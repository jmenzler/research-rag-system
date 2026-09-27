# long-ok-file
"""Production fetch pipeline via Scrapling Spider.

A single ``FetchPipelineSpider`` process handles every source URL using
Scrapling's multi-session routing, per-host concurrency control, and
block-detection hooks.

Escalation chain: ``http`` -> ``stealth`` -> ``stealth_max``.

Per-URL terminal-failure retry with exponential backoff (3s, 6s, 12s) sits on
top of session escalation: when the whole chain fails for a URL, the spider
sleeps and re-tries the same session once more before giving up. This fixes
non-deterministic CF WAF (Error 1020) races where a concurrent burst gets
challenged but a solo retry passes.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import anyio
from scrapling.spiders import Request, Response, Spider

from src.cleanup import (
    MIN_TEXT_LEN,
    PDF_MAGIC,
    detect_fail_signature,
    pdf_is_openable,
)
from src.fetch.classify import slug
from src.fetch.html_preprocess import extract_web_text
from src.fetch.page_metadata import harvest_page_metadata
from src.fetch.util import discover_free_pdf

_log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from scrapling.spiders.session import SessionManager

# ---------------------------------------------------------------------------
# Quality gate (thin wrapper over src.cleanup primitives)
# ---------------------------------------------------------------------------


def _quality_gate(text: str) -> tuple[bool, str]:
    """Return ``(pass, reason)``. Wraps ``detect_fail_signature`` + length floor."""
    sig = detect_fail_signature(text)
    if sig:
        return False, f"fail_sig:{sig}"
    if len(text) < MIN_TEXT_LEN:
        return False, f"too_short:{len(text)}"
    return True, "ok"


def _validate_pdf_bytes(path: Path) -> tuple[bool, str]:
    """Magic-byte + ``pypdfium2`` openability check on a downloaded PDF."""
    if path.stat().st_size < 1024:
        return False, "too_small"
    with path.open("rb") as f:
        head = f.read(5)
    if head != PDF_MAGIC:
        return False, f"not_pdf_magic:head={head!r}"
    return pdf_is_openable(path)


def _download_pdf(url: str, dest: Path) -> tuple[bool, str]:
    """Download via plain ``requests.GET`` and validate. Sync; call via ``anyio.to_thread``."""
    import requests as req  # type: ignore[import-untyped]  # noqa: PLC0415

    try:
        r = req.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; rag-system/1.0)"},
            timeout=30,
            stream=True,
        )
    except Exception as e:
        return False, f"pdf_dl_err:{type(e).__name__}"
    if r.status_code >= 400:
        return False, f"pdf_http_{r.status_code}"
    try:
        with dest.open("wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
    except Exception as e:
        return False, f"pdf_write_err:{type(e).__name__}"
    ok, reason = _validate_pdf_bytes(dest)
    if not ok:
        dest.unlink(missing_ok=True)
    return ok, reason


# ---------------------------------------------------------------------------
# Meta builder
# ---------------------------------------------------------------------------


def _build_meta(
    *,
    index: int,
    title: str,
    url: str,
    tier: str,
    source_id: str,
    fetch_results: dict[str, Any],
) -> dict[str, Any]:
    """Build a meta.json dict matching the production schema."""
    return {
        "index": index,
        "id": source_id,
        "title": title,
        "url": url,
        "host": urlparse(url).netloc.lower() if url else "",
        "tier": tier,
        "fetch": fetch_results,
    }


# ---------------------------------------------------------------------------
# Spider
# ---------------------------------------------------------------------------


# Maximum terminal-failure retries after the full escalation chain fails.
# Backoff: 3s, 6s, 12s. Lets concurrent CF WAF requests drain before re-trying.
_MAX_TERM_RETRIES = 3

# Maximum PDF-download retries inside _handle_pdf. PDFs served with the wrong
# content-type or from blocked hosts become retryable via session escalation.
_MAX_PDF_RETRIES = 3


class FetchPipelineSpider(Spider):
    """Fetch each source URL, extract text, escalate to stealth on failure.

    Uses explicit ``Request`` yielding for stealth escalation rather than
    Spider's ``is_blocked`` / ``retry_blocked_request`` -- those are designed
    for HTTP-level blocking (403/429/503) and silently drop items after max
    retries. Our content-level failures (JS-SPA too-short pages) need to
    produce output even when both tiers fail.
    """

    name = "fetch_pipeline"
    concurrent_requests = 30
    concurrent_requests_per_domain = 2
    download_delay = 0.5

    # Set in on_start; declared here so mypy can see the types.
    _out_root: Path
    _stats: dict[str, int]

    # Escalation chain: http -> stealth -> stealth_max -> stop
    _ESCALATION: dict[str, str] = {
        "http": "stealth",
        "stealth": "stealth_max",
        # stealth_max has no next tier
    }

    # ------------------------------------------------------------------
    # Shared helpers (DRY: result building, stat tracking, terminal retry)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_result(
        *,
        meta: dict[str, Any],
        url: str,
        final_url: str,
        session: str,
        ok: bool,
        reason: str,
        **extras: object,
    ) -> dict[str, Any]:
        """Return the base scraped-item dict with common fields filled in."""
        result: dict[str, Any] = {
            "index": meta["index"],
            "title": meta["title"],
            "url": meta.get("original_url", url),
            "final_url": final_url,
            "tier": meta["tier"],
            "session": session,
            "ok": ok,
            "reason": reason,
            "source_id": meta.get("source_id", ""),
            "nb_tag": meta.get("nb_tag", ""),
            "nb_id": meta.get("nb_id", ""),
            "slug_override": meta.get("slug_override", ""),
        }
        result.update(extras)
        return result

    def _record_success(self, session: str) -> None:
        """Increment the success counter for *session* (stealth/stealth_max)."""
        key_map = {
            "stealth": "stealth_rescue",
            "stealth_max": "stealth_max_rescue",
        }
        if key := key_map.get(session):
            self._stats[key] += 1
        self._stats["ok"] += 1

    async def _retry_terminal(
        self,
        meta: dict[str, Any],
        url: str,
        session: str,
        *,
        retries_key: str,
        max_retries: int,
        label: str,
    ) -> AsyncGenerator[Request, None]:
        """Yield a retry Request after exponential backoff, or return (void).

        Uses *retries_key* in meta to track per-URL retry count so the retry
        chain can survive across PDF/HTML handler boundaries without double-counting.
        """
        retries = meta.get(retries_key, 0)
        if retries >= max_retries:
            return
        delay = 3.0 * (2 ** retries)
        self.logger.warning(
            "Terminal failure on %s -- %s retry %d/%d after %.0fs",
            url, label, retries + 1, max_retries, delay,
        )
        await asyncio.sleep(delay)
        yield Request(
            url,
            sid=session,
            meta={**meta, retries_key: retries + 1},
            callback=self.parse,
        )

    # ------------------------------------------------------------------
    # Session setup
    # ------------------------------------------------------------------

    def configure_sessions(self, manager: SessionManager) -> None:
        from scrapling.fetchers import AsyncStealthySession, FetcherSession  # noqa: PLC0415

        manager.add("http", FetcherSession(impersonate="chrome"))
        manager.add(
            "stealth",
            AsyncStealthySession(headless=True, network_idle=True),
            lazy=True,
        )
        manager.add(
            "stealth_max",
            AsyncStealthySession(
                headless=True,
                network_idle=True,
                solve_cloudflare=True,
                humanize=True,  # type: ignore[call-arg]
                google_search=True,
                os_randomize=True,
            ),
            lazy=True,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def on_start(self, resuming: bool = False) -> None:
        self._out_root = Path(
            os.environ.get(
                "SPIDER_OUT_DIR",
                str(Path(__file__).resolve().parent / "_spider_output"),
            )
        )
        self._out_root.mkdir(parents=True, exist_ok=True)
        self._stats: dict[str, int] = {
            "ok": 0,
            "stealth_rescue": 0,
            "stealth_max_rescue": 0,
            "fail": 0,
        }

    async def on_close(self) -> None:
        total = self._stats["ok"] + self._stats["fail"]
        self.logger.info(
            "Pipeline done: %d/%d ok (%d stealth, %d stealth_max), %d failed",
            self._stats["ok"],
            total,
            self._stats["stealth_rescue"],
            self._stats["stealth_max_rescue"],
            self._stats["fail"],
        )

    async def on_error(self, request: Request, error: Exception) -> None:
        """Route transport-level fetch errors back into the escalation chain.

        Scrapling calls this when the fetch itself raises (DNS failure,
        connection refused, read timeout) — before ``parse`` ever runs, so the
        ``http -> stealth -> stealth_max`` escalation and terminal backoff that
        live in ``parse``/``_handle_pdf`` would otherwise never fire and a
        transient first-attempt hiccup would silently drop the source.

        Re-enqueue through the same chain here. Only when escalation AND
        terminal retries are exhausted do we give up — and then we write a
        failure artifact so the drop is visible on disk. Result-queue
        accounting is reconciled by ``spider_fetch_all`` (``spider_no_result``)
        and the disk walk in ``fetch.cli`` (``no_artifacts``), so on_error
        must NOT emit a synthetic item (it would double-count).
        """
        session = request.meta.get("_sid", request.sid or "http")
        if await self._reenqueue_on_error(request, session, error):
            return

        self._stats["fail"] += 1
        self.logger.error(
            "Fetch error [%s]: %s -- %s (escalation+retries exhausted)",
            request.meta.get("title", "?"),
            request.meta.get("original_url", request.url),
            error,
        )
        await anyio.to_thread.run_sync(
            self._write_transport_failure, dict(request.meta), request.url, session, str(error),
        )

    async def _reenqueue_on_error(
        self, request: Request, session: str, error: Exception,
    ) -> bool:
        """Re-enqueue *request* via escalation or terminal backoff. Returns True if requeued."""
        engine = getattr(self, "_engine", None)
        if engine is None:
            return False

        next_sid = self._ESCALATION.get(session)
        if next_sid is not None:
            self.logger.info(
                "Transport error escalating %s -> %s: %s (%s)",
                session, next_sid, request.url, type(error).__name__,
            )
            req = Request(
                request.url,
                sid=next_sid,
                meta={**request.meta, "_sid": next_sid, f"_err_{session}": str(error)},
                callback=self.parse,
            )
            await engine.scheduler.enqueue(req)
            return True

        # No next tier — terminal backoff on the same session.
        retries = request.meta.get("_term_retries", 0)
        if retries >= _MAX_TERM_RETRIES:
            return False
        delay = 3.0 * (2 ** retries)
        self.logger.warning(
            "Transport terminal failure on %s retry %d/%d after %.0fs (%s)",
            request.url, retries + 1, _MAX_TERM_RETRIES, delay, type(error).__name__,
        )
        await asyncio.sleep(delay)
        req = Request(
            request.url,
            sid=session,
            meta={**request.meta, "_term_retries": retries + 1},
            callback=self.parse,
            dont_filter=True,  # same sid+url as the failed request → bypass dedup
        )
        await engine.scheduler.enqueue(req)
        return True

    def _write_transport_failure(
        self, meta: dict[str, Any], url: str, session: str, error: str,
    ) -> None:
        """Persist a minimal failure meta.json + body so a dropped source is visible.

        Mirrors the terminal-failure artifact that parse() writes when the
        escalation chain is exhausted for a content-level failure.
        """
        out_dir = self._out_dir(meta)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "_failure_body.txt").write_text(
            f"transport_error session={session} url={url}\n{error}\n",
        )
        reason = f"transport_err:{error[:200]}"
        is_pdf = meta.get("original_url", "").lower().endswith(".pdf")
        kind = "pdf" if is_pdf else "web"
        fetch_results: dict[str, Any] = {
            "nlm": {"ok": False, "bytes": None},
            kind: {"ok": False, "bytes": None, "reason": reason, "session": session},
        }
        failure_meta = _build_meta(
            index=meta.get("index", -1),
            title=meta.get("title", "untitled"),
            url=meta.get("original_url", url),
            tier=meta.get("tier", "T5_web_blog"),
            source_id=meta.get("source_id", ""),
            fetch_results=fetch_results,
        )
        (out_dir / "meta.json").write_text(
            json.dumps(failure_meta, indent=2, ensure_ascii=False),
        )

    # ------------------------------------------------------------------
    # Request generation
    # ------------------------------------------------------------------

    async def start_requests(self) -> AsyncGenerator[Request, None]:
        for src in getattr(self, "_sources", []):
            url = src.get("url")
            # Defense-in-depth: w3lib.url.canonicalize_url crashes the entire
            # task group on a single None/non-str URL. Callers should filter
            # T6_nlm_native sources upstream, but skip here too just in case.
            if not isinstance(url, str) or not url:
                continue
            meta = {
                "index": src["index"],
                "title": src.get("title", "untitled"),
                "tier": src.get("tier", src.get("_tier", "T5_web_blog")),
                "source_id": src.get("id", ""),
                "original_url": url,
                "nb_tag": src.get("nb_tag", ""),
                "nb_id": src.get("nb_id", ""),
                "slug_override": src.get("slug_override", ""),
            }
            yield Request(url, sid="http", meta=meta, callback=self.parse)

    # ------------------------------------------------------------------
    # Block detection -- defer entirely to parse()
    # ------------------------------------------------------------------

    async def is_blocked(self, response: Response) -> bool:
        # Spider's default would retry the same session on 4xx/5xx and silently
        # drop after max retries. Our escalation chain handles content-level
        # blocks (CF challenge HTML returned with status 200), so let parse()
        # see every response and decide.
        return False

    # ------------------------------------------------------------------
    # Parse (called for HTTP, stealth, and stealth_max responses)
    # ------------------------------------------------------------------

    async def parse(
        self, response: Response,
    ) -> AsyncGenerator[Request | dict[str, Any], None]:
        """Thin safety wrapper around ``_parse_impl`` — any unhandled exception
        in the parse path is caught, logged, and converted to a fail result
        instead of crashing the entire spider via ``ExceptionGroup``."""
        try:
            async for item in self._parse_impl(response):
                yield item
        except Exception as exc:
            meta = response.meta
            self.logger.error(
                "Parse crash [%s] %s: %s",
                meta.get("title", "?"), response.url, exc, exc_info=True,
            )
            self._stats["fail"] += 1
            yield self._build_result(
                meta=meta,
                url=response.url,
                final_url=response.url,
                session=meta.get("_sid", "http"),
                ok=False,
                reason=f"parse_err:{type(exc).__name__}",
                bytes=0,
            )

    async def _parse_impl(
        self, response: Response,
    ) -> AsyncGenerator[Request | dict[str, Any], None]:
        meta = response.meta
        ct = response.headers.get("content-type", "").lower()
        session = meta.get("_sid", "http")
        url = response.url  # final URL after redirects

        # --- PDF path ------------------------------------------------------
        if "application/pdf" in ct or meta.get("original_url", "").lower().endswith(".pdf"):
            async for item in self._handle_pdf(response, meta, session):
                yield item
            return

        # --- HTML path -----------------------------------------------------
        out_dir = self._out_dir(meta)
        out_dir.mkdir(parents=True, exist_ok=True)
        text, img_registry = extract_web_text(response.body, base_url=url)
        if img_registry:
            (out_dir / "figures.json").write_text(json.dumps(img_registry, indent=2))

        ok, reason = _quality_gate(text)

        # Escalate if quality gate failed and there's a next tier.
        next_sid = self._ESCALATION.get(session)
        if not ok and next_sid is not None:
            self.logger.info(
                "Escalating %s -> %s: %s (reason=%s)", session, next_sid, url, reason,
            )
            yield Request(
                response.url,
                sid=next_sid,
                meta={**meta, "_sid": next_sid, f"_reason_{session}": reason},
                callback=self.parse,
            )
            return

        # Terminal -- persist result (pass or fail).
        pdf_upgrade: dict[str, Any] | None = None

        if ok:
            # PDF upgrade: scan raw HTML for free-PDF links.
            html_str = (
                response.body.decode("utf-8", errors="replace")
                if isinstance(response.body, bytes)
                else str(response.body)
            )
            pdf_url = discover_free_pdf(html_str, url)
            has_pdf = False
            if pdf_url and not (out_dir / "source.pdf").exists():
                pdf_ok, pdf_reason = await anyio.to_thread.run_sync(
                    _download_pdf, pdf_url, out_dir / "source.pdf",
                )
                if pdf_ok:
                    pdf_upgrade = {
                        "url": pdf_url,
                        "reason": pdf_reason,
                        "pdf_bytes": (out_dir / "source.pdf").stat().st_size,
                    }
                    has_pdf = True

            if not has_pdf:
                (out_dir / "web.txt").write_text(text)

            self._record_success(session)
        else:
            async for retry_req in self._retry_terminal(
                meta, url, session,
                retries_key="_term_retries", max_retries=_MAX_TERM_RETRIES, label="HTML",
            ):
                yield retry_req
                return  # _retry_terminal yields at most one Request

            (out_dir / "_failure_body.txt").write_text(
                text if text else response.body.decode("utf-8", errors="replace")[:4000],
            )
            self._stats["fail"] += 1

        result = self._build_result(
            meta=meta,
            url=url,
            final_url=url,
            session=session,
            ok=ok,
            reason=f"ok:{session}" if ok else reason,
            bytes=len(text.encode()) if text else 0,
        )
        if pdf_upgrade:
            result["pdf_upgrade"] = pdf_upgrade
        if ok:
            result["_raw_html"] = (
                response.body.decode("utf-8", errors="replace")
                if isinstance(response.body, bytes)
                else str(response.body)
            )
        yield result

    # ------------------------------------------------------------------
    # PDF handler (with retry through escalation chain on validation fail)
    # ------------------------------------------------------------------

    async def _handle_pdf(
        self, response: Response, meta: dict[str, Any], session: str,
    ) -> AsyncGenerator[Request | dict[str, Any], None]:
        out_dir = self._out_dir(meta)
        out_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = out_dir / "source.pdf"
        pdf_path.write_bytes(response.body)

        ok, reason = _validate_pdf_bytes(pdf_path)

        if not ok:
            pdf_path.unlink(missing_ok=True)

            # Escalate via session chain on validation failure: a PDF served
            # with the wrong content-type or from a blocked host can sometimes
            # be retrieved correctly by a real browser session.
            next_sid = self._ESCALATION.get(session)
            if next_sid is not None:
                self.logger.info(
                    "PDF escalating %s -> %s: %s (reason=%s)",
                    session, next_sid, response.url, reason,
                )
                yield Request(
                    response.url,
                    sid=next_sid,
                    meta={**meta, "_sid": next_sid, f"_pdf_reason_{session}": reason},
                    callback=self.parse,
                )
                return

            # All tiers exhausted -- terminal-retry with backoff (same as HTML).
            async for retry_req in self._retry_terminal(
                meta, response.url, session,
                retries_key="_pdf_retries", max_retries=_MAX_PDF_RETRIES, label="PDF",
            ):
                yield retry_req
                return

            self._stats["fail"] += 1
        else:
            self._record_success(session)

        yield self._build_result(
            meta=meta,
            url=response.url,
            final_url=response.url,
            session=session,
            ok=ok,
            reason=f"pdf:{reason}",
            pdf_bytes=pdf_path.stat().st_size if ok else None,
        )

    # ------------------------------------------------------------------
    # Item processing (write meta.json) + output dir helper
    # ------------------------------------------------------------------

    async def on_scraped_item(self, item: dict[str, Any]) -> dict[str, Any] | None:
        """Per-source post-processing: NLM fulltext, structured sidecar, meta.json.

        Runs once per scraped item after parse() yields. The HTTP/PDF fetch is
        already done by parse() — here we do the side work: NotebookLM
        fulltext download, content_list.json sidecar generation, and metadata
        enrichment (authors / year / DOI / etc).
        """
        is_pdf = item.get("reason", "").startswith("pdf:")
        out_dir = self._out_dir(item)
        out_dir.mkdir(parents=True, exist_ok=True)

        # --- NLM fulltext --------------------------------------------------
        nb_id = item.get("nb_id", "")
        source_id = item.get("source_id", "")
        nlm_path = out_dir / "nlm.txt"
        nlm_result: dict[str, Any] = {"ok": False, "bytes": None}
        if nb_id and source_id and not nlm_path.exists():
            from src.fetch.orchestrator import fetch_nlm_fulltext  # noqa: PLC0415

            n_bytes = await anyio.to_thread.run_sync(
                fetch_nlm_fulltext, nb_id, source_id, nlm_path,
            )
            nlm_result = {"ok": bool(n_bytes), "bytes": n_bytes}
        elif nlm_path.exists():
            nlm_result = {
                "ok": True,
                "bytes": nlm_path.stat().st_size,
                "cached": True,
            }

        # --- Structured sidecar from web.txt's HTML body --------------------
        # The Spider already wrote web.txt; structured extraction needs the
        # raw HTML which is held in item["_raw_html"] when present (only for
        # successful HTML fetches).
        if not is_pdf and item.get("ok") and (out_dir / "web.txt").exists():
            await anyio.to_thread.run_sync(self._write_structured_sidecar, out_dir, item)

        # --- NLM-only fallback: synthesize content_list.json from nlm.txt --
        # Only fires when the HTML path didn't produce a sidecar (or there
        # was no HTML — e.g. NotebookLM is the only signal). Idempotent and
        # best-effort: failures are logged, never raised. See
        # ``scripts/nlm_to_content_list.py``.
        if nlm_path.exists() and not (out_dir / "content_list.json").exists():
            await anyio.to_thread.run_sync(
                self._synthesize_from_nlm, out_dir, self._out_root,
            )

        # --- meta.json -----------------------------------------------------
        fetch_results: dict[str, Any] = {"nlm": nlm_result}
        kind = "pdf" if is_pdf else "web"
        fetch_results[kind] = {
            "ok": item["ok"],
            "bytes": item.get("bytes") or item.get("pdf_bytes"),
            "reason": item["reason"],
            "session": item["session"],
        }
        meta = _build_meta(
            index=item["index"],
            title=item["title"],
            url=item["url"],
            tier=item["tier"],
            source_id=source_id,
            fetch_results=fetch_results,
        )
        if item.get("final_url"):
            meta["final_url"] = item["final_url"]

        # --- Harvest page metadata (trafilatura + citation tags) from raw HTML.
        # Best-effort: failures fall through to text-scan in extract_metadata.
        # PDF fetches have no _raw_html.
        raw_html = item.get("_raw_html")
        if isinstance(raw_html, str) and raw_html:
            try:
                meta["page_metadata"] = harvest_page_metadata(raw_html)
            except Exception as exc:
                _log.warning("page_metadata harvest failed: %s", exc)

        # --- Metadata enrichment (authors / year / DOI / extraction quality)
        await anyio.to_thread.run_sync(self._enrich_and_write_meta, meta, out_dir)

        # --- Sources registry: mark as fetched (best-effort; sqlite errors
        # logged not raised — fetch must not fail because of bookkeeping).
        await anyio.to_thread.run_sync(
            self._mark_source_fetched, meta, item.get("notebook"),
        )
        return item

    @staticmethod
    def _mark_source_fetched(meta: dict[str, Any], notebook: str | None) -> None:
        try:
            from src.ingest.sources import upsert_fetched  # noqa: PLC0415
            from src.ingest.storage import _open_parents_db  # noqa: PLC0415

            conn = _open_parents_db()
            try:
                upsert_fetched(conn, meta, notebook, meta.get("source_type"))
            finally:
                conn.close()
        except Exception as exc:
            _log.warning("sources upsert_fetched failed: %s: %s", type(exc).__name__, exc)

    @staticmethod
    def _synthesize_from_nlm(out_dir: Path, root: Path) -> None:
        """Best-effort fallback: build content_list.json from nlm.txt.

        Fires only when the HTML structured-sidecar path didn't produce one
        (no raw HTML, or extraction returned nothing). Quarantined garbage
        goes to ``<root>/_quarantine_nlm_garbage/``. All exceptions are
        swallowed with a warning — fetch must not fail because of a synthesis
        problem.
        """
        try:
            from scripts.nlm_to_content_list import synthesize_from_nlm  # noqa: PLC0415

            result = synthesize_from_nlm(out_dir, root)
            if result.startswith("quarantined:"):
                _log.info("nlm synthesizer %s for %s", result, out_dir)
            elif result == "synthesized":
                _log.debug("nlm synthesized content_list.json for %s", out_dir)
        except Exception as exc:
            _log.warning(
                "nlm synthesizer failed for %s: %s: %s",
                out_dir, type(exc).__name__, exc,
            )

    @staticmethod
    def _write_structured_sidecar(out_dir: Path, item: dict[str, Any]) -> None:
        """Write content_list.json from raw HTML when available. Best-effort.

        Failures are logged at warning level rather than raised — a missing
        sidecar degrades chunking quality but doesn't block the rest of the
        pipeline. Each warning carries the source dir so failed extractions
        are attributable in CI/log review.
        """
        raw_html = item.get("_raw_html")
        if not raw_html:
            return
        try:
            from src.fetch.html_preprocess import preprocess_html  # noqa: PLC0415
            from src.fetch.structured import extract_structured  # noqa: PLC0415
        except Exception as exc:
            _log.warning(
                "structured sidecar import failed for %s: %s: %s",
                out_dir, type(exc).__name__, exc,
            )
            return
        try:
            processed_html, _ = preprocess_html(raw_html, base_url="")
            elements = extract_structured(processed_html, page_metadata={})
        except Exception as exc:
            _log.warning(
                "structured extraction failed for %s: %s: %s",
                out_dir, type(exc).__name__, exc,
            )
            return
        if elements is not None:
            (out_dir / "content_list.json").write_text(
                json.dumps(elements, ensure_ascii=False, indent=2),
            )

    @staticmethod
    def _enrich_and_write_meta(meta: dict[str, Any], out_dir: Path) -> None:
        """Stamp ``extracted_at`` + ``parser_versions``, run inline enrichment, persist.

        ``enrich_meta_inline`` failures are logged but don't block the meta.json
        write — best-effort metadata enrichment shouldn't kill the source.
        """
        import datetime  # noqa: PLC0415

        from src.fetch.metadata import _detect_parser_versions, enrich_meta_inline  # noqa: PLC0415

        meta["extracted_at"] = datetime.datetime.now(datetime.UTC).isoformat()
        meta["parser_versions"] = _detect_parser_versions()
        try:
            enrich_meta_inline(meta, out_dir)
        except Exception as exc:
            _log.warning(
                "metadata enrichment failed for %s: %s: %s",
                out_dir, type(exc).__name__, exc,
            )
        (out_dir / "meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False),
        )

    def _out_dir(self, meta: dict[str, Any]) -> Path:
        """Resolve target dir: ``<root>/<nb_tag>/<slug>`` or ``<root>/<slug>``.

        ``nb_tag`` carries the collection name (post-flatten). The slug
        derives from the source title — no ``NNN__`` extraction-order prefix.

        ``slug_override`` (when set in *meta*) wins over ``slug(title)`` so a
        caller that has already resolved a collision (e.g. fetch CLI suffixed
        the slug with ``__<sha6>``) writes into that exact dir.
        """
        nb_tag: str = meta.get("nb_tag", "")
        override = meta.get("slug_override")
        if override:
            leaf = str(override)
        else:
            title: str = meta.get("title", "untitled")
            leaf = slug(title)
        if nb_tag:
            return self._out_root / nb_tag / leaf
        return self._out_root / leaf
