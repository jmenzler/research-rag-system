"""Queue-based fetch+parse pipeline runner for `ragctl run`.

Streaming-queue design: a single ``FetchPipelineSpider`` (Scrapling) handles
all source URLs with internal concurrency=30, while N parse workers consume
ready PDFs from a parallel queue, keeping the MinerU daemon continuously fed.
Stage / contextualize / ingest stay sequential per-notebook in the main
process post-barrier.

Architecture::

    Main process
      ├─ Resolve sources, dedup, interleave by notebook
      ├─ Spawn Spider thread + parse_worker × M processes
      ├─ Coordinator loop drains ResultQueue, enqueues ParseItems
      ├─ Audit barrier (corpus-wide, single run)
      └─ Sequential post-barrier per notebook (stage → ctx → ingest)

Tests inject a per-URL fetch stub via ``_fetch_fn=...`` which routes through
``fetch_worker`` instead of the Spider; production never takes that path.
"""
from __future__ import annotations

import errno
import json
import multiprocessing
import os
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.pipeline import orchestrate
from src.pipeline.crasher_tracker import CrasherOutcome
from src.pipeline.daemon_supervisor import DaemonSupervisor
from src.pipeline.queues import FetchItem, ParseItem, SharedPacer

# ---------------------------------------------------------------------------
# Poison pill sentinel — None on a queue tells workers to exit
# ---------------------------------------------------------------------------
_POISON = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _classify_source(src: dict[str, Any]) -> dict[str, Any]:
    """Ensure source has a ``_tier`` key (idempotent). Uses classify from fetch_sources."""
    if "_tier" not in src:
        from src.fetch.classify import classify  # noqa: PLC0415
        src["_tier"] = classify(src)
    return src


def _resolve_sources_for_notebook(nb: dict[str, Any]) -> list[dict[str, Any]]:
    """Return classified source dicts for one notebook.

    Uses embedded ``_sources`` when present (test fixture), otherwise calls
    ``notebooklm source list`` via ``list_sources``.
    """
    if "_sources" in nb:
        sources = list(nb["_sources"])
        for s in sources:
            _classify_source(s)
        return sources

    from src.fetch.classify import classify, list_sources  # noqa: PLC0415
    sources = list_sources(nb["id"])
    for s in sources:
        s["_tier"] = classify(s)
    return sources


def _discover_existing_pdfs(
    collections: list[str], sources_root: Path,
) -> list[ParseItem]:
    """Scan ``sources/<collection>/<slug>/source.pdf`` for parse-only mode.

    The notebook tag attached to each ParseItem carries the collection name —
    downstream stages use it as the Milvus partition value at ingest time.
    """
    items: list[ParseItem] = []
    for coll in collections:
        coll_dir = sources_root / coll
        if not coll_dir.is_dir():
            continue
        for pdf_path in sorted(coll_dir.glob("*/source.pdf")):
            if any(p.startswith("_") for p in pdf_path.relative_to(sources_root).parts):
                continue
            items.append(ParseItem(nb_tag=coll, pdf_path=pdf_path))
    return items


def _filter_failed_only(
    items: list[FetchItem], sources_root: Path,
) -> tuple[list[FetchItem], list[str]]:
    """Keep only sources whose last fetch did not produce an ok ``web`` or ``pdf``.

    A source is "failed" iff its ``meta.json`` shows neither ``fetch.web.ok``
    nor ``fetch.pdf.ok`` is True. Sources without a ``meta.json`` (never
    fetched) also count as failed. NLM-only success doesn't count -- nlm.txt
    is a side artifact, not the primary fetch target.

    Returns ``(failed_items, urls_to_evict)``. ``urls_to_evict`` is the list
    of URLs for kept (failed) items so the caller can drop them from the
    URL cache before the spider runs -- otherwise stale partial files get
    hardlinked into the new fetch and shadow it.
    """
    from src.fetch.classify import canonical_url, slug  # noqa: PLC0415

    failed_items: list[FetchItem] = []
    urls_to_evict: list[str] = []
    for item in items:
        title = str(item.src.get("title") or "untitled")
        leaf = slug(title)
        # The on-disk dir is normally ``<nb_tag>/<slug>/`` but a slug-collision
        # at fetch time may suffix it with ``__<sha6>``. Try the bare slug
        # first, then any suffixed sibling whose meta.json has the same
        # canonical URL as this item.
        meta_path = sources_root / item.nb_tag / leaf / "meta.json"
        meta: dict[str, Any] = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
            except (json.JSONDecodeError, OSError):
                meta = {}
        # Walk suffixed-slug siblings when the bare-slug meta.json is absent
        # OR has no ``fetch`` block — both states mean the bare-slug dir is
        # not the canonical record for this source.
        if not meta or not meta.get("fetch"):
            target_cu = canonical_url(item.src)
            if target_cu:
                for sibling in (sources_root / item.nb_tag).glob(f"{leaf}__*"):
                    sib_meta = sibling / "meta.json"
                    if not sib_meta.exists():
                        continue
                    try:
                        cand = json.loads(sib_meta.read_text())
                    except (json.JSONDecodeError, OSError):
                        continue
                    if canonical_url(cand) == target_cu:
                        meta = cand
                        break
        fetch = meta.get("fetch") or {}
        web_ok = bool((fetch.get("web") or {}).get("ok"))
        pdf_ok = bool((fetch.get("pdf") or {}).get("ok"))
        if web_ok or pdf_ok:
            continue
        failed_items.append(item)
        url = item.src.get("url")
        if isinstance(url, str) and url:
            urls_to_evict.append(url)
    return failed_items, urls_to_evict


def _stage_ready(nb_tag: str, progress: dict[str, dict[str, int]]) -> bool:
    """True when all sources for a notebook are fetched AND all PDFs parsed."""
    p = progress.get(nb_tag)
    if p is None:
        return False
    return p["fetched"] >= p["total"] and p["pdfs_parsed"] >= p["pdfs_expected"]


# ---------------------------------------------------------------------------
# Progress tracking (plain dict, no dataclass)
# ---------------------------------------------------------------------------

@dataclass
class _RunProgress:
    """Mutable bag shared between coordinator and run_pipeline."""
    progress: dict[str, dict[str, int]] = field(default_factory=dict)
    fetch_total: int = 0
    fetch_done_count: int = 0
    parse_pending: int = 0


# ---------------------------------------------------------------------------
# Workers (module-level for multiprocessing pickling)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Run lockfile — prevents two `ragctl run` invocations from racing on the
# shared MinerU daemon, GPU, sources/ tree, and parents/ sqlite. Concurrent
# runs were observed to mutually SIGKILL each other's daemons via the
# orphan-sweep, then deadlock on bounded queues. One run at a time.
# ---------------------------------------------------------------------------
_RUN_LOCK_PATH = Path.home() / ".cache" / "rag-system" / "ragctl-run.lock"

# Holder rewrites the lockfile heartbeat this often. Takeover triggers when
# the heartbeat is older than the stale threshold. Both env-overridable so
# operators can tune for long-running stages.
_RUN_LOCK_HEARTBEAT_S = int(os.environ.get("RAG_RUNLOCK_HEARTBEAT_S", "30"))
_RUN_LOCK_STALE_S = int(os.environ.get("RAG_RUNLOCK_STALE_S", "300"))
_RUN_LOCK_TERM_GRACE_S = float(os.environ.get("RAG_RUNLOCK_TERM_GRACE_S", "10"))


class RunLockHeldError(RuntimeError):
    """Raised when another `ragctl run` already holds the lock and is live."""


def _read_lockfile_holder() -> dict[str, str]:
    """Parse `key=value` tokens from the lockfile. Empty dict on any failure."""
    try:
        text = _RUN_LOCK_PATH.read_text()
    except OSError:
        return {}
    out: dict[str, str] = {}
    for line in text.splitlines():
        for tok in line.split():
            if "=" in tok:
                k, _, v = tok.partition("=")
                out[k] = v
    return out


def _pid_alive(pid: int) -> bool:
    """True if a process with this PID is alive (or exists but owned by another user)."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _kill_holder(pid: int) -> None:
    """SIGTERM, wait up to _RUN_LOCK_TERM_GRACE_S, SIGKILL. Tolerates already-dead."""
    import signal  # noqa: PLC0415
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return
    deadline = time.time() + _RUN_LOCK_TERM_GRACE_S
    while time.time() < deadline:
        if not _pid_alive(pid):
            return
        time.sleep(0.5)
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass


def _write_holder_stamp(fd: int) -> None:
    """Rewrite lockfile body with current holder identity and heartbeat timestamp."""
    now = int(time.time())
    stamp = (
        f"pid={os.getpid()} host={os.uname().nodename} "
        f"ts={now} last_heartbeat={now}\n"
    )
    try:
        os.ftruncate(fd, 0)
        os.lseek(fd, 0, 0)
        os.write(fd, stamp.encode())
    except OSError:
        pass


_heartbeat_stop = threading.Event()
_heartbeat_thread: threading.Thread | None = None


def _heartbeat_loop(fd: int) -> None:
    while not _heartbeat_stop.wait(_RUN_LOCK_HEARTBEAT_S):
        _write_holder_stamp(fd)


def _start_heartbeat(fd: int) -> None:
    global _heartbeat_thread
    _heartbeat_stop.clear()
    t = threading.Thread(
        target=_heartbeat_loop,
        args=(fd,),
        name="run-lock-heartbeat",
        daemon=True,
    )
    t.start()
    _heartbeat_thread = t


def _stop_heartbeat() -> None:
    _heartbeat_stop.set()
    t = _heartbeat_thread
    if t is not None:
        t.join(timeout=2.0)


def _try_takeover_stale_lock() -> bool:
    """Kill stale holder + unlink lockfile if heartbeat exceeds stale threshold.

    Returns True if a takeover occurred (caller should retry acquire). Live
    holders return False so the caller raises RunLockHeldError.
    """
    holder = _read_lockfile_holder()
    if not holder:
        return False
    try:
        hb = int(holder.get("last_heartbeat", holder.get("ts", "0")))
    except ValueError:
        hb = 0
    age = int(time.time()) - hb
    if age < _RUN_LOCK_STALE_S:
        return False
    try:
        pid = int(holder.get("pid", "0"))
    except ValueError:
        pid = 0
    print(
        f"WARN: stale run lockfile (holder pid={pid}, heartbeat {age}s old, "
        f"threshold {_RUN_LOCK_STALE_S}s) — taking over",
        file=sys.stderr,
    )
    if pid > 0 and pid != os.getpid() and _pid_alive(pid):
        _kill_holder(pid)
    try:
        _RUN_LOCK_PATH.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        return False
    return True


def _acquire_run_lock() -> int:
    """Acquire the exclusive run lockfile or raise RunLockHeldError.

    fcntl.flock LOCK_EX | LOCK_NB is the primary gate. A 30s heartbeat thread
    rewrites the lockfile body so wedged-alive holders are detectable. On
    contention, heartbeat age is checked: if older than RAG_RUNLOCK_STALE_S
    (default 300s), the holder is SIGTERMed (SIGKILLed after RAG_RUNLOCK_TERM_GRACE_S)
    and the lockfile is unlinked so this invocation can proceed. Live holders
    still raise RunLockHeldError.
    """
    import fcntl  # noqa: PLC0415
    _RUN_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(2):
        fd = os.open(str(_RUN_LOCK_PATH), os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(fd)
            if exc.errno not in (errno.EWOULDBLOCK, errno.EAGAIN, errno.EACCES):
                raise
            if attempt == 0 and _try_takeover_stale_lock():
                continue
            holder = ""
            try:
                holder = _RUN_LOCK_PATH.read_text().strip()
            except OSError:
                pass
            msg = "another `ragctl run` is in progress"
            if holder:
                msg += f" (holder: {holder})"
            msg += f" — lockfile {_RUN_LOCK_PATH}"
            raise RunLockHeldError(msg) from exc
        _write_holder_stamp(fd)
        _start_heartbeat(fd)
        return fd
    raise RunLockHeldError("unreachable: run-lock acquire exhausted retries")


def _release_run_lock(fd: int) -> None:
    """Release the run lock, stop the heartbeat thread, close the fd."""
    import fcntl  # noqa: PLC0415
    _stop_heartbeat()
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    except OSError:
        pass
    try:
        os.close(fd)
    except OSError:
        pass


def fetch_worker(in_q: multiprocessing.Queue[Any],
                 result_q: multiprocessing.Queue[Any],
                 pacer: SharedPacer,
                 sources_root: Path,
                 no_cffi: bool,
                 fetch_fn: object) -> None:
    """Per-URL fetch loop — used ONLY by tests via ``_fetch_fn`` injection.

    Production fetch goes through ``spider_fetch_all`` (single Spider process,
    internal concurrency=30). This worker exists to keep the test-stub path
    decoupled from Scrapling so unit tests run without the browser stack.
    """
    while True:
        item = in_q.get()
        if item is None:          # poison pill — exit
            break
        try:
            meta, written = fetch_fn(  # type: ignore[operator]
                item.src,
                out_root=sources_root / item.nb_tag,
                no_cffi=item.no_cffi,
                pace_fn=pacer.wait if pacer is not None else None,
                nb_id=item.nb_id,
            )
            pdf = next((p for p in written if p.name == "source.pdf"), None)
            result_q.put((
                "fetch_ok", item.nb_tag, item.src["id"],
                pdf is not None,
                str(pdf) if pdf else None,
            ))
        except Exception as e:
            result_q.put((
                "fetch_fail", item.nb_tag, item.src["id"],
                f"{type(e).__name__}: {e}",
            ))


def spider_fetch_all(items: list[FetchItem],
                     result_q: multiprocessing.Queue[Any],
                     sources_root: Path) -> None:
    """Run a single ``FetchPipelineSpider`` over ALL fetch items.

    Run by ``run_pipeline`` in a background thread (Scrapling's anyio loop
    conflicts with multiprocessing contexts). The Spider handles concurrency
    internally (30 global, 2 per domain). Each scraped item produces a
    ``fetch_ok`` or ``fetch_fail`` event matching the tuple shape that
    ``coordinator_loop`` expects.
    """
    import os  # noqa: PLC0415
    os.environ["SPIDER_OUT_DIR"] = str(sources_root)

    import anyio  # noqa: PLC0415

    from src.fetch.spider import FetchPipelineSpider  # noqa: PLC0415

    # Track items the Spider has already produced a result for, so a mid-run
    # crash only fail-emits the remainder. Without this, the crash handler
    # double-counts items the coordinator has already accounted for, which
    # overshoots fetch_total and leaves parse_pending pinned > 0.
    reported: set[tuple[str, str]] = set()

    # Filter out URL-less sources (T6_nlm_native — fulltext lives in NLM only,
    # nothing to fetch over the wire). Scrapling's scheduler crashes the entire
    # task group on a single None URL via w3lib's to_unicode(); see
    # logs/spider_crash_traceback.txt for the failure mode. Emit a synthetic
    # fetch_ok per skipped source so the coordinator's _stage_ready check fires
    # and parse_pending stays balanced.
    sources = []
    for item in items:
        src = dict(item.src)
        url = src.get("url")
        if not isinstance(url, str) or not url:
            result_q.put((
                "fetch_ok", item.nb_tag, str(item.src["id"]), False, None,
            ))
            reported.add((item.nb_tag, str(item.src["id"])))
            continue
        src["nb_tag"] = item.nb_tag
        src["nb_id"] = item.nb_id
        src["tier"] = src.get("_tier", src.get("tier", "T5_web_blog"))
        sources.append(src)

    async def _run() -> None:
        spider = FetchPipelineSpider()
        spider._sources = sources  # type: ignore[attr-defined]
        async for spider_item in spider.stream():
            nb_tag = spider_item.get("nb_tag", "")
            src_id = spider_item.get("source_id", "")
            ok = bool(spider_item.get("ok"))
            is_pdf = spider_item.get("reason", "").startswith("pdf:")
            has_pdf = is_pdf or "pdf_upgrade" in spider_item

            if ok or has_pdf:
                pdf_path = None
                if has_pdf:
                    title = spider_item.get("title", "untitled")
                    leaf = spider_item.get("slug_override") or ""
                    if not leaf:
                        from src.fetch.classify import slug as _slug  # noqa: PLC0415
                        leaf = _slug(title)
                    pdf_path = str(
                        sources_root / nb_tag / leaf / "source.pdf",
                    )
                result_q.put((
                    "fetch_ok", nb_tag, src_id, has_pdf, pdf_path,
                ))
            else:
                result_q.put((
                    "fetch_fail", nb_tag, src_id,
                    spider_item.get("reason", "unknown"),
                ))
            reported.add((nb_tag, src_id))

    try:
        anyio.run(_run)
    except Exception as e:
        # Diagnostic: ExceptionGroup stringification hides the real error.
        # Dump the full traceback + every sub-exception to stderr AND a
        # persistent file so we can read the actual cause after the run.
        crash_log = sources_root.parent / "logs" / "spider_crash_traceback.txt"
        crash_log.parent.mkdir(parents=True, exist_ok=True)

        with crash_log.open("w") as f:
            for sink in (sys.stderr, f):
                traceback.print_exc(file=sink)
                subs = getattr(e, "exceptions", None)
                if subs:
                    for i, sub in enumerate(subs):
                        print(
                            f"\n--- sub-exception {i} ({type(sub).__name__}) ---",
                            file=sink,
                        )
                        traceback.print_exception(
                            type(sub), sub, sub.__traceback__, file=sink,
                        )

        # Build informative reason string for fail-emit.
        subs = getattr(e, "exceptions", None)
        if subs:
            sub_classes = ",".join(type(s).__name__ for s in subs)
            reason_str = f"spider_crash:{type(e).__name__}[{sub_classes}]: {e}"
        else:
            reason_str = f"spider_crash:{type(e).__name__}: {e}"

        # Spider crash: fail-emit only items the spider hasn't already reported.
        for item in items:
            key = (item.nb_tag, str(item.src["id"]))
            if key in reported:
                continue
            try:
                result_q.put((
                    "fetch_fail", item.nb_tag, item.src["id"],
                    reason_str,
                ))
            except Exception:
                pass
    else:
        # Transport-level failures hit Scrapling's engine, which awaits
        # Spider.on_error and discards its return — on_error emits no item, so
        # those sources never reach result_q and the coordinator stalls below
        # fetch_total forever. Fail-emit every source the spider never reported.
        for item in items:
            key = (item.nb_tag, str(item.src["id"]))
            if key in reported:
                continue
            result_q.put((
                "fetch_fail", item.nb_tag, item.src["id"],
                "spider_no_result",
            ))


def parse_worker(in_q: multiprocessing.Queue[Any],
                 result_q: multiprocessing.Queue[Any],
                 parse_fn: object = None) -> None:
    """Parse loop: pull ParseItems, POST to MinerU daemon, push results.

    Single-shot parse: any error → parse_fail. The worker does NOT restart the
    daemon on connection failures — that path caused cascading respawns (16
    daemons spawned + killed in ~140s on the same PDF range). Daemon lifecycle
    is owned solely by the main process; if it dies mid-run, all subsequent
    parses fail visibly until the next ragctl run.
    """
    if parse_fn is None:
        from src.pdf_parsers.mineru import parse_pipeline_only  # noqa: PLC0415
        parse_fn = parse_pipeline_only

    while True:
        item = in_q.get()
        if item is None:              # poison pill — exit
            break
        # Skip if pdf.txt already exists and is non-empty — MinerU is deterministic
        # and re-parsing wastes ~30s GPU per source. Lets ragctl restarts resume
        # cleanly without redoing prior parse work.
        pdf_txt = item.pdf_path.parent / "pdf.txt"
        if pdf_txt.exists() and pdf_txt.stat().st_size > 0:
            result_q.put(("parse_ok", item.nb_tag, str(item.pdf_path)))
            continue
        # Forensic logging: emit "starting" line so the next libpdfium SIGTRAP
        # has an attributable PDF in the orchestrator log. Goes to result_q
        # so it shows up in the main process's output (forkserver child
        # stdout is invisible otherwise).
        result_q.put(("parse_start", item.nb_tag, str(item.pdf_path)))

        try:
            text = parse_fn(item.pdf_path)  # type: ignore[operator]
            pdf_txt.write_text(text)
            _refresh_meta_extraction_quality(item.pdf_path.parent)
            result_q.put(("parse_ok", item.nb_tag, str(item.pdf_path)))
        except Exception as e:
            result_q.put((
                "parse_fail", item.nb_tag, str(item.pdf_path),
                f"{type(e).__name__}: {e}",
            ))


def _refresh_meta_extraction_quality(doc_dir: Path) -> None:
    """Re-run inline metadata enrichment after parse so ``extraction_quality``
    reflects the freshly-written ``content_list.json``.

    The Spider's ``on_scraped_item`` runs ``enrich_meta_inline`` at fetch
    time — but for PDF sources, ``content_list.json`` is written minutes
    later by the parse stage. Without this hook, every PDF source ships
    with all-zero ``n_tables`` / ``n_equations`` / ``n_figures`` / ``n_lists``.

    Best-effort: any failure (missing meta.json, JSON corruption, etc.) is
    logged but never propagated — parse_ok must not be downgraded to
    parse_fail by a metadata bookkeeping error.
    """
    meta_path = doc_dir / "meta.json"
    if not meta_path.exists():
        return
    try:
        from src.fetch.metadata import enrich_meta_inline  # noqa: PLC0415
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if not isinstance(meta, dict):
            return
        enrich_meta_inline(meta, doc_dir)
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    except Exception as exc:
        print(f"  ! refresh_meta failed for {doc_dir}: {type(exc).__name__}: {exc}")


def _advance_source_parsed(doc_dir: Path) -> None:
    """Mark a source as 'parsed' in the sources registry. Best-effort.

    The coordinator only has the PDF path; canonical_id comes from the
    sibling meta.json (the spider wrote it). Missing meta.json or db errors
    log + continue — pipeline correctness can't hinge on bookkeeping.
    """
    meta_path = doc_dir / "meta.json"
    if not meta_path.exists():
        return
    try:
        from src.ingest.sources import advance_status, canonical_id  # noqa: PLC0415
        from src.ingest.storage import _open_parents_db  # noqa: PLC0415

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if not isinstance(meta, dict):
            return
        cid, _ = canonical_id(meta)
        conn = _open_parents_db()
        try:
            advance_status(conn, cid, "parsed")
        finally:
            conn.close()
    except Exception as exc:
        print(f"  ! advance_source_parsed failed for {doc_dir}: {type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Coordinator loop
# ---------------------------------------------------------------------------

_COORDINATOR_POLL_TIMEOUT = 0.5
_BACKPRESSURE_WARN_S = 10.0


def coordinator_loop(rp: _RunProgress,
                     fetch_q: multiprocessing.Queue[Any],
                     parse_q: multiprocessing.Queue[Any],
                     result_q: multiprocessing.Queue[Any],
                     do_fetch: bool,
                     do_parse: bool,
                     supervisor: DaemonSupervisor | None = None,
                     sources_root: Path | None = None) -> None:
    """Drain ResultQueue. Update progress, enqueue ParseItems on fetch_ok+pdf.

    When *supervisor* is provided (production path with parse_pipeline), the
    coordinator runs daemon-liveness checks on every idle tick and reactively
    on parse_fail events that look like daemon-death. On circuit-break, fresh
    fetch_ok+pdf events synthesise immediate parse_fail with
    ``daemon_circuit_breaker`` rather than enqueueing more work — but the
    coordinator still exits cleanly so the post-barrier (audit/stage/ctx/ingest)
    runs on whatever parsed before the breaker tripped.

    Returns when every notebook reaches stage_ready AND all ParseItems consumed.
    """
    last_progress_log = time.monotonic()

    while True:
        # Check termination condition
        if rp.fetch_total == 0:
            done = True  # nothing was ever enqueued
        elif do_fetch and do_parse:
            all_fetched = rp.fetch_done_count >= rp.fetch_total
            done = all_fetched and rp.parse_pending == 0
        elif do_fetch:
            done = rp.fetch_done_count >= rp.fetch_total
        elif do_parse:
            done = rp.parse_pending == 0
        else:
            done = True

        if done:
            break

        # result_q.get(timeout=N) does NOT raise queue.Empty after N seconds —
        # the timeout only applies to the internal lock, which is uncontested
        # (always acquired instantly), then _recv_bytes() blocks on the pipe
        # forever. Use _reader.poll() (real select-based timeout) instead.
        if result_q._reader.poll(_COORDINATOR_POLL_TIMEOUT):  # type: ignore[attr-defined]
            event = result_q.get()
        else:
            # Log progress periodically
            now = time.monotonic()
            if now - last_progress_log > _BACKPRESSURE_WARN_S:
                _log_progress(rp)
                last_progress_log = now
            # Daemon heartbeat — cheap when alive, time-gated inside tick().
            if supervisor is not None:
                supervisor.tick()
            continue

        kind = event[0]

        if kind == "fetch_ok":
            _, nb, src_id, has_pdf, pdf_path = event
            rp.progress[nb]["fetched"] += 1
            rp.fetch_done_count += 1
            if has_pdf and do_parse:
                # Circuit-break path: don't enqueue more parse work, just
                # mark this PDF failed so the run can wind down on whatever
                # already parsed cleanly.
                if supervisor is not None and supervisor.is_circuit_open():
                    print(
                        f"  ✗ parse_fail: {nb} pdf={pdf_path} "
                        f"reason=daemon_circuit_breaker"
                    )
                else:
                    rp.progress[nb]["pdfs_expected"] += 1
                    parse_q.put(ParseItem(nb_tag=nb, pdf_path=Path(pdf_path)))
                    rp.parse_pending += 1

        elif kind == "fetch_fail":
            _, nb, src_id, reason = event
            rp.progress[nb]["fetched"] += 1
            rp.fetch_done_count += 1
            print(f"  ✗ fetch_fail: {nb} src={src_id} reason={reason}")

        elif kind == "parse_start":
            _, nb, pdf_path = event
            print(f"  → parse_start: {nb} pdf={pdf_path}")
            # gh #17: tell the supervisor which slug is now in flight so
            # a SIGABRT can be attributed to it for crasher correlation.
            if supervisor is not None:
                slug = Path(pdf_path).parent.name
                if hasattr(supervisor, "note_parse_start"):
                    supervisor.note_parse_start(slug)

        elif kind == "parse_ok":
            _, nb, pdf_path = event
            rp.progress[nb]["pdfs_parsed"] += 1
            rp.parse_pending -= 1
            if supervisor is not None and hasattr(supervisor, "note_parse_done"):
                supervisor.note_parse_done(Path(pdf_path).parent.name, ok=True)
            _advance_source_parsed(Path(pdf_path).parent)

        elif kind == "parse_fail":
            _, nb, pdf_path, reason = event
            rp.progress[nb]["pdfs_expected"] -= 1  # so stage_ready still reachable
            rp.parse_pending -= 1
            print(f"  ✗ parse_fail: {nb} pdf={pdf_path} reason={reason}")
            if supervisor is not None and hasattr(supervisor, "note_parse_done"):
                supervisor.note_parse_done(Path(pdf_path).parent.name, ok=False)
            # Reactive supervisor check — if the failure reason looks like
            # daemon death, probe NOW instead of waiting up to 10s for the
            # next heartbeat. Bounds death-to-respawn latency.
            if supervisor is not None and supervisor.note_parse_fail_reason(reason):
                ev = supervisor.tick(force=True)
                # gh #17: if the supervisor's tracker confirmed a crasher
                # on this death, quarantine the suspect's source dir so
                # subsequent runs skip it.
                outcome = getattr(supervisor, "last_crasher_outcome", None)
                if outcome is not None and sources_root is not None:
                    _maybe_quarantine_confirmed_crasher(
                        outcome=outcome, collection=nb,
                        sources_root=sources_root,
                        last_completed_pdf=getattr(
                            supervisor, "_last_completed_pdf", None,
                        ),
                    )
                # Circuit_break propagation (gh #16): if this tick tripped
                # the breaker, drain parse_q + synthesise fetch_fail for
                # un-arrived items so coordinator can terminate cleanly
                # instead of waiting minutes for workers to slow-fail
                # every queued item with Connection refused.
                if ev is not None and ev.kind == "circuit_break":
                    _propagate_circuit_break(rp, parse_q, do_fetch, do_parse)

        elif kind == "worker_crash":
            _, worker_kind, exc_str = event
            print(f"  ✗ worker_crash: {worker_kind} crashed: {exc_str}")


def _log_progress(rp: _RunProgress) -> None:
    """Print one-line progress summary for all notebooks."""
    parts = []
    for tag in sorted(rp.progress):
        p = rp.progress[tag]
        parts.append(
            f"{tag}:{p['fetched']}/{p['total']} "
            f"pdf:{p['pdfs_parsed']}/{p['pdfs_expected']}"
        )
    print(f"  [progress] {' | '.join(parts)} (parse_pending={rp.parse_pending})")


def _propagate_circuit_break(
    rp: _RunProgress,
    parse_q: multiprocessing.Queue[Any] | None,
    do_fetch: bool,
    do_parse: bool,
) -> None:
    """Unblock coordinator termination after the daemon breaker trips (gh #16).

    Two responsibilities:
      1. Drain ``parse_q`` of every pending ParseItem and decrement
         ``parse_pending`` / ``pdfs_expected`` once per drained item. Without
         this, parse workers spend minutes slow-failing each queued PDF
         with ConnectionRefused — wasteful and the log looks frozen.
      2. Synthesise ``fetch_fail`` accounting for un-arrived fetches so
         ``fetch_done_count`` catches up to ``fetch_total``. The spider
         thread is uninterruptible from here, so we pre-count its in-flight
         items as failures; any genuine ``fetch_ok`` events that still
         arrive will overshoot harmlessly (handled by the termination
         check, which is ``>=`` not ``==``).

    Idempotent within a single run — second call is a no-op (queue empty,
    fetch_total already reached).
    """
    if do_parse and parse_q is not None:
        import queue as _queue  # noqa: PLC0415
        n_drained = 0
        n_pills_held = 0
        # Multiprocessing queues use a background feeder thread, so a put()
        # from a producer may not be visible to get_nowait() for a few ms.
        # Retry briefly to give the feeder time to flush before declaring
        # the queue empty.
        empty_attempts = 0
        while empty_attempts < 5:
            try:
                item = parse_q.get_nowait()
                empty_attempts = 0  # got something — reset
            except (_queue.Empty, OSError):
                empty_attempts += 1
                time.sleep(0.05)
                continue
            if item is None:
                # Poison pill — count it and re-enqueue AFTER the drain
                # finishes (re-enqueueing inside the loop would re-pop
                # forever).
                n_pills_held += 1
                continue
            n_drained += 1
            nb = getattr(item, "nb_tag", None)
            pdf_path = getattr(item, "pdf_path", "?")
            if nb is not None and nb in rp.progress:
                rp.progress[nb]["pdfs_expected"] -= 1
            rp.parse_pending = max(0, rp.parse_pending - 1)
            print(
                f"  ✗ parse_fail: {nb} pdf={pdf_path} "
                f"reason=daemon_circuit_breaker"
            )
        # Restore any pills we removed so workers can shut down cleanly.
        for _ in range(n_pills_held):
            try:
                parse_q.put_nowait(None)
            except (_queue.Full, OSError):
                pass
        if n_drained:
            print(
                f"  ✗ daemon_event: drained {n_drained} pending parse items "
                f"after circuit_break"
            )

    if do_fetch:
        deficit = rp.fetch_total - rp.fetch_done_count
        if deficit > 0:
            # Distribute to notebooks proportionally to their remaining
            # un-fetched totals so per-notebook accounting stays honest.
            for tag, p in rp.progress.items():
                missing = max(0, p["total"] - p["fetched"])
                if missing == 0:
                    continue
                p["fetched"] += missing
                rp.fetch_done_count += missing
                print(
                    f"  ✗ fetch_fail: {tag} "
                    f"(synthesised {missing} on circuit_break)"
                )


def _maybe_quarantine_confirmed_crasher(
    *,
    outcome: CrasherOutcome,
    collection: str,
    sources_root: Path,
    last_completed_pdf: str | None,
) -> None:
    """If *outcome* confirms a crasher, quarantine its source dir.

    Called by the coordinator after each death-tick.
    """
    if not outcome.confirmed:
        return
    suspect = outcome.suspect
    if suspect is None:
        return
    from src.pipeline.quarantine import (  # noqa: PLC0415
        QuarantineSidecar,
        quarantine_source,
    )

    sidecar = QuarantineSidecar(
        reason="mineru_pipeline_glibc_corruption",
        deaths_observed=outcome.deaths_observed,
        predecessor_pdf=last_completed_pdf or "",
        daemon_log_refs=list(outcome.daemon_log_refs),
        url="",  # filled in if/when the sidecar grows a meta.json reader
    )
    new_path = quarantine_source(
        sources_root=sources_root, collection=collection,
        slug=suspect, sidecar=sidecar,
    )
    print(
        f"  ✗ quarantined: {collection}/{suspect} → {new_path} "
        f"(deaths={sidecar.deaths_observed})",
        flush=True,
    )


def _print_run_summary(
    *, supervisor: DaemonSupervisor | None,
    progress: dict[str, dict[str, int]],
) -> None:
    """Print a single human-readable summary line at run exit (gh #16).

    Surfaces per-collection ok / fail / pending counts so operators see
    the run outcome without grepping journalctl. Always prints, even on
    clean exits — the line tells you what landed in Milvus.
    """
    is_open = supervisor is not None and supervisor.is_circuit_open()
    result = "circuit_break" if is_open else "ok"

    parts: list[str] = []
    total_ok = 0
    total_fail = 0
    for tag in sorted(progress):
        p = progress[tag]
        ok = p.get("pdfs_parsed", 0)
        # pdfs_expected was decremented on parse_fail and on circuit_break
        # drain, so failures are not directly recoverable from the dict
        # without separate tracking. The cheapest honest metric is
        # "fetched but not parsed = unattempted-or-failed".
        fail = max(0, p.get("fetched", 0) - p.get("pdfs_parsed", 0)
                   - max(0, p.get("pdfs_expected", 0) - p.get("pdfs_parsed", 0)))
        total_ok += ok
        total_fail += fail
        parts.append(f"{tag}:ok={ok}/fail={fail}")

    print(
        f"=== ragctl run summary: result={result} "
        f"ok={total_ok} fail={total_fail} | {' '.join(parts)} ==="
    )


# ---------------------------------------------------------------------------
# Stage helpers (invoked in main process post-barrier)
# ---------------------------------------------------------------------------

def _run_audit(root: Path) -> int:
    """Run corpus-wide audit + apply. Returns worst rc (0 on success)."""
    audit_modules = [
        "src.cleanup.filename_content_match_audit",
        "src.cleanup.cross_partition_dedup_audit",
        "src.cleanup.build_cleanup_manifest",
    ]
    max_rc = 0
    for mod in audit_modules:
        rc = subprocess.run(
            ["uv", "run", "python", "-m", mod],
            cwd=root, check=False,
        ).returncode
        if rc != 0:
            max_rc = rc
    if max_rc != 0:
        return max_rc

    # Apply manifest
    manifest_path = root / "logs" / "cleanup_manifest.json"
    if not manifest_path.exists():
        return 1
    rc = subprocess.run(
        ["uv", "run", "python", "-m", "src.cleanup.apply_cleanup_manifest"],
        cwd=root, check=False,
    ).returncode
    return rc


def _print_dry_run(notebooks: list[dict[str, Any]], stages: list[str],
                   parse_workers: int) -> None:
    """Print a human-readable plan of what would run."""
    pre = [s for s in stages
           if s in orchestrate.PRE_BARRIER_STAGES and s != "stage"]
    post = [s for s in stages if s in orchestrate.POST_BARRIER_STAGES]
    do_audit = orchestrate.BARRIER_STAGE in stages

    print("WOULD RUN:")
    collections = sorted({nb["collection"] for nb in notebooks})
    if any(nb.get("id") for nb in notebooks):
        print(f"  fetch UUIDs:      {len(notebooks)}")
        for nb in notebooks:
            print(f"                     - {nb.get('id', '?'):<40} → "
                  f"{nb['collection']}")
    print(f"  collections:      {', '.join(collections) if collections else '(none)'}")
    print(f"  stages:           [{', '.join(stages)}]")

    has_fetch = "fetch" in pre
    has_parse = "parse_pipeline" in pre
    print("  workers:")
    if has_fetch:
        print("    fetch:          spider (concurrent_requests=30)")
    if has_parse:
        cap_note = (
            "  (MinerU daemon caps at ~3 — extra workers will queue)"
            if parse_workers > 3 else ""
        )
        print(f"    parse:          {parse_workers}{cap_note}")
    print(f"  post-barrier:     sequential per-collection "
          f"({' → '.join(post) if post else '(none)'})")
    if do_audit:
        print("  barrier:          audit + apply (corpus-wide, single run)")
    print("  estimated:        ~unknown (run without --dry-run to measure)")


# ---------------------------------------------------------------------------
# Main orchestrator entry point
# ---------------------------------------------------------------------------

def run_pipeline(
    notebooks: list[dict[str, str]],
    stages: list[str],
    *,
    root: Path | None = None,
    parallel: int = 5,
    parse_workers: int | None = None,
    no_cffi: bool = False,
    dry_run: bool = False,
    retry_failed: bool = False,
    # Test hooks (underscore-prefixed)
    _mp_ctx: object = None,
    _fetch_fn: object = None,
    _parse_fn: object = None,
) -> int:
    """Run fetch+parse streaming pipeline across *notebooks*.

    Args:
        notebooks: List of notebook dicts from ``select_notebooks``.
        stages: Ordered list from ``select_stages``.
        root: Project root. Defaults to the repo root.
        parallel: Number of parse workers (also the fetch worker count when a
            test stub is injected via ``_fetch_fn``; production fetch goes
            through the Spider with internal concurrency=30 regardless).
        parse_workers: Explicit parse worker count.
        no_cffi: Forwarded to test-stub fetch fns; ignored by the Spider.
        dry_run: Print plan and exit.
        retry_failed: Filter to sources whose last fetch did not produce an ok
            ``web.txt`` or ``source.pdf``; evict their URL-cache entries so the
            spider writes fresh.

    Returns:
        Exit code (0 = success). Non-zero if any notebook's post-barrier failed.
    """
    # Force line buffering on stdout/stderr so progress is visible when output
    # is piped to a log file (default Python uses block buffering for non-tty).
    try:
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]
        sys.stderr.reconfigure(line_buffering=True)  # type: ignore[union-attr]
    except (AttributeError, OSError):
        pass  # already line-buffered or non-reconfigurable stream

    if root is None:
        root = Path(__file__).resolve().parent.parent.parent

    pre = [s for s in stages if s in orchestrate.PRE_BARRIER_STAGES]
    post = [s for s in stages if s in orchestrate.POST_BARRIER_STAGES]
    do_audit = orchestrate.BARRIER_STAGE in stages

    do_fetch = "fetch" in pre
    do_parse = "parse_pipeline" in pre

    # Production fetch goes through the Spider (one process, internal concurrency
    # of 30 via FetchPipelineSpider.concurrent_requests). Test-stub injection via
    # ``_fetch_fn`` triggers the legacy per-URL worker pool with --parallel workers.
    using_spider = do_fetch and _fetch_fn is None
    n_fetch = 0 if using_spider else (parallel if do_fetch else 0)
    # MinerU pipeline daemon caps internal request concurrency at 3
    # (see "Request concurrency limited to 3" in its startup log). More than
    # 3 parse workers just queue at the daemon — wasted forkserver memory.
    # Explicit parse_workers= overrides the cap (escape hatch for future tuning).
    if parse_workers is not None:
        n_parse = parse_workers
    else:
        n_parse = min(parallel, 3) if do_parse else 0

    if dry_run:
        _print_dry_run(notebooks, stages, n_parse)
        return 0

    # Acquire the global run lock — only one ragctl run at a time.
    # See _acquire_run_lock for rationale (concurrent runs were observed
    # mutually SIGKILLing each other's daemons, then deadlocking).
    try:
        run_lock_fd = _acquire_run_lock()
    except RunLockHeldError as exc:
        print(f"REFUSING TO RUN: {exc}")
        return 2
    # The kernel releases flock on process death, but register atexit so
    # the lockfile bookkeeping (including the holder stamp) is cleaned up
    # promptly on normal exit too.
    import atexit  # noqa: PLC0415
    atexit.register(_release_run_lock, run_lock_fd)

    fetch_label = (
        "spider(concurrent_requests=30)" if using_spider
        else f"fetch_workers={n_fetch}" if do_fetch
        else "fetch=skipped"
    )
    print(f"=== ragctl run: {len(notebooks)} notebooks, "
          f"stages={stages}, {fetch_label}, parse_workers={n_parse} ===")

    # --- Resolve sources & seed queues ---
    rp = _RunProgress()
    fetch_items: list[FetchItem] = []
    parse_items: list[ParseItem] = []

    sources_root = root / "sources"

    # Build a per-collection progress bucket. Multi-UUID fetches into the
    # same collection share that bucket; parse-only mode has one bucket per
    # collection in ``collections``. ``nb_tag`` on FetchItems / ParseItems
    # is always the target collection name (the spider writes
    # ``sources/<collection>/<slug>/`` with no extra subroot).
    collections = sorted({nb["collection"] for nb in notebooks})
    for coll in collections:
        rp.progress[coll] = {
            "total": 0, "fetched": 0, "pdfs_expected": 0, "pdfs_parsed": 0,
        }

    if do_fetch:
        print(f"resolving sources for {len(notebooks)} notebooks via notebooklm CLI...")
    started_resolve = time.monotonic()
    for nb_idx, nb in enumerate(notebooks, 1):
        coll = nb["collection"]
        if do_fetch:
            t0 = time.monotonic()
            sources = _resolve_sources_for_notebook(nb)
            print(f"  [{nb_idx:2d}/{len(notebooks)}] {coll:<45} {len(sources):>4d} sources "
                  f"({time.monotonic() - t0:.1f}s)")
            rp.progress[coll]["total"] += len(sources)
            for src in sources:
                rp.fetch_total += 1
                fetch_items.append(FetchItem(
                    nb_tag=coll, nb_id=nb["id"],
                    src=src, no_cffi=no_cffi,
                ))

    if not do_fetch and do_parse:
        # Parse-only mode: discover every existing source.pdf under each
        # selected collection.
        pdfs = _discover_existing_pdfs(collections, sources_root)
        for pdf in pdfs:
            rp.progress[pdf.nb_tag]["pdfs_expected"] += 1
        rp.fetch_total = len(pdfs)
        parse_items.extend(pdfs)
    elif do_fetch and do_parse:
        # Mixed mode: discover any PDFs already on disk so prior fetches don't
        # need to re-run; new fetches enqueue ParseItems via the coordinator.
        pdfs = _discover_existing_pdfs(collections, sources_root)
        for pdf in pdfs:
            rp.progress[pdf.nb_tag]["pdfs_expected"] += 1
        parse_items.extend(pdfs)

    if do_fetch:
        print(f"resolved {rp.fetch_total} sources across {len(notebooks)} notebooks "
              f"in {time.monotonic() - started_resolve:.1f}s")

        # Seed the cross-run URL cache from existing meta.json files so
        # ``--retry-failed`` has accurate per-URL state.
        from src.fetch.util import (  # noqa: PLC0415
            evict_url_cache_entries,
            seed_url_cache_from_existing,
        )
        n_seeded = seed_url_cache_from_existing()
        if n_seeded:
            print(f"seeded URL cache: +{n_seeded} entries from existing source dirs")

        # --retry-failed: drop sources whose last fetch already produced an ok
        # web.txt or source.pdf, and evict their cache entries so the spider
        # writes fresh files (not hardlinks of stale partials).
        if retry_failed:
            n_before = len(fetch_items)
            fetch_items, urls_to_evict = _filter_failed_only(fetch_items, sources_root)
            n_evicted = evict_url_cache_entries(urls_to_evict)
            n_dropped = n_before - len(fetch_items)
            print(
                f"--retry-failed: kept {len(fetch_items)} failed sources "
                f"(dropped {n_dropped} successful), evicted {n_evicted} cache entries",
            )
            # Recompute per-notebook totals so _stage_ready doesn't deadlock
            # waiting for fetches that won't happen.
            for tag in rp.progress:
                rp.progress[tag]["total"] = 0
            for item in fetch_items:
                rp.progress[item.nb_tag]["total"] += 1
            rp.fetch_total = len(fetch_items)

        # Dedup at seed time: drop duplicate FetchItems (same canonical URL)
        # before they hit the queue. This prevents wasted fetch + parse work
        # and avoids the coordinator-stall bug where phantom items inflate
        # fetch_total without ever producing a fetch_ok event.
        from src.fetch.classify import canonical_url, slug  # noqa: PLC0415
        seen: dict[str, FetchItem] = {}
        deduped_items: list[FetchItem] = []
        n_within = 0
        n_cross = 0
        for item in fetch_items:
            cu = canonical_url(item.src)
            if not cu:
                deduped_items.append(item)
                continue
            if cu in seen:
                # Write dedup pointer in the duplicate's source dir so stage
                # can hardlink the canonical's text later.
                canon = seen[cu]
                item_title = str(item.src.get('title') or 'untitled')
                dup_dir = sources_root / item.nb_tag / slug(item_title)
                dup_dir.mkdir(parents=True, exist_ok=True)
                canon_name = slug(str(canon.src.get('title') or 'untitled'))
                (dup_dir / "_dedup_pointer.json").write_text(json.dumps({
                    "canonical_nb_tag": canon.nb_tag,
                    "canonical_dir_name": canon_name,
                    "canonical_url": cu,
                }))
                # Decrement the notebook's total so _stage_ready doesn't wait
                # for a fetch that will never happen.
                rp.progress[item.nb_tag]["total"] -= 1
                if item.nb_tag == canon.nb_tag:
                    n_within += 1
                else:
                    n_cross += 1
                continue
            seen[cu] = item
            deduped_items.append(item)

        n_orig = len(fetch_items)
        fetch_items = deduped_items
        # Critical: clobber fetch_total to match the deduped queue size.
        # Without this, the coordinator's termination check
        # (fetch_done_count >= fetch_total) never fires because phantom
        # items are still counted.
        rp.fetch_total = len(fetch_items)
        n_dupes = n_orig - len(fetch_items)
        if n_dupes:
            print(f"deduped {n_dupes} of {n_orig} sources "
                  f"({n_within} within + {n_cross} cross) "
                  f"— fetch queue size now {len(fetch_items)}")

        # Interleave items by notebook to spread per-host pacing across workers.
        # Without this, all 5 workers hit the same notebook's sources
        # consecutively, causing them to serialize on per-host pace() locks.
        # Round-robin across notebooks instead so adjacent items hit different
        # hosts → workers run in parallel instead of serializing on one host.
        from collections import defaultdict  # noqa: PLC0415
        by_nb: dict[str, list[FetchItem]] = defaultdict(list)
        for item in fetch_items:
            by_nb[item.nb_tag].append(item)
        interleaved: list[FetchItem] = []
        nb_iters = [iter(by_nb[k]) for k in by_nb]
        while nb_iters:
            next_iters = []
            for it in nb_iters:
                try:
                    interleaved.append(next(it))
                    next_iters.append(it)
                except StopIteration:
                    pass
            nb_iters = next_iters
        fetch_items = interleaved
        print(f"interleaved fetch queue across {len(by_nb)} notebooks "
              f"(round-robin) — adjacent items hit different hosts")

    # --- Phase 1: Fetch + Parse via multiprocessing queues ---
    # Only run the queue machinery when there is actual work to do.
    has_queue_work = (do_fetch and len(fetch_items) > 0) or (do_parse and len(parse_items) > 0)

    # Hoisted so the post-barrier overall_rc check sees it whether or not
    # has_queue_work fired.
    supervisor: DaemonSupervisor | None = None

    if has_queue_work:
        spawn_label = (
            f"spider(internal=30) + {n_parse} parse workers" if using_spider
            else f"{n_fetch} fetch + {n_parse} parse workers"
        )
        parse_q_str = "20" if do_parse else "n/a"
        fetch_q_str = "spider" if using_spider else ("200" if do_fetch else "n/a")
        print(f"spawning {spawn_label}, queues: fetch={fetch_q_str} parse={parse_q_str}")
        ctx: Any = _mp_ctx or multiprocessing.get_context("forkserver")
        mgr = ctx.Manager()
        pacer = SharedPacer(mgr, default_delay_s=1.0)

        # Production fetch goes through the Spider; only the test stub path
        # needs the per-URL fetch_q.
        fetch_q: Any = ctx.Queue(maxsize=200) if (do_fetch and not using_spider) else None
        parse_q: Any = ctx.Queue(maxsize=20) if do_parse else None
        result_q: Any = ctx.Queue()

        # Start MinerU daemon BEFORE spawning workers (parse workers need it)
        daemon_started = False
        if do_parse:
            try:
                from src.pdf_parsers import mineru_daemon  # noqa: PLC0415
                mineru_daemon.start_pipeline_daemon()
                daemon_started = True
                from src.pipeline.crasher_tracker import (  # noqa: PLC0415
                    CrasherTracker,
                )
                supervisor = DaemonSupervisor(crasher_tracker=CrasherTracker())
                print("  ✓ daemon_event: start")
            except Exception as exc:
                print(f"WARNING: Could not start MinerU daemon ({exc}); "
                      "parse_pipeline will be skipped")
                do_parse = False

        # Spawn workers BEFORE seeding queues — if queues are seeded first
        # and the number of items exceeds the queue maxsize, the main process
        # blocks on put() with no consumers to drain the queue (deadlock).
        workers: list[tuple[str, Any]] = []
        if using_spider:
            # Spider uses Scrapling's internal anyio event loop — anyio.run()
            # creates its own loop which conflicts with multiprocessing context
            # (both forkserver and spawn break).  Run it inline in a background
            # thread so it feeds result_q without a process boundary.
            threading.Thread(
                target=spider_fetch_all,
                args=(list(fetch_items), result_q, sources_root),
                name="pipeline-spider",
                daemon=True,
            ).start()
        elif do_fetch:
            # Test stub path: per-URL fetch workers driven by ``_fetch_fn``.
            for _ in range(n_fetch):
                p = ctx.Process(
                    target=fetch_worker,
                    args=(fetch_q, result_q, pacer, sources_root, no_cffi, _fetch_fn),
                )
                p.start()
                workers.append(("fetch", p))
        if do_parse:
            for _ in range(n_parse):
                p = ctx.Process(
                    target=parse_worker,
                    args=(parse_q, result_q, _parse_fn),
                )
                p.start()
                workers.append(("parse", p))

        # Seed queues in parallel daemon threads so coordinator_loop can run
        # concurrently. Two reasons:
        #   1. With large corpora (3446 items, queue maxsize=200), serial
        #      seed-then-coordinate would starve the coordinator for tens of
        #      minutes — fetch workers write source.pdf to disk but coordinator
        #      never runs to push ParseItems onto parse_q → daemon idle, GPU 0%.
        #   2. parse_q (maxsize=20) fills fast with existing PDFs; if seeded in
        #      the same thread as fetch_q, fetch workers stay idle until parse
        #      drains (~30s each = minutes of delay). Separate threads decouple.
        def _seed_parse() -> None:
            if do_parse and parse_q is not None:
                for item in parse_items:
                    parse_q.put(item)
        def _seed_fetch() -> None:
            if fetch_q is not None:
                for item in fetch_items:
                    fetch_q.put(item)

        # Pre-count parse_pending for static parse_items so coordinator's done check
        # is consistent regardless of seed thread progress.
        if do_parse and parse_items:
            rp.parse_pending += len(parse_items)

        if do_parse:
            threading.Thread(target=_seed_parse, name="pipeline-seed-parse", daemon=True).start()
        # Spider receives its source list directly (passed to spider_fetch_all
        # via Process args), so fetch_q seeding only happens for the test path.
        if do_fetch and not using_spider:
            threading.Thread(target=_seed_fetch, name="pipeline-seed-fetch", daemon=True).start()

        # Coordinator loop (blocks until all work done)
        try:
            coordinator_loop(
                rp, fetch_q, parse_q, result_q, do_fetch, do_parse,
                supervisor=supervisor,
                sources_root=sources_root,
            )
        except KeyboardInterrupt:
            print("\nInterrupted — draining queues and stopping workers...")
        finally:
            # Poison pills
            if do_fetch and fetch_q is not None:
                for _ in range(n_fetch):
                    try:
                        fetch_q.put(_POISON, timeout=1)
                    except Exception:
                        pass
            if do_parse and parse_q is not None:
                for _ in range(n_parse):
                    try:
                        parse_q.put(_POISON, timeout=1)
                    except Exception:
                        pass

            # Join workers
            for _kind, p in workers:
                p.join(timeout=30)
                if p.is_alive():
                    p.terminate()
                    p.join(timeout=5)

            # Stop daemon
            if daemon_started:
                try:
                    from src.pdf_parsers import mineru_daemon  # noqa: PLC0415
                    mineru_daemon.stop_pipeline_daemon()
                except Exception:
                    pass

            mgr.shutdown()

    overall_rc = 0
    # Daemon-supervisor circuit-break is a partial-success signal: post-barrier
    # still runs on whatever parsed before the breaker tripped (so the corpus
    # doesn't lose ground), but the run as a whole exits non-zero so CI / cron
    # callers see the failure.
    if supervisor is not None and supervisor.is_circuit_open():
        overall_rc = max(overall_rc, 1)

    # --- Phase 2: Audit barrier ---
    if do_audit:
        print("\n=== barrier (audit + apply, corpus-wide) ===")
        rc = _run_audit(root)
        if rc != 0:
            overall_rc = max(overall_rc, rc)
            # Audit failure is load-bearing — halt the run
            return overall_rc

    # --- Phase 3: Sequential post-barrier (per-collection, not per-notebook) ---
    for coll in collections:
        if (do_fetch or do_parse) and not _stage_ready(coll, rp.progress):
            print(f"  ✗ {coll}: not stage_ready — skipping post-barrier")
            overall_rc = max(overall_rc, 1)
            continue

        for s in post:
            cmd: list[str] | None = None
            if s == "contextualize":
                cmd = orchestrate.build_contextualize_command(coll)
            elif s == "ingest":
                cmd = orchestrate.build_ingest_command(coll)
            if cmd is None:
                continue
            rc = subprocess.run(cmd, cwd=root, check=False).returncode
            status = "✓" if rc == 0 else "✗"
            print(f"  {status} {coll:<32} {s} (rc={rc})")
            if rc != 0:
                overall_rc = max(overall_rc, rc)
                break

    # gh #16: surface a single summary line so operators can see at a glance
    # whether the run ended cleanly or via circuit_break, and how much
    # actually landed.
    _print_run_summary(supervisor=supervisor, progress=rp.progress)

    return overall_rc
