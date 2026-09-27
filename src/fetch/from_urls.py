"""Fetch sources from an explicit URL list — non-NotebookLM ingest path.

Mirrors ``src.fetch.cli`` end-to-end but accepts a hand-curated URL list
instead of a NotebookLM notebook UUID. Each URL becomes a synthetic ``src``
dict (the same shape ``classify`` and the spider expect) and is fed through
``FetchPipelineSpider`` unchanged. Downstream stages (parse_pdfs, audit,
apply, contextualize, ingest) operate on the resulting source dirs without
any awareness of how they were seeded.

Input file format (one URL per line):

  # Comments and blank lines are skipped
  https://arxiv.org/abs/2508.02435
  https://arxiv.org/pdf/2510.14278
  My Custom Title || https://example.com/blog/post

When the optional ``title || url`` form is used, the title is honored
verbatim; otherwise an initial slug is derived from the arxiv ID (when
present) or the URL path. The real document title is filled in later by
``src.fetch.metadata`` after the PDF is parsed.

CLI::

    python -m src.fetch.from_urls \\
        --urls urls.txt \\
        --collection system \\
        --notebook system
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from src.fetch.classify import classify, slug
from src.fetch.cli import VALID_COLLECTIONS, _resolve_target_dir, write_inventory
from src.fetch.metadata import extract_arxiv_id
from src.fetch.util import seed_url_cache_from_existing

logger = logging.getLogger(__name__)


def parse_url_file(path: Path) -> list[tuple[str, str]]:
    """Parse a URL file into ``[(title, url), ...]`` pairs.

    Empty lines and ``#``-prefixed comments are skipped. Lines may use the
    bare ``url`` form or ``title || url`` to override the auto-derived title.
    Whitespace around fields is stripped.
    """
    pairs: list[tuple[str, str]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "||" in line:
            title_str, url = (part.strip() for part in line.split("||", 1))
        else:
            title_str, url = "", line
        if not url:
            continue
        pairs.append((title_str, url))
    return pairs


def _derive_title(url: str, override: str) -> str:
    """Return the best initial title for ``url``.

    Honors a non-empty override. Otherwise uses the arxiv ID (when present)
    or the last non-empty URL path segment. Falls back to ``"untitled"``.
    """
    if override:
        return override
    arxiv_id = extract_arxiv_id(url)
    if arxiv_id:
        return f"arxiv:{arxiv_id}"
    parsed = urlparse(url)
    parts = [p for p in parsed.path.split("/") if p]
    if parts:
        return parts[-1].removesuffix(".pdf").removesuffix(".html")
    return parsed.netloc or "untitled"


def build_sources(pairs: list[tuple[str, str]]) -> list[dict[str, Any]]:
    """Build src-dict list compatible with ``classify`` and the spider.

    Each src has the minimum fields the downstream pipeline reads: ``index``,
    ``title``, ``url``, ``type``. The ``type`` is set to
    ``"SourceType.UNKNOWN"`` so ``classify`` routes purely on the URL host.
    """
    sources: list[dict[str, Any]] = []
    for idx, (title_override, url) in enumerate(pairs):
        sources.append({
            "index": idx,
            "title": _derive_title(url, title_override),
            "url": url,
            "type": "SourceType.UNKNOWN",
        })
    return sources


def _run_spider(
    sources: list[dict[str, Any]],
    *,
    collection: str,
    log_slug: str,
) -> list[Path]:
    """Drive ``FetchPipelineSpider`` over ``sources``; mirror ``fetch.cli`` IO.

    Pre-resolves per-source target dirs (same logic ``fetch.cli`` uses),
    writes meta.json + extract_summary, then walks the spider stream.
    Returns the list of doc_dirs that received fetched content (including
    skip-as-already-on-disk dirs — callers see the canonical on-disk slot
    for every input URL).
    """
    out_root = Path("sources") / collection
    out_root.mkdir(parents=True, exist_ok=True)

    inv_md = Path("logs") / f"{log_slug}_inventory.md"
    inv_json = Path("logs") / f"{log_slug}_inventory.json"
    summary_path = Path("logs") / f"{log_slug}_extract_summary.jsonl"
    summary_path.parent.mkdir(exist_ok=True)

    write_inventory(sources, log_slug, inv_md, inv_json)
    print(f"    inventory → {inv_md} / {inv_json}")

    n_seeded = seed_url_cache_from_existing()
    if n_seeded:
        print(f"[*] Seeded URL cache: +{n_seeded} entries from existing source dirs")

    spider_sources: list[dict[str, Any]] = []
    target_by_idx: dict[int, Path] = {}
    all_dirs: list[Path] = []
    for src in sources:
        target, action = _resolve_target_dir(out_root, src)
        all_dirs.append(target)  # callers want every URL's canonical slot
        if action == "skip":
            continue
        target_by_idx[src["index"]] = target
        s = dict(src)
        s["nb_tag"] = ""
        s["nb_id"] = log_slug
        s["slug_override"] = target.name
        s["tier"] = src.get("_tier", "T5_web_blog")
        spider_sources.append(s)

    skipped_dups = [d for d in all_dirs if d not in target_by_idx.values()]
    if skipped_dups:
        print(f"[*] Skipping {len(skipped_dups)} sources already on disk:")
        for d in skipped_dups[:8]:
            print(f"    - {d.name}")
        if len(skipped_dups) > 8:
            print(f"    … +{len(skipped_dups) - 8} more")

    if not spider_sources:
        print("Nothing to fetch.")
        return all_dirs

    from src.pipeline.pipeline_log import StageLogger  # noqa: PLC0415

    with StageLogger(log_slug, "fetch") as stage_logger, summary_path.open("w") as sf:
        import anyio  # noqa: PLC0415

        from src.fetch.spider import FetchPipelineSpider  # noqa: PLC0415

        os.environ["SPIDER_OUT_DIR"] = str(out_root)

        async def _run() -> list[dict[str, Any]]:
            spider = FetchPipelineSpider()
            spider._sources = spider_sources  # type: ignore[attr-defined]
            items = []
            async for item in spider.stream():
                items.append(item)
            return items

        items = anyio.run(_run)
        items_by_idx = {i["index"]: i for i in items}

        n = len(sources)
        for i, src in enumerate(sources, 1):
            idx = src["index"]
            title = src.get("title") or "untitled"
            tier = src.get("_tier", "T5_web_blog")

            target = target_by_idx.get(idx) or (out_root / slug(title))
            meta_path = target / "meta.json"
            meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"fetch": {}}
            sf.write(json.dumps(meta) + "\n")
            sf.flush()

            f = meta.get("fetch", {})
            any_ok = any((f.get(k) or {}).get("ok") for k in ("nlm", "pdf", "web"))
            if any_ok:
                stage_logger.item_ok(
                    target.name, tier=tier, title=title[:80],
                    nlm_ok=(f.get("nlm") or {}).get("ok", False),
                    pdf_ok=(f.get("pdf") or {}).get("ok", False),
                    web_ok=(f.get("web") or {}).get("ok", False),
                )
            else:
                reasons = []
                for k in ("nlm", "pdf", "web"):
                    v = f.get(k) or {}
                    if v and not v.get("ok"):
                        reasons.append(f"{k}:{v.get('reason', '?')}")
                stage_logger.item_fail(
                    target.name, tier=tier, title=title[:80],
                    reason="; ".join(reasons) or "no_artifacts",
                )

            spider_item = items_by_idx.get(idx, {})
            nlm_ok = "✓" if (f.get("nlm") or {}).get("ok") else "✗"
            pdf_state = "—"
            if "pdf" in f:
                f_pdf = f["pdf"]
                pdf_state = "✓" if f_pdf["ok"] else f"✗({(f_pdf.get('reason') or '?')[:25]})"
            web_state = "—"
            if "web" in f:
                f_web = f["web"]
                web_state = "✓" if f_web["ok"] else f"✗({(f_web.get('reason') or '?')[:25]})"
            session = spider_item.get("session", "?")
            print(
                f"[{i:3d}/{n}] {tier:22s} idx={idx:03d} "
                f"nlm={nlm_ok} pdf={pdf_state} web={web_state} sess={session} | {title[:50]}"
            )

    n_pdfs = sum(
        1
        for src in sources
        if (target_by_idx.get(src["index"]) or out_root / slug(src.get("title") or "untitled"))
            .joinpath("source.pdf").exists()
    )
    print(f"\nDone. Summary → {summary_path}")
    print(f"      Sources → {out_root}")
    if n_pdfs:
        print(f"\nNext step: parse {n_pdfs} downloaded PDF(s) via src.pdf_parsers.cli")
    return all_dirs


def fetch_from_url_file(
    urls_file: Path, collection: str, notebook: str,
) -> tuple[int, list[Path]]:
    """Programmatic entry point.

    Loads URLs from ``urls_file``, enriches each into the spider's src-dict
    shape, classifies for tier routing, and drives ``FetchPipelineSpider``.
    On-disk output lives at ``sources/<collection>/<slug>/``.

    Returns ``(rc, doc_dirs)``. ``rc`` is 0 on success, 1 if nothing to
    fetch. ``doc_dirs`` is the list of on-disk doc directories produced
    (including those skipped as already on disk, so callers can drive the
    full ingest pipeline restricted to exactly the URL set).
    """
    pairs = parse_url_file(urls_file)
    if not pairs:
        print(f"No URLs found in {urls_file}")
        return 1, []
    print(f"[*] Loaded {len(pairs)} URLs from {urls_file}")
    sources = build_sources(pairs)
    for src in sources:
        src["_tier"] = classify(src)
    dirs = _run_spider(sources, collection=collection, log_slug=notebook)
    return 0, dirs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch sources from an explicit URL list (non-NotebookLM path).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--urls", required=True, type=Path,
                        help="File with one URL per line. `title || url` form supported.")
    parser.add_argument("--collection", required=True, choices=VALID_COLLECTIONS,
                        help="Target collection root under sources/.")
    parser.add_argument("--notebook", required=True,
                        help="Notebook/partition tag for log file naming and ingest routing.")
    args = parser.parse_args()
    rc, _ = fetch_from_url_file(args.urls, args.collection, args.notebook)
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
