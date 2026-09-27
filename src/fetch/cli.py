"""Fetch all sources from a NotebookLM notebook UUID — NO PDF parsing.

This script ONLY downloads source artifacts (NLM fulltext, PDFs, web pages).
PDF→text parsing is a separate step — run ``src.pdf_parsers.cli`` after this.
The split lets the slow GPU-bound parse step run on different hardware (e.g. CUDA)
than the I/O-bound fetch step.

Pipeline (per source, run inside ``FetchPipelineSpider``):
  1. Spider fetches URL with http -> stealth -> stealth_max escalation
  2. NLM fulltext fallback   → nlm.txt
  3. Free-PDF discovery      → source.pdf (when an arxiv/PDF link is found)
  4. Trafilatura extraction  → web.txt

Tiers (download priority):
  T0_arxiv             arxiv abs/PDF — highest yield
  T1_doi               DOI/known publishers
  T2_ssrn              SSRN — direct download_id PDF
  T3_generic_pdf       direct .pdf URL not on ResearchGate
  T4_researchgate      DO NOT PROBE; IPv6 ban risk
  T5_web_blog          generic web — trafilatura
  T6_nlm_native        pasted_text / markdown — NLM only
  T7_nlm_uploaded_pdf  PDF uploaded to NLM, no public URL — NLM fulltext only

Outputs (per source dir):
  sources/<collection>/<slug>/{nlm.txt, source.pdf, web.txt, meta.json}
                                  pdf.txt is added by parse_pdfs.

CLI:
  python -m src.fetch.cli \\
      --notebook-id 3c96ff28-... \\
      --collection trading \\
      [--inventory-only] [--no-cffi]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from src.fetch.classify import canonical_url, classify, list_sources, slug
from src.fetch.util import seed_url_cache_from_existing

VALID_COLLECTIONS = ("trading", "ecology", "notes", "system", "poker", "security")


def _short_sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:6]


def _resolve_target_dir(out_root: Path, src: dict[str, Any]) -> tuple[Path, str]:
    """Return ``(target_dir, action)`` where action ∈ {"write", "skip"}.

    On slug collision with an existing dir for a different paper, suffix the
    new dir with a 6-char hash for disambiguation. Same-paper collision (matching
    canonical URL) returns ``("...", "skip")`` so the caller can reuse the
    existing dir without overwriting.
    """
    title = src.get("title") or "untitled"
    base_slug = slug(title)
    target = out_root / base_slug
    if not target.exists():
        return target, "write"
    existing_meta = target / "meta.json"
    new_cu = canonical_url(src)
    if existing_meta.exists() and new_cu:
        try:
            existing = json.loads(existing_meta.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
        if canonical_url(existing) == new_cu:
            return target, "skip"
    fingerprint_seed = new_cu or src.get("url") or json.dumps(src, sort_keys=True)
    return out_root / f"{base_slug}__{_short_sha(fingerprint_seed)}", "write"


def write_inventory(
    sources: list[dict[str, Any]], notebook_id: str, out_md: Path, out_json: Path,
) -> None:
    by_tier: dict[str, list[dict[str, Any]]] = defaultdict(list)
    host_counter: Counter[str] = Counter()
    for s in sources:
        s["_tier"] = classify(s)
        by_tier[s["_tier"]].append(s)
        if s.get("url"):
            host_counter[urlparse(s["url"]).netloc.lower()] += 1

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(sources, indent=2))

    md = [
        "# Notebook source inventory\n",
        f"**Notebook:** `{notebook_id}`  \n",
        f"**Total sources:** {len(sources)}\n\n",
        "## Tier breakdown\n\n| Tier | Count |\n|---|---:|\n",
    ]
    for tier in sorted(by_tier):
        md.append(f"| {tier} | {len(by_tier[tier])} |\n")
    md.append(f"| **TOTAL** | **{len(sources)}** |\n")

    md.append("\n## Top hosts\n\n")
    for host, n in host_counter.most_common(20):
        md.append(f"- {host}: {n}\n")

    md.append("\n## Per-tier source list\n")
    for tier in sorted(by_tier):
        md.append(f"\n### {tier} ({len(by_tier[tier])})\n")
        for s in by_tier[tier]:
            md.append(f"- `{s['index']:03d}` {s['title'][:90]}\n  {s.get('url') or ''}\n")
    out_md.write_text("".join(md))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract all sources from a NotebookLM notebook by UUID.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--notebook-id", required=True,
                        help="NotebookLM notebook UUID to fetch.")
    parser.add_argument("--collection", required=True,
                        choices=VALID_COLLECTIONS,
                        help="Target collection root under sources/. Required.")
    parser.add_argument("--slug",
                        default=None,
                        help="Short slug for log files (default: first 8 chars of notebook id)")
    parser.add_argument("--inventory-only", action="store_true",
                        help="Only classify; no fetching.")
    # Accepted as a no-op for compatibility with ``orchestrate.build_stage_command``
    # which still forwards the flag from ``ragctl run --no-cffi``. The Spider
    # owns its own escalation chain (http -> stealth -> stealth_max), so the
    # flag has no effect here.
    parser.add_argument("--no-cffi", action="store_true",
                        help="(no-op) Spider escalation handles all blocking layers.")
    args = parser.parse_args()

    nb_id: str = args.notebook_id
    collection: str = args.collection
    log_slug = args.slug or nb_id.split("-")[0]
    out_root = Path("sources") / collection
    out_root.mkdir(parents=True, exist_ok=True)

    inv_md = Path("logs") / f"{log_slug}_inventory.md"
    inv_json = Path("logs") / f"{log_slug}_inventory.json"
    summary_path = Path("logs") / f"{log_slug}_extract_summary.jsonl"
    summary_path.parent.mkdir(exist_ok=True)

    print(f"[*] Listing notebook {nb_id}…")
    sources = list_sources(nb_id)
    print(f"    found {len(sources)} sources")

    write_inventory(sources, nb_id, inv_md, inv_json)
    print(f"    inventory → {inv_md} / {inv_json}")

    if args.inventory_only:
        return

    n_seeded = seed_url_cache_from_existing()
    if n_seeded:
        print(f"[*] Seeded URL cache: +{n_seeded} entries from existing source dirs")

    n = len(sources)

    from src.pipeline.pipeline_log import StageLogger  # noqa: PLC0415

    # Pre-resolve per-source targets so the spider writes directly into the
    # final on-disk slot. Same-paper collisions become a no-op skip; slug
    # collisions across different papers get a suffix.
    spider_sources: list[dict[str, Any]] = []
    skipped_dups: list[tuple[str, str]] = []
    target_by_idx: dict[int, Path] = {}
    for src in sources:
        target, action = _resolve_target_dir(out_root, src)
        if action == "skip":
            skipped_dups.append((str(src.get("title") or "untitled")[:60], target.name))
            continue
        target_by_idx[src["index"]] = target
        s = dict(src)
        # Spider writes to <out_root>/<slug_override>/ — no extra subroot, no
        # NNN prefix. Title is preserved verbatim so meta.json carries the
        # original; slug_override is the disambiguated on-disk leaf name.
        s["nb_tag"] = ""
        s["nb_id"] = nb_id
        s["slug_override"] = target.name
        s["tier"] = src.get("_tier", "T5_web_blog")
        spider_sources.append(s)

    if skipped_dups:
        print(f"[*] Skipping {len(skipped_dups)} sources already on disk "
              f"(same canonical URL):")
        for title, slug_name in skipped_dups[:8]:
            print(f"    - {title:<60} → {slug_name}")
        if len(skipped_dups) > 8:
            print(f"    … +{len(skipped_dups) - 8} more")

    with StageLogger(log_slug, "fetch") as stage_logger, summary_path.open("w") as sf:

        import os  # noqa: PLC0415

        import anyio  # noqa: PLC0415

        from src.fetch.spider import FetchPipelineSpider  # noqa: PLC0415

        os.environ["SPIDER_OUT_DIR"] = str(out_root)

        async def _run_spider() -> list[dict[str, Any]]:
            spider = FetchPipelineSpider()
            spider._sources = spider_sources  # type: ignore[attr-defined]
            items = []
            async for item in spider.stream():
                items.append(item)
            return items

        items = anyio.run(_run_spider) if spider_sources else []
        items_by_idx = {i["index"]: i for i in items}

        for i, src in enumerate(sources, 1):
            idx = src["index"]
            title = src.get("title") or "untitled"
            tier = src["_tier"]

            target = target_by_idx.get(idx) or (out_root / slug(title))
            meta_path = target / "meta.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
            else:
                meta = {"fetch": {}}
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
        print(f"\nNext step: parse {n_pdfs} downloaded PDF(s) to text:")
        print(
            f"  uv run python -m src.pdf_parsers.cli "
            f"--collection {collection} --pass pipeline-only",
        )


if __name__ == "__main__":
    main()
