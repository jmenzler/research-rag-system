"""Ingest a NotebookLM Deep Research brief as a synthesized_brief.

Phase 2.7 path. Treats each Deep Research run as ephemeral compute: the user
exports the brief markdown manually, this script stages it as a single-doc
notebook under ``sources/<collection>/<slug>/`` and pushes it through ingest
with ``modality="synth_brief"`` so the chunks land in the ``research_briefs``
partition and remain filterable on modality at retrieval time.

Pipeline:
    1. Stage:   write source.md + content_list.json + meta.json
    2. Ingest:  call ``ingest_file`` directly with ``modality_override``.

Note on contextualization: the structural ingest path (which we use here, by
synthesizing a content_list.json from the markdown) does NOT consume the
``<source>.ctx.json`` sidecar today — only the hierarchical fallback does.
Contextualization is therefore skipped for briefs in MVP. Adding it later is
a structural-chunker change, not a brief-pipeline change.

Usage:
    python -m scripts.ingest_deep_research /tmp/brief_lvr.md \\
        --query "How does LVR scale with pool depth?" \\
        --notebook research_briefs \\
        --collection trading
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.nlm_to_content_list import parse_nlm  # noqa: E402
from src.ingest.chunker import _get_encoder  # noqa: E402
from src.ingest.ingest import _build_gemini_client, ingest_file  # noqa: E402
from src.ingest.storage import _open_parents_db  # noqa: E402
from src.milvus_client import ensure_collection, ensure_partition  # noqa: E402

MODALITY = "synth_brief"  # Milvus VARCHAR(16) cap → can't fit 'synthesized_brief'.

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slugify(text: str, max_len: int = 80) -> str:
    """Lowercase + dash-collapsed slug, capped at max_len."""
    slug = _SLUG_RE.sub("-", text.lower()).strip("-")
    return slug[:max_len].rstrip("-") or "untitled"


def _stage_brief(
    brief_path: Path,
    query: str,
    notebook: str,
    collection: str,
    sources_root: Path,
) -> Path:
    """Write source.md + content_list.json + meta.json to the slug dir.

    Returns the staged source.md path (the value to pass to ingest_file).
    Idempotent: re-staging overwrites content_list.json and meta.json so a
    re-run after fixing the brief markdown picks up the new structure.
    """
    text = brief_path.read_text(encoding="utf-8")
    if not text.strip():
        raise RuntimeError(f"brief is empty: {brief_path}")

    slug = _slugify(query)
    doc_dir = sources_root / collection / slug
    doc_dir.mkdir(parents=True, exist_ok=True)

    md_path = doc_dir / "source.md"
    md_path.write_text(text, encoding="utf-8")

    items = parse_nlm(text)
    if not items:
        raise RuntimeError(
            f"no structural items parsed from {brief_path} — check the markdown"
        )
    cl_path = doc_dir / "content_list.json"
    cl_path.write_text(
        json.dumps([it.to_dict() for it in items], ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )

    meta = {
        "title": query,
        "url": f"notebooklm://deep-research/{slug}",
        "host": "notebooklm.google.com",
        "tier": "T_synth_brief",
        "metadata": {
            "title": query,
            "authors": [],
            "year": datetime.now(UTC).year,
            "short_cite": f"Deep Research ({datetime.now(UTC).strftime('%Y-%m-%d')})",
            "confidence": "high",
            "arxiv_id": None,
            "doi": None,
            "source_kind": "synthesized_research_brief",
            "ingested_at": datetime.now(UTC).isoformat(),
            "query": query,
            "notebook": notebook,
            "n_items": len(items),
        },
    }
    (doc_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    return md_path


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="python -m scripts.ingest_deep_research",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("brief", type=Path, help="path to the Deep Research brief markdown")
    ap.add_argument("--query", required=True,
                    help="the original Deep Research question (used as title + slug)")
    ap.add_argument("--notebook", default="research_briefs",
                    help="partition tag (default: research_briefs)")
    ap.add_argument("--collection", default="trading",
                    help="Milvus collection (default: trading)")
    ap.add_argument("--sources-root", type=Path, default=ROOT / "sources",
                    help="corpus root for staging (default: ./sources)")
    args = ap.parse_args()

    if not args.brief.is_file():
        print(f"ERROR: brief not found: {args.brief}", file=sys.stderr)
        return 2

    print(f"staging {args.brief.name} → {args.collection}/{args.notebook}")
    md_path = _stage_brief(
        args.brief, args.query, args.notebook, args.collection, args.sources_root
    )
    print(f"  staged: {md_path.relative_to(ROOT)}")

    print("ingesting…")
    ensure_collection(args.collection)
    ensure_partition(args.collection, args.notebook)

    gemini_client = _build_gemini_client()
    enc = _get_encoder()
    db_conn = _open_parents_db()

    t0 = time.time()
    result = ingest_file(
        path=md_path,
        notebook=args.notebook,
        collection=args.collection,
        gemini_client=gemini_client,
        db_conn=db_conn,
        enc=enc,
        modality_override=MODALITY,
    )
    elapsed = time.time() - t0

    print(
        f"DONE in {elapsed:.1f}s  parents={result.get('n_parents', 0)}  "
        f"children={result.get('n_children', 0)}  "
        f"skipped={result.get('skipped', False)}"
        + (f"  reason={result.get('reason')}" if result.get('skipped') else "")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
