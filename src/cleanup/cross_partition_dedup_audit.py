"""Cross-partition near-dup audit for trading sources.

Reads every per-source text file under ``sources/<collection>/<slug>/`` and
groups by:
- exact-content hash (full-text SHA256, after the section-tail stripper)
- soft hash (sha256 of first 2KB after lowercasing + collapsing whitespace)

Reports any collisions where the same paper appears in 2+ collections.

Read-only — does not modify any files. Output: stdout summary + JSON manifest
at logs/cross_partition_dedup.json for review before any consolidation action.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.ingest import _strip_tail_sections

ROOT = Path(__file__).resolve().parent.parent.parent
# `sources/<collection>/<slug>/<leaf>.txt` — pdf.txt | nlm.txt | web.txt.
SOURCE_TEXT_GLOBS = ["sources/*/*/*.txt"]
SOFT_HASH_BYTES = 2048


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _exact_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:16]


def _soft_hash(text: str) -> str:
    norm = _normalize(text)[:SOFT_HASH_BYTES]
    return hashlib.sha256(norm.encode("utf-8", errors="ignore")).hexdigest()[:16]


def _partition_of(path: Path) -> str:
    """Collection is the grandparent of a leaf text in the post-flatten layout.

    ``sources/<collection>/<slug>/<leaf>.txt`` → ``<collection>``.
    """
    parts = path.parts
    try:
        idx = parts.index("sources")
    except ValueError:
        return path.parent.name
    return parts[idx + 1] if idx + 1 < len(parts) else path.parent.name


def main() -> None:
    files: list[Path] = []
    for pat in SOURCE_TEXT_GLOBS:
        files.extend(ROOT.glob(pat))
    files = sorted(
        f for f in set(files)
        if not f.name.endswith(".ctx.json")
        and not any(p.startswith("_") for p in f.relative_to(ROOT / "sources").parts)
    )

    by_exact: dict[str, list[Path]] = defaultdict(list)
    by_soft: dict[str, list[Path]] = defaultdict(list)

    for f in files:
        text = f.read_text(encoding="utf-8", errors="ignore")
        clean, _ = _strip_tail_sections(text)
        eh = _exact_hash(clean)
        sh = _soft_hash(clean)
        by_exact[eh].append(f)
        by_soft[sh].append(f)

    def collisions(buckets: dict[str, list[Path]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for h, paths in buckets.items():
            if len(paths) < 2:
                continue
            partitions = {_partition_of(p) for p in paths}
            if len(partitions) < 2:
                continue  # same partition dupe — different problem
            out.append({
                "hash": h,
                "n_files": len(paths),
                "partitions": sorted(partitions),
                "files": [str(p.relative_to(ROOT)) for p in paths],
            })
        return sorted(out, key=lambda d: -d["n_files"])

    exact_collisions = collisions(by_exact)
    soft_collisions = collisions(by_soft)
    soft_only = [
        c for c in soft_collisions
        if c["hash"] not in {e["hash"] for e in exact_collisions}
    ]

    summary = {
        "n_files_scanned": len(files),
        "exact_cross_partition_dupes": len(exact_collisions),
        "soft_cross_partition_dupes": len(soft_collisions),
        "soft_only_cross_partition_dupes": len(soft_only),
        "exact": exact_collisions[:50],
        "soft_only": soft_only[:50],
    }

    out_path = ROOT / "logs" / "cross_partition_dedup.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))

    print(f"scanned: {len(files)} files across {len({_partition_of(f) for f in files})} partitions")
    print(f"exact cross-partition dupes: {len(exact_collisions)}")
    print(f"soft (first {SOFT_HASH_BYTES}B) cross-partition dupes: {len(soft_collisions)}")
    print(f"soft-only (not exact): {len(soft_only)}")
    print(f"\ndetailed manifest -> {out_path}")

    from src.pipeline.pipeline_log import StageLogger  # noqa: PLC0415
    with StageLogger("_corpus_", "audit") as sl:
        sl.item_ok("cross_partition_dedup",
                   n_files=len(files),
                   exact=len(exact_collisions),
                   soft=len(soft_collisions),
                   soft_only=len(soft_only))

    if exact_collisions:
        print("\n--- top 10 exact cross-partition dupes ---")
        for c in exact_collisions[:10]:
            print(f"  hash={c['hash']}  partitions={c['partitions']}")
            for f in c["files"]:
                print(f"    {f}")


if __name__ == "__main__":
    main()
