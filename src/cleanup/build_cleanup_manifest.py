"""CLI wrapper around `src.cleanup.manifest.build_manifest`.

Reads the two audit JSONs, runs the manifest builder, writes the result, and
prints a summary plus a sample of canonical assignments.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.cleanup.manifest import build_manifest

ROOT = Path(__file__).resolve().parent.parent.parent

NAME_AUDIT = ROOT / "logs" / "filename_content_match.json"
DUPE_AUDIT = ROOT / "logs" / "cross_partition_dedup.json"
OUT = ROOT / "logs" / "cleanup_manifest.json"


def main() -> None:
    from src.pipeline.pipeline_log import StageLogger  # noqa: PLC0415
    with StageLogger("_corpus_", "manifest") as sl:
        manifest = build_manifest(NAME_AUDIT, DUPE_AUDIT, ROOT)
        OUT.write_text(json.dumps(manifest, indent=2))

        s = manifest["summary"]
        sl.item_ok("manifest",
                   n_quarantine_poisoned=s["n_quarantine_poisoned"],
                   n_quarantine_dupes=s["n_quarantine_dupes"],
                   n_canonical_groups=s["n_canonical_groups"])

    print("=== cleanup manifest ===")
    print(
        f"  poisoned (fail-page + name-mismatch combiner + no-canonical): "
        f"{s['n_quarantine_poisoned']}"
    )
    print(
        f"  dupes to move (real cross-partition dupes, non-canonical):     "
        f"{s['n_quarantine_dupes']}"
    )
    print(
        f"  canonical groups preserved:                                    "
        f"{s['n_canonical_groups']}"
    )
    print(f"\nwrote {OUT}")
    print()
    print("=== sample of canonical assignments (best filename-vs-content overlap wins) ===")
    for entry in manifest["keep_canonical"][:5]:
        print(f"  hash={entry['hash']}  best_overlap={entry['best_overlap']}")
        print(f"    KEEP  {entry['canonical']}")
        for d in entry["drop"]:
            print(f"    drop  {d}  (overlap={entry['scores'][d]})")
        print()


if __name__ == "__main__":
    main()
