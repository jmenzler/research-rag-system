"""Apply the corpus cleanup manifest by moving flagged files to quarantine dirs.

Reads logs/cleanup_manifest.json. Moves:
- quarantine_poisoned → sources/_quarantine_poisoned/
- quarantine_dupes    → sources/_quarantine_dupes/

Files are moved (not deleted) so they can be reviewed and recovered. Each
quarantine dir gets a per-file `<original-path>.json` sidecar with reason
metadata.

Use --dry-run to preview moves without touching the filesystem.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.pipeline.pipeline_log import StageLogger

ROOT = Path(__file__).resolve().parent.parent.parent

MANIFEST = ROOT / "logs" / "cleanup_manifest.json"
QUARANTINE_POISON = ROOT / "sources" / "_quarantine_poisoned"
QUARANTINE_DUPES = ROOT / "sources" / "_quarantine_dupes"


def _move(src: Path, dst_root: Path, meta: dict[str, Any], dry_run: bool,
          stage_logger: StageLogger | None = None) -> None:
    rel = src.relative_to(ROOT / "sources")
    dst = dst_root / rel
    sidecar = dst.with_suffix(dst.suffix + ".json")
    if not src.exists():
        print(f"  SKIP missing: {src.relative_to(ROOT)}")
        if stage_logger:
            stage_logger.item_skip(str(rel), reason="missing")
        return
    if dry_run:
        print(f"  DRY  {src.relative_to(ROOT)} -> {dst.relative_to(ROOT)}")
        if stage_logger:
            stage_logger.item_skip(str(rel), reason="dry_run",
                                   target=str(dst.relative_to(ROOT)))
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    sidecar.write_text(json.dumps(meta, indent=2))
    print(f"  MOVE {src.relative_to(ROOT)} -> {dst.relative_to(ROOT)}")
    if stage_logger:
        stage_logger.item_ok(str(rel),
                             target=str(dst.relative_to(ROOT)),
                             kind=meta.get("kind"),
                             reason=meta.get("reason"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--slug", default="_corpus_",
                        help="slug for observability tagging (default: _corpus_)")
    args = parser.parse_args()

    if not MANIFEST.exists():
        raise SystemExit(
            f"manifest not found: {MANIFEST}\n"
            "run scripts/build_cleanup_manifest.py first"
        )

    manifest = json.loads(MANIFEST.read_text())
    sm = manifest["summary"]

    print(f"manifest: poisoned={sm['n_quarantine_poisoned']}, dupes={sm['n_quarantine_dupes']}")
    if args.dry_run:
        print("DRY RUN — no files will be moved\n")

    from src.pipeline.pipeline_log import StageLogger  # noqa: PLC0415
    with StageLogger(args.slug, "apply") as sl:
        print("\n--- moving poisoned ---")
        for r in manifest["quarantine_poisoned"]:
            _move(
                ROOT / r["path"],
                QUARANTINE_POISON,
                {"reason": r["reason"], "kind": "poisoned"},
                args.dry_run,
                stage_logger=sl,
            )

        print("\n--- moving non-canonical dupes ---")
        for r in manifest["quarantine_dupes"]:
            _move(
                ROOT / r["path"],
                QUARANTINE_DUPES,
                {
                    "kind": "non_canonical_dupe",
                    "canonical_path": r["canonical_path"],
                    "canonical_in": r["canonical_in"],
                },
                args.dry_run,
                stage_logger=sl,
            )

    print("\ndone.")


if __name__ == "__main__":
    main()
