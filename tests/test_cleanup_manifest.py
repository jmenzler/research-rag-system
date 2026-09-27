# long-ok-file
"""Tests for src/cleanup/manifest.py — quarantine action-plan synthesis.

Focus: the canonical-promotion path. When the file the canonical-picker chose
is independently flagged poisoned, the next-best survivor is promoted to
canonical and must NOT also be quarantined as a dupe — doing so would move the
group's only retained copy away and lose the entire dupe group (data loss).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.cleanup.manifest import build_manifest


def _write_source(root: Path, rel: str, content: str) -> str:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return rel


def _write_audits(
    tmp_path: Path,
    *,
    fail_pages: list[dict[str, str]],
    name_mismatches: list[dict[str, object]],
    exact: list[dict[str, object]],
) -> tuple[Path, Path]:
    name_path = tmp_path / "filename_content_match.json"
    dupe_path = tmp_path / "cross_partition_dedup.json"
    name_path.write_text(json.dumps({"fail_pages": fail_pages, "name_mismatches": name_mismatches}))
    dupe_path.write_text(json.dumps({"exact": exact}))
    return name_path, dupe_path


def test_poisoned_canonical_keeps_one_live_copy(tmp_path: Path) -> None:
    """Group whose chosen canonical is poisoned keeps exactly one live copy.

    The promoted survivor must appear in keep_canonical and must NOT appear in
    quarantine_dupes (otherwise the only retained copy gets moved → data loss).
    """
    root = tmp_path / "repo"
    # Same content under two filenames; both filename token-sets appear in the
    # body, so the group is NOT wholesale-poisoned (best overlap >= 0.4).
    body = (
        "avellaneda stoikov market making inventory risk reservation price "
        "garch volatility estimation hawkes intensity calibration\n" * 50
    )
    # File A's tokens are all in body -> overlap 1.0 -> chosen canonical.
    a = _write_source(
        root,
        "sources/trading/grp/001__avellaneda_stoikov_market_making_inventory.txt",
        body,
    )
    # File B's tokens partly in body -> overlap < 1.0 but >= 0.4 -> survivor.
    b = _write_source(
        root,
        "sources/ecology/grp/001__market_making_garch_volatility_unrelatedtoken_xyzzy.txt",
        body,
    )
    digest = hashlib.sha256(body.encode()).hexdigest()

    # A (the best-scoring canonical) is independently flagged poisoned.
    name_path, dupe_path = _write_audits(
        tmp_path,
        fail_pages=[{"path": a, "reason": "vendor_block_page"}],
        name_mismatches=[],
        exact=[{"hash": digest, "files": [a, b]}],
    )

    manifest = build_manifest(name_path, dupe_path, root)

    poisoned_paths = {e["path"] for e in manifest["quarantine_poisoned"]}
    dupe_paths = {e["path"] for e in manifest["quarantine_dupes"]}
    canonicals = {e["canonical"] for e in manifest["keep_canonical"]}

    assert a in poisoned_paths
    # The promoted survivor is the canonical and is NOT quarantined as a dupe.
    assert b in canonicals
    assert b not in dupe_paths
    assert b not in poisoned_paths

    # Exactly one live copy survives: B is canonical, A is poisoned, nothing
    # from this group is in quarantine_dupes.
    assert dupe_paths == set()
    # keep_canonical group has no drops left (only A and B existed).
    grp = manifest["keep_canonical"][0]
    assert grp["canonical"] == b
    assert grp["drop"] == []


def test_normal_group_drops_noncanonical(tmp_path: Path) -> None:
    """Sanity: a clean group keeps the best file and quarantines the other."""
    root = tmp_path / "repo"
    body = (
        "avellaneda stoikov market making inventory risk reservation price "
        "garch volatility estimation hawkes intensity calibration\n" * 50
    )
    a = _write_source(
        root,
        "sources/trading/grp/001__avellaneda_stoikov_market_making_inventory.txt",
        body,
    )
    b = _write_source(
        root,
        "sources/ecology/grp/001__market_making_garch_unrelatedtoken_xyzzy.txt",
        body,
    )
    digest = hashlib.sha256(body.encode()).hexdigest()

    name_path, dupe_path = _write_audits(
        tmp_path,
        fail_pages=[],
        name_mismatches=[],
        exact=[{"hash": digest, "files": [a, b]}],
    )

    manifest = build_manifest(name_path, dupe_path, root)

    canonicals = {e["canonical"] for e in manifest["keep_canonical"]}
    dupe_paths = {e["path"] for e in manifest["quarantine_dupes"]}

    # A scores 1.0 (all tokens present), B < 1.0 -> A canonical, B quarantined.
    assert a in canonicals
    assert dupe_paths == {b}
