# long-ok-file
"""Cleanup manifest builder — synthesizes corpus-audit outputs into action plan.

Inputs (JSON files written by the two audit scripts):
- filename_content_match.json   fail-pages + name-mismatches + skipped
- cross_partition_dedup.json    cross-partition exact-content collisions

Output:
- cleanup_manifest.json — three buckets:
    * quarantine_poisoned: files to move to sources/_quarantine_poisoned/
    * quarantine_dupes:    non-canonical files in real cross-partition dupe groups
    * keep_canonical:      surviving canonical file per dupe group (informational)

Rules in force (algorithmic, no LLM):
- Rule 1 — Hash-collision canonical picking: when N files share content,
  pick the one whose filename tokens best match the content's first 5KB.
  Others go to quarantine_dupes. If even the best filename has overlap < 0.4,
  the whole group is corrupted-name saves of unknown-true-title content →
  quarantine_poisoned.
- Rule 3 — Longest-paragraph floor (handled in fail-page audit upstream).
- Rule 4 — Vendor-block-page tokens (handled in fail-page audit upstream).
- Rule 5 — DROPPED: cross_part_poison vocab-overlap-with-notebook check.
  Punished legit general/intro articles. Source of 21/48 false positives.
- Rule 6 — name_mismatch is combiner-only: only escalates to quarantine if it
  also has another signal (hash collision OR file<1KB OR longest_paragraph<70).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.cleanup.filename_match import filename_to_content_overlap
from src.cleanup.quality import MIN_LONGEST_PARAGRAPH, longest_paragraph_chars
from src.ingest import _strip_tail_sections

# Below this filename-vs-content overlap, even the "best" filename in a hash-
# collision group is wrong → whole group is corrupted-name saves, can't trust any.
CANONICAL_MIN_OVERLAP = 0.40

# Combiner thresholds for name_mismatch (Rule 6)
TINY_FILE_BYTES = 1024


def _partition_of(path: str) -> str:
    """Collection is the path component immediately under ``sources/``.

    ``sources/<collection>/<slug>/<leaf>.txt`` → ``<collection>``. Falls back
    to the parent directory name for non-standard layouts.
    """
    parts = Path(path).parts
    try:
        idx = parts.index("sources")
    except ValueError:
        return Path(path).parent.name
    return parts[idx + 1] if idx + 1 < len(parts) else Path(path).parent.name


def _file_size(root: Path, path: str) -> int:
    p = root / path
    return p.stat().st_size if p.exists() else 0


def _file_max_paragraph(root: Path, path: str) -> int:
    p = root / path
    if not p.exists():
        return 0
    return longest_paragraph_chars(p.read_text(encoding="utf-8", errors="ignore"))


def _file_canonical_score(root: Path, path: str) -> float:
    """Filename-vs-content overlap on the section-tail-stripped body."""
    p = root / path
    if not p.exists():
        return 0.0
    text = p.read_text(encoding="utf-8", errors="ignore")
    clean, _ = _strip_tail_sections(text)
    return filename_to_content_overlap(path, clean)


def build_manifest(
    name_audit_path: Path,
    dupe_audit_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    """Synthesize the two audit JSONs into a single quarantine action manifest.

    Pure function — does not write files or move anything. Caller persists output.
    """
    name_data = json.loads(name_audit_path.read_text())
    dupe_data = json.loads(dupe_audit_path.read_text())

    # Bucket A: poisoned (fail-page is strong; name-mismatch is combiner-only)
    poisoned: dict[str, str] = {}
    for r in name_data["fail_pages"]:
        poisoned[r["path"]] = f"fail_page:{r['reason']}"

    # Pre-compute hash-collision membership for combiner check (Rule 6)
    hash_collision_paths: set[str] = set()
    for c in dupe_data["exact"]:
        hash_collision_paths.update(c["files"])

    # Bucket B: cross-partition dupes — pick canonical via filename-vs-content (Rule 1)
    dupes_canonical: list[dict[str, Any]] = []

    for c in dupe_data["exact"]:
        files = c["files"]
        scored = sorted(
            ((p, _file_canonical_score(repo_root, p)) for p in files),
            key=lambda x: -x[1],
        )
        best_path, best_score = scored[0]

        if best_score < CANONICAL_MIN_OVERLAP:
            # All filenames disagree with content — whole group is corrupted-name saves.
            for p, s in scored:
                poisoned.setdefault(p, f"hash_collision_no_canonical:best={s:.2f}")
            continue

        canonical = best_path
        drops = [p for p, _ in scored[1:]]
        scores_map = {p: round(s, 2) for p, s in scored}
        dupes_canonical.append({
            "canonical": canonical,
            "drop": drops,
            "hash": c["hash"],
            "best_overlap": round(best_score, 2),
            "all_files": files,
            "scores": scores_map,
        })

    # Rule 6: name_mismatch only escalates with another signal.
    for r in name_data["name_mismatches"]:
        path = r["path"]
        if path in poisoned:
            continue
        signals: list[str] = []
        if path in hash_collision_paths:
            signals.append("hash_collision")
        if _file_size(repo_root, path) < TINY_FILE_BYTES:
            signals.append("<1KB")
        if _file_max_paragraph(repo_root, path) < MIN_LONGEST_PARAGRAPH:
            signals.append(f"max_paragraph<{MIN_LONGEST_PARAGRAPH}")
        if signals:
            poisoned[path] = f"name_mismatch+{'+'.join(signals)}:ratio={r['ratio']:.2f}"

    # If a poisoned path also appears in a dupe group, promote next-best canonical.
    poisoned_set = set(poisoned)
    dupes_canonical_filtered: list[dict[str, Any]] = []
    for entry in dupes_canonical:
        if entry["canonical"] in poisoned_set:
            survivors = [
                (p, entry["scores"][p])
                for p in entry["all_files"]
                if p not in poisoned_set
            ]
            if not survivors:
                continue
            survivors.sort(key=lambda x: -x[1])
            entry["canonical"] = survivors[0][0]
            entry["drop"] = [p for p, _ in survivors[1:]]
        dupes_canonical_filtered.append(entry)

    # Rebuild the move list from the final per-group drops so the promoted
    # canonical (which started life as a drop) is never quarantined — moving it
    # would strip the dupe group of its only retained copy.
    dupes_to_move: list[dict[str, Any]] = []
    for entry in dupes_canonical_filtered:
        canonical = entry["canonical"]
        for p in entry["drop"]:
            if p in poisoned_set:
                continue
            dupes_to_move.append({
                "path": p,
                "canonical_in": _partition_of(canonical),
                "canonical_path": canonical,
                "match_overlap": entry["scores"][p],
            })

    return {
        "summary": {
            "n_quarantine_poisoned": len(poisoned),
            "n_quarantine_dupes": len(dupes_to_move),
            "n_canonical_groups": len(dupes_canonical_filtered),
        },
        "quarantine_poisoned": [{"path": p, "reason": r} for p, r in sorted(poisoned.items())],
        "quarantine_dupes": dupes_to_move,
        "keep_canonical": dupes_canonical_filtered,
    }
