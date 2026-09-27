"""Storage paths relative to the project root.

Layout:
    sources/<collection>/<slug>/                  doc dir (content_list.json, meta.json, ...)
    sources/_quarantine_dedup/<orig_tag>/<orig_slug>/
    sources/_quarantine_nlm_artifacts/<orig_tag>/<orig_slug>/
    sources/_quarantine_nlm_garbage/<orig_tag>/<orig_slug>/
    sources/_quarantine_poisoned/<orig_tag>/<orig_slug>/   (legacy)
    sources/_quarantine_dupes/<orig_tag>/<orig_slug>/      (legacy)
    parents/parents.sqlite

Use the helpers below — never hardcode `sources/...` strings.
"""
from __future__ import annotations

from pathlib import Path

# Project root — this file is at <root>/src/config/paths.py
ROOT: Path = Path(__file__).resolve().parents[2]

SOURCES_DIR: Path = ROOT / "sources"
PARENTS_DB: str = "parents/parents.sqlite"

COLLECTIONS: tuple[str, ...] = ("trading", "ecology", "notes", "system", "poker", "security")


def collection_dir(collection: str) -> Path:
    """sources/<collection>/"""
    return SOURCES_DIR / collection


def doc_dir(collection: str, slug: str) -> Path:
    """sources/<collection>/<slug>/"""
    return SOURCES_DIR / collection / slug


def quarantine_dir(reason: str) -> Path:
    """sources/_quarantine_<reason>/"""
    return SOURCES_DIR / f"_quarantine_{reason}"


def iter_doc_dirs(collection: str | None = None) -> list[Path]:
    """Yield every doc dir under sources/<collection>/ (or all collections).

    Skips top-level dirs starting with `_` (quarantine, etc).
    """
    if collection is not None:
        coll_root = collection_dir(collection)
        if not coll_root.is_dir():
            return []
        return sorted(p for p in coll_root.iterdir() if p.is_dir())

    out: list[Path] = []
    for coll in COLLECTIONS:
        out.extend(iter_doc_dirs(coll))
    return out


def collection_glob_content_list(collection: str) -> str:
    """Glob pattern matching every content_list.json in a collection."""
    return str(SOURCES_DIR / collection / "*" / "content_list.json")
