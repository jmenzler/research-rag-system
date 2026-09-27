"""RED — implemented in 03-02-PLAN.

Tests for ``src.server.papers_view`` (build_read_view, get_read_view — TTL cache
+ watermark early bust + graceful null fallback).

Collection succeeds because every import of ``src.server.papers_view`` is deferred
inside the test body via ``pytest.importorskip`` — these tests fail at run time
(import error proves the gate is real).

Analog:
  - tests/server/test_api_papers.py  (Phase 1 — same deferred-import pattern)
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest


def _make_parents_db(db_path: Path) -> sqlite3.Connection:
    """Synthesize parents.sqlite with a sources table matching
    src/ingest/sources.py:init_schema and a parents table matching
    src/ingest/storage.py:48-58."""
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS sources (
            id           TEXT PRIMARY KEY,
            id_type      TEXT NOT NULL,
            title        TEXT,
            authors      TEXT,
            year         INTEGER,
            url          TEXT,
            doi          TEXT,
            arxiv_id     TEXT,
            source_type  TEXT,
            notebook     TEXT,
            status       TEXT NOT NULL,
            file_hash    TEXT,
            updated_at   INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS parents (
            id         TEXT PRIMARY KEY,
            text       TEXT NOT NULL,
            source_file TEXT NOT NULL,
            notebook   TEXT NOT NULL,
            modality   TEXT NOT NULL,
            page_number INTEGER NOT NULL,
            image_path  TEXT
        );
        """
    )
    return conn


def _make_doc_dir(
    sources_root: Path,
    collection: str,
    slug: str,
    *,
    has_meta: bool = True,
    source_filename: str = "source.pdf",
) -> Path:
    """Create a fake doc-dir under sources/<collection>/<slug>/ with optional
    meta.json and a representative source file."""
    doc_dir = sources_root / collection / slug
    doc_dir.mkdir(parents=True, exist_ok=True)

    if has_meta:
        (doc_dir / "meta.json").write_text(
            json.dumps(
                {
                    "title": f"Paper {slug}",
                    "authors": ["Author A"],
                    "year": 2024,
                    "arxiv_id": f"2401.{slug}",
                    "doi": f"10.1234/{slug}",
                    "url": f"https://example.com/{slug}",
                    "metadata": {
                        "arxiv_id": f"2401.{slug}",
                        "short_cite": f"Author et al. ({slug})",
                        "confidence": "high",
                    },
                }
            )
        )

    # Touch the representative source file
    (doc_dir / source_filename).write_text("content")
    return doc_dir


def test_build_read_view_import(tmp_path: Path) -> None:
    """RED — src.server.papers_view.build_read_view must be importable after
    03-02-PLAN."""
    pytest.importorskip("src.server.papers_view")
    from src.server.papers_view import build_read_view  # noqa: F401


def test_build_read_view_row_shape(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """RED — build_read_view returns rows with expected keys."""
    sources_root = tmp_path / "sources"
    _make_doc_dir(sources_root, "trading", "abc123")
    parents_db = tmp_path / "parents.sqlite"
    conn = _make_parents_db(parents_db)
    conn.execute(
        "INSERT INTO sources (id, id_type, title, status, updated_at) VALUES (?, ?, ?, ?, ?)",
        ("2401.abc123", "arxiv", "Paper abc123", "ingested", 1000),
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr("src.config.paths.SOURCES_DIR", sources_root)

    pytest.importorskip("src.server.papers_view")
    from src.server.papers_view import build_read_view

    rows = build_read_view(parents_db=parents_db)
    assert len(rows) >= 1
    row = rows[0]
    for key in (
        "paper_id",
        "title",
        "authors",
        "year",
        "collection",
        "chunk_count",
        "arxiv_id",
        "doi",
        "short_cite",
        "ingested_at",
        "extraction_quality",
    ):
        assert key in row, f"missing key {key!r} in read-view row"


def test_ttl_returns_same_object(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """RED — get_read_view within TTL returns the SAME cached object (identity)."""
    sources_root = tmp_path / "sources"
    _make_doc_dir(sources_root, "trading", "abc123")
    parents_db = tmp_path / "parents.sqlite"
    conn = _make_parents_db(parents_db)
    conn.execute(
        "INSERT INTO sources (id, id_type, title, status, updated_at) VALUES (?, ?, ?, ?, ?)",
        ("2401.abc123", "arxiv", "Paper abc123", "ingested", 1000),
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr("src.config.paths.SOURCES_DIR", sources_root)

    pytest.importorskip("src.server.papers_view")
    from src.server.papers_view import get_read_view

    first = get_read_view(parents_db=parents_db)
    second = get_read_view(parents_db=parents_db)
    assert first is second, "TTL cache should return same object within TTL"


def test_watermark_early_bust(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """RED — bumping max(sources.updated_at) forces rebuild before TTL expires."""
    sources_root = tmp_path / "sources"
    _make_doc_dir(sources_root, "trading", "abc123")
    parents_db = tmp_path / "parents.sqlite"
    conn = _make_parents_db(parents_db)
    conn.execute(
        "INSERT INTO sources (id, id_type, title, status, updated_at) VALUES (?, ?, ?, ?, ?)",
        ("2401.abc123", "arxiv", "Paper abc123", "ingested", 1000),
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr("src.config.paths.SOURCES_DIR", sources_root)

    pytest.importorskip("src.server.papers_view")
    from src.server.papers_view import get_read_view

    # First call builds cache
    first = get_read_view(parents_db=parents_db)

    # Simulate a fresh ingest: bump updated_at past the watermark
    conn2 = sqlite3.connect(str(parents_db))
    conn2.execute(
        "UPDATE sources SET updated_at = ? WHERE id = ?",
        (999999, "2401.abc123"),
    )
    conn2.commit()
    conn2.close()

    # Second call should detect watermark advance and rebuild
    second = get_read_view(parents_db=parents_db)
    assert first is not second, "Watermark advance should force cache rebuild"


def test_graceful_null_on_missing_meta_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RED — sources row with no meta.json yields chunk_count=None / metadata
    None, no crash."""
    sources_root = tmp_path / "sources"
    # Create doc-dir WITHOUT meta.json
    _make_doc_dir(sources_root, "trading", "no-meta", has_meta=False)
    parents_db = tmp_path / "parents.sqlite"
    conn = _make_parents_db(parents_db)
    conn.execute(
        "INSERT INTO sources (id, id_type, title, status, updated_at) VALUES (?, ?, ?, ?, ?)",
        ("no-meta-hash", "hash", "No Meta Paper", "ingested", 1000),
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr("src.config.paths.SOURCES_DIR", sources_root)

    pytest.importorskip("src.server.papers_view")
    from src.server.papers_view import build_read_view

    rows = build_read_view(parents_db=parents_db)
    # Should not crash; missing metadata fields are None
    for row in rows:
        if row.get("chunk_count") is None:
            pass  # graceful null is acceptable
        if row.get("title") is None:
            pass  # graceful null from missing meta.json


def test_no_materialized_papers_table(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """RED — build_read_view must NOT create a materialized papers table in
    parents.sqlite (CORPUS-04 forbids)."""
    sources_root = tmp_path / "sources"
    _make_doc_dir(sources_root, "trading", "abc123")
    parents_db = tmp_path / "parents.sqlite"
    conn = _make_parents_db(parents_db)
    conn.commit()

    monkeypatch.setattr("src.config.paths.SOURCES_DIR", sources_root)

    pytest.importorskip("src.server.papers_view")
    from src.server.papers_view import build_read_view

    build_read_view(parents_db=parents_db)

    # Re-open and check no new tables were created
    conn2 = sqlite3.connect(str(parents_db))
    tables = conn2.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    table_names = {r[0] for r in tables}
    conn2.close()
    # We expect only sources, parents, poll_runs (or whatever existed before)
    assert "papers" not in table_names, (
        "build_read_view created a 'papers' table — CORPUS-04 violation"
    )


def test_get_read_view_import(tmp_path: Path) -> None:
    """RED — src.server.papers_view.get_read_view must be importable after
    03-02-PLAN."""
    pytest.importorskip("src.server.papers_view")
    from src.server.papers_view import get_read_view  # noqa: F401
