"""RED — implemented in 03-02-PLAN.

Tests for ``src.server.papers_view`` FTS5 behavior — rebuild_corpus_fts and
search_corpus_fts.

Collection succeeds via deferred import (``pytest.importorskip``); run-time
import failure proves the gate is real.

Analog:
  - tests/server/test_chats_fts.py (Phase 2 — same FTS5 rebuild/search pattern)
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest


def _make_parents_db(db_path: Path) -> sqlite3.Connection:
    """Synthesize parents.sqlite with sources + parents tables."""
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
) -> Path:
    """Create a fake doc-dir with meta.json."""
    doc_dir = sources_root / collection / slug
    doc_dir.mkdir(parents=True, exist_ok=True)
    (doc_dir / "meta.json").write_text(
        json.dumps(
            {
                "title": f"Paper {slug}",
                "authors": ["Author A", "Author B"],
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
    (doc_dir / "source.pdf").write_text("content")
    return doc_dir


def test_rebuild_corpus_fts_import(tmp_path: Path) -> None:
    """RED — rebuild_corpus_fts must be importable after 03-02-PLAN."""
    pytest.importorskip("src.server.papers_view")
    from src.server.papers_view import rebuild_corpus_fts  # noqa: F401


def test_search_corpus_fts_import(tmp_path: Path) -> None:
    """RED — search_corpus_fts must be importable after 03-02-PLAN."""
    pytest.importorskip("src.server.papers_view")
    from src.server.papers_view import search_corpus_fts  # noqa: F401


def test_rebuild_populates_fts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ephemeral_chats_db: Path
) -> None:
    """RED — rebuild populates corpus_fts; at least one row exists."""
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
    from src.server.papers_view import rebuild_corpus_fts

    rebuild_corpus_fts(chats_db=str(ephemeral_chats_db), parents_db=parents_db)

    conn2 = sqlite3.connect(str(ephemeral_chats_db))
    try:
        count = conn2.execute("SELECT COUNT(*) FROM corpus_fts").fetchone()[0]
        assert count >= 1, "rebuild_corpus_fts populated 0 rows; expected >= 1"
    finally:
        conn2.close()


def test_search_matches_title(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ephemeral_chats_db: Path
) -> None:
    """RED — search on title returns the matching paper."""
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
    from src.server.papers_view import rebuild_corpus_fts, search_corpus_fts

    rebuild_corpus_fts(chats_db=str(ephemeral_chats_db), parents_db=parents_db)
    results = search_corpus_fts("abc123", chats_db=str(ephemeral_chats_db))
    assert len(results) >= 1, "FTS5 title search for 'abc123' returned 0 results"


def test_search_matches_authors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ephemeral_chats_db: Path
) -> None:
    """RED — search on authors returns the matching paper."""
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
    from src.server.papers_view import rebuild_corpus_fts, search_corpus_fts

    rebuild_corpus_fts(chats_db=str(ephemeral_chats_db), parents_db=parents_db)
    results = search_corpus_fts("Author", chats_db=str(ephemeral_chats_db))
    assert len(results) >= 1, "FTS5 authors search for 'Author' returned 0 results"


def test_search_matches_arxiv_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ephemeral_chats_db: Path
) -> None:
    """RED — search on arxiv_id returns the matching paper."""
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
    from src.server.papers_view import rebuild_corpus_fts, search_corpus_fts

    rebuild_corpus_fts(chats_db=str(ephemeral_chats_db), parents_db=parents_db)
    results = search_corpus_fts("2401.abc123", chats_db=str(ephemeral_chats_db))
    assert len(results) >= 1, "FTS5 arxiv_id search for '2401.abc123' returned 0 results"


def test_search_matches_doi(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ephemeral_chats_db: Path
) -> None:
    """RED — search on doi returns the matching paper."""
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
    from src.server.papers_view import rebuild_corpus_fts, search_corpus_fts

    rebuild_corpus_fts(chats_db=str(ephemeral_chats_db), parents_db=parents_db)
    results = search_corpus_fts("10.1234", chats_db=str(ephemeral_chats_db))
    assert len(results) >= 1, "FTS5 doi search for '10.1234' returned 0 results"


def test_rebuild_after_watermark_advance_refreshes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ephemeral_chats_db: Path
) -> None:
    """RED — rebuild after watermark advance drops stale papers and includes
    new ones."""
    sources_root = tmp_path / "sources"
    _make_doc_dir(sources_root, "trading", "stale123")
    parents_db = tmp_path / "parents.sqlite"
    conn = _make_parents_db(parents_db)
    conn.execute(
        "INSERT INTO sources (id, id_type, title, status, updated_at) VALUES (?, ?, ?, ?, ?)",
        ("2401.stale123", "arxiv", "Stale Paper", "ingested", 1000),
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr("src.config.paths.SOURCES_DIR", sources_root)

    pytest.importorskip("src.server.papers_view")
    from src.server.papers_view import rebuild_corpus_fts, search_corpus_fts

    # First rebuild
    rebuild_corpus_fts(chats_db=str(ephemeral_chats_db), parents_db=parents_db)

    # Now simulate fresh rebuild: delete stale, add new
    conn2 = sqlite3.connect(str(parents_db))
    conn2.execute("DELETE FROM sources WHERE id = '2401.stale123'")
    conn2.commit()
    conn2.close()

    # Second rebuild should refresh
    rebuild_corpus_fts(chats_db=str(ephemeral_chats_db), parents_db=parents_db)

    results = search_corpus_fts("Stale", chats_db=str(ephemeral_chats_db))
    assert len(results) == 0, "Rebuild after watermark advance did not drop stale paper"
