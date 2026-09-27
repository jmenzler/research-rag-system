# long-ok-file
"""Tests for ``src.ingest.sources`` — unified sources registry.

Replaces the two legacy dedup tables (``ingested_files`` + ``ingested_sources``)
with one ``sources`` table tracking a status lifecycle across the pipeline.

Coverage:
  * ``canonical_id`` priority order (arxiv > doi > url > hash) + stability
  * schema init creates the ``sources`` table with required indexes
  * status lifecycle: fetched → parsed → contextualized → ingested
  * ``is_ingested(canonical_id)`` reflects status='ingested'
  * ``has_been_ingested_by_hash`` short-circuit for ingest re-runs
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture
def parents_db(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[sqlite3.Connection]:
    """Open an isolated parents.sqlite per test — uses _open_parents_db so the
    new sources table is created alongside the legacy ones."""
    from src.ingest.storage import _open_parents_db  # noqa: PLC0415

    db_path = tmp_path / "parents.sqlite"
    monkeypatch.setattr("src.config.PARENTS_DB", str(db_path))
    conn = _open_parents_db()
    yield conn
    conn.close()


# ---------------------------------------------------------------------------
# canonical_id — priority ordering + stability
# ---------------------------------------------------------------------------


def test_canonical_id_prefers_arxiv_over_everything() -> None:
    from src.ingest.sources import canonical_id  # noqa: PLC0415

    meta = {
        "arxiv_id": "2204.12345",
        "doi": "10.1000/foo",
        "url": "https://example.com/x",
        "title": "x",
        "authors": ["a"],
    }
    assert canonical_id(meta) == ("2204.12345", "arxiv")


def test_canonical_id_prefers_doi_when_no_arxiv() -> None:
    from src.ingest.sources import canonical_id  # noqa: PLC0415

    meta = {"doi": "10.1000/FOO", "url": "https://example.com/x", "title": "x"}
    cid, kind = canonical_id(meta)
    assert kind == "doi"
    assert cid == "10.1000/foo"  # lowercased


def test_canonical_id_prefers_url_when_no_arxiv_or_doi() -> None:
    from src.ingest.sources import canonical_id  # noqa: PLC0415

    meta = {"url": "https://arxiv.org/abs/2204.12345v3", "title": "x"}
    cid, kind = canonical_id(meta)
    assert kind == "url"
    assert cid == "arxiv.org/abs/2204.12345"  # canonical_url normalizes


def test_canonical_id_falls_back_to_hash() -> None:
    from src.ingest.sources import canonical_id  # noqa: PLC0415

    meta = {"title": "My Paper", "authors": ["Alice Smith"]}
    cid, kind = canonical_id(meta)
    assert kind == "hash"
    assert len(cid) == 16


def test_canonical_id_stable() -> None:
    from src.ingest.sources import canonical_id  # noqa: PLC0415

    meta = {"title": "stable", "authors": ["a"]}
    assert canonical_id(meta) == canonical_id(meta)


def test_canonical_id_reads_nested_metadata_arxiv_id() -> None:
    """Enriched meta writes arxiv_id under meta['metadata']; canonical_id must read it."""
    from src.ingest.sources import canonical_id  # noqa: PLC0415

    meta = {
        "url": "https://arxiv.org/abs/2403.12345",
        "metadata": {"arxiv_id": "2403.12345", "doi": "10.1000/foo"},
    }
    assert canonical_id(meta) == ("2403.12345", "arxiv")


def test_canonical_id_reads_nested_metadata_doi_when_no_arxiv() -> None:
    """Nested doi wins over url when no arxiv at either level."""
    from src.ingest.sources import canonical_id  # noqa: PLC0415

    meta = {
        "url": "https://example.com/x",
        "metadata": {"arxiv_id": None, "doi": "10.1000/FOO"},
    }
    cid, kind = canonical_id(meta)
    assert kind == "doi"
    assert cid == "10.1000/foo"  # lowercased


def test_canonical_id_falls_back_to_top_level_for_legacy_meta() -> None:
    """Legacy meta dicts (no nested 'metadata' key) still resolved via top-level fields."""
    from src.ingest.sources import canonical_id  # noqa: PLC0415

    meta = {"arxiv_id": "9999.88888", "url": "https://example.com"}
    assert canonical_id(meta) == ("9999.88888", "arxiv")


# ---------------------------------------------------------------------------
# Schema — sources table exists with expected indexes
# ---------------------------------------------------------------------------


def test_init_creates_sources_table(parents_db: sqlite3.Connection) -> None:
    tables = [
        r[0]
        for r in parents_db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    ]
    assert "sources" in tables


def test_sources_indexes_present(parents_db: sqlite3.Connection) -> None:
    idx = [
        r[0]
        for r in parents_db.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='sources'"
        ).fetchall()
    ]
    assert "idx_sources_status" in idx
    assert "idx_sources_arxiv_id" in idx
    assert "idx_sources_file_hash" in idx


def test_legacy_tables_dropped(parents_db: sqlite3.Connection) -> None:
    tables = [
        r[0]
        for r in parents_db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    ]
    assert "ingested_files" not in tables
    assert "ingested_sources" not in tables


# ---------------------------------------------------------------------------
# Status lifecycle
# ---------------------------------------------------------------------------


def test_upsert_creates_fetched_row(parents_db: sqlite3.Connection) -> None:
    from src.ingest.sources import upsert  # noqa: PLC0415

    upsert(
        parents_db,
        canonical_id="2204.12345",
        id_type="arxiv",
        status="fetched",
        title="t",
        authors=["a"],
        year=2024,
        url="https://arxiv.org/abs/2204.12345",
        doi=None,
        arxiv_id="2204.12345",
        source_type="arxiv",
        notebook="trading",
        file_hash=None,
    )
    row = parents_db.execute(
        "SELECT id, status, arxiv_id FROM sources WHERE id = ?", ("2204.12345",)
    ).fetchone()
    assert row == ("2204.12345", "fetched", "2204.12345")


def test_advance_status_progresses_lifecycle(parents_db: sqlite3.Connection) -> None:
    from src.ingest.sources import advance_status, upsert  # noqa: PLC0415

    upsert(
        parents_db,
        canonical_id="x",
        id_type="hash",
        status="fetched",
        title=None,
        authors=None,
        year=None,
        url=None,
        doi=None,
        arxiv_id=None,
        source_type="web",
        notebook="trading",
        file_hash=None,
    )
    advance_status(parents_db, "x", "parsed")
    advance_status(parents_db, "x", "contextualized")
    advance_status(parents_db, "x", "ingested")
    row = parents_db.execute("SELECT status FROM sources WHERE id = ?", ("x",)).fetchone()
    assert row[0] == "ingested"


def test_is_ingested_true_only_when_status_ingested(
    parents_db: sqlite3.Connection,
) -> None:
    from src.ingest.sources import advance_status, is_ingested, upsert  # noqa: PLC0415

    upsert(
        parents_db,
        canonical_id="x",
        id_type="hash",
        status="fetched",
        title=None,
        authors=None,
        year=None,
        url=None,
        doi=None,
        arxiv_id=None,
        source_type="web",
        notebook="trading",
        file_hash=None,
    )
    assert is_ingested(parents_db, "x") is False
    advance_status(parents_db, "x", "ingested")
    assert is_ingested(parents_db, "x") is True


def test_is_ingested_false_for_unknown_id(parents_db: sqlite3.Connection) -> None:
    from src.ingest.sources import is_ingested  # noqa: PLC0415

    assert is_ingested(parents_db, "nope") is False


# ---------------------------------------------------------------------------
# Hash-based dedup — replaces _already_ingested / _mark_ingested
# ---------------------------------------------------------------------------


def test_has_been_ingested_by_hash_short_circuit(
    parents_db: sqlite3.Connection,
) -> None:
    from src.ingest.sources import (  # noqa: PLC0415
        has_been_ingested_by_hash,
        mark_ingested_with_hash,
        upsert,
    )

    upsert(
        parents_db,
        canonical_id="cid1",
        id_type="hash",
        status="fetched",
        title=None,
        authors=None,
        year=None,
        url=None,
        doi=None,
        arxiv_id=None,
        source_type="web",
        notebook="trading",
        file_hash=None,
    )
    mark_ingested_with_hash(parents_db, "cid1", "fhash-A", "trading")

    # Same (hash, notebook): idempotent re-run is skipped.
    assert has_been_ingested_by_hash(parents_db, "fhash-A", "trading") is True
    assert has_been_ingested_by_hash(parents_db, "missing-hash", "trading") is False
    # Same hash, DIFFERENT notebook: must NOT skip — partition B has no rows yet.
    assert has_been_ingested_by_hash(parents_db, "fhash-A", "notes") is False
    # notebook=None falls back to hash-only match (legacy callers).
    assert has_been_ingested_by_hash(parents_db, "fhash-A", None) is True


def test_mark_ingested_with_hash_idempotent(parents_db: sqlite3.Connection) -> None:
    from src.ingest.sources import mark_ingested_with_hash, upsert  # noqa: PLC0415

    upsert(
        parents_db,
        canonical_id="cid1",
        id_type="hash",
        status="fetched",
        title=None,
        authors=None,
        year=None,
        url=None,
        doi=None,
        arxiv_id=None,
        source_type="web",
        notebook="trading",
        file_hash=None,
    )
    mark_ingested_with_hash(parents_db, "cid1", "fh", "trading")
    mark_ingested_with_hash(parents_db, "cid1", "fh", "trading")  # no error
    row = parents_db.execute(
        "SELECT status, file_hash FROM sources WHERE id = ?", ("cid1",)
    ).fetchone()
    assert row == ("ingested", "fh")


def test_exists_returns_true_for_any_status(parents_db: sqlite3.Connection) -> None:
    from src.ingest.sources import exists, upsert  # noqa: PLC0415

    upsert(
        parents_db,
        canonical_id="cid1",
        id_type="hash",
        status="fetched",
        title=None,
        authors=None,
        year=None,
        url=None,
        doi=None,
        arxiv_id=None,
        source_type="web",
        notebook="trading",
        file_hash=None,
    )
    assert exists(parents_db, "cid1") is True
    assert exists(parents_db, "missing") is False
