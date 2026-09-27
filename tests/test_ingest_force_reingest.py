"""Force re-ingest: bypass the unchanged-hash dedup + delete stale chunks.
The full-PDF upgrade re-parses a paper whose source.pdf hash is unchanged, so
``ingest_file(force=True)`` must re-ingest, and ``delete_source_chunks`` must
clear the stub chunks from Milvus + parents.sqlite first.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.ingest.storage import _open_parents_db, _store_parent, delete_source_chunks
from src.models import ParentChunk


@pytest.fixture
def parents_db(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[sqlite3.Connection]:
    monkeypatch.setattr("src.config.PARENTS_DB", str(tmp_path / "parents.sqlite"))
    conn = _open_parents_db()
    yield conn
    conn.close()


def _parent(pid: str, source_file: str) -> ParentChunk:
    return ParentChunk(
        id=pid,
        text="x",
        source_file=source_file,
        notebook="arxiv_monitor",
        modality="pdf",
        page_number=0,
        image_path=None,
    )


@patch("src.ingest.storage.get_client")
def test_delete_source_chunks_removes_parents_and_calls_milvus(
    get_client_mock: MagicMock,
    parents_db: sqlite3.Connection,
) -> None:
    fake = MagicMock()
    get_client_mock.return_value = fake
    src = "/abs/sources/trading/2401.1/source.pdf"
    other = "/abs/sources/trading/2401.2/source.pdf"
    _store_parent(parents_db, _parent("p1", src))
    _store_parent(parents_db, _parent("p2", src))
    _store_parent(parents_db, _parent("p3", other))
    parents_db.commit()

    deleted = delete_source_chunks("trading", src, parents_db, notebook="arxiv_monitor")

    assert deleted == 2
    remaining = parents_db.execute("SELECT id FROM parents").fetchall()
    assert [r[0] for r in remaining] == ["p3"]
    fake.delete.assert_called_once()
    kwargs = fake.delete.call_args.kwargs
    assert kwargs["collection_name"] == "trading"
    assert kwargs["partition_name"] == "arxiv_monitor"
    assert src in kwargs["filter"]


@patch("src.ingest.storage.get_client")
def test_delete_source_chunks_rejects_quote(
    get_client_mock: MagicMock,
    parents_db: sqlite3.Connection,
) -> None:
    with pytest.raises(ValueError, match="embedded quote"):
        delete_source_chunks("trading", '/a/"x"/source.pdf', parents_db, notebook="n")


@patch("src.ingest.ingest._ingest_text_file", return_value=(1, 2))
@patch("src.ingest.ingest.has_been_ingested_by_hash", return_value=True)
def test_ingest_file_force_bypasses_hash_dedup(
    _hash_mock: MagicMock,
    ingest_text_mock: MagicMock,
    tmp_path: Path,
) -> None:
    from src.ingest.ingest import ingest_file

    pdf = tmp_path / "source.pdf"
    pdf.write_bytes(b"%PDF-1.4 payload")

    skipped = ingest_file(
        path=pdf,
        notebook="arxiv_monitor",
        collection="trading",
        gemini_client=MagicMock(),
        db_conn=MagicMock(),
        enc=MagicMock(),
    )
    assert skipped["skipped"] is True
    ingest_text_mock.assert_not_called()

    with patch("src.ingest.ingest.mark_ingested_with_hash"):
        forced = ingest_file(
            path=pdf,
            notebook="arxiv_monitor",
            collection="trading",
            gemini_client=MagicMock(),
            db_conn=MagicMock(),
            enc=MagicMock(),
            force=True,
        )
    assert forced["skipped"] is False
    ingest_text_mock.assert_called_once()
