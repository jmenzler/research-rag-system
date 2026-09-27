"""Tests for ``src.ingest.storage`` — parents.sqlite + Milvus insert layer.

Boundaries pinned here:

  * ``_open_parents_db`` / ``_init_parents_db`` — schema creation is
    idempotent; two opens see the same tables.
  * ``_file_hash`` — model+dim salt invalidates the hash on model swap
    (this is the dedup-cache invariant; if it breaks, a re-embed run will
    silently skip already-ingested files even with a different vector dim).
  * ``_truncate_utf8`` — byte-counted truncation that does NOT split UTF-8
    multi-byte sequences (Milvus VARCHAR is byte-counted; naive char slice
    can produce decode errors downstream).
  * ``_insert_children`` — happy-path row construction + upsert_count
    assertion + sparse_embedding deliberately omitted. ``upsert`` (not
    ``insert``) is the write op: deterministic child ids + delete-by-pk make
    re-ingest overwrite in place instead of appending duplicates.

Dedup is exercised in ``tests/test_sources.py`` (unified ``sources`` table).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.ingest.storage import (
    _file_hash,
    _init_parents_db,
    _insert_children,
    _open_parents_db,
    _store_parent,
    _truncate_utf8,
)
from src.models import ChildChunk, ParentChunk

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def parents_db(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[sqlite3.Connection]:
    """Open an isolated parents.sqlite per test, point config.PARENTS_DB at it."""
    db_path = tmp_path / "parents.sqlite"
    monkeypatch.setattr("src.config.PARENTS_DB", str(db_path))
    conn = _open_parents_db()
    yield conn
    conn.close()


@pytest.fixture
def text_file(tmp_path: Path) -> Iterator[str]:
    """A small file whose content is deterministic, returned as path string."""
    p = tmp_path / "doc.pdf"
    p.write_bytes(b"hello-world-payload")
    yield str(p)


def _make_parent(
    idx: int = 0,
    *,
    text: str | None = None,
    source_file: str | None = None,
    notebook: str = "trading",
    modality: str = "pdf",
    page_number: int = 0,
    image_path: str | None = None,
) -> ParentChunk:
    """Build a ParentChunk with sensible test defaults."""
    return ParentChunk(
        id=f"p{idx:04x}",
        text=text if text is not None else f"parent {idx} text",
        source_file=source_file if source_file is not None else f"/tmp/doc{idx}.pdf",
        notebook=notebook,
        modality=modality,
        page_number=page_number,
        image_path=image_path,
    )


def _make_child(
    idx: int = 0,
    *,
    parent_id: str = "p0000",
    text: str | None = None,
    source_file: str = "/tmp/doc.pdf",
    notebook: str = "trading",
    modality: str = "pdf",
    page_number: int = 0,
) -> ChildChunk:
    """Build a ChildChunk with sensible test defaults."""
    return ChildChunk(
        id=f"c{idx:04x}",
        parent_id=parent_id,
        text=text if text is not None else f"child {idx} text",
        source_file=source_file,
        notebook=notebook,
        modality=modality,
        page_number=page_number,
    )


# ---------------------------------------------------------------------------
# Schema init — idempotent, two opens see same tables
# ---------------------------------------------------------------------------


def test_init_parents_db_is_idempotent(tmp_path: Path) -> None:
    """Calling _init_parents_db twice must not raise (CREATE IF NOT EXISTS)."""
    conn = sqlite3.connect(tmp_path / "twice.sqlite")
    _init_parents_db(conn)
    _init_parents_db(conn)  # second call — must not raise
    tables = [
        r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    ]
    assert "parents" in tables
    assert "sources" in tables
    conn.close()


def test_open_parents_db_creates_parent_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If PARENTS_DB points to a path under a missing dir, open creates it."""
    nested = tmp_path / "deep" / "nest" / "parents.sqlite"
    monkeypatch.setattr("src.config.PARENTS_DB", str(nested))
    conn = _open_parents_db()
    assert nested.exists()
    conn.close()


# ---------------------------------------------------------------------------
# _file_hash — model salt invariant
# ---------------------------------------------------------------------------


def test_file_hash_is_deterministic_for_same_content_and_model(
    text_file: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two reads of the same file under the same config must produce identical hash."""
    monkeypatch.setattr("src.config.EMBED_MODEL", "model-A")
    monkeypatch.setattr("src.config.EMBED_DIM", 768)
    from pathlib import Path

    h1 = _file_hash(Path(text_file))
    h2 = _file_hash(Path(text_file))
    assert h1 == h2


@pytest.mark.parametrize(
    "label,model_a,dim_a,model_b,dim_b,should_differ",
    [
        ("model_change", "model-A", 768, "model-B", 768, True),
        ("dim_change", "model-A", 768, "model-A", 1024, True),
        ("both_change", "model-A", 768, "model-B", 1024, True),
        ("no_change", "model-A", 768, "model-A", 768, False),
    ],
)
def test_file_hash_model_salt_invalidates_cache(
    text_file: str,
    monkeypatch: pytest.MonkeyPatch,
    label: str,
    model_a: str,
    dim_a: int,
    model_b: str,
    dim_b: int,
    should_differ: bool,
) -> None:
    """Model or dim swap must change the hash — that's the cache-bust invariant."""
    from pathlib import Path

    monkeypatch.setattr("src.config.EMBED_MODEL", model_a)
    monkeypatch.setattr("src.config.EMBED_DIM", dim_a)
    h_a = _file_hash(Path(text_file))

    monkeypatch.setattr("src.config.EMBED_MODEL", model_b)
    monkeypatch.setattr("src.config.EMBED_DIM", dim_b)
    h_b = _file_hash(Path(text_file))

    if should_differ:
        assert h_a != h_b, f"{label}: hashes should differ but matched"
    else:
        assert h_a == h_b, f"{label}: hashes should match"


# ---------------------------------------------------------------------------
# _store_parent — round-trips through sqlite
# ---------------------------------------------------------------------------


def test_store_parent_round_trips(parents_db: sqlite3.Connection) -> None:
    p = _make_parent(0, text="hello")
    _store_parent(parents_db, p)
    parents_db.commit()
    row = parents_db.execute(
        "SELECT id, text, notebook, modality FROM parents WHERE id = ?",
        (p.id,),
    ).fetchone()
    assert row == (p.id, "hello", "trading", "pdf")


def test_store_parent_overwrites_on_id_collision(
    parents_db: sqlite3.Connection,
) -> None:
    """INSERT OR REPLACE — a re-store with the same id updates the row."""
    p1 = _make_parent(0, text="first version")
    _store_parent(parents_db, p1)
    p2 = _make_parent(0, text="second version")
    _store_parent(parents_db, p2)
    parents_db.commit()
    row = parents_db.execute("SELECT text FROM parents WHERE id = ?", (p1.id,)).fetchone()
    assert row[0] == "second version"


# ---------------------------------------------------------------------------
# _truncate_utf8 — byte-counted, multi-byte safe
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "label,text,max_bytes,expected_bytes_le",
    [
        # ASCII path — char count == byte count.
        ("ascii_under", "hello", 100, 5),
        ("ascii_at_limit", "x" * 100, 100, 100),
        ("ascii_over", "x" * 200, 100, 100),
        # German umlauts — each ä/ü is 2 bytes in UTF-8.
        ("umlauts_under", "schöne grüße", 50, 14),  # 12 chars / 14 bytes
        # Cap mid-multi-byte sequence — must NOT produce a decode error.
        ("emoji_over", "🙂" * 50, 10, 10),  # each emoji = 4 bytes; 10/4 = 2 emoji
    ],
)
def test_truncate_utf8_byte_count(
    label: str,
    text: str,
    max_bytes: int,
    expected_bytes_le: int,
) -> None:
    out = _truncate_utf8(text, max_bytes)
    out_bytes = out.encode("utf-8")
    assert len(out_bytes) <= max_bytes, (
        f"{label}: produced {len(out_bytes)} bytes, cap was {max_bytes}"
    )
    # Must be valid UTF-8 (decode → encode round-trip without error).
    assert out.encode("utf-8").decode("utf-8") == out, f"{label}: produced text isn't clean UTF-8"


def test_truncate_utf8_does_not_split_multi_byte_sequence() -> None:
    """4-byte emoji at the cap boundary must not produce a partial sequence."""
    text = "abc🙂xyz"  # 'abc' = 3 bytes, '🙂' = 4 bytes, 'xyz' = 3 bytes; total 10 bytes
    # Cap at 5 bytes — would slice into the middle of the emoji.
    out = _truncate_utf8(text, 5)
    assert out == "abc"  # emoji + xyz dropped cleanly
    assert len(out.encode("utf-8")) == 3


# ---------------------------------------------------------------------------
# _insert_children — happy path + upsert_count assertion
# ---------------------------------------------------------------------------


@patch("src.ingest.storage.get_client")
def test_insert_children_happy_path(get_client_mock: MagicMock) -> None:
    """Three children → three rows → Milvus upsert called once (not insert)."""
    fake_client = MagicMock()
    fake_client.upsert.return_value = {"upsert_count": 3}
    get_client_mock.return_value = fake_client

    children = [_make_child(i) for i in range(3)]
    vectors = [[0.0] * 4096 for _ in range(3)]

    _insert_children("trading", children, vectors)

    fake_client.upsert.assert_called_once()
    fake_client.insert.assert_not_called()
    call_kwargs = fake_client.upsert.call_args.kwargs
    assert call_kwargs["collection_name"] == "trading"
    rows = call_kwargs["data"]
    assert len(rows) == 3
    # Schema invariants: dense_embedding present, sparse_embedding absent.
    assert "dense_embedding" in rows[0]
    assert "sparse_embedding" not in rows[0], (
        "sparse_embedding must be omitted — Milvus populates it server-side via BM25"
    )


@patch("src.ingest.storage.get_client")
def test_insert_children_empty_input_is_noop(get_client_mock: MagicMock) -> None:
    """Empty children list must NOT call Milvus at all."""
    fake_client = MagicMock()
    get_client_mock.return_value = fake_client

    _insert_children("trading", [], [])

    fake_client.upsert.assert_not_called()


@patch("src.ingest.storage.get_client")
def test_insert_children_raises_on_count_mismatch(
    get_client_mock: MagicMock,
) -> None:
    """Milvus returning upsert_count != len(rows) is a partial write that must fail loud."""
    fake_client = MagicMock()
    fake_client.upsert.return_value = {"upsert_count": 2}  # we sent 3
    get_client_mock.return_value = fake_client

    children = [_make_child(i) for i in range(3)]
    vectors = [[0.0] * 4096 for _ in range(3)]

    with pytest.raises(RuntimeError, match="upsert_count mismatch"):
        _insert_children("trading", children, vectors)


@patch("src.ingest.storage.get_client")
def test_insert_children_caps_text_to_milvus_byte_limit(
    get_client_mock: MagicMock,
) -> None:
    """Child text exceeding 4096 bytes must be truncated before upsert."""
    fake_client = MagicMock()
    fake_client.upsert.return_value = {"upsert_count": 1}
    get_client_mock.return_value = fake_client

    big_child = _make_child(0, text="x" * 10000)
    _insert_children("trading", [big_child], [[0.0] * 4096])

    rows = fake_client.upsert.call_args.kwargs["data"]
    assert len(rows[0]["text"].encode("utf-8")) <= 4096, (
        "child text not truncated to Milvus VARCHAR(4096) byte limit"
    )
