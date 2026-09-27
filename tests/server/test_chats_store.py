"""Tests for src/server/chats_store.py — Phase 1 CRUD + optimistic locking + idempotency.

Covers CHAT-02 (persistence), CHAT-11 (optimistic lock + idempotency).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from src.server import chats_store
from src.server.chats_store import ConflictError
from src.server.migrations import migrate


def _fresh_db(tmp_path: Path) -> sqlite3.Connection:
    db = tmp_path / "chats.db"
    migrate.run(db)
    return chats_store.open_chats_conn(db)


def test_migrate_creates_chats_messages_citations_message_meta(tmp_path: Path) -> None:
    conn = _fresh_db(tmp_path)
    try:
        tables = {
            r[0]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        }
        assert {"chats", "messages", "citations", "message_meta"}.issubset(tables)
    finally:
        conn.close()


def test_create_chat_returns_id_and_inserts_row(tmp_path: Path) -> None:
    conn = _fresh_db(tmp_path)
    try:
        chat_id = chats_store.create_chat(
            conn,
            retriever="milvus",
            collections=["trading"],
            title="t1",
        )
        assert isinstance(chat_id, str) and chat_id
        row = conn.execute(
            "SELECT id, title, retriever, version FROM chats WHERE id = ?",
            (chat_id,),
        ).fetchone()
        assert row is not None
        assert row["id"] == chat_id
        assert row["title"] == "t1"
        assert row["retriever"] == "milvus"
        assert row["version"] == 0
    finally:
        conn.close()


def test_insert_message_idempotent_returns_existing_on_duplicate_uuid(
    tmp_path: Path,
) -> None:
    conn = _fresh_db(tmp_path)
    try:
        chat_id = chats_store.create_chat(conn)
        msg_id_1, was_new_1 = chats_store.insert_message_idempotent(
            conn,
            chat_id=chat_id,
            message_uuid="uuid-1",
            role="user",
            content="hello",
            expected_version=0,
        )
        assert was_new_1 is True
        # Second call with same uuid returns the same id; version not bumped twice.
        msg_id_2, was_new_2 = chats_store.insert_message_idempotent(
            conn,
            chat_id=chat_id,
            message_uuid="uuid-1",
            role="user",
            content="hello",
            expected_version=99,  # would normally raise; idempotent skips check
        )
        assert msg_id_1 == msg_id_2
        assert was_new_2 is False
        # version still 1, not 2.
        version = conn.execute("SELECT version FROM chats WHERE id = ?", (chat_id,)).fetchone()[
            "version"
        ]
        assert version == 1
    finally:
        conn.close()


def test_atomic_version_bump_raises_on_stale(tmp_path: Path) -> None:
    conn = _fresh_db(tmp_path)
    try:
        chat_id = chats_store.create_chat(conn)
        # First message: version 0 → 1.
        chats_store.insert_message_idempotent(
            conn,
            chat_id=chat_id,
            message_uuid="uuid-1",
            role="user",
            content="hello",
            expected_version=0,
        )
        # Stale write: expected_version=0 but actual is 1.
        with pytest.raises(ConflictError) as excinfo:
            chats_store.insert_message_idempotent(
                conn,
                chat_id=chat_id,
                message_uuid="uuid-2",
                role="user",
                content="stale",
                expected_version=0,
            )
        assert excinfo.value.current_version == 1
    finally:
        conn.close()


def test_get_chat_returns_chat_with_messages_in_order(tmp_path: Path) -> None:
    conn = _fresh_db(tmp_path)
    try:
        chat_id = chats_store.create_chat(conn, title="x")
        chats_store.insert_message_idempotent(
            conn,
            chat_id=chat_id,
            message_uuid="u1",
            role="user",
            content="a",
            expected_version=0,
        )
        chats_store.insert_message_idempotent(
            conn,
            chat_id=chat_id,
            message_uuid="u2",
            role="assistant",
            content="b",
            expected_version=1,
        )
        result = chats_store.get_chat(conn, chat_id)
        assert result is not None
        assert result["id"] == chat_id
        assert len(result["messages"]) == 2
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][1]["role"] == "assistant"
    finally:
        conn.close()


def test_attach_citations_bulk_inserts(tmp_path: Path) -> None:
    conn = _fresh_db(tmp_path)
    try:
        chat_id = chats_store.create_chat(conn)
        msg_id, _ = chats_store.insert_message_idempotent(
            conn,
            chat_id=chat_id,
            message_uuid="u1",
            role="assistant",
            content="ans",
            expected_version=0,
        )
        chats_store.attach_citations(
            conn,
            msg_id,
            [
                {
                    "marker": 1,
                    "child_id": "c1",
                    "parent_id": "p1",
                    "paper_id": "src.pdf",
                    "score_dense": 0.9,
                    "score_sparse": 0.4,
                    "score_rerank": 0.95,
                    "resolved": True,
                },
                {
                    "marker": 7,
                    "child_id": None,
                    "parent_id": None,
                    "paper_id": None,
                    "score_dense": None,
                    "score_sparse": None,
                    "score_rerank": None,
                    "resolved": False,
                },
            ],
        )
        rows = conn.execute(
            "SELECT marker, resolved FROM citations WHERE message_id = ? ORDER BY marker",
            (msg_id,),
        ).fetchall()
        assert [r["marker"] for r in rows] == [1, 7]
        assert [r["resolved"] for r in rows] == [1, 0]
    finally:
        conn.close()


def test_open_chats_conn_enables_foreign_keys(tmp_path: Path) -> None:
    conn = _fresh_db(tmp_path)
    try:
        fk = conn.execute("PRAGMA foreign_keys").fetchone()[0]
        assert fk == 1
    finally:
        conn.close()
