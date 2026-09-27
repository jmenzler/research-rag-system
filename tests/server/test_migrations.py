"""Tests for src/server/migrations/migrate.py — idempotency, fail-fast, schema dump match."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from src.server.migrations import migrate

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_DUMP = PROJECT_ROOT / "src" / "server" / "migrations" / "schema_dump.sql"


def _dump_schema(db_path: Path) -> str:
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT sql FROM sqlite_master WHERE sql IS NOT NULL ORDER BY name"
        ).fetchall()
        return "\n".join(r[0] for r in rows)
    finally:
        conn.close()


def test_migrate_applies_through_latest_and_sets_user_version(tmp_path: Path) -> None:
    """user_version tracks the highest applied migration prefix; bumped to 5 after 0005."""
    db = tmp_path / "chats.db"
    migrate.run(db)
    conn = sqlite3.connect(str(db))
    try:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 5
    finally:
        conn.close()


def test_migrate_is_idempotent(tmp_path: Path) -> None:
    db = tmp_path / "chats.db"
    migrate.run(db)
    migrate.run(db)  # second run = no-op
    conn = sqlite3.connect(str(db))
    try:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 5
    finally:
        conn.close()


def test_migrate_enables_wal(tmp_path: Path) -> None:
    db = tmp_path / "chats.db"
    migrate.run(db)
    conn = sqlite3.connect(str(db))
    try:
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode.lower() == "wal"
    finally:
        conn.close()


def test_schema_matches_committed_dump(tmp_path: Path) -> None:
    db = tmp_path / "chats.db"
    migrate.run(db)
    actual = _dump_schema(db).strip()
    expected = SCHEMA_DUMP.read_text().strip()
    assert actual == expected, (
        f"Schema drift detected. Actual:\n{actual!r}\nExpected:\n{expected!r}\n"
        "If this is an intentional change, regenerate schema_dump.sql."
    )


def test_migrate_creates_phase_1_tables(tmp_path: Path) -> None:
    db = tmp_path / "chats.db"
    migrate.run(db)
    conn = sqlite3.connect(str(db))
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        tables = {r[0] for r in rows}
        assert {"chats", "messages", "citations", "message_meta"}.issubset(tables), (
            f"Missing Phase 1 tables. Found: {tables}"
        )
    finally:
        conn.close()


def test_messages_uuid_unique_constraint(tmp_path: Path) -> None:
    db = tmp_path / "chats.db"
    migrate.run(db)
    conn = sqlite3.connect(str(db))
    try:
        conn.execute(
            "INSERT INTO chats (id, title, retriever, collections, version) VALUES (?, ?, ?, ?, ?)",
            ("chat1", "test", "milvus", "[]", 0),
        )
        conn.execute(
            "INSERT INTO messages (id, chat_id, role, content, message_uuid, version) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("msg1", "chat1", "user", "hello", "uuid-shared", 1),
        )
        conn.commit()
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO messages "
                "(id, chat_id, role, content, message_uuid, version) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                ("msg2", "chat1", "assistant", "world", "uuid-shared", 2),
            )
            conn.commit()
    finally:
        conn.close()


def test_messages_chat_id_fk_cascade_on_delete(tmp_path: Path) -> None:
    db = tmp_path / "chats.db"
    migrate.run(db)
    conn = sqlite3.connect(str(db))
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute(
            "INSERT INTO chats (id, title, retriever, collections, version) VALUES (?, ?, ?, ?, ?)",
            ("chat1", "test", "milvus", "[]", 0),
        )
        conn.execute(
            "INSERT INTO messages (id, chat_id, role, content, message_uuid, version) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("msg1", "chat1", "user", "hello", "uuid-1", 1),
        )
        conn.commit()
        # Sanity: message exists.
        assert (
            conn.execute("SELECT COUNT(*) FROM messages WHERE chat_id = ?", ("chat1",)).fetchone()[
                0
            ]
            == 1
        )
        # Delete the parent chat; CASCADE should remove the message.
        conn.execute("DELETE FROM chats WHERE id = ?", ("chat1",))
        conn.commit()
        assert (
            conn.execute("SELECT COUNT(*) FROM messages WHERE chat_id = ?", ("chat1",)).fetchone()[
                0
            ]
            == 0
        )
    finally:
        conn.close()


def test_failed_migration_rolls_back_user_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "chats.db"
    migrate.run(db)  # apply 0001 cleanly first
    # Inject a broken 0002 by pointing MIGRATIONS_DIR at a tmp dir with both 0001 + broken 0002
    broken_dir = tmp_path / "migs"
    broken_dir.mkdir()
    (broken_dir / "0001_schema_version.sql").write_text("PRAGMA journal_mode = WAL;")
    (broken_dir / "0002_broken.sql").write_text("CREATE TABLE good (id INTEGER); BUSTED_KEYWORD;")
    monkeypatch.setattr(migrate, "MIGRATIONS_DIR", broken_dir)
    db2 = tmp_path / "chats2.db"
    with pytest.raises(sqlite3.Error):
        migrate.run(db2)
    conn = sqlite3.connect(str(db2))
    try:
        # 0001 succeeded, 0002 rolled back → user_version is 1, not 2
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 1
        # 'good' table from 0002 must not exist (rolled back)
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        assert "good" not in tables
    finally:
        conn.close()
