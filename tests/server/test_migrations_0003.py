"""RED tests for src/server/migrations/0003_chats_fts.sql (Plan 02-03).

Pin the migration shape before it lands:
  - PRAGMA user_version bumps to 3
  - chats.archived INTEGER DEFAULT 0 column
  - messages.retriever TEXT column
  - chats_fts virtual table + four INSERT/UPDATE/DELETE triggers
  - Backfill messages.retriever from chats.retriever for existing rows
  - schema_dump.sql committed and matches actual schema

Tests use deferred imports so collection stays green before the SQL lands.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_DUMP = PROJECT_ROOT / "src" / "server" / "migrations" / "schema_dump.sql"


def _run_all(tmp_path: Path) -> Path:
    from src.server.migrations import migrate

    db = tmp_path / "chats.db"
    migrate.run(db)
    return db


def _columns(conn: sqlite3.Connection, table: str) -> dict[str, dict[str, object]]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    cols: dict[str, dict[str, object]] = {}
    for r in rows:
        # PRAGMA table_info: cid, name, type, notnull, dflt_value, pk
        cols[r[1]] = {
            "type": r[2],
            "notnull": r[3],
            "default": r[4],
            "pk": r[5],
        }
    return cols


def test_user_version_bumps_to_3(tmp_path: Path) -> None:
    """0003+ applied; latest user_version is 5 after 0005 maps."""
    db = _run_all(tmp_path)
    conn = sqlite3.connect(str(db))
    try:
        v = conn.execute("PRAGMA user_version").fetchone()[0]
        assert v == 5, f"expected user_version=5 after latest migration; got {v}"
    finally:
        conn.close()


def test_chats_archived_column_exists_default_0(tmp_path: Path) -> None:
    """RED — D-15: chats.archived INTEGER NOT NULL DEFAULT 0."""
    db = _run_all(tmp_path)
    conn = sqlite3.connect(str(db))
    try:
        cols = _columns(conn, "chats")
        assert "archived" in cols, f"chats.archived column missing; cols={list(cols)}"
        assert str(cols["archived"]["type"]).upper().startswith("INT")
        assert str(cols["archived"]["default"]) == "0"
    finally:
        conn.close()


def test_messages_retriever_column_exists(tmp_path: Path) -> None:
    """RED — D-01: messages.retriever TEXT column."""
    db = _run_all(tmp_path)
    conn = sqlite3.connect(str(db))
    try:
        cols = _columns(conn, "messages")
        assert "retriever" in cols, f"messages.retriever column missing; cols={list(cols)}"
    finally:
        conn.close()


def test_messages_retriever_backfilled_from_chats_retriever(
    tmp_path: Path,
) -> None:
    """RED — Pitfall 3: existing messages backfill messages.retriever from
    chats.retriever during migration.

    Inserts a chat + message under user_version=2 then re-runs migration
    to user_version=3; messages.retriever for the existing row must match
    chats.retriever.
    """
    # 1. Apply only 0001 + 0002.
    from src.server.migrations import migrate

    db = tmp_path / "chats.db"
    migrations_dir = migrate.MIGRATIONS_DIR
    only_phase1 = tmp_path / "migs"
    only_phase1.mkdir()
    for p in sorted(migrations_dir.glob("000[1-2]_*.sql")):
        (only_phase1 / p.name).write_text(p.read_text())

    import unittest.mock

    with unittest.mock.patch.object(migrate, "MIGRATIONS_DIR", only_phase1):
        migrate.run(db)
    conn = sqlite3.connect(str(db))
    try:
        conn.execute(
            "INSERT INTO chats (id, title, retriever, collections, version) VALUES (?, ?, ?, ?, ?)",
            ("c1", "t", "paperqa", "[]", 0),
        )
        conn.execute(
            "INSERT INTO messages "
            "(id, chat_id, role, content, message_uuid, version) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("m1", "c1", "user", "hi", "u1", 1),
        )
        conn.commit()
    finally:
        conn.close()

    # 2. Now run all migrations (which adds 0003).
    migrate.run(db)
    conn = sqlite3.connect(str(db))
    try:
        retriever = conn.execute("SELECT retriever FROM messages WHERE id = 'm1'").fetchone()[0]
        assert retriever == "paperqa", (
            f"Pitfall 3: backfill from chats.retriever failed; got {retriever!r}"
        )
    finally:
        conn.close()


def test_chats_fts_virtual_table_exists(tmp_path: Path) -> None:
    """RED — D-13: chats_fts virtual table is created."""
    db = _run_all(tmp_path)
    conn = sqlite3.connect(str(db))
    try:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='chats_fts'"
        ).fetchone()
        assert row is not None, "chats_fts virtual table missing"
    finally:
        conn.close()


def test_four_triggers_exist(tmp_path: Path) -> None:
    """RED — Pitfall 2: messages INSERT/UPDATE/DELETE + chats UPDATE-of-title.

    Four triggers keep chats_fts in sync.
    """
    db = _run_all(tmp_path)
    conn = sqlite3.connect(str(db))
    try:
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type='trigger'").fetchall()
        trig = {r[0] for r in rows}
        assert len(trig) >= 4, f"expected >=4 fts triggers; got {trig}"
    finally:
        conn.close()


def test_rename_updates_fts_snippet(tmp_path: Path) -> None:
    """RED — Pitfall 2: UPDATE chats SET title=... updates chats_fts.title."""
    db = _run_all(tmp_path)
    conn = sqlite3.connect(str(db))
    try:
        conn.execute(
            "INSERT INTO chats (id, title, retriever, collections, version) VALUES (?, ?, ?, ?, ?)",
            ("c1", "first title", "milvus", "[]", 0),
        )
        conn.commit()
        # Trigger fires on UPDATE OF title.
        conn.execute("UPDATE chats SET title = ? WHERE id = 'c1'", ("renamed",))
        conn.commit()

        rows = conn.execute(
            "SELECT chat_id FROM chats_fts WHERE chats_fts MATCH ?",
            ("renamed",),
        ).fetchall()
        assert any(r[0] == "c1" for r in rows), "FTS5 rename trigger did not update title column"
    finally:
        conn.close()


def test_delete_removes_fts_row(tmp_path: Path) -> None:
    """RED — DELETE FROM chats removes the matching chats_fts row."""
    db = _run_all(tmp_path)
    conn = sqlite3.connect(str(db))
    try:
        conn.execute(
            "INSERT INTO chats (id, title, retriever, collections, version) VALUES (?, ?, ?, ?, ?)",
            ("c1", "delete me", "milvus", "[]", 0),
        )
        conn.commit()
        conn.execute("DELETE FROM chats WHERE id = 'c1'")
        conn.commit()
        rows = conn.execute(
            "SELECT chat_id FROM chats_fts WHERE chats_fts MATCH ?",
            ("delete",),
        ).fetchall()
        assert not any(r[0] == "c1" for r in rows), "DELETE trigger did not clean up chats_fts row"
    finally:
        conn.close()


def test_idempotent_re_apply_no_error(tmp_path: Path) -> None:
    """RED — re-running migrations is a no-op (user_version >= 5 already)."""
    from src.server.migrations import migrate

    db = _run_all(tmp_path)
    migrate.run(db)  # second run — must not error
    conn = sqlite3.connect(str(db))
    try:
        v = conn.execute("PRAGMA user_version").fetchone()[0]
        assert v == 5
    finally:
        conn.close()


def test_schema_matches_committed_dump(tmp_path: Path) -> None:
    """RED — D-21 gate: schema_dump.sql is regenerated for the 0003 schema."""
    db = _run_all(tmp_path)
    conn = sqlite3.connect(str(db))
    try:
        rows = conn.execute(
            "SELECT sql FROM sqlite_master WHERE sql IS NOT NULL ORDER BY name"
        ).fetchall()
        actual = "\n".join(r[0] for r in rows).strip()
    finally:
        conn.close()
    expected = SCHEMA_DUMP.read_text().strip()
    assert actual == expected, "schema_dump.sql drift after 0003; regenerate dump and commit."
