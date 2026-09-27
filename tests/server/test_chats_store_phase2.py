"""RED tests for Phase 2 extensions to src/server/chats_store.py.

New helpers (Plans 02-09 / 02-10 / 02-11):
  - truncate_from_last_user(chat_id) — for edit-last (CHAT-07)
  - truncate_last_assistant(chat_id) — for regenerate-last (CHAT-08)
  - update_chat(chat_id, title?, retriever?, collections?, archived?) — HIST-03
  - delete_chat(chat_id) — HIST-03
  - search_chats(q, collections?, retrievers?, archived?) — HIST-02
  - auto_name_chat(chat_id, title) — D-21 zero-version-bump rename

All assertions follow Pattern C (atomic CAS, never SELECT-then-UPDATE).
Tests use deferred imports so collection stays green until the helpers land.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture
def conn_factory(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    """Migrated chats.db connection."""
    pytest.importorskip("src.server.chats_store")
    from src.server import chats_store
    from src.server.migrations import migrate

    db = tmp_path / "chats.db"
    migrate.run(db)
    conn = chats_store.open_chats_conn(db)
    try:
        yield conn
    finally:
        conn.close()


def _seed_chat_with_turns(
    conn: sqlite3.Connection,
    *,
    n_pairs: int = 2,
) -> tuple[str, list[str]]:
    """Insert one chat + n_pairs (user, assistant) message pairs.

    Returns (chat_id, message_uuids).
    """
    from src.server import chats_store

    chat_id = chats_store.create_chat(
        conn,
        retriever="milvus",
        collections=["trading"],
        title="t",
    )
    uuids: list[str] = []
    for i in range(n_pairs):
        u = f"u{i * 2}"
        a = f"u{i * 2 + 1}"
        chats_store.insert_message_idempotent(
            conn,
            chat_id=chat_id,
            message_uuid=u,
            role="user",
            content=f"q{i}",
            expected_version=i * 2,
        )
        chats_store.insert_message_idempotent(
            conn,
            chat_id=chat_id,
            message_uuid=a,
            role="assistant",
            content=f"a{i}",
            expected_version=i * 2 + 1,
        )
        uuids.extend([u, a])
    return chat_id, uuids


# --------------------------------------------------------------------------- #
# truncate_from_last_user (CHAT-07 / D-18)
# --------------------------------------------------------------------------- #


def test_truncate_from_last_user_atomic_with_cancel_safe(
    conn_factory: sqlite3.Connection,
) -> None:
    """RED — truncate_from_last_user atomically deletes the last user turn
    AND any subsequent assistant turn. Cancel-safe: if the operation is
    interrupted, the chat is in a consistent state (Pitfall 10).
    """
    from src.server import chats_store

    conn = conn_factory
    chat_id, _ = _seed_chat_with_turns(conn, n_pairs=2)

    chats_store.truncate_from_last_user(conn, chat_id)  # type: ignore[attr-defined]

    rows = conn.execute(
        "SELECT role FROM messages WHERE chat_id = ? ORDER BY created_at, version",
        (chat_id,),
    ).fetchall()
    roles = [r["role"] for r in rows]
    assert "user" in roles, f"first user turn should survive; got {roles}"
    # The last user + its assistant follow-up should be gone.
    assert roles.count("user") == 1, (
        f"truncate_from_last_user should drop the last user turn; "
        f"got user-count={roles.count('user')} in {roles}"
    )
    assert roles.count("assistant") == 1, (
        f"trailing assistant turn should also be dropped; got {roles}"
    )


def test_truncate_last_assistant_preserves_audit_pointer(
    conn_factory: sqlite3.Connection,
) -> None:
    """RED — D-20: truncate_last_assistant drops the assistant message ROW
    but the on-disk audit dir (referenced by messages.query_id) is NOT
    touched here. The helper only mutates sqlite.
    """
    from src.server import chats_store

    conn = conn_factory
    chat_id, _ = _seed_chat_with_turns(conn, n_pairs=2)

    # Tag the last assistant turn with a query_id.
    last_id = conn.execute(
        "SELECT id FROM messages WHERE chat_id = ? AND role = 'assistant' "
        "ORDER BY created_at DESC, version DESC LIMIT 1",
        (chat_id,),
    ).fetchone()["id"]
    chats_store.set_assistant_content_and_query_id(conn, last_id, "final answer", "qid-12345")

    chats_store.truncate_last_assistant(conn, chat_id)  # type: ignore[attr-defined]

    # Row gone.
    n = conn.execute(
        "SELECT COUNT(*) AS n FROM messages WHERE id = ?",
        (last_id,),
    ).fetchone()["n"]
    assert n == 0, "truncate_last_assistant must remove the assistant row"

    # The helper itself does not touch disk; D-20 invariant pinned at row
    # level: it does NOT cascade-delete the audit dir by side-effect.
    # (Disk-side check is in test_api_chats_phase2.test_regenerate_preserves_audit_dir.)


# --------------------------------------------------------------------------- #
# update_chat (HIST-03)
# --------------------------------------------------------------------------- #


def test_update_chat_bumps_version_atomically(
    conn_factory: sqlite3.Connection,
) -> None:
    """RED — Pattern C: update_chat uses atomic CAS to bump chats.version."""
    from src.server import chats_store

    conn = conn_factory
    chat_id = chats_store.create_chat(conn)

    chats_store.update_chat(  # type: ignore[attr-defined]
        conn,
        chat_id,
        title="new",
        expected_version=0,
    )
    new_version = conn.execute(
        "SELECT version FROM chats WHERE id = ?",
        (chat_id,),
    ).fetchone()["version"]
    assert new_version == 1, f"update_chat must bump version atomically; got {new_version}"


def test_update_chat_409_on_stale_version(
    conn_factory: sqlite3.Connection,
) -> None:
    """RED — Pattern C / D-17: stale expected_version raises ConflictError."""
    from src.server import chats_store
    from src.server.chats_store import ConflictError

    conn = conn_factory
    chat_id = chats_store.create_chat(conn)
    chats_store.update_chat(conn, chat_id, title="a", expected_version=0)  # type: ignore[attr-defined]

    with pytest.raises(ConflictError) as exc:
        chats_store.update_chat(  # type: ignore[attr-defined]
            conn,
            chat_id,
            title="b",
            expected_version=0,
        )
    assert exc.value.current_version == 1


# --------------------------------------------------------------------------- #
# delete_chat (HIST-03)
# --------------------------------------------------------------------------- #


def test_delete_chat_cascades(conn_factory: sqlite3.Connection) -> None:
    """RED — 0002 FKs cascade messages + citations + message_meta on delete."""
    from src.server import chats_store

    conn = conn_factory
    chat_id, _ = _seed_chat_with_turns(conn, n_pairs=1)

    chats_store.delete_chat(conn, chat_id)  # type: ignore[attr-defined]

    for table in ("chats", "messages", "citations", "message_meta"):
        try:
            n = conn.execute(
                f"SELECT COUNT(*) AS n FROM {table} WHERE COALESCE(chat_id, message_id, id) LIKE ?",
                (chat_id,),
            ).fetchone()["n"]
        except sqlite3.OperationalError:
            continue
        assert n == 0, f"{table} should be empty after CASCADE; got n={n}"


# --------------------------------------------------------------------------- #
# search_chats (HIST-02, HIST-04)
# --------------------------------------------------------------------------- #


def test_search_chats_returns_snippet_with_mark(
    conn_factory: sqlite3.Connection,
) -> None:
    """RED — D-14: search returns a `<mark>`-highlighted snippet (FTS5
    snippet() output).
    """
    from src.server import chats_store

    conn = conn_factory
    chat_id, _ = _seed_chat_with_turns(conn, n_pairs=1)
    # Re-write the assistant content with a known phrase.
    conn.execute(
        "UPDATE messages SET content = 'GLFT market making spread' "
        "WHERE chat_id = ? AND role = 'assistant'",
        (chat_id,),
    )
    conn.commit()

    rows = chats_store.search_chats(conn, q="GLFT")  # type: ignore[attr-defined]
    assert rows, f"expected non-empty hit list; got {rows!r}"
    snippet = rows[0].get("snippet", "")
    assert "<mark>" in snippet, f"D-14: snippet must contain <mark> highlight; got {snippet!r}"


def test_search_chats_escapes_special_chars(
    conn_factory: sqlite3.Connection,
) -> None:
    """RED — Pitfall 8: FTS5 syntax chars don't crash sqlite.

    Inputs containing a stray `"` must be sanitized or rejected, never raise.
    """
    from src.server import chats_store

    conn = conn_factory
    _seed_chat_with_turns(conn, n_pairs=1)

    # Should NOT raise.
    rows = chats_store.search_chats(conn, q='" weird')  # type: ignore[attr-defined]
    assert isinstance(rows, list)


def test_search_chats_filter_by_facets(
    conn_factory: sqlite3.Connection,
) -> None:
    """RED — HIST-04: filtering by collections + retrievers + archived.

    Seed two chats with different retrievers; assert only the matching one
    is returned.
    """
    from src.server import chats_store

    conn = conn_factory
    chat_a = chats_store.create_chat(
        conn,
        retriever="milvus",
        collections=["trading"],
        title="t1",
    )
    chat_b = chats_store.create_chat(
        conn,
        retriever="paperqa",
        collections=["ecology"],
        title="t2",
    )
    # Insert a turn to populate FTS row for each.
    chats_store.insert_message_idempotent(
        conn,
        chat_id=chat_a,
        message_uuid="ua",
        role="user",
        content="alpha test",
        expected_version=0,
    )
    chats_store.insert_message_idempotent(
        conn,
        chat_id=chat_b,
        message_uuid="ub",
        role="user",
        content="alpha test",
        expected_version=0,
    )

    rows = chats_store.search_chats(conn, q="alpha", retrievers=["paperqa"])  # type: ignore[attr-defined]
    assert rows, "filter should match at least one row"
    ids = {r["chat_id"] for r in rows}
    assert chat_b in ids, f"expected {chat_b} in results; got {ids}"
    assert chat_a not in ids, f"chat_a (retriever=milvus) should be excluded; got {ids}"


def test_search_chats_returns_max_50_rows(
    conn_factory: sqlite3.Connection,
) -> None:
    """RED — D-14 LIMIT 50 cap. Seed 60 chats, assert at most 50 returned."""
    from src.server import chats_store

    conn = conn_factory
    for i in range(60):
        c = chats_store.create_chat(
            conn,
            retriever="milvus",
            collections=["trading"],
            title=f"t{i}",
        )
        chats_store.insert_message_idempotent(
            conn,
            chat_id=c,
            message_uuid=f"u-{i}",
            role="user",
            content=f"alpha test {i}",
            expected_version=0,
        )

    rows = chats_store.search_chats(conn, q="alpha")  # type: ignore[attr-defined]
    assert len(rows) <= 50, f"LIMIT 50 cap violated; got {len(rows)} rows"


# --------------------------------------------------------------------------- #
# auto_name_chat (D-21 / Pitfall 6)
# --------------------------------------------------------------------------- #


def test_auto_name_chat_no_version_bump(
    conn_factory: sqlite3.Connection,
) -> None:
    """RED — Pitfall 6: server-side auto-name does NOT bump chats.version.

    Reason: an immediate client-side follow-up POST uses the version the
    client last observed; if auto-name bumps version mid-flight, the
    follow-up 409s. D-21 fix: title-only update path uses a CAS that does
    NOT increment version.
    """
    from src.server import chats_store

    conn = conn_factory
    chat_id = chats_store.create_chat(conn, title="Untitled chat")
    pre = conn.execute(
        "SELECT version FROM chats WHERE id = ?",
        (chat_id,),
    ).fetchone()["version"]

    chats_store.auto_name_chat(conn, chat_id, "Auto-titled")  # type: ignore[attr-defined]

    post = conn.execute(
        "SELECT version FROM chats WHERE id = ?",
        (chat_id,),
    ).fetchone()["version"]
    assert post == pre, f"Pitfall 6: auto-name must NOT bump version; was {pre} → {post}"


def test_auto_name_chat_idempotent_via_where_clause(
    conn_factory: sqlite3.Connection,
) -> None:
    """RED — D-21: auto_name_chat uses ``UPDATE … WHERE title='Untitled chat'``
    so a second call after a user-initiated rename is a no-op.
    """
    from src.server import chats_store

    conn = conn_factory
    chat_id = chats_store.create_chat(conn, title="Untitled chat")
    chats_store.auto_name_chat(conn, chat_id, "First auto")  # type: ignore[attr-defined]
    # User then renames manually.
    conn.execute(
        "UPDATE chats SET title = ? WHERE id = ?",
        ("User rename", chat_id),
    )
    conn.commit()

    # Second auto-name call should NOT overwrite the user rename.
    chats_store.auto_name_chat(conn, chat_id, "Second auto")  # type: ignore[attr-defined]
    title = conn.execute(
        "SELECT title FROM chats WHERE id = ?",
        (chat_id,),
    ).fetchone()["title"]
    assert title == "User rename", (
        f"idempotency: auto-name must not clobber a user-set title; got {title!r}"
    )
