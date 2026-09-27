"""sqlite CRUD over parents/chats.db. Atomic optimistic locking + idempotency.

Phase 1 brownfield-additive surface (D-16/D-17/D-18). Connection management is
caller-owned (open via :func:`open_chats_conn`; close via context manager or
``finally``). Schema is created by :mod:`src.server.migrations.migrate` at
startup, NOT here.

Every SQL statement uses ``?`` placeholders (CRIT: SQL injection prevention,
RESEARCH §Known Threat Patterns).

The atomic version bump (:func:`insert_message_idempotent`) is a single
``UPDATE chats SET version = version + 1 WHERE id = ? AND version = ?``
statement — NEVER a SELECT-then-UPDATE pair (TOCTOU race per RESEARCH
§Anti-Patterns).
"""
from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHATS_DB_PATH = PROJECT_ROOT / "parents" / "chats.db"


class ConflictError(Exception):
    """Raised when the optimistic-lock CAS fails (chats.version mismatch).

    Attributes:
        current_version: The actual version observed in the database, used by
            the API layer to populate the HTTP 409 response body (D-17: client
            UI shows a toast and refetches).
    """

    def __init__(self, message: str, *, current_version: int) -> None:
        super().__init__(message)
        self.current_version = current_version


def open_chats_conn(db_path: Path | None = None) -> sqlite3.Connection:
    """Open a connection to chats.db with foreign keys enabled.

    sqlite default is ``foreign_keys = OFF`` — without this PRAGMA the
    ``ON DELETE CASCADE`` declarations in 0002_chats.sql are no-ops (T-01-W1-06).
    """
    path = db_path if db_path is not None else CHATS_DB_PATH
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn


def create_chat(
    conn: sqlite3.Connection,
    *,
    retriever: str = "milvus",
    collections: list[str] | None = None,
    title: str | None = "Untitled chat",
) -> str:
    """Insert a new chat row and return its id (uuid4 hex)."""
    chat_id = uuid.uuid4().hex
    conn.execute(
        "INSERT INTO chats (id, title, retriever, collections, version) "
        "VALUES (?, ?, ?, ?, 0)",
        (chat_id, title, retriever, json.dumps(collections or [])),
    )
    conn.commit()
    return chat_id


def list_chats(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Return all chats ordered by most-recently-updated first."""
    rows = conn.execute(
        "SELECT id, title, retriever, collections, archived, version, "
        "created_at, updated_at "
        "FROM chats ORDER BY updated_at DESC"
    ).fetchall()
    out: list[dict[str, Any]] = []
    for r in rows:
        d = dict(r)
        d["archived"] = bool(d.get("archived"))
        out.append(d)
    return out


def get_chat(conn: sqlite3.Connection, chat_id: str) -> dict[str, Any] | None:
    """Return one chat plus its messages + per-message citations (append order).

    Each message dict carries a ``citations`` list (empty for user turns and for
    assistants with no extracted markers). Without this join the CitationToken
    pipeline on the frontend has no marker→chunk map on reload, so `[1]`-style
    markers fall through to the unresolved ``[?]`` pill even though the rows
    exist in the ``citations`` table. Live SSE delivers the same shape via the
    ``citations`` event — reload now matches.
    """
    chat_row = conn.execute(
        "SELECT id, title, retriever, collections, archived, version, "
        "created_at, updated_at "
        "FROM chats WHERE id = ?",
        (chat_id,),
    ).fetchone()
    if chat_row is None:
        return None
    messages = conn.execute(
        "SELECT id, role, content, query_id, message_uuid, created_at, version "
        "FROM messages WHERE chat_id = ? ORDER BY created_at ASC, version ASC",
        (chat_id,),
    ).fetchall()
    citation_rows = conn.execute(
        "SELECT message_id, marker, child_id, parent_id, paper_id, "
        "       score_dense, score_sparse, score_rerank, resolved "
        "FROM citations WHERE message_id IN ("
        "    SELECT id FROM messages WHERE chat_id = ?"
        ") "
        "ORDER BY message_id, marker",
        (chat_id,),
    ).fetchall()
    citations_by_message: dict[str, list[dict[str, Any]]] = {}
    for c in citation_rows:
        row = dict(c)
        msg_id = row.pop("message_id")
        # SQLite stores resolved as 0/1; surface as JSON boolean to match the
        # SSE citations event shape (CitationPayload.resolved: boolean).
        row["resolved"] = bool(row["resolved"])
        citations_by_message.setdefault(msg_id, []).append(row)
    result = dict(chat_row)
    result["archived"] = bool(result.get("archived"))
    result["messages"] = [
        {**dict(m), "citations": citations_by_message.get(m["id"], [])}
        for m in messages
    ]
    return result


def insert_message_idempotent(
    conn: sqlite3.Connection,
    chat_id: str,
    message_uuid: str,
    role: str,
    content: str,
    expected_version: int,
) -> tuple[str, bool]:
    """Insert a message row with optimistic locking + idempotency.

    Returns:
        ``(message_id, was_new)``. When ``was_new=False`` the ``message_uuid``
        was already present; caller should reuse the existing row's
        ``query_id`` (no new LLM call, no new audit dir — D-18).

    Raises:
        ConflictError: When ``chats.version != expected_version`` (HTTP 409 —
            D-17). The error carries the observed ``current_version`` so the
            API layer can surface it to the client.
    """
    # 1. Idempotency: UNIQUE on message_uuid — early return if present.
    cur = conn.execute(
        "SELECT id FROM messages WHERE message_uuid = ?", (message_uuid,)
    )
    existing = cur.fetchone()
    if existing is not None:
        return str(existing[0]), False

    # 2. Atomic version CAS — sqlite UPDATE is atomic. NEVER a
    #    SELECT-then-UPDATE pair (TOCTOU). cursor.rowcount == 1 is the only
    #    success path; rowcount == 0 means version mismatch OR chat missing.
    cur = conn.execute(
        "UPDATE chats SET version = version + 1, "
        "updated_at = CURRENT_TIMESTAMP "
        "WHERE id = ? AND version = ?",
        (chat_id, expected_version),
    )
    if cur.rowcount != 1:
        row = conn.execute(
            "SELECT version FROM chats WHERE id = ?", (chat_id,)
        ).fetchone()
        current = int(row["version"]) if row else -1
        raise ConflictError(
            f"chat {chat_id} version mismatch "
            f"(expected {expected_version}, current {current})",
            current_version=current,
        )

    # 3. Insert the message at the bumped version.
    new_id = uuid.uuid4().hex
    conn.execute(
        "INSERT INTO messages "
        "(id, chat_id, role, content, query_id, message_uuid, version) "
        "VALUES (?, ?, ?, ?, NULL, ?, ?)",
        (new_id, chat_id, role, content, message_uuid, expected_version + 1),
    )
    conn.commit()
    return new_id, True


def set_assistant_content_and_query_id(
    conn: sqlite3.Connection,
    message_id: str,
    content: str,
    query_id: str | None,
) -> None:
    """Update an already-inserted assistant row with final content + query_id.

    Used by the SSE handler in :mod:`src.server.api_chats` (Plan 03/04) after
    ``run_query_pipeline`` finishes — the assistant row is inserted with empty
    content first so the optimistic-lock CAS fires BEFORE any LLM work begins.
    """
    conn.execute(
        "UPDATE messages SET content = ?, query_id = ? WHERE id = ?",
        (content, query_id, message_id),
    )
    conn.commit()


def attach_citations(
    conn: sqlite3.Connection,
    message_id: str,
    citations: list[dict[str, Any]],
) -> None:
    """Bulk-insert validated citations rows for an assistant message.

    Uses ``INSERT OR REPLACE`` so a retry overwrites stale rows for the same
    ``(message_id, marker)`` PK rather than failing on conflict — safe because
    the caller (SSE handler) emits the full validated set per message.
    """
    conn.executemany(
        "INSERT OR REPLACE INTO citations "
        "(message_id, marker, child_id, parent_id, paper_id, "
        " score_dense, score_sparse, score_rerank, resolved) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                message_id,
                c["marker"],
                c["child_id"],
                c["parent_id"],
                c["paper_id"],
                c["score_dense"],
                c["score_sparse"],
                c["score_rerank"],
                1 if c["resolved"] else 0,
            )
            for c in citations
        ],
    )
    conn.commit()


def set_message_meta(
    conn: sqlite3.Connection,
    message_id: str,
    key: str,
    value: dict[str, Any] | list[Any] | str | int | float,
) -> None:
    """Write a ``(message_id, key, json.dumps(value))`` row into ``message_meta``.

    The bag holds flexible per-turn payloads (``usage``, ``cost_usd``,
    ``latency_ms``, ``unresolved_markers``, etc.) without schema churn — D-16's
    ``message_meta`` table.
    """
    conn.execute(
        "INSERT OR REPLACE INTO message_meta (message_id, key, value) "
        "VALUES (?, ?, ?)",
        (message_id, key, json.dumps(value)),
    )
    conn.commit()


# --------------------------------------------------------------------------- #
# Phase 2 helpers (Plan 02-04)
# --------------------------------------------------------------------------- #

# Allow-list of fields the PATCH path can mutate. Mass-assignment of
# id/version/created_at/updated_at is rejected at the API layer (Pydantic
# extra='forbid'); this list is the store-layer second line of defence.
_UPDATE_CHAT_FIELDS: tuple[str, ...] = (
    "title", "retriever", "collections", "archived",
)


def update_chat(
    conn: sqlite3.Connection,
    chat_id: str,
    *,
    title: str | None = None,
    retriever: str | None = None,
    collections: list[str] | None = None,
    archived: bool | None = None,
    expected_version: int,
) -> None:
    """Partial PATCH on ``chats`` with Pattern C atomic CAS (HIST-03).

    At least one of ``title``/``retriever``/``collections``/``archived`` must
    be non-None; passing all-None raises ``ValueError`` (the API layer surfaces
    that as 400).

    Atomicity: builds a dynamic ``SET col = ?, ... , version = version + 1,
    updated_at = CURRENT_TIMESTAMP WHERE id = ? AND version = ?`` — a single
    statement, never SELECT-then-UPDATE (TOCTOU per RESEARCH §Anti-Patterns).
    ``cursor.rowcount == 1`` is the only success signal; otherwise we SELECT
    the current version for the 409 body and raise :class:`ConflictError`.
    """
    sets: list[str] = []
    params: list[Any] = []
    if title is not None:
        sets.append("title = ?")
        params.append(title)
    if retriever is not None:
        sets.append("retriever = ?")
        params.append(retriever)
    if collections is not None:
        sets.append("collections = ?")
        params.append(json.dumps(collections))
    if archived is not None:
        sets.append("archived = ?")
        params.append(1 if archived else 0)

    if not sets:
        raise ValueError(
            "update_chat requires at least one of "
            "title/retriever/collections/archived"
        )

    sets.append("version = version + 1")
    sets.append("updated_at = CURRENT_TIMESTAMP")
    sql = (
        "UPDATE chats SET " + ", ".join(sets) + " "
        "WHERE id = ? AND version = ?"
    )
    params.extend([chat_id, expected_version])

    cur = conn.execute(sql, tuple(params))
    if cur.rowcount != 1:
        row = conn.execute(
            "SELECT version FROM chats WHERE id = ?", (chat_id,),
        ).fetchone()
        current = int(row["version"]) if row else -1
        raise ConflictError(
            f"chat {chat_id} version mismatch "
            f"(expected {expected_version}, current {current})",
            current_version=current,
        )
    conn.commit()


def delete_chat(conn: sqlite3.Connection, chat_id: str) -> bool:
    """Hard-delete a chat. FKs CASCADE messages/citations/message_meta.

    Sync triggers (``messages_ad``) remove message-derived ``chats_fts`` rows
    per row. The title-only sentinel rows ``chats_title_au`` writes for empty
    chats (rowid = -chats.rowid) are NOT covered by an FK, so we issue an
    explicit ``DELETE FROM chats_fts WHERE chat_id = ?`` first (per the Plan
    02-03 SUMMARY contract handoff).

    Returns:
        True if the chat row was removed, False if no row matched ``chat_id``.
    """
    conn.execute("DELETE FROM chats_fts WHERE chat_id = ?", (chat_id,))
    cur = conn.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
    conn.commit()
    return cur.rowcount > 0


def truncate_from_last_user(
    conn: sqlite3.Connection,
    chat_id: str,
    *,
    expected_version: int | None = None,
) -> None:
    """Drop the last user turn + any subsequent assistant turn (CHAT-07).

    Behaviour (RESEARCH §Pitfall 10):
      1. SELECT the created_at of the last role='user' message; raise
         ``ValueError`` if no user message exists.
      2. Pattern C version CAS (when ``expected_version`` is supplied).
      3. DELETE FROM messages WHERE chat_id = ? AND created_at >= cutoff.
      4. Does NOT ``conn.commit()`` — the caller (SSE edit-last handler in
         Plan 02-11) owns the transaction so the new user-row INSERT joins
         this truncation atomically. The cancel-during-stream path then
         matches D-22 (cancel = lost) without leaving the chat in a
         half-truncated state.

    Args:
        expected_version: Optional optimistic-lock guard. When ``None`` the
            CAS is skipped — used by the bare unit-test fixture; production
            callers (Plan 02-11) always pass a version.
    """
    # Use (created_at, version) compound ordering — sqlite CURRENT_TIMESTAMP
    # has 1-second resolution so messages inserted in the same call all share
    # a created_at; messages.version (the chats.version snapshot at insert) is
    # the tiebreaker that gives us a total order.
    row = conn.execute(
        "SELECT created_at, version FROM messages "
        "WHERE chat_id = ? AND role = 'user' "
        "ORDER BY created_at DESC, version DESC LIMIT 1",
        (chat_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"chat {chat_id} has no user message to truncate")
    cutoff_ts = row["created_at"]
    cutoff_ver = row["version"]

    if expected_version is not None:
        cur = conn.execute(
            "UPDATE chats SET version = version + 1, "
            "updated_at = CURRENT_TIMESTAMP "
            "WHERE id = ? AND version = ?",
            (chat_id, expected_version),
        )
        if cur.rowcount != 1:
            row2 = conn.execute(
                "SELECT version FROM chats WHERE id = ?", (chat_id,),
            ).fetchone()
            current = int(row2["version"]) if row2 else -1
            raise ConflictError(
                f"chat {chat_id} version mismatch "
                f"(expected {expected_version}, current {current})",
                current_version=current,
            )

    # Delete rows at the cutoff (same created_at AND version >= cutoff_ver)
    # plus anything strictly after the cutoff timestamp.
    conn.execute(
        "DELETE FROM messages "
        "WHERE chat_id = ? AND ("
        "       created_at > ? "
        "    OR (created_at = ? AND version >= ?)"
        ")",
        (chat_id, cutoff_ts, cutoff_ts, cutoff_ver),
    )
    # NOTE: intentionally no conn.commit() — Plan 02-11's SSE handler will
    # commit after also inserting the replacement user row, so the
    # truncation + new-turn insert form one atomic unit (Pitfall 10).


def atomic_edit_last(
    conn: sqlite3.Connection,
    chat_id: str,
    *,
    new_user_uuid: str,
    new_content: str,
    expected_version: int,
) -> str:
    """Atomic edit-last: truncate from last user + insert new user row in
    one transaction. Returns the new message_id.

    Combines :func:`truncate_from_last_user` (no-commit) with the
    new-user-row INSERT so that a cancel mid-stream (after this returns but
    before the SSE finishes) does NOT leave the chat half-truncated
    (RESEARCH §Pitfall 10). The commit happens exactly once at the end of
    this helper — either both mutations land or neither does.

    On UNIQUE(message_uuid) violation: raises ``ValueError``. Edit-last must
    mint a FRESH uuid each time (RESEARCH §Pitfall 4) — the idempotency
    replay path from :func:`insert_message_idempotent` is NOT desirable here
    because the client's previous edit attempt produced different content.

    Args:
        new_user_uuid: Fresh ``message_uuid`` for the replacement user row.
            Must NOT match any existing message_uuid in the database.
        new_content: Replacement user-turn text.
        expected_version: Pattern C atomic CAS guard. Mismatched → raises
            :class:`ConflictError` (transaction rolled back via the sqlite3
            implicit rollback on exception; the connection's autocommit
            behaviour ensures no partial state survives).

    Returns:
        The new user row's ``message_id``.
    """
    # Pre-check UNIQUE(message_uuid) so we fail fast before truncate_from_last_user
    # touches any rows. truncate_from_last_user does NOT commit, but the chats
    # version bump it performs would partially apply if we let the INSERT below
    # raise — easier to short-circuit here.
    existing = conn.execute(
        "SELECT 1 FROM messages WHERE message_uuid = ?", (new_user_uuid,),
    ).fetchone()
    if existing is not None:
        raise ValueError(
            f"message_uuid {new_user_uuid!r} already exists; "
            "edit-last must mint a fresh uuid (RESEARCH §Pitfall 4)"
        )

    # truncate_from_last_user performs the version CAS + deletes the rows
    # WITHOUT committing — the caller (this helper) owns the transaction.
    truncate_from_last_user(conn, chat_id, expected_version=expected_version)

    # Insert the replacement user row at the bumped version (expected_version + 1
    # is the version truncate_from_last_user advanced chats.version to).
    new_id = uuid.uuid4().hex
    conn.execute(
        "INSERT INTO messages "
        "(id, chat_id, role, content, query_id, message_uuid, version) "
        "VALUES (?, ?, 'user', ?, NULL, ?, ?)",
        (new_id, chat_id, new_content, new_user_uuid, expected_version + 1),
    )
    conn.commit()
    return new_id


def truncate_last_assistant(
    conn: sqlite3.Connection,
    chat_id: str,
    *,
    expected_version: int | None = None,
) -> str | None:
    """Drop the last assistant message ROW only (regenerate-last, CHAT-08).

    The on-disk audit dir (``logs/queries/<query_id>/``) is NOT touched here:
    the D-20 invariant "audit dir survives row deletion" is preserved at the
    forensic layer (any caller that wants to remove the dir does it
    explicitly, never as a side-effect of this helper).

    Args:
        expected_version: Optional Pattern C CAS. ``None`` skips the check
            (used by raw unit tests; Plan 02-11 always supplies a version).

    Returns:
        The deleted assistant's ``query_id`` (so the caller can reference the
        preserved audit dir if needed), or ``None`` when the chat had no
        assistant message.
    """
    if expected_version is not None:
        cur = conn.execute(
            "UPDATE chats SET version = version + 1, "
            "updated_at = CURRENT_TIMESTAMP "
            "WHERE id = ? AND version = ?",
            (chat_id, expected_version),
        )
        if cur.rowcount != 1:
            row2 = conn.execute(
                "SELECT version FROM chats WHERE id = ?", (chat_id,),
            ).fetchone()
            current = int(row2["version"]) if row2 else -1
            raise ConflictError(
                f"chat {chat_id} version mismatch "
                f"(expected {expected_version}, current {current})",
                current_version=current,
            )

    row = conn.execute(
        "SELECT id, query_id FROM messages "
        "WHERE chat_id = ? AND role = 'assistant' "
        "ORDER BY created_at DESC, version DESC LIMIT 1",
        (chat_id,),
    ).fetchone()
    if row is None:
        conn.commit()
        return None
    msg_id = row["id"]
    query_id = row["query_id"]
    conn.execute("DELETE FROM messages WHERE id = ?", (msg_id,))
    conn.commit()
    return str(query_id) if query_id is not None else None


def search_chats(
    conn: sqlite3.Connection,
    q: str,
    *,
    collections: list[str] | None = None,
    retrievers: list[str] | None = None,
    archived: bool = False,
) -> list[dict[str, Any]]:
    """FTS5-backed history search (HIST-02 / HIST-04).

    Sanitizes user input by wrapping it as a phrase-match expression
    (``safe_q = '"' + q.replace('"', '""') + '"'``, RESEARCH §Pitfall 8) so
    FTS5 operator characters in the input cannot raise OperationalError. A
    final try/except still catches any pathological case and returns an
    empty list — the API layer surfaces a 400 only on the (now impossible)
    syntax-error path.

    Args:
        q: Free-text query. Empty/whitespace-only returns ``[]`` without a
            sqlite roundtrip.
        collections: Optional intersection filter; an empty ``chat.collections``
            list is treated as "matches every requested collection" (D-03
            ergonomics).
        retrievers: Optional set filter on ``chats.retriever``.
        archived: Filter on ``chats.archived`` (default: only active chats).

    Returns:
        A de-duplicated, ranked, at-most-50 list of ``{chat_id, title,
        snippet}`` dicts (D-14 LIMIT 50 cap).
    """
    if not q.strip():
        return []
    safe_q = '"' + q.replace('"', '""') + '"'

    try:
        rows = conn.execute(
            "SELECT c.id            AS chat_id, "
            "       c.title         AS title, "
            "       c.retriever     AS retriever, "
            "       c.collections   AS collections_json, "
            "       snippet(chats_fts, -1, '<mark>', '</mark>', '…', 32) AS snippet "
            "  FROM chats_fts "
            "  JOIN chats c ON c.id = chats_fts.chat_id "
            " WHERE chats_fts MATCH ? "
            "   AND c.archived = ? "
            " ORDER BY rank "
            " LIMIT 200",
            (safe_q, 1 if archived else 0),
        ).fetchall()
    except sqlite3.OperationalError:
        return []

    retriever_set: set[str] | None = (
        set(retrievers) if retrievers else None
    )
    requested_collections: set[str] | None = (
        set(collections) if collections else None
    )

    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for r in rows:
        chat_id = r["chat_id"]
        if chat_id in seen:
            continue
        if retriever_set is not None and r["retriever"] not in retriever_set:
            continue
        if requested_collections is not None:
            chat_cols_raw = r["collections_json"] or "[]"
            try:
                chat_cols = json.loads(chat_cols_raw)
            except (json.JSONDecodeError, TypeError):
                chat_cols = []
            chat_col_set = set(chat_cols) if isinstance(chat_cols, list) else set()
            # Empty chat.collections means "all four collections" (D-03).
            if chat_col_set and not (chat_col_set & requested_collections):
                continue
        out.append({
            "chat_id": chat_id,
            "title": r["title"],
            "snippet": r["snippet"],
        })
        seen.add(chat_id)
        if len(out) >= 50:
            break
    return out


# D-22: hard cap on auto-generated chat titles.
_AUTO_NAME_MAX_CHARS = 80


def auto_name_chat(
    conn: sqlite3.Connection,
    chat_id: str,
    title: str,
) -> bool:
    """Server-side rename of an Untitled chat WITHOUT bumping ``version``.

    Pitfall 6: an immediate client-side follow-up POST uses the version the
    client last observed; if auto-name bumped ``version`` mid-flight the
    follow-up would 409. The idempotency guard is therefore
    ``WHERE title = 'Untitled chat'`` instead of a version CAS — calling this
    helper twice, or after a user-initiated rename, is a safe no-op.

    Args:
        title: Raw title hint from the routing plan or LLM fallback. Stripped
            and truncated to 80 chars (D-22 hard cap; D-25 target is 60).

    Returns:
        ``True`` if the row was updated, ``False`` if the chat already had a
        non-default title (idempotent no-op).
    """
    truncated = title.strip()[:_AUTO_NAME_MAX_CHARS]
    cur = conn.execute(
        "UPDATE chats SET title = ?, updated_at = CURRENT_TIMESTAMP "
        "WHERE id = ? AND title = 'Untitled chat'",
        (truncated, chat_id),
    )
    conn.commit()
    return cur.rowcount == 1
