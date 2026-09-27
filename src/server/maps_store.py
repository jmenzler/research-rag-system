"""SQLite CRUD for saved Maps (graph-page entities, not chat-bound).

Maps are D-09 entities whose lifecycle is independent of any chat.  Schema is
applied by ``0005_maps.sql`` at server startup via the migration runner.

Every SQL statement uses ``?`` placeholders only (T-05-05: SQL injection prevention).
Snapshot size is validated before INSERT (T-05-06: DoS guard).
"""
from __future__ import annotations

import logging
import sqlite3
import uuid
from pathlib import Path
from typing import Any, cast

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHATS_DB_PATH = PROJECT_ROOT / "parents" / "chats.db"

MAX_SNAPSHOT_BYTES = 5 * 1024 * 1024


def open_maps_conn(db_path: Path | None = None) -> sqlite3.Connection:
    """Open a connection to chats.db for maps operations."""
    path = db_path if db_path is not None else CHATS_DB_PATH
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def create_map(
    conn: sqlite3.Connection,
    *,
    name: str,
    collection: str,
    snapshot: str,
) -> str:
    """Insert a new map row and return its id (uuid4 hex).

    Raises:
        ValueError: If snapshot exceeds MAX_SNAPSHOT_BYTES (T-05-06 DoS guard).
    """
    if len(snapshot.encode("utf-8")) > MAX_SNAPSHOT_BYTES:
        raise ValueError(
            f"snapshot too large: {len(snapshot.encode('utf-8'))} bytes "
            f"exceeds limit of {MAX_SNAPSHOT_BYTES} bytes"
        )
    map_id = uuid.uuid4().hex
    conn.execute(
        "INSERT INTO maps (id, name, collection, snapshot) VALUES (?, ?, ?, ?)",
        (map_id, name, collection, snapshot),
    )
    conn.commit()
    return map_id


def get_map(conn: sqlite3.Connection, map_id: str) -> sqlite3.Row | None:
    """Return the map row for ``map_id``, or None if not found."""
    row = conn.execute(
        "SELECT id, name, collection, snapshot, created_at, updated_at "
        "FROM maps WHERE id = ?",
        (map_id,),
    ).fetchone()
    return cast("sqlite3.Row | None", row)


def list_maps(
    conn: sqlite3.Connection,
    collection: str | None = None,
) -> list[sqlite3.Row]:
    """Return maps ordered by most-recently-updated first.

    Args:
        collection: When provided, filter to only rows with this collection value.
    """
    if collection is not None:
        rows = conn.execute(
            "SELECT id, name, collection, snapshot, created_at, updated_at "
            "FROM maps WHERE collection = ? ORDER BY updated_at DESC",
            (collection,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, name, collection, snapshot, created_at, updated_at "
            "FROM maps ORDER BY updated_at DESC",
        ).fetchall()
    return list(rows)


def delete_map(conn: sqlite3.Connection, map_id: str) -> bool:
    """Delete the map row. Returns True if a row was removed, False otherwise."""
    cur = conn.execute("DELETE FROM maps WHERE id = ?", (map_id,))
    conn.commit()
    return cur.rowcount > 0


def compute_coverage(
    corpus_ids: list[int],
    in_corpus_map: dict[int, Any],
) -> dict[str, Any]:
    """Pure function computing D-10 scorecard for a saved map's node set.

    Args:
        corpus_ids: List of node corpus IDs from the saved snapshot.
        in_corpus_map: Mapping of corpus_id to either a bool or a dict with an
            ``in_corpus`` key (produced by the graph resolver).

    Returns:
        ``{"in_corpus": N, "total": M, "missing_ids": [...]}`` where ``in_corpus``
        counts nodes confirmed present in the corpus and ``missing_ids`` lists those
        that are not.
    """
    missing: list[int] = []
    for cid in corpus_ids:
        val = in_corpus_map.get(cid, False)
        if isinstance(val, dict):
            present = bool(val.get("in_corpus", False))
        else:
            present = bool(val)
        if not present:
            missing.append(cid)
    return {
        "in_corpus": len(corpus_ids) - len(missing),
        "total": len(corpus_ids),
        "missing_ids": missing,
    }
