"""SQLite helpers for the arXiv monitor.

Each function opens — or accepts — its own connection so callers in different
threads can never share a connection (sqlite3 connections are not thread-safe).

Dedup (``is_ingested`` / ``mark_ingested``) moved to ``src.ingest.sources``.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import Any


def create_run(conn: sqlite3.Connection, started_at: datetime) -> int:
    """Insert a new ``poll_runs`` row with status='running'. Return its id."""
    cur = conn.execute(
        "INSERT INTO poll_runs (started_at, status) VALUES (?, 'running')",
        (int(started_at.timestamp()),),
    )
    conn.commit()
    return int(cur.lastrowid)  # type: ignore[arg-type]


def update_run(
    conn: sqlite3.Connection,
    run_id: int,
    *,
    completed_at: datetime | None = None,
    status: str | None = None,
    papers_found: int | None = None,
    papers_ingested: int | None = None,
    papers_skipped: int | None = None,
    error_detail: str | None = None,
) -> None:
    """Update a ``poll_runs`` row. Only non-None fields are written."""
    col_map: dict[str, int | str | None] = {}
    if completed_at is not None:
        col_map["completed_at"] = int(completed_at.timestamp())
    if status is not None:
        col_map["status"] = status
    if papers_found is not None:
        col_map["papers_found"] = papers_found
    if papers_ingested is not None:
        col_map["papers_ingested"] = papers_ingested
    if papers_skipped is not None:
        col_map["papers_skipped"] = papers_skipped
    if error_detail is not None:
        col_map["error_detail"] = error_detail
    if not col_map:
        return
    set_clause = ", ".join(f"{k} = ?" for k in col_map)
    values: list[int | str | None] = list(col_map.values()) + [run_id]
    conn.execute(f"UPDATE poll_runs SET {set_clause} WHERE id = ?", values)  # noqa: S608
    conn.commit()


def get_last_run(conn: sqlite3.Connection) -> dict[str, Any] | None:
    """Return the most-recent ``poll_runs`` row as a dict, or None."""
    row = conn.execute(
        "SELECT id, started_at, completed_at, status,"
        " papers_found, papers_ingested, papers_skipped, error_detail"
        " FROM poll_runs ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if row is None:
        return None
    return {
        "run_id": row[0],
        "started_at": float(row[1]),
        "completed_at": float(row[2]) if row[2] is not None else None,
        "status": row[3],
        "papers_found": row[4],
        "papers_ingested": row[5],
        "papers_skipped": row[6],
        "error_detail": row[7],
    }
