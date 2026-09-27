# long-ok-file
"""Unified sources registry — single dedup + lifecycle table in parents.sqlite.

Replaces the two legacy tables (``ingested_files``, ``ingested_sources``) with
one ``sources`` table keyed on a canonical ID. Tracks lifecycle status
(``fetched`` → ``parsed`` → ``contextualized`` → ``ingested``) so spider,
parser, contextualizer and ingest all write to the same row.

Schema lives here; ``_init_parents_db`` in ``src.ingest.storage`` calls
``init_schema`` at db-open time so creation is idempotent.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from src.fetch.classify import canonical_url

_VALID_STATUSES = {"fetched", "parsed", "contextualized", "ingested"}


def canonical_id(meta: dict[str, Any]) -> tuple[str, str]:
    """Return ``(id, id_type)`` for a source meta dict.

    Priority: arxiv > doi > url > hash(title/first-author). Hash fallback is
    16 hex chars — enough entropy for dedup, short enough to read.

    Enrichment writes arxiv_id/doi under ``meta["metadata"]``; legacy meta
    dicts stored them at the top level. Read nested first, fall back to top.
    """
    nested_raw = meta.get("metadata")
    nested: dict[str, Any] = nested_raw if isinstance(nested_raw, dict) else {}
    arxiv = nested.get("arxiv_id") or meta.get("arxiv_id")
    if arxiv:
        return str(arxiv), "arxiv"
    doi = nested.get("doi") or meta.get("doi")
    if doi:
        return str(doi).lower(), "doi"
    if meta.get("url"):
        return canonical_url({"url": meta["url"]}), "url"
    authors = meta.get("authors") or [""]
    first = authors[0] if authors else ""
    key = f"{meta.get('title','')}/{first}"
    return hashlib.sha256(key.encode()).hexdigest()[:16], "hash"


def init_schema(conn: sqlite3.Connection) -> None:
    """Create the ``sources`` table and its indexes if missing."""
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS sources (
            id           TEXT PRIMARY KEY,
            id_type      TEXT NOT NULL,
            title        TEXT,
            authors      TEXT,
            year         INTEGER,
            url          TEXT,
            doi          TEXT,
            arxiv_id     TEXT,
            source_type  TEXT,
            notebook     TEXT,
            status       TEXT NOT NULL,
            file_hash    TEXT,
            updated_at   INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_sources_status    ON sources(status);
        CREATE INDEX IF NOT EXISTS idx_sources_notebook  ON sources(notebook);
        CREATE INDEX IF NOT EXISTS idx_sources_arxiv_id  ON sources(arxiv_id);
        CREATE INDEX IF NOT EXISTS idx_sources_file_hash ON sources(file_hash);
        """
    )
    conn.commit()


def _now() -> int:
    return int(datetime.now(UTC).timestamp())


def _serialize_authors(authors: Sequence[str] | None) -> str | None:
    if authors is None:
        return None
    return json.dumps(list(authors), ensure_ascii=False)


def upsert(
    conn: sqlite3.Connection,
    *,
    canonical_id: str,
    id_type: str,
    status: str,
    title: str | None,
    authors: Sequence[str] | None,
    year: int | None,
    url: str | None,
    doi: str | None,
    arxiv_id: str | None,
    source_type: str | None,
    notebook: str | None,
    file_hash: str | None,
) -> None:
    """Insert or update a sources row. Status must be valid; never regresses."""
    if status not in _VALID_STATUSES:
        raise ValueError(f"invalid status: {status!r}")
    conn.execute(
        """
        INSERT INTO sources
            (id, id_type, title, authors, year, url, doi, arxiv_id,
             source_type, notebook, status, file_hash, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            id_type     = excluded.id_type,
            title       = COALESCE(excluded.title, sources.title),
            authors     = COALESCE(excluded.authors, sources.authors),
            year        = COALESCE(excluded.year, sources.year),
            url         = COALESCE(excluded.url, sources.url),
            doi         = COALESCE(excluded.doi, sources.doi),
            arxiv_id    = COALESCE(excluded.arxiv_id, sources.arxiv_id),
            source_type = COALESCE(excluded.source_type, sources.source_type),
            notebook    = COALESCE(excluded.notebook, sources.notebook),
            status      = excluded.status,
            file_hash   = COALESCE(excluded.file_hash, sources.file_hash),
            updated_at  = excluded.updated_at
        """,
        (
            canonical_id, id_type, title, _serialize_authors(authors), year,
            url, doi, arxiv_id, source_type, notebook, status, file_hash, _now(),
        ),
    )
    conn.commit()


def upsert_fetched(
    conn: sqlite3.Connection,
    meta: dict[str, Any],
    notebook: str | None,
    source_type: str | None,
) -> str:
    """Write a row at the ``fetched`` stage from a spider meta dict.

    Returns the canonical_id so callers can keep using it for subsequent
    advance_status calls. Idempotent — never regresses status.
    """
    cid, ctype = canonical_id(meta)
    existing = conn.execute(
        "SELECT status FROM sources WHERE id = ?", (cid,)
    ).fetchone()
    new_status = "fetched" if existing is None else existing[0]
    upsert(
        conn,
        canonical_id=cid, id_type=ctype, status=new_status,
        title=meta.get("title"), authors=meta.get("authors"),
        year=meta.get("year"), url=meta.get("url"), doi=meta.get("doi"),
        arxiv_id=meta.get("arxiv_id"), source_type=source_type,
        notebook=notebook, file_hash=None,
    )
    return cid


def advance_status(
    conn: sqlite3.Connection, canonical_id: str, status: str,
) -> None:
    """Move a source forward in the lifecycle. No-op if row doesn't exist.

    The status column is overwritten, but ``updated_at`` always advances.
    Earlier-stage callers calling this after a later-stage write would
    regress the row — that's OK by design (re-runs are valid).
    """
    if status not in _VALID_STATUSES:
        raise ValueError(f"invalid status: {status!r}")
    conn.execute(
        "UPDATE sources SET status = ?, updated_at = ? WHERE id = ?",
        (status, _now(), canonical_id),
    )
    conn.commit()


def mark_ingested_with_hash(
    conn: sqlite3.Connection,
    canonical_id: str,
    file_hash: str,
    notebook: str | None,
) -> None:
    """Mark a source as ingested and record its content hash.

    Upserts the row even when the caller never went through the
    fetched/parsed stages (eg. files dropped into ``sources/`` manually).
    """
    existing = conn.execute(
        "SELECT 1 FROM sources WHERE id = ?", (canonical_id,)
    ).fetchone()
    if existing is None:
        upsert(
            conn,
            canonical_id=canonical_id, id_type="hash", status="ingested",
            title=None, authors=None, year=None, url=None, doi=None,
            arxiv_id=None, source_type=None, notebook=notebook,
            file_hash=file_hash,
        )
        return
    conn.execute(
        "UPDATE sources SET status='ingested', file_hash = ?, notebook = COALESCE(notebook, ?),"
        " updated_at = ? WHERE id = ?",
        (file_hash, notebook, _now(), canonical_id),
    )
    conn.commit()


def has_been_ingested_by_hash(
    conn: sqlite3.Connection, file_hash: str, notebook: str | None,
) -> bool:
    """True if this content hash is already ingested under ``notebook``.

    Dedup is scoped to (file_hash, notebook) because child rows carry the
    notebook as their partition key: a file ingested under notebook A produces
    no rows for notebook B, so skipping the B ingest on a hash-only match would
    leave the file unretrievable under B. The same-notebook re-run is still
    skipped (idempotency). When ``notebook`` is None the gate matches on hash
    alone (legacy callers that don't partition).
    """
    if notebook is None:
        row = conn.execute(
            "SELECT 1 FROM sources WHERE file_hash = ? AND status = 'ingested'",
            (file_hash,),
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT 1 FROM sources"
            " WHERE file_hash = ? AND notebook = ? AND status = 'ingested'",
            (file_hash, notebook),
        ).fetchone()
    return row is not None


def is_ingested(conn: sqlite3.Connection, canonical_id: str) -> bool:
    """True only if the row exists AND status='ingested' — replaces the
    arxiv-monitor ``is_ingested(arxiv_id)`` check."""
    row = conn.execute(
        "SELECT 1 FROM sources WHERE id = ? AND status = 'ingested'",
        (canonical_id,),
    ).fetchone()
    return row is not None


def exists(conn: sqlite3.Connection, canonical_id: str) -> bool:
    """True if any row exists for this canonical_id, regardless of status."""
    row = conn.execute(
        "SELECT 1 FROM sources WHERE id = ?", (canonical_id,),
    ).fetchone()
    return row is not None
