# long-ok-file
"""Persistence layer for the ingest pipeline: parents.sqlite + Milvus insert.

Two stores, two responsibilities:

  * **parents.sqlite** — owns ``ParentChunk`` rows (looked up by ``id`` at
    query time after rerank). Lifecycle + dedup state lives in the unified
    ``sources`` table (see ``src.ingest.sources``); ``_file_hash`` here is
    salted with embed model+dim so a model swap (eg. Gemini → Qwen3)
    invalidates the cache and re-ingests everything.

  * **Milvus** — owns ``ChildChunk`` rows with their dense embeddings. The
    sparse_embedding is populated server-side via the BM25 built-in function
    over the ``text`` field (we don't pass sparse vectors at insert time).

UTF-8 truncation (``_truncate_utf8``) enforces the schema's byte-counted
``VARCHAR(4096)`` cap on child text. Parent text is uncapped because parents
live in sqlite, which has no such limit.
"""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from src import config
from src.milvus_client import get_client
from src.models import ChildChunk, ParentChunk

# ---------------------------------------------------------------------------
# parents.sqlite
# ---------------------------------------------------------------------------


def _open_parents_db() -> sqlite3.Connection:
    """Open (and initialise if needed) the parents SQLite database."""
    db_path = Path(config.PARENTS_DB)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    _init_parents_db(conn)
    return conn


def _init_parents_db(conn: sqlite3.Connection) -> None:
    """Create tables if they do not exist. Drops legacy dedup tables on first
    open post-migration — the unified ``sources`` table absorbs both."""
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS parents (
            id         TEXT PRIMARY KEY,
            text       TEXT NOT NULL,
            source_file TEXT NOT NULL,
            notebook   TEXT NOT NULL,
            modality   TEXT NOT NULL,
            page_number INTEGER NOT NULL,
            image_path  TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_parents_notebook
            ON parents(notebook);

        CREATE TABLE IF NOT EXISTS poll_runs (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at      INTEGER NOT NULL,
            completed_at    INTEGER,
            status          TEXT NOT NULL DEFAULT 'running',
            papers_found    INTEGER NOT NULL DEFAULT 0,
            papers_ingested INTEGER NOT NULL DEFAULT 0,
            papers_skipped  INTEGER NOT NULL DEFAULT 0,
            error_detail    TEXT
        );

        DROP TABLE IF EXISTS ingested_files;
        DROP TABLE IF EXISTS ingested_sources;
        """
    )
    from src.ingest.sources import init_schema  # noqa: PLC0415
    init_schema(conn)
    conn.commit()


def _file_hash(path: Path) -> str:
    """Return sha256 of file content salted with embed model+dim.

    Model changes invalidate the cache.
    """
    salt = f"{config.EMBED_MODEL}:{config.EMBED_DIM}".encode()
    sha = hashlib.sha256(salt)
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            sha.update(block)
    return sha.hexdigest()


def _store_parent(conn: sqlite3.Connection, parent: ParentChunk) -> None:
    """Persist a ParentChunk to SQLite (upsert)."""
    conn.execute(
        "INSERT OR REPLACE INTO parents"
        " (id, text, source_file, notebook, modality, page_number, image_path)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            parent.id,
            parent.text,
            parent.source_file,
            parent.notebook,
            parent.modality,
            parent.page_number,
            parent.image_path,
        ),
    )


# ---------------------------------------------------------------------------
# Milvus insert
# ---------------------------------------------------------------------------


def _truncate_utf8(text: str, max_bytes: int) -> str:
    """Truncate text to <= max_bytes UTF-8 bytes (Milvus VARCHAR max_length is byte-counted)."""
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode("utf-8", errors="ignore")


# Milvus query(ids=…) caps the id list per call; batch larger inputs.
_QUERY_ID_BATCH = 4096


def _insert_children(
    collection: str,
    children: list[ChildChunk],
    vectors: list[list[float]],
) -> None:
    """Upsert child chunks + their dense embeddings into Milvus.

    Upsert (not insert) makes re-ingest idempotent: ``child.id`` is a
    deterministic content hash, so Milvus' delete-by-pk-then-insert overwrites
    the row in place instead of appending a duplicate. sparse_embedding is
    deliberately omitted — Milvus populates it server-side via the BM25
    built-in function over the ``text`` field.
    """
    if not children:
        return

    client = get_client()
    now = int(datetime.now(UTC).timestamp())

    rows = [
        {
            "id": child.id,
            "dense_embedding": vector,
            "text": _truncate_utf8(child.text, 4096),
            "parent_chunk_id": child.parent_id,
            "source_file": child.source_file,
            "notebook": child.notebook,
            "modality": child.modality,
            "page_number": child.page_number,
            "created_at": now,
        }
        for child, vector in zip(children, vectors)
    ]
    result = client.upsert(collection_name=collection, data=rows)
    upsert_count: int = result.get("upsert_count", -1)
    if upsert_count != len(rows):
        raise RuntimeError(
            f"Milvus upsert_count mismatch: sent {len(rows)} rows,"
            f" acknowledged {upsert_count}. Full response: {result}"
        )


def existing_child_ids(collection: str, ids: list[str]) -> set[str]:
    """Return the subset of ``ids`` already present in ``collection``.

    Drives the pre-embed existence gate (idempotent cost). On first ingest the
    collection doesn't exist yet; query raises, and an empty set means "embed
    everything". A query failure degrades to re-embedding (cost, not data loss —
    upsert collapses on the deterministic id), never a hard stop on the run.

    A missing collection (legitimately empty) returns an empty set up-front. A
    transient mid-stream batch failure returns the hits gathered so far rather
    than discarding them — losing confirmed presence forces a full re-embed of
    an already-ingested >4096-child doc.
    """
    if not ids:
        return set()

    client = get_client()
    if not client.has_collection(collection):
        return set()

    present: set[str] = set()
    for start in range(0, len(ids), _QUERY_ID_BATCH):
        batch = ids[start : start + _QUERY_ID_BATCH]
        try:
            rows = client.query(
                collection_name=collection,
                ids=batch,
                output_fields=["id"],
            )
        except Exception:  # noqa: BLE001 — transient query failure ⇒ keep confirmed hits
            return present
        present.update(str(row["id"]) for row in rows)
    return present


def delete_source_chunks(
    collection: str,
    source_file: str,
    db_conn: sqlite3.Connection,
    *,
    notebook: str,
) -> int:
    """Delete a source's children (Milvus) + parents (sqlite) before re-ingest.
    Removing the abstract stub chunks first keeps retrieval from mixing both
    representations of the same paper. Returns the number of parent rows deleted.
    """
    if '"' in source_file:
        raise ValueError(f"source_file with embedded quote is unsupported: {source_file!r}")
    get_client().delete(
        collection_name=collection,
        filter=f'source_file == "{source_file}"',
        partition_name=notebook,
    )
    cur = db_conn.execute("DELETE FROM parents WHERE source_file = ?", (source_file,))
    db_conn.commit()
    return cur.rowcount
