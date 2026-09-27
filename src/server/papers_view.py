"""In-memory TTL read-view over the 3-substrate corpus join (CORPUS-04).

Joins:
  1. parents.sqlite.sources   — lifecycle + canonical_id per paper
  2. sources/<coll>/<slug>/meta.json  — Tier-1 metadata sidecars
  3. Milvus chunk counts      — single batched pass per collection (D-10)

No materialised papers table is created; the join is computed on read and cached
by a TTL + watermark pattern. The cache is coherent because the server is
single-worker (D-08).

Public API (imported by api_papers.py):
  - build_read_view           — full join, sorted by ingested_at DESC
  - get_read_view             — TTL + watermark-cached wrapper
  - rebuild_corpus_fts        — contentless FTS5 DELETE+INSERT; bump watermark
  - search_corpus_fts         — FTS5 MATCH -> list[str] paper_ids
  - apply_facets              — filter rows by collection/year/arxiv/doi/ingested
  - encode_cursor / decode_cursor   — base64url (sort_value, paper_id) pagination
"""
from __future__ import annotations

import base64
import json
import logging
import socket
import sqlite3
import time
from collections import Counter
from pathlib import Path
from typing import Any, TypedDict

from src.config import paths as config_paths
from src.ingest.sources import canonical_id
from src.milvus_client import _MILVUS_LOCK, get_client

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CORPUS_VIEW_TTL_S = 60  # D-09 discretion — re-check watermark after 60s
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class PaperRow(TypedDict, total=False):
    """Single paper in the corpus read-view.

    All metadata fields may be ``None`` under substrate-scarcity (no meta.json,
    Milvus down, sources row missing). Never crashes — ``None`` is the contract
    for missing data.
    """

    paper_id: str
    title: str | None
    authors: str | None  # JSON-encoded list or None
    year: int | None
    collection: str
    chunk_count: int | None
    arxiv_id: str | None
    doi: str | None
    short_cite: str | None
    ingested_at: int | None  # sources.updated_at epoch
    extraction_quality: str | None  # from meta.json metadata
    fetch_source: str | None  # from meta.json metadata
    parser_version: str | None  # from meta.json metadata


# ---------------------------------------------------------------------------
# Cache state (keyed by parents_db+sources_root so concurrent test runs with
# different params don't collide)
# ---------------------------------------------------------------------------

_cache_rows: list[PaperRow] | None = None
_cache_watermark: int = 0
_cache_ts: float = 0.0
_cache_key: str | None = None


# ---------------------------------------------------------------------------
# Helpers — source-file resolution (mirrors _resolve_doc_representative)
# ---------------------------------------------------------------------------


def _resolve_source_file(doc_dir: Path) -> Path | None:
    """Resolve the representative source file for a doc-dir.

    Mirrors ``src.ingest.ingest._resolve_doc_representative`` logic so the
    computed ``source_file`` matches what was written to parents + Milvus at
    ingest time.

    Returns ``None`` if no ingestable file exists (counts as chunk_count=0).
    """
    for fname in ("source.pdf", "pdf.txt", "nlm.txt", "web.txt"):
        cand = doc_dir / fname
        if cand.is_file():
            return cand
    return None


def _load_meta(doc_dir: Path) -> dict[str, Any] | None:  # noqa: ANN401
    """Read and parse ``meta.json`` from a doc-dir.

    Returns ``None`` when the file doesn't exist or is malformed.
    """
    meta_path = doc_dir / "meta.json"
    if not meta_path.is_file():
        return None
    try:
        return json.loads(meta_path.read_text())  # type: ignore[no-any-return]
    except (json.JSONDecodeError, OSError):
        logger.warning("malformed meta.json at %s", meta_path)
        return None


def _get_meta_value(
    meta: dict[str, Any], key: str, nested_key: str | None = None,
) -> Any:  # noqa: ANN401
    """Read a value from meta, optionally checking ``meta["metadata"]`` first."""
    if nested_key:
        nested = meta.get("metadata")
        if isinstance(nested, dict) and nested_key in nested:
            return nested[nested_key]
    return meta.get(key)


# ---------------------------------------------------------------------------
# Doc-dir iteration
# ---------------------------------------------------------------------------


def _iter_doc_dirs(sources_root: Path | None = None) -> list[tuple[Path, str]]:
    """Yield ``(doc_dir, collection)`` for every non-quarantine doc-dir.

    Skips top-level source dirs starting with ``_`` (quarantine buckets).
    """
    root = sources_root if sources_root is not None else config_paths.SOURCES_DIR
    if not root.is_dir():
        return []
    results: list[tuple[Path, str]] = []
    for coll_dir in sorted(root.iterdir()):
        if not coll_dir.is_dir() or coll_dir.name.startswith("_"):
            continue
        collection = coll_dir.name
        for doc_dir in sorted(coll_dir.iterdir()):
            if doc_dir.is_dir():
                results.append((doc_dir, collection))
    return results


# ---------------------------------------------------------------------------
# Milvus single-pass chunk-count tally (D-10 / Pattern 5)
# ---------------------------------------------------------------------------


def _milvus_reachable() -> bool:
    """Quick socket-level reachability check for the Milvus URI host:port.

    Avoids hanging on the SDK's long default timeout when no Milvus is running.
    """
    uri = "http://localhost:19530"  # matches src/config/env.py default
    # Parse host:port from the URI
    host = "localhost"
    port = 19530
    if "://" in uri:
        netloc = uri.split("://", 1)[1].split("/")[0]
    else:
        netloc = uri
    if ":" in netloc:
        host, port_str = netloc.rsplit(":", 1)
        try:
            port = int(port_str)
        except ValueError:
            pass
    try:
        sock = socket.create_connection((host, port), timeout=2.0)
        sock.close()
        return True
    except OSError:
        return False


def _tally_chunk_counts() -> dict[str, int]:
    """Return {source_file: chunk_count} via one batched query per collection.

    Degrades to empty dict (all counts None) when Milvus is unreachable.
    One query_iterator per collection (4 max), NEVER N+1.
    """
    if not _milvus_reachable():
        logger.warning("Milvus unreachable; chunk_count=None for all rows")
        return {}

    counter: Counter[str] = Counter()
    client = get_client()

    with _MILVUS_LOCK:
        for coll in config_paths.COLLECTIONS:
            try:
                if not client.has_collection(coll):
                    continue
                iterator = client.query_iterator(
                    collection_name=coll,
                    batch_size=1000,
                    output_fields=["source_file"],
                )
                while True:
                    batch = iterator.next()
                    if not batch:
                        break
                    for row in batch:
                        sf = row.get("source_file")
                        if sf is not None:
                            counter[str(sf)] += 1
            except Exception:
                logger.warning(
                    "Milvus query_iterator failed for collection '%s'; skipping", coll,
                )
                continue
    return dict(counter)


# ---------------------------------------------------------------------------
# Core read-view builders
# ---------------------------------------------------------------------------


def build_read_view(
    parents_db: Path | None = None,
    sources_root: Path | None = None,
) -> list[PaperRow]:
    """Full three-substrate join: doc-dirs + meta.json + sources + Milvus counts.

    Args:
        parents_db: Path to parents.sqlite (default: project parents dir).
        sources_root: Path to sources/ directory (default: config SOURCES_DIR).

    Returns:
        Rows sorted by ``ingested_at`` DESC (most recently added first).
    """
    if parents_db is None:
        parents_db = PROJECT_ROOT / "parents" / "parents.sqlite"

    # 1. Load sources lookup by canonical_id
    sources_by_id: dict[str, dict[str, Any]] = {}
    try:
        conn = sqlite3.connect(str(parents_db))
        conn.row_factory = sqlite3.Row
        for row in conn.execute("SELECT * FROM sources"):
            sources_by_id[row["id"]] = dict(row)
        conn.close()
    except (sqlite3.OperationalError, sqlite3.DatabaseError):
        logger.warning("parents.sqlite not available at %s", parents_db)

    # 2. Walk doc-dirs, read meta.json, compute canonical_id, join to sources
    rows_map: dict[str, PaperRow] = {}  # paper_id -> row

    for doc_dir, collection in _iter_doc_dirs(sources_root=sources_root):
        meta = _load_meta(doc_dir)

        # Derive paper_id from canonical_id(meta) when meta exists; else use slug
        if meta is not None:
            cid, _ = canonical_id(meta)
            paper_id = cid
        else:
            paper_id = doc_dir.name

        src = sources_by_id.get(paper_id)

        title: str | None = None
        if meta is not None:
            title = meta.get("title")
        if title is None and src is not None:
            title = src.get("title")

        authors: str | None = None
        if meta is not None:
            raw = meta.get("authors")
            if raw is not None:
                authors = json.dumps(list(raw) if isinstance(raw, list) else [raw])

        year: int | None = None
        if meta is not None:
            year = meta.get("year")

        arxiv_id: str | None = None
        if meta is not None:
            arxiv_id = _get_meta_value(meta, "arxiv_id", "arxiv_id")

        doi: str | None = None
        if meta is not None:
            doi = meta.get("doi")

        short_cite: str | None = None
        if meta is not None:
            short_cite = _get_meta_value(meta, "short_cite", "short_cite")

        extraction_quality: str | None = None
        if meta is not None:
            extraction_quality = _get_meta_value(meta, "extraction_quality", "extraction_quality")

        fetch_source: str | None = None
        if meta is not None:
            fetch_source = _get_meta_value(meta, "fetch_source", "fetch_source")

        parser_version: str | None = None
        if meta is not None:
            parser_version = _get_meta_value(meta, "parser_version", "parser_version")

        ingested_at: int | None = src.get("updated_at") if src is not None else None

        rows_map[paper_id] = PaperRow(
            paper_id=paper_id,
            title=title,
            authors=authors,
            year=year,
            collection=collection,
            chunk_count=None,  # filled below
            arxiv_id=arxiv_id,
            doi=doi,
            short_cite=short_cite,
            ingested_at=ingested_at,
            extraction_quality=extraction_quality,
            fetch_source=fetch_source,
            parser_version=parser_version,
        )

    # 3. Milvus chunk count tally — single pass per collection
    chunk_counts = _tally_chunk_counts()

    # Map source_file -> paper_id by resolving each doc-dir
    source_to_paper: dict[str, str] = {}
    for doc_dir, _collection in _iter_doc_dirs(sources_root=sources_root):
        sf = _resolve_source_file(doc_dir)
        if sf is not None:
            # Try both the resolved path and the slug-derived paper_id
            meta = _load_meta(doc_dir)
            if meta is not None:
                cid, _ = canonical_id(meta)
            else:
                cid = doc_dir.name
            source_to_paper[str(sf)] = cid

    # Apply chunk counts to rows
    for paper_id, row in rows_map.items():
        # Find source_file for this paper and look up its chunk count
        for src_path, pid in source_to_paper.items():
            if pid == paper_id:
                cnt = chunk_counts.get(src_path)
                if cnt is not None:
                    row["chunk_count"] = cnt
                break

    # 4. Sort by ingested_at DESC; papers with no timestamp sort last
    rows = list(rows_map.values())
    rows.sort(key=lambda r: r.get("ingested_at") or 0, reverse=True)
    return rows


def get_read_view(
    parents_db: Path | None = None, sources_root: Path | None = None
) -> list[PaperRow]:
    """Return the cached read-view, rebuilding when TTL or watermark changes.

    Simple module-level cache, key-checked so parallel test runs with different
    fixtures invalidate. Coherent for a single-worker server (D-08).
    """
    global _cache_rows, _cache_watermark, _cache_ts, _cache_key

    current_key = f"{parents_db or ''}|{sources_root or ''}"
    now = time.monotonic()
    cache_miss = (
        _cache_rows is None
        or _cache_key != current_key
        or (now - _cache_ts > CORPUS_VIEW_TTL_S)
    )

    if cache_miss:
        _cache_rows = build_read_view(parents_db=parents_db, sources_root=sources_root)
        _cache_watermark = _read_max_updated_at(parents_db)
        _cache_ts = now
        _cache_key = current_key
    else:
        # Sub-TTL: check watermark for early bust
        current_watermark = _read_max_updated_at(parents_db)
        if current_watermark > _cache_watermark:
            _cache_rows = build_read_view(
                parents_db=parents_db, sources_root=sources_root
            )
            _cache_watermark = current_watermark
            _cache_ts = now
            _cache_key = current_key

    return _cache_rows  # type: ignore[return-value]


def _read_max_updated_at(parents_db: Path | None = None) -> int:
    """Return max(sources.updated_at) or 0 if unreachable."""
    if parents_db is None:
        parents_db = PROJECT_ROOT / "parents" / "parents.sqlite"
    try:
        conn = sqlite3.connect(str(parents_db))
        val: int = conn.execute(
            "SELECT COALESCE(MAX(updated_at), 0) FROM sources"
        ).fetchone()[0]
        conn.close()
        return val
    except (sqlite3.OperationalError, sqlite3.DatabaseError):
        return 0


# ---------------------------------------------------------------------------
# FTS5 rebuild and search
# ---------------------------------------------------------------------------


def rebuild_corpus_fts(
    chats_db: str | Path,
    parents_db: Path | None = None,
    sources_root: Path | None = None,
) -> None:
    """Rebuild the contentless corpus_fts table from the current read-view.

    DELETE then INSERT all rows; bump ``corpus_fts_meta.watermark`` to the
    current max(sources.updated_at). Only rebuilds when the watermark lags.
    """
    conn = sqlite3.connect(str(chats_db))

    # Read current watermark
    current_watermark = conn.execute(
        "SELECT COALESCE(MAX(watermark), 0) FROM corpus_fts_meta"
    ).fetchone()[0]

    # Check if sources have changed
    sources_watermark = _read_max_updated_at(parents_db)

    if current_watermark >= sources_watermark:
        conn.close()
        return

    # Build the read-view
    rows = build_read_view(parents_db=parents_db, sources_root=sources_root)

    conn.execute("BEGIN")
    try:
        conn.execute("DELETE FROM corpus_fts")

        for row in rows:
            conn.execute(
                "INSERT INTO corpus_fts (paper_id, title, authors, year,"
                " short_cite, arxiv_id, doi) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    row.get("paper_id", ""),
                    row.get("title") or "",
                    row.get("authors") or "",
                    str(row.get("year") or ""),
                    row.get("short_cite") or "",
                    row.get("arxiv_id") or "",
                    row.get("doi") or "",
                ),
            )

        conn.execute(
            "UPDATE corpus_fts_meta SET watermark = ? WHERE id = 1",
            (sources_watermark,),
        )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()


def search_corpus_fts(q: str, chats_db: str | Path) -> list[str]:
    """Search corpus_fts and return matching ``paper_id`` values.

    The query is wrapped in double quotes to prevent FTS5 syntax errors from
    special characters (``.``, ``*``, ``^``, etc.). Malformed queries that still
    error (e.g., unmatched quotes embedded in the raw input) return ``[]`` instead
    of crashing.

    The caller is responsible for calling ``rebuild_corpus_fts`` before this
    if the FTS needs to be refreshed.
    """
    conn = sqlite3.connect(str(chats_db))
    try:
        # Wrap in double quotes to avoid FTS5 query syntax special characters.
        # A quoted query is a phrase match across tokens from the tokenizer.
        results = conn.execute(
            "SELECT paper_id FROM corpus_fts WHERE corpus_fts MATCH ?",
            (f'"{q}"',),
        ).fetchall()
        return [r[0] for r in results]
    except sqlite3.OperationalError:
        # Malformed FTS5 query (syntax error, unmatched quotes, etc.)
        return []
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Facet helpers
# ---------------------------------------------------------------------------


def apply_facets(
    rows: list[PaperRow],
    *,
    collections: list[str] | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
    ingested_bucket: str | None = None,
    has_arxiv: bool | None = None,
    has_doi: bool | None = None,
) -> list[PaperRow]:
    """Filter the row list by facet criteria. All facets are AND-ed."""
    result = rows

    if collections:
        coll_set = set(collections)
        result = [r for r in result if r.get("collection") in coll_set]

    if year_from is not None:
        result = [r for r in result if (r.get("year") or 0) >= year_from]

    if year_to is not None:
        result = [r for r in result if (r.get("year") or 0) <= year_to]

    if ingested_bucket is not None:
        now = int(time.time())
        if ingested_bucket == "7d":
            cutoff = now - 7 * 86400
            result = [r for r in result if (r.get("ingested_at") or 0) >= cutoff]
        elif ingested_bucket == "30d":
            cutoff = now - 30 * 86400
            result = [r for r in result if (r.get("ingested_at") or 0) >= cutoff]
        elif ingested_bucket == "1y":
            cutoff = now - 365 * 86400
            result = [r for r in result if (r.get("ingested_at") or 0) >= cutoff]

    if has_arxiv is True:
        result = [r for r in result if r.get("arxiv_id")]
    elif has_arxiv is False:
        result = [r for r in result if not r.get("arxiv_id")]

    if has_doi is True:
        result = [r for r in result if r.get("doi")]
    elif has_doi is False:
        result = [r for r in result if not r.get("doi")]

    return result


# ---------------------------------------------------------------------------
# Cursor pagination (Pattern 4)
# ---------------------------------------------------------------------------


def encode_cursor(sort_value: str, paper_id: str) -> str:
    """Encode (sort_value, paper_id) as a base64url opaque cursor."""
    pair = json.dumps([sort_value, paper_id])
    return base64.urlsafe_b64encode(pair.encode()).decode()


def decode_cursor(cursor: str) -> tuple[str, str] | None:
    """Decode an opaque base64url cursor back to (sort_value, paper_id).

    Returns ``None`` when the cursor is malformed.
    """
    try:
        raw = base64.urlsafe_b64decode(cursor).decode()
        pair: list[str] = json.loads(raw)
        if isinstance(pair, list) and len(pair) == 2:
            return (str(pair[0]), str(pair[1]))
        return None
    except (ValueError, json.JSONDecodeError, UnicodeDecodeError):
        return None
