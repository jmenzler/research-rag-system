"""Read-only /api/papers and /api/chunks surface (CHAT-04 + Phase 3 corpus browser).

Reads the existing parents/parents.sqlite (owned by src/ingest/) — NEVER writes.
All SELECTs use parameterized ``?`` placeholders.

Phase 1: per-id endpoints for the Citation Sheet (D-08).
Phase 3 (03-02-PLAN): expands with:
  - GET /api/papers          cursor-paginated list + FTS5 + facets + citedInChat cross-link
  - GET /api/papers/{id}     provenance detail (ingested_at, collection, chunk_count, …)
  - GET /api/papers/{id}/content-list  content_list.json file serving (path-guarded)
"""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from src.config.paths import SOURCES_DIR
from src.server.papers_view import (
    PaperRow,
    apply_facets,
    decode_cursor,
    encode_cursor,
    get_read_view,
    rebuild_corpus_fts,
    search_corpus_fts,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARENTS_DB_PATH = PROJECT_ROOT / "parents" / "parents.sqlite"
CHATS_DB_PATH = PROJECT_ROOT / "parents" / "chats.db"

router = APIRouter(prefix="/api")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _open_parents_readonly(db_path: Path | None = None) -> sqlite3.Connection:
    """Open parents.sqlite read-only (uri=True with mode=ro).

    Even if a future bug added a write statement, sqlite refuses writes on a
    read-only connection (defence-in-depth — T-01-W1B-09).
    """
    path = db_path if db_path is not None else PARENTS_DB_PATH
    uri = f"file:{path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------------------
# Phase 1 — per-id endpoints (KEpt, unbroken)
# ---------------------------------------------------------------------------


@router.get("/chunks/{parent_id}")
def get_chunk(parent_id: str) -> dict[str, Any]:
    """Return the parent chunk text and source metadata for the D-08 Citation Sheet.

    404 when no row matches; 503 when parents.sqlite is missing.
    """
    try:
        conn = _open_parents_readonly()
    except sqlite3.OperationalError as exc:  # parents.sqlite missing or unreadable
        raise HTTPException(status_code=503, detail="parents store unavailable") from exc
    try:
        row = conn.execute(
            "SELECT id, text, source_file, notebook, modality, page_number "
            "FROM parents WHERE id = ?",
            (parent_id,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        raise HTTPException(status_code=404, detail=f"parent {parent_id} not found")
    return {
        "id": row["id"],
        "text": row["text"],
        "source_file": row["source_file"],
        "notebook": row["notebook"],
        "modality": row["modality"],
        "page_number": row["page_number"],
    }


# ---------------------------------------------------------------------------
# Phase 3 — corpus endpoints
# ---------------------------------------------------------------------------


def _get_cited_paper_ids(chat_id: str) -> set[str]:
    """Return paper_ids cited in messages belonging to *chat_id*.

    Returns an empty set when chats.db is unreachable or the chat has no
    citations (never raises).
    """
    if not CHATS_DB_PATH.is_file():
        return set()
    try:
        conn = sqlite3.connect(str(CHATS_DB_PATH), uri=True)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT DISTINCT c.paper_id"
                "  FROM citations c"
                "  JOIN messages m ON m.id = c.message_id"
                " WHERE m.chat_id = ? AND c.paper_id IS NOT NULL",
                (chat_id,),
            ).fetchall()
            return {r["paper_id"] for r in rows}
        finally:
            conn.close()
    except (sqlite3.OperationalError, sqlite3.DatabaseError):
        return set()


def _simple_search(rows: list[PaperRow], q: str) -> list[PaperRow]:
    """Simple in-memory substring search fallback when FTS5 is unavailable."""
    q_lower = q.lower()
    return [
        r for r in rows
        if (r.get("title") or "").lower().find(q_lower) >= 0
        or (r.get("short_cite") or "").lower().find(q_lower) >= 0
    ]


@router.get("/papers")
def list_papers(
    q: str | None = None,
    collection: str | None = None,
    year: int | None = None,
    year_from: int | None = Query(None, alias="yearFrom"),
    year_to: int | None = Query(None, alias="yearTo"),
    ingested: str | None = None,
    has_arxiv: bool | None = Query(None, alias="hasArxiv"),
    has_doi: bool | None = Query(None, alias="hasDoi"),
    sort: str = "ingested",
    dir: str = "desc",
    cited_in_chat: str | None = Query(None, alias="citedInChat"),
    cursor: str | None = None,
    limit: int = 100,
) -> dict[str, Any]:
    """Return a cursor-paginated, facet-filtered, optionally FTS5-searched list of papers.

    The read-view is built from the 3-substrate join (doc-dirs + meta.json +
    sources + Milvus). All filters are AND-ed.

    T-03-03: ``limit`` is clamped server-side to ``< =200`` (``max(1, min(limit, 200))``).
    Malformed FTS5 queries return ``[]`` (safe degradation).
    """
    limit = max(1, min(limit, 200))

    # 1. Get the read-view (TTL+watermark cached)
    try:
        rows = get_read_view(parents_db=PARENTS_DB_PATH, sources_root=SOURCES_DIR)
    except Exception:
        logger.exception("failed to build read-view")
        return {"papers": [], "next_cursor": None, "total": 0}

    # 2. FTS5 search — try chats.db first, fall back to in-memory substring
    if q:
        try:
            if CHATS_DB_PATH.is_file():
                rebuild_corpus_fts(
                    chats_db=str(CHATS_DB_PATH),
                    parents_db=PARENTS_DB_PATH,
                    sources_root=SOURCES_DIR,
                )
                fts_ids = search_corpus_fts(q, chats_db=str(CHATS_DB_PATH))
                if fts_ids is not None:
                    id_set = set(fts_ids)
                    rows = [r for r in rows if r.get("paper_id") in id_set]
                else:
                    rows = []
            else:
                rows = _simple_search(rows, q)
        except Exception:
            rows = _simple_search(rows, q)

    # 3. citedInChat cross-link (CORPUS-06)
    if cited_in_chat:
        cited_ids = _get_cited_paper_ids(cited_in_chat)
        if cited_ids:
            rows = [r for r in rows if r.get("paper_id") in cited_ids]
        else:
            # Chat exists but has no citations → empty result
            rows = []

    # 4. Apply facets
    coll_filter = collection.split(",") if collection else None
    # Support both ``year`` (exact) and ``year_from/year_to`` (range)
    yf = year_from if year_from is not None else year
    yt = year_to if year_to is not None else year
    rows = apply_facets(
        rows,
        collections=coll_filter,
        year_from=yf,
        year_to=yt,
        ingested_bucket=ingested,
        has_arxiv=has_arxiv,
        has_doi=has_doi,
    )

    # 5. Sort
    def _sort_key(row: PaperRow) -> Any:  # noqa: ANN401
        if sort == "title":
            return (row.get("title") or "").lower()
        if sort == "year":
            return row.get("year") or 0
        return row.get("ingested_at") or 0  # default: ingested

    rows.sort(key=_sort_key, reverse=dir.lower() != "asc")

    # 6. Cursor pagination
    total = len(rows)
    start = 0
    if cursor:
        decoded = decode_cursor(cursor)
        if decoded:
            cursor_val, cursor_pid = decoded
            for i, row in enumerate(rows):
                current_key = str(_sort_key(row))
                past_k = current_key > cursor_val
                same_k_and_past_id = (
                    current_key == cursor_val
                    and row.get("paper_id") == cursor_pid
                )
                if past_k or same_k_and_past_id:
                    start = i + 1
                    break

    page = rows[start:start + limit]
    next_cursor: str | None = None
    if start + limit < total:
        last = page[-1]
        next_cursor = encode_cursor(str(_sort_key(last)), last.get("paper_id") or "")

    return {"papers": page, "next_cursor": next_cursor, "total": total}


@router.get("/papers/{paper_id}/content-list")
def get_paper_content(paper_id: str) -> Any:  # noqa: ANN401
    """Serve the ``content_list.json`` file for a paper.

    Security (T-03-02):
      1. Regex-validate the ``paper_id`` segment.
      2. Build candidate path ``sources/<collection>/<paper_id>/content_list.json``
         by searching across all collections.
      3. ``resolve()`` the candidate.
      4. Assert ``SOURCES_DIR.resolve()`` is in ``candidate.parents`` (canonicalize-and-assert).
      5. 400 on escape, 404 on missing file.
    """
    # Regex guard: reject paths containing ``..`` or other dangerous chars
    import re

    if not re.match(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$", paper_id):
        raise HTTPException(status_code=400, detail="invalid paper_id format")

    sources_root = SOURCES_DIR.resolve()
    # Search across all collections for the doc-dir with matching name
    try:
        for coll_dir in sources_root.iterdir():
            if not coll_dir.is_dir() or coll_dir.name.startswith("_"):
                continue
            doc_dir = coll_dir / paper_id
            if doc_dir.is_dir():
                target = (doc_dir / "content_list.json").resolve()
                # Canonicalize-and-assert (T-03-02)
                if sources_root not in target.parents:
                    raise HTTPException(
                        status_code=400,
                        detail="path escape detected",
                    )
                if not target.is_file():
                    raise HTTPException(
                        status_code=404,
                        detail="content_list.json not found",
                    )
                return FileResponse(target, media_type="application/json")
    except PermissionError:
        raise HTTPException(status_code=500, detail="filesystem read error")  # noqa: B904

    raise HTTPException(status_code=404, detail="paper not found")


@router.get("/papers/{paper_id:path}")
def get_paper(paper_id: str) -> dict[str, Any]:
    """Return paper provenance detail (Phase 3) OR the Phase-1 citation-sheet stub.

    Phase 3 (CORPUS-05): First tries the read-view (``get_read_view``) and
    returns provenance fields (``fetch_source``, ``parser_version``, ``extraction_quality``,
    ``ingested_at``, ``collection``, ``chunk_count``) + ``content_list`` link.

    Fallback: Phase-1 behavior — look up by ``source_file`` in the parents table
    (for backwards compatibility with Citation-Sheet URLs that pass source_file).
    """
    # Phase 3: read-view lookup by paper_id (canonical_id)
    try:
        rows = get_read_view(parents_db=PARENTS_DB_PATH, sources_root=SOURCES_DIR)
        for row in rows:
            if row.get("paper_id") == paper_id:
                result = dict(row)
                result["content_list"] = f"/api/papers/{paper_id}/content-list"
                return result
    except Exception:
        pass  # Fall through to Phase-1 fallback

    # Fallback: Phase-1 parents-table lookup by source_file
    try:
        conn = _open_parents_readonly()
    except sqlite3.OperationalError as exc:
        raise HTTPException(status_code=503, detail="parents store unavailable") from exc
    try:
        row = conn.execute(
            "SELECT source_file, notebook FROM parents WHERE source_file = ? LIMIT 1",
            (paper_id,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        raise HTTPException(status_code=404, detail=f"paper {paper_id} not found")
    return {
        "id": paper_id,
        "source_file": row["source_file"],
        "notebook": row["notebook"],
        # Tier-1 fields (title/authors/year/short_cite) land in Phase 3 from the
        # parents.sqlite.sources table. Stub them as None for Phase 1.
        "title": None,
        "authors": None,
        "year": None,
        "short_cite": row["source_file"],
    }
