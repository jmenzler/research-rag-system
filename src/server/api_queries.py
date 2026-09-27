"""D-14a per-id + list/facet/cross-link audit-trail endpoints.

Per-id: READ-ONLY filesystem reads paths guarded by regex + canonicalize-and-assert.
List: in-memory index via query_index.py, enriched by messages join in chats.db.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import PlainTextResponse

from src.server.query_index import (
    QueryRow,
    apply_cursor,
    apply_query_facets,
    get_query_index,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Default queries-log root. Resolved per-request via :func:`_queries_root` so the
# ``QUERY_LOG_ROOT`` env var (honoured by src/server/api_chats.py:_record_cancellation_outcome
# and src/query/query_logger.py:make_query_logger) is also respected here. Keeping
# this module-level constant pinned to the default keeps existing imports working
# and the per-request resolution falls back to it when QUERY_LOG_ROOT is unset.
QUERIES_LOG_DIR = PROJECT_ROOT / "logs" / "queries"
CHATS_DB_PATH = PROJECT_ROOT / "parents" / "chats.db"


def _queries_root() -> Path:
    """Return the active queries-log root, honouring ``QUERY_LOG_ROOT``.

    Re-evaluated per-request so tests / deploys that redirect the log root via
    the env var see the correct directory (mirrors the resolution pattern in
    ``src/server/api_chats.py::_record_cancellation_outcome``).
    """
    env_root = os.getenv("QUERY_LOG_ROOT")
    return Path(env_root) if env_root else QUERIES_LOG_DIR


# query_id is the leaf dirname under <date>/, e.g.
#   014523123456Z_a1b2c3d4_12345_140735  (HHMMSSffffffZ_qhash8_pid_tid)
# qhash8 is lowercase hex from sha256().hexdigest()[:8]; timestamp is uppercase Z + digits.
QUERY_ID_RE = re.compile(r"^[A-Za-z0-9_]+$")
# stage name selects the NN_<name>.json file; the {name} portion alone.
STAGE_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
# chunk filename: NN_<child_id>.txt — child_id is hex/uuid-ish.
CHUNK_FILENAME_RE = re.compile(r"^[0-9]{2}_[A-Za-z0-9_-]+\.txt$")

router = APIRouter(prefix="/api")


def _resolve_query_dir(query_id: str) -> Path:
    """Find the per-query directory under logs/queries/<date>/.

    Raises HTTPException(400) on invalid query_id; 404 when no dir is found.
    Performs canonicalize-and-assert (01-RESEARCH §Known Threat Patterns line 1188).
    """
    if not QUERY_ID_RE.fullmatch(query_id):
        raise HTTPException(status_code=400, detail="invalid query_id format")
    root = _queries_root()
    if not root.is_dir():
        raise HTTPException(status_code=404, detail="no audit trail on disk")
    # Search every date dir; first match wins (query_ids are unique by construction
    # — they embed a microsecond timestamp + qhash + pid + tid).
    queries_root = root.resolve()
    for date_dir in root.iterdir():
        if not date_dir.is_dir():
            continue
        candidate = (date_dir / query_id).resolve()
        # Canonicalize-and-assert: confirm the resolved path is still under the
        # queries root. Defends against symlink escape + .. traversal.
        if queries_root not in candidate.parents:
            continue
        if candidate.is_dir():
            return candidate
    raise HTTPException(status_code=404, detail=f"query {query_id} not found")


# List-endpoint helpers

_SORT_COL: dict[str, str] = {
    "ts": "ts_started",
    "cost": "cost_usd",
    "latencyMs": "latency_ms",
}


def _open_chats_ro() -> sqlite3.Connection | None:
    """Open chats.db read-only, or None if absent/degraded."""
    if not CHATS_DB_PATH.is_file():
        return None
    uri = f"file:{CHATS_DB_PATH}?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.OperationalError:
        logger.warning("api_queries: cannot open chats.db read-only")
        return None


def _enrich_from_messages(rows: list[QueryRow]) -> list[QueryRow]:
    """Fill retriever and chat_id from messages.query_id join."""
    conn = _open_chats_ro()
    if conn is None:
        return rows
    try:
        cur = conn.execute(
            "SELECT query_id, retriever, chat_id FROM messages WHERE query_id IS NOT NULL"
        )
        lookup: dict[str, tuple[str | None, str | None]] = {}
        for r in cur:
            lookup[r["query_id"]] = (r["retriever"], r["chat_id"])
    except sqlite3.OperationalError:
        # chats.db present but pre-Phase-1 / empty (no messages table) — degrade.
        return rows
    finally:
        conn.close()
    result: list[QueryRow] = []
    for row in rows:
        qid = row["query_id"]
        if qid in lookup:
            row["retriever"] = lookup[qid][0]
            row["chat_id"] = lookup[qid][1]
        result.append(row)
    return result


def _paper_queries_set(paper_id: str) -> set[str]:
    """Return query_ids that cited *paper_id* (via citations.paper_id join)."""
    conn = _open_chats_ro()
    if conn is None:
        return set()
    try:
        cur = conn.execute(
            "SELECT DISTINCT m.query_id FROM citations c "
            "JOIN messages m ON m.id = c.message_id "
            "WHERE c.paper_id = ? AND m.query_id IS NOT NULL",
            (paper_id,),
        )
        return {r["query_id"] for r in cur}
    except sqlite3.OperationalError:
        # chats.db present but pre-Phase-1 / empty (no citations/messages) — degrade.
        return set()
    finally:
        conn.close()


# ---------------------------------------------------------------------------


@router.get("/queries/{query_id}")
def get_query(query_id: str) -> dict[str, Any]:
    """Return meta.json + raw report.md text + index of available stage / chunk files."""
    qdir = _resolve_query_dir(query_id)
    meta_path = qdir / "meta.json"
    report_path = qdir / "report.md"
    meta: dict[str, Any] = {}
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text())
        except json.JSONDecodeError:
            logger.warning("malformed meta.json at %s", meta_path)
    report_md = report_path.read_text() if report_path.is_file() else ""
    stages = sorted(p.name for p in qdir.glob("[0-9][0-9]_*.json"))
    chunks_dir = qdir / "chunks"
    chunks = (
        sorted(p.name for p in chunks_dir.glob("[0-9][0-9]_*.txt")) if chunks_dir.is_dir() else []
    )
    return {
        "query_id": query_id,
        "meta": meta,
        "report_md": report_md,
        "stages": stages,
        "chunks": chunks,
    }


@router.get("/queries/{query_id}/stages/{name}")
def get_query_stage(query_id: str, name: str) -> dict[str, Any]:
    """Return the parsed JSON payload of a stage file (NN_<name>.json)."""
    qdir = _resolve_query_dir(query_id)
    if not STAGE_NAME_RE.fullmatch(name):
        raise HTTPException(status_code=400, detail="invalid stage name")
    # Find NN_<name>.json — there's exactly one per stage prefix in a well-formed
    # audit dir. The glob pattern is fixed (no user-controlled wildcards).
    matches = sorted(qdir.glob(f"[0-9][0-9]_{name}.json"))
    if not matches:
        raise HTTPException(status_code=404, detail=f"stage {name} not found")
    target = matches[0].resolve()
    qdir_resolved = qdir.resolve()
    if qdir_resolved not in target.parents:
        raise HTTPException(status_code=400, detail="path escapes query dir")
    try:
        payload = json.loads(target.read_text())
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail="malformed stage payload") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=500, detail="stage payload must be a JSON object")
    # Flatten the payload at top level and inject `stage` (the stage name) — matches the
    # contract asserted by tests/server/test_api_queries.py (body["stage"], body["latency_ms"]).
    return {"stage": name, **payload}


@router.get("/queries/{query_id}/chunks/{filename}", response_class=PlainTextResponse)
def get_query_chunk(query_id: str, filename: str) -> str:
    """Return the raw text of a retrieved-chunk file (chunks/NN_<child_id>.txt)."""
    qdir = _resolve_query_dir(query_id)
    if not CHUNK_FILENAME_RE.fullmatch(filename):
        raise HTTPException(status_code=400, detail="invalid chunk filename")
    target = (qdir / "chunks" / filename).resolve()
    # canonicalize-and-assert: target must be under qdir/chunks specifically.
    chunks_dir = (qdir / "chunks").resolve()
    if chunks_dir not in target.parents:
        raise HTTPException(status_code=400, detail="path escapes chunks dir")
    if not target.is_file():
        raise HTTPException(status_code=404, detail="chunk not found")
    return target.read_text()


@router.get("/queries")
def list_queries(
    retriever: str | None = Query(default=None),
    outcome: str | None = Query(default=None),
    cost_min: float | None = Query(default=None, ge=0, alias="costMin"),
    latency_min: float | None = Query(default=None, ge=0, alias="latencyMin"),
    sort: str = Query(default="ts"),
    dir: str = Query(default="desc"),
    paper: str | None = Query(default=None),
    cursor: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
) -> dict[str, Any]:
    """List queries with per-row totals, facets, and paper cross-link."""
    rows = get_query_index()
    rows = _enrich_from_messages(rows)

    if paper:
        qids = _paper_queries_set(paper)
        rows = [r for r in rows if r["query_id"] in qids]

    ret_list = retriever.split(",") if retriever else None
    out_list = outcome.split(",") if outcome else None
    lat_val = int(latency_min) if latency_min is not None else None

    rows = apply_query_facets(
        rows,
        retrievers=ret_list,
        outcomes=out_list,
        cost_min=cost_min,
        latency_min=lat_val,
    )

    if sort != "ts" or dir != "desc":
        col = _SORT_COL.get(sort, "ts_started")
        reverse = dir == "desc"
        if col == "cost_usd":
            rows = sorted(rows, key=lambda r: r.get("cost_usd") or 0.0, reverse=reverse)
        elif col == "latency_ms":
            rows = sorted(rows, key=lambda r: r.get("latency_ms") or 0, reverse=reverse)
        else:
            rows = sorted(rows, key=lambda r: r.get("ts_started") or "", reverse=reverse)

    page, next_cursor, total = apply_cursor(
        rows,
        cursor=cursor,
        limit=limit,
        sort=sort,
        direction=dir,
    )
    return {"queries": page, "next_cursor": next_cursor, "total": total}
