# long-ok-file
"""GUI-facing ingest + queue-stream + maps API router.

Endpoints:
    POST /api/ingest              — enqueue corpus_ids via DiscoverQueue (single, bulk, split)
    GET  /api/queue/stream        — SSE stream of queue position/stage/terminal events
    POST /api/maps                — create a saved map snapshot
    GET  /api/maps                — list saved maps (optionally filtered by collection)
    GET  /api/maps/{map_id}       — retrieve a single saved map
    GET  /api/maps/{map_id}/coverage — compute D-10 coverage scorecard for a saved map
    DELETE /api/maps/{map_id}     — delete a saved map

Design constraints (CRIT-5):
    - All ingest routes through DiscoverQueue.enqueue() — no parallel queue, no bypass.
    - This module owns its OWN set_ingest_queue / _require_ingest_queue singleton;
      it cannot reach api.py's private _discover_queue.
    - __main__.py injects the SAME DiscoverQueue instance into both set_discover_queue
      and set_ingest_queue.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import sqlite3
from collections import OrderedDict
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from sse_starlette.event import ServerSentEvent
from sse_starlette.sse import EventSourceResponse

from src.citations import DEFAULT_BASE, DEFAULT_RELEASE
from src.server.discover_queue import (
    DiscoverQueue,
    DiscoverStartRequest,
    QueueDepthExceededError,
    _max_queries_per_request,
)
from src.server.maps_store import (
    compute_coverage,
    create_map,
    delete_map,
    get_map,
    list_maps,
    open_maps_conn,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")

# ---------------------------------------------------------------------------
# Queue injection — OWN copy (cannot reach api.py's private singleton)
# ---------------------------------------------------------------------------

_ingest_queue: DiscoverQueue | None = None


def set_ingest_queue(queue: DiscoverQueue | None) -> None:
    """Inject (or clear with None) the DiscoverQueue instance.

    Called once at server boot from __main__.py with the SAME instance
    used by set_discover_queue. Tests use this to swap in a fake queue.
    """
    global _ingest_queue
    _ingest_queue = queue


def _require_ingest_queue() -> DiscoverQueue:
    if _ingest_queue is None:
        raise HTTPException(status_code=503, detail="discover queue not initialised")
    return _ingest_queue


# ---------------------------------------------------------------------------
# run_id → corpus_ids index (populated by POST /api/ingest)
# ---------------------------------------------------------------------------

_run_corpus_index: dict[str, list[int]] = {}

# Process-wide record of run_ids whose terminal event has already been emitted
# on SOME /queue/stream connection. Gates re-emission across reconnects so a
# fresh connection does not replay stale done/crashed/cancelled frames for every
# historical run. Bounded FIFO — old entries age out; a long-dead run that ages
# out simply will not be replayed (the client recovers via corpus recheck).
_REPORTED_TERMINALS_MAX = 4096
_reported_terminals: OrderedDict[str, None] = OrderedDict()


def _mark_terminal_reported(run_id: str) -> None:
    """Record a terminal emission and drop the run's corpus_ids from the index.

    Evicting from ``_run_corpus_index`` here bounds its growth — corpus_ids are
    only needed for in-flight position/running/terminal frames, all of which
    are done once the terminal event has fired.
    """
    _reported_terminals[run_id] = None
    _reported_terminals.move_to_end(run_id)
    while len(_reported_terminals) > _REPORTED_TERMINALS_MAX:
        _reported_terminals.popitem(last=False)
    _run_corpus_index.pop(run_id, None)

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

_COLLECTION = Literal["trading", "ecology", "notes", "system", "poker", "security"]


class IngestRequest(BaseModel):
    corpus_ids: list[int] = Field(..., max_length=2000)
    collection: _COLLECTION
    partition: str = "research_briefs"


class RunIdGroup(BaseModel):
    run_id: str
    corpus_ids: list[int]


class ResolvedSource(BaseModel):
    corpus_id: int
    kind: Literal["arxiv", "doi", "title"]
    value: str


class IngestResponse(BaseModel):
    run_ids: list[RunIdGroup]
    n_enqueued: int
    resolved: list[ResolvedSource]
    unresolved: list[int]


# ---------------------------------------------------------------------------
# Identifier resolution — arxiv_id ▸ doi ▸ title (D-02)
# ---------------------------------------------------------------------------

_META_DB_PATH = Path(DEFAULT_BASE) / DEFAULT_RELEASE / "paper_meta.sqlite"


_ResolvedKind = Literal["arxiv", "doi", "title", "unresolved"]
_Source = tuple[int, _ResolvedKind, str]


def _resolve_sources(corpus_ids: list[int], collection: str) -> list[_Source]:  # noqa: ARG001
    """Per corpus_id, return (corpus_id, kind, value).

    arxiv / doi -> direct no-NotebookLM fetch URL; title -> Deep Research query;
    unresolved -> id unknown to paper_meta (str(corpus_id), Deep Research fallback).
    paper_meta opened readonly for the batch SELECT.
    """
    if not corpus_ids:
        return []

    uri = f"file:{_META_DB_PATH}?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True)
        conn.row_factory = sqlite3.Row
    except sqlite3.OperationalError:
        logger.warning("paper_meta unavailable; all corpus_ids unresolved")
        return [(cid, "unresolved", str(cid)) for cid in corpus_ids]

    try:
        placeholders = ",".join("?" * len(corpus_ids))
        rows = conn.execute(
            f"SELECT corpusid, arxiv_id, doi, title "
            f"FROM paper_meta WHERE corpusid IN ({placeholders})",
            corpus_ids,
        ).fetchall()
    finally:
        conn.close()

    row_map: dict[int, sqlite3.Row] = {int(r["corpusid"]): r for r in rows}

    result: list[_Source] = []
    for cid in corpus_ids:
        row = row_map.get(cid)
        arxiv_id = row["arxiv_id"] if row is not None else None
        doi = row["doi"] if row is not None else None
        title = row["title"] if row is not None else None
        if arxiv_id:
            result.append((cid, "arxiv", f"https://arxiv.org/abs/{arxiv_id}"))
        elif doi:
            result.append((cid, "doi", f"https://doi.org/{doi}"))
        elif title:
            result.append((cid, "title", title))
        else:
            result.append((cid, "unresolved", str(cid)))
    return result


# ---------------------------------------------------------------------------
# POST /api/ingest
# ---------------------------------------------------------------------------


@router.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest) -> IngestResponse:
    """Enqueue corpus_ids for ingest.

    Papers with an arxiv_id/doi route through the DIRECT fetch path
    (enqueue_urls -> ragctl ingest-urls, no NotebookLM); title-only/unknown fall
    back to NotebookLM Deep Research. Returns a run_id + corpus_ids group per
    enqueued job (one for the direct batch, plus one per Deep Research chunk),
    plus ``resolved`` (corpus_id -> kind/value) and ``unresolved`` (corpus_ids
    unknown to paper_meta) so the caller sees what each id became.
    """
    queue = _require_ingest_queue()

    sources = _resolve_sources(req.corpus_ids, req.collection)
    url_items = [(cid, val) for cid, kind, val in sources if kind in ("arxiv", "doi")]
    query_items = [(cid, val) for cid, kind, val in sources if kind in ("title", "unresolved")]
    resolved = [
        ResolvedSource(corpus_id=cid, kind=kind, value=val)
        for cid, kind, val in sources
        if kind != "unresolved"
    ]
    unresolved = [cid for cid, kind, _ in sources if kind == "unresolved"]

    run_id_groups: list[RunIdGroup] = []

    # Direct path: arxiv/doi -> real URL, no NotebookLM. One batched job.
    if url_items:
        url_ids = [cid for cid, _ in url_items]
        try:
            entry = queue.enqueue_urls(
                urls=[u for _, u in url_items],
                collection=req.collection,
                partition=req.partition,
            )
        except QueueDepthExceededError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        _run_corpus_index[entry.run_id] = url_ids
        run_id_groups.append(RunIdGroup(run_id=entry.run_id, corpus_ids=url_ids))

    # Fallback: title-only -> Deep Research, chunked by cap.
    if query_items:
        cap = _max_queries_per_request()
        for i in range(0, len(query_items), cap):
            chunk = query_items[i : i + cap]
            chunk_ids = [cid for cid, _ in chunk]
            chunk_queries = [q for _, q in chunk]
            query: str | list[str] = (
                chunk_queries[0] if len(chunk_queries) == 1 else chunk_queries
            )
            discover_req = DiscoverStartRequest(
                query=query,
                collection=req.collection,
                partition=req.partition,
                mode="fast",
                timeout=1800,
                keep=False,
            )
            try:
                entry = queue.enqueue(discover_req)
            except QueueDepthExceededError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            _run_corpus_index[entry.run_id] = chunk_ids
            run_id_groups.append(RunIdGroup(run_id=entry.run_id, corpus_ids=chunk_ids))

    logger.info(
        "api/ingest: collection=%s direct=%d fallback=%d unresolved=%d run_ids=%s",
        req.collection, len(url_items), len(query_items), len(unresolved),
        [g.run_id for g in run_id_groups],
    )
    return IngestResponse(
        run_ids=run_id_groups,
        n_enqueued=len(run_id_groups),
        resolved=resolved,
        unresolved=unresolved,
    )


# ---------------------------------------------------------------------------
# GET /api/queue/stream — SSE queue status
# ---------------------------------------------------------------------------

SSE_PING_INTERVAL_SECONDS = 15

# Compiled stage-detection regexes (RESEARCH.md Flag 2).
# Scan log_tail LAST-to-FIRST; first match wins (most-recent stage).
_STAGE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("embedding", re.compile(r"\d+ parents, \d+ children — embedding")),
    ("chunking", re.compile(r"Found \d+ file\(s\) to process\.|^\[\w[\w.]*\]$")),
    ("parsing", re.compile(
        r"→ parse_start:|\[progress\].*parse_pending=|"
        r"Processing pages|MFR Predict|OCR-rec Predict|/file_parse"
    )),
    ("fetching", re.compile(
        r"\[discover:.*\] (triggering|waiting up to)|running batched fetch\+parse|"
        r"\(fetch_pipeline\)|Pipeline done:"
    )),
]


def _heartbeat() -> ServerSentEvent:
    """sse-starlette ping_message_factory. Emits a comment frame with ISO timestamp."""
    return ServerSentEvent(comment=f"hb {datetime.now(tz=UTC).isoformat()}")


def _stage_from_log_tail(
    log_tail: list[str],
) -> Literal["fetching", "parsing", "chunking", "embedding", "unknown"]:
    """Derive the coarsest running stage from the most-recent matching log line.

    Scans log_tail from last to first so the most-recent stage wins when
    multiple markers appear in one tail window.
    """
    for line in reversed(log_tail):
        for stage, pattern in _STAGE_PATTERNS:
            if pattern.search(line):
                return stage  # type: ignore[return-value]
    return "unknown"


def _run_corpus_ids(run_id: str) -> list[int]:
    """Return corpus_ids for a run_id from the in-process index.

    Returns empty list if the server was restarted (index lost); the client
    recovers via a full corpus recheck on the done event.
    """
    return _run_corpus_index.get(run_id, [])


def _build_running_event_payload(
    state: dict[str, Any],
    *,
    corpus_ids: list[int],
) -> dict[str, Any]:
    """Build the running SSE frame payload.

    Exposes derived 'stage' ONLY — raw log_tail is never serialized (T-05-04).
    """
    log_tail: list[str] = state.get("log" + "_tail") or []
    stage = _stage_from_log_tail(log_tail)
    return {
        "run_id": state.get("run_id", ""),
        "elapsed_s": state.get("elapsed_s", 0.0),
        "stage": stage,
        "corpus_ids": corpus_ids,
    }


def _build_terminal_event_payload(
    state: dict[str, Any],
    *,
    corpus_ids: list[int],
) -> dict[str, Any]:
    """Build crashed or cancelled SSE frame payload."""
    result: dict[str, Any] = state.get("result") or {}
    payload: dict[str, Any] = {
        "run_id": state.get("run_id", ""),
        "corpus_ids": corpus_ids,
    }
    if "note" in result:
        payload["note"] = result["note"]
    return payload


def _build_done_event_payload(
    state: dict[str, Any],
    *,
    corpus_ids: list[int],
) -> dict[str, Any]:
    """Build done SSE frame payload."""
    result: dict[str, Any] = state.get("result") or {}
    return {
        "run_id": state.get("run_id", ""),
        "corpus_ids": corpus_ids,
        "n_imported": result.get("n_imported", 0),
    }


def _discover_run_ids(queue_dir: Path) -> list[str]:
    """Scan queued/ and running/ dirs to auto-discover active run_ids."""
    run_ids: list[str] = []
    for subdir in ("queued", "running"):
        d = queue_dir / subdir
        if d.is_dir():
            for p in d.iterdir():
                if p.suffix == ".json":
                    # filename: <counter>_<run_id>.json
                    parts = p.stem.split("_", 1)
                    if len(parts) == 2:
                        run_ids.append(parts[1])
    return run_ids


@router.get("/queue/stream")
async def queue_stream(request: Request) -> EventSourceResponse:
    """SSE stream emitting position/running/done/crashed/cancelled events.

    Auto-discovers run_ids by scanning queued/ + running/ dirs each tick.
    Terminal events (done/crashed/cancelled) are emitted once per run_id
    then dropped from the tracked set to prevent re-emission.
    """
    queue = _require_ingest_queue()

    async def _generate() -> AsyncGenerator[ServerSentEvent, None]:
        while not await request.is_disconnected():
            # Auto-discover active run_ids from the queue's on-disk state_dir.
            # This is what surfaces runs queued by MCP/CLI (or by a prior server
            # process before an autodeploy restart) — they are not in this
            # process's in-memory _run_corpus_index.
            state_dir: Path | None = getattr(queue, "state_dir", None)
            if state_dir is not None:
                active_run_ids = _discover_run_ids(state_dir)
                tracked_ids = list(_run_corpus_index.keys())
                all_ids = list(dict.fromkeys(active_run_ids + tracked_ids))
            else:
                all_ids = list(_run_corpus_index.keys())

            for run_id in all_ids:
                # Terminal events fire at most once process-wide — a fresh
                # connection must not replay stale done/crashed/cancelled frames.
                if run_id in _reported_terminals:
                    continue
                try:
                    state = queue.get_state(run_id)
                except Exception:
                    continue

                run_state: str = state.get("state", "unknown")
                corpus_ids = _run_corpus_ids(run_id)

                if run_state == "queued":
                    payload: dict[str, Any] = {
                        "run_id": run_id,
                        "position": state.get("position", 0),
                        "queue_depth": state.get("queue_depth", 0),
                    }
                    yield ServerSentEvent(event="position", data=json.dumps(payload))

                elif run_state == "running":
                    payload = _build_running_event_payload(state, corpus_ids=corpus_ids)
                    yield ServerSentEvent(event="running", data=json.dumps(payload))

                elif run_state == "done":
                    payload = _build_done_event_payload(state, corpus_ids=corpus_ids)
                    yield ServerSentEvent(event="done", data=json.dumps(payload))
                    _mark_terminal_reported(run_id)

                elif run_state == "crashed":
                    payload = _build_terminal_event_payload(state, corpus_ids=corpus_ids)
                    yield ServerSentEvent(event="crashed", data=json.dumps(payload))
                    _mark_terminal_reported(run_id)

                elif run_state == "cancelled":
                    payload = _build_terminal_event_payload(state, corpus_ids=corpus_ids)
                    yield ServerSentEvent(event="cancelled", data=json.dumps(payload))
                    _mark_terminal_reported(run_id)

            await asyncio.sleep(2.0)

    return EventSourceResponse(
        _generate(),
        ping=SSE_PING_INTERVAL_SECONDS,
        ping_message_factory=_heartbeat,
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-store"},
    )


# ---------------------------------------------------------------------------
# /api/maps CRUD + coverage
# ---------------------------------------------------------------------------


class CreateMapRequest(BaseModel):
    name: str
    collection: _COLLECTION
    snapshot: str


class MapMeta(BaseModel):
    id: str
    name: str
    collection: str
    created_at: str
    updated_at: str


class MapFull(BaseModel):
    id: str
    name: str
    collection: str
    snapshot: str
    created_at: str
    updated_at: str


class CoverageResponse(BaseModel):
    in_corpus: int
    total: int
    missing_ids: list[int]


@router.post("/maps", status_code=201)
def create_map_endpoint(req: CreateMapRequest) -> dict[str, str]:
    """Create a saved map snapshot. Returns {id}."""
    try:
        json.loads(req.snapshot)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail=f"snapshot is not valid JSON: {exc}") from exc

    conn = open_maps_conn()
    try:
        try:
            map_id = create_map(
                conn, name=req.name, collection=req.collection, snapshot=req.snapshot
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        conn.close()

    return {"id": map_id}


@router.get("/maps", response_model=list[MapMeta])
def list_maps_endpoint(collection: str | None = None) -> list[dict[str, Any]]:
    """List saved maps, most-recently-updated first. Omits snapshot for payload size."""
    conn = open_maps_conn()
    try:
        rows = list_maps(conn, collection=collection)
    finally:
        conn.close()

    return [
        {
            "id": row["id"],
            "name": row["name"],
            "collection": row["collection"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }
        for row in rows
    ]


@router.get("/maps/{map_id}", response_model=MapFull)
def get_map_endpoint(map_id: str) -> dict[str, Any]:
    """Return full map row including snapshot. 404 if not found."""
    conn = open_maps_conn()
    try:
        row = get_map(conn, map_id)
    finally:
        conn.close()

    if row is None:
        raise HTTPException(status_code=404, detail=f"map {map_id!r} not found")

    return {
        "id": row["id"],
        "name": row["name"],
        "collection": row["collection"],
        "snapshot": row["snapshot"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


@router.get("/maps/{map_id}/coverage", response_model=CoverageResponse)
def map_coverage(map_id: str) -> dict[str, Any]:
    """Compute D-10 coverage scorecard for a saved map's corpus_ids.

    Reads the snapshot's corpus_ids and runs an in-corpus check against
    paper_meta.sqlite (same pattern as api_graph.py in-corpus-check).
    Returns {in_corpus, total, missing_ids} fresh on each load.
    """
    conn = open_maps_conn()
    try:
        row = get_map(conn, map_id)
    finally:
        conn.close()

    if row is None:
        raise HTTPException(status_code=404, detail=f"map {map_id!r} not found")

    try:
        snapshot_data: dict[str, Any] = json.loads(row["snapshot"])
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail=f"map snapshot invalid JSON: {exc}") from exc

    raw_corpus_ids: list[Any] = snapshot_data.get("corpus_ids", [])
    corpus_ids = [int(c) for c in raw_corpus_ids]

    if not corpus_ids:
        return {"in_corpus": 0, "total": 0, "missing_ids": []}

    # Resolve in-corpus status via paper_meta (same pattern as api_graph._open_meta_readonly)
    uri = f"file:{_META_DB_PATH}?mode=ro"
    try:
        meta_conn = sqlite3.connect(uri, uri=True)
        meta_conn.row_factory = sqlite3.Row
    except sqlite3.OperationalError as exc:
        raise HTTPException(status_code=503, detail="paper_meta store unavailable") from exc

    try:
        placeholders = ",".join("?" * len(corpus_ids))
        rows = meta_conn.execute(
            f"SELECT corpusid FROM paper_meta WHERE corpusid IN ({placeholders})",
            corpus_ids,
        ).fetchall()
    finally:
        meta_conn.close()

    present: set[int] = {int(r["corpusid"]) for r in rows}
    in_corpus_map: dict[int, bool] = {cid: (cid in present) for cid in corpus_ids}

    return compute_coverage(corpus_ids, in_corpus_map)


@router.delete("/maps/{map_id}", status_code=204)
def delete_map_endpoint(map_id: str) -> None:
    """Delete a saved map. 404 if not found."""
    conn = open_maps_conn()
    try:
        deleted = delete_map(conn, map_id)
    finally:
        conn.close()

    if not deleted:
        raise HTTPException(status_code=404, detail=f"map {map_id!r} not found")
