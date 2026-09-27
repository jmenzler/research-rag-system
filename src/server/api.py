"""HTTP routes for CLI-backed retrieval, discovery, and monitoring."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pymilvus import MilvusClient

from src import config
from src.server.discover_queue import (
    DiscoverQueue,
    DiscoverStartRequest,
    QueueDepthExceededError,
    UnknownRunIDError,
)
from src.server.notebooks import list_notebooks

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAG_BIN = PROJECT_ROOT / "bin" / "rag"
DISCOVER_LOG_DIR = PROJECT_ROOT / "logs" / "discover"

app = FastAPI(title="rag-system query API", version="1.0.0")


class QueryRequest(BaseModel):
    query: str
    collection: str = "trading"
    notebook: str | None = None
    top_k: int = 15
    decompose: bool = False
    synthesize: bool = False
    model: str | None = None


@app.post("/query")
async def query_endpoint(req: QueryRequest) -> dict[str, Any]:
    """Run ``bin/rag --json`` as a subprocess and forward its JSON output.

    The CLI is the canonical pipeline (decompose, stepback, CRAG, retrieval,
    rerank, synthesis, logging). Server only adds HTTP + arg translation.
    """
    cmd: list[str] = [
        str(RAG_BIN),
        "--query", req.query,
        "--collection", req.collection,
        "--top-k-rerank", str(req.top_k),
        "--json",
    ]
    if req.notebook:
        cmd += ["--notebook", req.notebook]
    if not req.decompose:
        cmd += ["--no-decompose"]
    if not req.synthesize:
        cmd += ["--raw"]
    if req.model:
        cmd += ["--model", req.model]

    logger.info("running: %s", " ".join(cmd))
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=PROJECT_ROOT,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=300.0)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        raise HTTPException(status_code=504, detail="bin/rag exceeded 300s timeout")

    if proc.returncode != 0:
        msg = stderr.decode("utf-8", errors="replace")[:2000]
        raise HTTPException(status_code=500, detail=f"bin/rag exit={proc.returncode}: {msg}")

    try:
        return json.loads(stdout.decode("utf-8"))  # type: ignore[no-any-return]
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"bin/rag emitted non-JSON: {exc}; stdout head: {stdout[:500]!r}",
        ) from exc


@app.get("/notebooks")
def notebooks_endpoint(collection: str | None = None) -> dict[str, Any]:
    """List collections + per-notebook chunk counts."""
    return list_notebooks(collection=collection)


# ---------------------------------------------------------------------------
# Discover endpoints — delegated to the persistent DiscoverQueue.
# ---------------------------------------------------------------------------
#
# Discovery is long-running (parallel Deep Research jobs + batched
# fetch+parse+ingest pass; minutes to ~1 hour). The queue serialises runs
# so concurrent /discover/start calls cannot race on the ragctl-run.lock.
# The HTTP layer here only enqueues and reads status; the worker thread
# in DiscoverQueue spawns and tracks the subprocess.

_discover_queue: DiscoverQueue | None = None


def set_discover_queue(queue: DiscoverQueue | None) -> None:
    """Inject (or clear, with None) the DiscoverQueue instance.

    Called once at server boot from __main__.py; tests use it to swap in
    a tmp-path-isolated queue and unset between cases.
    """
    global _discover_queue
    _discover_queue = queue


def get_discover_queue() -> DiscoverQueue | None:
    """Return the DiscoverQueue instance, or None before boot wires it.

    The MonitorScheduler reads this lazily to enqueue full-PDF upgrade jobs
    after a poll, without a hard import-time dependency on server boot order.
    """
    return _discover_queue


def _require_queue() -> DiscoverQueue:
    if _discover_queue is None:
        raise HTTPException(status_code=503, detail="discover queue not initialised")
    return _discover_queue


@app.post("/discover/start")
def discover_start(req: DiscoverStartRequest) -> dict[str, Any]:
    """Enqueue a discover run. Returns immediately with run_id + state."""
    queue = _require_queue()
    try:
        entry = queue.enqueue(req)
    except QueueDepthExceededError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    state = queue.get_state(entry.run_id)
    logger.info(
        "discover/start: run_id=%s queries=%s collection=%s",
        entry.run_id, entry.queries, entry.collection,
    )
    return {
        "run_id": entry.run_id,
        "state": state["state"],
        "position": state.get("position"),
        "queue_depth": state.get("queue_depth"),
        "started_at": entry.enqueued_at,
        "log_path": entry.log_path,
    }


class IngestUrlsRequest(BaseModel):
    urls: list[str] = Field(..., max_length=100)
    collection: str = "trading"
    partition: str = "trading"


@app.post("/ingest/urls")
def ingest_urls(req: IngestUrlsRequest) -> dict[str, Any]:
    """Enqueue a direct URL-list ingest (no NotebookLM). Returns run_id + state.

    Reuses /discover/status + /discover/cancel for polling and cancellation
    (same run_id namespace).
    """
    queue = _require_queue()
    try:
        entry = queue.enqueue_urls(
            urls=req.urls, collection=req.collection, partition=req.partition
        )
    except QueueDepthExceededError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    state = queue.get_state(entry.run_id)
    logger.info(
        "ingest/urls: run_id=%s n_urls=%d collection=%s",
        entry.run_id, len(entry.queries), entry.collection,
    )
    return {
        "run_id": entry.run_id,
        "state": state["state"],
        "position": state.get("position"),
        "queue_depth": state.get("queue_depth"),
        "started_at": entry.enqueued_at,
        "log_path": entry.log_path,
    }


@app.get("/discover/status")
def discover_status(run_id: str, log_tail_lines: int = 30) -> dict[str, Any]:
    """Poll a discover run. State: queued | running | done | crashed | cancelled."""
    del log_tail_lines  # log_tail length is fixed at 30 in the queue
    queue = _require_queue()
    try:
        return queue.get_state(run_id)
    except UnknownRunIDError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


class DiscoverCancelRequest(BaseModel):
    run_id: str


@app.post("/discover/cancel")
def discover_cancel(req: DiscoverCancelRequest) -> dict[str, Any]:
    """Cancel a queued or running discover run."""
    queue = _require_queue()
    try:
        return queue.cancel(req.run_id)
    except UnknownRunIDError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/health")
def health_endpoint() -> dict[str, Any]:
    """Liveness probe — checks Milvus and the reranker."""
    status: dict[str, Any] = {"status": "ok"}
    failures: list[str] = []

    try:
        client = MilvusClient(uri=config.MILVUS_URI, timeout=2)
        _ = client.list_collections()
        status["milvus"] = "ok"
    except Exception as exc:
        status["milvus"] = f"fail: {exc!r}"
        failures.append("milvus")

    backend = os.getenv("RERANK_BACKEND", "remote_llamacpp").lower()
    if backend == "remote_nvidia":
        # Hosted NIM endpoint — a liveness poll must not issue a billed external
        # call, so verify the credential is present instead of probing.
        if os.getenv("NVIDIA_API_KEY"):
            status["reranker"] = "ok (remote_nvidia)"
        else:
            status["reranker"] = "fail: NVIDIA_API_KEY unset"
            failures.append("reranker")
    else:
        rerank_url = os.getenv("RERANK_REMOTE_URL", "http://127.0.0.1:8090")
        try:
            with httpx.Client(timeout=2.0) as h:
                r = h.post(
                    f"{rerank_url}/v1/rerank",
                    json={"query": "ping", "documents": ["pong"]},
                )
                r.raise_for_status()
            status["reranker"] = "ok"
        except Exception as exc:
            status["reranker"] = f"fail: {exc!r}"
            failures.append("reranker")

    if failures:
        status["status"] = "degraded"
        raise HTTPException(status_code=503, detail=status)

    return status


# ---------------------------------------------------------------------------
# Monitor endpoints
# ---------------------------------------------------------------------------
#
# MonitorScheduler is started in __main__.py and stored in _monitor_scheduler.
# api.py references it via get_monitor_scheduler() to avoid a circular import.

from src.monitor.scheduler import MonitorScheduler as _MonitorScheduler  # noqa: E402

_monitor_scheduler: _MonitorScheduler | None = None


def set_monitor_scheduler(scheduler: _MonitorScheduler) -> None:
    """Called once at server boot to inject the running MonitorScheduler."""
    global _monitor_scheduler
    _monitor_scheduler = scheduler


@app.post("/monitor/trigger")
def monitor_trigger() -> dict[str, Any]:
    """Fire a poll cycle immediately. Returns 409 if already running."""
    if _monitor_scheduler is None:
        raise HTTPException(status_code=503, detail="monitor not initialised")
    triggered = _monitor_scheduler.trigger()
    if not triggered:
        raise HTTPException(status_code=409, detail="poll already running")
    return {"status": "triggered", "started_at": time.time()}


@app.get("/monitor/status")
def monitor_status() -> dict[str, Any]:
    """Return last poll run result and current running state."""
    if _monitor_scheduler is None:
        raise HTTPException(status_code=503, detail="monitor not initialised")
    return _monitor_scheduler.status()
