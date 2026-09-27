"""MCP transport forwarding retrieval and notebook requests to RAG_API_URL."""

from __future__ import annotations

import os
from typing import Any, Literal

import httpx
from mcp.server.fastmcp import FastMCP

RAG_API_URL = os.getenv("RAG_API_URL", "http://127.0.0.1:8766")
HTTP_TIMEOUT = float(os.getenv("RAG_API_TIMEOUT", "330"))

_transport_raw = os.getenv("MCP_TRANSPORT", "stdio")
_transport: Literal["stdio", "sse", "streamable-http"] = (
    _transport_raw  # type: ignore[assignment]
    if _transport_raw in ("stdio", "sse", "streamable-http")
    else "stdio"
)
mcp = (
    FastMCP(
        "rag-system",
        host=os.getenv("MCP_HOST", "127.0.0.1"),
        port=int(os.getenv("MCP_PORT", "8765")),
    )
    if _transport == "sse"
    else FastMCP("rag-system")
)


def _post(path: str, body: dict[str, Any]) -> dict[str, Any]:
    """POST to the PC HTTP API and return parsed JSON, or raise with detail."""
    with httpx.Client(timeout=HTTP_TIMEOUT) as h:
        r = h.post(f"{RAG_API_URL}{path}", json=body)
    if r.status_code != 200:
        raise RuntimeError(
            f"PC API {path} -> {r.status_code}: {r.text[:1000]}"
        )
    return r.json()  # type: ignore[no-any-return]


def _get(path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    with httpx.Client(timeout=HTTP_TIMEOUT) as h:
        r = h.get(f"{RAG_API_URL}{path}", params=params or {})
    if r.status_code != 200:
        raise RuntimeError(
            f"PC API {path} -> {r.status_code}: {r.text[:1000]}"
        )
    return r.json()  # type: ignore[no-any-return]


@mcp.tool()
def rag_retrieve(
    query: str,
    collection: str = "trading",
    notebook: str | None = None,
    decompose: bool = False,
    synthesize: bool = False,
    model: str | None = None,
) -> dict[str, Any]:
    """Retrieve parent chunks from the RAG corpus.

    Set ``synthesize=True`` for an LLM-written answer with citations;
    leave it False to get raw chunks. Scope via ``collection`` (trading |
    ecology | notes | system) and optional ``notebook`` partition tag.
    """
    return _post(
        "/query",
        {
            "query": query,
            "collection": collection,
            "notebook": notebook,
            "decompose": decompose,
            "synthesize": synthesize,
            "model": model,
        },
    )


@mcp.tool()
def list_notebooks(collection: str | None = None) -> dict[str, Any]:
    """List collections and their notebooks with chunk counts.

    Use to inform ``rag_retrieve`` collection/notebook args.
    """
    params = {"collection": collection} if collection else None
    return _get("/notebooks", params=params)


# ---------------------------------------------------------------------------
# Discover tools — async runner over the PC's NotebookLM Deep Research +
# fetch + ingest pipeline. Runs are long (minutes to ~1h), so the surface
# is split: kick off, poll, optionally cancel.
# ---------------------------------------------------------------------------


@mcp.tool()
def research_and_ingest(
    query: str | list[str],
    collection: str = "trading",
    partition: str = "research_briefs",
    mode: str = "deep",
    timeout: int = 1800,
    keep: bool = False,
) -> dict[str, Any]:
    """DISCOVER new sources from a natural-language query (NotebookLM Deep
    Research), then ingest them into the RAG.

    Use this ONLY when you do not yet know which papers you want. If you
    already have Semantic Scholar corpus_ids (e.g. from a citation-graph
    walk) use ``ingest_papers``; if you have direct paper URLs use
    ``ingest_url`` — both skip Deep Research and are faster + deterministic.

    Async — returns a ``run_id`` immediately; poll with
    ``research_and_ingest_status``. Use ``mode="fast"`` for ~1 min,
    ``mode="deep"`` for ~5-10 min.

    ``query`` accepts a single string OR a list of strings. Passing a list
    runs all queries' Deep Research jobs in parallel (capped by
    ``MAX_PARALLEL_DR``) and folds the results into a SINGLE fetch + parse +
    ingest pass — recommended over firing N separate calls because it
    eliminates per-notebook MinerU/GPU warmup and the lock contention that
    used to drop concurrent runs.

    Server-side queue: rapid-fire calls (single or bulk) enqueue safely on
    disk and drain one at a time; you never lose work even if you call this
    tool faster than it can finish.

    ``collection`` is the top-level Milvus index (trading | ecology | notes |
    system). ``partition`` is the within-collection tag for filtered retrieval
    via ``rag_retrieve(notebook=...)``. Default ``research_briefs`` keeps new
    discoveries in a side partition; pass the collection name (e.g.
    ``partition="trading"``) to ingest into the main partition so the chunks
    appear in default unfiltered queries.
    """
    return _post(
        "/discover/start",
        {
            "query": query,
            "collection": collection,
            "partition": partition,
            "mode": mode,
            "timeout": timeout,
            "keep": keep,
        },
    )


@mcp.tool()
def research_and_ingest_status(
    run_id: str, log_tail_lines: int = 30,
) -> dict[str, Any]:
    """Poll a research_and_ingest run. State: running | done | crashed | unknown."""
    return _get(
        "/discover/status",
        params={"run_id": run_id, "log_tail_lines": log_tail_lines},
    )


@mcp.tool()
def research_and_ingest_cancel(run_id: str) -> dict[str, Any]:
    """SIGTERM a running research_and_ingest run."""
    return _post("/discover/cancel", {"run_id": run_id})


@mcp.tool()
def ingest_papers(
    corpus_ids: list[int] | int,
    collection: str = "trading",
    partition: str = "research_briefs",
) -> dict[str, Any]:
    """Ingest specific papers you ALREADY have by Semantic Scholar corpus_id.

    The natural pairing for citation-graph-traversal: when a walk surfaces a
    noteworthy paper, pass its ``corpus_id`` here to add it to the corpus.
    (To DISCOVER unknown papers from a query instead, use ``research_and_ingest``.)

    Each corpus_id is resolved to its best identifier (arxiv_id > doi > title),
    then fetched + parsed + ingested via the shared DiscoverQueue — the same queue
    research_and_ingest uses, so calls enqueue safely and drain one at a time.

    Async. The response reports, per corpus_id, which identifier it resolved to
    (``resolved``) and which could not be resolved at all (``unresolved``), and
    splits the work across one or more ``run_ids`` (arxiv/doi go in one direct
    batch; title-only ids are peeled into Deep Research runs). Poll each run_id
    with ``research_and_ingest_status``.

    ``partition`` defaults to ``research_briefs`` (side partition, filtered
    retrieval); pass the collection name (e.g. ``"trading"``) to land chunks in
    the main partition so they show up in default unfiltered queries.
    """
    ids = [corpus_ids] if isinstance(corpus_ids, int) else list(corpus_ids)
    resp = _post(
        "/api/ingest",
        {"corpus_ids": ids, "collection": collection, "partition": partition},
    )
    resp["poll_hint"] = "research_and_ingest_status(run_id=...) for each run_id"
    return resp


@mcp.tool()
def ingest_url(
    sources: str | list[str],
    collection: str = "trading",
    partition: str = "trading",
) -> dict[str, Any]:
    """Ingest specific papers you ALREADY have by direct URL — NO NotebookLM
    Deep Research.

    Use when you hold the paper URLs directly. (Have Semantic Scholar
    corpus_ids instead? Use ``ingest_papers``. Only a topic, not specific
    papers? Use ``research_and_ingest``.) Each source is an http(s) URL
    (arxiv /abs/ or /pdf/, journal/proceedings PDF) or a ``Title || url`` line;
    local file paths are NOT supported (the fetch pipeline is HTTP-only).

    Async + queued via the shared DiscoverQueue. Returns one ``run_id`` for the
    batch; poll it with ``research_and_ingest_status``. partition="trading"
    lands chunks in the main partition (default unfiltered retrieval).
    """
    urls = [sources] if isinstance(sources, str) else list(sources)
    resp = _post(
        "/ingest/urls",
        {"urls": urls, "collection": collection, "partition": partition},
    )
    resp["poll_hint"] = "research_and_ingest_status(run_id=...)"
    return resp


if __name__ == "__main__":
    mcp.run(transport=_transport)
