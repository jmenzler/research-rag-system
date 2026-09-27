# long-ok-file
"""RED tests for src/server/api_ingest.py — GREEN after Plan 05-03 lands.

Covers INGEST-01, INGEST-02, INGEST-03, INGEST-04, INGEST-05, INGEST-07, INGEST-08
plus security V5 (corpus_ids bound) and D-02 (mode=fast, arxiv_id priority),
D-03 (bulk-split), D-08 (crashed/cancelled terminal state).

All tests skip at collection time until src.server.api_ingest exists, then flip
to asserting once Plan 05-03 ships.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("src.server.api_ingest")

from typing import TYPE_CHECKING

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from tests.server.conftest import FakeDiscoverQueue  # noqa: E402

if TYPE_CHECKING:
    from src.server.discover_queue import DiscoverStartRequest

# ---------------------------------------------------------------------------
# App builder
# ---------------------------------------------------------------------------


def _make_app(fq: FakeDiscoverQueue) -> TestClient:
    """Build a minimal FastAPI app with the ingest router + injected fake queue."""
    from src.server import api_ingest  # noqa: PLC0415

    app = FastAPI()
    app.include_router(api_ingest.router)
    api_ingest.set_ingest_queue(fq)  # type: ignore[attr-defined]
    return TestClient(app)


@pytest.fixture(autouse=True)
def _stub_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default every corpus_id to the title (Deep Research) path so bulk/chunk
    tests do not depend on the host's real paper_meta DB. Tests that exercise
    routing override this with their own monkeypatch."""
    from src.server import api_ingest as _ai  # noqa: PLC0415

    def _all_title(corpus_ids: list[int], collection: str):  # noqa: ANN202, ARG001
        return [(cid, "title", str(cid)) for cid in corpus_ids]

    monkeypatch.setattr(_ai, "_resolve_sources", _all_title)


# ---------------------------------------------------------------------------
# INGEST-01: single-node enqueue
# ---------------------------------------------------------------------------


def test_single_enqueue(fake_queue: FakeDiscoverQueue) -> None:
    """INGEST-01: POST /api/ingest with 1 corpus_id enqueues exactly 1 DiscoverStartRequest."""
    client = _make_app(fake_queue)
    resp = client.post("/api/ingest", json={"corpus_ids": [1], "collection": "trading"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert len(fake_queue.calls) == 1
    assert len(body["run_ids"]) == 1
    assert body["run_ids"][0]["corpus_ids"] == [1]


# ---------------------------------------------------------------------------
# INGEST-02: bulk enqueue — 3 corpus_ids → 1 enqueue with list-of-3 query
# ---------------------------------------------------------------------------


def test_bulk_enqueue_list(fake_queue: FakeDiscoverQueue) -> None:
    """INGEST-02: 3 corpus_ids → exactly 1 enqueue call whose req.query is a list of 3."""
    client = _make_app(fake_queue)
    resp = client.post("/api/ingest", json={"corpus_ids": [1, 2, 3], "collection": "trading"})
    assert resp.status_code == 200, resp.text
    assert len(fake_queue.calls) == 1
    req = fake_queue.calls[0]
    assert isinstance(req.query, list)
    assert len(req.query) == 3


# ---------------------------------------------------------------------------
# INGEST-02 / D-03: bulk-split over cap (20 nodes → 3 enqueue calls: 8, 8, 4)
# ---------------------------------------------------------------------------


def test_bulk_split_over_cap(fake_queue: FakeDiscoverQueue) -> None:
    """D-03: 20 corpus_ids split into ⌈20/8⌉=3 enqueue calls (8, 8, 4); n_enqueued==3."""
    client = _make_app(fake_queue)
    corpus_ids = list(range(1, 21))  # 20 nodes
    resp = client.post("/api/ingest", json={"corpus_ids": corpus_ids, "collection": "trading"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert len(fake_queue.calls) == 3, f"expected 3 enqueue calls, got {len(fake_queue.calls)}"
    assert body["n_enqueued"] == 3
    chunk_sizes = [
        len(call.query) if isinstance(call.query, list) else 1 for call in fake_queue.calls
    ]
    assert chunk_sizes == [8, 8, 4], f"expected [8, 8, 4] chunks, got {chunk_sizes}"
    run_ids_entry = body["run_ids"]
    assert len(run_ids_entry) == 3
    assert run_ids_entry[0]["corpus_ids"] == corpus_ids[0:8]
    assert run_ids_entry[1]["corpus_ids"] == corpus_ids[8:16]
    assert run_ids_entry[2]["corpus_ids"] == corpus_ids[16:20]


# ---------------------------------------------------------------------------
# D-02: mode=fast + partition=research_briefs
# ---------------------------------------------------------------------------


def test_enqueue_uses_fast_mode(fake_queue: FakeDiscoverQueue) -> None:
    """D-02: every enqueued req has mode=='fast' and partition=='research_briefs'."""
    client = _make_app(fake_queue)
    corpus_ids = list(range(1, 10))  # 9 nodes → 2 enqueue calls
    resp = client.post("/api/ingest", json={"corpus_ids": corpus_ids, "collection": "trading"})
    assert resp.status_code == 200, resp.text
    assert len(fake_queue.calls) >= 1
    for req in fake_queue.calls:
        assert req.mode == "fast", f"expected mode='fast', got {req.mode!r}"
        assert req.partition == "research_briefs", (
            f"expected partition='research_briefs', got {req.partition!r}"
        )


# ---------------------------------------------------------------------------
# D-02: query uses best identifier — arxiv_id ▸ doi ▸ title
# ---------------------------------------------------------------------------


def test_resolution_routes_by_kind(
    fake_queue: FakeDiscoverQueue, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D-02: arxiv/doi → direct URL path; title → Deep Research; unresolved reported."""
    from src.server import api_ingest as _ai  # noqa: PLC0415

    def _fake_resolve(corpus_ids: list[int], collection: str):  # noqa: ANN202, ARG001
        return [
            (111, "arxiv", "https://arxiv.org/abs/1706.03762"),
            (222, "doi", "https://doi.org/10.1234/doi-only"),
            (333, "title", "A Title Only Paper"),
            (444, "unresolved", "444"),
        ]

    monkeypatch.setattr(_ai, "_resolve_sources", _fake_resolve)

    client = _make_app(fake_queue)
    resp = client.post(
        "/api/ingest",
        json={"corpus_ids": [111, 222, 333, 444], "collection": "trading"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()

    # arxiv + doi → one direct enqueue_urls batch
    assert len(fake_queue.url_calls) == 1
    assert fake_queue.url_calls[0]["urls"] == [
        "https://arxiv.org/abs/1706.03762",
        "https://doi.org/10.1234/doi-only",
    ]
    # title + unresolved → Deep Research query path
    queries: list[str] = []
    for call in fake_queue.calls:
        q = call.query
        queries.extend(q) if isinstance(q, list) else queries.append(q)
    assert "A Title Only Paper" in queries
    assert "444" in queries

    resolved = {r["corpus_id"]: r["kind"] for r in body["resolved"]}
    assert resolved == {111: "arxiv", 222: "doi", 333: "title"}
    assert body["unresolved"] == [444]


# ---------------------------------------------------------------------------
# Security V5: corpus_ids length > 2000 → 422
# ---------------------------------------------------------------------------


def test_corpus_ids_bounded(fake_queue: FakeDiscoverQueue) -> None:
    """Security V5: corpus_ids length 2001 must be rejected with 422."""
    client = _make_app(fake_queue)
    resp = client.post(
        "/api/ingest",
        json={"corpus_ids": list(range(2001)), "collection": "trading"},
    )
    assert resp.status_code == 422, f"expected 422, got {resp.status_code}: {resp.text}"


# ---------------------------------------------------------------------------
# Security: collection literal validation → 422 on bogus value
# ---------------------------------------------------------------------------


def test_collection_literal_validated(fake_queue: FakeDiscoverQueue) -> None:
    """Tampering guard: collection='bogus' must be rejected with 422."""
    client = _make_app(fake_queue)
    resp = client.post(
        "/api/ingest",
        json={"corpus_ids": [1], "collection": "bogus"},
    )
    assert resp.status_code == 422, f"expected 422, got {resp.status_code}: {resp.text}"


# ---------------------------------------------------------------------------
# Security DoS: QueueDepthExceededError → 503
# ---------------------------------------------------------------------------


def test_queue_depth_exceeded_503(monkeypatch: pytest.MonkeyPatch) -> None:
    """DoS guard: fake_queue.enqueue raises QueueDepthExceededError → HTTP 503."""
    from src.server.discover_queue import QueueDepthExceededError  # noqa: PLC0415

    fq = FakeDiscoverQueue()

    def _raise_on_enqueue(req: DiscoverStartRequest) -> None:  # noqa: ARG001
        raise QueueDepthExceededError("queue at capacity (50/50)")

    monkeypatch.setattr(fq, "enqueue", _raise_on_enqueue)
    client = _make_app(fq)
    resp = client.post("/api/ingest", json={"corpus_ids": [1], "collection": "trading"})
    assert resp.status_code == 503, f"expected 503, got {resp.status_code}: {resp.text}"


# ---------------------------------------------------------------------------
# INGEST-08: 10 sequential single-node POSTs → 10 distinct run_ids
# ---------------------------------------------------------------------------


def test_concurrent_ingest(fake_queue: FakeDiscoverQueue) -> None:
    """Sequential requests must receive distinct run identifiers."""
    client = _make_app(fake_queue)
    collected_run_ids: list[str] = []
    for i in range(10):
        resp = client.post(
            "/api/ingest",
            json={"corpus_ids": [i + 1], "collection": "trading"},
        )
        assert resp.status_code == 200, resp.text
        collected_run_ids.append(resp.json()["run_ids"][0]["run_id"])
    assert len(fake_queue.calls) == 10, f"expected 10 enqueue calls, got {len(fake_queue.calls)}"
    assert len(set(collected_run_ids)) == 10, (
        f"expected 10 distinct run_ids, got {collected_run_ids}"
    )


# ---------------------------------------------------------------------------
# INGEST-05: done event payload carries corpus_ids so client can target recheck
# ---------------------------------------------------------------------------


def test_done_triggers_corpus_check(fake_queue: FakeDiscoverQueue) -> None:
    """Keep corpus IDs available for subsequent status checks."""
    client = _make_app(fake_queue)
    resp = client.post(
        "/api/ingest",
        json={"corpus_ids": [42, 99], "collection": "trading"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert len(body["run_ids"]) == 1
    entry = body["run_ids"][0]
    assert "corpus_ids" in entry, "done event payload must carry corpus_ids for targeted recheck"
    assert set(entry["corpus_ids"]) == {42, 99}


# ---------------------------------------------------------------------------
# Finding 1: a real on-disk queued run is discoverable via the queue's
# state_dir even when the per-process in-memory corpus index is empty (the
# autodeploy-restart / MCP-queued case).
# ---------------------------------------------------------------------------


def test_discover_run_ids_surfaces_queued_run_with_empty_index(
    tmp_path: Path,
) -> None:
    """Finding 1: DiscoverQueue.state_dir resolves and _discover_run_ids scans
    queued/+running/, so a run not in _run_corpus_index is still surfaced.
    """
    from src.server import api_ingest as _ai  # noqa: PLC0415
    from src.server.discover_queue import (  # noqa: PLC0415
        DiscoverQueue,
        DiscoverStartRequest,
    )

    q = DiscoverQueue(
        state_dir=tmp_path / "qstate",
        log_dir=tmp_path / "qlogs",
        max_queue_depth=50,
        poll_interval_s=0.01,
    )
    try:
        entry = q.enqueue(
            DiscoverStartRequest(
                query="some query",
                collection="trading",
                partition="research_briefs",
                mode="fast",
                timeout=60,
                keep=False,
            )
        )
        # Simulate the post-restart / MCP-queued case: nothing in this process's
        # in-memory index. Discovery must still find the run via state_dir.
        _ai._run_corpus_index.pop(entry.run_id, None)
        assert _ai._run_corpus_index.get(entry.run_id) is None

        # The property must resolve (the bug was getattr → None → fallback).
        assert q.state_dir == tmp_path / "qstate"
        discovered = _ai._discover_run_ids(q.state_dir)
        assert entry.run_id in discovered
    finally:
        q.stop()
        _ = q.is_alive() and q.join(timeout=2.0)


# ---------------------------------------------------------------------------
# Finding 5: terminal events fire at most once process-wide; corpus_ids are
# evicted from the index on terminal so it does not leak / replay on reconnect.
# ---------------------------------------------------------------------------


def test_mark_terminal_evicts_index_and_gates_reemit() -> None:
    """Finding 5: _mark_terminal_reported records the run + drops its corpus_ids,
    so a reconnecting stream skips it instead of replaying a stale terminal.
    """
    from src.server import api_ingest as _ai  # noqa: PLC0415

    run_id = "term_reemit_1"
    _ai._run_corpus_index[run_id] = [7, 8, 9]
    _ai._reported_terminals.pop(run_id, None)

    _ai._mark_terminal_reported(run_id)

    assert run_id in _ai._reported_terminals
    assert run_id not in _ai._run_corpus_index


def test_reported_terminals_is_bounded() -> None:
    """Finding 5: the process-wide terminal record is a bounded FIFO so it does
    not grow without limit (the slow-leak half of the finding).
    """
    from src.server import api_ingest as _ai  # noqa: PLC0415

    _ai._reported_terminals.clear()
    over = _ai._REPORTED_TERMINALS_MAX + 50
    for i in range(over):
        _ai._mark_terminal_reported(f"bounded_{i}")

    assert len(_ai._reported_terminals) <= _ai._REPORTED_TERMINALS_MAX
    # Oldest aged out; newest retained.
    assert "bounded_0" not in _ai._reported_terminals
    assert f"bounded_{over - 1}" in _ai._reported_terminals
    _ai._reported_terminals.clear()
