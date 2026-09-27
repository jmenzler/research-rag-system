# long-ok-file
"""Integration tests for /discover/{start,status,cancel} backed by DiscoverQueue.

Goal: rapid-fire POST /discover/start calls enqueue safely; status reflects
queued → running → done transitions; bulk queries flow through as a list.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.server import api as api_module
from src.server.discover_queue import DiscoverQueue, QueueEntry

# ---------------------------------------------------------------------------
# Test double: same shape as _StubSpawner from test_discover_queue.py.
# Imported separately so the two test files stay decoupled.
# ---------------------------------------------------------------------------


class _StubSpawner:
    def __init__(self) -> None:
        self.calls: list[tuple[QueueEntry, Path, Path]] = []
        self.next_delay_s: float = 0.05
        self._fake_pid = 100_000
        self._alive: set[int] = set()
        self._lock = threading.Lock()

    def spawn(self, entry: QueueEntry, log_path: Path, result_path: Path) -> int:
        with self._lock:
            self._fake_pid += 1
            pid = self._fake_pid
            self._alive.add(pid)
        self.calls.append((entry, log_path, result_path))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            f"stubbed pid={pid} queries={entry.queries}\n",
            encoding="utf-8",
        )

        delay = self.next_delay_s

        def _finish() -> None:
            time.sleep(delay)
            import json

            result_path.write_text(
                json.dumps(
                    {
                        "status": "ok",
                        "queries": entry.queries,
                        "per_query": [
                            {
                                "query": q,
                                "notebook_id": f"nb-{i}",
                                "status": "ok",
                                "n_imported": 1,
                                "n_deduped": 0,
                                "n_kept": 1,
                                "notebook_kept": False,
                                "error": None,
                            }
                            for i, q in enumerate(entry.queries)
                        ],
                        "n_imported": len(entry.queries),
                        "n_deduped": 0,
                        "n_kept": len(entry.queries),
                        "collection": entry.collection,
                        "partition": entry.partition,
                        "mode": entry.mode,
                    }
                ),
                encoding="utf-8",
            )
            with self._lock:
                self._alive.discard(pid)

        threading.Thread(target=_finish, daemon=True).start()
        return pid

    def is_alive(self, pid: int, pid_start: str | None = None) -> bool:
        with self._lock:
            return pid in self._alive

    def terminate(self, pid: int) -> bool:
        with self._lock:
            if pid not in self._alive:
                return False
            self._alive.discard(pid)
        return True


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def discover_queue(tmp_path: Path) -> Iterator[DiscoverQueue]:
    """Replace the api's queue with a worker-started, tmp-path-isolated one."""
    spawner = _StubSpawner()
    queue = DiscoverQueue(
        state_dir=tmp_path / "queue",
        log_dir=tmp_path / "logs",
        spawner=spawner,
        max_queue_depth=50,
        poll_interval_s=0.01,
    )
    api_module.set_discover_queue(queue)
    queue.start()
    try:
        yield queue
    finally:
        queue.stop()
        api_module.set_discover_queue(None)


@pytest.fixture()
def client(discover_queue: DiscoverQueue) -> TestClient:
    del discover_queue  # fixture used for setup side effect
    return TestClient(api_module.app)


def _wait_until_state(
    client: TestClient, run_id: str, target: str | set[str], *, timeout: float = 3.0
) -> dict[str, Any]:
    deadline = time.time() + timeout
    targets = {target} if isinstance(target, str) else target
    last: dict[str, Any] = {}
    while time.time() < deadline:
        r = client.get(f"/discover/status?run_id={run_id}")
        if r.status_code == 200:
            last = r.json()
            if last.get("state") in targets:
                return last
        time.sleep(0.02)
    raise AssertionError(
        f"run_id={run_id} did not reach {targets} within {timeout}s "
        f"(last state: {last.get('state')})"
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_post_start_single_query(client: TestClient) -> None:
    r = client.post(
        "/discover/start",
        json={"query": "what is GLFT", "collection": "trading"},
    )
    assert r.status_code == 200
    body = r.json()
    assert "run_id" in body
    assert body["state"] in ("queued", "running")
    assert "position" in body or body["state"] == "running"


def test_post_start_bulk_query(client: TestClient) -> None:
    r = client.post(
        "/discover/start",
        json={
            "query": ["a", "b", "c"],
            "collection": "trading",
            "partition": "research_briefs",
            "mode": "deep",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["state"] in ("queued", "running")
    assert "run_id" in body


def test_post_start_empty_list_rejected(client: TestClient) -> None:
    r = client.post(
        "/discover/start",
        json={"query": [], "collection": "trading"},
    )
    assert r.status_code == 422


def test_post_start_too_many_queries_rejected(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MAX_QUERIES_PER_REQUEST", "3")
    r = client.post(
        "/discover/start",
        json={"query": ["a", "b", "c", "d"], "collection": "trading"},
    )
    assert r.status_code == 422


def test_status_transitions_to_done(client: TestClient) -> None:
    r = client.post(
        "/discover/start",
        json={"query": "x", "collection": "trading"},
    )
    run_id = r.json()["run_id"]
    final = _wait_until_state(client, run_id, "done", timeout=4.0)
    assert final["state"] == "done"
    assert final["result"]["status"] == "ok"
    assert final["result"]["queries"] == ["x"]


def test_get_status_unknown_returns_404(client: TestClient) -> None:
    r = client.get("/discover/status?run_id=does-not-exist")
    assert r.status_code == 404


def test_cancel_queued_entry(client: TestClient, discover_queue: DiscoverQueue) -> None:
    # Make sure the first entry occupies the worker so the second stays queued.
    r1 = client.post("/discover/start", json={"query": "first", "collection": "trading"})
    r2 = client.post("/discover/start", json={"query": "second", "collection": "trading"})
    rid2 = r2.json()["run_id"]
    del r1
    # Brief wait so worker picks up #1.
    time.sleep(0.02)

    cancel = client.post("/discover/cancel", json={"run_id": rid2})
    assert cancel.status_code == 200

    # Either already cancelled (queued path) or in cancellation flow (running).
    final = _wait_until_state(client, rid2, {"cancelled", "crashed", "done"}, timeout=3.0)
    assert final["state"] in {"cancelled", "crashed", "done"}
    del discover_queue


def test_concurrent_enqueue_no_500s(client: TestClient) -> None:
    """Hammer /discover/start from 5 threads; every call succeeds."""
    n = 5
    results: list[int] = []
    lock = threading.Lock()
    errors: list[str] = []

    def worker(i: int) -> None:
        try:
            r = client.post(
                "/discover/start",
                json={"query": f"q-{i}", "collection": "trading"},
            )
            with lock:
                results.append(r.status_code)
        except Exception as exc:  # noqa: BLE001
            with lock:
                errors.append(repr(exc))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert results == [200] * n
