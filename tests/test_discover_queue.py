# long-ok-file
"""Tests for src.server.discover_queue — persistent FIFO + single worker.

The queue serialises `research_and_ingest` runs so concurrent MCP calls cannot
race on the ragctl-run.lock. Disk-backed (survives uvicorn restart). Single
worker thread mirrors the MonitorScheduler pattern.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from src.server.discover_queue import (
    DiscoverQueue,
    DiscoverStartRequest,
    QueueDepthExceededError,
    QueueEntry,
    UnknownRunIDError,
    _pid_start_token,
)

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _StubSpawner:
    """Records spawn calls and optionally writes a result.json after a delay.

    Matches the SubprocessSpawner protocol the queue depends on. Tests inject
    this so no real subprocesses fire.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[QueueEntry, Path, Path]] = []
        self.next_result: dict[str, Any] | None = {
            "status": "ok",
            "queries": ["x"],
            "per_query": [],
        }
        self.next_delay_s: float = 0.05
        self.next_crash: bool = False
        # When True, terminate() records the SIGTERM but the pid stays "alive"
        # — simulates a process (e.g. ragctl's mp parent blocked on Queue.get)
        # that ignores SIGTERM. force_kill() always kills regardless.
        self.terminate_ignored: bool = False
        self._fake_pid_counter = 100_000
        self._alive_pids: set[int] = set()
        self._lock = threading.Lock()
        self.terminated_pids: list[int] = []
        self.force_killed_pids: list[int] = []
        self.parallel_max = 0
        self.parallel_now = 0
        self._parallel_lock = threading.Lock()

    def spawn(self, entry: QueueEntry, log_path: Path, result_path: Path) -> int:
        with self._lock:
            self._fake_pid_counter += 1
            pid = self._fake_pid_counter
            self._alive_pids.add(pid)
        self.calls.append((entry, log_path, result_path))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(f"spawn pid={pid} queries={entry.queries}\n", encoding="utf-8")

        with self._parallel_lock:
            self.parallel_now += 1
            if self.parallel_now > self.parallel_max:
                self.parallel_max = self.parallel_now

        result = None if self.next_crash else dict(self.next_result or {})
        delay = self.next_delay_s
        crash = self.next_crash

        def _finish() -> None:
            time.sleep(delay)
            if not crash and result is not None:
                result_path.write_text(json.dumps(result), encoding="utf-8")
            with self._lock:
                self._alive_pids.discard(pid)
            with self._parallel_lock:
                self.parallel_now -= 1

        threading.Thread(target=_finish, daemon=True).start()
        return pid

    def is_alive(self, pid: int, pid_start: str | None = None) -> bool:
        with self._lock:
            if pid in self._alive_pids:
                return True
        # Pids we didn't spawn (reconcile tests pass real os pids) — defer to OS.
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def terminate(self, pid: int) -> bool:
        with self._lock:
            if pid not in self._alive_pids:
                return False
            self.terminated_pids.append(pid)
            if not self.terminate_ignored:
                self._alive_pids.discard(pid)
                with self._parallel_lock:
                    self.parallel_now -= 1
        return True

    def force_kill(self, pid: int) -> bool:
        with self._lock:
            was_alive = pid in self._alive_pids
            if was_alive:
                self._alive_pids.discard(pid)
                self.force_killed_pids.append(pid)
        if was_alive:
            with self._parallel_lock:
                self.parallel_now -= 1
        return was_alive


def _make_req(queries: str | list[str] = "anything") -> DiscoverStartRequest:
    return DiscoverStartRequest(
        query=queries,
        collection="trading",
        partition="research_briefs",
        mode="deep",
        timeout=60,
        keep=False,
    )


def _wait_until(
    predicate: Callable[[], bool], *, timeout: float = 3.0, interval: float = 0.02
) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if predicate():
                return True
        except UnknownRunIDError:
            # Transient: the entry is mid-rename between queued/running/done.
            # Retry — the worker is moving forward, will land in done/ shortly.
            pass
        time.sleep(interval)
    return False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def queue_dirs(tmp_path: Path) -> tuple[Path, Path]:
    state_dir = tmp_path / "discover-queue"
    log_dir = tmp_path / "logs"
    return state_dir, log_dir


@pytest.fixture()
def spawner() -> _StubSpawner:
    return _StubSpawner()


@pytest.fixture()
def queue(queue_dirs: tuple[Path, Path], spawner: _StubSpawner) -> Iterator[DiscoverQueue]:
    state_dir, log_dir = queue_dirs
    q = DiscoverQueue(
        state_dir=state_dir,
        log_dir=log_dir,
        spawner=spawner,
        max_queue_depth=50,
        poll_interval_s=0.01,
    )
    yield q
    q.stop()
    _ = q.is_alive() and q.join(timeout=2.0)


# ---------------------------------------------------------------------------
# Test cases (numbered per plan)
# ---------------------------------------------------------------------------


def test_1_enqueue_writes_queued_file(queue: DiscoverQueue, queue_dirs: tuple[Path, Path]) -> None:
    state_dir, _ = queue_dirs
    entry = queue.enqueue(_make_req(["foo", "bar"]))

    assert entry.run_id
    assert entry.queries == ["foo", "bar"]

    queued_files = list((state_dir / "queued").iterdir())
    assert len(queued_files) == 1
    payload = json.loads(queued_files[0].read_text(encoding="utf-8"))
    assert payload["run_id"] == entry.run_id
    assert payload["queries"] == ["foo", "bar"]
    assert payload["collection"] == "trading"


def test_2_fifo_position_ordering(queue: DiscoverQueue) -> None:
    a = queue.enqueue(_make_req("a"))
    b = queue.enqueue(_make_req("b"))
    c = queue.enqueue(_make_req("c"))

    sa = queue.get_state(a.run_id)
    sb = queue.get_state(b.run_id)
    sc = queue.get_state(c.run_id)

    assert sa["state"] == "queued" and sa["position"] == 0
    assert sb["state"] == "queued" and sb["position"] == 1
    assert sc["state"] == "queued" and sc["position"] == 2


def test_3_cancel_queued_entry(queue: DiscoverQueue, queue_dirs: tuple[Path, Path]) -> None:
    state_dir, _ = queue_dirs
    entry = queue.enqueue(_make_req("x"))
    expected = state_dir / "queued" / f"{entry.counter:05d}_{entry.run_id}.json"
    assert expected.is_file()

    result = queue.cancel(entry.run_id)
    assert result["state"] == "cancelled"

    assert not list((state_dir / "queued").glob(f"*_{entry.run_id}.json"))
    done_files = list((state_dir / "done").glob(f"*_{entry.run_id}.json"))
    assert len(done_files) == 1

    state = queue.get_state(entry.run_id)
    assert state["state"] == "cancelled"
    assert state["result"]["status"] == "cancelled"


def test_4_unknown_run_id_raises(queue: DiscoverQueue) -> None:
    with pytest.raises(UnknownRunIDError):
        queue.get_state("does-not-exist")


def test_5_reject_when_depth_exceeded(queue_dirs: tuple[Path, Path], spawner: _StubSpawner) -> None:
    state_dir, log_dir = queue_dirs
    q = DiscoverQueue(
        state_dir=state_dir,
        log_dir=log_dir,
        spawner=spawner,
        max_queue_depth=2,
        poll_interval_s=0.01,
    )
    try:
        q.enqueue(_make_req("a"))
        q.enqueue(_make_req("b"))
        with pytest.raises(QueueDepthExceededError):
            q.enqueue(_make_req("c"))
    finally:
        q.stop()
        _ = q.is_alive() and q.join(timeout=2.0)


def test_6_worker_processes_in_order(
    queue: DiscoverQueue, spawner: _StubSpawner, queue_dirs: tuple[Path, Path]
) -> None:
    state_dir, _ = queue_dirs
    spawner.next_delay_s = 0.10

    a = queue.enqueue(_make_req("a"))
    b = queue.enqueue(_make_req("b"))
    queue.start()

    assert _wait_until(
        lambda: (
            queue.get_state(a.run_id)["state"] == "done"
            and queue.get_state(b.run_id)["state"] == "done"
        ),
        timeout=4.0,
    ), "both entries should reach done"

    assert spawner.parallel_max == 1
    assert len(list((state_dir / "done").iterdir())) == 2
    assert len(list((state_dir / "queued").iterdir())) == 0
    assert len(list((state_dir / "running").iterdir())) == 0
    assert [c[0].run_id for c in spawner.calls] == [a.run_id, b.run_id]


def test_7_worker_crash_marks_crashed_and_continues(
    queue: DiscoverQueue, spawner: _StubSpawner
) -> None:
    spawner.next_crash = True
    spawner.next_delay_s = 0.05
    a = queue.enqueue(_make_req("a"))

    queue.start()
    assert _wait_until(lambda: queue.get_state(a.run_id)["state"] == "crashed", timeout=4.0)

    spawner.next_crash = False
    spawner.next_result = {"status": "ok", "queries": ["b"], "per_query": []}
    b = queue.enqueue(_make_req("b"))
    assert _wait_until(lambda: queue.get_state(b.run_id)["state"] == "done", timeout=4.0)


def test_8_reconcile_dead_running_entry(
    queue_dirs: tuple[Path, Path], spawner: _StubSpawner
) -> None:
    state_dir, log_dir = queue_dirs
    running_dir = state_dir / "running"
    running_dir.mkdir(parents=True)

    fake_entry = {
        "run_id": "deadbeef",
        "counter": 1,
        "queries": ["q"],
        "collection": "trading",
        "partition": "research_briefs",
        "mode": "deep",
        "timeout": 60,
        "keep": False,
        "enqueued_at": time.time(),
        "pid": 999_999_999,
        "log_path": str(log_dir / "deadbeef.log"),
        "result_path": str(log_dir / "deadbeef.result.json"),
    }
    (running_dir / "00001_deadbeef.json").write_text(json.dumps(fake_entry), encoding="utf-8")

    q = DiscoverQueue(
        state_dir=state_dir,
        log_dir=log_dir,
        spawner=spawner,
        max_queue_depth=50,
        poll_interval_s=0.01,
    )
    try:
        q.reconcile_on_startup()
        state = q.get_state("deadbeef")
        assert state["state"] == "crashed"
        assert len(list((state_dir / "running").iterdir())) == 0
        assert len(list((state_dir / "done").iterdir())) == 1
    finally:
        q.stop()
        _ = q.is_alive() and q.join(timeout=2.0)


def test_9_reconcile_live_running_entry_left_alone(
    queue_dirs: tuple[Path, Path], spawner: _StubSpawner
) -> None:
    state_dir, log_dir = queue_dirs
    running_dir = state_dir / "running"
    running_dir.mkdir(parents=True)

    live_pid = os.getpid()
    fake_entry = {
        "run_id": "livepid1",
        "counter": 1,
        "queries": ["q"],
        "collection": "trading",
        "partition": "research_briefs",
        "mode": "deep",
        "timeout": 60,
        "keep": False,
        "enqueued_at": time.time(),
        "pid": live_pid,
        "log_path": str(log_dir / "livepid1.log"),
        "result_path": str(log_dir / "livepid1.result.json"),
    }
    (running_dir / "00001_livepid1.json").write_text(json.dumps(fake_entry), encoding="utf-8")

    q = DiscoverQueue(
        state_dir=state_dir,
        log_dir=log_dir,
        spawner=spawner,
        max_queue_depth=50,
        poll_interval_s=0.01,
    )
    try:
        q.reconcile_on_startup()
        assert (running_dir / "00001_livepid1.json").is_file()
        state = q.get_state("livepid1")
        assert state["state"] == "running"
    finally:
        q.stop()
        _ = q.is_alive() and q.join(timeout=2.0)


def test_10_concurrent_enqueue_distinct_counters(queue: DiscoverQueue) -> None:
    n = 10
    entries: list[QueueEntry] = []
    lock = threading.Lock()

    def worker() -> None:
        e = queue.enqueue(_make_req("c"))
        with lock:
            entries.append(e)

    threads = [threading.Thread(target=worker) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    counters = [e.counter for e in entries]
    run_ids = [e.run_id for e in entries]
    assert len(set(counters)) == n
    assert len(set(run_ids)) == n


def test_cancel_running_terminates(queue: DiscoverQueue, spawner: _StubSpawner) -> None:
    spawner.next_delay_s = 1.0  # spawner won't finish on its own
    a = queue.enqueue(_make_req("slow"))
    queue.start()

    assert _wait_until(lambda: queue.get_state(a.run_id)["state"] == "running", timeout=2.0)

    result = queue.cancel(a.run_id)
    assert result["killed"] is True
    assert len(spawner.terminated_pids) == 1

    assert _wait_until(
        lambda: queue.get_state(a.run_id)["state"] in ("cancelled", "crashed"),
        timeout=4.0,
    )


def test_empty_query_list_rejected() -> None:
    with pytest.raises(ValueError):
        DiscoverStartRequest(
            query=[],
            collection="trading",
            partition="research_briefs",
            mode="deep",
            timeout=60,
            keep=False,
        )


def test_str_query_normalised_to_list() -> None:
    req = _make_req("just one")
    assert req.queries == ["just one"]


def test_cancel_running_no_escalation_when_sigterm_works(
    queue: DiscoverQueue, spawner: _StubSpawner
) -> None:
    """Normal path: terminate() kills the pid, no SIGKILL escalation."""
    spawner.next_delay_s = 1.0  # spawner won't finish on its own
    a = queue.enqueue(_make_req("slow"))
    queue.start()
    assert _wait_until(lambda: queue.get_state(a.run_id)["state"] == "running", timeout=2.0)

    queue.cancel(a.run_id)
    assert _wait_until(
        lambda: queue.get_state(a.run_id)["state"] in ("cancelled", "crashed"),
        timeout=2.0,
    )
    # SIGTERM did the job; SIGKILL never needed.
    assert len(spawner.terminated_pids) == 1
    assert len(spawner.force_killed_pids) == 0


def test_cancel_escalates_to_sigkill_when_sigterm_ignored(
    queue_dirs: tuple[Path, Path], spawner: _StubSpawner
) -> None:
    """A process that ignores SIGTERM (mp Queue deadlock) gets SIGKILL'd.

    Reproduces the production deadlock: ragctl orchestrator stays alive after
    SIGTERM because its multiprocessing parent blocks on Queue.get() from
    already-dead children. The queue worker must escalate.
    """
    state_dir, log_dir = queue_dirs
    q = DiscoverQueue(
        state_dir=state_dir,
        log_dir=log_dir,
        spawner=spawner,
        max_queue_depth=50,
        poll_interval_s=0.01,
        cancel_kill_timeout_s=0.3,
    )
    try:
        spawner.terminate_ignored = True
        spawner.next_delay_s = 10.0  # never finishes; only SIGKILL can stop it

        a = q.enqueue(_make_req("stuck"))
        q.start()
        assert _wait_until(lambda: q.get_state(a.run_id)["state"] == "running", timeout=2.0)

        q.cancel(a.run_id)
        # SIGTERM landed but pid stayed alive (ignored).
        assert _wait_until(lambda: len(spawner.terminated_pids) >= 1, timeout=1.0)

        # Within cancel_kill_timeout_s + slack, worker escalates to SIGKILL
        # and moves the entry to done/.
        assert _wait_until(lambda: q.get_state(a.run_id)["state"] == "crashed", timeout=2.0), (
            "worker did not escalate SIGTERM to SIGKILL within timeout"
        )
        assert len(spawner.force_killed_pids) == 1
        assert spawner.force_killed_pids[0] == spawner.terminated_pids[0]
    finally:
        q.stop()
        _ = q.is_alive() and q.join(timeout=2.0)


def test_cancel_escalation_blocks_next_entry_only_until_kill(
    queue_dirs: tuple[Path, Path], spawner: _StubSpawner
) -> None:
    """Queued entries cannot start until a stuck running entry is SIGKILL'd."""
    state_dir, log_dir = queue_dirs
    q = DiscoverQueue(
        state_dir=state_dir,
        log_dir=log_dir,
        spawner=spawner,
        max_queue_depth=50,
        poll_interval_s=0.01,
        cancel_kill_timeout_s=0.2,
    )
    try:
        spawner.terminate_ignored = True
        spawner.next_delay_s = 10.0

        stuck = q.enqueue(_make_req("stuck"))
        waiting = q.enqueue(_make_req("waiting"))
        q.start()
        assert _wait_until(lambda: q.get_state(stuck.run_id)["state"] == "running", timeout=2.0)

        # While stuck is running and SIGTERM hasn't fired, waiting stays queued.
        assert q.get_state(waiting.run_id)["state"] == "queued"

        # After cancel + escalation, waiting must finally get its turn.
        # Re-arm the spawner so the second entry can actually complete.
        spawner.terminate_ignored = True  # still ignored for the stuck one
        q.cancel(stuck.run_id)

        # Reset for the next spawn so the waiting entry finishes normally.
        # (cancel-related state is per-pid in the stub; new spawn = new pid.)
        spawner.next_delay_s = 0.05

        assert _wait_until(lambda: q.get_state(waiting.run_id)["state"] == "done", timeout=4.0)
        assert spawner.force_killed_pids == [stuck.run_id and spawner.terminated_pids[0]]
    finally:
        q.stop()
        _ = q.is_alive() and q.join(timeout=2.0)


def _seed_done_entry(
    state_dir: Path, log_dir: Path, *, run_id: str, result: dict[str, Any]
) -> None:
    """Place a done/ entry whose result.json carries ``result``."""
    done_dir = state_dir / "done"
    done_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    result_path = log_dir / f"{run_id}.result.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    entry = {
        "run_id": run_id,
        "counter": 1,
        "queries": ["q"],
        "collection": "trading",
        "partition": "research_briefs",
        "mode": "deep",
        "timeout": 60,
        "keep": False,
        "enqueued_at": time.time(),
        "pid": 12345,
        "log_path": str(log_dir / f"{run_id}.log"),
        "result_path": str(result_path),
    }
    (done_dir / f"00001_{run_id}.json").write_text(json.dumps(entry), encoding="utf-8")


def test_error_result_maps_to_crashed(queue_dirs: tuple[Path, Path], spawner: _StubSpawner) -> None:
    """Finding 2: a fully-failed discover run writes status='error' and exits;
    get_state must surface it as 'crashed', NOT a clean 'done' (n_imported=0).
    """
    state_dir, log_dir = queue_dirs
    _seed_done_entry(
        state_dir,
        log_dir,
        run_id="errrun01",
        result={"status": "error", "n_imported": 0, "error": "all queries failed"},
    )
    q = DiscoverQueue(
        state_dir=state_dir,
        log_dir=log_dir,
        spawner=spawner,
        max_queue_depth=50,
        poll_interval_s=0.01,
    )
    try:
        assert q.get_state("errrun01")["state"] == "crashed"
    finally:
        q.stop()
        _ = q.is_alive() and q.join(timeout=2.0)


@pytest.mark.parametrize("status", ["ok", "no_new_sources"])
def test_success_results_map_to_done(
    queue_dirs: tuple[Path, Path], spawner: _StubSpawner, status: str
) -> None:
    """Finding 2 guard: genuine success statuses stay 'done'."""
    state_dir, log_dir = queue_dirs
    _seed_done_entry(
        state_dir,
        log_dir,
        run_id=f"ok_{status[:5]}",
        result={"status": status, "n_imported": 2},
    )
    q = DiscoverQueue(
        state_dir=state_dir,
        log_dir=log_dir,
        spawner=spawner,
        max_queue_depth=50,
        poll_interval_s=0.01,
    )
    try:
        assert q.get_state(f"ok_{status[:5]}")["state"] == "done"
    finally:
        q.stop()
        _ = q.is_alive() and q.join(timeout=2.0)


def test_reconcile_recycled_pid_treated_as_dead(queue_dirs: tuple[Path, Path]) -> None:
    """Finding 3: a post-restart running/ entry whose PID is alive but whose
    persisted start token no longer matches (PID recycled) must reconcile as
    crashed, not be left running forever.

    Uses the real DefaultSpawner (untracked-pid branch) with this test's own
    live PID but a deliberately wrong pid_start, so the start-token comparison
    is the only thing that can detect the staleness.
    """
    from src.server.discover_queue import DefaultSpawner  # noqa: PLC0415

    state_dir, log_dir = queue_dirs
    running_dir = state_dir / "running"
    running_dir.mkdir(parents=True)

    live_pid = os.getpid()
    real_token = _pid_start_token(live_pid)
    if real_token is None:
        pytest.skip("pid start token unavailable on this platform")

    fake_entry = {
        "run_id": "recyc001",
        "counter": 1,
        "queries": ["q"],
        "collection": "trading",
        "partition": "research_briefs",
        "mode": "deep",
        "timeout": 60,
        "keep": False,
        "enqueued_at": time.time(),
        "pid": live_pid,
        "pid_start": real_token + "_STALE",
        "log_path": str(log_dir / "recyc001.log"),
        "result_path": str(log_dir / "recyc001.result.json"),
    }
    (running_dir / "00001_recyc001.json").write_text(json.dumps(fake_entry), encoding="utf-8")

    q = DiscoverQueue(
        state_dir=state_dir,
        log_dir=log_dir,
        spawner=DefaultSpawner(),
        max_queue_depth=50,
        poll_interval_s=0.01,
    )
    try:
        q.reconcile_on_startup()
        assert q.get_state("recyc001")["state"] == "crashed"
        assert len(list((state_dir / "running").iterdir())) == 0
    finally:
        q.stop()
        _ = q.is_alive() and q.join(timeout=2.0)


def test_reconcile_matching_pid_token_left_alone(queue_dirs: tuple[Path, Path]) -> None:
    """Finding 3 guard: a running/ entry whose PID is alive AND whose start
    token still matches is a genuinely live run — must NOT be reconciled away.
    """
    from src.server.discover_queue import DefaultSpawner  # noqa: PLC0415

    state_dir, log_dir = queue_dirs
    running_dir = state_dir / "running"
    running_dir.mkdir(parents=True)

    live_pid = os.getpid()
    real_token = _pid_start_token(live_pid)
    if real_token is None:
        pytest.skip("pid start token unavailable on this platform")

    fake_entry = {
        "run_id": "livetok1",
        "counter": 1,
        "queries": ["q"],
        "collection": "trading",
        "partition": "research_briefs",
        "mode": "deep",
        "timeout": 60,
        "keep": False,
        "enqueued_at": time.time(),
        "pid": live_pid,
        "pid_start": real_token,
        "log_path": str(log_dir / "livetok1.log"),
        "result_path": str(log_dir / "livetok1.result.json"),
    }
    (running_dir / "00001_livetok1.json").write_text(json.dumps(fake_entry), encoding="utf-8")

    q = DiscoverQueue(
        state_dir=state_dir,
        log_dir=log_dir,
        spawner=DefaultSpawner(),
        max_queue_depth=50,
        poll_interval_s=0.01,
    )
    try:
        q.reconcile_on_startup()
        assert q.get_state("livetok1")["state"] == "running"
    finally:
        q.stop()
        _ = q.is_alive() and q.join(timeout=2.0)


def test_state_dir_property_exposes_root(
    queue_dirs: tuple[Path, Path], spawner: _StubSpawner
) -> None:
    """Finding 1: DiscoverQueue.state_dir is a public property so api_ingest's
    queue_stream can scan queued/ + running/ to surface MCP/CLI-queued runs.
    """
    state_dir, log_dir = queue_dirs
    q = DiscoverQueue(
        state_dir=state_dir,
        log_dir=log_dir,
        spawner=spawner,
        max_queue_depth=50,
        poll_interval_s=0.01,
    )
    try:
        assert q.state_dir == state_dir
    finally:
        q.stop()
        _ = q.is_alive() and q.join(timeout=2.0)
