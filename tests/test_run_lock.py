"""Tests for the `ragctl run` lockfile heartbeat + stale-takeover machinery.

Covers issues #28 (stale lockfile blocking subsequent runs) and #37
(heartbeat-based liveness for zombie/wedged holders).
"""

from __future__ import annotations

import fcntl
import os
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from src.pipeline import runner


@pytest.fixture
def tmp_lock(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Path]:
    """Redirect _RUN_LOCK_PATH to a tmp file and reset module heartbeat state."""
    lock_path = tmp_path / "ragctl-run.lock"
    monkeypatch.setattr(runner, "_RUN_LOCK_PATH", lock_path)
    runner._heartbeat_stop.set()
    runner._heartbeat_thread = None
    yield lock_path
    runner._heartbeat_stop.set()
    if lock_path.exists():
        lock_path.unlink()


def _hold_flock_with_body(path: Path, body: str) -> int:
    """Acquire an exclusive flock on `path` and write `body`. Returns the fd."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_RDWR | os.O_CREAT, 0o644)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.ftruncate(fd, 0)
    os.lseek(fd, 0, 0)
    os.write(fd, body.encode())
    return fd


def _release_flock(fd: int) -> None:
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    except OSError:
        pass
    try:
        os.close(fd)
    except OSError:
        pass


def test_acquire_then_release(tmp_lock: Path) -> None:
    fd = runner._acquire_run_lock()
    try:
        assert tmp_lock.exists()
        holder = runner._read_lockfile_holder()
        assert int(holder["pid"]) == os.getpid()
        assert "last_heartbeat" in holder
    finally:
        runner._release_run_lock(fd)


def test_second_acquire_blocked_when_live(tmp_lock: Path) -> None:
    fd = runner._acquire_run_lock()
    try:
        with pytest.raises(runner.RunLockHeldError):
            runner._acquire_run_lock()
    finally:
        runner._release_run_lock(fd)


def test_stale_heartbeat_triggers_takeover(tmp_lock: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A stale holder (dead PID + old heartbeat) is taken over and re-acquired."""
    monkeypatch.setattr(runner, "_RUN_LOCK_STALE_S", 60)
    monkeypatch.setattr(runner, "_pid_alive", lambda pid: False)
    monkeypatch.setattr(runner, "_kill_holder", lambda pid: None)

    stale_ts = int(time.time()) - 600
    held_fd = _hold_flock_with_body(
        tmp_lock,
        f"pid=999999 host=ghost ts={stale_ts} last_heartbeat={stale_ts}\n",
    )
    try:
        new_fd = runner._acquire_run_lock()
        try:
            holder = runner._read_lockfile_holder()
            assert int(holder["pid"]) == os.getpid()
        finally:
            runner._release_run_lock(new_fd)
    finally:
        _release_flock(held_fd)


def test_stale_takeover_kills_live_holder(tmp_lock: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """If the stale holder PID is still alive, it is killed before takeover."""
    monkeypatch.setattr(runner, "_RUN_LOCK_STALE_S", 60)

    killed: list[int] = []
    monkeypatch.setattr(runner, "_kill_holder", lambda pid: killed.append(pid))
    monkeypatch.setattr(runner, "_pid_alive", lambda pid: True)

    stale_ts = int(time.time()) - 600
    held_fd = _hold_flock_with_body(
        tmp_lock,
        f"pid=12345 host=zombie ts={stale_ts} last_heartbeat={stale_ts}\n",
    )
    try:
        new_fd = runner._acquire_run_lock()
        try:
            assert killed == [12345]
        finally:
            runner._release_run_lock(new_fd)
    finally:
        _release_flock(held_fd)


def test_fresh_heartbeat_blocks_takeover(tmp_lock: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A holder with a recent heartbeat is NOT taken over — raises RunLockHeldError."""
    monkeypatch.setattr(runner, "_RUN_LOCK_STALE_S", 60)
    fresh_ts = int(time.time())
    held_fd = _hold_flock_with_body(
        tmp_lock,
        f"pid=12345 host=live ts={fresh_ts} last_heartbeat={fresh_ts}\n",
    )
    try:
        with pytest.raises(runner.RunLockHeldError) as exc:
            runner._acquire_run_lock()
        assert "12345" in str(exc.value)
    finally:
        _release_flock(held_fd)


def test_heartbeat_thread_updates_timestamp(
    tmp_lock: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The heartbeat thread rewrites last_heartbeat at the configured interval."""
    monkeypatch.setattr(runner, "_RUN_LOCK_HEARTBEAT_S", 1)
    fd = runner._acquire_run_lock()
    try:
        first = int(runner._read_lockfile_holder()["last_heartbeat"])
        time.sleep(2.2)
        second = int(runner._read_lockfile_holder()["last_heartbeat"])
        assert second > first
    finally:
        runner._release_run_lock(fd)
