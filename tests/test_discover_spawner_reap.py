"""Regression tests for #35-1a: DefaultSpawner must reap zombie children.

Before the fix, `DefaultSpawner.is_alive` used `os.kill(pid, 0)`, which
returns alive for zombie processes (kernel keeps the entry until parent
calls waitpid). Because `spawn()` dropped the `Popen` reference, no one
ever reaped — so after SIGKILL, the discover-queue's inner loop spun
forever, never wrote a crashed result.json, and never moved the entry
from `running/` to `done/`.

The fix tracks the `Popen` instance and uses `proc.poll()` in `is_alive`,
which performs a non-blocking waitpid and reaps the zombie.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from src.server.discover_queue import DefaultSpawner, QueueEntry


def _make_entry(tmp_path: Path) -> QueueEntry:
    return QueueEntry(
        run_id="t",
        counter=1,
        queries=["x"],
        collection="trading",
        partition="research_briefs",
        mode="fast",
        timeout=10,
        keep=False,
        enqueued_at=time.time(),
        pid=None,
        log_path=str(tmp_path / "l.log"),
        result_path=str(tmp_path / "r.json"),
    )


def test_default_spawner_is_alive_returns_false_after_subprocess_exits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Zombie children are reaped via Popen.poll(), so is_alive flips to False."""
    spawner = DefaultSpawner()

    # Override _build_cmd to avoid the heavyweight `uv run python ...` path.
    # The test only cares about subprocess lifecycle tracking.
    monkeypatch.setattr(
        spawner,
        "_build_cmd",
        lambda entry, result_path: ["sleep", "0.05"],
    )

    entry = _make_entry(tmp_path)
    pid = spawner.spawn(entry, Path(entry.log_path), Path(entry.result_path))

    deadline = time.time() + 5.0
    while time.time() < deadline:
        if not spawner.is_alive(pid):
            return
        time.sleep(0.05)
    pytest.fail(
        f"is_alive(pid={pid}) never returned False after subprocess exit — "
        "zombie not reaped (regression of #35-1a)"
    )


def test_default_spawner_is_alive_returns_true_for_live_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spawner = DefaultSpawner()
    monkeypatch.setattr(
        spawner,
        "_build_cmd",
        lambda entry, result_path: ["sleep", "5"],
    )

    entry = _make_entry(tmp_path)
    pid = spawner.spawn(entry, Path(entry.log_path), Path(entry.result_path))

    try:
        assert spawner.is_alive(pid) is True
    finally:
        spawner.force_kill(pid)
        deadline = time.time() + 5.0
        while time.time() < deadline and spawner.is_alive(pid):
            time.sleep(0.05)


def test_default_spawner_is_alive_falls_back_to_os_kill_for_untracked_pid() -> None:
    """Reconcile-on-startup uses pids from prior server runs (no Popen tracked).
    is_alive must still work — falling back to os.kill(pid, 0)."""
    spawner = DefaultSpawner()
    # This process is definitely alive.
    assert spawner.is_alive(os.getpid()) is True
