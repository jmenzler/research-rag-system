"""Test fixtures shared across the suite.

Currently just cleans up the global ragctl run lockfile between tests so a
test that fails after acquiring the lock doesn't poison the next one.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from src.pipeline import runner


@pytest.fixture(autouse=True)
def _isolate_ragctl_lockfile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "_RUN_LOCK_PATH", tmp_path / "ragctl-run.lock")


class _DummyReranker:
    """Deterministic stand-in for the reranker singleton.

    Returns a length-of-pairs score list that descends from 1.0 by 0.01,
    so callers can still pick a "top-k" without hitting the network and
    without loading sentence-transformers.
    """

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [max(0.0, 1.0 - i * 0.01) for i in range(len(pairs))]

    def identity(self) -> dict[str, str | None]:
        return {"backend": "dummy", "model": "test-stub", "endpoint": None}


@pytest.fixture(autouse=True)
def _stub_reranker_for_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    """Autouse: replace the reranker singleton with a deterministic stub.

    Tests must never depend on the live PC reranker endpoint (it is unreachable
    from Mac CI and from agent worktrees), and the local cross-encoder backend
    triggers a HuggingFace download on first call. Both paths cause hangs that
    look like silent failures. Use the stub everywhere unless a specific test
    explicitly overrides it.

    Implemented as a no-op when ``src.query.rerankers`` is not importable so
    tests that do not touch the rerankers module pay zero cost.
    """
    try:
        from src.query import rerankers
    except ImportError:
        return

    stub = _DummyReranker()
    monkeypatch.setattr(rerankers, "_singleton", stub, raising=False)


class _FakeBroker:
    """Recording stub matching the HttpClient surface used by daemon modules."""

    def __init__(self) -> None:
        self.acquire_calls: list[dict[str, Any]] = []
        self.release_calls: list[int] = []
        self.next_acquire_raises: Exception | None = None

    def acquire(self, *, pid: int, **kwargs: object) -> SimpleNamespace:
        self.acquire_calls.append({"pid": pid, **kwargs})
        if self.next_acquire_raises is not None:
            raise self.next_acquire_raises
        return SimpleNamespace(
            claim_id=pid,
            external_mib=0,
            budget_mib=8000,
            headroom_mib=1000,
            evicted=[],
        )

    def release(self, *, pid: int) -> bool:
        self.release_calls.append(pid)
        return True


class _FakeBrokerBusyError(RuntimeError):
    pass


class _FakeGpuBusyError(RuntimeError):
    pass


@pytest.fixture()
def fake_broker(monkeypatch: pytest.MonkeyPatch) -> _FakeBroker:
    """Inject a stub HttpClient into the reranker daemon module.

    Tests that exercise _spawn/_shutdown no longer need a real broker daemon.
    Assertions on acquire/release use the call-log lists instead of SQLite.
    """
    from scripts import reranker_daemon

    fake = _FakeBroker()
    monkeypatch.setattr(reranker_daemon, "_BROKER", fake)
    monkeypatch.setattr(reranker_daemon, "MODEL_PATH", "models/test-reranker.gguf")
    monkeypatch.setattr(reranker_daemon, "_BrokerGPUBusy", _FakeBrokerBusyError)
    monkeypatch.setattr(reranker_daemon, "GPUBusy", _FakeGpuBusyError)
    return fake


@pytest.fixture()
def mineru_fake_broker(monkeypatch: pytest.MonkeyPatch) -> _FakeBroker:
    """Inject a stub HttpClient into the MinerU pipeline-daemon module.

    Mirrors ``fake_broker`` for the reranker — same recording semantics so
    tests can assert acquire kwargs and release pids without a real daemon.
    """
    from src.pdf_parsers import mineru_daemon

    fake = _FakeBroker()
    monkeypatch.setattr(mineru_daemon, "_BROKER", fake)
    monkeypatch.setattr(mineru_daemon, "_BrokerGPUBusy", _FakeBrokerBusyError)
    monkeypatch.setattr(mineru_daemon, "GPUBusy", _FakeGpuBusyError)
    monkeypatch.setattr(mineru_daemon, "_free_port", lambda: 50123)
    return fake
