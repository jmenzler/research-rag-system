"""Acceptance tests for src/pdf_parsers/mineru_daemon.py broker wiring.

Verifies the daemon-level claim contract over the gpu-broker HTTP daemon:
- start_pipeline_daemon() calls broker.acquire with the documented kwargs.
- stop_pipeline_daemon() calls broker.release.
- A failure in subprocess.Popen OR _wait_for_api releases the claim before
  re-raising (no orphan registry rows).
- broker.acquire raising GPUBusy propagates as GPUBusy (no spawn, no release).

Mocks subprocess.Popen and _wait_for_api so no real mineru-api launches.
Uses the shared ``mineru_fake_broker`` fixture from conftest.py instead of
spinning up a real broker daemon.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING
from unittest import mock

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator

    from tests.conftest import _FakeBroker


@pytest.fixture()
def state_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Override the daemon state file location."""
    p = tmp_path / "mineru-pipeline.json"
    monkeypatch.setenv("RAG_MINERU_STATE", str(p))
    return p


@pytest.fixture(autouse=True)
def _reset_run_claim() -> Iterator[None]:
    """Clear module-level _RUN_CLAIM between tests so they're independent."""
    from src.pdf_parsers import mineru_daemon

    mineru_daemon._RUN_CLAIM = None
    yield
    mineru_daemon._RUN_CLAIM = None


def _make_fake_proc(pid: int = 88888) -> mock.MagicMock:
    """Mock subprocess.Popen() return: a 'live' process."""
    fake = mock.MagicMock()
    fake.pid = pid
    fake.poll.return_value = None  # alive
    fake.terminate.return_value = None
    fake.wait.return_value = 0
    return fake


def _patch_spawn_dependencies(monkeypatch: pytest.MonkeyPatch) -> mock.MagicMock:
    """Patch out the side-effectful pieces of start_pipeline_daemon."""
    from src.pdf_parsers import mineru_daemon

    fake_popen = mock.MagicMock(return_value=_make_fake_proc())
    monkeypatch.setattr(mineru_daemon.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(mineru_daemon, "_wait_for_api", lambda *a, **kw: None)
    monkeypatch.setattr(mineru_daemon, "_sweep_orphan_mineru", lambda: None)
    return fake_popen


# ---------------------------------------------------------------------------
# 1. start_pipeline_daemon acquires the documented claim
# ---------------------------------------------------------------------------


def test_start_pipeline_daemon_acquires_with_broker(
    state_file: Path,
    mineru_fake_broker: _FakeBroker,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.pdf_parsers import mineru_daemon

    _patch_spawn_dependencies(monkeypatch)

    url = mineru_daemon.start_pipeline_daemon()
    assert url.startswith("http://")

    assert len(mineru_fake_broker.acquire_calls) == 1
    call = mineru_fake_broker.acquire_calls[0]
    assert call["pid"] == os.getpid()
    assert call["role"] == "mineru"
    assert call["priority"] == 30
    assert call["vram_mib"] == 6500
    assert call["yield_mode"] == "fail"
    assert call["blocking"] is True
    assert mineru_daemon._RUN_CLAIM == os.getpid()


# ---------------------------------------------------------------------------
# 2. stop_pipeline_daemon releases the claim
# ---------------------------------------------------------------------------


def test_stop_pipeline_daemon_releases_with_broker(
    state_file: Path,
    mineru_fake_broker: _FakeBroker,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.pdf_parsers import mineru_daemon

    _patch_spawn_dependencies(monkeypatch)
    # stop_pipeline_daemon tries to killpg — neutralise so the test doesn't
    # actually signal random pids on a CI host.
    monkeypatch.setattr(mineru_daemon.os, "killpg", lambda *a, **kw: None)
    monkeypatch.setattr(mineru_daemon.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(mineru_daemon, "_pid_alive", lambda pid: False)

    mineru_daemon.start_pipeline_daemon()
    assert mineru_fake_broker.release_calls == []

    mineru_daemon.stop_pipeline_daemon()

    assert mineru_fake_broker.release_calls == [os.getpid()]
    assert mineru_daemon._RUN_CLAIM is None


# ---------------------------------------------------------------------------
# 3. _wait_for_api failure releases claim before re-raise
# ---------------------------------------------------------------------------


def test_start_pipeline_daemon_releases_claim_on_wait_for_api_failure(
    state_file: Path,
    mineru_fake_broker: _FakeBroker,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.pdf_parsers import mineru_daemon

    _patch_spawn_dependencies(monkeypatch)

    def _fail(*_a: object, **_kw: object) -> None:
        raise RuntimeError("API never responded")

    monkeypatch.setattr(mineru_daemon, "_wait_for_api", _fail)

    with pytest.raises(RuntimeError, match="API never responded"):
        mineru_daemon.start_pipeline_daemon()

    assert mineru_fake_broker.release_calls == [os.getpid()]
    assert mineru_daemon._RUN_CLAIM is None


# ---------------------------------------------------------------------------
# 4. Popen failure releases claim before re-raise
# ---------------------------------------------------------------------------


def test_start_pipeline_daemon_releases_claim_on_popen_failure(
    state_file: Path,
    mineru_fake_broker: _FakeBroker,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.pdf_parsers import mineru_daemon

    def _boom(*_a: object, **_kw: object) -> mock.MagicMock:
        raise OSError("ENOENT: mineru-api not found")

    monkeypatch.setattr(mineru_daemon.subprocess, "Popen", _boom)
    monkeypatch.setattr(mineru_daemon, "_wait_for_api", lambda *a, **kw: None)
    monkeypatch.setattr(mineru_daemon, "_sweep_orphan_mineru", lambda: None)

    with pytest.raises(OSError, match="ENOENT"):
        mineru_daemon.start_pipeline_daemon()

    assert mineru_fake_broker.release_calls == [os.getpid()]
    assert mineru_daemon._RUN_CLAIM is None


# ---------------------------------------------------------------------------
# 5. broker GPUBusy propagates without spawn / release
# ---------------------------------------------------------------------------


def test_start_pipeline_daemon_propagates_gpubusy(
    state_file: Path,
    mineru_fake_broker: _FakeBroker,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the broker refuses the claim, no subprocess is started and no
    release is issued — there's nothing to release. The rag-system-facing
    ``GPUBusy`` is raised so callers (e.g. runner.py) can fall back."""
    from src.pdf_parsers import mineru_daemon

    fake_popen = _patch_spawn_dependencies(monkeypatch)
    mineru_fake_broker.next_acquire_raises = mineru_daemon._BrokerGPUBusy("simulated")

    with pytest.raises(mineru_daemon.GPUBusy, match="simulated"):
        mineru_daemon.start_pipeline_daemon()

    assert fake_popen.call_count == 0
    assert mineru_fake_broker.release_calls == []
    assert mineru_daemon._RUN_CLAIM is None
