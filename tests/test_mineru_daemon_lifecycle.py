"""Integration tests for the real mineru-api daemon lifecycle.

Spawns the actual `.venv-mineru/bin/mineru-api` server, hits its health
endpoint, then tears it down. Does NOT parse a PDF — that's covered in the
e2e suite. These tests catch:

  - mineru-api startup flag changes across MinerU versions
  - process-group cleanup regressions (orphaned uvicorn workers)
  - state-file write/read cycle

Marked @pytest.mark.slow because cold start is 15-45s. Skipped by default;
run with `pytest -m slow`. Skipped automatically when .venv-mineru is absent.
"""

from __future__ import annotations

import json
import os
import signal
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
MINERU_API_BIN = REPO / ".venv-mineru" / "bin" / "mineru-api"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not MINERU_API_BIN.exists(), reason=".venv-mineru not installed"),
]


@pytest.fixture()
def state_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    p = tmp_path / "mineru-pipeline.json"
    monkeypatch.setenv("RAG_MINERU_STATE", str(p))
    return p


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def test_start_writes_state_file_with_live_pid(state_file: Path) -> None:
    from src.pdf_parsers import mineru_daemon

    try:
        url = mineru_daemon.start_pipeline_daemon()
        assert url.startswith("http://")
        assert state_file.exists()
        data = json.loads(state_file.read_text())
        assert data["url"] == url
        assert _pid_alive(data["pid"])
    finally:
        mineru_daemon.stop_pipeline_daemon()


def test_health_endpoint_responds(state_file: Path) -> None:
    import urllib.request

    from src.pdf_parsers import mineru_daemon

    try:
        url = mineru_daemon.start_pipeline_daemon()
        # mineru-api uses /docs as the canonical liveness probe; FastAPI default.
        # If the daemon helper changes the probe path, update this test.
        with urllib.request.urlopen(f"{url}/docs", timeout=10) as resp:
            assert resp.status == 200
    finally:
        mineru_daemon.stop_pipeline_daemon()


def test_stop_kills_process_group_and_clears_state(state_file: Path) -> None:
    from src.pdf_parsers import mineru_daemon

    mineru_daemon.start_pipeline_daemon()
    pid = json.loads(state_file.read_text())["pid"]
    assert _pid_alive(pid)

    mineru_daemon.stop_pipeline_daemon()

    # Allow brief grace for process group teardown
    deadline = time.time() + 10
    while time.time() < deadline and _pid_alive(pid):
        time.sleep(0.2)
    assert not _pid_alive(pid)
    assert not state_file.exists()


def test_start_reuses_existing_alive_daemon(state_file: Path) -> None:
    """Calling start twice in the same process must return the same URL,
    not spawn a second mineru-api."""
    from src.pdf_parsers import mineru_daemon

    try:
        url1 = mineru_daemon.start_pipeline_daemon()
        pid1 = json.loads(state_file.read_text())["pid"]

        url2 = mineru_daemon.start_pipeline_daemon()
        pid2 = json.loads(state_file.read_text())["pid"]

        assert url1 == url2
        assert pid1 == pid2
    finally:
        mineru_daemon.stop_pipeline_daemon()


def test_start_overwrites_stale_state_file(state_file: Path) -> None:
    """A leftover state file with a dead PID should not block a new daemon."""
    state_file.write_text(json.dumps({"url": "http://127.0.0.1:1", "pid": 999_999}))
    from src.pdf_parsers import mineru_daemon

    try:
        url = mineru_daemon.start_pipeline_daemon()
        data = json.loads(state_file.read_text())
        assert data["url"] == url
        assert data["pid"] != 999_999
        assert _pid_alive(data["pid"])
    finally:
        mineru_daemon.stop_pipeline_daemon()


def test_stop_is_idempotent(state_file: Path) -> None:
    """Calling stop without an active daemon must not raise."""
    from src.pdf_parsers import mineru_daemon

    mineru_daemon.stop_pipeline_daemon()  # no-op
    assert not state_file.exists()


def test_pipeline_daemon_url_returns_url_after_real_start(state_file: Path) -> None:
    """Cross-module check: pipeline_daemon_url() correctly reads what
    start_pipeline_daemon() wrote."""
    from src.pdf_parsers import mineru_daemon

    try:
        url = mineru_daemon.start_pipeline_daemon()
        assert mineru_daemon.pipeline_daemon_url() == url
    finally:
        mineru_daemon.stop_pipeline_daemon()
        assert mineru_daemon.pipeline_daemon_url() is None


# Keep the SIGTERM-doesn't-orphan-children test from the existing VLM pattern
# in mind (mineru.py:99-121). The killpg behaviour is exercised implicitly by
# test_stop_kills_process_group_and_clears_state; if it ever regresses, the
# parent dies but uvicorn workers leak — surfaces as later test failures from
# the leaked port. Don't add a separate subprocess-tree test; it's brittle.
_ = signal  # imported for future explicit kill-tree test if needed
