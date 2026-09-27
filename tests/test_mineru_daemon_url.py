"""Unit tests for src/pdf_parsers/mineru_daemon.py URL discovery.

These tests define the contract for `pipeline_daemon_url()` — the read-only
lookup that parse workers use to find a live MinerU daemon. The CLI-subprocess
fallback was removed; ``parse_pipeline_only`` now requires the daemon to be
running so that GPU access goes through the gpu-broker's claim.

The state file lives at $RAG_MINERU_STATE (overridable for tests) and contains
JSON `{"url": "...", "pid": ...}` written by start_pipeline_daemon().
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


@pytest.fixture()
def state_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Override the daemon state file location for the test."""
    p = tmp_path / "mineru-pipeline.json"
    monkeypatch.setenv("RAG_MINERU_STATE", str(p))
    return p


def test_pipeline_daemon_url_returns_none_when_no_state_file(state_file: Path) -> None:
    from src.pdf_parsers import mineru_daemon

    assert not state_file.exists()
    assert mineru_daemon.pipeline_daemon_url() is None


def test_pipeline_daemon_url_returns_url_when_alive(state_file: Path) -> None:
    """If the recorded PID is the current process, treat as alive (proxy for live)."""
    from src.pdf_parsers import mineru_daemon

    state_file.write_text(json.dumps({"url": "http://127.0.0.1:5000", "pid": os.getpid()}))
    assert mineru_daemon.pipeline_daemon_url() == "http://127.0.0.1:5000"


def test_pipeline_daemon_url_returns_none_when_stale_pid(state_file: Path) -> None:
    """A PID that does not exist means a previous run crashed; ignore the file."""
    from src.pdf_parsers import mineru_daemon

    # PID 1 is init/launchd — it exists but isn't us. Use a high impossible PID
    # that os.kill(0) will reject with ProcessLookupError.
    stale_pid = 999_999
    state_file.write_text(json.dumps({"url": "http://127.0.0.1:5000", "pid": stale_pid}))
    assert mineru_daemon.pipeline_daemon_url() is None


def test_pipeline_daemon_url_returns_none_when_state_file_malformed(state_file: Path) -> None:
    from src.pdf_parsers import mineru_daemon

    state_file.write_text("not json")
    assert mineru_daemon.pipeline_daemon_url() is None


def test_pipeline_daemon_url_returns_none_when_state_file_missing_keys(state_file: Path) -> None:
    from src.pdf_parsers import mineru_daemon

    state_file.write_text(json.dumps({"url": "http://127.0.0.1:5000"}))  # no pid
    assert mineru_daemon.pipeline_daemon_url() is None


def test_parse_pipeline_only_requires_daemon(
    state_file: Path,
    tmp_path: Path,
) -> None:
    """Removed CLI-subprocess fallback — must raise loudly when no daemon."""
    from src.pdf_parsers.mineru import parse_pipeline_only

    pdf = tmp_path / "fake.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    assert not state_file.exists()  # no daemon registered

    with pytest.raises(RuntimeError, match="pipeline daemon not running"):
        parse_pipeline_only(pdf)
