from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.pdf_parsers import mineru_daemon


def _write_state(tmp: Path) -> Path:
    sf = tmp / "state.json"
    sf.write_text(json.dumps({"url": "http://127.0.0.1:9999", "pid": 1}))
    return sf


def test_busy_daemon_alive_via_open_socket(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HTTP probe fails (busy parsing) but the listening socket is open => alive."""
    sf = _write_state(tmp_path)
    monkeypatch.setenv("RAG_MINERU_STATE", str(sf))
    monkeypatch.setattr(mineru_daemon, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(mineru_daemon, "_http_probe", lambda *a, **k: False)
    monkeypatch.setattr(mineru_daemon, "_socket_open", lambda *a, **k: True)
    assert mineru_daemon.is_alive() is True


def test_crashed_daemon_dead_when_socket_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """API segfaulted: HTTP fails AND socket refused => dead (so it gets respawned)."""
    sf = _write_state(tmp_path)
    monkeypatch.setenv("RAG_MINERU_STATE", str(sf))
    monkeypatch.setattr(mineru_daemon, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(mineru_daemon, "_http_probe", lambda *a, **k: False)
    monkeypatch.setattr(mineru_daemon, "_socket_open", lambda *a, **k: False)
    assert mineru_daemon.is_alive() is False


def test_responsive_daemon_alive_via_http(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sf = _write_state(tmp_path)
    monkeypatch.setenv("RAG_MINERU_STATE", str(sf))
    monkeypatch.setattr(mineru_daemon, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(mineru_daemon, "_http_probe", lambda *a, **k: True)
    # socket probe must not even be needed
    monkeypatch.setattr(mineru_daemon, "_socket_open", lambda *a, **k: False)
    assert mineru_daemon.is_alive() is True


def test_dead_pid_is_dead(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sf = _write_state(tmp_path)
    monkeypatch.setenv("RAG_MINERU_STATE", str(sf))
    monkeypatch.setattr(mineru_daemon, "_pid_alive", lambda pid: False)
    monkeypatch.setattr(mineru_daemon, "_http_probe", lambda *a, **k: True)
    monkeypatch.setattr(mineru_daemon, "_socket_open", lambda *a, **k: True)
    assert mineru_daemon.is_alive() is False


def test_socket_open_refused_is_dead(monkeypatch: pytest.MonkeyPatch) -> None:
    def _refuse(*a: object, **k: object) -> object:
        raise ConnectionRefusedError(111, "refused")

    monkeypatch.setattr(mineru_daemon.socket, "create_connection", _refuse)
    assert mineru_daemon._socket_open("127.0.0.1", 9999, timeout=0.1) is False


def test_socket_open_other_oserror_is_alive(monkeypatch: pytest.MonkeyPatch) -> None:
    def _timeout(*a: object, **k: object) -> object:
        raise TimeoutError("slow accept queue")

    monkeypatch.setattr(mineru_daemon.socket, "create_connection", _timeout)
    assert mineru_daemon._socket_open("127.0.0.1", 9999, timeout=0.1) is True
