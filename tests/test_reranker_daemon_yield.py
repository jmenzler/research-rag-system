"""Acceptance tests for reranker-daemon /_yield endpoint.

Verifies the daemon can be evicted: on POST /_yield it responds 202
immediately and tears down the llama-server child + releases the GPU claim
asynchronously in a background thread.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any
from unittest import mock

import pytest

if TYPE_CHECKING:
    from scripts.reranker_daemon import RerankerManager


def _make_fake_popen() -> mock.MagicMock:
    fake_proc = mock.MagicMock()
    fake_proc.poll.return_value = None  # alive
    fake_proc.pid = 424242
    fake_proc.terminate.return_value = None
    fake_proc.wait.return_value = 0
    fake_proc.kill.return_value = None
    return fake_proc


def _fresh_manager(monkeypatch: pytest.MonkeyPatch) -> RerankerManager:
    from scripts import reranker_daemon

    monkeypatch.setattr(reranker_daemon, "LOG_DIR", "/tmp")
    monkeypatch.setattr(reranker_daemon, "STARTUP_TIMEOUT_S", 1)
    monkeypatch.setattr(reranker_daemon, "IDLE_TIMEOUT_S", 60)
    return reranker_daemon.RerankerManager()


class _FakeHandlerBase:
    """Minimal FakeHandler scaffolding — bypass socket init, capture writes."""

    def __init__(self) -> None:
        self.command = "POST"
        self.path = "/_yield"
        self.headers: dict[str, str] = {}

    def send_response(self, code: int, message: str | None = None) -> None:
        type(self)._captured["code"] = code

    def send_header(self, key: str, value: str) -> None:
        type(self)._captured.setdefault("headers", []).append((key, value))

    def end_headers(self) -> None:
        type(self)._captured["end_headers"] = True

    class _Buf:
        buf = b""

        def write(self, data: bytes) -> None:
            type(self).buf += data

    wfile = _Buf()  # type: ignore[assignment]
    _captured: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# 1. POST /_yield returns 202 immediately
# ---------------------------------------------------------------------------


def test_yield_endpoint_returns_202_immediately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """POST /_yield must return 202; it must not block waiting for teardown."""
    from scripts import reranker_daemon

    mgr = _fresh_manager(monkeypatch)
    monkeypatch.setattr(reranker_daemon, "_manager", mgr)

    # Stub the teardown so the test doesn't try to talk to a real broker.
    teardown_called: list[bool] = []

    def _fake_teardown(m: RerankerManager) -> None:
        time.sleep(0.05)  # simulate async work
        teardown_called.append(True)

    monkeypatch.setattr(reranker_daemon, "_yield_teardown", _fake_teardown)

    class FakeHandler(_FakeHandlerBase, reranker_daemon.ProxyHandler):  # type: ignore[misc]
        _captured: dict[str, Any] = {}

        def __init__(self) -> None:
            _FakeHandlerBase.__init__(self)

    # Reset shared state.
    FakeHandler._captured = {}

    start = time.monotonic()
    h = FakeHandler()
    h.do_POST()
    elapsed = time.monotonic() - start

    # Must return 202.
    assert FakeHandler._captured.get("code") == 202, (
        f"/_yield must return 202, got {FakeHandler._captured.get('code')}"
    )
    # Must not block — handler should return well under 1s even if teardown is slow.
    assert elapsed < 1.0, f"/_yield handler blocked for {elapsed:.2f}s"


# ---------------------------------------------------------------------------
# 2. /_yield triggers teardown of llama-server child + broker release
# ---------------------------------------------------------------------------


def test_yield_terminates_llama_and_releases_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """POST /_yield must terminate the llama-server child and release the GPU claim."""
    from scripts import reranker_daemon

    mgr = _fresh_manager(monkeypatch)
    # Put the manager into a "running" state: _proc set, _claim set.
    fake_proc = _make_fake_popen()
    mgr._proc = fake_proc
    # Fake claim — after HttpClient switch, _claim is the PID passed to broker.release().
    mgr._claim = 999

    # Inject a fake broker so _yield_teardown can call broker.release() without
    # a real daemon.
    fake_release_calls: list[int] = []

    class FakeBroker:
        def release(self, *, pid: int) -> object:
            fake_release_calls.append(pid)
            return object()

    monkeypatch.setattr(reranker_daemon, "_BROKER", FakeBroker())
    monkeypatch.setattr(reranker_daemon, "_manager", mgr)

    class FakeHandler(_FakeHandlerBase, reranker_daemon.ProxyHandler):  # type: ignore[misc]
        _captured: dict[str, Any] = {}

        def __init__(self) -> None:
            _FakeHandlerBase.__init__(self)

    FakeHandler._captured = {}
    FakeHandler._Buf.buf = b""

    h = FakeHandler()
    h.do_POST()

    # Give the background thread up to 2s to complete teardown.
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if fake_proc.terminate.called or mgr._claim is None:
            break
        time.sleep(0.05)

    # llama-server child must have been terminated.
    assert fake_proc.terminate.called, "/_yield must terminate the llama-server child process"
    # GPU claim must be released.
    assert mgr._claim is None, f"/_yield must clear _claim after release; got {mgr._claim!r}"
