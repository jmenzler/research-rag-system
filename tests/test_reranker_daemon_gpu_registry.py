"""Acceptance tests for scripts/reranker_daemon.py GPU broker integration.

Verifies the lifecycle wiring:
- _spawn() acquires a broker slot at role=reranker, priority=0, vram=7000
  with yield_mode='evict' (the production behavior change from library mode).
- _shutdown() releases the slot via broker.release().
- GPUBusy raised by acquire propagates out and releases the manager lock.
- ProxyHandler translates RuntimeError("GPU busy") into HTTP 503.
- atexit handler releases the slot on normal exit.
- Double-release (SIGTERM then atexit) is safe.

We never spawn a real llama-server. ``subprocess.Popen`` is patched and the
health probe (``_poll_until_healthy``) is replaced with an immediate-ready
stub so ``ensure_running`` never blocks.

The ``fake_broker`` fixture (conftest.py) injects a stub HttpClient into
``reranker_daemon._BROKER`` so tests run without a real broker daemon.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any
from unittest import mock

if TYPE_CHECKING:
    from scripts.reranker_daemon import RerankerManager

import pytest

if TYPE_CHECKING:
    from tests.conftest import _FakeBroker


def _make_fake_popen() -> mock.MagicMock:
    """Build a ``subprocess.Popen`` mock that looks like a live process."""
    fake_proc = mock.MagicMock()
    fake_proc.poll.return_value = None  # alive
    fake_proc.pid = 424242
    fake_proc.terminate.return_value = None
    fake_proc.wait.return_value = 0
    fake_proc.kill.return_value = None
    return fake_proc


def _fresh_manager(monkeypatch: pytest.MonkeyPatch) -> RerankerManager:
    """Return a brand-new RerankerManager and patch its env to be Mac-friendly."""
    from scripts import reranker_daemon

    # _spawn() opens a log file at LOG_DIR; redirect to /tmp so test works on Mac.
    monkeypatch.setattr(reranker_daemon, "LOG_DIR", "/tmp")
    # Don't burn 90s waiting for an unmocked health probe.
    monkeypatch.setattr(reranker_daemon, "STARTUP_TIMEOUT_S", 1)
    return reranker_daemon.RerankerManager()


# ---------------------------------------------------------------------------
# 1. _spawn acquires GPU claim
# ---------------------------------------------------------------------------


def test_spawn_acquires_gpu_claim_on_success(
    fake_broker: _FakeBroker,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import reranker_daemon

    mgr = _fresh_manager(monkeypatch)

    fake_popen = mock.MagicMock(return_value=_make_fake_popen())
    monkeypatch.setattr(reranker_daemon.subprocess, "Popen", fake_popen)

    mgr._spawn()

    # Assert subprocess.Popen called once.
    assert fake_popen.call_count == 1
    command = fake_popen.call_args.args[0]
    assert command[command.index("--host") + 1] == "127.0.0.1"

    # Assert broker.acquire was called with the right args.
    assert len(fake_broker.acquire_calls) == 1, (
        f"Expected 1 acquire call, got {fake_broker.acquire_calls}"
    )
    call = fake_broker.acquire_calls[0]
    assert call["pid"] == os.getpid()
    assert call["role"] == "reranker"
    assert call["priority"] == 0
    assert call["vram_mib"] == 7000
    assert call["yield_mode"] == "evict", (
        "reranker must register with yield_mode='evict' to enable ollama eviction"
    )
    assert call["blocking"] is False

    # _claim must be set to the current PID.
    assert mgr._claim == os.getpid()

    # Cleanup: release the fake claim.
    fake_broker.release(pid=os.getpid())


# ---------------------------------------------------------------------------
# 2. _shutdown releases GPU claim
# ---------------------------------------------------------------------------


def test_shutdown_releases_gpu_claim(
    fake_broker: _FakeBroker,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import reranker_daemon

    mgr = _fresh_manager(monkeypatch)
    fake_popen = mock.MagicMock(return_value=_make_fake_popen())
    monkeypatch.setattr(reranker_daemon.subprocess, "Popen", fake_popen)

    mgr._spawn()
    assert mgr._claim == os.getpid()

    # _shutdown should release.
    mgr._shutdown()

    assert len(fake_broker.release_calls) >= 1, "broker.release must be called after _shutdown"
    assert os.getpid() in fake_broker.release_calls
    assert mgr._claim is None, "_claim must be None after _shutdown"


# ---------------------------------------------------------------------------
# 3. GPUBusy releases the manager lock
# ---------------------------------------------------------------------------


def test_spawn_releases_lock_on_gpubusy(
    fake_broker: _FakeBroker,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reranker uses blocking=False. If a higher-priority holder is active,
    GPUBusy must propagate out of _spawn (and out of ensure_running's
    `with self._lock:` block, releasing the lock)."""
    from scripts import reranker_daemon

    mgr = _fresh_manager(monkeypatch)

    # Make the broker's acquire raise GPUBusy.
    fake_broker.next_acquire_raises = reranker_daemon._BrokerGPUBusy("simulated: device busy")

    fake_popen = mock.MagicMock(return_value=_make_fake_popen())
    monkeypatch.setattr(reranker_daemon.subprocess, "Popen", fake_popen)

    # Drive _spawn through ensure_running so the `with self._lock:` is exercised.
    with pytest.raises(RuntimeError, match="GPU busy: simulated: device busy"):
        mgr.ensure_running()

    # Lock must be released — non-blocking acquire must succeed.
    assert mgr._lock.acquire(blocking=False), (
        "Manager lock still held after GPUBusy unwind — lock leaked!"
    )
    mgr._lock.release()

    # Popen must NOT have been called (acquire failed before spawn).
    assert fake_popen.call_count == 0

    # No release should have been called (never acquired).
    assert fake_broker.release_calls == []


# ---------------------------------------------------------------------------
# 4. ProxyHandler returns 503 on GPU-busy
# ---------------------------------------------------------------------------


def test_proxy_handler_returns_503_on_gpubusy(
    fake_broker: _FakeBroker,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ensure_running raises RuntimeError('GPU busy'), the proxy responds 503."""
    from scripts import reranker_daemon

    captured: dict[str, Any] = {}

    class FakeHandler(reranker_daemon.ProxyHandler):
        # Bypass BaseHTTPRequestHandler __init__ which expects a real socket.
        def __init__(self) -> None:  # noqa: D401  -- fake bootstrap, no super().__init__()
            self.command = "POST"
            self.path = "/v1/rerank"
            self.headers = {}  # type: ignore[assignment]

        def send_response(self, code: int, message: str | None = None) -> None:
            captured["code"] = code

        def send_header(self, key: str, value: str) -> None:
            captured.setdefault("headers", []).append((key, value))

        def end_headers(self) -> None:
            captured["end_headers"] = True

        class _StubBuffer:
            def write(self, _: bytes) -> None:
                pass

        wfile = _StubBuffer()  # type: ignore[assignment]

    # Make ensure_running explode with the busy-string contract.
    monkeypatch.setattr(
        reranker_daemon._manager,
        "ensure_running",
        mock.MagicMock(side_effect=RuntimeError("GPU busy: blocked by foo")),
    )

    h = FakeHandler()
    h._proxy()

    assert captured.get("code") == 503, f"Expected 503, got {captured.get('code')}"


# ---------------------------------------------------------------------------
# 5. atexit handler releases on normal exit
# ---------------------------------------------------------------------------


def test_atexit_releases_claim_on_normal_exit(
    fake_broker: _FakeBroker,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The module's atexit hook (registered in __main__) must release the claim."""
    from scripts import reranker_daemon

    mgr = _fresh_manager(monkeypatch)
    fake_popen = mock.MagicMock(return_value=_make_fake_popen())
    monkeypatch.setattr(reranker_daemon.subprocess, "Popen", fake_popen)

    mgr._spawn()
    assert mgr._claim == os.getpid()

    # Build the safe-release closure the daemon registers, point it at our
    # local manager, and invoke directly (don't kill the test process).
    safe_release = reranker_daemon._build_safe_release_on_exit(mgr)
    safe_release()

    assert mgr._claim is None, "_claim must be None after safe_release"
    assert os.getpid() in fake_broker.release_calls, (
        "broker.release must be called by the atexit handler"
    )


# ---------------------------------------------------------------------------
# 6. Double-release (SIGTERM then atexit) is safe.
# ---------------------------------------------------------------------------


def test_idempotent_release_via_sigterm_then_atexit(
    fake_broker: _FakeBroker,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import reranker_daemon

    mgr = _fresh_manager(monkeypatch)
    fake_popen = mock.MagicMock(return_value=_make_fake_popen())
    monkeypatch.setattr(reranker_daemon.subprocess, "Popen", fake_popen)

    mgr._spawn()

    # First call simulates the SIGTERM handler.
    mgr._shutdown()
    # Second call simulates atexit firing afterwards — must not raise.
    safe_release = reranker_daemon._build_safe_release_on_exit(mgr)
    safe_release()  # must not raise; _claim is already None

    assert mgr._claim is None


# ---------------------------------------------------------------------------
# 7. Post-healthy backend crash → next _spawn does not wedge (gh #19).
# ---------------------------------------------------------------------------


def test_spawn_after_backend_died_does_not_wedge(
    fake_broker: _FakeBroker,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repro for gh #19: backend dies after _ready.set() (no path through
    _shutdown / _reap_failed_spawn), leaving self._claim set. The next
    ensure_running → _spawn must not wedge — the broker's idempotent
    same-PID acquire lets the second _spawn succeed.
    """
    from scripts import reranker_daemon

    mgr = _fresh_manager(monkeypatch)

    # Distinct fake-Popens so we can tell the second spawn happened.
    first_proc = _make_fake_popen()
    second_proc = _make_fake_popen()
    second_proc.pid = 525252
    fake_popen = mock.MagicMock(side_effect=[first_proc, second_proc])
    monkeypatch.setattr(reranker_daemon.subprocess, "Popen", fake_popen)

    # First spawn: claim acquired, _claim set.
    mgr._spawn()
    assert mgr._claim == os.getpid()

    # Simulate backend dying post-healthy: poll() now reports the proc gone,
    # but _claim is still set (no cleanup path fired). This is the exact
    # state the daemon was wedged in on 2026-05-10.
    first_proc.poll.return_value = 1

    # Second spawn must succeed: _spawn releases stale claim before re-acquiring.
    mgr._spawn()

    # Two acquire calls: one for first spawn, one for second.
    assert fake_popen.call_count == 2
    assert len(fake_broker.acquire_calls) == 2
    # The second acquire comes after a release of the stale claim.
    assert len(fake_broker.release_calls) >= 1, "stale claim must be released before second acquire"
    # _claim is still set (second spawn succeeded).
    assert mgr._claim == os.getpid()

    # Cleanup.
    fake_broker.release(pid=os.getpid())
