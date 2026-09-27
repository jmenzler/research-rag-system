"""Acceptance tests for gh #15 — reranker proxy idle-timer leak fixes.

Three orphan-leak vectors covered:

1. ``_poll_until_healthy`` times out → ``_proc`` and ``_claim`` are NOT
   cleaned up by the existing code path. A live llama-server process keeps
   running and pinning ~7 GiB of VRAM. The fix: on timeout, terminate the
   spawned proc and release the claim.

2. ``_poll_until_healthy`` succeeds → ``_idle_timer`` must be armed
   immediately so an idle backend gets evicted after ``IDLE_TIMEOUT_S``,
   even if the very first proxied request raises BrokenPipeError before
   ``touch()`` runs.

3. Startup orphan sweep: a previous daemon's llama-server may still be
   running with PPID=1 after a parent crash. The daemon must SIGTERM such
   orphans on startup so a fresh spawn isn't joined by a phantom backend.

4. ``/health`` proxy endpoint exposes the live state so operators don't
   have to grep /var/log.

The ``fake_broker`` fixture (conftest.py) injects a stub HttpClient into
``reranker_daemon._BROKER`` so tests run without a real broker daemon.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any
from unittest import mock

import pytest

if TYPE_CHECKING:
    from tests.conftest import _FakeBroker

if TYPE_CHECKING:
    from scripts.reranker_daemon import RerankerManager


def _make_fake_popen(*, alive: bool = True) -> mock.MagicMock:
    fake_proc = mock.MagicMock()
    fake_proc.poll.return_value = None if alive else 1
    fake_proc.pid = 4242
    fake_proc.terminate.return_value = None
    fake_proc.wait.return_value = 0
    fake_proc.kill.return_value = None
    return fake_proc


def _fresh_manager(monkeypatch: pytest.MonkeyPatch) -> RerankerManager:
    from scripts import reranker_daemon

    monkeypatch.setattr(reranker_daemon, "LOG_DIR", "/tmp")
    # Make timeout deterministic & fast.
    monkeypatch.setattr(reranker_daemon, "STARTUP_TIMEOUT_S", 1)
    monkeypatch.setattr(reranker_daemon, "IDLE_TIMEOUT_S", 60)
    return reranker_daemon.RerankerManager()


# ---------------------------------------------------------------------------
# 1. Health-probe timeout must NOT leak _proc + _claim
# ---------------------------------------------------------------------------


class TestHealthProbeTimeoutCleanup:
    """If the backend never becomes healthy, ensure_running raises — but
    today the spawned llama-server keeps running and the GPU claim is
    held until the daemon process itself exits. Both must be released
    on the timeout path."""

    def test_timeout_terminates_spawned_proc(
        self,
        fake_broker: _FakeBroker,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from scripts import reranker_daemon

        mgr = _fresh_manager(monkeypatch)
        fake_proc = _make_fake_popen()
        monkeypatch.setattr(
            reranker_daemon.subprocess,
            "Popen",
            mock.MagicMock(return_value=fake_proc),
        )
        # Force _poll_until_healthy to be a no-op (never sets _ready).
        # ensure_running's wait(timeout=STARTUP_TIMEOUT_S) will then time out.
        monkeypatch.setattr(
            reranker_daemon.RerankerManager,
            "_poll_until_healthy",
            lambda self: None,
        )

        with pytest.raises(RuntimeError, match="did not become healthy"):
            mgr.ensure_running()

        # Today's bug: the proc is left alive. Fix: terminate must have been
        # called so the orphan llama-server is reaped.
        assert fake_proc.terminate.called, (
            "spawned llama-server must be terminated on health-probe timeout"
        )

    def test_timeout_releases_gpu_claim(
        self,
        fake_broker: _FakeBroker,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from scripts import reranker_daemon

        mgr = _fresh_manager(monkeypatch)
        fake_proc = _make_fake_popen()
        monkeypatch.setattr(
            reranker_daemon.subprocess,
            "Popen",
            mock.MagicMock(return_value=fake_proc),
        )
        monkeypatch.setattr(
            reranker_daemon.RerankerManager,
            "_poll_until_healthy",
            lambda self: None,
        )

        with pytest.raises(RuntimeError, match="did not become healthy"):
            mgr.ensure_running()

        # The GPU claim must be released — otherwise the next ragctl run
        # blocks forever waiting for a phantom holder.
        assert os.getpid() in fake_broker.release_calls, (
            "GPU claim must be released on health-probe timeout"
        )
        assert mgr._claim is None, "_claim must be cleared on timeout"

    def test_timeout_clears_proc_handle(
        self,
        fake_broker: _FakeBroker,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Subsequent ensure_running() calls must be free to spawn fresh —
        not blocked by a stale ``_proc`` whose process is already SIGTERM'd."""
        from scripts import reranker_daemon

        mgr = _fresh_manager(monkeypatch)
        fake_proc = _make_fake_popen()
        monkeypatch.setattr(
            reranker_daemon.subprocess,
            "Popen",
            mock.MagicMock(return_value=fake_proc),
        )
        monkeypatch.setattr(
            reranker_daemon.RerankerManager,
            "_poll_until_healthy",
            lambda self: None,
        )

        with pytest.raises(RuntimeError, match="did not become healthy"):
            mgr.ensure_running()

        # _proc must be None or its poll() must show exited (so the
        # ensure_running guard at the top doesn't misclassify it as alive).
        assert mgr._proc is None or mgr._proc.poll() is not None


# ---------------------------------------------------------------------------
# 2. _poll_until_healthy success must arm the idle timer
# ---------------------------------------------------------------------------


class TestIdleTimerArmedOnHealthy:
    """The idle timer arms inside _poll_until_healthy itself (not just
    inside touch()), so a backend that became healthy is guaranteed an
    eventual eviction even if the first proxied request fails before
    touch() ever runs."""

    def test_idle_timer_armed_when_health_probe_succeeds(
        self,
        fake_broker: _FakeBroker,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from scripts import reranker_daemon

        mgr = _fresh_manager(monkeypatch)
        fake_proc = _make_fake_popen()
        monkeypatch.setattr(
            reranker_daemon.subprocess,
            "Popen",
            mock.MagicMock(return_value=fake_proc),
        )

        # Stub _http_probe to succeed immediately so _poll_until_healthy
        # takes the success branch on its very first iteration.
        class _FakeResp:
            status = 200

            def __enter__(self) -> _FakeResp:
                return self

            def __exit__(self, *_a: object) -> None:
                pass

        monkeypatch.setattr(
            reranker_daemon.urllib.request,
            "urlopen",
            mock.MagicMock(return_value=_FakeResp()),
        )

        mgr.ensure_running()

        # The backend is healthy and the idle timer must be armed —
        # this is the load-bearing fix from gh #15: don't depend on
        # the request-handler path arming the timer, do it at health-set time.
        assert mgr._idle_timer is not None, "idle timer must be armed once backend is healthy"

        # Cleanup: cancel timer + release claim so the test process exits cleanly.
        mgr._idle_timer.cancel()
        if mgr._claim is not None:
            fake_broker.release(pid=mgr._claim)


# ---------------------------------------------------------------------------
# 3. Startup orphan sweep
# ---------------------------------------------------------------------------


class TestStartupOrphanSweep:
    """A previous proxy crashed, leaving its llama-server reparented to
    init. The new proxy must SIGTERM it on startup so the new spawn
    doesn't join a phantom backend pinning the GPU."""

    def test_sweep_terminates_orphan_with_ppid_one(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from scripts import reranker_daemon

        # Force Linux branch so the sweep runs on Mac CI too.
        monkeypatch.setattr(reranker_daemon.sys, "platform", "linux")

        # Pretend pgrep finds two llama-server PIDs.
        fake_pgrep = mock.MagicMock()
        fake_pgrep.return_value = mock.MagicMock(
            stdout="1234\n5678\n",
            returncode=0,
        )
        monkeypatch.setattr(reranker_daemon.subprocess, "run", fake_pgrep)

        # 1234 is an orphan (ppid=1), 5678 is owned by current proc.
        ppids = {1234: 1, 5678: os.getpid()}

        def fake_ppid_of(pid: int) -> int | None:
            return ppids.get(pid)

        monkeypatch.setattr(reranker_daemon, "_ppid_of", fake_ppid_of, raising=False)

        kills: list[tuple[int, int]] = []

        def fake_kill(pid: int, sig: int) -> None:
            kills.append((pid, sig))

        monkeypatch.setattr(reranker_daemon.os, "kill", fake_kill)

        # Sweep is a module-level helper; on first call it must terminate
        # only the ppid=1 orphan, never the live sibling.
        reranker_daemon._sweep_orphan_llama_server()

        # Only the orphan was killed.
        signal_sent = {pid for pid, _ in kills}
        assert 1234 in signal_sent, "orphan llama-server (ppid=1) must be killed"
        assert 5678 not in signal_sent, "live sibling (ppid=self) must NOT be killed"


# ---------------------------------------------------------------------------
# 4. /health proxy endpoint
# ---------------------------------------------------------------------------


class TestProxyHealthEndpoint:
    """``GET /proxy/health`` returns JSON describing the proxy's view of its
    backend so MCP clients and operators can detect undead-backend states
    without grep."""

    def test_health_endpoint_returns_json_state(
        self,
        fake_broker: _FakeBroker,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from scripts import reranker_daemon

        mgr = _fresh_manager(monkeypatch)
        fake_proc = _make_fake_popen()
        monkeypatch.setattr(
            reranker_daemon.subprocess,
            "Popen",
            mock.MagicMock(return_value=fake_proc),
        )

        class _FakeResp:
            status = 200

            def __enter__(self) -> _FakeResp:
                return self

            def __exit__(self, *_a: object) -> None:
                pass

        monkeypatch.setattr(
            reranker_daemon.urllib.request,
            "urlopen",
            mock.MagicMock(return_value=_FakeResp()),
        )

        # Drive the manager into a healthy state.
        # Re-point the module-level singleton at our local manager so
        # ProxyHandler._health() reads the same state.
        monkeypatch.setattr(reranker_daemon, "_manager", mgr)
        mgr.ensure_running()

        # Exercise the /health route via a fake handler.
        captured: dict[str, Any] = {}

        class FakeHandler(reranker_daemon.ProxyHandler):
            def __init__(self) -> None:  # noqa: D401  -- bypass socket init
                self.command = "GET"
                self.path = "/proxy/health"
                self.headers = {}  # type: ignore[assignment]

            def send_response(self, code: int, message: str | None = None) -> None:
                captured["code"] = code

            def send_header(self, key: str, value: str) -> None:
                captured.setdefault("headers", []).append((key, value))

            def end_headers(self) -> None:
                captured["end_headers"] = True

            class _Buf:
                buf = b""

                def write(self, data: bytes) -> None:
                    type(self).buf += data

            wfile = _Buf()  # type: ignore[assignment]

        h = FakeHandler()
        h.do_GET()

        assert captured.get("code") == 200, (
            f"/proxy/health must return 200, got {captured.get('code')!r}"
        )
        # Body must be parseable JSON with the contract keys.
        import json as _json

        body = _json.loads(FakeHandler._Buf.buf.decode())  # type: ignore[union-attr]
        assert "backend_alive" in body
        assert "backend_pid" in body
        assert "idle_timer_armed" in body
        assert body["backend_alive"] is True
        assert body["backend_pid"] == fake_proc.pid
        assert body["idle_timer_armed"] is True

        # cleanup
        if mgr._idle_timer is not None:
            mgr._idle_timer.cancel()
        if mgr._claim is not None:
            fake_broker.release(pid=mgr._claim)
        # Reset shared buf for any subsequent run.
        FakeHandler._Buf.buf = b""  # type: ignore[union-attr]
