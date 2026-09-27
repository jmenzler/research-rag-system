#!/usr/bin/env python3
"""On-demand reranker proxy daemon.

Sits on RERANKER_PROXY_PORT (:8090). Spawns llama-server on RERANKER_BACKEND_PORT
(:8091) on the first request after idle; kills it after RERANKER_IDLE_TIMEOUT
seconds of silence. Callers see no change — same port, same llama.cpp API.

Usage:
    python scripts/reranker_daemon.py          # production (via systemd)
    RERANKER_IDLE_TIMEOUT=300 python ...       # 5-min idle for testing
"""
from __future__ import annotations

import atexit
import http.server
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path
from socketserver import ThreadingMixIn
from typing import Protocol

import requests  # type: ignore[import-untyped]


class _BrokerClient(Protocol):
    def acquire(self, **kwargs: object) -> object: ...

    def release(self, *, pid: int) -> bool: ...


_BROKER: _BrokerClient | None

try:
    from gpu_broker import GPUBusy  # type: ignore[import-not-found]
    from gpu_broker.client import HttpClient  # type: ignore[import-not-found]
    from gpu_broker.client.exceptions import (  # type: ignore[import-not-found]
        GPUBusy as _BrokerGPUBusy,
    )
except ModuleNotFoundError:
    GPUBusy = RuntimeError
    _BrokerGPUBusy = RuntimeError
    _BROKER = None
else:
    _BROKER = HttpClient.from_url(os.environ.get("GPU_BROKER_URL", "http://127.0.0.1:7090"))


def _require_broker() -> _BrokerClient:
    if _BROKER is None:
        raise RuntimeError(
            "Starting the reranker GPU daemon requires the optional gpu-broker package."
        )
    return _BROKER

# `scripts/` is not a package on the import path when this file is run as
# `python scripts/reranker_daemon.py` (production / systemd mode). Pytest
# imports it as `scripts.reranker_daemon` via the project's pyproject root,
# but production needs the explicit sys.path tweak so `from src...` resolves.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

PROXY_PORT = int(os.getenv("RERANKER_PROXY_PORT", "8090"))
BACKEND_PORT = int(os.getenv("RERANKER_BACKEND_PORT", "8091"))
IDLE_TIMEOUT_S = int(os.getenv("RERANKER_IDLE_TIMEOUT", str(30 * 60)))
STARTUP_TIMEOUT_S = 90
LLAMA_BIN = os.getenv("LLAMA_SERVER_BIN", "llama-server")
MODEL_PATH = os.getenv("RERANKER_MODEL_PATH", "")
LOG_DIR = os.getenv("RERANKER_LOG_DIR", "logs/reranker")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [reranker-daemon] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)


class RerankerManager:
    """Lazy-start / idle-shutdown lifecycle manager for llama-server."""

    def __init__(self) -> None:
        self._proc: subprocess.Popen[bytes] | None = None
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._idle_timer: threading.Timer | None = None
        self._claim: int | None = None

    def ensure_running(self) -> None:
        """Block until the backend is healthy, starting it first if needed.

        Translates ``GPUBusy`` (raised by the registry when a higher-priority
        consumer is active) into ``RuntimeError('GPU busy: ...')`` so the
        proxy handler returns HTTP 503. The ``with self._lock:`` block
        propagates the exception out, releasing the lock on stack unwind.
        """
        try:
            with self._lock:
                if (
                    self._proc is not None
                    and self._proc.poll() is None
                    and self._ready.is_set()
                ):
                    return
                if self._proc is None or self._proc.poll() is not None:
                    self._ready.clear()
                    self._spawn()
                    threading.Thread(target=self._poll_until_healthy, daemon=True).start()
        except GPUBusy as exc:
            raise RuntimeError(f"GPU busy: {exc}") from exc
        if not self._ready.wait(timeout=STARTUP_TIMEOUT_S):
            # Health probe timed out. The spawned llama-server is either still
            # warming up (will eventually become healthy with no idle timer
            # armed → orphan + GPU leak) or genuinely stuck. Either way we
            # must terminate it and release the GPU claim, otherwise the
            # next ensure_running() guard at the top mis-classifies the
            # stale _proc as healthy and we accumulate orphans.
            self._reap_failed_spawn()
            raise RuntimeError(f"llama-server did not become healthy within {STARTUP_TIMEOUT_S}s")

    def _reap_failed_spawn(self) -> None:
        """Cleanup path for spawn → unhealthy transition.

        Called when the health probe times out. Terminates the spawned
        process, releases the GPU claim, and clears local state so the
        next ensure_running() can spawn fresh. Best-effort — never raises.
        """
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                pid = self._proc.pid
                log.warning(
                    "Reaping unhealthy llama-server (PID %d) after %ds health timeout",
                    pid, STARTUP_TIMEOUT_S,
                )
                try:
                    self._proc.terminate()
                    try:
                        self._proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self._proc.kill()
                        self._proc.wait()
                except Exception as exc:  # noqa: BLE001
                    log.warning("terminate during reap raised: %s", exc)
            self._proc = None
            self._ready.clear()
            if self._claim is not None:
                try:
                    _require_broker().release(pid=self._claim)
                except Exception as exc:  # noqa: BLE001
                    log.warning("gpu release during reap raised: %s", exc)
                self._claim = None

    def _spawn(self) -> None:
        broker = _require_broker()
        if not MODEL_PATH.strip():
            raise RuntimeError("RERANKER_MODEL_PATH must point to a local GGUF model.")
        # If our backend died post-healthy (no path through _shutdown or
        # _reap_failed_spawn), a stale registry row from the previous
        # _spawn would block this acquire (gh #19). Release best-effort
        # before re-acquiring; gpu_registry.acquire is also idempotent on
        # same-PID + same-role as a defence in depth.
        if self._claim is not None:
            try:
                broker.release(pid=self._claim)
            except Exception as exc:  # noqa: BLE001  -- best-effort
                log.warning("gpu release of stale claim raised: %s", exc)
            self._claim = None
        # P0 path: never block. Broker's attempt_evictions will evict idle
        # ollama (yield_mode='evict') before raising GPUBusy. If ollama cannot
        # be evicted fast enough, _BrokerGPUBusy propagates (caught in ensure_running).
        try:
            broker.acquire(
                pid=os.getpid(),
                role="reranker",
                priority=0,
                vram_mib=7000,
                yield_mode="evict",
                yield_endpoint=f"http://127.0.0.1:{PROXY_PORT}/_yield",
                yield_cost_s=2,
                blocking=False,
            )
        except _BrokerGPUBusy as exc:
            log.warning("reranker: broker.acquire failed: %s", exc)
            raise GPUBusy(str(exc)) from exc
        self._claim = os.getpid()
        try:
            os.makedirs(LOG_DIR, exist_ok=True)
            log_path = os.path.join(LOG_DIR, f"reranker_{int(time.time())}.log")
            log.info("Spawning llama-server on :%d  log=%s", BACKEND_PORT, log_path)
            log_f = open(log_path, "ab")
            self._proc = subprocess.Popen(
                [
                    LLAMA_BIN, "--model", MODEL_PATH, "--alias", "Qwen3-Reranker-4B",
                    "--host", "127.0.0.1", "--port", str(BACKEND_PORT),
                    "-ngl", "99", "--ctx-size", "4096", "-b", "4096", "-ub", "4096",
                    "--reranking", "--pooling", "rank", "--embedding",
                    "--threads", "4", "--parallel", "1",
                ],
                stdout=log_f,
                stderr=subprocess.STDOUT,
            )
            log_f.close()  # safe: Popen already dup'd the fd into the child
        except Exception:
            # Roll back the claim so a retry isn't blocked by our orphan row.
            if self._claim is not None:
                try:
                    _require_broker().release(pid=self._claim)
                except Exception as rel_exc:  # noqa: BLE001
                    log.warning("gpu release rollback raised: %s", rel_exc)
                self._claim = None
            raise

    def _poll_until_healthy(self) -> None:
        deadline = time.time() + STARTUP_TIMEOUT_S
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(
                    f"http://localhost:{BACKEND_PORT}/health", timeout=2
                ) as r:
                    if r.status == 200:
                        pid = self._proc.pid if self._proc else -1
                        log.info("llama-server ready (PID %d)", pid)
                        # Arm the idle timer at health-set time so the
                        # backend is guaranteed eviction even if the very
                        # first proxied request raises BrokenPipeError
                        # before touch() can fire (gh #15).
                        self.touch()
                        self._ready.set()
                        return
            except Exception:
                pass
            time.sleep(2)
        log.error("llama-server did not become healthy within %ds", STARTUP_TIMEOUT_S)

    def touch(self) -> None:
        """Reset the idle shutdown timer on every handled request."""
        if self._idle_timer is not None:
            self._idle_timer.cancel()
        t = threading.Timer(IDLE_TIMEOUT_S, self._shutdown)
        t.daemon = True
        t.start()
        self._idle_timer = t

    def _shutdown(self) -> None:
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                # Process already gone; still release any orphan claim so the
                # registry doesn't carry a stale row.
                if self._claim is not None:
                    try:
                        _require_broker().release(pid=self._claim)
                    except Exception as exc:  # noqa: BLE001
                        log.warning("gpu release on proc-gone path raised: %s", exc)
                    self._claim = None
                return
            log.info("Idle timeout — stopping llama-server (PID %d)", self._proc.pid)
            self._ready.clear()
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
            self._proc = None
            if self._claim is not None:
                try:
                    _require_broker().release(pid=self._claim)
                except Exception as exc:  # noqa: BLE001
                    log.warning("gpu release during shutdown raised: %s", exc)
                self._claim = None
            log.info("llama-server stopped; GPU released")


def _ppid_of(pid: int) -> int | None:
    """Return parent PID of *pid*, or None if unreadable.

    Linux-only via /proc/{pid}/status. Other platforms return None
    (caller treats as "unknown" → does not touch the process).
    """
    if sys.platform != "linux":
        return None
    try:
        for line in Path(f"/proc/{pid}/status").read_text(errors="replace").splitlines():
            if line.startswith("PPid:"):
                return int(line.split()[1])
    except (OSError, ValueError):
        return None
    return None


def _sweep_orphan_llama_server() -> None:
    """SIGTERM llama-server processes whose parent is init (ppid=1).

    A previous proxy crashed without a clean shutdown; its child
    llama-server got reparented to init and is still pinning ~7 GiB
    of VRAM. We must reap it before spawning a fresh backend, otherwise
    the new daemon stacks on top.

    Linux-only (no /proc on macOS). Live siblings (ppid != 1) are
    NEVER touched — that would turn into mutual destruction across
    concurrent ragctl runs.
    """
    if sys.platform != "linux":
        return
    try:
        result = subprocess.run(
            ["pgrep", "-f", "llama-server.*Qwen3-Reranker"],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        log.warning("pgrep for orphan llama-server failed: %s", exc)
        return
    if result.returncode not in (0, 1):  # 1 = no matches, 0 = matches
        return
    for pid_str in result.stdout.strip().splitlines():
        try:
            pid = int(pid_str.strip())
        except ValueError:
            continue
        if pid == os.getpid():
            continue
        ppid = _ppid_of(pid)
        if ppid != 1:
            continue
        try:
            os.kill(pid, signal.SIGTERM)
            log.warning("Killed orphaned llama-server pid=%d (ppid=1)", pid)
        except (ProcessLookupError, PermissionError):
            pass


_manager = RerankerManager()


def _yield_teardown(mgr: RerankerManager) -> None:
    """Background: terminate llama-server child and release the GPU claim.

    Called from a daemon thread when the broker sends POST /_yield.
    Best-effort — logs exceptions, never raises so the thread exits cleanly.
    """
    try:
        mgr._shutdown()
    except Exception as exc:  # noqa: BLE001
        log.exception("yield teardown: error during _shutdown: %s", exc)
        # If _shutdown failed partway, still try to release any remaining claim.
        claim = getattr(mgr, "_claim", None)
        if claim is not None:
            try:
                _require_broker().release(pid=claim)
            except Exception as rel_exc:  # noqa: BLE001
                log.warning("yield teardown: release fallback raised: %s", rel_exc)
            mgr._claim = None


class ProxyHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        if self.path == "/proxy/health":
            self._health()
            return
        if self.path == "/_yield":
            self._yield()
            return
        self._proxy()

    def _yield(self) -> None:
        """Handle POST /_yield — respond 202 immediately, tear down async.

        The broker calls this endpoint when it needs to evict the reranker
        (yield_mode='evict'). We respond 202 within 1s per the yield protocol
        and do the actual teardown in a daemon thread.
        """
        self.send_response(202)
        self.send_header("Content-Length", "0")
        self.end_headers()
        threading.Thread(
            target=_yield_teardown,
            args=(_manager,),
            daemon=True,
        ).start()

    def do_GET(self) -> None:
        if self.path == "/proxy/health":
            self._health()
            return
        self._proxy()

    def _health(self) -> None:
        """Return JSON describing the proxy's view of its backend.

        Operators / MCP clients use this instead of grepping journalctl
        to detect undead-backend states (gh #15). Cheap: no backend HTTP
        call, just reads in-process state.
        """
        proc = _manager._proc
        backend_alive = proc is not None and proc.poll() is None
        body = json.dumps({
            "backend_alive": backend_alive,
            "backend_pid": proc.pid if (backend_alive and proc is not None) else None,
            "idle_timer_armed": _manager._idle_timer is not None,
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _proxy(self) -> None:
        try:
            _manager.ensure_running()
        except RuntimeError as exc:
            self._respond(503, str(exc))
            return

        _manager.touch()

        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else None
        fwd_headers = {
            k: v for k, v in self.headers.items()
            if k.lower() not in ("host", "content-length", "transfer-encoding")
        }

        try:
            resp = requests.request(
                self.command,
                f"http://localhost:{BACKEND_PORT}{self.path}",
                headers=fwd_headers,
                data=body,
                timeout=120,
            )
        except Exception as exc:
            log.exception("Backend request failed: %s", exc)
            self._respond(502, str(exc))
            return

        self.send_response(resp.status_code)
        for k, v in resp.headers.items():
            if k.lower() in ("content-type", "content-length", "content-encoding"):
                self.send_header(k, v)
        self.end_headers()
        self.wfile.write(resp.content)

    def _respond(self, code: int, msg: str) -> None:
        body = msg.encode()
        self.send_response(code)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: object) -> None:
        log.debug("HTTP %s", fmt % args)


class ThreadedHTTPServer(ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


def _build_safe_release_on_exit(manager: RerankerManager) -> Callable[[], None]:
    """Return a closure suitable for atexit / SIGTERM that always returns cleanly.

    Calls ``manager._shutdown()`` and swallows any exception — running this
    twice (e.g. SIGTERM handler then atexit) is a no-op the second time
    because ``_shutdown`` clears ``self._proc`` and ``self._claim``.
    """
    def _safe_release() -> None:
        try:
            manager._shutdown()
        except Exception as exc:  # noqa: BLE001  -- best-effort, never crash exit
            log.warning("shutdown on exit raised: %s", exc)

    return _safe_release


if __name__ == "__main__":
    # Reap orphan llama-server processes from a previous crashed daemon
    # before listening — otherwise the new spawn stacks on top and the
    # GPU still sees the old 7 GiB pinned (gh #15).
    _sweep_orphan_llama_server()

    server = ThreadedHTTPServer(("127.0.0.1", PROXY_PORT), ProxyHandler)
    log.info(
        "Reranker proxy listening on :%d → backend :%d  idle_timeout=%ds",
        PROXY_PORT, BACKEND_PORT, IDLE_TIMEOUT_S,
    )

    _safe_release_on_exit = _build_safe_release_on_exit(_manager)
    atexit.register(_safe_release_on_exit)
    signal.signal(signal.SIGTERM, lambda *_: _safe_release_on_exit())

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Shutting down proxy.")
