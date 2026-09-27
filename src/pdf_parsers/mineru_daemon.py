# long-ok-file
"""Long-lived mineru-api pipeline daemon lifecycle.

Manages a singleton `mineru-api` process (from `.venv-mineru/bin/`) whose
pipeline backend cold-starts once per orchestrator run instead of once per PDF.

Mirrors the VLM server pattern in `mineru.py` but without vLLM-specific bits —
no `--enable-vlm-preload`, no sleep/wake, no vLLM env. Uses `_BASE_ENV`.

State is persisted to `$RAG_MINERU_STATE` (default: `~/.cache/rag-system/mineru-pipeline.json`)
so parallel parse workers in a ProcessPoolExecutor can discover the daemon URL
via a lightweight `pipeline_daemon_url()` read.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlsplit


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
            "Starting the MinerU GPU daemon requires the optional gpu-broker package."
        )
    return _BROKER


logger = logging.getLogger(__name__)

# Daemon-level GPU claim — held for the lifetime of one ragctl run. Set in
# start_pipeline_daemon() after a successful subprocess.Popen + _wait_for_api,
# cleared in stop_pipeline_daemon(). Module-level (not per-call) because
# start/stop are called by separate callers in runner.py.
_RUN_CLAIM: int | None = None

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_MINERU_API_BIN = _REPO_ROOT / ".venv-mineru" / "bin" / "mineru-api"

# Prefer reclaiming the restartable parser under Linux memory pressure.
_OOM_SCORE_ADJ = os.environ.get("RAG_MINERU_OOM_SCORE_ADJ", "500")

# mineru-api writes per-parse scratch (a copy of the input PDF + every extracted
# image, ~97% of which the chunker discards) under MINERU_API_OUTPUT_ROOT —
# default "./output", inheriting ragctl's CWD so it lands inside the repo. We
# never read it (content_list comes back in the HTTP body) and it is never
# reclaimed: the daemon is SIGKILLed per run before mineru-api's 24h in-process
# GC fires, and upstream heap-corruption segfaults skip the GC entirely.
# Unbounded it grew to 14G. Relocate outside the repo, wipe on each cold spawn
# (purges leftovers from crashed prior runs), and shorten the in-process
# retention so completed-task dirs are reaped mid-run instead of after 24h. Set
# RAG_KEEP_PARSE_SCRATCH=1 to keep the scratch (skip wipe, leave 24h retention)
# when debugging a parse. Root overridable via $RAG_MINERU_SCRATCH_ROOT.
_SCRATCH_ROOT = Path(
    os.environ.get("RAG_MINERU_SCRATCH_ROOT")
    or str(Path.home() / ".cache" / "rag-system" / "mineru-scratch")
).expanduser()
_KEEP_SCRATCH = bool(os.environ.get("RAG_KEEP_PARSE_SCRATCH"))
_SCRATCH_RETENTION_SECONDS = "300"


_BASE_ENV: dict[str, str] = {
    **os.environ,
    # Reduce GPU memory fragmentation:
    #   expandable_segments        — grow one segment per stream instead of
    #                                scattering allocations across many segments.
    #   max_split_size_mb:128      — prevent blocks >128 MiB from splitting,
    #                                avoids the fragmentation death spiral.
    #   garbage_collection_threshold:0.6 — when GPU hits 60% usage, actively
    #                                reclaim cached freed blocks before OOM.
    "PYTORCH_CUDA_ALLOC_CONF":
        "expandable_segments:True,"
        "max_split_size_mb:128,"
        "garbage_collection_threshold:0.6",
}
if sys.platform == "darwin":
    _BASE_ENV["DYLD_LIBRARY_PATH"] = "/opt/homebrew/opt/expat/lib"

# Off by default. Set RAG_MINERU_MALLOC_CHECK=1 (or any truthy) when reproducing
# the upstream glibc heap-corruption SIGABRTs (gh #14 Bug A) for vendor escalation.
# MALLOC_CHECK_=3 makes glibc abort immediately at the first detected corruption
# with a usable backtrace, instead of crashing later in unrelated code.
if os.environ.get("RAG_MINERU_MALLOC_CHECK"):
    _BASE_ENV["MALLOC_CHECK_"] = "3"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _state_file_path() -> Path:
    env = os.environ.get("RAG_MINERU_STATE")
    if env:
        return Path(env)
    return Path.home() / ".cache" / "rag-system" / "mineru-pipeline.json"


def pipeline_daemon_url() -> str | None:
    """Return the pipeline daemon's base URL if a live daemon exists, else None.

    Read-only lookup used by parse workers to decide between the warm daemon
    path and the CLI subprocess fallback.

    "Live" means the PID exists AND is not a zombie. ``os.kill(pid, 0)``
    succeeds on zombie processes so we additionally read /proc/<pid>/status
    on Linux to filter Z-state. (Non-Linux: PID-existence is enough.)
    """
    sf = _state_file_path()
    if not sf.exists():
        return None
    try:
        data: dict[str, Any] = json.loads(sf.read_text())
        url: str = data["url"]
        pid: int = data["pid"]
    except (json.JSONDecodeError, KeyError):
        return None
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return None
    # On Linux, ignore zombies — they pass kill(pid,0) but their socket is dead.
    if sys.platform == "linux":
        try:
            status_text = Path(f"/proc/{pid}/status").read_text(errors="replace")
            for line in status_text.splitlines():
                if line.startswith("State:") and " Z" in line:
                    return None  # zombie — treat as dead
        except OSError:
            pass  # /proc not readable — fall through to "alive"
    return url


def _socket_open(host: str, port: int, timeout: float = 2.0) -> bool:
    """True iff a TCP connection to the daemon's listening socket succeeds.

    A daemon mid-parse can't serve an HTTP *request* (its event loop is blocked
    on synchronous OCR), but its listening socket stays open — the kernel
    completes the handshake into the accept backlog regardless. A crashed daemon
    leaves no socket, so connect() gets ECONNREFUSED. This is the discriminator
    an HTTP-response probe lacks: it separates "busy" (socket open) from "dead"
    (refused) without the ambiguous log-mtime heartbeat, which surviving worker
    processes keep warm even after the API segfaults.
    """
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except ConnectionRefusedError:
        return False
    except OSError:
        # Socket exists but the accept queue is slow, or a transient error.
        # Conservative: a real death surfaces as ECONNREFUSED or a dead PID, so
        # never false-kill here.
        return True


def is_alive() -> bool:
    """True iff the pipeline daemon is alive — responsive OR busy, not crashed.

    Layers:
      1. State file exists and is parseable
      2. Recorded PID exists and is not a zombie (filters Linux Z-state)
      3. EITHER HTTP `/docs` responds within 2s (idle/responsive daemon)
         OR the listening socket still accepts a TCP connection (busy daemon
         mid-parse — blocked on OCR so it can't serve HTTP, but the kernel still
         accepts connections; a crashed API's socket is gone → ECONNREFUSED).

    Socket-connectability is the busy-vs-dead discriminator. An earlier version
    used daemon log freshness instead, but surviving worker processes keep the
    log warm after the API segfaults (gh #14), masking the death and blocking
    the crasher-tracker respawn/quarantine recovery.
    """
    sf = _state_file_path()
    if not sf.exists():
        return False
    try:
        data: dict[str, Any] = json.loads(sf.read_text())
        url: str = data["url"]
        pid: int = data["pid"]
    except (json.JSONDecodeError, KeyError, OSError):
        return False
    if not _pid_alive(pid):
        return False
    if _http_probe(url, endpoint="/docs", timeout=2.0):
        return True
    parts = urlsplit(url)
    if parts.hostname is None or parts.port is None:
        return False
    return _socket_open(parts.hostname, parts.port, timeout=2.0)


def current_daemon_log_path() -> Path | None:
    """Return the log path persisted in the state file, or None if unavailable.

    Used by the supervisor on detected death to print the dying daemon's
    last lines (forensic context for glibc-SIGABRT-class crashes).
    """
    sf = _state_file_path()
    if not sf.exists():
        return None
    try:
        data: dict[str, Any] = json.loads(sf.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    raw = data.get("log_path")
    if not isinstance(raw, str):
        return None
    return Path(raw)


def _http_probe(url: str, endpoint: str = "/docs", timeout: float = 2.0) -> bool:
    """Single HTTP probe — True iff endpoint responds within ``timeout`` seconds."""
    try:
        with urllib.request.urlopen(f"{url}{endpoint}", timeout=timeout):
            return True
    except Exception:
        return False


def _wait_for_api(url: str, endpoint: str = "/docs", timeout: float = 600.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _http_probe(url, endpoint=endpoint, timeout=2.0):
            return
        time.sleep(2)
    raise RuntimeError(f"Pipeline API at {url} did not become healthy within {timeout}s")


def _prepare_scratch_root() -> Path:
    """Create — and unless RAG_KEEP_PARSE_SCRATCH, first wipe — the scratch root.

    Called only on a cold spawn (never the reused-daemon path) so the wipe can
    never pull the working dir out from under a live daemon. Returns the root to
    hand mineru-api via MINERU_API_OUTPUT_ROOT.
    """
    if not _KEEP_SCRATCH:
        shutil.rmtree(_SCRATCH_ROOT, ignore_errors=True)
    _SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
    return _SCRATCH_ROOT


def start_pipeline_daemon() -> str:
    """Start the mineru-api pipeline daemon, returning its base URL.

    Reuses an existing live daemon (via `pipeline_daemon_url()`).
    Spawns `mineru-api` on a free port with `start_new_session=True` so
    `stop_pipeline_daemon` can kill the entire process group.
    """
    # Kill orphaned mineru-api processes from prior crashed runs before
    # checking state file — each orphan holds a CUDA context, and 6+ of
    # them will OOM an 8 GB GPU even if the new daemon barely loads.
    _sweep_orphan_mineru()

    existing = pipeline_daemon_url()
    if existing is not None:
        # Reused-daemon path: another process spawned this daemon and owns the
        # GPU claim. Do NOT acquire here — _RUN_CLAIM stays None for this
        # caller. In the rare case the original spawning process died, its
        # registry row is dead-pid-cleaned on the next acquire by anyone else.
        return existing

    broker = _require_broker()

    # State file points at a dead/zombie daemon — clean up before spawning.
    # Try to SIGKILL the zombie's process group so the kernel reaps it; the
    # parent of the original spawn is the worker that called start_pipeline_daemon
    # the first time, which may not be us. SIGKILL is best-effort.
    sf = _state_file_path()
    if sf.exists():
        try:
            stale = json.loads(sf.read_text())
            stale_pid = int(stale.get("pid", 0))
            if stale_pid > 0 and _is_mineru_api(stale_pid):
                try:
                    pgid = os.getpgid(stale_pid)
                    os.killpg(pgid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
        except (json.JSONDecodeError, ValueError, KeyError):
            pass
        sf.unlink(missing_ok=True)

    scratch_root = _prepare_scratch_root()

    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    log_dir = _REPO_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"pipeline_daemon_{int(time.time())}_{port}.log"
    logger.info("Starting MinerU pipeline daemon on port %d (stdout → %s)", port, log_path)

    def _pdeath() -> None:
        """Set PR_SET_PDEATHSIG so the daemon receives SIGTERM when the
        parent process (ragctl) dies — cleanly or otherwise. Linux only;
        fails silently on other platforms or if prctl is unavailable."""
        import ctypes  # noqa: PLC0415
        import signal  # noqa: PLC0415
        try:
            libc = ctypes.CDLL("libc.so.6", use_errno=True)
            PR_SET_PDEATHSIG = 1  # noqa: N806
            libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM)
        except Exception:
            pass

    # GPU broker — daemon-level claim for the entire ragctl run via HTTP daemon
    # (not the library API) so the broker's attempt_evictions loop runs and the
    # reranker can be evicted via /_yield if priorities ever flip.
    # blocking=True so mineru polls politely if reranker (priority=0) is hot;
    # the reranker's idle-timeout bounds the wait. Cadence tunable via
    # RAG_GPU_POLL_S (default 5s). yield_mode='fail' because MinerU has no
    # in-process teardown that frees VRAM mid-parse.
    global _RUN_CLAIM
    poll_s = float(os.environ.get("RAG_GPU_POLL_S", "5.0"))
    try:
        broker.acquire(
            pid=os.getpid(),
            role="mineru",
            priority=30,
            vram_mib=6500,
            yield_mode="fail",
            blocking=True,
            poll_interval_s=poll_s,
        )
    except _BrokerGPUBusy as exc:
        logger.warning("mineru: broker.acquire failed: %s", exc)
        raise GPUBusy(str(exc)) from exc
    _RUN_CLAIM = os.getpid()

    daemon_env = {**_BASE_ENV, "MINERU_API_OUTPUT_ROOT": str(scratch_root)}
    if not _KEEP_SCRATCH:
        daemon_env["MINERU_API_TASK_RETENTION_SECONDS"] = _SCRATCH_RETENTION_SECONDS

    # Opened after a successful broker.acquire so a GPUBusy re-raise can't leak
    # this write FD in the long-lived ragctl process across supervisor respawns.
    log_handle = log_path.open("w")

    try:
        proc = subprocess.Popen(
            [str(_MINERU_API_BIN), "--port", str(port)],
            env=daemon_env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            preexec_fn=_pdeath,
        )

        # Retain the file handle so the daemon can write to it, but close our copy
        # in the parent so we don't leak (the child has its own fd).
        log_handle.close()

        # Bias the OOM killer toward this daemon. Best-effort — if /proc isn't
        # writable (e.g. inside a constrained container) we silently move on.
        if sys.platform == "linux" and _OOM_SCORE_ADJ:
            try:
                Path(f"/proc/{proc.pid}/oom_score_adj").write_text(_OOM_SCORE_ADJ)
                logger.info("Daemon oom_score_adj=%s (preferred OOM victim)", _OOM_SCORE_ADJ)
            except OSError as e:
                logger.warning(
                    "Could not set oom_score_adj for daemon pid=%d: %s", proc.pid, e
                )

        _wait_for_api(url, endpoint="/docs")

        sf = _state_file_path()
        sf.parent.mkdir(parents=True, exist_ok=True)
        sf.write_text(json.dumps({
            "url": url,
            "pid": proc.pid,
            "log_path": str(log_path),
        }))

        logger.info("Pipeline daemon ready at %s", url)
        return url
    except Exception:
        # Release the claim on any spawn-or-wait failure so a retry / second
        # ragctl run isn't blocked by an orphan registry row.
        log_handle.close()
        if _RUN_CLAIM is not None:
            broker.release(pid=_RUN_CLAIM)
            _RUN_CLAIM = None
        raise


def _pid_alive(pid: int) -> bool:
    """True if *pid* exists and is not a zombie."""
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    if sys.platform == "linux":
        try:
            status_text = Path(f"/proc/{pid}/status").read_text(errors="replace")
            for line in status_text.splitlines():
                if line.startswith("State:"):
                    if " Z" in line:
                        return False
                    break
        except OSError:
            pass
    return True


def _ppid_of(pid: int) -> int | None:
    """Return the parent PID of *pid*, or None if unreadable.

    Linux-only via /proc/{pid}/status. Other platforms return None.
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


def _is_mineru_api(pid: int) -> bool:
    """True iff *pid*'s argv contains the mineru-api binary path.

    Guards killpg paths that target a PID read from a possibly-stale state file:
    a recycled PID (its own group leader under start_new_session) would otherwise
    take down an unrelated session. Reads /proc/{pid}/cmdline (NUL-separated
    argv). Non-Linux has no /proc — returns True (no-op safety, preserving the
    prior unconditional-kill behavior on macOS dev).
    """
    if sys.platform != "linux":
        return True
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\x00", b" ")
    except OSError:
        return False
    return b".venv-mineru/bin/mineru-api" in cmdline


def _sweep_orphan_mineru() -> None:
    """Kill genuinely-orphaned mineru-api processes — never live siblings.

    A process is "orphaned" iff its parent PID is 1 (init-reparented after the
    spawning ragctl died). A live concurrent ragctl run owns its daemon and
    will be the daemon's PPid; we must NOT kill those — doing so on a shared
    box turns into mutual destruction across runs.

    PR_SET_PDEATHSIG already kills daemons on clean parent crashes, so this
    sweep only matters for kernel-level OOM kills or signals that bypass the
    pdeath path. Falls back to no-op on non-Linux (no /proc).
    """
    if sys.platform != "linux":
        return
    import subprocess  # noqa: PLC0415
    try:
        result = subprocess.run(
            ["pgrep", "-f", r"\.venv-mineru.*mineru-api"],
            capture_output=True, text=True, timeout=10,
        )
    except Exception:
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
            # Parent is alive (or unreadable) — not an orphan, skip.
            continue
        try:
            os.kill(pid, signal.SIGKILL)
            logger.warning("Killed orphaned mineru-api pid=%d (ppid=1)", pid)
        except (ProcessLookupError, PermissionError):
            pass


def stop_pipeline_daemon() -> None:
    """Kill the pipeline daemon process group and remove the state file.

    Escalates from SIGTERM to SIGKILL after 5s if the daemon doesn't respond
    (uvicorn graceful-shutdown can block on in-flight asyncio requests).

    Releases the GPU registry claim acquired in ``start_pipeline_daemon`` no
    matter which exit path is taken (no state file, malformed JSON, normal
    shutdown). Idempotent — no-op if no state file exists.
    """
    global _RUN_CLAIM
    try:
        sf = _state_file_path()
        if not sf.exists():
            return
        try:
            data = json.loads(sf.read_text())
            pid = data["pid"]
        except (json.JSONDecodeError, KeyError):
            sf.unlink(missing_ok=True)
            return

        # A stale state file may name a recycled PID that is now its own group
        # leader — killpg would take down an unrelated session. Only kill if the
        # target's argv confirms it is our mineru-api.
        if _is_mineru_api(pid):
            try:
                pgid = os.getpgid(pid)
                os.killpg(pgid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass

            # Give uvicorn graceful shutdown 5s, then escalate to SIGKILL
            for _ in range(50):
                if not _pid_alive(pid):
                    break
                time.sleep(0.1)
            else:
                logger.warning("Daemon pid=%d ignored SIGTERM — escalating to SIGKILL", pid)
                try:
                    pgid = os.getpgid(pid)
                    os.killpg(pgid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass

        sf.unlink(missing_ok=True)
        logger.info("Pipeline daemon stopped")
    finally:
        if _RUN_CLAIM is not None:
            try:
                _require_broker().release(pid=_RUN_CLAIM)
            except Exception as exc:  # noqa: BLE001
                logger.warning("gpu release during stop raised: %s", exc)
            _RUN_CLAIM = None


__all__ = [
    "current_daemon_log_path",
    "is_alive",
    "pipeline_daemon_url",
    "start_pipeline_daemon",
    "stop_pipeline_daemon",
]
