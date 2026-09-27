# long-ok-file
"""Persistent FIFO queue + single worker for /discover/start runs.

Serialises `research_and_ingest` invocations so concurrent MCP calls cannot
race on the ragctl-run.lock. Disk-backed under
``~/.cache/rag-system/discover-queue/`` so it survives uvicorn restart and
the PC's GitHub Actions autodeploy. State transitions are atomic via
``os.rename`` between sibling directories.

Layout::

    <state_dir>/
        queued/       <NNNNN>_<run_id>.json     FIFO by counter prefix
        running/      <NNNNN>_<run_id>.json     worker is executing this
        done/         <NNNNN>_<run_id>.json     terminal (ok|cancelled|crashed)
        quarantine/   corrupt entries
"""
from __future__ import annotations

import json
import logging
import os
import shlex
import signal
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, field_validator

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = Path.home() / ".cache" / "rag-system" / "discover-queue"
DEFAULT_LOG_DIR = _PROJECT_ROOT / "logs" / "discover"
DEFAULT_MAX_QUEUE_DEPTH = int(os.getenv("MAX_QUEUE_DEPTH", "50"))

# Result-file ``status`` values that mean the run terminated successfully.
# Mirrors scripts.discover_via_deep_research's top_status set + the urls path's
# "ok". Anything else terminal (e.g. "error") is a crash, not a clean done.
_SUCCESS_RESULT_STATUSES = frozenset({"ok", "no_new_sources"})


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


def _pid_start_token(pid: int) -> str | None:
    """Return a stable token identifying *this* incarnation of ``pid``.

    Guards against PID recycling: a dead child's PID can be reassigned to an
    unrelated process, making a bare ``os.kill(pid, 0)`` liveness check report
    a zombie as alive. The token (process start time) changes when the PID is
    reused, so a stale entry can be detected as dead.

    Linux: field 22 (starttime, clock ticks since boot) of /proc/<pid>/stat.
    Darwin/BSD: ``ps -o lstart=`` (process start wall-clock).
    Returns ``None`` when neither source is available — callers then degrade
    to the legacy os.kill check (no worse than before).
    """
    proc_stat = Path(f"/proc/{pid}/stat")
    if proc_stat.exists():
        try:
            raw = proc_stat.read_text(encoding="utf-8")
        except OSError:
            return None
        # comm (field 2) may contain spaces/parens; split on the last ')'.
        rparen = raw.rfind(")")
        if rparen != -1:
            fields = raw[rparen + 2 :].split()
            # After comm, field 3 is index 0 here; starttime is field 22 → idx 19.
            if len(fields) > 19:
                return fields[19]
        return None
    try:
        out = subprocess.run(
            ["ps", "-o", "lstart=", "-p", str(pid)],
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, ValueError):
        return None
    token = out.stdout.strip()
    return token or None


class QueueDepthExceededError(RuntimeError):
    """Raised by ``enqueue`` when ``queued/`` is already at capacity."""


class UnknownRunIDError(LookupError):
    """Raised by ``get_state`` / ``cancel`` when the run_id is not on disk."""


class _RaceMissError(Exception):
    """Internal: a file vanished mid-lookup; retry get_state."""


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def _max_queries_per_request() -> int:
    return int(os.getenv("MAX_QUERIES_PER_REQUEST", "8"))


class DiscoverStartRequest(BaseModel):
    """Request body for /discover/start; same shape as the MCP tool args.

    ``query`` accepts either a single string (existing behavior) or a list
    of strings (bulk mode). Internal code reads ``.queries`` which always
    returns a list.
    """

    query: str | list[str]
    collection: Literal["trading", "ecology", "notes", "system", "poker", "security"]
    partition: str = "research_briefs"
    mode: Literal["fast", "deep"] = "deep"
    timeout: int = 1800
    keep: bool = False

    @field_validator("query")
    @classmethod
    def _validate_query(cls, v: str | list[str]) -> str | list[str]:
        if isinstance(v, str):
            if not v.strip():
                raise ValueError("query must be a non-empty string")
            return v
        if not v:
            raise ValueError("query list must be non-empty")
        max_q = _max_queries_per_request()
        if len(v) > max_q:
            raise ValueError(
                f"query list length {len(v)} exceeds MAX_QUERIES_PER_REQUEST={max_q}"
            )
        for q in v:
            if not isinstance(q, str) or not q.strip():
                raise ValueError("every query must be a non-empty string")
        return v

    @property
    def queries(self) -> list[str]:
        if isinstance(self.query, str):
            return [self.query]
        return list(self.query)


class QueueEntry(BaseModel):
    """On-disk record for one queued/running/done discover run."""

    run_id: str
    counter: int
    kind: Literal["discover", "upgrade", "urls"] = "discover"
    queries: list[str]
    collection: str
    partition: str
    mode: str
    timeout: int
    keep: bool
    enqueued_at: float
    pid: int | None = None
    pid_start: str | None = None
    log_path: str
    result_path: str


# ---------------------------------------------------------------------------
# Spawner — injectable so tests can stub the subprocess
# ---------------------------------------------------------------------------


class SubprocessSpawner(Protocol):
    """The queue's only dependency on the OS process layer."""

    def spawn(
        self, entry: QueueEntry, log_path: Path, result_path: Path
    ) -> int: ...

    def is_alive(self, pid: int, pid_start: str | None = None) -> bool: ...

    def terminate(self, pid: int) -> bool: ...

    def force_kill(self, pid: int) -> bool: ...


class DefaultSpawner:
    """Real spawner: detached subprocess to scripts.discover_via_deep_research.

    Tracks each ``Popen`` instance so ``is_alive`` can reap zombies via
    ``proc.poll()``. Without that, a SIGKILL'd child stays in the kernel's
    process table forever (parent never calls waitpid), ``os.kill(pid, 0)``
    keeps returning alive, and the queue's inner loop spins indefinitely
    instead of moving the entry from ``running/`` to ``done/``. (#35-1a)
    """

    def __init__(self) -> None:
        self._procs: dict[int, subprocess.Popen[bytes]] = {}
        self._procs_lock = threading.Lock()

    def _build_cmd(self, entry: QueueEntry, result_path: Path) -> list[str]:
        if entry.kind == "upgrade":
            return [
                "uv", "run", "python", "-m", "scripts.upgrade_abstract_only",
                "--collection", entry.collection,
                "--result-json", str(result_path),
            ]
        if entry.kind == "urls":
            urls_file = result_path.parent / f"{entry.run_id}.urls.txt"
            inner = " ".join(
                shlex.quote(c)
                for c in [
                    "uv", "run", "python", "./bin/ragctl", "ingest-urls",
                    str(urls_file), "--collection", entry.collection,
                    "--notebook", entry.partition,
                ]
            )
            rp = shlex.quote(str(result_path))
            # ragctl ingest-urls has no --result-json; write the terminal marker
            # the queue done-detection needs, keyed off the exit code.
            ok = "'{\"status\": \"ok\"}'"
            bad = "'{\"status\": \"crashed\"}'"
            script = (
                f"{inner}; rc=$?; "
                f"if [ \"$rc\" = 0 ]; then printf {ok} > {rp}; "
                f"else printf {bad} > {rp}; fi; exit \"$rc\""
            )
            return ["bash", "-c", script]
        cmd: list[str] = [
            "uv", "run", "python", "-m", "scripts.discover_via_deep_research",
            "--collection", entry.collection,
            "--partition", entry.partition,
            "--mode", entry.mode,
            "--timeout", str(entry.timeout),
            "--result-json", str(result_path),
        ]
        if entry.keep:
            cmd.append("--keep")
        cmd.append("--query")
        cmd.extend(entry.queries)
        return cmd

    def spawn(self, entry: QueueEntry, log_path: Path, result_path: Path) -> int:
        cmd = self._build_cmd(entry, result_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fh = log_path.open("w", encoding="utf-8")
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=_PROJECT_ROOT,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        finally:
            log_fh.close()
        with self._procs_lock:
            self._procs[proc.pid] = proc
        return proc.pid

    def is_alive(self, pid: int, pid_start: str | None = None) -> bool:
        with self._procs_lock:
            proc = self._procs.get(pid)
        if proc is not None:
            if proc.poll() is not None:
                with self._procs_lock:
                    self._procs.pop(pid, None)
                return False
            return True
        # Fallback: untracked pid (eg. reconcile_on_startup looking at a pid
        # from a prior server run). os.kill confirms a process exists, but the
        # PID may have been recycled — compare the persisted start token to the
        # live one so a recycled PID does not keep a zombie entry alive forever.
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        if pid_start is not None:
            live_start = _pid_start_token(pid)
            if live_start is not None and live_start != pid_start:
                return False
        return True

    def terminate(self, pid: int) -> bool:
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except ProcessLookupError:
            return False
        return True

    def force_kill(self, pid: int) -> bool:
        """Send SIGKILL to the process group. Last-resort for SIGTERM-deaf children
        (ragctl's multiprocessing parent blocking on a dead-children Queue.get is
        the motivating case).
        """
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except ProcessLookupError:
            return False
        return True


# ---------------------------------------------------------------------------
# Queue
# ---------------------------------------------------------------------------


class DiscoverQueue(threading.Thread):
    """Persistent FIFO + single worker thread for discover runs."""

    def __init__(
        self,
        *,
        state_dir: Path | None = None,
        log_dir: Path | None = None,
        spawner: SubprocessSpawner | None = None,
        max_queue_depth: int | None = None,
        poll_interval_s: float = 0.5,
        cancel_kill_timeout_s: float | None = None,
    ) -> None:
        super().__init__(name="DiscoverQueue", daemon=True)
        self._state_dir = state_dir if state_dir is not None else DEFAULT_STATE_DIR
        self._log_dir = log_dir if log_dir is not None else DEFAULT_LOG_DIR
        self._spawner: SubprocessSpawner = (
            spawner if spawner is not None else DefaultSpawner()
        )
        self._max_queue_depth = (
            max_queue_depth if max_queue_depth is not None else DEFAULT_MAX_QUEUE_DEPTH
        )
        self._poll_interval_s = poll_interval_s
        # Seconds to wait after SIGTERM before escalating to SIGKILL on a
        # cancel that's stuck (e.g. ragctl mp parent ignoring SIGTERM).
        if cancel_kill_timeout_s is None:
            cancel_kill_timeout_s = float(os.getenv("DISCOVER_CANCEL_KILL_TIMEOUT_S", "10"))
        self._cancel_kill_timeout_s = cancel_kill_timeout_s

        self._queued_dir = self._state_dir / "queued"
        self._running_dir = self._state_dir / "running"
        self._done_dir = self._state_dir / "done"
        self._quarantine_dir = self._state_dir / "quarantine"
        for d in (
            self._queued_dir,
            self._running_dir,
            self._done_dir,
            self._quarantine_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)
        self._log_dir.mkdir(parents=True, exist_ok=True)

        self._enqueue_lock = threading.Lock()
        self._wake = threading.Event()
        self._stop_event = threading.Event()
        self._counter = self._compute_initial_counter()
        # In-memory cancel markers: run_id -> when cancel was requested.
        # Worker's poll loop reads these to decide when to escalate SIGTERM
        # to SIGKILL. In-memory is fine because cancel state is ephemeral —
        # a server restart kills any in-flight cancellation alongside the
        # subprocess group (re-issue via POST /discover/cancel if needed).
        self._cancel_marker_lock = threading.Lock()
        self._cancel_markers: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def state_dir(self) -> Path:
        """Root of the on-disk queue layout (queued/ running/ done/)."""
        return self._state_dir

    def enqueue(self, req: DiscoverStartRequest) -> QueueEntry:
        with self._enqueue_lock:
            depth = self._depth()
            if depth >= self._max_queue_depth:
                raise QueueDepthExceededError(
                    f"queue at capacity ({depth}/{self._max_queue_depth})"
                )
            run_id = uuid.uuid4().hex[:12]
            counter = self._counter
            self._counter += 1
            log_path = self._log_dir / f"{run_id}.log"
            result_path = self._log_dir / f"{run_id}.result.json"
            entry = QueueEntry(
                run_id=run_id,
                counter=counter,
                queries=req.queries,
                collection=req.collection,
                partition=req.partition,
                mode=req.mode,
                timeout=req.timeout,
                keep=req.keep,
                enqueued_at=time.time(),
                log_path=str(log_path),
                result_path=str(result_path),
            )
            self._write_entry_atomically(self._queued_dir, entry)
        self._wake.set()
        return entry

    def enqueue_urls(
        self, *, urls: list[str], collection: str, partition: str
    ) -> QueueEntry:
        """Enqueue a direct fetch+parse+ingest of an explicit URL list.

        Bypasses NotebookLM Deep Research: the worker runs ``ragctl ingest-urls``
        against a per-run urls file written here. URL lines are verbatim — the
        ``Title || url`` form passes through unchanged.
        """
        with self._enqueue_lock:
            depth = self._depth()
            if depth >= self._max_queue_depth:
                raise QueueDepthExceededError(
                    f"queue at capacity ({depth}/{self._max_queue_depth})"
                )
            run_id = uuid.uuid4().hex[:12]
            counter = self._counter
            self._counter += 1
            log_path = self._log_dir / f"{run_id}.log"
            result_path = self._log_dir / f"{run_id}.result.json"
            urls_path = self._log_dir / f"{run_id}.urls.txt"
            urls_path.write_text("\n".join(urls) + "\n", encoding="utf-8")
            entry = QueueEntry(
                run_id=run_id,
                counter=counter,
                kind="urls",
                queries=urls,
                collection=collection,
                partition=partition,
                mode="direct",
                timeout=1800,
                keep=False,
                enqueued_at=time.time(),
                log_path=str(log_path),
                result_path=str(result_path),
            )
            self._write_entry_atomically(self._queued_dir, entry)
        self._wake.set()
        return entry

    def enqueue_upgrade(self, *, collection: str = "trading") -> QueueEntry | None:
        """Enqueue an abstract-only → full-PDF upgrade job. Deduped.
        The worker scans the whole sources tree, so one pending job drains the
        entire backlog — a second would be redundant. Returns the new entry, or
        None when an upgrade job is already queued/running.
        """
        with self._enqueue_lock:
            if self._upgrade_pending():
                return None
            depth = self._depth()
            if depth >= self._max_queue_depth:
                raise QueueDepthExceededError(
                    f"queue at capacity ({depth}/{self._max_queue_depth})"
                )
            run_id = uuid.uuid4().hex[:12]
            counter = self._counter
            self._counter += 1
            entry = QueueEntry(
                run_id=run_id,
                counter=counter,
                kind="upgrade",
                queries=["upgrade-abstract-only"],
                collection=collection,
                partition="arxiv_monitor",
                mode="fast",
                timeout=1800,
                keep=False,
                enqueued_at=time.time(),
                log_path=str(self._log_dir / f"{run_id}.log"),
                result_path=str(self._log_dir / f"{run_id}.result.json"),
            )
            self._write_entry_atomically(self._queued_dir, entry)
        self._wake.set()
        return entry

    def _upgrade_pending(self) -> bool:
        """True if an upgrade job is already queued or running."""
        for d in (self._queued_dir, self._running_dir):
            for f in d.iterdir():
                if f.suffix != ".json":
                    continue
                try:
                    if self._read_entry(f).kind == "upgrade":
                        return True
                except Exception:  # noqa: BLE001
                    continue
        return False

    def get_state(self, run_id: str) -> dict[str, Any]:
        # Files move queued → running → done; a get_state call can race the
        # worker's os.rename and find the entry has disappeared mid-read.
        # Retry briefly so callers see a consistent state instead of a
        # transient FileNotFound / position miss. Total budget ~500ms covers
        # multi-step worker transitions (rename queued→running, write entry
        # with pid, rename running→done) under stress.
        for _ in range(50):
            try:
                return self._get_state_once(run_id)
            except (FileNotFoundError, _RaceMissError):
                time.sleep(0.01)
        raise UnknownRunIDError(f"unknown run_id: {run_id}")

    def _get_state_once(self, run_id: str) -> dict[str, Any]:
        # Snapshot all three dirs in close succession so a worker rename that
        # happens mid-lookup is bounded — we either see the file in its old
        # location or its new one, not "vanished".
        #
        # Snapshots are taken in lifecycle order (queued → running → done) so
        # that any entry present at the moment we start looking is captured
        # in at least one snapshot, even if the worker zips it through all
        # three states (queued → running → done) between snapshot calls.
        # Iteration below stays in the reverse order so we prefer the most
        # recent state seen across the three snapshots.
        queued_snap = sorted(self._queued_dir.iterdir())
        running_snap = list(self._running_dir.iterdir())
        done_snap = list(self._done_dir.iterdir())
        suffix = f"_{run_id}.json"

        for f in done_snap:
            if f.name.endswith(suffix):
                entry = self._read_entry(f)
                result = self._read_result(entry)
                state: str = "done"
                if isinstance(result, dict):
                    status = result.get("status")
                    if status == "cancelled":
                        state = "cancelled"
                    elif status not in _SUCCESS_RESULT_STATUSES:
                        # Any non-success terminal status (e.g. the discover
                        # script's "error" when all queries fail) is a failure,
                        # not a clean done — surface it as crashed so the SSE
                        # consumer / caller can retry instead of seeing n=0 ok.
                        state = "crashed"
                return {
                    "state": state,
                    "run_id": run_id,
                    "result": result,
                    "log_tail": self._read_log_tail(Path(entry.log_path)),
                }

        for f in running_snap:
            if f.name.endswith(suffix):
                entry = self._read_entry(f)
                return {
                    "state": "running",
                    "run_id": run_id,
                    "pid": entry.pid,
                    "started_at": entry.enqueued_at,
                    "elapsed_s": round(time.time() - entry.enqueued_at, 1),
                    "log_tail": self._read_log_tail(Path(entry.log_path)),
                }

        for i, f in enumerate(queued_snap):
            if f.name.endswith(suffix):
                return {
                    "state": "queued",
                    "run_id": run_id,
                    "position": i,
                    "queue_depth": len(queued_snap),
                }

        raise UnknownRunIDError(f"unknown run_id: {run_id}")

    def cancel(self, run_id: str) -> dict[str, Any]:
        # Already-done: no-op, surface the final state. Avoids racing the
        # worker's running→done rename — by the time a cancel call lands,
        # the run may have crashed/completed on its own.
        f_done = self._find_entry_file(self._done_dir, run_id)
        if f_done is not None:
            return {
                "state": "already_terminal",
                "killed": False,
                "run_id": run_id,
            }

        f = self._find_entry_file(self._queued_dir, run_id)
        if f is not None:
            entry = self._read_entry(f)
            self._write_cancelled_result(entry)
            os.replace(f, self._done_dir / f.name)
            return {"state": "cancelled", "killed": False, "run_id": run_id}

        # Running path. Race: the worker may have just moved the file from
        # queued/ to running/ but not yet written the spawn pid back. Retry
        # briefly so cancel still lands.
        for _ in range(40):
            # Re-check done/ each iteration in case worker raced us there.
            if self._find_entry_file(self._done_dir, run_id) is not None:
                return {
                    "state": "already_terminal",
                    "killed": False,
                    "run_id": run_id,
                }
            f = self._find_entry_file(self._running_dir, run_id)
            if f is None:
                # Re-check queued/ in case the entry hasn't been picked up yet.
                f_q = self._find_entry_file(self._queued_dir, run_id)
                if f_q is not None:
                    entry = self._read_entry(f_q)
                    self._write_cancelled_result(entry)
                    os.replace(f_q, self._done_dir / f_q.name)
                    return {"state": "cancelled", "killed": False, "run_id": run_id}
                # Not in queued either — small wait so a worker rename in
                # flight can settle, then loop continues.
                time.sleep(0.01)
                continue
            try:
                entry = self._read_entry(f)
            except FileNotFoundError:
                continue
            if entry.pid is None:
                time.sleep(0.01)
                continue
            killed = False
            if self._spawner.is_alive(entry.pid, entry.pid_start):
                # Mark cancel BEFORE SIGTERM so the worker's next poll
                # iteration sees the marker and starts the SIGKILL countdown
                # even if SIGTERM is ignored.
                with self._cancel_marker_lock:
                    self._cancel_markers[run_id] = time.time()
                killed = self._spawner.terminate(entry.pid)
            return {"state": "cancelling", "killed": killed, "run_id": run_id}

        raise UnknownRunIDError(f"unknown run_id: {run_id}")

    def stop(self) -> None:
        self._stop_event.set()
        self._wake.set()

    # ------------------------------------------------------------------
    # Worker thread
    # ------------------------------------------------------------------

    def run(self) -> None:
        logger.info("DiscoverQueue worker started")
        while not self._stop_event.is_set():
            try:
                self._tick()
            except Exception:
                logger.exception("DiscoverQueue tick failed")
                time.sleep(self._poll_interval_s)
        logger.info("DiscoverQueue worker stopped")

    def _tick(self) -> None:
        queued = sorted(
            f for f in self._queued_dir.iterdir() if f.suffix == ".json"
        )
        if not queued:
            self._wake.wait(timeout=self._poll_interval_s)
            self._wake.clear()
            return

        src = queued[0]
        dst = self._running_dir / src.name
        try:
            os.rename(src, dst)
        except FileNotFoundError:
            return

        try:
            entry = self._read_entry(dst)
        except Exception:
            logger.exception("corrupt queued entry %s; quarantining", dst.name)
            os.replace(dst, self._quarantine_dir / dst.name)
            return

        pid = self._spawner.spawn(
            entry, Path(entry.log_path), Path(entry.result_path)
        )
        entry.pid = pid
        entry.pid_start = _pid_start_token(pid)
        self._write_entry_atomically(self._running_dir, entry)

        result_path = Path(entry.result_path)
        force_killed = False
        while not self._stop_event.is_set():
            if result_path.is_file():
                break
            if not self._spawner.is_alive(pid, entry.pid_start):
                if not result_path.is_file():
                    note = (
                        "process exited after SIGKILL escalation"
                        if force_killed
                        else "process exited without writing result.json"
                    )
                    result_path.parent.mkdir(parents=True, exist_ok=True)
                    result_path.write_text(
                        json.dumps(
                            {
                                "status": "crashed",
                                "queries": entry.queries,
                                "per_query": [],
                                "note": note,
                            }
                        ),
                        encoding="utf-8",
                    )
                break

            # Escalate stuck cancels to SIGKILL. cancel() sets the in-memory
            # marker before sending SIGTERM; the worker's countdown begins
            # the moment the next poll sees the marker.
            if not force_killed:
                with self._cancel_marker_lock:
                    requested_at = self._cancel_markers.get(entry.run_id)
                if (
                    requested_at is not None
                    and time.time() - requested_at >= self._cancel_kill_timeout_s
                ):
                    logger.warning(
                        "cancel escalation: SIGKILL pid=%s run_id=%s "
                        "(SIGTERM ignored for %.1fs)",
                        pid, entry.run_id, time.time() - requested_at,
                    )
                    self._spawner.force_kill(pid)
                    force_killed = True

            time.sleep(self._poll_interval_s)

        target = self._done_dir / dst.name
        try:
            os.rename(dst, target)
        except FileNotFoundError:
            pass

        # Clean cancel marker for completed run (whether by SIGKILL, natural
        # exit, or never-cancelled).
        with self._cancel_marker_lock:
            self._cancel_markers.pop(entry.run_id, None)

    # ------------------------------------------------------------------
    # Startup reconcile
    # ------------------------------------------------------------------

    def reconcile_on_startup(self) -> None:
        for f in list(self._running_dir.iterdir()):
            try:
                entry = self._read_entry(f)
            except Exception:
                logger.warning("reconcile: corrupt %s; quarantining", f.name)
                os.replace(f, self._quarantine_dir / f.name)
                continue
            alive = entry.pid is not None and self._spawner.is_alive(
                entry.pid, entry.pid_start
            )
            if alive:
                continue
            result_path = Path(entry.result_path)
            if not result_path.is_file():
                result_path.parent.mkdir(parents=True, exist_ok=True)
                result_path.write_text(
                    json.dumps(
                        {
                            "status": "crashed",
                            "queries": entry.queries,
                            "per_query": [],
                            "note": "process not running at startup reconcile",
                        }
                    ),
                    encoding="utf-8",
                )
            os.replace(f, self._done_dir / f.name)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _compute_initial_counter(self) -> int:
        existing: list[int] = []
        for d in (self._queued_dir, self._running_dir, self._done_dir):
            for f in d.iterdir():
                try:
                    existing.append(int(f.name.split("_", 1)[0]))
                except (ValueError, IndexError):
                    continue
        return (max(existing) + 1) if existing else 1

    def _depth(self) -> int:
        return len(list(self._queued_dir.iterdir())) + len(
            list(self._running_dir.iterdir())
        )

    def _write_entry_atomically(self, dir_: Path, entry: QueueEntry) -> None:
        # Write the tmp file under a sibling .tmp/ subdir, not inside `dir_`,
        # so the worker's `iterdir(queued_dir)` never picks up a half-written
        # entry as if it were a ready job. os.replace across sibling dirs on
        # the same filesystem is still atomic.
        final = dir_ / f"{entry.counter:05d}_{entry.run_id}.json"
        tmp_dir = dir_.parent / ".tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp = tmp_dir / f"{entry.counter:05d}_{entry.run_id}.{os.getpid()}.tmp"
        tmp.write_text(entry.model_dump_json(), encoding="utf-8")
        os.replace(tmp, final)

    def _find_entry_file(self, dir_: Path, run_id: str) -> Path | None:
        suffix = f"_{run_id}.json"
        for f in dir_.iterdir():
            if f.name.endswith(suffix):
                return f
        return None

    def _read_entry(self, path: Path) -> QueueEntry:
        return QueueEntry.model_validate_json(path.read_text(encoding="utf-8"))

    def _read_result(self, entry: QueueEntry) -> dict[str, Any] | None:
        p = Path(entry.result_path)
        if not p.is_file():
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        return data if isinstance(data, dict) else None

    def _read_log_tail(self, log_path: Path, n: int = 30) -> list[str]:
        if not log_path.is_file():
            return []
        try:
            with log_path.open("r", encoding="utf-8", errors="replace") as fh:
                lines = fh.readlines()
        except OSError:
            return []
        return [line.rstrip("\n") for line in lines[-n:]]

    def _write_cancelled_result(self, entry: QueueEntry) -> None:
        p = Path(entry.result_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(
                {
                    "status": "cancelled",
                    "queries": entry.queries,
                    "per_query": [],
                }
            ),
            encoding="utf-8",
        )
