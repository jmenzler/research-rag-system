"""Single-coordinator supervisor for the long-lived MinerU pipeline daemon.

Detects daemon death (glibc SIGABRT crashes — see gh #14), respawns on the
fly, and opens a circuit breaker if the daemon is unstable so a heisenbug
does not crashloop the run.

Lives in the **main ragctl process** alongside ``coordinator_loop``. Parse
workers stay fail-fast (per the post-1b2df18 invariant: only one process
may own respawn). Workers pick up the new daemon transparently because
``pipeline_daemon_url()`` re-reads the state file on every parse, and
``start_pipeline_daemon()`` rewrites it on respawn.
"""
from __future__ import annotations

import os
import subprocess
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from src.pdf_parsers import mineru_daemon
from src.pipeline.crasher_tracker import CrasherOutcome, CrasherTracker

# Tunable via env. Defaults match the gh #14 acceptance criteria.
_HEALTH_INTERVAL_S = float(os.environ.get("RAG_DAEMON_HEALTH_INTERVAL_S", "10.0"))
_CIRCUIT_DEATHS = int(os.environ.get("RAG_DAEMON_CIRCUIT_DEATHS", "3"))
_CIRCUIT_WINDOW_S = float(os.environ.get("RAG_DAEMON_CIRCUIT_WINDOW_S", "600.0"))

# Substrings that indicate a parse_fail reason was caused by daemon death,
# not by a bad PDF. Keep narrow — we don't want broad URLLib failures (DNS,
# proxy, etc.) to count as daemon deaths. The daemon is on 127.0.0.1, so
# "Connection refused" / "Connection reset" / "Remote disconnected" are
# unambiguous.
_DAEMON_DEAD_MARKERS = (
    "ConnectionRefusedError",
    "ConnectionResetError",
    "RemoteDisconnected",
    "[Errno 111]",                # urllib.error.URLError wrapping ConnRefused
    "[Errno 104]",                # ConnReset
    "URLError",                   # broader net — but combined with the above is fine
)


@dataclass(frozen=True)
class SupervisorEvent:
    """Returned by ``DaemonSupervisor.tick`` on a state transition.

    ``kind`` is one of ``"died"``, ``"respawned"``, ``"circuit_break"``.
    ``death_index`` is 1-based; only meaningful for "died" / "respawned".
    """
    kind: str
    death_index: int = 0


class DaemonSupervisor:
    """Owns daemon liveness for one ragctl run."""

    def __init__(
        self,
        *,
        check_interval_s: float = _HEALTH_INTERVAL_S,
        circuit_deaths: int = _CIRCUIT_DEATHS,
        circuit_window_s: float = _CIRCUIT_WINDOW_S,
        crasher_tracker: CrasherTracker | None = None,
    ) -> None:
        self._check_interval_s = check_interval_s
        self._circuit_deaths = circuit_deaths
        self._circuit_window_s = circuit_window_s
        self._death_times: deque[float] = deque()
        self._circuit_open = False
        # Initialised to -inf so the first tick() always probes regardless
        # of the time gate. monotonic() can be 0 or any positive float at
        # process start; -inf forces the first run.
        self._last_check_ts = float("-inf")
        self._crasher_tracker = crasher_tracker
        # Coordinator updates these as parse_start / parse_ok / parse_fail
        # events flow through. The supervisor reads them in death-handling
        # to attribute the SIGABRT to a specific PDF (gh #17).
        self._last_completed_pdf: str | None = None
        self._pending_pdfs: set[str] = set()
        # CrasherOutcome from the most recent confirmed correlation, or
        # None. Coordinator polls this after a death tick and triggers
        # quarantine when set.
        self.last_crasher_outcome: CrasherOutcome | None = None

    # -- public API ---------------------------------------------------------

    def is_circuit_open(self) -> bool:
        return self._circuit_open

    # Parse-event hooks for crasher correlation (gh #17). Coordinator
    # calls these as events flow; cheap, no I/O.

    def note_parse_start(self, pdf_slug: str) -> None:
        self._pending_pdfs.add(pdf_slug)

    def note_parse_done(self, pdf_slug: str, *, ok: bool) -> None:
        self._pending_pdfs.discard(pdf_slug)
        if ok:
            self._last_completed_pdf = pdf_slug

    def note_parse_fail_reason(self, reason: str) -> bool:
        """True iff *reason* looks like the daemon died (vs a bad PDF).

        Caller uses this to decide whether to call ``tick(force=True)``
        immediately on a parse_fail event, instead of waiting for the
        next 10s heartbeat.
        """
        return any(m in reason for m in _DAEMON_DEAD_MARKERS)

    def tick(self, *, force: bool = False) -> SupervisorEvent | None:
        """Check daemon liveness; respawn or open circuit breaker on death.

        Idempotent + cheap when the daemon is alive (one PID stat + one
        2-second HTTP GET). Returns None on green; otherwise a
        :class:`SupervisorEvent` describing the state change.

        ``force=True`` skips the 10-second time gate — used for reactive
        checks triggered by a parse_fail event.

        Side effects:
          * On detected death: prints the dying daemon's last 50 log lines
            to stdout (forensic context).
          * On respawn: calls ``mineru_daemon.start_pipeline_daemon()``,
            which rewrites the shared state file so workers transparently
            pick up the new daemon.
          * On circuit-break: sets ``self._circuit_open=True``; subsequent
            ticks become no-ops.
          * Always emits a structured ``daemon_event:`` line on state change
            so logs can be grep-filtered.
        """
        if self._circuit_open:
            return None

        now = time.monotonic()
        if not force and (now - self._last_check_ts) < self._check_interval_s:
            return None
        self._last_check_ts = now

        if mineru_daemon.is_alive():
            return None

        # Daemon is dead. Capture forensics, then decide: respawn or break.
        log_path = mineru_daemon.current_daemon_log_path()
        self._dump_log_tail(log_path)
        self._death_times.append(now)
        self._evict_old_deaths(now)
        death_index = len(self._death_times)

        print(f"  ✗ daemon_event: died (death #{death_index})", flush=True)

        # Crasher correlation (gh #17). With multiple parse workers,
        # several PDFs may be in flight at SIGABRT time. We pick the
        # lexicographically-lowest slug — deterministic across ticks
        # AND across PYTHONHASHSEED permutations, so the second death
        # converges on the same suspect as the first whenever both
        # were pending. ``next(iter(set))`` would rotate with set
        # composition changes (a different suspect after each respawn)
        # and could confirm an innocent PDF.
        if self._crasher_tracker is not None:
            pending = min(self._pending_pdfs) if self._pending_pdfs else None
            log_ref = str(log_path) if log_path is not None else "?"
            outcome = self._crasher_tracker.record_death(
                last_completed_pdf=self._last_completed_pdf,
                pending_pdf=pending,
                log_ref=log_ref,
            )
            self.last_crasher_outcome = outcome
            if outcome.confirmed:
                print(
                    f"  ✗ daemon_event: crasher_confirmed "
                    f"slug={outcome.suspect} deaths={outcome.deaths_observed}",
                    flush=True,
                )
            elif outcome.suspect is not None:
                print(
                    f"  ✗ daemon_event: crasher_suspect "
                    f"slug={outcome.suspect} attempts={outcome.deaths_observed}",
                    flush=True,
                )

        if death_index >= self._circuit_deaths:
            self._circuit_open = True
            print(
                f"  ✗ daemon_event: circuit_break "
                f"({death_index} deaths in ≤{int(self._circuit_window_s)}s)",
                flush=True,
            )
            return SupervisorEvent(kind="circuit_break", death_index=death_index)

        try:
            mineru_daemon.start_pipeline_daemon()
        except Exception as exc:
            # Respawn failed — treat as another death and check breaker.
            print(
                f"  ✗ daemon_event: respawn_failed "
                f"({type(exc).__name__}: {exc})",
                flush=True,
            )
            self._death_times.append(time.monotonic())
            self._evict_old_deaths(time.monotonic())
            if len(self._death_times) >= self._circuit_deaths:
                self._circuit_open = True
                print(
                    f"  ✗ daemon_event: circuit_break "
                    f"({len(self._death_times)} deaths in "
                    f"≤{int(self._circuit_window_s)}s)",
                    flush=True,
                )
                return SupervisorEvent(
                    kind="circuit_break", death_index=len(self._death_times)
                )
            return SupervisorEvent(kind="died", death_index=death_index)

        print(f"  ↻ daemon_event: respawned (death #{death_index})", flush=True)
        return SupervisorEvent(kind="respawned", death_index=death_index)

    # -- internals ----------------------------------------------------------

    def _evict_old_deaths(self, now: float) -> None:
        cutoff = now - self._circuit_window_s
        while self._death_times and self._death_times[0] < cutoff:
            self._death_times.popleft()

    def _dump_log_tail(self, log_path: Path | None) -> None:
        if log_path is None or not log_path.exists():
            print(
                "  ✗ daemon_event: died — no log path recorded "
                "(state file missing or pre-supervisor daemon)",
                flush=True,
            )
            return
        # Use system `tail` rather than reading the whole file — the daemon
        # log can grow to many MB. Capture stdout, fall back to a short
        # marker line on any subprocess failure.
        try:
            result = subprocess.run(
                ["tail", "-n", "50", str(log_path)],
                capture_output=True, text=True, timeout=5,
            )
            tail = result.stdout
        except (OSError, subprocess.TimeoutExpired) as exc:
            tail = f"<could not tail {log_path}: {type(exc).__name__}: {exc}>"
        print(
            f"  ✗ daemon_event: died — last 50 lines of {log_path}:\n{tail}",
            flush=True,
        )


__all__ = ["DaemonSupervisor", "SupervisorEvent"]
