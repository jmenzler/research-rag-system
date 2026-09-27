"""Crasher correlation across daemon SIGABRTs (gh #17).

Each death is paired with two anchors:

* ``last_completed_pdf`` — the slug of the last 200-OK
  ``POST /file_parse`` the dying daemon served.
* ``pending_pdf`` — the slug of the next-in-flight POST a parse worker
  was waiting on at SIGABRT time. May be ``None`` if the supervisor
  could not determine it (race window, log line not flushed yet).

A pending PDF is **confirmed** as a deterministic crasher only after
TWO observed deaths for which it was the pending suspect. One death
may be transient (paddleocr heap state contaminated by an unrelated
predecessor); two correlated deaths is policy enough to quarantine.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field


@dataclass(frozen=True)
class CrasherOutcome:
    """Returned by ``CrasherTracker.record_death``.

    * ``confirmed`` — caller should quarantine ``suspect``.
    * ``deaths_observed`` — total deaths attributed to ``suspect`` so far.
    * ``daemon_log_refs`` — every log path implicated, for the sidecar.
    """
    confirmed: bool
    suspect: str | None
    deaths_observed: int
    daemon_log_refs: list[str] = field(default_factory=list)


_CONFIRM_THRESHOLD = 2


class CrasherTracker:
    """In-memory tracker — survives one ragctl run, not across restarts.

    Cross-run persistence (e.g. SQLite) is not needed yet: a single bad
    PDF gets quarantined within the run that first hits it. Subsequent
    runs skip it because the source-walker filters
    ``sources/_quarantine/`` already.
    """

    def __init__(self, *, confirm_threshold: int = _CONFIRM_THRESHOLD) -> None:
        self._counts: dict[str, int] = defaultdict(int)
        self._logs: dict[str, list[str]] = defaultdict(list)
        self._confirm_threshold = confirm_threshold

    def attempts_for(self, suspect: str) -> int:
        return self._counts.get(suspect, 0)

    def record_death(
        self,
        *,
        last_completed_pdf: str | None,
        pending_pdf: str | None,
        log_ref: str,
    ) -> CrasherOutcome:
        """Record a daemon SIGABRT. ``last_completed_pdf`` is unused for
        confirmation today (we trust ``pending_pdf``), but the parameter
        stays in the signature so future correlation logic (e.g. blame
        the predecessor on the second death of a different pending) has
        the data without an API change."""
        del last_completed_pdf  # unused — reserved for future correlation logic.

        if pending_pdf is None:
            # Can't attribute — return informational outcome without
            # bumping any counter. Caller logs and continues; daemon
            # supervisor falls back to its existing respawn loop.
            return CrasherOutcome(
                confirmed=False, suspect=None, deaths_observed=0,
                daemon_log_refs=[log_ref],
            )

        self._counts[pending_pdf] += 1
        self._logs[pending_pdf].append(log_ref)
        n = self._counts[pending_pdf]
        return CrasherOutcome(
            confirmed=n >= self._confirm_threshold,
            suspect=pending_pdf,
            deaths_observed=n,
            daemon_log_refs=list(self._logs[pending_pdf]),
        )


__all__ = ["CrasherOutcome", "CrasherTracker"]
