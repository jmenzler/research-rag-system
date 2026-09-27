"""Background daemon thread: poll paper sources every 24h, ingest relevant papers."""
from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from src.ingest.storage import _open_parents_db
from src.monitor import discord as _discord
from src.monitor.arxiv import fetch_papers as _arxiv_fetch
from src.monitor.db import create_run, get_last_run, update_run
from src.monitor.filter import classify_papers
from src.monitor.ingest_paper import ingest_paper
from src.monitor.paper import Paper

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_LOG_DIR = _PROJECT_ROOT / "logs" / "monitor"
_DAILY_HOUR_UTC = 7  # fire at 07:00 UTC every day
_STARTUP_COOLDOWN_H = 6  # skip startup poll if last run was within this many hours


def _next_scheduled_utc() -> datetime:
    now = datetime.now(UTC)
    target = now.replace(hour=_DAILY_HOUR_UTC, minute=0, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return target


class MonitorScheduler(threading.Thread):
    """Daemon thread that runs a poll cycle every 24 hours.

    Usage::

        scheduler = MonitorScheduler()
        scheduler.start()      # from __main__.py on server boot
        scheduler.trigger()    # from POST /monitor/trigger
        scheduler.status()     # from GET /monitor/status
        scheduler.stop()       # on shutdown
    """

    def __init__(
        self,
        *,
        run_on_startup: bool = True,
        sources: list[Callable[[], list[Paper]]] | None = None,
    ) -> None:
        super().__init__(name="MonitorScheduler", daemon=True)
        self._run_on_startup = run_on_startup
        self._sources: list[Callable[[], list[Paper]]] = sources or [_arxiv_fetch]
        self._trigger = threading.Event()
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._running: bool = False
        self._last_run: dict[str, Any] | None = None
        self._next_run_at: float | None = None

    # ------------------------------------------------------------------
    # Public control API (thread-safe)
    # ------------------------------------------------------------------

    def trigger(self) -> bool:
        """Wake the scheduler immediately. Returns False if already running."""
        with self._lock:
            if self._running:
                return False
        self._trigger.set()
        return True

    def stop(self) -> None:
        """Signal the thread to exit after the current poll (if any)."""
        self._stop_event.set()
        self._trigger.set()  # unblock the wait

    def status(self) -> dict[str, Any]:
        """Return current run state for GET /monitor/status."""
        with self._lock:
            running = self._running
            last_run = self._last_run
            next_run_at = self._next_run_at

        if last_run is None:
            # Attempt DB fallback after a server restart.
            try:
                conn = _open_parents_db()
                try:
                    last_run = get_last_run(conn)
                finally:
                    conn.close()
            except Exception:  # noqa: BLE001
                logger.warning("status: failed to read last_run from DB", exc_info=True)

        return {
            "running": running,
            "last_run": last_run,
            "next_run_at": None if running else next_run_at,
        }

    # ------------------------------------------------------------------
    # Thread entrypoint
    # ------------------------------------------------------------------

    def _recent_poll_exists(self) -> bool:
        """Return True if a poll completed within _STARTUP_COOLDOWN_H hours."""
        try:
            conn = _open_parents_db()
            try:
                last = get_last_run(conn)
            finally:
                conn.close()
        except Exception:  # noqa: BLE001
            return False
        if last is None or last.get("completed_at") is None:
            return False
        age_h = (datetime.now(UTC).timestamp() - float(last["completed_at"])) / 3600
        return age_h < _STARTUP_COOLDOWN_H

    def run(self) -> None:
        logger.info("MonitorScheduler started")
        skip_startup = self._run_on_startup and self._recent_poll_exists()
        if skip_startup:
            logger.info(
                "MonitorScheduler: last poll was recent (<%dh ago) — skipping startup poll",
                _STARTUP_COOLDOWN_H,
            )
        if not self._run_on_startup or skip_startup:
            self._schedule_next()
            self._wait_for_trigger()

        while not self._stop_event.is_set():
            self._run_poll()
            if self._stop_event.is_set():
                break
            self._schedule_next()
            self._wait_for_trigger()

        logger.info("MonitorScheduler stopped")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _schedule_next(self) -> None:
        next_run = _next_scheduled_utc()
        with self._lock:
            self._next_run_at = next_run.timestamp()

    def _wait_for_trigger(self) -> None:
        with self._lock:
            next_run_at = self._next_run_at
        delay = max(0.0, next_run_at - datetime.now(UTC).timestamp()) if next_run_at else 86400.0
        self._trigger.wait(timeout=delay)
        self._trigger.clear()

    def _run_poll(self) -> None:
        started_at = datetime.now(UTC)
        run_id: int | None = None

        with self._lock:
            self._running = True
            self._next_run_at = None

        papers_found = papers_ingested = papers_skipped = 0

        conn = _open_parents_db()
        try:
            run_id = create_run(conn, started_at)
        except Exception:
            logger.warning("Failed to create poll_runs row", exc_info=True)
        finally:
            conn.close()

        try:
            all_papers: list[Paper] = []
            for source_fn in self._sources:
                try:
                    all_papers.extend(source_fn())
                except Exception:  # noqa: BLE001
                    logger.error("source %s failed", source_fn.__name__, exc_info=True)
            papers_found = len(all_papers)

            results = classify_papers(all_papers)
            relevant = [r for r in results if r.relevant]

            for r in relevant:
                ok = ingest_paper(r.paper)
                if ok:
                    papers_ingested += 1
                else:
                    papers_skipped += 1

            papers_skipped += papers_found - len(relevant)

            status = "done"
            error_detail: str | None = None

        except Exception as exc:  # noqa: BLE001
            logger.error("poll cycle failed", exc_info=True)
            status = "failed"
            error_detail = str(exc)
            relevant = []

        completed_at = datetime.now(UTC)
        duration_s = (completed_at - started_at).total_seconds()

        last_run: dict[str, Any] = {
            "run_id": run_id,
            "started_at": started_at.timestamp(),
            "completed_at": completed_at.timestamp(),
            "status": status,
            "papers_found": papers_found,
            "papers_ingested": papers_ingested,
            "papers_skipped": papers_skipped,
            "error_detail": error_detail,
        }

        if run_id is not None:
            conn = _open_parents_db()
            try:
                update_run(
                    conn,
                    run_id,
                    completed_at=completed_at,
                    status=status,
                    papers_found=papers_found,
                    papers_ingested=papers_ingested,
                    papers_skipped=papers_skipped,
                    error_detail=error_detail,
                )
            except Exception:
                logger.warning("Failed to update poll_runs row", exc_info=True)
            finally:
                conn.close()

        self._write_log(started_at, last_run, duration_s, run_id)

        with self._lock:
            self._running = False
            self._last_run = last_run

        logger.info(
            "poll done: found=%d ingested=%d skipped=%d status=%s duration=%.1fs",
            papers_found,
            papers_ingested,
            papers_skipped,
            status,
            duration_s,
        )

        try:
            _discord.post_digest(relevant, started_at)
        except Exception:  # noqa: BLE001
            logger.error("Discord post_digest raised unexpectedly", exc_info=True)

        if status == "done" and papers_ingested > 0:
            self._enqueue_full_pdf_upgrade()

    @staticmethod
    def _enqueue_full_pdf_upgrade() -> None:
        """Queue a full-PDF upgrade for papers ingested abstract-only this poll.
        Inline parsing here is unsafe (the GPU daemon is single-owner; a
        poll-scoped start/stop races ragctl and can kill the server), so the
        heavy parse runs in the serialized DiscoverQueue worker. Best-effort —
        not-ready or full just defers to the next poll."""
        from src.server.api import get_discover_queue  # noqa: PLC0415

        queue = get_discover_queue()
        if queue is None:
            logger.info("upgrade enqueue skipped — discover queue not ready")
            return
        try:
            entry = queue.enqueue_upgrade(collection="trading")
            if entry is not None:
                logger.info("enqueued full-PDF upgrade job run_id=%s", entry.run_id)
        except Exception:  # noqa: BLE001
            logger.warning("failed to enqueue full-PDF upgrade", exc_info=True)

    @staticmethod
    def _write_log(
        started_at: datetime, last_run: dict[str, Any], duration_s: float, run_id: int | None
    ) -> None:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_path = _LOG_DIR / f"{started_at.strftime('%Y-%m-%d')}.jsonl"
        entry = {
            "ts": started_at.isoformat(),
            "run_id": run_id,
            "status": last_run["status"],
            "papers_found": last_run["papers_found"],
            "papers_ingested": last_run["papers_ingested"],
            "papers_skipped": last_run["papers_skipped"],
            "duration_s": round(duration_s, 1),
            "error": last_run["error_detail"],
        }
        try:
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")
        except OSError:
            logger.warning("Failed to write monitor log to %s", log_path, exc_info=True)
