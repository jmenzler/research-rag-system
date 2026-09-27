"""Local server entry point; host and port are configurable through the environment."""
from __future__ import annotations

import logging
import os

import uvicorn

from src.monitor.scheduler import MonitorScheduler
from src.server.api import set_discover_queue, set_monitor_scheduler
from src.server.api_ingest import set_ingest_queue
from src.server.discover_queue import DiscoverQueue

logger = logging.getLogger(__name__)


def main() -> None:
    host = os.getenv("RAG_SERVER_HOST", "127.0.0.1")
    port = int(os.getenv("RAG_SERVER_PORT", "8766"))

    # Queue first: the scheduler's startup poll enqueues full-PDF upgrade jobs
    # onto it, so it must be wired before the scheduler thread can run a poll.
    queue = DiscoverQueue()
    queue.reconcile_on_startup()
    set_discover_queue(queue)
    set_ingest_queue(queue)
    queue.start()
    logger.info("DiscoverQueue started on boot")

    scheduler = MonitorScheduler(run_on_startup=True)
    set_monitor_scheduler(scheduler)
    scheduler.start()
    logger.info("MonitorScheduler started on boot")

    uvicorn.run("src.server.app:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
