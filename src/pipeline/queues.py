"""Queue item types and multi-process-safe per-host rate limiter.

FetchItem / ParseItem are frozen dataclasses carried by multiprocessing queues.
SharedPacer serializes HTTP requests to the same host across worker processes
using a Manager-backed shared dict, fixing the broken in-process-only ``pace()``.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from multiprocessing.managers import SyncManager
from pathlib import Path


@dataclass(frozen=True)
class FetchItem:
    """One source to download from a notebook."""
    nb_tag: str
    nb_id: str
    src: dict[str, object]  # NotebookLM source dict
    no_cffi: bool


@dataclass(frozen=True)
class ParseItem:
    """One PDF file to parse via the MinerU daemon."""
    nb_tag: str
    pdf_path: Path


# ResultQueue event tuples (no dataclass — plain tuples for speed):
#
#   ("fetch_ok", nb_tag, src_id, has_pdf: bool, pdf_path: str | None)
#   ("fetch_fail", nb_tag, src_id, reason: str)
#   ("parse_ok", nb_tag, pdf_path: str)
#   ("parse_fail", nb_tag, pdf_path: str, reason: str)
#   ("worker_crash", worker_kind: str, exc_str: str)


class SharedPacer:
    """Multi-process-safe per-host rate limiter, backed by a Manager.

    The lock guards only the dict update; ``time.sleep`` happens outside the
    critical section so different hosts aren't blocked by each other. The
    future timestamp is reserved atomically so subsequent callers queue behind it.

    Usage::

        mgr = multiprocessing.Manager()
        pacer = SharedPacer(mgr, default_delay_s=1.0)
        pacer.wait("arxiv.org")         # uses default 1.0s
        pacer.wait("arxiv.org", 2.0)    # explicit 2.0s delay
    """

    def __init__(self, manager: SyncManager, default_delay_s: float = 1.0) -> None:
        self._last_seen = manager.dict()
        self._lock = manager.Lock()
        self._default = default_delay_s

    def wait(self, host: str, delay_s: float | None = None) -> None:
        """Wait at least ``delay_s`` seconds since the last call to *host*.

        If *delay_s* is ``None``, the instance-level ``default_delay_s`` is used.
        The first call for a given host returns immediately (no prior timestamp).
        """
        delay = delay_s if delay_s is not None else self._default
        now = time.time()
        with self._lock:
            last = self._last_seen.get(host, 0.0)
            wait_for = max(0.0, last + delay - now)
            # Reserve the future timestamp so subsequent callers queue behind it
            self._last_seen[host] = max(now, last) + delay
        if wait_for > 0:
            time.sleep(wait_for)
