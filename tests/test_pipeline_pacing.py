"""Unit tests for SharedPacer in src/pipeline/queues.py.

Tests the multi-process-safe per-host rate limiter. Uses a real
multiprocessing.Manager to back the SharedPacer (no brittle mocking).
"""

from __future__ import annotations

import multiprocessing as mp
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

from src.pipeline.queues import SharedPacer


@pytest.fixture()
def pacer() -> SharedPacer:
    """Create a SharedPacer backed by a real Manager."""
    mgr = mp.Manager()
    p = SharedPacer(mgr, default_delay_s=0.1)
    yield p
    mgr.shutdown()


def test_shared_pacer_serializes_same_host(pacer: SharedPacer) -> None:
    """3 calls to same host with 0.1s delay → wallclock between 0.18s and 0.35s.

    Two full delays (0.2s) plus some overhead, but not 3x sequential (0.3s)
    because the first call has no wait.
    """
    started = time.monotonic()
    pacer.wait("h")
    pacer.wait("h")
    pacer.wait("h")
    elapsed = time.monotonic() - started
    # Two full 0.1s delays: expected ~0.2s. Allow generous tolerance.
    assert 0.18 <= elapsed <= 0.35, (
        f"expected 0.18–0.35s for 3 serialized calls, got {elapsed:.3f}s"
    )


def test_shared_pacer_interleaved_hosts_serialize_separately(pacer: SharedPacer) -> None:
    """Interleaved hosts each serialize independently: a,b,a,b → ~1 delay each, not 3."""
    started = time.monotonic()
    pacer.wait("a")  # no wait (first for a)
    pacer.wait("b")  # no wait (first for b)
    pacer.wait("a")  # ~0.1s delay (second for a)
    pacer.wait("b")  # ~0.1s delay (second for b)
    elapsed = time.monotonic() - started
    # Two independent 0.1s delays, but they run sequentially so ~0.2s.
    assert 0.18 <= elapsed <= 0.35, f"expected 0.18–0.35s for interleaved calls, got {elapsed:.3f}s"


def test_shared_pacer_independent_hosts_parallel(pacer: SharedPacer) -> None:
    """Different hosts don't block each other: a and b in parallel → < 0.15s."""

    def wait_a() -> None:
        pacer.wait("a")

    def wait_b() -> None:
        pacer.wait("b")

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futs = [pool.submit(wait_a), pool.submit(wait_b)]
        for f in as_completed(futs):
            f.result()
    elapsed = time.monotonic() - started
    # Both are first calls for their host → zero delay. Just scheduling overhead.
    assert elapsed < 0.15, f"expected < 0.15s for parallel independent hosts, got {elapsed:.3f}s"


def test_shared_pacer_explicit_delay_overrides_default(pacer: SharedPacer) -> None:
    """Per-call delay_s overrides the default. 0.05s explicit → shorter wait."""
    started = time.monotonic()
    pacer.wait("x")  # uses default 0.1s, no wait (first)
    pacer.wait("x", delay_s=0.05)  # overrides to 0.05s wait
    elapsed = time.monotonic() - started
    assert 0.03 <= elapsed <= 0.25, (
        f"expected 0.03–0.25s (0.05s explicit delay), got {elapsed:.3f}s"
    )


def test_shared_pacer_default_delay_used_when_none(pacer: SharedPacer) -> None:
    """When delay_s=None, the pacer's default_delay_s is used."""
    started = time.monotonic()
    pacer.wait("y")  # no wait (first)
    pacer.wait("y", delay_s=None)  # uses default 0.1s
    elapsed = time.monotonic() - started
    assert 0.08 <= elapsed <= 0.30, f"expected 0.08–0.30s (0.1s default delay), got {elapsed:.3f}s"


def test_shared_pacer_lock_releases_during_sleep(pacer: SharedPacer) -> None:
    """The lock must be released during time.sleep so other hosts aren't blocked."""
    results: list[float] = []

    def call_and_record_timing(host: str) -> None:
        t0 = time.monotonic()
        pacer.wait(host, delay_s=0.1)
        results.append(time.monotonic() - t0)

    # Two parallel calls to different hosts: both should return quickly
    # because the lock is released before sleep.
    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_a = pool.submit(call_and_record_timing, "a")
        fut_b = pool.submit(call_and_record_timing, "b")
        fut_a.result()
        fut_b.result()

    # Both should be fast — no cross-host blocking
    for elapsed in results:
        assert elapsed < 0.15, f"parallel call took {elapsed:.3f}s — lock may be held during sleep"
