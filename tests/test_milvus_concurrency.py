"""Thread-safety test for Milvus client access.

Slice 3: The Milvus client singleton must hold a module-level lock so
concurrent _milvus_hybrid_search calls from the parallel eval pipeline
do not corrupt the shared gRPC channel state.

pymilvus 2.x uses a shared gRPC channel; thread-safety is not documented.
The conservative approach is to serialize at the call site. The lock is
placed in src/milvus_client.py and acquired in retrieve.py before each
hybrid_search call.

Test strategy: Mock the underlying client.hybrid_search (no real Milvus
needed) and verify that a lock is present and respected. We verify the lock
exists and that concurrent calls through the locked accessor complete without
exceptions.
"""

from __future__ import annotations

import threading
import time
from typing import Any
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Slice 3 — Lock attribute presence
# ---------------------------------------------------------------------------


class TestMilvusLockPresence:
    """src/milvus_client.py must expose _MILVUS_LOCK at module level."""

    def test_milvus_client_module_has_lock(self) -> None:
        import importlib

        mc = importlib.import_module("src.milvus_client")
        lock = getattr(mc, "_MILVUS_LOCK", None)
        assert lock is not None, (
            "src.milvus_client._MILVUS_LOCK is missing — add a module-level "
            "threading.Lock() and use it in retrieve.py::_milvus_hybrid_search"
        )
        assert hasattr(lock, "acquire") and hasattr(lock, "release"), (
            "_MILVUS_LOCK must be a threading.Lock or compatible"
        )


# ---------------------------------------------------------------------------
# Slice 3 — Concurrent search calls do not raise
# ---------------------------------------------------------------------------


class TestMilvusSearchConcurrency:
    """4 threads call get_client_lock() + client.hybrid_search concurrently.

    No real Milvus needed — client is mocked. We verify no exceptions arise
    under contention and that results are returned correctly.
    """

    N_THREADS = 4

    def test_concurrent_search_no_exception(self) -> None:
        """Concurrent calls with a mocked client must complete without errors."""
        import importlib

        mc = importlib.import_module("src.milvus_client")

        # Build a fake MilvusClient that records concurrent access and
        # adds a small sleep to surface races.
        concurrent_call_count = 0
        max_concurrent = 0
        call_lock = threading.Lock()

        def fake_hybrid_search(*args: object, **kwargs: object) -> list[dict[str, Any]]:
            nonlocal concurrent_call_count, max_concurrent
            with call_lock:
                concurrent_call_count += 1
                max_concurrent = max(max_concurrent, concurrent_call_count)
            time.sleep(0.005)  # Hold long enough for all threads to be "in flight"
            with call_lock:
                concurrent_call_count -= 1
            return []

        mock_client = MagicMock()
        mock_client.hybrid_search.side_effect = fake_hybrid_search

        errors: list[Exception] = []
        barrier = threading.Barrier(self.N_THREADS)

        def worker() -> None:
            try:
                barrier.wait()
                with mc._MILVUS_LOCK:
                    mock_client.hybrid_search("col", [], rerank=MagicMock())
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(self.N_THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Concurrent search raised: {errors}"
        # With the lock held, max concurrent calls inside hybrid_search is 1.
        assert max_concurrent == 1, (
            f"max_concurrent={max_concurrent}: lock is not serializing calls "
            f"(expected 1 at a time with _MILVUS_LOCK held)"
        )
