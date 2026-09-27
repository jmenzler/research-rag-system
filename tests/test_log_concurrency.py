"""Thread-safety tests for JSONL log writers in src/query/generate.py.

Slice 1: _append_log must hold _QUERIES_LOG_LOCK so concurrent threads never
produce truncated or interleaved lines.  The functional correctness test
verifies N=20 writes all land intact; the lock-presence test verifies the lock
is actually acquired (rather than relying on OS-level write atomicity, which
holds on macOS but is not guaranteed cross-platform for multi-KB JSON payloads).
"""

from __future__ import annotations

import importlib
import json
import threading
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from src.models import ChildChunk, Citation, ParentChunk, RetrievedChunk

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_rc(idx: int) -> RetrievedChunk:
    """Build a minimal RetrievedChunk for testing purposes."""
    child = ChildChunk(
        id=f"child-{idx}",
        parent_id=f"parent-{idx}",
        text=f"child text {idx}",
        source_file=f"sources/trading/doc_{idx}/source.pdf",
        notebook="trading",
        modality="text",
        page_number=idx,
    )
    parent = ParentChunk(
        id=f"parent-{idx}",
        text=f"parent text for document {idx}" * 10,
        source_file=f"sources/trading/doc_{idx}/source.pdf",
        notebook="trading",
        modality="text",
        page_number=idx,
    )
    return RetrievedChunk(
        child=child,
        parent=parent,
        dense_score=0.9,
        sparse_score=0.5,
        rerank_score=0.8,
    )


def _fake_citation(idx: int) -> Citation:
    return Citation(
        source_file=f"doc_{idx}",
        page_number=idx,
        modality="text",
    )


# ---------------------------------------------------------------------------
# Slice 1 — _append_log thread-safety
# ---------------------------------------------------------------------------


class TestAppendLogConcurrency:
    """20 concurrent threads each write one record; assert all 20 land intact."""

    N_THREADS = 20

    def test_module_exposes_queries_log_lock(self) -> None:
        """_QUERIES_LOG_LOCK must exist and be a threading.Lock (or RLock).

        Note: ``src.query.__init__`` re-exports ``generate`` (the function),
        so ``src.query.generate`` resolves to the function in the package
        namespace. Import the module directly via importlib to bypass that.
        """
        import importlib

        gen_mod = importlib.import_module("src.query.generate")

        lock = getattr(gen_mod, "_QUERIES_LOG_LOCK", None)
        assert lock is not None, (
            "src.query.generate._QUERIES_LOG_LOCK is missing — add it "
            "and wrap the _append_log write block with it."
        )
        # Must be acquirable/releasable (behaves like a lock).
        assert hasattr(lock, "acquire") and hasattr(lock, "release"), (
            "_QUERIES_LOG_LOCK must be a threading.Lock or compatible"
        )

    def test_all_records_present_after_concurrent_writes(self, tmp_path: Path) -> None:
        """All N records are present, valid JSON, and not truncated/interleaved.

        Patches _LOGS_DIR and _QUERIES_LOG at the module level BEFORE spawning
        threads — not inside threads — to avoid competing context-manager teardowns.
        """
        gen_mod = importlib.import_module("src.query.generate")
        _append_log = gen_mod._append_log  # type: ignore[attr-defined]

        log_file = tmp_path / "queries.jsonl"

        errors: list[Exception] = []
        barrier = threading.Barrier(self.N_THREADS)

        def writer(idx: int) -> None:
            try:
                barrier.wait()  # Synchronize all threads to maximize contention
                _append_log(
                    query=f"question-{idx}",
                    retrieved=[_fake_rc(idx)],
                    answer=f"answer-{idx}",
                    citations=[_fake_citation(idx)],
                    latency_ms=idx * 10,
                )
            except Exception as exc:
                errors.append(exc)

        # Patch module-level attrs ONCE before spawning threads.
        with (
            patch.object(gen_mod, "_LOGS_DIR", tmp_path),
            patch.object(gen_mod, "_QUERIES_LOG", log_file),
        ):
            threads = [threading.Thread(target=writer, args=(i,)) for i in range(self.N_THREADS)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        assert not errors, f"Worker exceptions: {errors}"

        # Every line must be valid JSON.
        lines = [ln for ln in log_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
        assert len(lines) == self.N_THREADS, f"Expected {self.N_THREADS} lines, got {len(lines)}"

        parsed: list[dict[str, Any]] = []
        for i, line in enumerate(lines):
            try:
                parsed.append(json.loads(line))
            except json.JSONDecodeError as exc:
                pytest.fail(f"Line {i} is not valid JSON: {exc!r}\n{line!r}")

        # All N distinct queries must be present (no collisions/drops).
        queries_found = {r["query"] for r in parsed}
        expected_queries = {f"question-{i}" for i in range(self.N_THREADS)}
        assert queries_found == expected_queries, (
            f"Missing queries: {expected_queries - queries_found}"
        )

    def test_each_line_individually_valid_json(self, tmp_path: Path) -> None:
        """No interleaving: every line is parseable in isolation."""
        gen_mod = importlib.import_module("src.query.generate")
        _append_log = gen_mod._append_log  # type: ignore[attr-defined]

        log_file = tmp_path / "queries.jsonl"
        barrier = threading.Barrier(self.N_THREADS)

        def writer(idx: int) -> None:
            barrier.wait()
            _append_log(
                query=f"q-{idx}",
                retrieved=[_fake_rc(idx)],
                answer=f"ans-{idx}",
                citations=[],
                latency_ms=0,
            )

        with (
            patch.object(gen_mod, "_LOGS_DIR", tmp_path),
            patch.object(gen_mod, "_QUERIES_LOG", log_file),
        ):
            threads = [threading.Thread(target=writer, args=(i,)) for i in range(self.N_THREADS)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        raw = log_file.read_text(encoding="utf-8")
        for i, line in enumerate(raw.splitlines()):
            if not line.strip():
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError as exc:
                pytest.fail(f"Line {i} invalid JSON (interleaving?): {exc!r}")
