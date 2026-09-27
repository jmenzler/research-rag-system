"""Tests for parallelized _build_rows_from_pipeline in src/evaluate.py.

Slice 5: _build_rows_from_pipeline must accept a pipeline_workers parameter.
When workers > 1, it uses ThreadPoolExecutor and preserves question order.
Results must be identical across workers=1, 4, 8. Worker exceptions must
propagate to the caller (no silent swallowing).

Slice 6: Per-question print must show elapsed time in seconds.
"""

from __future__ import annotations

import re
from typing import Any
from unittest.mock import patch

import pytest

from src.models import Citation, RAGResponse, Usage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_rag_response(query: str) -> RAGResponse:
    """Return a deterministic RAGResponse based on the query string."""
    return RAGResponse(
        query=query,
        answer=f"answer-for-{query}",
        citations=[Citation(source_file="doc.pdf", page_number=1, modality="text")],
        retrieved=[],
        latency_ms=10,
        usage=Usage(input=100, output=50, total=150),
    )


def _make_golden_rows(n: int) -> list[dict[str, Any]]:
    """Build N unique golden rows."""
    return [
        {
            "question": f"question-{i}",
            "ground_truth": f"truth-{i}",
            "notebook": "trading",
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Slice 5 — _build_rows_from_pipeline signature accepts pipeline_workers
# ---------------------------------------------------------------------------


class TestBuildRowsSignature:
    def test_accepts_pipeline_workers_kwarg(self) -> None:
        import inspect

        from src.evaluate import _build_rows_from_pipeline

        sig = inspect.signature(_build_rows_from_pipeline)
        assert "pipeline_workers" in sig.parameters, (
            "_build_rows_from_pipeline must accept pipeline_workers keyword arg (int, default 1)"
        )

    def test_default_pipeline_workers_is_one(self) -> None:
        import inspect

        from src.evaluate import _build_rows_from_pipeline

        sig = inspect.signature(_build_rows_from_pipeline)
        param = sig.parameters["pipeline_workers"]
        assert param.default == 1, f"pipeline_workers default must be 1, got {param.default}"


# ---------------------------------------------------------------------------
# Slice 5 — order preservation and correctness across worker counts
# ---------------------------------------------------------------------------


class TestBuildRowsParallel:
    """Results at workers=1,4,8 must be identical and order-preserving."""

    N = 10

    def _mock_pipeline(self, query: str, **kwargs: object) -> tuple[RAGResponse, dict[str, Any]]:
        """Deterministic fake pipeline: answer depends only on query text."""
        return _fake_rag_response(query), {}

    def _run_at_workers(
        self,
        workers: int,
        golden_rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        from src.evaluate import _build_rows_from_pipeline

        with patch(
            "src.evaluate.run_query_pipeline",
            side_effect=lambda query, **kw: self._mock_pipeline(query, **kw),
        ):
            return _build_rows_from_pipeline(
                golden_rows=golden_rows,
                notebook="trading",
                collection="trading",
                use_stepback=None,
                use_crag=None,
                pipeline_workers=workers,
            )

    def test_sequential_results_match_golden_questions(self) -> None:
        golden = _make_golden_rows(self.N)
        rows = self._run_at_workers(1, golden)
        assert len(rows) == self.N
        for i, row in enumerate(rows):
            assert row["question"] == f"question-{i}"
            assert row["answer"] == f"answer-for-question-{i}"
            assert row["ground_truth"] == f"truth-{i}"

    def test_parallel_4_same_as_sequential(self) -> None:
        golden = _make_golden_rows(self.N)
        seq_rows = self._run_at_workers(1, golden)
        par_rows = self._run_at_workers(4, golden)
        assert len(par_rows) == len(seq_rows) == self.N
        for seq, par in zip(seq_rows, par_rows):
            assert seq["question"] == par["question"]
            assert seq["answer"] == par["answer"]
            assert seq["ground_truth"] == par["ground_truth"]

    def test_parallel_8_same_as_sequential(self) -> None:
        golden = _make_golden_rows(self.N)
        seq_rows = self._run_at_workers(1, golden)
        par_rows = self._run_at_workers(8, golden)
        for seq, par in zip(seq_rows, par_rows):
            assert seq["question"] == par["question"]
            assert seq["answer"] == par["answer"]

    def test_order_preserved_at_workers_4(self) -> None:
        """Questions must appear in original order regardless of thread completion order."""
        golden = _make_golden_rows(self.N)
        par_rows = self._run_at_workers(4, golden)
        for i, row in enumerate(par_rows):
            assert row["question"] == f"question-{i}", (
                f"Row {i} has wrong question: {row['question']!r}"
            )

    def test_contexts_are_empty_list_when_no_retrieved(self) -> None:
        """RAGResponse.retrieved=[] → contexts is []."""
        golden = _make_golden_rows(3)
        rows = self._run_at_workers(1, golden)
        for row in rows:
            assert row["contexts"] == []


# ---------------------------------------------------------------------------
# Slice 5 — exception propagation from worker
# ---------------------------------------------------------------------------


class TestBuildRowsExceptionPropagation:
    """A failing pipeline call must propagate, not be swallowed silently."""

    def test_exception_propagates_to_caller(self) -> None:
        from src.evaluate import _build_rows_from_pipeline

        golden = _make_golden_rows(5)

        def bad_pipeline(query: str, **kwargs: object) -> tuple[RAGResponse, dict[str, Any]]:
            if query == "question-2":
                raise RuntimeError("Injected pipeline failure")
            return _fake_rag_response(query), {}

        with patch(
            "src.evaluate.run_query_pipeline",
            side_effect=lambda query, **kw: bad_pipeline(query, **kw),
        ):
            with pytest.raises(RuntimeError, match="Injected pipeline failure"):
                _build_rows_from_pipeline(
                    golden_rows=golden,
                    notebook="trading",
                    collection="trading",
                    use_stepback=None,
                    use_crag=None,
                    pipeline_workers=4,
                )

    def test_exception_propagates_sequential_too(self) -> None:
        """Even at workers=1, pipeline errors must propagate."""
        from src.evaluate import _build_rows_from_pipeline

        golden = _make_golden_rows(3)

        def bad_pipeline(query: str, **kwargs: object) -> tuple[RAGResponse, dict[str, Any]]:
            if query == "question-1":
                raise ValueError("Sequential failure")
            return _fake_rag_response(query), {}

        with patch(
            "src.evaluate.run_query_pipeline",
            side_effect=lambda query, **kw: bad_pipeline(query, **kw),
        ):
            with pytest.raises(ValueError, match="Sequential failure"):
                _build_rows_from_pipeline(
                    golden_rows=golden,
                    notebook="trading",
                    collection="trading",
                    use_stepback=None,
                    use_crag=None,
                    pipeline_workers=1,
                )


# ---------------------------------------------------------------------------
# Slice 5 — no-chunks fallback
# ---------------------------------------------------------------------------


class TestNoChunksFallback:
    """When pipeline returns None, a placeholder RAGResponse is used."""

    def test_none_response_becomes_placeholder(self) -> None:
        from src.evaluate import _build_rows_from_pipeline

        golden = _make_golden_rows(3)

        def nil_pipeline(query: str, **kwargs: object) -> tuple[None, dict[str, Any]]:
            return None, {}

        with patch(
            "src.evaluate.run_query_pipeline",
            side_effect=lambda query, **kw: nil_pipeline(query, **kw),
        ):
            rows = _build_rows_from_pipeline(
                golden_rows=golden,
                notebook="trading",
                collection="trading",
                use_stepback=None,
                use_crag=None,
                pipeline_workers=1,
            )

        assert len(rows) == 3
        for row in rows:
            assert row["answer"] == "(no chunks retrieved)"


# ---------------------------------------------------------------------------
# Slice 5 — argparse --pipeline-workers flag
# ---------------------------------------------------------------------------


class TestArgparsePipelineWorkers:
    def test_pipeline_workers_flag_exists(self) -> None:
        from src.evaluate import _parse_args

        args = _parse_args(
            ["--notebook", "trading", "--collection", "trading", "--pipeline-workers", "4"]
        )
        assert args.pipeline_workers == 4

    def test_pipeline_workers_default_is_1(self) -> None:
        from src.evaluate import _parse_args

        args = _parse_args(["--notebook", "trading", "--collection", "trading"])
        assert args.pipeline_workers == 1


# ---------------------------------------------------------------------------
# Slice 6 — per-question print format shows elapsed time
# ---------------------------------------------------------------------------


class TestPerQuestionPrint:
    """The print after each question must show elapsed time in seconds."""

    def test_print_shows_elapsed_time(self, capsys: pytest.CaptureFixture[str]) -> None:
        from src.evaluate import _build_rows_from_pipeline

        golden = _make_golden_rows(2)

        def fake_pipeline(query: str, **kwargs: object) -> tuple[RAGResponse, dict[str, Any]]:
            return _fake_rag_response(query), {}

        with patch(
            "src.evaluate.run_query_pipeline",
            side_effect=lambda query, **kw: fake_pipeline(query, **kw),
        ):
            _build_rows_from_pipeline(
                golden_rows=golden,
                notebook="trading",
                collection="trading",
                use_stepback=None,
                use_crag=None,
                pipeline_workers=1,
            )

        out = capsys.readouterr().out
        # Each line should contain elapsed time like "done in X.Xs"
        lines = [ln for ln in out.splitlines() if "[evaluate]" in ln]
        assert len(lines) >= 2
        for line in lines:
            assert "done in" in line, f"Expected 'done in X.Xs' in print output, got: {line!r}"
            assert re.search(r"done in \d+\.\d+s", line), (
                f"Expected 'done in N.Ns' pattern, got: {line!r}"
            )
