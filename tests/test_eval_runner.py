"""Tests for src/eval/runner.py — orchestrator over the four metrics.

``run_our_metrics(rows, judge_model, max_workers)`` wires the four metric
functions over a list of eval rows (``question``, ``answer``, ``contexts``,
``ground_truth``), maps them onto our internal row shape (``user_input``,
``response``, ``retrieved_contexts``, ``reference``), and returns a
per-question + aggregate score dict.

Tests mock the four metric functions so the runner is exercised in isolation
— no LLM, no embed.
"""

from __future__ import annotations

import asyncio
import math
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _eval_row(i: int) -> dict[str, Any]:
    return {
        "question": f"q{i}",
        "answer": f"a{i}",
        "contexts": [f"c{i}-1", f"c{i}-2"],
        "ground_truth": f"gt{i}",
    }


# ---------------------------------------------------------------------------
# Slice 8 — runner shape and aggregate math
# ---------------------------------------------------------------------------


class TestRunOurMetricsShape:
    def test_returns_per_question_and_aggregate(self) -> None:
        from src.eval.runner import run_our_metrics

        rows = [_eval_row(i) for i in range(3)]
        with (
            patch("src.eval.runner.faithfulness", new=AsyncMock(return_value=0.8)),
            patch("src.eval.runner.answer_relevancy", new=AsyncMock(return_value=0.9)),
            patch("src.eval.runner.context_precision", new=AsyncMock(return_value=0.7)),
            patch("src.eval.runner.context_recall", new=AsyncMock(return_value=0.6)),
        ):
            result = asyncio.run(
                run_our_metrics(rows=rows, judge_model="gemini-3-flash-preview", max_workers=4)
            )

        assert "per_question" in result
        assert "aggregate" in result
        assert len(result["per_question"]) == 3

        # Each per-question entry has all 4 metrics
        for entry in result["per_question"]:
            assert entry["faithfulness"] == 0.8
            assert entry["answer_relevancy"] == 0.9
            assert entry["context_precision"] == 0.7
            assert entry["context_recall"] == 0.6
            assert "question" in entry

        # Aggregate is the mean of per-row scores
        assert result["aggregate"]["faithfulness"] == pytest.approx(0.8)
        assert result["aggregate"]["answer_relevancy"] == pytest.approx(0.9)
        assert result["aggregate"]["context_precision"] == pytest.approx(0.7)
        assert result["aggregate"]["context_recall"] == pytest.approx(0.6)

    def test_handles_n_equals_one(self) -> None:
        from src.eval.runner import run_our_metrics

        with (
            patch("src.eval.runner.faithfulness", new=AsyncMock(return_value=0.5)),
            patch("src.eval.runner.answer_relevancy", new=AsyncMock(return_value=0.5)),
            patch("src.eval.runner.context_precision", new=AsyncMock(return_value=0.5)),
            patch("src.eval.runner.context_recall", new=AsyncMock(return_value=0.5)),
        ):
            result = asyncio.run(
                run_our_metrics(
                    rows=[_eval_row(0)],
                    judge_model="gemini-3-flash-preview",
                    max_workers=4,
                )
            )
        assert len(result["per_question"]) == 1

    def test_aggregate_excludes_nans(self) -> None:
        """NaN per-row scores should be skipped in the aggregate."""
        from src.eval.runner import run_our_metrics

        rows = [_eval_row(i) for i in range(2)]
        scores = [math.nan, 0.6]

        async def _faith(row: dict[str, object], judge_model: str) -> float:
            return scores.pop(0)

        with (
            patch("src.eval.runner.faithfulness", side_effect=_faith),
            patch("src.eval.runner.answer_relevancy", new=AsyncMock(return_value=0.5)),
            patch("src.eval.runner.context_precision", new=AsyncMock(return_value=0.5)),
            patch("src.eval.runner.context_recall", new=AsyncMock(return_value=0.5)),
        ):
            result = asyncio.run(
                run_our_metrics(rows=rows, judge_model="gemini-3-flash-preview", max_workers=4)
            )

        # Aggregate of [nan, 0.6] = 0.6 (NaN skipped)
        assert result["aggregate"]["faithfulness"] == 0.6


class TestRowFieldAdapter:
    """Runner must rename eval row fields to our metric input shape."""

    def test_passes_renamed_row_to_metric(self) -> None:
        from src.eval.runner import run_our_metrics

        captured: list[dict[str, Any]] = []

        async def capturing_faithfulness(row: dict[str, Any], judge_model: str) -> float:
            captured.append(dict(row))
            return 1.0

        with (
            patch("src.eval.runner.faithfulness", side_effect=capturing_faithfulness),
            patch("src.eval.runner.answer_relevancy", new=AsyncMock(return_value=1.0)),
            patch("src.eval.runner.context_precision", new=AsyncMock(return_value=1.0)),
            patch("src.eval.runner.context_recall", new=AsyncMock(return_value=1.0)),
        ):
            asyncio.run(
                run_our_metrics(
                    rows=[_eval_row(0)],
                    judge_model="gemini-3-flash-preview",
                    max_workers=4,
                )
            )

        assert len(captured) == 1
        adapted = captured[0]
        assert adapted["user_input"] == "q0"
        assert adapted["response"] == "a0"
        assert adapted["retrieved_contexts"] == ["c0-1", "c0-2"]
        assert adapted["reference"] == "gt0"


class TestConcurrencyCap:
    """The semaphore must cap concurrent metric calls to ``max_workers``."""

    def test_max_workers_caps_concurrency(self) -> None:
        from src.eval.runner import run_our_metrics

        in_flight = 0
        peak = 0

        async def slow(row: dict[str, object], judge_model: str) -> float:
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            # Yield so other tasks can pile up if not capped
            await asyncio.sleep(0.01)
            in_flight -= 1
            return 1.0

        rows = [_eval_row(i) for i in range(8)]
        with (
            patch("src.eval.runner.faithfulness", side_effect=slow),
            patch("src.eval.runner.answer_relevancy", side_effect=slow),
            patch("src.eval.runner.context_precision", side_effect=slow),
            patch("src.eval.runner.context_recall", side_effect=slow),
        ):
            asyncio.run(
                run_our_metrics(rows=rows, judge_model="gemini-3-flash-preview", max_workers=2)
            )

        # With 8 rows × 4 metrics = 32 coroutines but max_workers=2,
        # peak in-flight must NOT exceed 2.
        assert peak <= 2, f"Concurrency cap breached: peak={peak}, expected <=2"


# ---------------------------------------------------------------------------
# Slice 9 — extras flag plumbing
# ---------------------------------------------------------------------------


class TestExtrasPlumbing:
    def test_no_extras_keeps_existing_behavior(self) -> None:
        """Default extras=None must produce the same output shape as before."""
        from src.eval.runner import run_our_metrics

        rows = [_eval_row(i) for i in range(2)]
        with (
            patch("src.eval.runner.faithfulness", new=AsyncMock(return_value=0.8)),
            patch("src.eval.runner.answer_relevancy", new=AsyncMock(return_value=0.9)),
            patch("src.eval.runner.context_precision", new=AsyncMock(return_value=0.7)),
            patch("src.eval.runner.context_recall", new=AsyncMock(return_value=0.6)),
        ):
            result = asyncio.run(
                run_our_metrics(
                    rows=rows,
                    judge_model="g",
                    max_workers=4,
                    extras=None,
                )
            )

        # Same shape as before.
        assert set(result["aggregate"].keys()) == {
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
        }
        assert set(result["per_question"][0].keys()) == {
            "question",
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
        }

    def test_extras_empty_dict_same_as_none(self) -> None:
        """Empty dict is equivalent to None — no extra metrics."""
        from src.eval.runner import run_our_metrics

        with (
            patch("src.eval.runner.faithfulness", new=AsyncMock(return_value=0.8)),
            patch("src.eval.runner.answer_relevancy", new=AsyncMock(return_value=0.9)),
            patch("src.eval.runner.context_precision", new=AsyncMock(return_value=0.7)),
            patch("src.eval.runner.context_recall", new=AsyncMock(return_value=0.6)),
        ):
            result = asyncio.run(
                run_our_metrics(
                    rows=[_eval_row(0)],
                    judge_model="g",
                    max_workers=4,
                    extras={},
                )
            )
        assert "citation_accuracy" not in result["per_question"][0]

    def test_citation_extra_adds_field_to_output(self) -> None:
        from src.eval.runner import run_our_metrics

        with (
            patch("src.eval.runner.faithfulness", new=AsyncMock(return_value=0.8)),
            patch("src.eval.runner.answer_relevancy", new=AsyncMock(return_value=0.9)),
            patch("src.eval.runner.context_precision", new=AsyncMock(return_value=0.7)),
            patch("src.eval.runner.context_recall", new=AsyncMock(return_value=0.6)),
            patch("src.eval.aspects.citation_accuracy", new=AsyncMock(return_value=0.85)),
        ):
            result = asyncio.run(
                run_our_metrics(
                    rows=[_eval_row(0)],
                    judge_model="g",
                    max_workers=4,
                    extras={"citation": True},
                )
            )
        pq = result["per_question"][0]
        assert pq["citation_accuracy"] == 0.85
        assert "citation_accuracy" in result["aggregate"]

    def test_disambiguation_extra_adds_field_to_output(self) -> None:
        from src.eval.runner import run_our_metrics

        with (
            patch("src.eval.runner.faithfulness", new=AsyncMock(return_value=0.8)),
            patch("src.eval.runner.answer_relevancy", new=AsyncMock(return_value=0.9)),
            patch("src.eval.runner.context_precision", new=AsyncMock(return_value=0.7)),
            patch("src.eval.runner.context_recall", new=AsyncMock(return_value=0.6)),
            patch("src.eval.aspects.concept_disambiguation", new=AsyncMock(return_value=1.0)),
        ):
            result = asyncio.run(
                run_our_metrics(
                    rows=[_eval_row(0)],
                    judge_model="g",
                    max_workers=4,
                    extras={"disambiguation": True},
                )
            )
        pq = result["per_question"][0]
        assert pq["concept_disambiguation"] == 1.0
        assert "concept_disambiguation" in result["aggregate"]

    def test_both_extras_together(self) -> None:
        from src.eval.runner import run_our_metrics

        with (
            patch("src.eval.runner.faithfulness", new=AsyncMock(return_value=0.8)),
            patch("src.eval.runner.answer_relevancy", new=AsyncMock(return_value=0.9)),
            patch("src.eval.runner.context_precision", new=AsyncMock(return_value=0.7)),
            patch("src.eval.runner.context_recall", new=AsyncMock(return_value=0.6)),
            patch("src.eval.aspects.citation_accuracy", new=AsyncMock(return_value=0.85)),
            patch("src.eval.aspects.concept_disambiguation", new=AsyncMock(return_value=1.0)),
        ):
            result = asyncio.run(
                run_our_metrics(
                    rows=[_eval_row(0)],
                    judge_model="g",
                    max_workers=4,
                    extras={"citation": True, "disambiguation": True},
                )
            )
        pq = result["per_question"][0]
        assert pq["citation_accuracy"] == 0.85
        assert pq["concept_disambiguation"] == 1.0
        assert result["aggregate"]["citation_accuracy"] == 0.85
        assert result["aggregate"]["concept_disambiguation"] == 1.0

    def test_extras_aggregate_excludes_nans(self) -> None:
        import math

        from src.eval.runner import run_our_metrics

        calls = 0

        async def _citation(row: dict[str, Any], judge_model: str) -> float:
            nonlocal calls
            calls += 1
            return math.nan if calls == 1 else 1.0

        with (
            patch("src.eval.runner.faithfulness", new=AsyncMock(return_value=0.8)),
            patch("src.eval.runner.answer_relevancy", new=AsyncMock(return_value=0.9)),
            patch("src.eval.runner.context_precision", new=AsyncMock(return_value=0.7)),
            patch("src.eval.runner.context_recall", new=AsyncMock(return_value=0.6)),
            patch("src.eval.aspects.citation_accuracy", side_effect=_citation),
        ):
            result = asyncio.run(
                run_our_metrics(
                    rows=[_eval_row(0), _eval_row(1)],
                    judge_model="g",
                    max_workers=4,
                    extras={"citation": True},
                )
            )
        # Aggregate of [nan, 1.0] = 1.0
        assert result["aggregate"]["citation_accuracy"] == 1.0


# ---------------------------------------------------------------------------
# Slice 10 — noise sensitivity wiring (extras["noise"])
# ---------------------------------------------------------------------------


class TestNoiseSensitivityWiring:
    """The noise extra is shape-different from aspect critics (returns a dict
    of fields, needs collection/notebook context). Verify it plumbs through
    run_our_metrics correctly: per-row fields appear in per_question, and
    the delta_faithfulness aggregate appears in aggregate."""

    def test_noise_extra_adds_per_row_fields(self) -> None:
        from src.eval.runner import run_our_metrics

        async def _mock_compute_all_noise(
            rows: list[dict[str, Any]],
            judge_model: str,
            collection: str,
            notebook: str,
            extras: dict[str, bool] | None = None,
        ) -> list[dict[str, Any] | None]:
            return [
                {
                    "noise_delta_faithfulness": 0.10,
                    "noise_baseline_f": 0.95,
                    "noise_f_with_noise": 0.85,
                    "noise_n_injected": 3,
                }
            ]

        with (
            patch("src.eval.runner.faithfulness", new=AsyncMock(return_value=0.95)),
            patch("src.eval.runner.answer_relevancy", new=AsyncMock(return_value=0.9)),
            patch("src.eval.runner.context_precision", new=AsyncMock(return_value=0.7)),
            patch("src.eval.runner.context_recall", new=AsyncMock(return_value=0.6)),
            patch("src.eval.noise.compute_all_noise", side_effect=_mock_compute_all_noise),
        ):
            result = asyncio.run(
                run_our_metrics(
                    rows=[_eval_row(0)],
                    judge_model="g",
                    max_workers=4,
                    extras={"noise": True},
                    collection="trading",
                    notebook="trading",
                )
            )
        pq = result["per_question"][0]
        assert pq["noise_delta_faithfulness"] == 0.10
        assert pq["noise_baseline_f"] == 0.95
        assert pq["noise_f_with_noise"] == 0.85
        assert pq["noise_n_injected"] == 3
        assert "noise_delta_faithfulness" in result["aggregate"]
        assert result["aggregate"]["noise_delta_faithfulness"] == 0.10

    def test_noise_extra_skipped_without_collection(self) -> None:
        """When extras['noise']=True but collection/notebook missing, skip
        gracefully and leave noise fields out — don't crash."""
        from src.eval.runner import run_our_metrics

        with (
            patch("src.eval.runner.faithfulness", new=AsyncMock(return_value=0.95)),
            patch("src.eval.runner.answer_relevancy", new=AsyncMock(return_value=0.9)),
            patch("src.eval.runner.context_precision", new=AsyncMock(return_value=0.7)),
            patch("src.eval.runner.context_recall", new=AsyncMock(return_value=0.6)),
        ):
            result = asyncio.run(
                run_our_metrics(
                    rows=[_eval_row(0)],
                    judge_model="g",
                    max_workers=4,
                    extras={"noise": True},
                    collection=None,
                    notebook=None,
                )
            )
        pq = result["per_question"][0]
        assert "noise_delta_faithfulness" not in pq
        assert "noise_delta_faithfulness" not in result["aggregate"]

    def test_noise_extra_handles_none_results(self) -> None:
        """compute_all_noise returns None for rows where noise injection fails
        (e.g. no BM25 hits, no baseline F). Those rows must NOT crash and
        must be excluded from the aggregate."""
        from src.eval.runner import run_our_metrics

        async def _mock_mixed(
            rows: list[dict[str, Any]],
            judge_model: str,
            collection: str,
            notebook: str,
            extras: dict[str, bool] | None = None,
        ) -> list[dict[str, Any] | None]:
            return [
                {
                    "noise_delta_faithfulness": 0.05,
                    "noise_baseline_f": 0.90,
                    "noise_f_with_noise": 0.85,
                    "noise_n_injected": 2,
                },
                None,  # second row failed
            ]

        with (
            patch("src.eval.runner.faithfulness", new=AsyncMock(return_value=0.9)),
            patch("src.eval.runner.answer_relevancy", new=AsyncMock(return_value=0.9)),
            patch("src.eval.runner.context_precision", new=AsyncMock(return_value=0.7)),
            patch("src.eval.runner.context_recall", new=AsyncMock(return_value=0.6)),
            patch("src.eval.noise.compute_all_noise", side_effect=_mock_mixed),
        ):
            result = asyncio.run(
                run_our_metrics(
                    rows=[_eval_row(0), _eval_row(1)],
                    judge_model="g",
                    max_workers=4,
                    extras={"noise": True},
                    collection="trading",
                    notebook="trading",
                )
            )
        # Row 0 has noise fields, row 1 doesn't.
        assert result["per_question"][0]["noise_delta_faithfulness"] == 0.05
        assert "noise_delta_faithfulness" not in result["per_question"][1]
        # Aggregate is the mean over surviving rows only.
        assert result["aggregate"]["noise_delta_faithfulness"] == 0.05
