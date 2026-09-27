"""Tests for src/eval/noise.py — BM25-adversarial noise injection + sensitivity scoring.

Tests mock Milvus search and synthesis so no real API calls are made.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _eval_row(i: int) -> dict[str, Any]:
    return {
        "question": f"question {i}",
        "answer": f"answer {i}",
        "contexts": [f"context {i}-1", f"context {i}-2"],
        "ground_truth": f"gt {i}",
    }


# ---------------------------------------------------------------------------
# Slice 1 — noun extraction
# ---------------------------------------------------------------------------


class TestExtractContentNouns:
    def test_extracts_content_words_excluding_stopwords(self) -> None:
        from src.eval.noise import _extract_content_nouns

        nouns = _extract_content_nouns(
            "What is the reservation price formula in Avellaneda-Stoikov?"
        )
        # Content words (4+ chars, non-stopword)
        assert "reservation" in nouns
        assert "price" in nouns
        assert "formula" in nouns
        # "what" is a stopword and < 4 chars anyway, should be excluded
        assert "what" not in nouns
        # short words excluded
        assert "the" not in nouns
        assert "is" not in nouns

    def test_deduplicates_words(self) -> None:
        from src.eval.noise import _extract_content_nouns

        nouns = _extract_content_nouns("market market market making making")
        # Each unique word appears once
        assert nouns.count("market") == 1
        assert nouns.count("making") == 1

    def test_returns_empty_for_no_content_words(self) -> None:
        from src.eval.noise import _extract_content_nouns

        nouns = _extract_content_nouns("the is at in on")
        assert nouns == []

    def test_limit_to_max_words(self) -> None:
        from src.eval.noise import _extract_content_nouns

        long_q = " ".join(f"important_term_{i}" for i in range(20))
        nouns = _extract_content_nouns(long_q, max_nouns=8)
        assert len(nouns) <= 8


# ---------------------------------------------------------------------------
# Slice 2 — noise injection core
# ---------------------------------------------------------------------------


class TestRunNoiseSensitivity:
    def test_returns_none_when_no_noise_found(self) -> None:
        from src.eval.noise import run_noise_sensitivity

        # Mock BM25 to return no hits
        mock_milvus = MagicMock()
        mock_milvus.search.return_value = [[]]

        row = _eval_row(0)
        with patch("src.eval.noise._bm25_search", return_value=[]):
            result = asyncio.run(
                run_noise_sensitivity(
                    row=row,
                    judge_model="test-model",
                    collection="trading",
                    notebook="test",
                    milvus_client=mock_milvus,
                )
            )
        # No noise chunks → cannot compute sensitivity.
        assert result is None

    def test_computes_delta_when_noise_found(self) -> None:
        from src.eval.noise import run_noise_sensitivity

        # Mock BM25 noise chunks.
        noise_texts = ["noise chunk A", "noise chunk B", "noise chunk C"]
        # Mock synthesis → returns a new answer.
        synth_answer = "answer with noise"

        async def mock_faithfulness(row: dict[str, Any], judge_model: str) -> float:
            return 0.85

        with (
            patch(
                "src.eval.noise._bm25_search",
                return_value=noise_texts,
            ),
            patch(
                "src.query.generate.synthesize_answer",
                return_value=(synth_answer, 100, MagicMock()),
            ),
            patch(
                "src.eval.metrics.faithfulness",
                side_effect=mock_faithfulness,
            ),
        ):
            # Inject a fake baseline faithfulness on the row.
            row = _eval_row(0)
            row["faithfulness"] = 0.95
            result = asyncio.run(
                run_noise_sensitivity(
                    row=row,
                    judge_model="test-model",
                    collection="trading",
                    notebook="test",
                )
            )

        assert result is not None
        # delta = baseline_F - F_with_noise = 0.95 - 0.85 = 0.10
        assert result["noise_delta_faithfulness"] == pytest.approx(0.10)
        assert result["noise_baseline_f"] == 0.95
        assert result["noise_f_with_noise"] == 0.85
        assert result["noise_n_injected"] == 3

    def test_returns_none_when_no_baseline_faithfulness(self) -> None:
        from src.eval.noise import run_noise_sensitivity

        noise_texts = ["noise A", "noise B"]
        synth_answer = "answer"

        async def mock_faithfulness(row: dict[str, Any], judge_model: str) -> float:
            return 0.80

        with (
            patch(
                "src.eval.noise._bm25_search",
                return_value=noise_texts,
            ),
            patch(
                "src.query.generate.synthesize_answer",
                return_value=(synth_answer, 100, MagicMock()),
            ),
            patch(
                "src.eval.metrics.faithfulness",
                side_effect=mock_faithfulness,
            ),
        ):
            row = _eval_row(0)
            # No "faithfulness" key on the row.
            result = asyncio.run(
                run_noise_sensitivity(
                    row=row,
                    judge_model="test-model",
                    collection="trading",
                    notebook="test",
                )
            )
        assert result is None

    def test_filters_out_already_retrieved_chunks(self) -> None:
        from src.eval.noise import run_noise_sensitivity

        # BM25 returns one chunk that's already in the original contexts.
        noise_texts = ["context 0-1", "fresh noise A", "fresh noise B"]
        synth_answer = "answer"

        async def mock_faithfulness(row: dict[str, Any], judge_model: str) -> float:
            return 0.75

        with (
            patch(
                "src.eval.noise._bm25_search",
                return_value=noise_texts,
            ),
            patch(
                "src.query.generate.synthesize_answer",
                return_value=(synth_answer, 100, MagicMock()),
            ),
            patch(
                "src.eval.metrics.faithfulness",
                side_effect=mock_faithfulness,
            ),
        ):
            row = _eval_row(0)
            row["faithfulness"] = 0.90
            result = asyncio.run(
                run_noise_sensitivity(
                    row=row,
                    judge_model="test-model",
                    collection="trading",
                    notebook="test",
                )
            )

        assert result is not None
        # "context 0-1" was filtered (already in original contexts).
        # Only 2 fresh noise chunks injected.
        assert result["noise_n_injected"] == 2


# ---------------------------------------------------------------------------
# Slice 3 — compute_all_noise wrapper
# ---------------------------------------------------------------------------


class TestComputeAllNoise:
    def test_skips_when_noise_flag_not_set(self) -> None:
        from src.eval.noise import compute_all_noise

        result = asyncio.run(
            compute_all_noise(
                rows=[_eval_row(0)],
                judge_model="test",
                collection="trading",
                notebook="test",
                extras=None,
            )
        )
        assert result == []

    def test_skips_when_noise_false_in_extras(self) -> None:
        from src.eval.noise import compute_all_noise

        result = asyncio.run(
            compute_all_noise(
                rows=[_eval_row(0)],
                judge_model="test",
                collection="trading",
                notebook="test",
                extras={"noise": False},
            )
        )
        assert result == []


# ---------------------------------------------------------------------------
# Slice 4 — _bm25_search Milvus schema integration
# ---------------------------------------------------------------------------


class TestBm25SearchMilvusSchema:
    """The BM25 search must request fields that actually exist on the Milvus
    schema (`parent_chunk_id`, NOT `parent_text` — parent text lives in the
    parents sqlite db, looked up by id). Verifies the two-step lookup works
    end-to-end with the same path retrieve.py uses for production queries."""

    def test_requests_parent_chunk_id_and_resolves_parent_text(self) -> None:
        from src.eval import noise

        # Mock Milvus returning two BM25 hits keyed on parent_chunk_id.
        # Note: NOT parent_text — schema only has parent_chunk_id.
        fake_hits = [
            {"id": "child-1", "entity": {"parent_chunk_id": "parent-1"}},
            {"id": "child-2", "entity": {"parent_chunk_id": "parent-2"}},
        ]
        mock_client = MagicMock()
        mock_client.hybrid_search.return_value = [fake_hits]

        # Mock parent sqlite returning text by parent_id.
        class FakeParent:
            def __init__(self, text: str) -> None:
                self.text = text

        fake_parents = {
            "parent-1": FakeParent("parent text one"),
            "parent-2": FakeParent("parent text two"),
        }

        with (
            patch("src.milvus_client.get_client", return_value=mock_client),
            patch("src.query.retrieve._lookup_parents", return_value=fake_parents),
        ):
            texts = noise._bm25_search(
                query="market making",
                collection="trading",
                notebook="trading",
            )

        # Verify the call used the correct output_fields (no parent_text).
        call_kwargs = mock_client.hybrid_search.call_args.kwargs
        assert call_kwargs["output_fields"] == ["parent_chunk_id"]

        # Verify resolved parent texts are returned.
        assert texts == ["parent text one", "parent text two"]

    def test_returns_empty_when_no_hits(self) -> None:
        from src.eval import noise

        mock_client = MagicMock()
        mock_client.hybrid_search.return_value = [[]]
        with patch("src.milvus_client.get_client", return_value=mock_client):
            texts = noise._bm25_search(query="anything", collection="trading", notebook="trading")
        assert texts == []

    def test_dedups_parent_ids(self) -> None:
        """Multiple child hits sharing the same parent_chunk_id collapse to
        one parent text. (Common in our pipeline because each parent has
        ~2 children.)"""
        from src.eval import noise

        fake_hits = [
            {"id": "child-1", "entity": {"parent_chunk_id": "parent-1"}},
            {"id": "child-2", "entity": {"parent_chunk_id": "parent-1"}},  # dup
            {"id": "child-3", "entity": {"parent_chunk_id": "parent-2"}},
        ]
        mock_client = MagicMock()
        mock_client.hybrid_search.return_value = [fake_hits]

        class FakeParent:
            def __init__(self, text: str) -> None:
                self.text = text

        fake_parents = {
            "parent-1": FakeParent("p1"),
            "parent-2": FakeParent("p2"),
        }
        with (
            patch("src.milvus_client.get_client", return_value=mock_client),
            patch("src.query.retrieve._lookup_parents", return_value=fake_parents),
        ):
            texts = noise._bm25_search(query="q", collection="trading", notebook="trading")

        assert texts == ["p1", "p2"]
