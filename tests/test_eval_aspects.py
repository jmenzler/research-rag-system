"""Tests for src/eval/aspects.py — citation accuracy + concept disambiguation critics.

Both aspects are LLM-judge critics that score a RAG answer on a specific dimension.
Tests mock the LLM call (``call_with_schema``) so no real API calls are made.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, patch

# ---------------------------------------------------------------------------
# Slice 1 — Schema validation (both aspects produce parseable output)
# ---------------------------------------------------------------------------


class TestCitationAccuracySchema:
    def test_parses_valid_output(self) -> None:
        from src.eval.schemas import CitationAccuracyOutput

        raw = {
            "checks": [
                {
                    "citation_index": 1,
                    "claim": "The reservation price is r = s - q*gamma*sigma^2*(T-t)",
                    "verdict": 1,
                    "reason": "This formula appears verbatim in Avellaneda-Stoikov (2008).",
                },
                {
                    "citation_index": 2,
                    "claim": "Glosten-Milgrom uses a different information structure.",
                    "verdict": 0,
                    "reason": "The cited paper does not discuss Glosten-Milgrom.",
                },
            ]
        }
        parsed = CitationAccuracyOutput.model_validate(raw)
        assert len(parsed.checks) == 2
        assert parsed.checks[0].verdict == 1
        assert parsed.checks[1].verdict == 0

    def test_empty_checks_is_valid(self) -> None:
        from src.eval.schemas import CitationAccuracyOutput

        parsed = CitationAccuracyOutput.model_validate({"checks": []})
        assert parsed.checks == []


class TestConceptDisambiguationSchema:
    def test_parses_disambiguated_verdict(self) -> None:
        from src.eval.schemas import ConceptDisambiguationOutput

        raw = {
            "verdict": 1,
            "reason": "Answer distinguishes AS inventory control from GM adverse selection.",
        }
        parsed = ConceptDisambiguationOutput.model_validate(raw)
        assert parsed.verdict == 1

    def test_parses_not_disambiguated_verdict(self) -> None:
        from src.eval.schemas import ConceptDisambiguationOutput

        raw = {
            "verdict": 0,
            "reason": "The answer conflates LVR with total PnL without explanation.",
        }
        parsed = ConceptDisambiguationOutput.model_validate(raw)
        assert parsed.verdict == 0


# ---------------------------------------------------------------------------
# Slice 2 — citation_accuracy critic (mocked LLM)
# ---------------------------------------------------------------------------


class TestCitationAccuracy:
    def test_returns_fraction_of_citations_that_pass(self) -> None:
        from src.eval.aspects import citation_accuracy
        from src.eval.schemas import CitationAccuracyOutput, CitationCheckResult

        # 4 citations in the answer, judge says 3 pass.
        mock_output = CitationAccuracyOutput(
            checks=[
                CitationCheckResult(citation_index=1, claim="X", verdict=1, reason="ok"),
                CitationCheckResult(citation_index=2, claim="Y", verdict=1, reason="ok"),
                CitationCheckResult(citation_index=3, claim="Z", verdict=0, reason="wrong"),
                CitationCheckResult(citation_index=4, claim="W", verdict=1, reason="ok"),
            ]
        )

        with patch(
            "src.eval.aspects.call_with_schema",
            new=AsyncMock(return_value=(mock_output, None)),
        ):
            score = asyncio.run(
                citation_accuracy(
                    row={
                        "user_input": "What is LRV?",
                        "response": "LRV is ... [1] ... [2] ... [3] ... [4]",
                        "retrieved_contexts": ["ctx1", "ctx2"],
                    },
                    judge_model="test-model",
                )
            )
        assert score == 0.75  # 3 of 4

    def test_returns_nan_when_no_citations_detected(self) -> None:
        from src.eval.aspects import citation_accuracy

        # Answer with no citation markers → no citations to check.
        with patch(
            "src.eval.aspects.call_with_schema",
            new=AsyncMock(return_value=(None, None)),
        ):
            score = asyncio.run(
                citation_accuracy(
                    row={
                        "user_input": "What is LVR?",
                        "response": "LVR is loss-versus-rebalancing.",
                        "retrieved_contexts": ["ctx1"],
                    },
                    judge_model="test-model",
                )
            )
        assert not (score == score)  # NaN check

    def test_returns_nan_on_parse_failure(self) -> None:
        from src.eval.aspects import citation_accuracy

        with patch(
            "src.eval.aspects.call_with_schema",
            new=AsyncMock(return_value=(None, None)),
        ):
            score = asyncio.run(
                citation_accuracy(
                    row={
                        "user_input": "q?",
                        "response": "answer [1]",
                        "retrieved_contexts": ["ctx1"],
                    },
                    judge_model="test-model",
                )
            )
        import math

        assert math.isnan(score)

    def test_uses_correct_row_fields(self) -> None:
        from src.eval.aspects import citation_accuracy

        captured_input: dict[str, Any] = {}

        async def capture_call(
            model: str,
            system: str,
            user: str,
            schema: object,
        ) -> tuple:
            captured_input["user_prompt"] = user
            from src.eval.schemas import CitationAccuracyOutput

            return CitationAccuracyOutput(checks=[]), None

        with patch("src.eval.aspects.call_with_schema", side_effect=capture_call):
            asyncio.run(
                citation_accuracy(
                    row={
                        "user_input": "Explain market making.",
                        "response": "The answer [1] explains it.",
                        "retrieved_contexts": ["context text one", "context text two"],
                    },
                    judge_model="gemini-3-flash",
                )
            )

        prompt = captured_input["user_prompt"]
        assert "Explain market making" in prompt
        assert "The answer [1] explains it" in prompt
        assert "context text one" in prompt


# ---------------------------------------------------------------------------
# Slice 3 — concept_disambiguation critic (mocked LLM)
# ---------------------------------------------------------------------------


class TestConceptDisambiguation:
    def test_returns_one_when_disambiguated(self) -> None:
        from src.eval.aspects import concept_disambiguation
        from src.eval.schemas import ConceptDisambiguationOutput

        mock_output = ConceptDisambiguationOutput(
            verdict=1,
            reason="Correctly distinguishes AS from Glosten-Milgrom.",
        )

        with patch(
            "src.eval.aspects.call_with_schema",
            new=AsyncMock(return_value=(mock_output, None)),
        ):
            score = asyncio.run(
                concept_disambiguation(
                    row={
                        "user_input": "How does AS compare to GM?",
                        "response": "AS uses inventory control; GM uses adverse selection.",
                        "retrieved_contexts": ["ctx1"],
                    },
                    judge_model="test-model",
                )
            )
        assert score == 1.0

    def test_returns_zero_when_conflated(self) -> None:
        from src.eval.aspects import concept_disambiguation
        from src.eval.schemas import ConceptDisambiguationOutput

        mock_output = ConceptDisambiguationOutput(
            verdict=0,
            reason="Answer merges two distinct concepts.",
        )

        with patch(
            "src.eval.aspects.call_with_schema",
            new=AsyncMock(return_value=(mock_output, None)),
        ):
            score = asyncio.run(
                concept_disambiguation(
                    row={
                        "user_input": "Compare AS and GM.",
                        "response": "They are the same thing.",
                        "retrieved_contexts": ["ctx1"],
                    },
                    judge_model="test-model",
                )
            )
        assert score == 0.0

    def test_returns_nan_on_parse_failure(self) -> None:
        from src.eval.aspects import concept_disambiguation

        with patch(
            "src.eval.aspects.call_with_schema",
            new=AsyncMock(return_value=(None, None)),
        ):
            score = asyncio.run(
                concept_disambiguation(
                    row={
                        "user_input": "q?",
                        "response": "a",
                        "retrieved_contexts": ["ctx1"],
                    },
                    judge_model="test-model",
                )
            )
        import math

        assert math.isnan(score)
