"""Tests for src/eval/* — schemas, prompts, ensembler, metrics.

Slice-by-slice TDD harness. Each test class corresponds to one slice.

Slice 2: schemas.py + prompts.py — Pydantic models + render_prompt
Slice 3: ensembler.py — majority_vote_discrete
Slice 4: metrics.faithfulness
Slice 5: metrics.answer_relevancy
Slice 6: metrics.context_precision
Slice 7: metrics.context_recall

The metrics tests mock `call_with_schema` (so no real LLM calls) and
`embed_queries_batch` (so no real embed calls). The whole point of this
module is deterministic, fast, schema-enforced metric computation — the
tests reflect that.
"""

from __future__ import annotations

import asyncio
import math
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel, ValidationError

if TYPE_CHECKING:
    from src.eval.schemas import NLIStatementOutput, StatementsOutput

# ---------------------------------------------------------------------------
# Slice 2 — Pydantic schemas
# ---------------------------------------------------------------------------


class TestStatementGenerator:
    def test_input_valid(self) -> None:
        from src.eval.schemas import StatementGeneratorInput

        m = StatementGeneratorInput(question="Who?", answer="Einstein.")
        assert m.question == "Who?"
        assert m.answer == "Einstein."

    def test_input_invalid_missing_field(self) -> None:
        from src.eval.schemas import StatementGeneratorInput

        with pytest.raises(ValidationError):
            StatementGeneratorInput(question="Who?")  # type: ignore[call-arg]

    def test_output_valid(self) -> None:
        from src.eval.schemas import StatementsOutput

        m = StatementsOutput(statements=["a", "b"])
        assert m.statements == ["a", "b"]

    def test_output_round_trip_json(self) -> None:
        from src.eval.schemas import StatementsOutput

        m = StatementsOutput(statements=["a", "b"])
        roundtrip = StatementsOutput.model_validate_json(m.model_dump_json())
        assert roundtrip == m


class TestNLIStatement:
    def test_full_round_trip(self) -> None:
        from src.eval.schemas import (
            NLIStatementInput,
            NLIStatementOutput,
            StatementFaithfulnessAnswer,
        )

        inp = NLIStatementInput(context="ctx", statements=["a", "b"])
        assert inp.context == "ctx"

        out = NLIStatementOutput(
            statements=[
                StatementFaithfulnessAnswer(statement="a", reason="r1", verdict=1),
                StatementFaithfulnessAnswer(statement="b", reason="r2", verdict=0),
            ]
        )
        roundtrip = NLIStatementOutput.model_validate_json(out.model_dump_json())
        assert roundtrip == out
        assert roundtrip.statements[0].verdict == 1


class TestResponseRelevance:
    def test_round_trip(self) -> None:
        from src.eval.schemas import (
            ResponseRelevanceInput,
            ResponseRelevanceOutput,
        )

        inp = ResponseRelevanceInput(response="Einstein was born in Germany.")
        out = ResponseRelevanceOutput(question="Where was Einstein born?", noncommittal=0)
        rt_in = ResponseRelevanceInput.model_validate_json(inp.model_dump_json())
        rt_out = ResponseRelevanceOutput.model_validate_json(out.model_dump_json())
        assert rt_in == inp
        assert rt_out == out


class TestContextPrecisionVerification:
    def test_round_trip(self) -> None:
        from src.eval.schemas import QAC, Verification

        qac = QAC(question="q", context="c", answer="a")
        v = Verification(reason="useful", verdict=1)
        rt = Verification.model_validate_json(v.model_dump_json())
        assert qac.question == "q"
        assert rt == v


class TestContextRecallClassification:
    def test_round_trip(self) -> None:
        from src.eval.schemas import (
            QCA,
            ContextRecallClassification,
            ContextRecallClassifications,
        )

        qca = QCA(question="q", context="c", answer="a")
        c = ContextRecallClassifications(
            classifications=[ContextRecallClassification(statement="s1", reason="r", attributed=1)]
        )
        rt = ContextRecallClassifications.model_validate_json(c.model_dump_json())
        assert qca.context == "c"
        assert rt == c


class TestPromptRenderer:
    def test_render_includes_instruction_examples_and_input(self) -> None:
        from src.eval.prompts import render_prompt

        class _In(BaseModel):
            x: str

        class _Out(BaseModel):
            y: int

        rendered = render_prompt(
            instruction="Do the thing.",
            examples=[(_In(x="ex1"), _Out(y=1)), (_In(x="ex2"), _Out(y=2))],
            actual_input=_In(x="actual"),
        )
        assert "Do the thing." in rendered
        assert '{"x":"ex1"}' in rendered
        assert '{"y":1}' in rendered
        assert '{"x":"ex2"}' in rendered
        assert '{"y":2}' in rendered
        assert '{"x":"actual"}' in rendered
        assert rendered.rstrip().endswith("Output:")

    def test_render_no_examples(self) -> None:
        from src.eval.prompts import render_prompt

        class _In(BaseModel):
            x: str

        rendered = render_prompt(
            instruction="Do the thing.",
            examples=[],
            actual_input=_In(x="actual"),
        )
        assert "Do the thing." in rendered
        assert '{"x":"actual"}' in rendered


# RAGAS prompt-parity tests removed alongside the ragas dependency.
# Original verification: our STATEMENT_GENERATOR_INSTRUCTION,
# NLI_STATEMENT_INSTRUCTION, RESPONSE_RELEVANCE_INSTRUCTION,
# CONTEXT_PRECISION_INSTRUCTION, CONTEXT_RECALL_INSTRUCTION were verbatim
# copies of the corresponding RAGAS prompts at port time. Source provenance
# is documented in src/eval/prompts.py inline comments.


# ---------------------------------------------------------------------------
# Slice 3 — Ensembler
# ---------------------------------------------------------------------------


class TestEnsembler:
    def test_majority_vote_simple_case(self) -> None:
        from src.eval.ensembler import majority_vote_discrete

        # Two samples, two items each. Item 0: verdicts (1,0) → tie → first encountered (1)
        # Wait — RAGAS uses Counter.most_common which for ties returns insertion order.
        # In Python 3.7+ Counter preserves insertion. (1,0) → Counter[1]=1, Counter[0]=1
        # most_common → [(1,1),(0,1)] → 1 wins. So plan example was wrong; let's test
        # a clear majority.
        result = majority_vote_discrete(
            [
                [{"verdict": 1, "extra": "a"}, {"verdict": 0, "extra": "b"}],
                [{"verdict": 1, "extra": "a"}, {"verdict": 0, "extra": "b"}],
                [{"verdict": 0, "extra": "a"}, {"verdict": 0, "extra": "b"}],
            ],
            "verdict",
        )
        assert result == [
            {"verdict": 1, "extra": "a"},
            {"verdict": 0, "extra": "b"},
        ]

    def test_empty_returns_empty_list(self) -> None:
        from src.eval.ensembler import majority_vote_discrete

        assert majority_vote_discrete([], "verdict") == []

    def test_single_sample_returned_as_is(self) -> None:
        from src.eval.ensembler import majority_vote_discrete

        only = [{"verdict": 1}, {"verdict": 0}]
        assert majority_vote_discrete([only], "verdict") == only

    def test_unanimous(self) -> None:
        from src.eval.ensembler import majority_vote_discrete

        result = majority_vote_discrete(
            [[{"verdict": 1}], [{"verdict": 1}], [{"verdict": 1}]], "verdict"
        )
        assert result == [{"verdict": 1}]

    def test_tie_picks_first_encountered(self) -> None:
        """RAGAS uses Counter.most_common which preserves insertion order on
        ties (Python 3.7+ Counter preserves insertion). The first verdict
        encountered for an item wins."""
        from src.eval.ensembler import majority_vote_discrete

        # Item 0: verdicts (0,1) — tied. Counter sees 0 first → 0 wins.
        result = majority_vote_discrete(
            [[{"verdict": 0}], [{"verdict": 1}]],
            "verdict",
        )
        assert result == [{"verdict": 0}]

    def test_missing_attribute_returns_first_sample(self) -> None:
        from src.eval.ensembler import majority_vote_discrete

        with pytest.warns(UserWarning):
            result = majority_vote_discrete([[{"other": 1}], [{"verdict": 0}]], "verdict")
        assert result == [{"other": 1}]

    def test_does_not_mutate_input(self) -> None:
        """We copy dicts before mutating — caller's data must be untouched."""
        from src.eval.ensembler import majority_vote_discrete

        sample = [{"verdict": 1, "reason": "a"}]
        inputs = [sample, [{"verdict": 0, "reason": "b"}]]
        majority_vote_discrete(inputs, "verdict")
        assert sample == [{"verdict": 1, "reason": "a"}]


# ---------------------------------------------------------------------------
# Slice 4 — Faithfulness
# ---------------------------------------------------------------------------


class TestFaithfulness:
    def _make_call_with_schema_mock(
        self,
        statements_output: StatementsOutput,
        nli_output: NLIStatementOutput,
    ) -> AsyncMock:
        """Mock call_with_schema that returns canned outputs in order:
        first call → statements, second call → NLI verdicts."""

        call_log: list[type] = []

        async def fake(model: str, system: str, user: str, schema: type) -> tuple[Any, Any]:
            call_log.append(schema)
            if len(call_log) == 1:
                return statements_output, MagicMock()
            return nli_output, MagicMock()

        return AsyncMock(side_effect=fake)

    def test_score_all_one_verdicts(self) -> None:
        from src.eval.metrics import faithfulness
        from src.eval.schemas import (
            NLIStatementOutput,
            StatementFaithfulnessAnswer,
            StatementsOutput,
        )

        stmt_out = StatementsOutput(statements=["s1", "s2"])
        nli_out = NLIStatementOutput(
            statements=[
                StatementFaithfulnessAnswer(statement="s1", reason="r", verdict=1),
                StatementFaithfulnessAnswer(statement="s2", reason="r", verdict=1),
            ]
        )
        mock = self._make_call_with_schema_mock(stmt_out, nli_out)
        with patch("src.eval.metrics.call_with_schema", mock):
            score = asyncio.run(
                faithfulness(
                    {
                        "user_input": "Q?",
                        "response": "A.",
                        "retrieved_contexts": ["c1", "c2"],
                    },
                    judge_model="gemini-3-flash-preview",
                )
            )
        assert score == 1.0

    def test_score_all_zero_verdicts(self) -> None:
        from src.eval.metrics import faithfulness
        from src.eval.schemas import (
            NLIStatementOutput,
            StatementFaithfulnessAnswer,
            StatementsOutput,
        )

        stmt_out = StatementsOutput(statements=["s1", "s2"])
        nli_out = NLIStatementOutput(
            statements=[
                StatementFaithfulnessAnswer(statement="s1", reason="r", verdict=0),
                StatementFaithfulnessAnswer(statement="s2", reason="r", verdict=0),
            ]
        )
        mock = self._make_call_with_schema_mock(stmt_out, nli_out)
        with patch("src.eval.metrics.call_with_schema", mock):
            score = asyncio.run(
                faithfulness(
                    {
                        "user_input": "Q?",
                        "response": "A.",
                        "retrieved_contexts": ["c1"],
                    },
                    judge_model="gemini-3-flash-preview",
                )
            )
        assert score == 0.0

    def test_score_mixed_verdicts(self) -> None:
        from src.eval.metrics import faithfulness
        from src.eval.schemas import (
            NLIStatementOutput,
            StatementFaithfulnessAnswer,
            StatementsOutput,
        )

        stmt_out = StatementsOutput(statements=["s1", "s2", "s3", "s4"])
        nli_out = NLIStatementOutput(
            statements=[
                StatementFaithfulnessAnswer(statement="s1", reason="", verdict=1),
                StatementFaithfulnessAnswer(statement="s2", reason="", verdict=0),
                StatementFaithfulnessAnswer(statement="s3", reason="", verdict=1),
                StatementFaithfulnessAnswer(statement="s4", reason="", verdict=0),
            ]
        )
        mock = self._make_call_with_schema_mock(stmt_out, nli_out)
        with patch("src.eval.metrics.call_with_schema", mock):
            score = asyncio.run(
                faithfulness(
                    {
                        "user_input": "Q?",
                        "response": "A.",
                        "retrieved_contexts": ["c1"],
                    },
                    judge_model="gemini-3-flash-preview",
                )
            )
        assert score == 0.5

    def test_score_zero_statements_is_nan(self) -> None:
        from src.eval.metrics import faithfulness
        from src.eval.schemas import StatementsOutput

        stmt_out = StatementsOutput(statements=[])
        # NLI mock not used (we should short-circuit), but provide it anyway
        nli_out = MagicMock()
        mock = self._make_call_with_schema_mock(stmt_out, nli_out)
        with patch("src.eval.metrics.call_with_schema", mock):
            score = asyncio.run(
                faithfulness(
                    {
                        "user_input": "Q?",
                        "response": "A.",
                        "retrieved_contexts": ["c1"],
                    },
                    judge_model="gemini-3-flash-preview",
                )
            )
        assert math.isnan(score)


# ---------------------------------------------------------------------------
# Slice 5 — Answer Relevancy
# ---------------------------------------------------------------------------


class TestAnswerRelevancy:
    def test_all_questions_match_original_score_one(self) -> None:
        """If 3 generated questions are identical to the original (in vector
        space), cosine_sim.mean() = 1.0 and noncommittal=0 → score 1.0."""
        from src.eval.metrics import answer_relevancy
        from src.eval.schemas import ResponseRelevanceOutput

        outputs = [
            ResponseRelevanceOutput(question="What is X?", noncommittal=0),
            ResponseRelevanceOutput(question="What is X?", noncommittal=0),
            ResponseRelevanceOutput(question="What is X?", noncommittal=0),
        ]

        async def fake_call(model: str, system: str, user: str, schema: type) -> tuple[Any, Any]:
            return outputs.pop(0), MagicMock()

        # Identical embedding vectors for all 4 → cosine = 1.0
        fake_vec = [1.0, 0.0, 0.0]
        embed_returns = [fake_vec, fake_vec, fake_vec, fake_vec]

        with patch("src.eval.metrics.call_with_schema", side_effect=fake_call):
            with patch(
                "src.eval.metrics.embed_queries_batch",
                return_value=embed_returns,
            ):
                score = asyncio.run(
                    answer_relevancy(
                        {"user_input": "What is X?", "response": "X is foo."},
                        judge_model="gemini-3-flash-preview",
                    )
                )
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_orthogonal_vectors_score_zero(self) -> None:
        """Generated questions orthogonal to original → cosine 0 → score 0."""
        from src.eval.metrics import answer_relevancy
        from src.eval.schemas import ResponseRelevanceOutput

        outputs = [
            ResponseRelevanceOutput(question="?1", noncommittal=0),
            ResponseRelevanceOutput(question="?2", noncommittal=0),
            ResponseRelevanceOutput(question="?3", noncommittal=0),
        ]

        async def fake_call(model: str, system: str, user: str, schema: type) -> tuple[Any, Any]:
            return outputs.pop(0), MagicMock()

        # First vec orthogonal to others
        embed_returns = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ]

        with patch("src.eval.metrics.call_with_schema", side_effect=fake_call):
            with patch(
                "src.eval.metrics.embed_queries_batch",
                return_value=embed_returns,
            ):
                score = asyncio.run(
                    answer_relevancy(
                        {"user_input": "X?", "response": "Y."},
                        judge_model="gemini-3-flash-preview",
                    )
                )
        assert score == pytest.approx(0.0, abs=1e-6)

    def test_all_noncommittal_zeros_score(self) -> None:
        from src.eval.metrics import answer_relevancy
        from src.eval.schemas import ResponseRelevanceOutput

        outputs = [
            ResponseRelevanceOutput(question="q", noncommittal=1),
            ResponseRelevanceOutput(question="q", noncommittal=1),
            ResponseRelevanceOutput(question="q", noncommittal=1),
        ]

        async def fake_call(model: str, system: str, user: str, schema: type) -> tuple[Any, Any]:
            return outputs.pop(0), MagicMock()

        embed_returns = [[1.0, 0.0]] * 4

        with patch("src.eval.metrics.call_with_schema", side_effect=fake_call):
            with patch(
                "src.eval.metrics.embed_queries_batch",
                return_value=embed_returns,
            ):
                score = asyncio.run(
                    answer_relevancy(
                        {"user_input": "X?", "response": "I don't know."},
                        judge_model="gemini-3-flash-preview",
                    )
                )
        assert score == 0.0


# ---------------------------------------------------------------------------
# Slice 6 — Context Precision
# ---------------------------------------------------------------------------


class TestContextPrecision:
    def test_all_relevant_score_one(self) -> None:
        from src.eval.metrics import context_precision
        from src.eval.schemas import Verification

        verdicts = [
            Verification(reason="r", verdict=1),
            Verification(reason="r", verdict=1),
            Verification(reason="r", verdict=1),
        ]

        async def fake_call(model: str, system: str, user: str, schema: type) -> tuple[Any, Any]:
            return verdicts.pop(0), MagicMock()

        with patch("src.eval.metrics.call_with_schema", side_effect=fake_call):
            score = asyncio.run(
                context_precision(
                    {
                        "user_input": "Q?",
                        "response": "A.",
                        "retrieved_contexts": ["c1", "c2", "c3"],
                        "reference": "ref",
                    },
                    judge_model="gemini-3-flash-preview",
                )
            )
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_all_zero_score_zero(self) -> None:
        from src.eval.metrics import context_precision
        from src.eval.schemas import Verification

        verdicts = [Verification(reason="r", verdict=0) for _ in range(3)]

        async def fake_call(model: str, system: str, user: str, schema: type) -> tuple[Any, Any]:
            return verdicts.pop(0), MagicMock()

        with patch("src.eval.metrics.call_with_schema", side_effect=fake_call):
            score = asyncio.run(
                context_precision(
                    {
                        "user_input": "Q?",
                        "response": "A.",
                        "retrieved_contexts": ["c1", "c2", "c3"],
                        "reference": "ref",
                    },
                    judge_model="gemini-3-flash-preview",
                )
            )
        assert score == pytest.approx(0.0, abs=1e-6)

    def test_realistic_ap_at_k(self) -> None:
        """Verdicts [1,0,1,0]: AP = (1/1)*1 + (1/2)*0 + (2/3)*1 + (2/4)*0 = 1.667
        denominator = 2 + 1e-10 → score ≈ 0.8333"""
        from src.eval.metrics import context_precision
        from src.eval.schemas import Verification

        verdicts = [
            Verification(reason="r", verdict=1),
            Verification(reason="r", verdict=0),
            Verification(reason="r", verdict=1),
            Verification(reason="r", verdict=0),
        ]

        async def fake_call(model: str, system: str, user: str, schema: type) -> tuple[Any, Any]:
            return verdicts.pop(0), MagicMock()

        expected = ((1.0 / 1) * 1 + (1.0 / 2) * 0 + (2.0 / 3) * 1 + (2.0 / 4) * 0) / (2 + 1e-10)
        with patch("src.eval.metrics.call_with_schema", side_effect=fake_call):
            score = asyncio.run(
                context_precision(
                    {
                        "user_input": "Q?",
                        "response": "A.",
                        "retrieved_contexts": ["c1", "c2", "c3", "c4"],
                        "reference": "ref",
                    },
                    judge_model="gemini-3-flash-preview",
                )
            )
        assert score == pytest.approx(expected, abs=1e-6)


# ---------------------------------------------------------------------------
# Slice 7 — Context Recall
# ---------------------------------------------------------------------------


class TestContextRecall:
    def test_all_attributed(self) -> None:
        from src.eval.metrics import context_recall
        from src.eval.schemas import (
            ContextRecallClassification,
            ContextRecallClassifications,
        )

        out = ContextRecallClassifications(
            classifications=[
                ContextRecallClassification(statement="s1", reason="", attributed=1),
                ContextRecallClassification(statement="s2", reason="", attributed=1),
            ]
        )

        async def fake_call(model: str, system: str, user: str, schema: type) -> tuple[Any, Any]:
            return out, MagicMock()

        with patch("src.eval.metrics.call_with_schema", side_effect=fake_call):
            score = asyncio.run(
                context_recall(
                    {
                        "user_input": "Q?",
                        "response": "A.",
                        "retrieved_contexts": ["c"],
                        "reference": "ref",
                    },
                    judge_model="gemini-3-flash-preview",
                )
            )
        assert score == 1.0

    def test_mixed(self) -> None:
        from src.eval.metrics import context_recall
        from src.eval.schemas import (
            ContextRecallClassification,
            ContextRecallClassifications,
        )

        out = ContextRecallClassifications(
            classifications=[
                ContextRecallClassification(statement="s1", reason="", attributed=1),
                ContextRecallClassification(statement="s2", reason="", attributed=0),
                ContextRecallClassification(statement="s3", reason="", attributed=1),
                ContextRecallClassification(statement="s4", reason="", attributed=0),
            ]
        )

        async def fake_call(model: str, system: str, user: str, schema: type) -> tuple[Any, Any]:
            return out, MagicMock()

        with patch("src.eval.metrics.call_with_schema", side_effect=fake_call):
            score = asyncio.run(
                context_recall(
                    {
                        "user_input": "Q?",
                        "response": "A.",
                        "retrieved_contexts": ["c"],
                        "reference": "ref",
                    },
                    judge_model="gemini-3-flash-preview",
                )
            )
        assert score == 0.5

    def test_empty_classifications_nan(self) -> None:
        from src.eval.metrics import context_recall
        from src.eval.schemas import ContextRecallClassifications

        out = ContextRecallClassifications(classifications=[])

        async def fake_call(model: str, system: str, user: str, schema: type) -> tuple[Any, Any]:
            return out, MagicMock()

        with patch("src.eval.metrics.call_with_schema", side_effect=fake_call):
            score = asyncio.run(
                context_recall(
                    {
                        "user_input": "Q?",
                        "response": "A.",
                        "retrieved_contexts": ["c"],
                        "reference": "ref",
                    },
                    judge_model="gemini-3-flash-preview",
                )
            )
        assert math.isnan(score)
