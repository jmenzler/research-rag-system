"""Async metric functions: faithfulness, answer_relevancy, context_precision,
context_recall.

Each metric mirrors RAGAS' arithmetic exactly (verified per-slice). Differences
from RAGAS:

1. LLM calls go through ``src.query.usage_track.call_text`` with
   ``schema=PydanticModel`` — provider-side enforcement (Gemini
   ``response_schema``, OpenAI ``json_schema`` ``response_format``). No
   parse-and-retry loop. A malformed response raises immediately.
2. ``answer_relevancy`` runs N=3 *parallel* LLM calls via ``asyncio.gather``
   instead of one ``generate_multiple(n=3)`` request. Variance from different
   sampling seeds vs RAGAS is expected and acceptable.
3. ``answer_relevancy`` uses our configured ``EMBEDDING_PROVIDER`` (via
   ``src.query.retrieve.embed_queries_batch``), NOT the hardcoded
   ``gemini-embedding-001`` RAGAS uses. Calibration may show some drift on
   this metric — flagged in the docstring.

Row format expected (already produced by ``src.evaluate``):
    {
        "user_input": str,
        "response": str,
        "retrieved_contexts": list[str],
        "reference": str,        # ground truth — used by precision/recall only
    }

Concurrency / robustness (added 2026-05-08 after a 503 storm):

- Total inflight LLM calls are bounded by ``_LLM_INFLIGHT`` (env var
  ``METRICS_MAX_INFLIGHT``, default 16). The runner's per-metric semaphore
  was insufficient because each metric internally fans out K parallel sub-calls
  (e.g. context_precision → one call per chunk), so the actual peak inflight
  was ``max_workers × K`` not ``max_workers``. This module-level semaphore
  bounds the true ceiling.
- Transient provider errors (503 UNAVAILABLE, RESOURCE_EXHAUSTED, network
  blips) trigger exponential backoff retry inside ``call_with_schema``.
  After ``_MAX_RETRIES`` attempts, the call returns a ``_NanSentinel`` so
  the affected question scores nan — the rest of the eval continues.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import random
from typing import Any, TypeVar

import numpy as np
from pydantic import BaseModel, ValidationError

from src.eval.prompts import (
    CONTEXT_PRECISION_EXAMPLES,
    CONTEXT_PRECISION_INSTRUCTION,
    CONTEXT_RECALL_EXAMPLES,
    CONTEXT_RECALL_INSTRUCTION,
    NLI_STATEMENT_EXAMPLES,
    NLI_STATEMENT_INSTRUCTION,
    RESPONSE_RELEVANCE_EXAMPLES,
    RESPONSE_RELEVANCE_INSTRUCTION,
    STATEMENT_GENERATOR_EXAMPLES,
    STATEMENT_GENERATOR_INSTRUCTION,
    render_prompt,
)
from src.eval.schemas import (
    QAC,
    QCA,
    ContextRecallClassifications,
    NLIStatementInput,
    NLIStatementOutput,
    ResponseRelevanceInput,
    ResponseRelevanceOutput,
    StatementGeneratorInput,
    StatementsOutput,
    Verification,
)
from src.models import Usage
from src.query.retrieve import embed_queries_batch
from src.query.usage_track import LLMCallResult, call_text

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


# ---------------------------------------------------------------------------
# Concurrency / retry config
# ---------------------------------------------------------------------------

# Module-level semaphore bounds total inflight LLM calls across ALL metrics.
# Each metric (especially context_precision) fans out K parallel sub-calls,
# so the runner-level semaphore is not enough — a single context_precision
# call alone can spawn K coroutines, blowing past per-row limits.
# Tune via env var METRICS_MAX_INFLIGHT (default 16). Created lazily so
# the limit is read at first use, not at module import time.
_LLM_MAX_INFLIGHT: int = int(os.getenv("METRICS_MAX_INFLIGHT", "16"))
_LLM_INFLIGHT: asyncio.Semaphore | None = None


def _get_inflight_sem() -> asyncio.Semaphore:
    """Lazy semaphore — must be created on the running event loop."""
    global _LLM_INFLIGHT
    if _LLM_INFLIGHT is None:
        _LLM_INFLIGHT = asyncio.Semaphore(_LLM_MAX_INFLIGHT)
    return _LLM_INFLIGHT


# Retry policy for transient provider errors (503 UNAVAILABLE, 429
# RESOURCE_EXHAUSTED, transient network errors). Schema-validation failures
# do NOT retry — they're deterministic and costly to re-attempt.
_MAX_RETRIES: int = 3
_BASE_BACKOFF_S: float = 4.0  # 4s, 16s, 64s with jitter


def _is_transient_provider_error(exc: BaseException) -> bool:
    """True for retriable errors: HTTP 5xx, 429, network blips."""
    msg = str(exc)
    # Gemini SDK: google.genai.errors.ServerError ("503 UNAVAILABLE")
    # OpenAI/DeepSeek: openai.APIStatusError with status_code attribute
    code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if isinstance(code, int) and (code in (408, 429, 500, 502, 503, 504)):
        return True
    if any(s in msg for s in ("UNAVAILABLE", "RESOURCE_EXHAUSTED", "503", "429", "502", "504")):
        return True
    # httpx ConnectError / ReadTimeout / RemoteProtocolError
    name = type(exc).__name__
    if name in ("ConnectError", "ReadTimeout", "ReadError", "RemoteProtocolError", "WriteError"):
        return True
    return False


# ---------------------------------------------------------------------------
# Parse-failure sentinel
# ---------------------------------------------------------------------------


class _NanSentinel:
    """Returned in place of a parsed Pydantic model when schema parsing fails.

    Any attribute access returns ``math.nan`` so downstream metric arithmetic
    (which always produces nan when nan enters a sum/mean) degrades gracefully
    instead of raising. The per-question score will be nan; other questions
    are unaffected.
    """

    def __getattr__(self, name: str) -> float:  # noqa: ANN401
        return math.nan

    def __bool__(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# Shared helper: schema-enforced LLM call with parse step
# ---------------------------------------------------------------------------


async def call_with_schema(
    model: str,
    system: str,
    user: str,
    schema: type[T],
) -> tuple[T, LLMCallResult]:
    """Single schema-enforced LLM call with concurrency cap + transient-error retry.

    ``call_text`` is sync; we ``asyncio.to_thread`` it so callers can fan-out
    via ``asyncio.gather``. Total inflight calls are bounded by
    ``_LLM_INFLIGHT`` (env: ``METRICS_MAX_INFLIGHT``). The returned text is
    parsed via ``schema.model_validate_json``.

    Retry policy:
    - Transient provider errors (503 UNAVAILABLE, 429, network) → retry up to
      ``_MAX_RETRIES`` times with exponential backoff + jitter, then return
      ``_NanSentinel``. The eval continues; the affected question scores nan.
    - Schema validation failures (deterministic) → log and return
      ``_NanSentinel`` immediately, no retry.
    - Any other exception bubbles up (caller is expected to be wrapped in
      ``asyncio.gather(..., return_exceptions=True)`` at the runner level).
    """
    sem = _get_inflight_sem()
    last_transient_exc: BaseException | None = None

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            async with sem:
                result = await asyncio.to_thread(
                    call_text,
                    model,
                    system,
                    user,
                    json_mode=False,
                    temperature=0.0,
                    schema=schema,
                )
            try:
                parsed = schema.model_validate_json(result.text)
            except (ValidationError, json.JSONDecodeError) as exc:
                logger.error(
                    "call_with_schema: parse failure for model=%r schema=%s — "
                    "scoring this question as nan. Error: %s. Response text: %.200r",
                    model,
                    schema.__name__,
                    exc,
                    result.text,
                )
                return _NanSentinel(), result  # type: ignore[return-value]
            return parsed, result
        except Exception as exc:  # noqa: BLE001 — re-raise unless transient
            if not _is_transient_provider_error(exc):
                raise
            last_transient_exc = exc
            if attempt >= _MAX_RETRIES:
                break
            backoff = _BASE_BACKOFF_S * (4 ** (attempt - 1))
            jitter = backoff * 0.25 * random.random()  # noqa: S311 — non-crypto
            wait_s = backoff + jitter
            logger.warning(
                "call_with_schema: transient error (attempt %d/%d) on model=%r — "
                "retrying in %.1fs. Error: %s",
                attempt,
                _MAX_RETRIES,
                model,
                wait_s,
                exc,
            )
            await asyncio.sleep(wait_s)

    logger.error(
        "call_with_schema: exhausted %d retries on model=%r schema=%s — "
        "scoring this question as nan. Last error: %s",
        _MAX_RETRIES,
        model,
        schema.__name__,
        last_transient_exc,
    )
    fallback_result = LLMCallResult(
        text="",
        latency_ms=0,
        usage=Usage(),
        raw_response=None,
        cost_usd=None,
        model=model,
    )
    return _NanSentinel(), fallback_result  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Faithfulness — 2 LLM calls (statements then NLI verdicts)
# ---------------------------------------------------------------------------


async def faithfulness(row: dict[str, Any], judge_model: str) -> float:
    """Score in [0, 1]: fraction of generated statements supported by context.

    Mirrors ``ragas.metrics._faithfulness.Faithfulness._ascore``:
    1. LLM extracts atomic statements from (question, answer) → ``StatementsOutput``.
    2. If no statements → NaN.
    3. LLM judges each statement against ``"\\n".join(retrieved_contexts)`` →
       ``NLIStatementOutput``.
    4. Score = ``sum(verdict) / len(verdicts)``.
    """
    question: str = row["user_input"]
    answer: str = row["response"]
    contexts: list[str] = list(row["retrieved_contexts"])

    # Call 1 — extract statements
    stmt_user = render_prompt(
        STATEMENT_GENERATOR_INSTRUCTION,
        STATEMENT_GENERATOR_EXAMPLES,
        StatementGeneratorInput(question=question, answer=answer),
    )
    try:
        stmt_out, _ = await call_with_schema(judge_model, "", stmt_user, StatementsOutput)
    except (ValidationError, json.JSONDecodeError) as exc:
        logger.error("faithfulness: statement extraction parse failed: %s; returning NaN", exc)
        return math.nan
    if isinstance(stmt_out, _NanSentinel):
        logger.warning("faithfulness: statement extraction parse failed; returning NaN")
        return math.nan
    if not stmt_out.statements:
        logger.warning("faithfulness: no statements generated; returning NaN")
        return math.nan

    # Call 2 — NLI verdicts
    nli_user = render_prompt(
        NLI_STATEMENT_INSTRUCTION,
        NLI_STATEMENT_EXAMPLES,
        NLIStatementInput(
            context="\n".join(contexts),
            statements=list(stmt_out.statements),
        ),
    )
    try:
        nli_out, _ = await call_with_schema(judge_model, "", nli_user, NLIStatementOutput)
    except (ValidationError, json.JSONDecodeError) as exc:
        logger.error("faithfulness: NLI parse failed: %s; returning NaN", exc)
        return math.nan
    if isinstance(nli_out, _NanSentinel):
        logger.warning("faithfulness: NLI parse failed; returning NaN")
        return math.nan

    n = len(nli_out.statements)
    if n == 0:
        logger.warning("faithfulness: NLI returned no verdicts; returning NaN")
        return math.nan
    return sum(1 if s.verdict else 0 for s in nli_out.statements) / n


# ---------------------------------------------------------------------------
# Answer relevancy — 3 parallel LLM calls + cosine similarity
# ---------------------------------------------------------------------------

_ANSWER_RELEVANCY_N = 3


async def answer_relevancy(row: dict[str, Any], judge_model: str) -> float:
    """Score in [0, 1]: cosine similarity between original question and
    questions reverse-engineered from the answer; zeroed out if all generated
    questions are flagged noncommittal.

    NOTE on embedder: this metric uses our configured ``EMBEDDING_PROVIDER``
    (via ``embed_queries_batch``), NOT the hardcoded
    ``gemini-embedding-001`` RAGAS uses. Scores may drift vs RAGAS for that
    reason — calibration is required if bit-for-bit parity matters.

    Mirrors ``ragas.metrics._answer_relevance.ResponseRelevancy._calculate_score``.
    """
    question: str = row["user_input"]
    answer: str = row["response"]

    # N parallel LLM calls — RAGAS does generate_multiple(n=3) in one request;
    # our wrapper does single completions, so we fan-out.
    rendered = render_prompt(
        RESPONSE_RELEVANCE_INSTRUCTION,
        RESPONSE_RELEVANCE_EXAMPLES,
        ResponseRelevanceInput(response=answer),
    )

    async def _one() -> ResponseRelevanceOutput | _NanSentinel:
        out, _ = await call_with_schema(
            judge_model, "", rendered, ResponseRelevanceOutput
        )
        return out

    outputs: list[ResponseRelevanceOutput | _NanSentinel] = await asyncio.gather(
        *(_one() for _ in range(_ANSWER_RELEVANCY_N))
    )

    if any(isinstance(o, _NanSentinel) for o in outputs):
        logger.warning("answer_relevancy: one or more LLM calls failed to parse; returning NaN")
        return math.nan
    # All sentinel outputs are already excluded by the isinstance check above;
    # cast to narrow types for mypy.
    from typing import cast  # noqa: PLC0415

    safe_outputs = cast("list[ResponseRelevanceOutput]", outputs)
    gen_questions = [o.question for o in safe_outputs]
    all_noncommittal = all(o.noncommittal for o in safe_outputs)
    if all(q == "" for q in gen_questions):
        logger.warning("answer_relevancy: all generated questions empty; NaN")
        return math.nan

    # Single batched embed call: original + N generated → N+1 vectors.
    vectors = await asyncio.to_thread(
        embed_queries_batch, [question, *gen_questions]
    )
    arr = np.asarray(vectors, dtype=np.float64)
    q_vec = arr[0:1]  # shape (1, D)
    g_vec = arr[1:]  # shape (N, D)

    # Cosine similarity per generated question vs original.
    norms = np.linalg.norm(g_vec, axis=1) * np.linalg.norm(q_vec, axis=1)
    cosine_sim = (g_vec @ q_vec.T).reshape(-1) / norms

    score = float(cosine_sim.mean()) * int(not all_noncommittal)
    return score


# ---------------------------------------------------------------------------
# Context precision — K parallel LLM calls + AP@k formula
# ---------------------------------------------------------------------------


async def context_precision(row: dict[str, Any], judge_model: str) -> float:
    """Score in [0, 1]: average-precision at K of the retrieved contexts.

    For each retrieved context, the LLM judges whether it is useful for
    arriving at the reference answer. AP@k is then computed over the
    binary verdicts in retrieval order.

    Mirrors ``ragas.metrics._context_precision.LLMContextPrecisionWithReference``
    with n=1 (we skip the ensembler — single sample).

    Returns NaN if no contexts were retrieved.
    """
    question: str = row["user_input"]
    contexts: list[str] = list(row["retrieved_contexts"])
    reference: str = row["reference"]

    if not contexts:
        logger.warning("context_precision: no retrieved contexts; returning NaN")
        return math.nan

    async def _verify(ctx: str) -> Verification | _NanSentinel:
        rendered = render_prompt(
            CONTEXT_PRECISION_INSTRUCTION,
            CONTEXT_PRECISION_EXAMPLES,
            QAC(question=question, context=ctx, answer=reference),
        )
        out, _ = await call_with_schema(judge_model, "", rendered, Verification)
        return out

    verifications: list[Verification | _NanSentinel] = await asyncio.gather(
        *(_verify(ctx) for ctx in contexts)
    )

    if any(isinstance(v, _NanSentinel) for v in verifications):
        logger.warning("context_precision: one or more LLM calls failed to parse; returning NaN")
        return math.nan
    verdict_list = [1 if v.verdict else 0 for v in verifications]
    denom = sum(verdict_list) + 1e-10
    numerator = sum(
        (sum(verdict_list[: i + 1]) / (i + 1)) * verdict_list[i]
        for i in range(len(verdict_list))
    )
    return numerator / denom


# ---------------------------------------------------------------------------
# Context recall — single LLM call, attribution ratio
# ---------------------------------------------------------------------------


async def context_recall(row: dict[str, Any], judge_model: str) -> float:
    """Score in [0, 1]: fraction of reference-answer sentences attributable
    to the retrieved context.

    One LLM call: the judge classifies each sentence in the reference answer
    as attributed (1) or not (0) given ``"\\n".join(retrieved_contexts)``.
    Score = attributed / total.

    Mirrors ``ragas.metrics._context_recall.LLMContextRecall._compute_score``
    with n=1 (no ensembler).

    Returns NaN if the LLM produced zero classifications.
    """
    question: str = row["user_input"]
    contexts: list[str] = list(row["retrieved_contexts"])
    reference: str = row["reference"]

    rendered = render_prompt(
        CONTEXT_RECALL_INSTRUCTION,
        CONTEXT_RECALL_EXAMPLES,
        QCA(question=question, context="\n".join(contexts), answer=reference),
    )
    out, _ = await call_with_schema(
        judge_model, "", rendered, ContextRecallClassifications
    )
    if isinstance(out, _NanSentinel):
        logger.warning("context_recall: LLM parse failed; returning NaN")
        return math.nan

    n = len(out.classifications)
    if n == 0:
        logger.warning("context_recall: LLM returned 0 classifications; NaN")
        return math.nan
    return sum(1 if c.attributed else 0 for c in out.classifications) / n
