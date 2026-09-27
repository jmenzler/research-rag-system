"""CRAG-lite: groundedness grader + query reformulator.

After synthesis, ``grade_groundedness`` asks the configured judge model to
score whether the generated answer's claims are supported by the retrieved
chunks (0.0 - 1.0). If the score falls below ``CRAG_THRESHOLD``,
``reformulate_query`` produces a new query optimized for retrieval, and the
pipeline retries once.
"""

from __future__ import annotations

import json
import logging
import re

from src import config
from src.query.decompose import _generate_text_via_provider
from src.query.usage_track import LLMCallResult

logger = logging.getLogger(__name__)

_GRADER_SYSTEM = (
    "You are a strict groundedness grader for a RAG system. Given a generated ANSWER "
    "and the retrieved CONTEXTS used to produce it, score how well the answer's "
    "substantive claims are supported by the contexts.\n\n"
    "Scoring rubric:\n"
    "- 1.0: Every claim traces cleanly to a context. No hallucinations.\n"
    "- 0.7-0.9: Most claims grounded; minor unsupported elaboration.\n"
    "- 0.4-0.6: Mixed — some claims grounded, others appear invented or extrapolated.\n"
    "- 0.0-0.3: Most claims not supported by contexts; likely hallucination.\n\n"
    "Output ONLY a JSON object: {\"score\": <float>, \"reason\": \"<one sentence>\"}\n"
    "No prose, no markdown."
)

_REFORMULATE_SYSTEM = (
    "You are a query reformulator for a RAG retrieval system. The original query was "
    "answered with low groundedness — the retriever likely missed the key chunks.\n\n"
    "Given the ORIGINAL question and the WEAK ANSWER produced from poor retrieval, "
    "produce ONE reformulated query optimized for retrieval that targets the gap. "
    "Add specific terminology, model names, or formula keywords "
    "likely present in source documents.\n\n"
    "Output ONLY the reformulated query string. No quotes, no prose, no labels."
)


def grade_groundedness(
    answer: str,
    contexts: list[str],
) -> tuple[float, str, LLMCallResult | None]:
    """Score *answer*'s groundedness in *contexts* (0.0 - 1.0).

    Returns ``(1.0, "grader unavailable", None)`` on any failure — fail-open
    so the pipeline never gets stuck in a retry loop due to grader errors.

    Returns:
        (score, reason, llm_result) — the logger uses the third element to
        record usage/latency/cost; callers that only care about score/reason
        can ignore it.
    """
    if not answer.strip() or not contexts:
        return 0.0, "empty answer or contexts", None

    contexts_block = "\n\n".join(f"[CTX {i}]\n{c[:1500]}" for i, c in enumerate(contexts, 1))
    prompt = f"ANSWER:\n{answer}\n\nCONTEXTS:\n{contexts_block}\n\nReturn the JSON now."

    try:
        result = _generate_text_via_provider(
            config.JUDGE_MODEL,
            _GRADER_SYSTEM,
            prompt,
            json_mode=True,
            temperature=0.0,
        )
        raw = result.text
    except Exception as exc:
        logger.warning("grade_groundedness: API call failed (%s); fail-open.", exc)
        return 1.0, f"grader unavailable ({exc.__class__.__name__})", None

    try:
        parsed = json.loads(raw)
        score = float(parsed.get("score", 1.0))
        reason = str(parsed.get("reason", ""))
    except (json.JSONDecodeError, ValueError, TypeError):
        match = re.search(r'"score"\s*:\s*([0-9.]+)', raw)
        if not match:
            logger.warning("grade_groundedness: malformed output %r; fail-open.", raw[:200])
            return 1.0, "grader output malformed", result
        score = float(match.group(1))
        reason = "score extracted via regex"

    score = max(0.0, min(1.0, score))
    logger.info("grade_groundedness: score=%.2f reason=%r", score, reason[:80])
    return score, reason, result


def reformulate_query(
    original_query: str,
    weak_answer: str,
) -> tuple[str | None, LLMCallResult | None]:
    """Produce a retrieval-optimized rewrite given the failed first attempt.

    Returns (None, None) on any failure (caller should skip the retry).

    Returns:
        (reformulated_query_text, llm_result) — the logger uses the second
        element to record usage/latency/cost.
    """
    prompt = f"ORIGINAL: {original_query}\n\nWEAK ANSWER:\n{weak_answer[:2000]}"

    try:
        result = _generate_text_via_provider(
            config.DECOMPOSE_MODEL,
            _REFORMULATE_SYSTEM,
            prompt,
            json_mode=False,
            temperature=0.2,
        )
    except Exception as exc:
        logger.warning("reformulate_query: API call failed (%s); skipping.", exc)
        return None, None

    text = result.text.strip().strip('"').strip("'")
    if not text or len(text) < 8:
        return None, result
    logger.info("reformulate_query: %r → %r", original_query[:60], text[:80])
    return text, result
