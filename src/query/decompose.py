"""Query decomposition: split a multi-part research question into sub-queries.

Pattern: an LLM call (Gemini Flash) reads the user query and returns 2-N atomic
sub-questions covering distinct facets. Each sub-query is then retrieved
separately and the union of chunks is passed to the synthesis generator.

Per the deep-research recommendation, this targets +15-25% accuracy on multi-hop
synthesis questions ("how does X interact with Y under Z?") at the cost of one
extra Flash call (~1s, ~$0.0001).

Usage::

    from src.query.decompose import decompose_query
    subs = decompose_query("How does X relate to Y in context Z?")
    # → ["What is X?", "What is Y in context Z?", "How does X affect Y?"]
"""

from __future__ import annotations

import json
import logging
import re

from pydantic import BaseModel

from src import config
from src.query.usage_track import LLMCallResult, call_text

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You are a query-decomposition agent for a RAG retrieval pipeline. "
    "Given a multi-part or compound research question, return 2-N atomic sub-questions "
    "that, when answered together, fully answer the original.\n\n"
    "RULES:\n"
    "- Output ONLY a JSON array of strings. No prose, no markdown, no commentary.\n"
    "- Each sub-question must be self-contained and retrievable independently.\n"
    "- Cover distinct facets: definitions, mechanisms, comparisons, edge cases, math.\n"
    "- For comparison questions (\"X vs Y\"), produce a sub-question for each side AND "
    "  one for the comparison axis itself.\n"
    "- For interaction questions (\"how does X affect Y under Z?\"), produce sub-questions "
    "  for X, Y, Z, and the interaction.\n"
    "- If the original is already atomic (single-fact lookup), return [original_query].\n"
    f"- Cap output at {config.DECOMPOSE_MAX_SUBQUERIES} sub-questions.\n\n"
    "EXAMPLES:\n"
    'Input: "What is LVR?"\n'
    'Output: ["What is LVR?"]\n\n'
    'Input: "How does volatility estimation (EWMA, GARCH, HAR-RV) influence A-S spread params?"\n'
    'Output: ["What is the Avellaneda-Stoikov spread formula and the role of volatility sigma?", '
    '"How does EWMA estimate volatility and what are its lag properties?", '
    '"How does GARCH estimate volatility and capture clustering?", '
    '"How does HAR-RV estimate realized variance with daily/weekly/monthly components?"]\n'
)

_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


def _generate_text_via_provider(
    model: str,
    system_prompt: str,
    user_prompt: str,
    *,
    json_mode: bool,
    temperature: float,
    schema: type[BaseModel] | None = None,
) -> LLMCallResult:
    """Single-shot text generation across gemini / deepseek / openrouter.

    Returns a ``LLMCallResult`` so callers can log usage/latency/cost.
    Use ``.text`` to get the bare string (backward-compatible with old callers).

    When ``schema`` is provided, the call uses provider-side JSON schema
    enforcement (Gemini ``response_schema`` or OpenAI ``json_schema``
    response_format). The caller still parses the returned text via
    ``schema.model_validate_json``.
    """
    return call_text(
        model,
        system_prompt,
        user_prompt,
        json_mode=json_mode,
        temperature=temperature,
        schema=schema,
    )


def decompose_query(
    query: str,
) -> tuple[list[str], LLMCallResult | None]:
    """Split *query* into atomic sub-questions.

    Routes to gemini / deepseek / openrouter based on ``config.DECOMPOSE_MODEL``.
    Returns ``([query], None)`` if decomposition fails or yields nothing usable.
    Never raises — degrades to single-query retrieval on any error.

    Returns:
        (sub_queries, llm_result) — callers that only need the sub-queries can
        ignore the second element; the logger uses it to record usage/latency/cost.
    """
    # We don't pass response_format=json_object: DeepSeek/OpenRouter strict-JSON
    # mode requires an object, but _SYSTEM specifies a JSON array. The regex
    # fallback below copes with any leading/trailing prose.
    try:
        result = _generate_text_via_provider(
            config.DECOMPOSE_MODEL,
            _SYSTEM,
            f"Question: {query}",
            json_mode=False,
            temperature=0.0,
        )
        raw = result.text
    except Exception as exc:
        logger.warning("decompose_query: API call failed (%s); using original query.", exc)
        return [query], None

    # Try strict JSON parse first; fall back to regex extraction.
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        match = _JSON_ARRAY_RE.search(raw)
        if not match:
            logger.warning(
                "decompose_query: no JSON array in response %r; using original.",
                raw[:200],
            )
            return [query], result
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            logger.warning("decompose_query: JSON parse failed (%s); using original.", exc)
            return [query], result

    if not isinstance(parsed, list) or not all(isinstance(s, str) for s in parsed):
        logger.warning("decompose_query: malformed output %r; using original.", parsed)
        return [query], result

    cleaned = [s.strip() for s in parsed if s.strip()]
    if not cleaned:
        return [query], result

    capped = cleaned[: config.DECOMPOSE_MAX_SUBQUERIES]
    logger.info("decompose_query: %d sub-queries from %r", len(capped), query[:80])
    return capped, result


_STEPBACK_SYSTEM = (
    "You are a step-back query reformulator. Given a specific research question, "
    "produce ONE more abstract, foundational question that captures the underlying "
    "concept the original is asking about. The step-back query retrieves textbook / "
    "survey material that grounds the specific answer.\n\n"
    "Output ONLY the abstract question. No prose, no quotes, no labels.\n\n"
    "EXAMPLES:\n"
    'Input: "How do you calibrate GLFT κ when CLMM fills are deterministic, not Poisson?"\n'
    'Output: What is the GLFT model and what role does the fill-intensity parameter κ play?\n\n'
    'Input: "EWMA λ=0.94 vs GARCH/HAR-RV for sub-minute basis vol"\n'
    'Output: What are the standard volatility estimation methods and their assumptions?\n\n'
    'Input: "What is VPIN?"\n'
    'Output: What is order-flow toxicity and why does it matter for market making?\n'
)


def step_back_query(
    query: str,
) -> tuple[str | None, LLMCallResult | None]:
    """Generate one abstract foundational question for *query*.

    Routes to gemini / deepseek / openrouter based on ``config.DECOMPOSE_MODEL``.
    Returns (None, None) on any failure (caller should treat as "no step-back").

    Returns:
        (stepback_query_text, llm_result) — the logger uses the second element
        to record usage/latency/cost; callers that only care about the text
        can ignore it.
    """
    try:
        result = _generate_text_via_provider(
            config.DECOMPOSE_MODEL,
            _STEPBACK_SYSTEM,
            f"Question: {query}",
            json_mode=False,
            temperature=0.0,
        )
    except Exception as exc:
        logger.warning("step_back_query: API call failed (%s); skipping step-back.", exc)
        return None, None

    text = result.text.strip().strip('"').strip("'")
    if not text or len(text) < 8:
        return None, result
    logger.info("step_back_query: %r → %r", query[:60], text[:80])
    return text, result
