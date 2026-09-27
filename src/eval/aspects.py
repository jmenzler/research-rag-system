"""Aspect-level LLM critics for citation accuracy and concept disambiguation.

Each aspect is an async function with the same signature as the core metrics
in ``src.eval.metrics``: takes a ``row`` dict (with ``user_input``, ``response``,
``retrieved_contexts`` keys) and a ``judge_model`` string, returns a float in
[0, 1] or NaN on parse failure.

Both critics reuse ``call_with_schema`` from ``src.eval.metrics`` for
concurrency-capped, retried LLM calls with provider-side schema enforcement.

Aspect scores are opt-in (toggled by CLI flags ``--with-aspect-citation`` and
``--with-aspect-disambiguation``) and are NOT included in the default eval run.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any

from src.eval.metrics import _NanSentinel, call_with_schema
from src.eval.prompts import (
    CITATION_ACCURACY_EXAMPLES,
    CITATION_ACCURACY_INSTRUCTION,
    CONCEPT_DISAMBIGUATION_EXAMPLES,
    CONCEPT_DISAMBIGUATION_INSTRUCTION,
    render_prompt,
)
from src.eval.schemas import (
    CitationAccuracyOutput,
    CitationCheckInput,
    ConceptDisambiguationInput,
    ConceptDisambiguationOutput,
)

logger = logging.getLogger(__name__)

# Matches citation markers like [1], [12], [3,4,5] in answer text.
_CITATION_RE = re.compile(r"\[(\d+(?:,\s*\d+)*)\]")


def _extract_citation_indices(answer: str) -> list[int]:
    """Return sorted unique citation indices found in the answer text."""
    indices: set[int] = set()
    for match in _CITATION_RE.finditer(answer):
        for part in match.group(1).split(","):
            try:
                n = int(part.strip())
                if n > 0:
                    indices.add(n)
            except ValueError:
                continue
    return sorted(indices)


async def citation_accuracy(row: dict[str, Any], judge_model: str) -> float:
    """Fraction of citations in the answer that correctly attribute their claims.

    Extracts citation markers [N] from the answer, builds a formatted citations
    list, then asks the LLM judge to verify each citation against the retrieved
    contexts. Returns the fraction of citations that pass (verdict=1), or NaN
    if no citations are present or the judge call fails.
    """
    question: str = row["user_input"]
    answer: str = row["response"]
    contexts: list[str] = list(row["retrieved_contexts"])

    # Build a formatted citation list from the answer + contexts.
    # We don't have structured citation data here — the aspect works with
    # whatever markers appear in the answer prose.
    citation_indices = _extract_citation_indices(answer)
    if not citation_indices:
        logger.debug(
            "citation_accuracy: no citation markers found in answer; returning NaN"
        )
        return math.nan

    # Build a minimal citations reference string.
    citations_lines: list[str] = []
    for idx in citation_indices:
        # Check if we have at least enough contexts to reference.
        ctx_idx = idx - 1  # citation [1] refers to first context
        source = f"context_{ctx_idx + 1}" if ctx_idx < len(contexts) else "unknown"
        citations_lines.append(f"[{idx}] {source}")

    citations_formatted = "\n".join(citations_lines)
    contexts_joined = "\n\n".join(contexts)

    rendered = render_prompt(
        CITATION_ACCURACY_INSTRUCTION,
        CITATION_ACCURACY_EXAMPLES,
        CitationCheckInput(
            question=question,
            answer=answer,
            citations_formatted=citations_formatted,
            contexts=contexts_joined,
        ),
    )

    try:
        out, _ = await call_with_schema(
            judge_model, "", rendered, CitationAccuracyOutput
        )
    except Exception:
        logger.warning("citation_accuracy: LLM call failed; returning NaN", exc_info=True)
        return math.nan

    if isinstance(out, _NanSentinel) or out is None:
        logger.warning("citation_accuracy: parse failure or sentinel; returning NaN")
        return math.nan

    checks = out.checks
    if not checks:
        logger.warning("citation_accuracy: no checks returned; returning NaN")
        return math.nan

    n = len(checks)
    passed = sum(1 for c in checks if c.verdict == 1)
    return passed / n


async def concept_disambiguation(row: dict[str, Any], judge_model: str) -> float:
    """Binary critic: does the answer correctly distinguish closely-related concepts?

    Returns 1.0 when the judge determines concepts are correctly distinguished,
    0.0 when they are conflated or ambiguous. NaN on parse/failure.
    The aggregate across rows is the fraction of answers that pass.
    """
    question: str = row["user_input"]
    answer: str = row["response"]
    contexts: list[str] = list(row["retrieved_contexts"])
    contexts_joined = "\n\n".join(contexts)

    rendered = render_prompt(
        CONCEPT_DISAMBIGUATION_INSTRUCTION,
        CONCEPT_DISAMBIGUATION_EXAMPLES,
        ConceptDisambiguationInput(
            question=question,
            answer=answer,
            contexts=contexts_joined,
        ),
    )

    try:
        out, _ = await call_with_schema(
            judge_model, "", rendered, ConceptDisambiguationOutput
        )
    except Exception:
        logger.warning(
            "concept_disambiguation: LLM call failed; returning NaN", exc_info=True
        )
        return math.nan

    if isinstance(out, _NanSentinel) or out is None:
        logger.warning(
            "concept_disambiguation: parse failure or sentinel; returning NaN"
        )
        return math.nan

    return float(out.verdict)
