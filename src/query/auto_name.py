"""Auto-name fallback — single cheap LLM call to generate a chat title.

Phase 2 D-22 / CHAT-10. Used only when the decompose LLM did NOT emit a
``title_hint`` (e.g., ``emit_title_hint`` flag was off, or the model returned
``title_hint=null`` for a vague query).

Architectural note: this lives in ``src/query/`` (not ``src/server/``) because
CLAUDE.md mandates every LLM call originate from ``src/query/``. The Phase 0
CI lint enforces this: direct ``genai`` / ``openai`` imports outside
``src/query/usage_track.py`` fail the build.

Cost attribution: ``auto_name_fallback`` calls ``audit.accumulate_usage`` with
``stage="decompose"`` so the fallback's cost folds into the existing decompose
stage in ``meta.totals`` (no new ``_STAGE_PREFIX`` entry). Per the plan
contract, only one of decompose-with-hint OR decompose-without-hint+fallback
fires per query, so this is the right attribution boundary.
"""
from __future__ import annotations

import logging

from src import config
from src.query import usage_track
from src.query.audit import accumulate_usage
from src.query.query_logger import QueryLogger

logger = logging.getLogger(__name__)

# Title generation prompt — kept minimal so the model emits the title alone
# (no JSON wrapper, no explanation). Mirrors the rules in
# ``prompts/router/sections/_title_hint.md`` so the fallback path produces
# the same shape as the primary path.
_AUTO_NAME_SYSTEM = (
    "You generate a concise chat title from a user's first question. "
    "Output ONLY the title text — no JSON, no quotes, no punctuation at "
    "the end. ≤60 characters target. Sentence case. Acronyms stay "
    "uppercase. Describe the topic, not the question form. Example: "
    "'GLFT optimal spread inventory penalty' (NOT 'How does GLFT work?')."
)

# D-22 hard cap mirrors the server-side truncation in router.py / chats_store.
_AUTO_NAME_HARD_CAP = 80


def auto_name_fallback(query: str, audit: QueryLogger) -> str | None:
    """Generate a chat title from the first user question via one LLM call.

    Returns the cleaned title (≤80 chars, no surrounding whitespace, no
    surrounding quotes, no trailing terminal punctuation) or ``None`` if the
    model output was empty / the call failed.

    Side effect: calls ``audit.accumulate_usage("decompose", ...)`` so the
    smoke test sees this cost in ``meta.totals.cost_usd`` (T-02-10-04
    mitigation). The ``audit`` argument is accepted for API symmetry with
    other ``src/query/`` helpers, but the recording goes through the
    module-level ``src.query.audit`` ContextVar facade so retriever
    sub-stages and this fallback land on the same logger.

    Args:
        query: The user's first message in this chat.
        audit: The active ``QueryLogger`` (kept for typed-API symmetry; the
            actual recording is routed via the contextvar in ``src.query.audit``
            because the SSE handler installs the logger there).

    Returns:
        Cleaned title string, or ``None`` if generation failed / was empty.
    """
    _ = audit  # symmetry argument; recording goes via the contextvar facade
    if not query.strip():
        return None
    try:
        # Module-attribute call so test monkeypatching of
        # `usage_track.call_text` is observed (RED test
        # test_auto_name_fallback_uses_usage_track relies on this).
        result = usage_track.call_text(
            config.DECOMPOSE_MODEL,
            _AUTO_NAME_SYSTEM,
            query.strip(),
            json_mode=False,
            temperature=0.0,
            max_tokens=64,
        )
    except Exception as exc:  # noqa: BLE001 — auto-name is best-effort
        logger.warning("auto_name_fallback LLM call failed: %r", exc)
        return None

    # Fold cost into the existing decompose stage so meta.totals.cost_usd
    # reflects this call without needing a new _STAGE_PREFIX entry.
    # Goes through the contextvar facade in src.query.audit so the SSE
    # handler's installed logger is the one that records (the `audit` arg
    # is the same logger in practice — both refer to the active one).
    accumulate_usage(
        "decompose",
        result.usage,
        result.cost_usd,
        result.latency_ms,
    )

    title = result.text.strip()
    # Strip surrounding quotes if the model couldn't resist.
    if (title.startswith('"') and title.endswith('"')) or (
        title.startswith("'") and title.endswith("'")
    ):
        title = title[1:-1].strip()
    # Strip trailing terminal punctuation (model may emit "?" or "." despite prompt).
    while title and title[-1] in ".?!":
        title = title[:-1].strip()
    # Hard cap at 80 chars (D-22).
    title = title[:_AUTO_NAME_HARD_CAP]
    return title or None
