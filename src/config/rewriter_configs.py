"""Global rewriter configuration.

Design: a single global ``RewriterConfig`` read from ``REWRITER_*`` env vars.
Used for A/B testing — flip ``REWRITER_EMIT_HYDE=true`` to enable HyDE for
ALL queries (no per-intent gating).

Contents:
- ``RewriterConfig`` frozen dataclass — parameters for the rewriter LLM call.
- ``_baseline_rewriter_config()`` — reads env knobs from ``src.config.rewriter``.
- ``_validate_rewriter_configs(overrides)`` — validator for test-injected
  override dicts (still used to guard against typo'd field names in tests).

Removed in PR #30 follow-up (HIGH #1):
- ``INTENT_REWRITER_CONFIG`` — per-intent override dict.
- ``rewriter_config_for_intent(intent)`` — resolver.

Per-intent gating was never the intended design. The single global config is
the A/B knob; the pipeline calls ``_baseline_rewriter_config()`` once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RewriterConfig:
    """Frozen config resolved at query-plan time.

    All emit_* flags default to ``False``; new output fields only appear in
    the router's JSON response when a flag is ``True``.
    """

    model: str
    emit_hyde: bool
    emit_stepback: bool
    emit_disambiguation: bool
    emit_filters: bool
    dense_max_tokens: int
    hyde_max_tokens: int
    stepback_max_tokens: int
    sub_queries_min: int
    sub_queries_max: int
    max_output_tokens: int
    # Phase 2 D-21 / CHAT-10 — when True, the router prompt includes the
    # _title_hint.md section and the LLM emits a title_hint field. Default
    # False preserves bit-identical behavior for existing call sites.
    # Listed last (with default) so positional construction still works.
    emit_title_hint: bool = False


def _baseline_rewriter_config() -> RewriterConfig:
    """Build the global ``RewriterConfig`` from env knobs.

    Reads from ``src.config.rewriter`` at call time (not at import time) so
    that monkeypatching env vars in tests takes effect.
    """
    import importlib

    # Re-import the module each call so tests that monkeypatch env vars
    # see fresh values. In production this is cheap (module is cached after
    # the first import; the attribute reads are O(1)).
    rw = importlib.import_module("src.config.rewriter")
    return RewriterConfig(
        model=rw.REWRITER_MODEL,
        emit_hyde=rw.REWRITER_EMIT_HYDE,
        emit_stepback=rw.REWRITER_EMIT_STEPBACK,
        emit_disambiguation=rw.REWRITER_EMIT_DISAMBIGUATION,
        emit_filters=rw.REWRITER_EMIT_FILTERS,
        dense_max_tokens=rw.REWRITER_DENSE_MAX_TOKENS,
        hyde_max_tokens=rw.REWRITER_HYDE_MAX_TOKENS,
        stepback_max_tokens=rw.REWRITER_STEPBACK_MAX_TOKENS,
        sub_queries_min=rw.REWRITER_SUB_QUERIES_MIN,
        sub_queries_max=rw.REWRITER_SUB_QUERIES_MAX,
        max_output_tokens=rw.REWRITER_MAX_OUTPUT_TOKENS,
        # Phase 2 D-21 — default False. The api_chats auto-name hook flips
        # this per-call via the emit_title_hint kwarg on route_query, NOT
        # via env var (the env var would emit a title for EVERY query, which
        # wastes tokens on non-first turns).
        emit_title_hint=False,
    )


# Field-name allowlist used by the validator below — kept here as the single
# source of truth for which RewriterConfig fields exist.
_BOOL_FIELDS = frozenset({
    "emit_hyde", "emit_stepback", "emit_disambiguation", "emit_filters",
    "emit_title_hint",
})
_INT_FIELDS = frozenset({
    "dense_max_tokens", "hyde_max_tokens", "stepback_max_tokens",
    "sub_queries_min", "sub_queries_max", "max_output_tokens",
})
_STR_FIELDS = frozenset({"model"})
_ALL_REWRITER_FIELDS = _BOOL_FIELDS | _INT_FIELDS | _STR_FIELDS


def _validate_rewriter_configs(
    overrides: dict[str, dict[str, Any]] | None = None,
) -> None:
    """Raise ``ValueError`` on any misconfiguration of an override dict.

    Validates:
    - All field names in each override dict must exist on ``RewriterConfig``.
    - Bool fields must receive ``bool`` values (not str, not int).

    Args:
        overrides: dict to validate (e.g. ``{"factoid": {"emit_hyde": True}}``).
            Kept for test scenarios that inject synthetic dicts. The production
            code path no longer has a per-intent override dict to validate —
            this function is now only useful for unit tests.
    """
    if overrides is None:
        return

    for intent, override_dict in overrides.items():
        for field_name, value in override_dict.items():
            if field_name not in _ALL_REWRITER_FIELDS:
                msg = (
                    f"rewriter override [{intent!r}]: unknown field {field_name!r}. "
                    f"Allowed: {sorted(_ALL_REWRITER_FIELDS)}"
                )
                raise ValueError(msg)
            if field_name in _BOOL_FIELDS and not isinstance(value, bool):
                msg = (
                    f"rewriter override [{intent!r}][{field_name!r}]: "
                    f"must be bool, got {type(value).__name__}={value!r}."
                )
                raise ValueError(msg)
