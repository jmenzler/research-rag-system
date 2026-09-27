"""Per-intent retrieval policy data.

Two dicts consumed by ``src.query.policies.policy_for_intent``:

- ``INTENT_MULTIPLIERS`` — scale numeric baseline fields per intent. Stays in
  lock-step with baseline tuning (e.g. ``RERANK_TOP_K`` 20 → 30) so that a
  ×2 multi_hop policy resolves to 60 instead of being pinned at 40.
- ``INTENT_OVERRIDES`` — absolute, non-numeric overrides (e.g. ``use_mmr=False``)
  or pinned values that must ignore baseline drift. Applied AFTER multipliers.

Resolution order in ``policy_for_intent`` (lowest → highest priority):
    baseline  →  multipliers  →  absolute overrides  →  llm_use_mmr

Both dicts are validated at module import (see ``_validate_policy_data`` at the
bottom of this file). Misconfiguration — unknown field, multiplier on a bool
field, non-numeric multiplier value — raises ``ValueError`` immediately so the
process fails fast rather than silently no-op'ing every query.

Empirical basis (2026-05-09 prototype on Corsi paper, Q4 multi-hop):
    ``top_k_rerank=40`` (RRF pool 60→120) lets table-summary children land in
    cross-encoder input pool (6/8 ground-truth needles found vs 2/8 at
    baseline=20). Q1/Q2 factoid queries don't need the wider pool — single-doc
    recall, larger pool only adds noise + cost. Hence intent-class gating.
"""
from __future__ import annotations

# Per-intent numeric multipliers applied to the baseline policy. Each inner
# dict maps a numeric ``RetrievalPolicy`` field name to a scaling factor.
# Integer fields are rounded; float fields keep precision.
#
# Only ``top_k_rerank`` has empirical support (2026-05-09). Other numeric
# fields stay at baseline (×1.0, omitted) until eval evidence motivates them.
INTENT_MULTIPLIERS: dict[str, dict[str, float]] = {
    # 4 baseline intents — single-doc lookups, baseline (×1.0) is correct.
    # Intentionally empty: RetrievalPolicy stays identical to _baseline_policy().
    "factoid": {},
    "definitional": {},
    "computational": {},
    "unknown": {},
    # Multi-hop / comparative — wider CE input pool reaches indirect-evidence
    # chunks (e.g. table-summary children) that single-pool retrieval misses.
    "multi_hop": {"top_k_rerank": 2.0},
    "comparative": {"top_k_rerank": 2.0},
    # Synthesis intents — even broader survey of the candidate space; the
    # answer often lives across many sub-arguments rather than one passage.
    "literature_synthesis": {"top_k_rerank": 3.0},
    "sensemaking": {"top_k_rerank": 3.0},
}

# Absolute per-intent overrides. Applied AFTER multipliers, so any field
# present here wins regardless of the multiplier outcome. Use for non-numeric
# fields (e.g. ``use_mmr=False``) or for pinned numeric values that must
# ignore baseline drift.
#
# Currently empty — 2026-05-08 ablation falsified the static
# multi_hop/comparative MMR-off override (CR 0.572 → 0.505). Per-query
# ``use_mmr`` decisions now come from the LLM router (see
# ``RoutingPlan.use_mmr``); intent-level absolute overrides stay empty.
INTENT_OVERRIDES: dict[str, dict[str, object]] = {
    "factoid": {},
    "definitional": {},
    "multi_hop": {},
    "comparative": {},
    "literature_synthesis": {},
    "sensemaking": {},
    "computational": {},
    "unknown": {},
}


# --- Module-import validation (fail fast on misconfig) ---------------------
#
# Both dicts are static, hand-edited, and small. A typo (unknown field name,
# multiplier targeting a bool field, non-numeric multiplier value) would
# silently no-op the per-query resolver and only surface in logs. Per the
# project rule "Fail loudly", we validate at import time and raise
# ``ValueError`` so a misconfigured commit fails CI, not production.
#
# Field-type allowlist mirrors ``RetrievalPolicy``. Kept in this module (not
# imported from policies.py) to avoid a circular dependency: policies.py
# already imports this module.
_NUMERIC_INT_FIELDS = frozenset(
    {"top_k_retrieve", "top_k_rerank", "mmr_pool_mult", "max_children_per_parent"}
)
_NUMERIC_FLOAT_FIELDS = frozenset({"mmr_lambda", "score_threshold"})
_BOOL_FIELDS = frozenset({"use_mmr"})
_ALL_FIELDS = _NUMERIC_INT_FIELDS | _NUMERIC_FLOAT_FIELDS | _BOOL_FIELDS


def _validate_policy_data() -> None:
    """Raise ``ValueError`` on any misconfiguration of the two dicts.

    Validation rules:
      - All keys must be known intents (the 8 IntentClass values).
      - Multipliers may only target int or float fields (never bool).
      - Multiplier values must be ``int`` or ``float`` (no bool, no str).
      - Override field names must exist on ``RetrievalPolicy``.
    """
    expected_intents = set(INTENT_MULTIPLIERS.keys())
    if set(INTENT_OVERRIDES.keys()) != expected_intents:
        msg = (
            "INTENT_OVERRIDES keys do not match INTENT_MULTIPLIERS keys. "
            f"Diff: {set(INTENT_OVERRIDES.keys()) ^ expected_intents}"
        )
        raise ValueError(msg)

    for intent, mults in INTENT_MULTIPLIERS.items():
        for field, mult in mults.items():
            if field not in _ALL_FIELDS:
                msg = (
                    f"INTENT_MULTIPLIERS[{intent!r}]: unknown field {field!r}. "
                    f"Allowed: {sorted(_ALL_FIELDS)}"
                )
                raise ValueError(msg)
            if field in _BOOL_FIELDS:
                msg = (
                    f"INTENT_MULTIPLIERS[{intent!r}][{field!r}]: cannot multiply "
                    f"a bool field — use INTENT_OVERRIDES instead."
                )
                raise ValueError(msg)
            if isinstance(mult, bool) or not isinstance(mult, (int, float)):
                msg = (
                    f"INTENT_MULTIPLIERS[{intent!r}][{field!r}]: multiplier must "
                    f"be int or float, got {type(mult).__name__}={mult!r}."
                )
                raise ValueError(msg)

    for intent, overrides in INTENT_OVERRIDES.items():
        for field in overrides:
            if field not in _ALL_FIELDS:
                msg = (
                    f"INTENT_OVERRIDES[{intent!r}]: unknown field {field!r}. "
                    f"Allowed: {sorted(_ALL_FIELDS)}"
                )
                raise ValueError(msg)


_validate_policy_data()
