"""Per-intent retrieval policy resolver.

The router classifies each query's *intent* (factoid, multi_hop, comparative,
…). This module maps that intent to a ``RetrievalPolicy`` — the discrete
parameter set the retrieve→rerank→MMR stages should use for queries of that
class. Centralising the mapping makes it auditable and version-controlled:
no LLM trust on raw param values, no scattered if-intent-then-X branches in
the pipeline.

Policy *data* lives in ``src.config.intent_policies`` (multipliers + absolute
overrides). This module holds only the resolver + the ``RetrievalPolicy``
dataclass — logic separated from data per the ``src/config/<topic>.py`` pattern.

The defaults preserve the validated 2026-05-08 baseline (pool100, λ=0.6,
USE_MMR=True). Per-intent multipliers/overrides should only be added with
eval evidence.
"""

from __future__ import annotations

import dataclasses
import math
import os
from dataclasses import dataclass

from src import config


def _active_score_threshold() -> float:
    """Backend-calibrated rerank drop threshold.

    The NVIDIA NIM is bimodal (valid matches land at 0.02-0.18), so the 0.20
    threshold calibrated for the Qwen3 cross-encoders dropped every valid hit.
    remote_nvidia gets a low floor that rejects only ~0 garbage (errors / the
    identity fallback's would-be zeros) and defers relevance to the rerank floor
    + CRAG groundedness. Any other/unset backend keeps the legacy 0.20.
    """
    if os.getenv("RERANK_BACKEND", "").lower() == "remote_nvidia":
        return config.RERANK_SCORE_THRESHOLD_NVIDIA
    return config.RERANK_SCORE_THRESHOLD


@dataclass(frozen=True)
class RetrievalPolicy:
    """Frozen policy resolved at query-plan time, then threaded through the
    retrieve→rerank→MMR stages. Every field has a default that matches the
    validated production baseline; per-intent variants change only the fields
    the eval data justifies.
    """

    top_k_retrieve: int
    top_k_rerank: int
    use_mmr: bool
    mmr_lambda: float
    mmr_pool_mult: int
    score_threshold: float
    max_children_per_parent: int


def _baseline_policy() -> RetrievalPolicy:
    """The validated 2026-05-08 baseline. Returned for any unknown intent."""
    return RetrievalPolicy(
        top_k_retrieve=config.RETRIEVE_TOP_K,
        top_k_rerank=config.RERANK_TOP_K,
        use_mmr=config.USE_MMR,
        mmr_lambda=config.MMR_LAMBDA,
        mmr_pool_mult=config.MMR_POOL_MULT,
        score_threshold=_active_score_threshold(),
        max_children_per_parent=config.MAX_CHILDREN_PER_PARENT,
    )


def _scale_field(base_value: int | float, mult: float, field_name: str) -> int | float:
    """Scale a numeric field by ``mult``, preserving the input numeric type.

    - Int fields use ``math.ceil`` so retrieval pool sizes only ever scale UP
      (never silently down via banker's rounding on half-integer products).
    - Float fields keep precision via plain multiplication.
    - Bools (an int subclass) raise — bool multiplication is meaningless and
      indicates a misconfiguration that ``intent_policies._validate_policy_data``
      should have caught at module import. The runtime check is defence in
      depth for tests that inject synthetic dicts.
    """
    if isinstance(base_value, bool):
        msg = (
            f"_scale_field: refusing to multiply bool field {field_name!r} "
            f"(use INTENT_OVERRIDES for bool toggles, not INTENT_MULTIPLIERS)."
        )
        raise ValueError(msg)
    if isinstance(base_value, int):
        return int(math.ceil(base_value * mult))
    if isinstance(base_value, float):
        return float(base_value * mult)
    msg = (
        f"_scale_field: field {field_name!r} has non-numeric type "
        f"{type(base_value).__name__}; cannot apply multiplier."
    )
    raise ValueError(msg)


def policy_for_intent(
    intent: str | None,
    *,
    llm_use_mmr: bool | None = None,
) -> RetrievalPolicy:
    """Resolve a ``RetrievalPolicy`` for the given router intent.

    Resolution order (lowest → highest priority):
        1. baseline (validated config defaults)
        2. multipliers (from ``config.INTENT_MULTIPLIERS`` — numeric fields only,
           int fields use ``math.ceil``)
        3. absolute overrides (from ``config.INTENT_OVERRIDES`` — wins over
           multipliers, used for non-numeric fields or pinned values)
        4. ``llm_use_mmr`` if non-None (router's per-query call — wins over
           everything; field-isolated to ``use_mmr``)

    Unknown intents fall back to the baseline policy. ``llm_use_mmr=None``
    means the router didn't make a call and the intent baseline applies.

    Raises ``ValueError`` if a multiplier targets a bool or non-numeric field
    (defence in depth — module-import validation in ``intent_policies`` is
    the primary gate).
    """
    policy = _baseline_policy()
    key = intent or "unknown"

    # Defensive shallow copy so callers cannot mutate module-global config
    # by retaining a reference into the returned dict structure.
    multipliers = dict(config.INTENT_MULTIPLIERS.get(key, {}))
    overrides = dict(config.INTENT_OVERRIDES.get(key, {}))

    # ``dataclasses.replace`` accepts **kwargs but mypy can't reconcile a
    # heterogeneous dict (int|float for multipliers, object for overrides)
    # with the dataclass's per-field types. The runtime correctness is
    # enforced by ``_scale_field`` (numeric guard) and module-import
    # ``_validate_policy_data`` (field-name + type allowlist), not by
    # static types here.
    if multipliers:
        scaled: dict[str, int | float] = {}
        for field_name, mult in multipliers.items():
            base_value = getattr(policy, field_name)
            scaled[field_name] = _scale_field(base_value, mult, field_name)
        policy = dataclasses.replace(policy, **scaled)  # type: ignore[arg-type]
    if overrides:
        policy = dataclasses.replace(policy, **overrides)  # type: ignore[arg-type]
    if llm_use_mmr is not None:
        policy = dataclasses.replace(policy, use_mmr=llm_use_mmr)
    return policy
