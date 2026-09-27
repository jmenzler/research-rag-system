"""Tests for src/query/policies.py — intent → RetrievalPolicy mapping.

These tests pin the policy contract so future per-intent overrides have a
regression target.
"""

from __future__ import annotations

import math

import pytest

from src import config
from src.config import intent_policies
from src.query.policies import RetrievalPolicy, policy_for_intent


class TestPolicyDefaults:
    def test_unknown_intent_returns_baseline(self) -> None:
        p = policy_for_intent("unknown")
        assert p.top_k_retrieve == config.RETRIEVE_TOP_K
        assert p.top_k_rerank == config.RERANK_TOP_K
        assert p.use_mmr == config.USE_MMR
        assert p.mmr_lambda == config.MMR_LAMBDA
        assert p.mmr_pool_mult == config.MMR_POOL_MULT
        assert p.score_threshold == config.RERANK_SCORE_THRESHOLD
        assert p.max_children_per_parent == config.MAX_CHILDREN_PER_PARENT

    def test_none_intent_treated_as_unknown(self) -> None:
        p = policy_for_intent(None)
        baseline = policy_for_intent("unknown")
        assert p == baseline

    def test_garbage_intent_treated_as_unknown(self) -> None:
        p = policy_for_intent("totally_made_up_intent")
        baseline = policy_for_intent("unknown")
        assert p == baseline

    def test_all_known_intents_return_a_policy(self) -> None:
        # The 8 IntentClass values from src/query/router.py.
        for intent in [
            "factoid",
            "definitional",
            "multi_hop",
            "comparative",
            "literature_synthesis",
            "sensemaking",
            "computational",
            "unknown",
        ]:
            p = policy_for_intent(intent)
            assert isinstance(p, RetrievalPolicy)

    def test_baseline_intents_match_baseline(self) -> None:
        """The 4 'baseline' intents (factoid, definitional, computational,
        unknown) have empty multipliers AND empty overrides — single-doc
        recall, baseline pool sizing is correct. They MUST resolve identical
        to ``policy_for_intent("unknown")``. The 4 'wide-pool' intents
        (multi_hop, comparative, literature_synthesis, sensemaking) carry
        non-trivial multipliers and are covered by their own tests."""
        baseline = policy_for_intent("unknown")
        for intent in ["factoid", "definitional", "computational"]:
            p = policy_for_intent(intent)
            assert p == baseline, f"intent={intent} unexpectedly diverges from baseline"

    def test_multipliers_applied_for_multi_hop_comparative(self) -> None:
        """multi_hop and comparative apply a 2.0x multiplier to top_k_rerank
        (2026-05-09 evidence: Q4 needles 2/8 → 6/8 with rerank pool 20 → 40).
        All other fields stay at baseline. Int scaling uses ``math.ceil``."""
        baseline = policy_for_intent("unknown")
        expected_top_k_rerank = math.ceil(baseline.top_k_rerank * 2.0)
        for intent in ["multi_hop", "comparative"]:
            p = policy_for_intent(intent)
            assert p.top_k_rerank == expected_top_k_rerank, (
                f"intent={intent} top_k_rerank={p.top_k_rerank}, expected {expected_top_k_rerank}"
            )
            # All other numeric fields stay at baseline.
            assert p.top_k_retrieve == baseline.top_k_retrieve
            assert p.use_mmr == baseline.use_mmr
            assert p.mmr_lambda == baseline.mmr_lambda
            assert p.mmr_pool_mult == baseline.mmr_pool_mult
            assert p.score_threshold == baseline.score_threshold
            assert p.max_children_per_parent == baseline.max_children_per_parent

    def test_multipliers_applied_for_synthesis_intents(self) -> None:
        """literature_synthesis and sensemaking apply a 3.0x multiplier on
        top_k_rerank — even broader candidate sweep for survey-style queries.

        NOTE: these multipliers are unevidenced (no current golden Q has
        intent=literature_synthesis or sensemaking). See
        ``test_synthesis_multipliers_unevidenced_marker`` below — that test
        is intentionally skipped to surface the missing eval data in CI."""
        baseline = policy_for_intent("unknown")
        expected_top_k_rerank = math.ceil(baseline.top_k_rerank * 3.0)
        for intent in ["literature_synthesis", "sensemaking"]:
            p = policy_for_intent(intent)
            assert p.top_k_rerank == expected_top_k_rerank, (
                f"intent={intent} top_k_rerank={p.top_k_rerank}, expected {expected_top_k_rerank}"
            )
            assert p.top_k_retrieve == baseline.top_k_retrieve

    @pytest.mark.skip(
        reason=(
            "literature_synthesis and sensemaking carry a 3.0x multiplier on "
            "top_k_rerank that is NOT validated by any current golden Q. "
            "Tracking gap in CI; remove skip once golden v2 seeds queries with "
            "these intents and N=10 eval confirms CR/CP do not regress."
        )
    )
    def test_synthesis_multipliers_unevidenced_marker(self) -> None:
        """Placeholder — surfaces in CI as a skipped test with reason so the
        unevidenced multiplier is visible during PR review."""

    def test_int_field_scales_via_ceil_not_round(self) -> None:
        """Int multiplier uses ``math.ceil`` so half-integer products always
        round UP. Banker's rounding (``round(22.5)==22``) would be unexpected
        for retrieval pool sizes — practitioners expect "scale up, never
        silently down". Patch a synthetic 1.5x multiplier on ``mmr_pool_mult``
        (baseline=3 → 1.5*3=4.5 → ceil=5) and verify."""
        baseline = policy_for_intent("unknown")
        original_mults = intent_policies.INTENT_MULTIPLIERS["multi_hop"]
        try:
            intent_policies.INTENT_MULTIPLIERS["multi_hop"] = {"mmr_pool_mult": 1.5}
            p = policy_for_intent("multi_hop")
            expected = math.ceil(baseline.mmr_pool_mult * 1.5)
            assert p.mmr_pool_mult == expected
            assert isinstance(p.mmr_pool_mult, int)
        finally:
            intent_policies.INTENT_MULTIPLIERS["multi_hop"] = original_mults

    def test_lockstep_with_baseline_drift(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """If ``RERANK_TOP_K`` is bumped from 20 to 30, multi_hop (×2) MUST
        resolve to 60 — never pinned at 40. This pins the headline contract
        of the multiplier mechanism: stays in lock-step with baseline tuning."""
        monkeypatch.setattr(config, "RERANK_TOP_K", 30)
        p = policy_for_intent("multi_hop")
        assert p.top_k_rerank == 60, (
            f"baseline drift broke lock-step contract: top_k_rerank={p.top_k_rerank}, "
            f"expected 60 (= 30 * 2.0)"
        )
        # Synthesis intents likewise scale.
        p_synth = policy_for_intent("literature_synthesis")
        assert p_synth.top_k_rerank == 90, (
            f"baseline drift broke synthesis lock-step: top_k_rerank={p_synth.top_k_rerank}, "
            f"expected 90 (= 30 * 3.0)"
        )

    def test_policy_is_immutable(self) -> None:
        # frozen=True dataclass → mutation raises FrozenInstanceError
        p = policy_for_intent("unknown")
        try:
            p.top_k_retrieve = 999  # type: ignore[misc]
        except Exception:
            return
        msg = "RetrievalPolicy should be frozen"
        raise AssertionError(msg)


class TestLLMUseMMROverride:
    """Resolver order: baseline → intent override → llm_use_mmr override."""

    def test_llm_true_overrides_baseline(self) -> None:
        baseline = policy_for_intent("factoid")
        # Skip if config baseline is already True (test would be vacuous).
        if baseline.use_mmr is True:
            p = policy_for_intent("factoid", llm_use_mmr=True)
            assert p.use_mmr is True
            return
        p = policy_for_intent("factoid", llm_use_mmr=True)
        assert p.use_mmr is True
        assert p.top_k_retrieve == baseline.top_k_retrieve

    def test_llm_false_overrides_baseline(self) -> None:
        p = policy_for_intent("factoid", llm_use_mmr=False)
        assert p.use_mmr is False

    def test_llm_none_falls_through_to_intent_baseline(self) -> None:
        p_none = policy_for_intent("multi_hop", llm_use_mmr=None)
        p_baseline = policy_for_intent("multi_hop")
        assert p_none == p_baseline

    def test_llm_override_does_not_touch_other_fields(self) -> None:
        baseline = policy_for_intent("comparative")
        p = policy_for_intent("comparative", llm_use_mmr=False)
        assert p.use_mmr is False
        assert p.top_k_retrieve == baseline.top_k_retrieve
        assert p.top_k_rerank == baseline.top_k_rerank
        assert p.mmr_lambda == baseline.mmr_lambda
        assert p.score_threshold == baseline.score_threshold


class TestIntentMultipliers:
    """Resolution order: baseline → multipliers → absolute overrides → llm_use_mmr.
    Multipliers stay numeric only; absolute overrides win over multipliers;
    llm_use_mmr (when non-None) wins over both."""

    def test_int_field_multiplier_yields_int(self) -> None:
        """Integer fields must remain int after multiplier."""
        p = policy_for_intent("multi_hop")
        assert isinstance(p.top_k_rerank, int)
        assert not isinstance(p.top_k_rerank, bool)  # bool is an int subclass

    def test_absolute_override_wins_over_multiplier(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When INTENT_OVERRIDES[intent] sets a numeric field, that absolute
        value wins over the multiplied value. Use ``monkeypatch.setitem`` for
        interrupt-safe teardown."""
        override_value = 7
        monkeypatch.setitem(
            intent_policies.INTENT_OVERRIDES,
            "multi_hop",
            {"top_k_rerank": override_value},
        )
        p = policy_for_intent("multi_hop")
        assert p.top_k_rerank == override_value

    def test_llm_use_mmr_wins_over_intent_overrides(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """llm_use_mmr (when non-None) is the highest-priority resolver. Even
        if INTENT_OVERRIDES sets use_mmr explicitly, llm_use_mmr wins."""
        monkeypatch.setitem(
            intent_policies.INTENT_OVERRIDES,
            "multi_hop",
            {"use_mmr": False},
        )
        p_no_llm = policy_for_intent("multi_hop")
        assert p_no_llm.use_mmr is False  # absolute override wins (no llm signal)

        p_llm_true = policy_for_intent("multi_hop", llm_use_mmr=True)
        assert p_llm_true.use_mmr is True  # llm_use_mmr beats absolute override

    def test_missing_intent_in_multipliers_falls_back_to_empty(self) -> None:
        """Garbage / unknown intents → no multiplier applied, baseline returned."""
        p = policy_for_intent("totally_made_up_intent")
        baseline = policy_for_intent("unknown")
        assert p == baseline

    def test_multiplier_on_bool_field_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Multipliers MUST refuse to scale a bool field (would silently
        cast True/False to 1/0 without ``_scale_field``'s bool guard).
        Per project rule "Fail loudly", a bool multiplier raises ValueError
        rather than logging a warning."""
        monkeypatch.setitem(
            intent_policies.INTENT_MULTIPLIERS,
            "multi_hop",
            {"use_mmr": 2.0},
        )
        with pytest.raises(ValueError, match="bool field"):
            policy_for_intent("multi_hop")


class TestPolicyDataValidation:
    """Module-import validation in ``intent_policies._validate_policy_data``
    must reject misconfiguration (typos, bool multipliers, non-numeric
    multipliers) at startup, not silently no-op every query."""

    def test_unknown_field_in_multipliers_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Patch a typo'd field name and re-run the validator.
        monkeypatch.setitem(
            intent_policies.INTENT_MULTIPLIERS,
            "multi_hop",
            {"top_k_rerannk": 2.0},  # typo intentional
        )
        with pytest.raises(ValueError, match="unknown field"):
            intent_policies._validate_policy_data()

    def test_bool_field_multiplier_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(
            intent_policies.INTENT_MULTIPLIERS,
            "multi_hop",
            {"use_mmr": 2.0},
        )
        with pytest.raises(ValueError, match="cannot multiply"):
            intent_policies._validate_policy_data()

    def test_non_numeric_multiplier_value_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(
            intent_policies.INTENT_MULTIPLIERS,
            "multi_hop",
            {"top_k_rerank": "two"},  # type: ignore[dict-item]
        )
        with pytest.raises(ValueError, match="must be int or float"):
            intent_policies._validate_policy_data()

    def test_unknown_field_in_overrides_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(
            intent_policies.INTENT_OVERRIDES,
            "multi_hop",
            {"unknown_field": 42},
        )
        with pytest.raises(ValueError, match="unknown field"):
            intent_policies._validate_policy_data()

    def test_overrides_and_multipliers_keys_must_match(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Drop one intent from OVERRIDES while leaving MULTIPLIERS intact.
        new_overrides = {
            k: v for k, v in intent_policies.INTENT_OVERRIDES.items() if k != "factoid"
        }
        monkeypatch.setattr(intent_policies, "INTENT_OVERRIDES", new_overrides)
        with pytest.raises(ValueError, match="do not match"):
            intent_policies._validate_policy_data()


class TestBackendAwareThreshold:
    """The baseline score_threshold is calibrated per reranker backend: the
    bimodal NVIDIA NIM gets a low floor, every other/unset backend keeps 0.20."""

    def test_nvidia_backend_uses_low_threshold(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RERANK_BACKEND", "remote_nvidia")
        p = policy_for_intent("unknown")
        assert p.score_threshold == config.RERANK_SCORE_THRESHOLD_NVIDIA

    def test_non_nvidia_backend_uses_default_threshold(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("RERANK_BACKEND", "remote_llamacpp")
        p = policy_for_intent("unknown")
        assert p.score_threshold == config.RERANK_SCORE_THRESHOLD

    def test_unset_backend_uses_default_threshold(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("RERANK_BACKEND", raising=False)
        p = policy_for_intent("unknown")
        assert p.score_threshold == config.RERANK_SCORE_THRESHOLD

    def test_nvidia_threshold_is_below_default(self) -> None:
        # The whole point: NVIDIA's valid-match band (0.02-0.18) sits under 0.20.
        assert config.RERANK_SCORE_THRESHOLD_NVIDIA < config.RERANK_SCORE_THRESHOLD
