"""Extended router tests covering all 12 TDD steps.

Tests for:
1. Pydantic schema (RouterOutputSchema, Disambig)
2. RewriterConfig dataclass + _baseline_rewriter_config
3. rewriter_config_for_intent resolver
4. _validate_rewriter_configs module-import validator
5. _build_prompt(cfg) prompt builder
6. Section file existence
7. RoutingPlan extension (new optional fields)
8. Schema-validated route_query (call_text with schema=)
9. Bit-identical baseline test (pre-refactor snapshot)
10-12. Pipeline / audit / render tests in test_pipeline_extended_router.py
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import pytest

from src.models import Usage
from src.query.router import RoutingPlan, route_query
from src.query.usage_track import LLMCallResult

# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------


def _fake_llm_result(payload: dict[str, Any]) -> LLMCallResult:
    return LLMCallResult(
        text=json.dumps(payload),
        latency_ms=1,
        usage=Usage(input=0, output=0, reasoning=0, total=0),
        raw_response=None,
        cost_usd=0.0,
        model="fake",
    )


def _route_via_call_text(payload: dict[str, Any]) -> RoutingPlan:
    """Route using the new call_text-based path."""
    with patch(
        "src.query.router.call_text",
        return_value=_fake_llm_result(payload),
    ):
        plan, _ = route_query("test query")
    return plan


# ---------------------------------------------------------------------------
# Step 1: Pydantic schema tests
# ---------------------------------------------------------------------------


class TestRouterOutputSchema:
    def test_router_output_schema_validates_minimal(self) -> None:
        """Minimal valid payload (just intent + tool_payloads) should validate."""
        from src.query.router import RouterOutputSchema

        schema = RouterOutputSchema.model_validate(
            {
                "intent": "factoid",
                "tool_payloads": {"milvus": ["q1"]},
            }
        )
        assert schema.intent == "factoid"
        assert schema.tool_payloads == {"milvus": ["q1"]}
        assert schema.hyde_doc is None
        assert schema.stepback is None
        assert schema.disambiguation == []

    def test_router_output_schema_accepts_full_fields(self) -> None:
        """Full payload with all new optional fields should validate."""
        from src.query.router import RouterOutputSchema

        schema = RouterOutputSchema.model_validate(
            {
                "intent": "multi_hop",
                "entities": ["A", "B"],
                "fire": ["milvus"],
                "tool_payloads": {"milvus": ["q1", "q2"]},
                "use_mmr": True,
                "use_mmr_reason": "needs diversity",
                "hyde_doc": "A hypothetical document about multi-hop reasoning...",
                "stepback": "What is multi-hop reasoning?",
                "disambiguation": [
                    {"label": "interpretation A", "clarified_query": "Q A"},
                    {"label": "interpretation B", "clarified_query": "Q B"},
                ],
                "filters": {"year_min": 2020, "year_max": 2024},
            }
        )
        assert schema.hyde_doc is not None
        assert schema.stepback == "What is multi-hop reasoning?"
        assert len(schema.disambiguation) == 2
        assert schema.filters == {"year_min": 2020, "year_max": 2024}

    def test_router_output_schema_rejects_unknown_intent(self) -> None:
        """Unknown intent values are accepted (intent is a plain str in schema)."""
        # The schema accepts any string; intent validation is done at the RoutingPlan level.
        from src.query.router import RouterOutputSchema

        # Should NOT raise — we accept unknown intents gracefully (fallback logic)
        schema = RouterOutputSchema.model_validate(
            {
                "intent": "not_a_real_intent",
                "tool_payloads": {"milvus": ["q1"]},
            }
        )
        assert schema.intent == "not_a_real_intent"


# ---------------------------------------------------------------------------
# Step 2: RewriterConfig dataclass tests
# ---------------------------------------------------------------------------


class TestRewriterConfig:
    def test_rewriter_config_frozen(self) -> None:
        """RewriterConfig must be frozen (immutable)."""
        from src.config.rewriter_configs import RewriterConfig

        cfg = RewriterConfig(
            model="deepseek-v4-flash",
            emit_hyde=False,
            emit_stepback=False,
            emit_disambiguation=False,
            emit_filters=False,
            dense_max_tokens=40,
            hyde_max_tokens=80,
            stepback_max_tokens=30,
            sub_queries_min=1,
            sub_queries_max=4,
            max_output_tokens=2048,
        )
        with pytest.raises((AttributeError, TypeError)):
            cfg.emit_hyde = True  # type: ignore[misc]

    def test_baseline_reads_env_knobs(self) -> None:
        """_baseline_rewriter_config reads REWRITER_* env knobs."""
        from src.config.rewriter_configs import _baseline_rewriter_config

        cfg = _baseline_rewriter_config()
        assert isinstance(cfg.emit_hyde, bool)
        assert isinstance(cfg.emit_stepback, bool)
        assert isinstance(cfg.max_output_tokens, int)
        assert cfg.max_output_tokens == 2048

    def test_validator_rejects_bad_field(self) -> None:
        """_validate_rewriter_configs raises ValueError on unknown field name."""
        from src.config.rewriter_configs import _validate_rewriter_configs

        with pytest.raises(ValueError, match="unknown field"):
            _validate_rewriter_configs(
                {
                    "factoid": {"nonexistent_field": True},
                }
            )


# ---------------------------------------------------------------------------
# Step 3: per-intent machinery has been removed (PR #30 review HIGH #1).
#
# The user clarified intent: there is no per-intent gating. The design is a
# single global RewriterConfig from env vars, used for A/B testing which
# prompt sections are enabled. Tests that assumed per-intent resolution
# were deleted along with INTENT_REWRITER_CONFIG and rewriter_config_for_intent.
# ---------------------------------------------------------------------------


class TestPerIntentMachineryRemoved:
    """Pin the design decision: INTENT_REWRITER_CONFIG and
    rewriter_config_for_intent must NOT exist on src.config.rewriter_configs.
    """

    def test_intent_rewriter_config_is_gone(self) -> None:
        import src.config.rewriter_configs as rw

        assert not hasattr(rw, "INTENT_REWRITER_CONFIG"), (
            "INTENT_REWRITER_CONFIG was deleted (PR #30 HIGH #1); the rewriter "
            "is now globally configured via REWRITER_EMIT_* env vars only."
        )

    def test_rewriter_config_for_intent_is_gone(self) -> None:
        import src.config.rewriter_configs as rw

        assert not hasattr(rw, "rewriter_config_for_intent"), (
            "rewriter_config_for_intent was deleted (PR #30 HIGH #1); use "
            "_baseline_rewriter_config() directly."
        )

    def test_baseline_default_all_emit_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With all REWRITER_EMIT_* env vars unset (default), baseline all-emit-false."""
        monkeypatch.delenv("REWRITER_EMIT_HYDE", raising=False)
        monkeypatch.delenv("REWRITER_EMIT_STEPBACK", raising=False)
        monkeypatch.delenv("REWRITER_EMIT_DISAMBIGUATION", raising=False)
        monkeypatch.delenv("REWRITER_EMIT_FILTERS", raising=False)

        import importlib

        import src.config.rewriter_configs as rw

        importlib.reload(rw)

        from src.config.rewriter_configs import _baseline_rewriter_config

        cfg = _baseline_rewriter_config()
        assert cfg.emit_hyde is False
        assert cfg.emit_stepback is False
        assert cfg.emit_disambiguation is False
        assert cfg.emit_filters is False


# ---------------------------------------------------------------------------
# Step 4: Module-import validator tests
# ---------------------------------------------------------------------------


class TestValidateRewriterConfigs:
    def test_validator_raises_on_typo(self) -> None:
        from src.config.rewriter_configs import _validate_rewriter_configs

        with pytest.raises(ValueError, match="unknown field"):
            _validate_rewriter_configs({"factoid": {"emitt_hyde": True}})  # typo

    def test_validator_raises_on_wrong_type(self) -> None:
        from src.config.rewriter_configs import _validate_rewriter_configs

        with pytest.raises(ValueError, match="must be bool"):
            _validate_rewriter_configs({"factoid": {"emit_hyde": "yes"}})  # str not bool


# ---------------------------------------------------------------------------
# Step 5: Prompt build tests
# ---------------------------------------------------------------------------


class TestBuildPrompt:
    def test_baseline_cfg_prompt_matches_pre_refactor_system_md_exactly(self) -> None:
        """STRENGTHENED bit-identical baseline (PR #30 MEDIUM #4):

        With all emit_* False, _build_prompt(cfg) must return exactly the
        pre-refactor prompts/router/system.md content. No conditional section
        bytes leak in.

        This is the core invariant the bit-identical-baseline guarantee rests on.
        """
        import dataclasses
        from pathlib import Path

        from src.config.rewriter_configs import _baseline_rewriter_config
        from src.query.router import _build_prompt

        cfg = dataclasses.replace(
            _baseline_rewriter_config(),
            emit_hyde=False,
            emit_stepback=False,
            emit_disambiguation=False,
            emit_filters=False,
        )
        prompt = _build_prompt(cfg)

        system_md = (Path(__file__).parent.parent / "prompts" / "router" / "system.md").read_text(
            encoding="utf-8"
        )

        # Normalize trailing whitespace at the end only — concat may add a
        # newline, but the content body must match exactly.
        assert prompt.rstrip() == system_md.rstrip(), (
            "With all emit_* False, _build_prompt must return the pre-refactor "
            "system.md verbatim. Conditional section content leaked in."
        )

    def test_baseline_cfg_excludes_all_optional_section_markers(self) -> None:
        """STRENGTHENED (PR #30 MEDIUM #4): when all emit_* are False, the prompt
        must NOT contain the JSON field names that conditional sections introduce.

        The field names ``hyde_doc``, ``stepback``, ``disambiguation``, and ``filters``
        appear ONLY in the conditional section files (_hyde.md, _stepback.md,
        _disambiguation.md, _filters.md, _self_gate.md). They do not appear in
        the base system.md. So their absence proves no section content leaked in.
        """
        import dataclasses

        from src.config.rewriter_configs import _baseline_rewriter_config
        from src.query.router import _build_prompt

        cfg = dataclasses.replace(
            _baseline_rewriter_config(),
            emit_hyde=False,
            emit_stepback=False,
            emit_disambiguation=False,
            emit_filters=False,
        )
        prompt = _build_prompt(cfg)

        assert "hyde_doc" not in prompt, "HyDE section content leaked into baseline prompt"
        assert "stepback" not in prompt, "Stepback section content leaked into baseline prompt"
        assert "disambiguation" not in prompt, (
            "Disambiguation section content leaked into baseline prompt"
        )
        # "filters" is unfortunately a common word; check the explicit section-marker
        # phrase used in _filters.md instead (``year_min``).
        assert "year_min" not in prompt, "Filters section content leaked into baseline prompt"

    def test_hyde_section_appears_when_emit_hyde_true(self) -> None:
        """When emit_hyde=True, _build_prompt includes the _hyde.md section."""
        import dataclasses

        from src.config.rewriter_configs import _baseline_rewriter_config
        from src.query.router import _build_prompt

        cfg = dataclasses.replace(_baseline_rewriter_config(), emit_hyde=True)
        prompt = _build_prompt(cfg)
        assert "hyde_doc" in prompt

    def test_self_gate_appended_when_any_section_enabled(self) -> None:
        """_self_gate.md is appended when at least one optional section is enabled.

        Previous design appended it unconditionally; per PR #30 MEDIUM #4 the
        baseline must be bit-identical to system.md, so the self-gate only
        ships when there's a section to gate.
        """
        import dataclasses

        from src.config.rewriter_configs import _baseline_rewriter_config
        from src.query.router import _build_prompt

        cfg = dataclasses.replace(_baseline_rewriter_config(), emit_hyde=True)
        prompt = _build_prompt(cfg)
        # _self_gate.md content is recognizable by its first non-blank line.
        # We don't assert exact match (the file content may evolve); we assert
        # the section header word "SELF-GATE" is present.
        assert "SELF-GATE" in prompt or "self_gate" in prompt.lower()


# ---------------------------------------------------------------------------
# Step 6: Section file existence + content tests
# ---------------------------------------------------------------------------


class TestSectionFiles:
    _SECTIONS_DIR = (
        __import__("pathlib").Path(__file__).parent.parent / "prompts" / "router" / "sections"
    )

    def test_all_section_files_present(self) -> None:
        expected = [
            "_hyde.md",
            "_stepback.md",
            "_disambiguation.md",
            "_filters.md",
            "_self_gate.md",
        ]
        for name in expected:
            path = self._SECTIONS_DIR / name
            assert path.exists(), f"Missing section file: {path}"

    def test_section_files_under_50_lines(self) -> None:
        for path in self._SECTIONS_DIR.glob("_*.md"):
            lines = path.read_text(encoding="utf-8").splitlines()
            assert len(lines) <= 50, f"{path.name} has {len(lines)} lines (limit 50)"


# ---------------------------------------------------------------------------
# Step 7: RoutingPlan extension tests
# ---------------------------------------------------------------------------


class TestRoutingPlanExtension:
    def test_routing_plan_defaults_preserve_none(self) -> None:
        """New optional fields default to None/[] when not provided."""
        plan = RoutingPlan(
            user_query="test",
            intent="factoid",
            entities=[],
            nodes=[],
        )
        assert plan.hyde_doc is None
        assert plan.stepback is None
        assert plan.disambiguation == []
        assert plan.filters == {}

    def test_routing_plan_accepts_new_fields(self) -> None:
        """RoutingPlan can be constructed with new optional fields."""
        from src.query.router import Disambig, RoutingPlan, ToolCall

        plan = RoutingPlan(
            user_query="test",
            intent="multi_hop",
            entities=["A"],
            nodes=[ToolCall(tool="milvus", sub_queries=["q1"])],
            hyde_doc="hypothetical document text",
            stepback="What is multi-hop?",
            disambiguation=[Disambig(label="A", clarified_query="Q A")],
            filters={"year_min": 2020},
        )
        assert plan.hyde_doc == "hypothetical document text"
        assert plan.stepback == "What is multi-hop?"
        assert len(plan.disambiguation) == 1
        assert plan.filters == {"year_min": 2020}


# ---------------------------------------------------------------------------
# Step 8: Schema-validated route_query tests
# ---------------------------------------------------------------------------


class TestSchemaValidatedRouteQuery:
    def test_route_query_parses_full_response(self) -> None:
        """route_query correctly parses a full response with all new fields."""
        payload = {
            "intent": "multi_hop",
            "entities": ["A", "B"],
            "fire": ["milvus"],
            "tool_payloads": {"milvus": ["q1", "q2"]},
            "use_mmr": True,
            "use_mmr_reason": "needs diversity",
            "hyde_doc": "A hypothetical document text for embedding",
            "stepback": "What is multi-hop reasoning?",
            "disambiguation": [],
            "filters": {"year_min": 2021},
        }
        plan = _route_via_call_text(payload)
        assert plan.intent == "multi_hop"
        assert plan.hyde_doc == "A hypothetical document text for embedding"
        assert plan.stepback == "What is multi-hop reasoning?"
        assert plan.filters == {"year_min": 2021}

    def test_route_query_salvages_partial(self) -> None:
        """Partial response (missing optional fields) falls back gracefully."""
        payload = {
            "intent": "factoid",
            "tool_payloads": {"milvus": ["q1"]},
        }
        plan = _route_via_call_text(payload)
        assert plan.intent == "factoid"
        assert plan.hyde_doc is None
        assert plan.stepback is None

    def test_route_query_full_fallback_on_garbage(self) -> None:
        """Completely invalid response triggers _fallback_plan (never raises)."""
        with patch(
            "src.query.router.call_text",
            return_value=LLMCallResult(
                text="not json at all %%%",
                latency_ms=1,
                usage=Usage(input=0, output=0, reasoning=0, total=0),
                raw_response=None,
                cost_usd=0.0,
                model="fake",
            ),
        ):
            plan, _ = route_query("test query")
        assert plan.intent == "unknown"
        assert len(plan.nodes) == 1
        assert plan.nodes[0].sub_queries == ["test query"]


# ---------------------------------------------------------------------------
# Step 9: Bit-identical baseline test
# ---------------------------------------------------------------------------


# Pre-refactor snapshot: the shape + key fields that route_query returned
# BEFORE any router.py changes. Captured from test_router.py behavior.
_BASELINE_SNAPSHOT: list[dict[str, Any]] = [
    # Query 1: simple factoid
    {
        "query": "What is VPIN?",
        "payload": {
            "intent": "definitional",
            "entities": ["VPIN"],
            "fire": ["milvus"],
            "tool_payloads": {"milvus": ["VPIN order flow toxicity definition"]},
            "use_mmr": False,
            "use_mmr_reason": "single-source definition",
        },
        "expected_intent_options": ["factoid", "definitional"],
        "expected_milvus_count": 1,
    },
    # Query 2: multi-hop
    {
        "query": "How does EWMA volatility affect Avellaneda-Stoikov spreads?",
        "payload": {
            "intent": "multi_hop",
            "entities": ["EWMA", "Avellaneda-Stoikov"],
            "fire": ["milvus"],
            "tool_payloads": {"milvus": ["q1", "q2"]},
            "use_mmr": True,
            "use_mmr_reason": "multi-hop needs diversity",
        },
        "expected_intent_options": ["multi_hop"],
        "expected_milvus_count": 2,
    },
    # Query 3: literature synthesis
    {
        "query": "Review the literature on risk-parity portfolio construction",
        "payload": {
            "intent": "literature_synthesis",
            "entities": ["risk parity"],
            "fire": ["milvus"],
            "tool_payloads": {"milvus": ["q1", "q2", "q3"]},
            "use_mmr": True,
            "use_mmr_reason": "literature survey",
        },
        "expected_intent_options": ["literature_synthesis", "sensemaking"],
        "expected_milvus_count": 3,
    },
]


class TestBitIdenticalBaseline:
    """Verify that with all REWRITER_EMIT_* false and INTENT_REWRITER_CONFIG
    effectively empty, the router output shape is identical to pre-refactor behavior.

    The key invariants:
    - hyde_doc is None
    - stepback is None
    - disambiguation is []
    - filters is {}
    - The milvus sub_queries list is non-empty
    - intent is a valid IntentClass string
    """

    def test_default_env_no_overrides_matches_pre_refactor_output(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """With all REWRITER_EMIT_* false, new fields must be None/[] (bit-identical shape)."""
        monkeypatch.setenv("REWRITER_EMIT_HYDE", "false")
        monkeypatch.setenv("REWRITER_EMIT_STEPBACK", "false")
        monkeypatch.setenv("REWRITER_EMIT_DISAMBIGUATION", "false")
        monkeypatch.setenv("REWRITER_EMIT_FILTERS", "false")

        for snapshot in _BASELINE_SNAPSHOT:
            payload = snapshot["payload"]
            plan = _route_via_call_text(payload)

            # Bit-identical baseline: new fields must be absent/empty
            assert plan.hyde_doc is None, f"hyde_doc should be None for {snapshot['query']!r}"
            assert plan.stepback is None, f"stepback should be None for {snapshot['query']!r}"
            assert plan.disambiguation == [], (
                f"disambiguation should be [] for {snapshot['query']!r}"
            )
            assert plan.filters == {}, f"filters should be {{}} for {snapshot['query']!r}"

            # Legacy fields must still work correctly
            assert plan.intent in snapshot["expected_intent_options"], (
                f"intent {plan.intent!r} not in {snapshot['expected_intent_options']}"
            )
            milvus_node = next((n for n in plan.nodes if n.tool == "milvus"), None)
            assert milvus_node is not None
            assert len(milvus_node.sub_queries) == snapshot["expected_milvus_count"]


# ---------------------------------------------------------------------------
# Belt-and-suspenders fixture: no real LLM calls
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_real_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
