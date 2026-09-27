"""Tests for src/query/router.py — RoutingPlan parsing.

Covers the use_mmr / use_mmr_reason fields added 2026-05-08. The LLM call
itself is monkey-patched; these tests pin the parser contract.

Updated 2026-05-13: router.py switched from _generate_text_via_provider to
call_text (src.query.usage_track). Patching updated accordingly.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import pytest

from src.models import Usage
from src.query.router import RoutingPlan, route_query
from src.query.usage_track import LLMCallResult


def _fake_llm_result(payload: dict[str, Any]) -> LLMCallResult:
    return LLMCallResult(
        text=json.dumps(payload),
        latency_ms=1,
        usage=Usage(input=0, output=0, reasoning=0, total=0),
        raw_response=None,
        cost_usd=0.0,
        model="fake",
    )


def _route(payload: dict[str, Any]) -> RoutingPlan:
    with patch(
        "src.query.router.call_text",
        return_value=_fake_llm_result(payload),
    ):
        plan, _ = route_query("test query")
    return plan


class TestUseMmrParsing:
    def test_parses_explicit_true(self) -> None:
        plan = _route(
            {
                "intent": "literature_synthesis",
                "entities": ["a", "b"],
                "fire": ["milvus"],
                "tool_payloads": {"milvus": ["sub-q-1"]},
                "use_mmr": True,
                "use_mmr_reason": "needs broad coverage",
            }
        )
        assert plan.use_mmr is True
        assert plan.use_mmr_reason == "needs broad coverage"

    def test_parses_explicit_false(self) -> None:
        plan = _route(
            {
                "intent": "factoid",
                "entities": ["x"],
                "fire": ["milvus"],
                "tool_payloads": {"milvus": ["sub-q-1"]},
                "use_mmr": False,
                "use_mmr_reason": "single-source factual",
            }
        )
        assert plan.use_mmr is False
        assert plan.use_mmr_reason == "single-source factual"

    def test_null_stays_none(self) -> None:
        plan = _route(
            {
                "intent": "unknown",
                "entities": [],
                "fire": ["milvus"],
                "tool_payloads": {"milvus": ["sub-q-1"]},
                "use_mmr": None,
                "use_mmr_reason": "",
            }
        )
        assert plan.use_mmr is None
        assert plan.use_mmr_reason == ""

    def test_missing_field_defaults_to_none(self) -> None:
        plan = _route(
            {
                "intent": "factoid",
                "entities": [],
                "fire": ["milvus"],
                "tool_payloads": {"milvus": ["sub-q-1"]},
            }
        )
        assert plan.use_mmr is None
        assert plan.use_mmr_reason == ""

    def test_garbage_use_mmr_value_defaults_to_none(self) -> None:
        # "yes" is not a bool — must coerce to None, not crash
        plan = _route(
            {
                "intent": "factoid",
                "entities": [],
                "fire": ["milvus"],
                "tool_payloads": {"milvus": ["sub-q-1"]},
                "use_mmr": "yes",
                "use_mmr_reason": "should still parse",
            }
        )
        assert plan.use_mmr is None
        assert plan.use_mmr_reason == "should still parse"

    def test_long_reason_truncated(self) -> None:
        plan = _route(
            {
                "intent": "factoid",
                "entities": [],
                "fire": ["milvus"],
                "tool_payloads": {"milvus": ["sub-q-1"]},
                "use_mmr": True,
                "use_mmr_reason": "x" * 500,
            }
        )
        assert len(plan.use_mmr_reason) == 200

    def test_non_string_reason_drops_to_empty(self) -> None:
        plan = _route(
            {
                "intent": "factoid",
                "entities": [],
                "fire": ["milvus"],
                "tool_payloads": {"milvus": ["sub-q-1"]},
                "use_mmr": True,
                "use_mmr_reason": 42,
            }
        )
        assert plan.use_mmr_reason == ""


@pytest.fixture(autouse=True)
def _no_real_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Belt-and-suspenders: never accidentally hit a real provider."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
