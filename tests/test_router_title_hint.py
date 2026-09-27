"""RED tests for router.py title_hint field (CHAT-10 / D-21).

Phase 2 ADDITIVE: ``RouterOutputSchema`` and ``RoutingPlan`` gain a
``title_hint: str | None`` field, defaulting to ``None`` so existing
callers stay bit-identical. The field is emitted by the LLM only when
``RewriterConfig.emit_title_hint`` is True; api_chats flips the flag for
the first turn of an "Untitled chat" so auto-name lands inside the existing
decompose call with zero extra LLM cost.

Wave-1 tests use deferred imports so collection stays green before the
schema gains the new field (Plan 02-04 / 02-09 lands the change).
"""

from __future__ import annotations

import dataclasses
from typing import Any
from unittest.mock import patch

import pytest


def _build_fake_call_result(text: str) -> Any:  # noqa: ANN401 — duck-typed in tests
    """Synthesize an LLMCallResult-like object with the minimum fields route_query reads.

    The router accesses ``result.text`` only; usage/cost/latency are only used
    downstream by the caller's audit-trail bookkeeping (not exercised here).
    """
    from src.models import Usage  # noqa: PLC0415
    from src.query.usage_track import LLMCallResult  # noqa: PLC0415

    return LLMCallResult(
        text=text,
        latency_ms=10,
        usage=Usage(input=10, output=20, reasoning=0, total=30),
        raw_response=None,
        cost_usd=0.0,
        model="fake",
    )


def _build_cfg_with_title_hint(*, emit_title_hint: bool) -> Any:  # noqa: ANN401
    """Build a RewriterConfig with ``emit_title_hint`` set.

    Uses ``dataclasses.replace`` so the field-list lives in the dataclass — the
    Wave 1 RED state is precisely "AttributeError on emit_title_hint" until the
    field exists on RewriterConfig (Plan 02-04 / 02-09 adds it).
    """
    from src.config.rewriter_configs import _baseline_rewriter_config  # noqa: PLC0415

    base = _baseline_rewriter_config()
    return dataclasses.replace(
        base,
        emit_title_hint=emit_title_hint,  # type: ignore[call-arg]
    )


def test_title_hint_field_defaults_to_none() -> None:
    """RED — RouterOutputSchema.title_hint defaults to None.

    Backward-compatible default: providers that emit only the legacy fields
    still parse, mirroring how ``hyde_doc`` / ``stepback`` / ``disambiguation``
    were added in Phase 1.
    """
    from src.query.router import RouterOutputSchema  # noqa: PLC0415

    inst = RouterOutputSchema(
        intent="factoid",
        tool_payloads={"milvus": ["q"]},
    )
    assert hasattr(inst, "title_hint"), (
        "Phase 2 D-21: RouterOutputSchema is missing the title_hint field. "
        "Add `title_hint: str | None = None` to the pydantic model."
    )
    assert inst.title_hint is None


def test_title_hint_emitted_when_section_active() -> None:
    """RED — RoutingPlan.title_hint is populated when the LLM emits it.

    Drives ``route_query`` with a RewriterConfig that has
    ``emit_title_hint=True``; monkeypatch ``call_text`` to return a synthetic
    JSON payload carrying ``title_hint``; assert the plan field is populated.
    """
    pytest.importorskip("src.query.router")

    from src.query.router import route_query  # noqa: PLC0415

    fake_payload = (
        '{"intent": "factoid", "entities": [], "fire": ["milvus"], '
        '"tool_payloads": {"milvus": ["test sub"]}, '
        '"title_hint": "test title hint"}'
    )
    fake_result = _build_fake_call_result(fake_payload)
    cfg = _build_cfg_with_title_hint(emit_title_hint=True)

    with patch("src.query.router.call_text", return_value=fake_result):
        plan, _ = route_query("test query", cfg=cfg)

    assert plan.title_hint == "test title hint", (  # type: ignore[attr-defined]
        f"D-21: expected plan.title_hint='test title hint', got {plan.title_hint!r}"  # type: ignore[attr-defined]
    )


def test_title_hint_truncated_to_80_chars() -> None:
    """RED — D-22: title_hint hard-stops at 80 chars on the server.

    The LLM may emit longer; the routing-plan layer truncates so any caller
    consumes a bounded string.
    """
    pytest.importorskip("src.query.router")

    from src.query.router import route_query  # noqa: PLC0415

    long_hint = "a" * 200
    fake_payload = (
        '{"intent": "factoid", "entities": [], "fire": ["milvus"], '
        '"tool_payloads": {"milvus": ["test"]}, '
        f'"title_hint": "{long_hint}"}}'
    )
    fake_result = _build_fake_call_result(fake_payload)
    cfg = _build_cfg_with_title_hint(emit_title_hint=True)

    with patch("src.query.router.call_text", return_value=fake_result):
        plan, _ = route_query("test", cfg=cfg)

    assert plan.title_hint is not None  # type: ignore[attr-defined]
    assert len(plan.title_hint) <= 80, (  # type: ignore[attr-defined]
        f"D-22: title_hint must be truncated to <=80 chars; got len={len(plan.title_hint)}"  # type: ignore[attr-defined]
    )


def test_existing_decompose_calls_pass_without_title_hint() -> None:
    """RED — backward compat: callers that pass no cfg still parse responses
    that omit title_hint.

    Mirrors the bit-identical guarantee that already covers hyde_doc/stepback:
    when the emit flag is off, the response without the field still produces a
    valid RoutingPlan with title_hint = None.
    """
    pytest.importorskip("src.query.router")

    from src.query.router import route_query  # noqa: PLC0415

    fake_payload = (
        '{"intent": "factoid", "entities": [], "fire": ["milvus"], '
        '"tool_payloads": {"milvus": ["q"]}}'
    )
    fake_result = _build_fake_call_result(fake_payload)

    with patch("src.query.router.call_text", return_value=fake_result):
        plan, _ = route_query("test")

    assert plan.title_hint is None, (  # type: ignore[attr-defined]
        f"backward compat: omitted title_hint should default to None, got {plan.title_hint!r}"  # type: ignore[attr-defined]
    )
