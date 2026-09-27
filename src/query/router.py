"""Decomposition router: turns a user question into per-tool retrieval payloads.

Phase 1 wires only the ``milvus`` tool. The router emits a routing plan so that
Phase 2 (PaperQA2 / HippoRAG 2 / LazyGraphRAG) can be added by extending the
``ToolName`` literal and the prompt manifest, without changing the call sites
in the query pipeline.

Pattern: SSRAG-style vocabulary-aware decomposition. Single LLM call to
``config.DECOMPOSE_MODEL`` (deepseek-v4-flash by default) returns:

    {
      "intent": "<class>",
      "entities": [...],
      "fire": ["milvus"],
      "tool_payloads": {"milvus": ["<sub-query 1>", ...]}
    }

Extended fields (activated by RewriterConfig — single global config from
``REWRITER_EMIT_*`` env vars):
    hyde_doc, stepback, disambiguation, filters

Failure mode is fail-open: if anything goes wrong, fall back to a single-tool
plan that retrieves on the original query.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from src import config
from src.config.rewriter_configs import RewriterConfig, _baseline_rewriter_config
from src.query.usage_track import LLMCallResult, call_text

logger = logging.getLogger(__name__)


ToolName = Literal["milvus", "paperqa", "hipporag", "lazygraph"]
IntentClass = Literal[
    "factoid",
    "definitional",
    "multi_hop",
    "comparative",
    "literature_synthesis",
    "sensemaking",
    "computational",
    "unknown",
]


# ---------------------------------------------------------------------------
# Pydantic schema for provider-side structured output (step 1 + 8)
# ---------------------------------------------------------------------------


class Disambig(BaseModel):
    """One candidate interpretation for disambiguation."""

    label: str
    clarified_query: str


class RouterOutputSchema(BaseModel):
    """Full router output schema — used for provider-side JSON schema enforcement.

    All new optional fields default to None/[] so the schema is backward-
    compatible: providers that emit only the legacy fields will still parse.
    """

    intent: str  # IntentClass string — kept as str to allow graceful unknown fallback
    entities: list[str] = []
    fire: list[str] = ["milvus"]
    tool_payloads: dict[str, list[str]]
    use_mmr: bool | None = None
    use_mmr_reason: str = ""
    hyde_doc: str | None = None
    stepback: str | None = None
    disambiguation: list[Disambig] = []
    filters: dict[str, Any] = {}
    # Phase 2 D-21 / CHAT-10 — concise auto-name hint for the first turn of
    # an "Untitled chat". Server-side truncation to 80 chars applied in
    # ``route_query`` after parsing. None when ``RewriterConfig.emit_title_hint``
    # is False or the LLM declined to emit a label (e.g. vague query).
    title_hint: str | None = None


# ---------------------------------------------------------------------------
# RoutingPlan dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ToolCall:
    """One node in the routing plan — a tool plus its tool-specific payload.

    Phase 1: ``tool == "milvus"`` and ``sub_queries`` is the only payload field.
    Phase 2: PaperQA2 will use ``payload_dict``, HippoRAG will use
    ``entity_pairs``, LazyGraphRAG will use ``topic_phrase``. Add fields here as
    each retriever lands; do not break the existing shape.
    """

    tool: ToolName
    sub_queries: list[str] = field(default_factory=list)


@dataclass
class RoutingPlan:
    user_query: str
    intent: IntentClass
    entities: list[str]
    nodes: list[ToolCall]
    # LLM-emitted retrieval policy override. None = fall through to intent
    # baseline in policies.policy_for_intent. Reasoning recorded for audit.
    # See prompts/router/system.md for the literature-backed criteria.
    use_mmr: bool | None = None
    use_mmr_reason: str = ""
    # Extended fields — activated by RewriterConfig (emit_* flags).
    # Default None/[] preserves bit-identical behavior when flags are off.
    hyde_doc: str | None = None
    stepback: str | None = None
    disambiguation: list[Disambig] = field(default_factory=list)
    filters: dict[str, Any] = field(default_factory=dict)
    # Phase 2 D-21 / CHAT-10 — concise auto-name hint produced by the
    # decompose call when ``RewriterConfig.emit_title_hint`` is True.
    # Hard-capped at 80 chars server-side after parsing.
    title_hint: str | None = None


# ---------------------------------------------------------------------------
# Prompt loading + building
# ---------------------------------------------------------------------------


_SYSTEM_PROMPT_PATH = (
    Path(__file__).resolve().parent.parent.parent / "prompts" / "router" / "system.md"
)
_SECTIONS_DIR = (
    Path(__file__).resolve().parent.parent.parent / "prompts" / "router" / "sections"
)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _load_system_prompt() -> str:
    return _SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")


def _build_prompt(cfg: RewriterConfig | None = None) -> str:
    """Build the full system prompt by concatenating base + enabled section files.

    The base ``system.md`` is always included. Conditional sections are appended
    only when the corresponding ``emit_*`` flag is ``True`` in ``cfg``.

    The ``_self_gate.md`` section is appended ONLY when at least one optional
    section was enabled — it tells the LLM which optional fields to emit. With
    all flags off, no self-gate is needed and the prompt is bit-identical to
    the pre-refactor ``system.md``.

    Args:
        cfg: ``RewriterConfig`` with emit flags. If None, reads baseline config.

    Returns:
        Full system prompt string.
    """

    if cfg is None:
        cfg = _baseline_rewriter_config()

    base = _load_system_prompt()
    sections: list[str] = []

    if cfg.emit_hyde:
        hyde_path = _SECTIONS_DIR / "_hyde.md"
        if hyde_path.exists():
            sections.append(hyde_path.read_text(encoding="utf-8"))

    if cfg.emit_stepback:
        sb_path = _SECTIONS_DIR / "_stepback.md"
        if sb_path.exists():
            sections.append(sb_path.read_text(encoding="utf-8"))

    if cfg.emit_disambiguation:
        dis_path = _SECTIONS_DIR / "_disambiguation.md"
        if dis_path.exists():
            sections.append(dis_path.read_text(encoding="utf-8"))

    if cfg.emit_filters:
        filt_path = _SECTIONS_DIR / "_filters.md"
        if filt_path.exists():
            sections.append(filt_path.read_text(encoding="utf-8"))

    if cfg.emit_title_hint:
        th_path = _SECTIONS_DIR / "_title_hint.md"
        if th_path.exists():
            sections.append(th_path.read_text(encoding="utf-8"))

    # Self-gate only when at least one optional section is active — otherwise
    # we'd leak section-marker tokens (hyde_doc, stepback, ...) into the
    # baseline prompt and break the bit-identical guarantee.
    if sections:
        gate_path = _SECTIONS_DIR / "_self_gate.md"
        if gate_path.exists():
            sections.append(gate_path.read_text(encoding="utf-8"))
        return base + "\n\n---\n\n" + "\n\n---\n\n".join(sections)
    return base


# ---------------------------------------------------------------------------
# Fallback plan
# ---------------------------------------------------------------------------


def _fallback_plan(query: str, reason: str) -> RoutingPlan:
    logger.warning("router: falling back to original query (%s)", reason)
    return RoutingPlan(
        user_query=query,
        intent="unknown",
        entities=[],
        nodes=[ToolCall(tool="milvus", sub_queries=[query])],
    )


# ---------------------------------------------------------------------------
# Public route_query
# ---------------------------------------------------------------------------


def route_query(
    query: str,
    available_tools: list[ToolName] | None = None,
    cfg: RewriterConfig | None = None,
    *,
    emit_title_hint: bool = False,
) -> tuple[RoutingPlan, LLMCallResult | None]:
    """Plan how to fan *query* out across *available_tools*.

    Phase 1 always returns a plan with one ``milvus`` ``ToolCall``. The
    ``available_tools`` argument exists so Phase 2 can opt new tools in
    without changing this signature.

    Never raises — falls back to ``([query], None)`` on any error.

    Args:
        query: The user's question.
        available_tools: Tools to route to (default ``["milvus"]``).
        cfg: ``RewriterConfig`` controlling which optional fields the LLM emits.
             If None, uses ``_baseline_rewriter_config()`` (env-knob defaults).
        emit_title_hint: Phase 2 D-21 / CHAT-10. When True, the router prompt
             includes the ``_title_hint.md`` section asking the LLM to emit a
             concise topic label (used for auto-naming the chat). When False
             (default), behavior is bit-identical to Phase 1. The
             ``api_chats._stream_assistant_turn`` hook flips this to True only
             for the first assistant turn of an ``Untitled chat`` — every
             other call site stays opt-out.

    Returns:
        (routing_plan, llm_result) — the logger uses the second element to
        record usage/latency/cost; callers that only care about the plan
        can ignore it.
    """

    available_tools = available_tools or ["milvus"]

    if "milvus" not in available_tools:
        return _fallback_plan(query, f"no supported tool in {available_tools}"), None

    if cfg is None:
        cfg = _baseline_rewriter_config()

    # Per-call opt-in for title_hint — flip the flag on the cfg without
    # mutating the frozen instance (dataclasses.replace returns a new one).
    if emit_title_hint and not cfg.emit_title_hint:
        import dataclasses as _dc  # noqa: PLC0415

        cfg = _dc.replace(cfg, emit_title_hint=True)

    system_prompt = _build_prompt(cfg)

    try:
        result = call_text(
            config.DECOMPOSE_MODEL,
            system_prompt,
            f"Question: {query}",
            json_mode=False,
            temperature=0.0,
            max_tokens=cfg.max_output_tokens,
            schema=RouterOutputSchema,
        )
        raw = result.text
    except Exception as exc:
        return _fallback_plan(query, f"router LLM call failed: {exc}"), None

    # Parse: try schema validation first, then regex fallback, then _fallback_plan.
    # Also keep the raw dict for fields that need pre-coercion access (use_mmr, use_mmr_reason)
    # since Pydantic coerces "yes"→True and int→str which breaks the old behavior contracts.
    raw_dict: dict[str, Any] | None = None
    parsed_schema: RouterOutputSchema | None = None

    def _try_parse(text: str) -> tuple[RouterOutputSchema | None, dict[str, Any] | None]:
        try:
            d: dict[str, Any] = json.loads(text)
            s = _salvage_from_dict(d)
            return s, d
        except Exception:
            return None, None

    try:
        raw_dict = json.loads(raw)
        parsed_schema = _salvage_from_dict(raw_dict)
    except Exception:
        match = _JSON_OBJECT_RE.search(raw)
        if match:
            parsed_schema, raw_dict = _try_parse(match.group(0))

    if parsed_schema is None:
        return _fallback_plan(query, f"no JSON in router output: {raw[:200]!r}"), result

    # Validate milvus payload
    milvus_subs = (parsed_schema.tool_payloads or {}).get("milvus") or []
    if not isinstance(milvus_subs, list) or not all(isinstance(s, str) for s in milvus_subs):
        return _fallback_plan(query, f"malformed milvus payload: {milvus_subs!r}"), result

    cleaned = [s.strip() for s in milvus_subs if s.strip()]
    if not cleaned:
        return _fallback_plan(query, "router returned empty milvus payload"), result

    capped = cleaned[: config.DECOMPOSE_MAX_SUBQUERIES]

    intent_raw = parsed_schema.intent
    intent: IntentClass = intent_raw if isinstance(intent_raw, str) else "unknown"  # type: ignore[assignment]

    entities = [e for e in parsed_schema.entities if isinstance(e, str)]

    # Extract use_mmr from the raw dict (pre-Pydantic-coercion) to preserve the
    # legacy contract: only native JSON bool true/false maps to bool; strings like
    # "yes" coerce to None. raw_dict is always available here since parsed_schema is not None.
    raw_use_mmr_val = (raw_dict or {}).get("use_mmr") if raw_dict is not None else None
    use_mmr: bool | None = raw_use_mmr_val if isinstance(raw_use_mmr_val, bool) else None
    raw_reason_val = (raw_dict or {}).get("use_mmr_reason", "")
    use_mmr_reason = raw_reason_val.strip()[:200] if isinstance(raw_reason_val, str) else ""

    # D-22 — hard cap title_hint at 80 chars server-side. The LLM is asked
    # for ≤60; this is a safety belt for adversarial outputs (Pitfall: a
    # 200-char overrun would blow up sidebar rendering).
    title_hint_raw = (
        parsed_schema.title_hint if isinstance(parsed_schema.title_hint, str) else None
    )
    title_hint = title_hint_raw.strip()[:80] if title_hint_raw is not None else None
    if title_hint == "":
        title_hint = None

    plan = RoutingPlan(
        user_query=query,
        intent=intent,
        entities=entities,
        nodes=[ToolCall(tool="milvus", sub_queries=capped)],
        use_mmr=use_mmr,
        use_mmr_reason=use_mmr_reason,
        hyde_doc=parsed_schema.hyde_doc if isinstance(parsed_schema.hyde_doc, str) else None,
        stepback=parsed_schema.stepback if isinstance(parsed_schema.stepback, str) else None,
        disambiguation=list(parsed_schema.disambiguation) if parsed_schema.disambiguation else [],
        filters=dict(parsed_schema.filters) if parsed_schema.filters else {},
        title_hint=title_hint,
    )
    logger.info(
        "router: intent=%s entities=%d milvus_subs=%d use_mmr=%s hyde=%s stepback=%s",
        plan.intent,
        len(plan.entities),
        len(capped),
        plan.use_mmr,
        plan.hyde_doc is not None,
        plan.stepback is not None,
    )
    return plan, result


def _salvage_from_dict(parsed: dict[str, Any]) -> RouterOutputSchema | None:
    """Attempt to salvage a RouterOutputSchema from a raw parsed dict.

    Keeps valid fields, defaults the rest. Returns None if the result is
    unusable (e.g. no milvus payload).
    """
    try:
        return RouterOutputSchema.model_validate(parsed)
    except Exception:
        pass
    # Manual salvage: try to build with whatever keys are present
    try:
        tool_payloads = parsed.get("tool_payloads") or {}
        if not tool_payloads:
            return None
        return RouterOutputSchema(
            intent=str(parsed.get("intent", "unknown")),
            entities=parsed.get("entities") or [],
            fire=parsed.get("fire") or ["milvus"],
            tool_payloads=tool_payloads,
            use_mmr=parsed.get("use_mmr"),
            use_mmr_reason=str(parsed.get("use_mmr_reason") or ""),
            hyde_doc=parsed.get("hyde_doc"),
            stepback=parsed.get("stepback"),
            disambiguation=parsed.get("disambiguation") or [],
            filters=parsed.get("filters") or {},
            title_hint=parsed.get("title_hint"),
        )
    except Exception:
        return None
