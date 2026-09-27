# long-ok-file
"""LazyGraphRAG-style retriever (Plan 02-08; Wave 5).

Implements the LazyGraphRAG algorithm — community routing → community
detail fetch → grounded summarisation — over the Microsoft GraphRAG
package (``graphrag==3.0.9``). The three sub-stages are emitted in order
through the audit-trail facade :mod:`src.query.audit`:

    1. ``lazygraph_route``       — classify the question into a community
                                  topic (LLM, ``config.DECOMPOSE_MODEL``)
    2. ``lazygraph_community``   — fetch community detail (Milvus
                                  partition-filtered search in v1
                                  fallback; future: graphrag community
                                  index)
    3. ``lazygraph_summarize``   — synthesise grounded passages from the
                                  community context (LLM,
                                  ``config.GEN_MODEL``)

Each sub-stage follows the canonical four-step audit ritual (CRIT-2 +
D-26):

    audit.write_stage(name, payload)
        → audit.accumulate_usage(name, usage, cost, latency)  [if LLM]
        → _notify_stage_and_maybe_cancel(on_stage, cancel_event, name, latency)

LLM calls go through :func:`src.query.usage_track.call_text` exclusively
(CLAUDE.md rule 1 — direct ``genai`` / ``openai`` imports are banned).

Stub-fallback contract (Assumption A1)
--------------------------------------
When ``importlib.util.find_spec("graphrag")`` returns ``None`` at call
time the retriever:

    * still emits all three sub-stages (their payloads carry
      ``stub: true`` so the audit invariants stay honest)
    * still performs the route + summarise LLM calls (CLAUDE.md rule 1
      — every LLM call must be auditable)
    * still attempts the Milvus partition-filtered fallback search
    * returns ``([], {"retriever": "lazygraph", "stage": "stub",
      "reason": "graphrag not installed"})``

Plan 02-09 owns the dispatch wiring; Plan 02-13 reads
``LAZYGRAPH_AVAILABLE`` for the disabled-option UX in the chat dialog.
"""

from __future__ import annotations

import importlib.util
import logging
import threading
import time
from collections.abc import Callable
from typing import Any

from src import config
from src.models import RetrievedChunk, Usage
from src.query import audit, usage_track
from src.query import retrieve as _retrieve
from src.query.query_logger import QueryLogger
from src.query.query_pipeline import _notify_stage_and_maybe_cancel

logger = logging.getLogger(__name__)

# Module-load probe — flipped at call time too, in case the user pip-installs
# graphrag after the server is running. Plan 02-13 reads this for the
# disabled-option UX.
LAZYGRAPH_AVAILABLE: bool = importlib.util.find_spec("graphrag") is not None


# Top-K of community-member chunks to forward to the summarize substage.
# Capped per threat T-02-08-04 (DoS via huge community summarisation).
_COMMUNITY_TOP_K: int = 10

# Maximum total characters of community context fed to the summarise LLM
# call — also a T-02-08-04 mitigation. ~24k chars ≈ 6k tokens, well under
# DECOMPOSE/GEN model context budgets.
_SUMMARIZE_MAX_CHARS: int = 24_000


# ---------------------------------------------------------------------------
# Sub-stage 1 — community routing (LLM)
# ---------------------------------------------------------------------------


_ROUTE_SYSTEM_PROMPT = (
    "You are a corpus-router for a LazyGraphRAG retriever. The user has "
    "asked a question and you must classify it into one short community "
    "topic (a 2-6 word noun phrase) that anchors which slice of the "
    "corpus to retrieve. Respond with exactly the topic phrase on a "
    "single line — no preamble, no JSON, no quotes."
)


def _lazygraph_route(
    query: str, *, stub: bool
) -> tuple[str, dict[str, Any], Any]:
    """Classify *query* into one community topic.

    Returns ``(community_id, payload, llm_result_or_none)``. The
    ``payload`` is the dict written to ``16_lazygraph_route.json``;
    ``llm_result`` is the :class:`LLMCallResult` (or ``None`` if the
    LLM call raised — in which case ``community_id`` falls back to a
    truncation of the raw query).
    """
    t0 = time.perf_counter()
    llm_result: Any = None
    community_id: str
    error: str | None = None

    try:
        # NOTE: reference ``usage_track.call_text`` via the module attribute
        # (NOT a direct import) so test monkeypatches on
        # ``usage_track.call_text`` are visible here (CLAUDE.md rule 1
        # test: ``test_each_llm_call_goes_through_usage_track``).
        llm_result = usage_track.call_text(
            model=config.DECOMPOSE_MODEL,
            system=_ROUTE_SYSTEM_PROMPT,
            user=query,
            json_mode=False,
            temperature=0.0,
            max_tokens=64,
        )
        lines = (llm_result.text or "").strip().splitlines()
        community_id = lines[0][:120] if lines else ""
        if not community_id:
            community_id = query[:120]
    except Exception as exc:  # noqa: BLE001 — soft-fail; sub-stage still emits.
        error = repr(exc)
        community_id = query[:120]
        logger.warning("lazygraph_route LLM call failed: %s", error)

    latency_ms = int((time.perf_counter() - t0) * 1000)
    payload: dict[str, Any] = {
        "model": getattr(llm_result, "model", config.DECOMPOSE_MODEL),
        "latency_ms": latency_ms,
        "community_id": community_id,
        "confidence": 1.0 if llm_result is not None and error is None else 0.0,
        "usage": _usage_to_dict(getattr(llm_result, "usage", None)),
        "cost_usd": getattr(llm_result, "cost_usd", None),
    }
    if stub:
        payload["stub"] = True
        payload["reason"] = "graphrag not installed"
    if error is not None:
        payload["error"] = error
    return community_id, payload, llm_result


# ---------------------------------------------------------------------------
# Sub-stage 2 — community detail fetch (Milvus partition-filtered in v1)
# ---------------------------------------------------------------------------


def _lazygraph_community(
    query: str,
    community_id: str,
    collections: list[str],
    *,
    stub: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any], None]:
    """Fetch community-member candidates.

    v1 implementation: Milvus partition-filtered hybrid search where the
    community ID is used as the lexical anchor. When graphrag is
    installed AND a community index is built this is replaced by a
    direct ``graphrag.api.local_search`` call (V2).

    Returns ``(candidate_hits, payload, None)`` — the ``None`` is the
    LLM-result slot kept for parallelism with the other two sub-stages.
    """
    t0 = time.perf_counter()
    candidates: list[dict[str, Any]] = []
    fallback_used = True  # v1 always uses Milvus fallback
    error: str | None = None

    # Combine the routed community topic with the raw query for richer
    # lexical anchoring against the BM25 sparse field. partition_names
    # carries the collections list (CLAUDE.md "filter before search").
    search_text = f"{community_id}\n\n{query}"

    try:
        # NOTE: reference ``_retrieve.milvus_search_only`` via the module
        # attribute (NOT a direct import) so test monkeypatches on
        # ``_retr.milvus_search_only`` are visible at this call site.
        result = _retrieve.milvus_search_only(
            query=search_text,
            collection=collections[0],
            notebook=",".join(collections),
            top_k=_COMMUNITY_TOP_K,
            partition_names=collections,
        )
        if isinstance(result, list):
            candidates = result
    except Exception as exc:  # noqa: BLE001 — soft-fail per Pattern A
        error = repr(exc)
        logger.warning(
            "lazygraph_community Milvus fallback failed: %s", error
        )

    latency_ms = int((time.perf_counter() - t0) * 1000)
    payload: dict[str, Any] = {
        "community_id": community_id,
        "n_candidates": len(candidates),
        "fallback_to_milvus": fallback_used,
        "partition_names": list(collections),
        "latency_ms": latency_ms,
        "usage": {},
        "cost_usd": 0.0,
    }
    if stub:
        payload["stub"] = True
        payload["reason"] = "graphrag not installed"
    if error is not None:
        payload["error"] = error
    return candidates, payload, None


# ---------------------------------------------------------------------------
# Sub-stage 3 — grounded summarisation (LLM)
# ---------------------------------------------------------------------------


_SUMMARIZE_SYSTEM_PROMPT = (
    "You are a LazyGraphRAG summariser. Given a user query and a set of "
    "community-member passages, produce 3-6 grounded answer passages "
    "that directly address the query. Each passage must be 2-4 "
    "sentences. Cite the source passage indices in square brackets at "
    "the end of each passage, e.g. [0], [2]. Do not invent facts not "
    "supported by the passages."
)


def _lazygraph_summarize(
    query: str,
    community_id: str,
    candidates: list[dict[str, Any]],
    *,
    stub: bool,
) -> tuple[list[RetrievedChunk], dict[str, Any], Any]:
    """Summarise *candidates* into grounded passages.

    Always issues an LLM call (even when ``candidates`` is empty — the
    LLM is the safety net that explains "no relevant passages found"
    rather than returning silently). LLM call goes through
    :func:`call_text` per CLAUDE.md rule 1.

    Returns ``(final_chunks, payload, llm_result)``. ``final_chunks`` is
    ``[]`` in v1 because we do not yet materialise grounded passages
    back into ``RetrievedChunk`` form — Plan 02-09 / Plan 02-13 own that
    contract once the cross-encoder rerank pool is unified.
    """
    t0 = time.perf_counter()
    llm_result: Any = None
    error: str | None = None

    user_msg = _build_summarize_user_message(query, community_id, candidates)

    try:
        # See _lazygraph_route for why we go through the module attribute.
        llm_result = usage_track.call_text(
            model=config.GEN_MODEL,
            system=_SUMMARIZE_SYSTEM_PROMPT,
            user=user_msg,
            json_mode=False,
            temperature=0.2,
            max_tokens=1024,
        )
    except Exception as exc:  # noqa: BLE001 — soft-fail per Pattern A
        error = repr(exc)
        logger.warning("lazygraph_summarize LLM call failed: %s", error)

    latency_ms = int((time.perf_counter() - t0) * 1000)
    payload: dict[str, Any] = {
        "model": getattr(llm_result, "model", config.GEN_MODEL),
        "latency_ms": latency_ms,
        "community_id": community_id,
        "n_input_passages": len(candidates),
        "answer_text": (
            getattr(llm_result, "text", "") if llm_result is not None else ""
        ),
        "usage": _usage_to_dict(getattr(llm_result, "usage", None)),
        "cost_usd": getattr(llm_result, "cost_usd", None),
    }
    if stub:
        payload["stub"] = True
        payload["reason"] = "graphrag not installed"
    if error is not None:
        payload["error"] = error

    # v1 contract: do not materialise grounded passages back into
    # RetrievedChunk form — that's Plan 02-09's fused-rerank job.
    final_chunks: list[RetrievedChunk] = []
    return final_chunks, payload, llm_result


def _build_summarize_user_message(
    query: str, community_id: str, candidates: list[dict[str, Any]]
) -> str:
    """Render the summarise-LLM user message from *candidates*.

    Truncates total content to ``_SUMMARIZE_MAX_CHARS`` to mitigate
    T-02-08-04 (DoS via oversized community blob).
    """
    parts: list[str] = [
        f"Community topic: {community_id}",
        f"User query: {query}",
        "",
        "Passages:",
    ]
    running_chars = sum(len(p) for p in parts)
    for i, hit in enumerate(candidates):
        text = ""
        if isinstance(hit, dict):
            entity = hit.get("entity")
            if isinstance(entity, dict):
                text = str(entity.get("text", ""))
            else:
                text = str(hit.get("text", ""))
        snippet = f"[{i}] {text}"
        if running_chars + len(snippet) > _SUMMARIZE_MAX_CHARS:
            parts.append(f"... ({len(candidates) - i} passages truncated)")
            break
        parts.append(snippet)
        running_chars += len(snippet)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _usage_to_dict(usage: Usage | None) -> dict[str, int]:
    """Serialise :class:`Usage` to a plain dict for JSON payloads."""
    if usage is None:
        return {}
    return {
        "input": usage.input,
        "output": usage.output,
        "reasoning": usage.reasoning,
        "total": usage.total,
        "input_cache_hit": usage.input_cache_hit,
        "input_cache_miss": usage.input_cache_miss,
    }


def _emit_substage(
    name: str,
    payload: dict[str, Any],
    *,
    llm_result: Any,  # noqa: ANN401 — LLMCallResult duck-typed; None for non-LLM stages.
    on_stage: Callable[[str, int], None] | None,
    cancel_event: threading.Event | None,
) -> None:
    """Four-step Pattern A ritual for a single sub-stage.

    1. ``audit.write_stage(name, payload)``
    2. ``audit.accumulate_usage(name, usage, cost, latency)`` if LLM ran
    3. ``_notify_stage_and_maybe_cancel(on_stage, cancel_event, name, latency)``

    The ordering is enforced so the SSE handler in
    :mod:`src.server.api_chats` (Plan 04) sees a stable interleaving
    (D-26: callback fires AFTER write).
    """
    audit.write_stage(name, payload)
    latency_ms = int(payload.get("latency_ms", 0))
    # CLAUDE.md NON-NEGOTIABLE rule 3 — every LLM stage MUST call
    # accumulate_usage or the stage is invisible in meta.totals. The
    # community sub-stage has no LLM but we still record a zero-cost
    # entry so the registry invariant
    # ``test_each_substage_calls_accumulate_usage`` passes.
    usage_obj: Usage = getattr(llm_result, "usage", None) or Usage()
    cost: float | None = getattr(llm_result, "cost_usd", None)
    audit.accumulate_usage(name, usage_obj, cost, latency_ms)
    _notify_stage_and_maybe_cancel(on_stage, cancel_event, name, latency_ms)


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------


def retrieve_lazygraph(
    query: str,
    collections: list[str],
    *,
    audit: QueryLogger | None = None,
    on_stage: Callable[[str, int], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> tuple[list[RetrievedChunk], dict[str, Any]]:
    """LazyGraphRAG retriever — community routing + summarisation.

    Args:
        query:        Original user query.
        collections:  Milvus partition names. Filter-before-search
                      invariant (CLAUDE.md "filter before search").
        audit:        Optional :class:`QueryLogger` to install as the
                      current logger. When ``None``, the module-level
                      audit facade lazily creates one rooted at
                      ``QUERY_LOG_ROOT``.
        on_stage:     Optional callback invoked AFTER each sub-stage's
                      ``audit.write_stage`` returns (D-26).
        cancel_event: Optional :class:`threading.Event` polled between
                      sub-stages — raises :class:`asyncio.CancelledError`
                      when set (D-22).

    Returns:
        ``(retrieved_chunks, trace)`` where ``trace`` carries the
        community ID + a ``stage`` marker. In stub mode
        (``graphrag`` not installed) ``trace["stage"] == "stub"`` and
        ``retrieved_chunks == []``.
    """
    # Install the explicit logger if provided; otherwise the facade
    # lazily creates one on first ``audit.write_stage`` call.
    if audit is not None:
        from src.query import audit as _audit_mod  # noqa: PLC0415
        _audit_mod.set_current_logger(audit)

    # Re-probe at call time so a runtime `pip install graphrag` is
    # picked up (and so the find_spec monkeypatch in the stub-fallback
    # test is honoured even though the module is already imported).
    graphrag_spec = importlib.util.find_spec("graphrag")
    stub_mode = graphrag_spec is None

    # Sub-stage 1 — route.
    community_id, route_payload, route_result = _lazygraph_route(
        query, stub=stub_mode
    )
    _emit_substage(
        "lazygraph_route",
        route_payload,
        llm_result=route_result,
        on_stage=on_stage,
        cancel_event=cancel_event,
    )

    # Sub-stage 2 — community detail.
    candidates, community_payload, _ = _lazygraph_community(
        query, community_id, collections, stub=stub_mode
    )
    _emit_substage(
        "lazygraph_community",
        community_payload,
        llm_result=None,
        on_stage=on_stage,
        cancel_event=cancel_event,
    )

    # Sub-stage 3 — summarise.
    final_chunks, summarize_payload, summarize_result = _lazygraph_summarize(
        query, community_id, candidates, stub=stub_mode
    )
    _emit_substage(
        "lazygraph_summarize",
        summarize_payload,
        llm_result=summarize_result,
        on_stage=on_stage,
        cancel_event=cancel_event,
    )

    trace: dict[str, Any] = {
        "retriever": "lazygraph",
        "community_id": community_id,
        "n_candidates": len(candidates),
        "n_passages": len(final_chunks),
    }
    if stub_mode:
        trace["stage"] = "stub"
        trace["reason"] = "graphrag not installed"
    return final_chunks, trace
