# long-ok-file
"""Unified query pipeline shared by the CLI and the eval harness.

Pipeline:
    decompose (optional) → step-back (optional flag) → multi-sub-query Milvus
    hybrid_search → batched cross-encoder rerank → parent lookup → generate
    → CRAG-lite groundedness check (optional flag, retries once on low score)

Both ``src.generate._cli`` and ``src.evaluate.evaluate_notebook`` should call
``run_query_pipeline`` so they exercise identical code paths. Eval scores then
reflect the real production pipeline including any flag-gated features.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src import config
from src.config.models import compute_cost_usd
from src.models import RAGResponse, RetrievedChunk
from src.query.query_logger import QueryLogger

logger = logging.getLogger(__name__)


def _notify_stage_and_maybe_cancel(
    on_stage: Callable[[str, int], None] | None,
    cancel_event: threading.Event | None,
    stage_name: str,
    latency_ms: int,
) -> None:
    """Fire the optional ``on_stage`` callback, then poll cancellation.

    Contract (CONTEXT.md D-26 + D-22):
      * MUST be called AFTER ``audit.write_stage(name, payload)`` returns
        successfully — never before, never instead of.
      * NEVER raises from the callback (try/except wrap).
      * Raises ``asyncio.CancelledError`` when ``cancel_event.is_set()``.

    The SSE handler in :mod:`src.server.api_chats` (Plan 04) supplies both
    arguments; CLI / eval callers pass neither so the pipeline behavior is
    unchanged for them.
    """
    if on_stage is not None:
        try:
            on_stage(stage_name, int(latency_ms))
        except Exception as exc:  # noqa: BLE001 — D-26 callback isolation
            logger.warning("on_stage callback raised: %r", exc)
    if cancel_event is not None and cancel_event.is_set():
        raise asyncio.CancelledError("user-requested cancel")

# RRF k-constant — controls how aggressively the score decays with rank.
# k=60 is the standard from Cormack-Clarke-Buettcher 2009 and used by every
# major hybrid-search implementation (Elastic, Milvus, OpenSearch, RAG-Fusion).
_RRF_K = 60


# ---------------------------------------------------------------------------
# Extended-router pipeline helpers (steps 10-11)
# ---------------------------------------------------------------------------


def _extract_sub_queries_from_plan(plan: Any) -> list[str]:  # noqa: ANN401
    """Extract milvus sub-queries from a RoutingPlan, appending hyde_doc if present.

    Args:
        plan: A ``RoutingPlan`` instance.

    Returns:
        List of sub-query strings. If ``plan.hyde_doc`` is non-null, it is
        appended as an additional sub-query (HyDE-based vocabulary expansion).
    """
    milvus_node = next((n for n in plan.nodes if n.tool == "milvus"), None)
    sub_queries: list[str] = milvus_node.sub_queries[:] if milvus_node else [plan.user_query]

    if plan.hyde_doc and isinstance(plan.hyde_doc, str):
        sub_queries.append(plan.hyde_doc)

    return sub_queries


def _should_call_legacy_stepback(plan: Any) -> bool:  # noqa: ANN401
    """Return True when the legacy step_back_query call should still run.

    Single-cycle deprecation: if the router already emitted a stepback string,
    skip the legacy call. Only call legacy when plan.stepback is None.

    Args:
        plan: A ``RoutingPlan`` instance.
    """
    return plan.stepback is None


def _accumulate_crag_call(
    audit: QueryLogger, stage: str, result: Any  # noqa: ANN401  — LLMCallResult | None
) -> int:
    """Feed one CRAG-stage LLMCallResult into meta.totals; return its latency_ms.

    A ``None`` result means the underlying LLM call short-circuited (empty
    inputs / library unavailable) — nothing to accumulate, zero latency.
    """
    if result is None:
        return 0
    audit.accumulate_usage(stage, result.usage, result.cost_usd, result.latency_ms)
    return int(result.latency_ms)


def _build_decompose_payload(
    plan: Any,  # noqa: ANN401
    sub_queries: list[str],
    latency_ms: int,
    llm_result: Any = None,  # noqa: ANN401  — LLMCallResult, kept Any to avoid heavy import
) -> dict[str, Any]:
    """Build the router audit payload dict for ``01_decompose.json``.

    Extends the legacy payload shape with the new optional fields from
    ``RoutingPlan``: ``hyde_doc``, ``stepback``, ``disambiguation``, ``filters``.

    Args:
        plan: A ``RoutingPlan`` instance.
        sub_queries: Effective sub-query list (after hyde_doc appended).
        latency_ms: Elapsed time for the router call.
        llm_result: Optional ``LLMCallResult`` from the router call. When non-None,
            ``usage`` and ``cost_usd`` are populated from it instead of being
            hardcoded to None. Per CLAUDE.md NON-NEGOTIABLE rule: usage MUST
            reach the audit payload.

    Returns:
        Payload dict ready for ``audit.write_stage("decompose", payload)``.
    """
    import dataclasses  # noqa: PLC0415

    usage_dict: dict[str, Any] | None = None
    cost_usd: float | None = None
    if llm_result is not None:
        usage_dict = dataclasses.asdict(llm_result.usage)
        cost_usd = llm_result.cost_usd

    return {
        "router": True,
        "intent": plan.intent,
        "entities": plan.entities,
        "sub_queries": sub_queries,
        "use_mmr": plan.use_mmr,
        "use_mmr_reason": plan.use_mmr_reason,
        "latency_ms": latency_ms,
        "model": config.DECOMPOSE_MODEL,
        "usage": usage_dict,
        "cost_usd": cost_usd,
        "parsed_subqueries": sub_queries,
        # Extended fields (None/[] when not activated)
        "hyde_doc": plan.hyde_doc,
        "stepback": plan.stepback,
        "disambiguation": [
            {"label": d.label, "clarified_query": d.clarified_query}
            for d in (plan.disambiguation or [])
        ],
        "filters": dict(plan.filters) if plan.filters else {},
    }


def _rrf_merge(
    per_sub_query_hits: list[list[dict[str, Any]]],
) -> list[tuple[dict[str, Any], float, list[int]]]:
    """Merge ranked Milvus hit lists across sub-queries via Reciprocal Rank Fusion.

    Implements canonical RAG-Fusion: each chunk's RRF score is the sum of
    1/(k+rank) across every sub-query list it appears in (k=_RRF_K=60).
    Chunks that surface in MULTIPLE sub-query lists get boosted naturally,
    no scale normalization required.

    Returns:
        List of ``(hit, rrf_score, source_list_indices)`` tuples sorted by
        rrf_score descending. ``source_list_indices`` records which
        sub-query lists (by index) the chunk appeared in — preserved for
        audit even though downstream cross-encoder scores against the
        original query, not the sub-queries.
    """
    scores: dict[str, float] = {}
    chunk_by_id: dict[str, dict[str, Any]] = {}
    sources: dict[str, list[int]] = {}
    for list_idx, hits in enumerate(per_sub_query_hits):
        for rank, hit in enumerate(hits, start=1):
            cid = hit["id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (_RRF_K + rank)
            chunk_by_id.setdefault(cid, hit)
            sources.setdefault(cid, []).append(list_idx)
    sorted_ids = sorted(scores, key=lambda c: scores[c], reverse=True)
    return [(chunk_by_id[c], scores[c], sources[c]) for c in sorted_ids]


def _retrieve(
    sub_queries: list[str],
    original_query: str,
    collection: str,
    notebook: str,
    top_k_retrieve: int,
    top_k_rerank: int,
    *,
    audit: QueryLogger | None = None,
    use_mmr: bool | None = None,
    mmr_lambda: float | None = None,
    mmr_pool_mult: int | None = None,
    score_threshold: float | None = None,
    max_children_per_parent: int | None = None,
    on_stage: Callable[[str, int], None] | None = None,
    cancel_event: threading.Event | None = None,
    partition_names: list[str] | None = None,
) -> tuple[list[RetrievedChunk], dict[str, Any]]:
    """Multi-sub-query Milvus → RRF merge → cross-encoder rerank → MMR.

    Implements the canonical RAG-Fusion pipeline (Raudaschl 2024):

    1. For each sub-query: Milvus hybrid search → top_k_retrieve hits
    2. RRF merge across sub-query lists → single ranked candidate pool
    3. Truncate to ``top_k_rerank * mmr_pool_mult`` candidates
    4. Cross-encoder rerank pairs (original_query, chunk) — NOT sub-queries
       — so the CE judges relevance against the user's actual question
       rather than an LLM-paraphrased rephrase
    5. Threshold + per-parent cap + MMR → final top_k_rerank chunks

    The previous implementation flattened all (sub_query, chunk) pairs into
    one big CE batch and let the CE rescore globally. That dropped the
    per-sub-query rank info and let whichever sub-query produced the
    cleanest CE matches dominate — a structural imbalance that
    cross-paper-synthesis queries (multi_hop, comparative) suffered from.
    RRF merge fixes the imbalance by giving every sub-query equal weight
    in the merge step; the CE then sharpens the final ordering.
    """
    from src.query.retrieve import batched_rerank_and_lookup, milvus_search_only  # noqa: PLC0415

    eff_mmr_pool_mult = (
        config.MMR_POOL_MULT if mmr_pool_mult is None else mmr_pool_mult
    )
    rrf_pool_size = top_k_rerank * eff_mmr_pool_mult

    sub_traces_summary: list[dict[str, Any]] = []
    sub_traces_audit: list[dict[str, Any]] = []
    per_list_hits: list[list[dict[str, Any]]] = []
    t0 = time.perf_counter()
    for sq in sub_queries:
        sq_t0 = time.perf_counter()
        # Multi-collection (D-03): when partition_names is set, push the
        # selected-collections filter down to Milvus. milvus_search_only loops
        # over each collection with the notebook IN-clause pinned to the same
        # set, so only the chosen collections are searched (filter-before-search).
        if partition_names is not None:
            hits = milvus_search_only(
                query=sq,
                collection=collection,
                notebook=notebook,
                top_k=top_k_retrieve,
                partition_names=partition_names,
            )
        else:
            hits = milvus_search_only(
                query=sq,
                collection=collection,
                notebook=notebook,
                top_k=top_k_retrieve,
            )
        sq_ms = int((time.perf_counter() - sq_t0) * 1000)
        per_list_hits.append(hits)
        sub_traces_summary.append(
            {
                "sub_query": sq,
                "n_candidates": len(hits),
                "milvus_latency_ms": sq_ms,
            }
        )
        if audit is not None:
            sub_traces_audit.append(
                {
                    "sub_query": sq,
                    "top_k": top_k_retrieve,
                    "partition": notebook,
                    "latency_ms": sq_ms,
                    "candidates": [
                        {
                            "id": h["id"],
                            "source_file": h["entity"]["source_file"],
                            "page": h["entity"]["page_number"],
                            "rrf_score": float(h["distance"]),
                        }
                        for h in hits
                    ],
                }
            )

    rrf_t0 = time.perf_counter()
    fused = _rrf_merge(per_list_hits)
    rrf_ms = int((time.perf_counter() - rrf_t0) * 1000)
    fused_truncated = fused[:rrf_pool_size]
    n_unique = len(fused)

    logger.info(
        "rrf: %d sub-queries → %d unique chunks → top %d for cross-encoder",
        len(sub_queries),
        n_unique,
        len(fused_truncated),
    )

    if audit is not None:
        audit.write_stage(
            "milvus",
            {
                "sub_queries": sub_traces_audit,
                "rrf_merge": {
                    "k_constant": _RRF_K,
                    "n_sub_queries": len(sub_queries),
                    "n_unique_after_merge": n_unique,
                    "n_truncated_to": len(fused_truncated),
                    "latency_ms": rrf_ms,
                    "top_post_rrf": [
                        {
                            "id": h["id"],
                            "source_file": h["entity"]["source_file"],
                            "rrf_score": float(s),
                            "appeared_in_sub_query_idx": src,
                        }
                        for h, s, src in fused_truncated[:30]
                    ],
                },
            },
        )
        _notify_stage_and_maybe_cancel(on_stage, cancel_event, "milvus", rrf_ms)

    pairs = [(original_query, h) for h, _, _ in fused_truncated]

    rerank_t0 = time.perf_counter()
    eff_score_threshold = (
        config.RERANK_SCORE_THRESHOLD if score_threshold is None else score_threshold
    )
    eff_max_children = (
        config.MAX_CHILDREN_PER_PARENT
        if max_children_per_parent is None
        else max_children_per_parent
    )
    retrieved, rerank_trace = batched_rerank_and_lookup(
        pairs,
        top_k=top_k_rerank,
        score_threshold=eff_score_threshold,
        max_children_per_parent=eff_max_children,
        use_mmr=use_mmr,
        mmr_lambda=mmr_lambda,
        mmr_pool_mult=mmr_pool_mult,
    )
    rerank_ms = int((time.perf_counter() - rerank_t0) * 1000)

    if audit is not None:
        from src.query.rerankers import get_reranker  # noqa: PLC0415

        try:
            rr_identity: dict[str, str | None] = get_reranker().identity()
        except Exception:
            rr_identity = {"backend": "unknown", "model": None, "endpoint": None}
        audit.write_stage(
            "rerank",
            {"reranker": rr_identity, "latency_ms": rerank_ms, **rerank_trace},
        )
        _notify_stage_and_maybe_cancel(on_stage, cancel_event, "rerank", rerank_ms)
        if "mmr" in rerank_trace:
            audit.write_stage("mmr", rerank_trace["mmr"])
            _notify_stage_and_maybe_cancel(
                on_stage,
                cancel_event,
                "mmr",
                int(rerank_trace["mmr"].get("latency_ms", 0)),
            )

    total_latency_ms = int((time.perf_counter() - t0) * 1000)
    trace = {
        "sub_queries": sub_queries,
        "sub_traces": sub_traces_summary,
        "rrf": {
            "n_unique_after_merge": n_unique,
            "n_truncated_to": len(fused_truncated),
            "latency_ms": rrf_ms,
        },
        "rerank_input_pairs": len(pairs),
        "rerank_survivors": rerank_trace,
        "rerank_latency_ms": rerank_ms,
        "total_latency_ms": total_latency_ms,
    }
    if audit is not None:
        audit.set_retrieval_latency(total_latency_ms)
    return retrieved, trace


def run_query_pipeline(
    query: str,
    collection: str,
    notebook: str,
    *,
    decompose: bool = True,
    use_router: bool = True,
    use_stepback: bool | None = None,
    use_crag: bool | None = None,
    top_k_retrieve: int | None = None,
    top_k_rerank: int | None = None,
    synthesize: bool = True,
    gen_model: str | None = None,
    on_stage: Callable[[str, int], None] | None = None,
    cancel_event: threading.Event | None = None,
    on_audit_dir: Callable[[Path], None] | None = None,
    retriever: str = "milvus",
    partition_names: list[str] | None = None,
    emit_title_hint: bool = False,
) -> tuple[RAGResponse | None, dict[str, Any]]:
    """End-to-end query pipeline. Returns (response_or_None_if_no_chunks, trace).

    Flag resolution order: explicit kwarg > config default.

    ``use_router`` (default True) swaps the legacy ``decompose_query`` for the
    SSRAG-style ``route_query`` planner. Set False to fall back to the old
    section-shaped decomposition for A/B comparison.

    ``synthesize`` (default True) controls whether the LLM synthesis + CRAG
    blocks run. Set False for raw-retrieval workflows (CLI ``--raw``): retrieval
    + rerank + MMR + the full audit trail still run, but synthesis and CRAG are
    skipped and the returned ``response`` is ``None``. ``trace["retrieved_chunks"]``
    holds the parent-chunks list so callers can render them without re-retrieving.

    ``on_stage`` (default None) is an optional ``Callable[[name, latency_ms], None]``
    invoked AFTER every ``audit.write_stage`` returns successfully — never
    before, never instead of (CONTEXT.md D-26). Callback exceptions are caught
    and logged; they never propagate into the pipeline.

    ``cancel_event`` (default None) is an optional ``threading.Event`` polled
    BETWEEN stages (immediately after the on_stage callback). When ``is_set()``
    the pipeline raises ``asyncio.CancelledError`` so callers can short-circuit;
    cancellation never interrupts an in-flight ``audit.write_stage`` (D-22 /
    PITFALLS Pitfall 5). The SSE handler in :mod:`src.server.api_chats` (Plan
    04) overwrites ``meta.outcome.cancelled = True`` atomically after the
    pipeline returns.

    ``on_audit_dir`` (default None) is an optional ``Callable[[Path], None]``
    invoked exactly once, immediately after the per-query audit directory is
    created (BEFORE the first stage). The SSE handler uses this to capture the
    exact audit dir for the in-flight stream so cancellation marks the right
    ``meta.json`` (WR-05) instead of the latest-mtime guess. Callback
    exceptions are caught and logged; they never propagate into the pipeline.

    Trace contains: original_query, decomposition, retrieval, synthesis, crag,
    retrieved_chunks (always; empty list when retrieval returned nothing).
    Always creates a per-query audit directory (via ``QueryLogger``) so eval
    stage-decomposed recall@k and observability aggregates have data to read.
    """
    from src.query.generate import generate  # noqa: PLC0415
    from src.query.policies import policy_for_intent  # noqa: PLC0415
    from src.query.query_logger import make_query_logger  # noqa: PLC0415

    use_stepback = config.USE_STEPBACK if use_stepback is None else use_stepback
    use_crag = config.USE_CRAG_LITE if use_crag is None else use_crag

    # Caller-provided overrides take precedence over the intent-derived
    # policy. We resolve the snapshot values up-front (used in config_snapshot
    # before the router fires), then refine after the router classifies intent.
    explicit_top_k_retrieve = top_k_retrieve
    explicit_top_k_rerank = top_k_rerank
    top_k_retrieve = top_k_retrieve or config.RETRIEVE_TOP_K
    top_k_rerank = top_k_rerank or config.RERANK_TOP_K

    effective_gen_model = gen_model or config.GEN_MODEL

    config_snapshot: dict[str, Any] = {
        "gen_model": effective_gen_model,
        "decompose_model": config.DECOMPOSE_MODEL,
        "judge_model": config.JUDGE_MODEL,
        "embed_provider": config.EMBEDDING_PROVIDER,
        "embed_dim": config.EMBED_DIM,
        "top_k_retrieve": top_k_retrieve,
        "top_k_rerank": top_k_rerank,
        "score_threshold": config.RERANK_SCORE_THRESHOLD,
        "max_children_per_parent": config.MAX_CHILDREN_PER_PARENT,
        "decompose": decompose,
        "use_router": use_router,
        "stepback": use_stepback,
        "crag": use_crag,
        "crag_threshold": config.CRAG_THRESHOLD,
        "use_mmr": config.USE_MMR,
        "mmr_lambda": config.MMR_LAMBDA,
        "mmr_top_k": config.MMR_TOP_K,
        "decompose_max_subqueries": config.DECOMPOSE_MAX_SUBQUERIES,
        "collection": collection,
        "notebook": notebook,
        "entry_point": "run_query_pipeline",
        "retriever": retriever,
        "partition_names": list(partition_names) if partition_names else None,
        "emit_title_hint": emit_title_hint,
    }
    audit = make_query_logger(query, config_snapshot=config_snapshot)
    # WR-05: fire on_audit_dir BEFORE any stages run so the SSE handler can
    # record the exact per-stream audit_dir and target the right meta.json on
    # cancellation (instead of "latest mtime on disk", which may pick another
    # concurrent run's meta.json or race the pipeline's own finalize).
    if on_audit_dir is not None:
        try:
            on_audit_dir(audit.dir_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("on_audit_dir callback raised: %r", exc)
    from src.query.query_logger import _PROJECT_ROOT  # noqa: PLC0415

    # Normally the audit dir lives under _PROJECT_ROOT/logs/queries/...,
    # but tests redirect QUERY_LOG_ROOT into tmp_path (outside the repo).
    # Fall back to the absolute path string when the audit dir is not under
    # _PROJECT_ROOT so the pipeline does not crash in sandboxed test envs.
    try:
        audit_dir_rel = str(audit.dir_path.relative_to(_PROJECT_ROOT))
    except ValueError:
        audit_dir_rel = str(audit.dir_path)

    pipeline_t0 = time.perf_counter()
    trace: dict[str, Any] = {
        "ts": datetime.now(tz=UTC).isoformat(),
        "original_query": query,
        "collection": collection,
        "notebook": notebook,
        "audit_dir": audit_dir_rel,
        "config": {
            "decompose": decompose,
            "use_router": use_router,
            "use_stepback": use_stepback,
            "use_crag": use_crag,
            "top_k_retrieve": top_k_retrieve,
            "top_k_rerank": top_k_rerank,
            "gen_model": effective_gen_model,
            "judge_model": config.JUDGE_MODEL,
            "decompose_model": config.DECOMPOSE_MODEL,
        },
    }
    rag_response: RAGResponse | None = None

    try:
        # Build sub-query list
        sub_queries: list[str]
        intent: str = "unknown"
        llm_use_mmr: bool | None = None
        llm_use_mmr_reason: str = ""
        # _router_plan holds the RoutingPlan when use_router=True (None otherwise).
        # Used by the stepback deprecation check below.
        _router_plan: Any = None
        if decompose and use_router:
            from src.query.router import route_query  # noqa: PLC0415

            r_t0 = time.perf_counter()
            _router_plan, _router_result = route_query(
                query, emit_title_hint=emit_title_hint
            )
            # TODO(router-filters): wire plan.filters into milvus_search_only
            # expr param. See PR #30 review. Filters are emitted + logged in
            # this PR (so we can observe what the LLM produces under
            # REWRITER_EMIT_FILTERS=true) but NOT wired into Milvus retrieval.
            # Use helper to get milvus sub-queries + append hyde_doc if present.
            sub_queries = _extract_sub_queries_from_plan(_router_plan)
            intent = _router_plan.intent
            llm_use_mmr = _router_plan.use_mmr
            llm_use_mmr_reason = _router_plan.use_mmr_reason
            decompose_payload: dict[str, Any] = _build_decompose_payload(
                plan=_router_plan,
                sub_queries=sub_queries,
                latency_ms=int((time.perf_counter() - r_t0) * 1000),
                llm_result=_router_result,
            )
            trace["decomposition"] = decompose_payload
            # Phase 2 D-21 — surface title_hint on the trace so the SSE
            # handler's auto-name hook can read it after the pipeline returns.
            # Keep the full plan fields the SSE caller cares about under a
            # stable key (no leakage of the dataclass into other consumers).
            trace["routing_plan"] = {
                "title_hint": _router_plan.title_hint,
                "intent": _router_plan.intent,
            }
            audit.write_stage("decompose", decompose_payload)
            _notify_stage_and_maybe_cancel(
                on_stage,
                cancel_event,
                "decompose",
                int(decompose_payload.get("latency_ms", 0)),
            )
            # CLAUDE.md NON-NEGOTIABLE rule 3: every LLM stage MUST call
            # accumulate_usage or the stage is invisible in meta.totals.
            if _router_result is not None:
                audit.accumulate_usage(
                    "decompose",
                    _router_result.usage,
                    _router_result.cost_usd,
                    _router_result.latency_ms,
                )
        elif decompose:
            from src.query.decompose import decompose_query  # noqa: PLC0415

            d_t0 = time.perf_counter()
            sub_queries, _decompose_result = decompose_query(query)
            decompose_payload = {
                "router": False,
                "sub_queries": sub_queries,
                "latency_ms": int((time.perf_counter() - d_t0) * 1000),
                "model": config.DECOMPOSE_MODEL,
                "usage": None,
                "cost_usd": None,
                "parsed_subqueries": sub_queries,
            }
            trace["decomposition"] = decompose_payload
            audit.write_stage("decompose", decompose_payload)
            _notify_stage_and_maybe_cancel(
                on_stage,
                cancel_event,
                "decompose",
                int(decompose_payload.get("latency_ms", 0)),
            )
        else:
            sub_queries = [query]
            decompose_payload = {
                "router": False,
                "sub_queries": sub_queries,
                "latency_ms": 0,
                "skipped": True,
                "model": None,
                "usage": None,
                "cost_usd": None,
                "parsed_subqueries": sub_queries,
            }
            trace["decomposition"] = decompose_payload
            audit.write_stage("decompose", decompose_payload)
            _notify_stage_and_maybe_cancel(
                on_stage,
                cancel_event,
                "decompose",
                int(decompose_payload.get("latency_ms", 0)),
            )

        if use_stepback:
            # Single-cycle deprecation: if the router already emitted a stepback
            # string, skip the legacy step_back_query call (saves one LLM round-trip).
            # If _router_plan is None (non-router path) or router returned no stepback,
            # fall back to the legacy call.
            if _router_plan is not None and not _should_call_legacy_stepback(_router_plan):
                # Router already emitted stepback — use it directly.
                router_stepback = _router_plan.stepback
                sb_payload = {
                    "query": router_stepback,
                    "latency_ms": 0,
                    "source": "router",
                }
                trace["stepback"] = sb_payload
                audit.write_stage("stepback", sb_payload)
                _notify_stage_and_maybe_cancel(
                    on_stage,
                    cancel_event,
                    "stepback",
                    int(sb_payload.get("latency_ms", 0)),
                )
                if router_stepback:
                    sub_queries = [*sub_queries, router_stepback]
            else:
                from src.query.decompose import step_back_query  # noqa: PLC0415

                sb_t0 = time.perf_counter()
                stepback, _sb_result = step_back_query(query)
                sb_payload = {
                    "query": stepback,
                    "latency_ms": int((time.perf_counter() - sb_t0) * 1000),
                }
                trace["stepback"] = sb_payload
                audit.write_stage("stepback", sb_payload)
                _notify_stage_and_maybe_cancel(
                    on_stage,
                    cancel_event,
                    "stepback",
                    int(sb_payload.get("latency_ms", 0)),
                )
                if stepback:
                    sub_queries = [*sub_queries, stepback]

        # Resolve retrieval policy. Resolution order:
        #   baseline → intent override → router LLM use_mmr override
        # Caller-provided top-k still wins over the policy. The resolved
        # policy + provenance lands in trace["policy"] and audit decompose.
        policy = policy_for_intent(intent, llm_use_mmr=llm_use_mmr)
        eff_top_k_retrieve = (
            explicit_top_k_retrieve if explicit_top_k_retrieve else policy.top_k_retrieve
        )
        eff_top_k_rerank = (
            explicit_top_k_rerank if explicit_top_k_rerank else policy.top_k_rerank
        )
        trace["policy"] = {
            "intent": intent,
            "top_k_retrieve": eff_top_k_retrieve,
            "top_k_rerank": eff_top_k_rerank,
            "use_mmr": policy.use_mmr,
            "use_mmr_source": "router_llm" if llm_use_mmr is not None else "intent_baseline",
            "use_mmr_reason": llm_use_mmr_reason,
            "mmr_lambda": policy.mmr_lambda,
            "mmr_pool_mult": policy.mmr_pool_mult,
            "score_threshold": policy.score_threshold,
            "max_children_per_parent": policy.max_children_per_parent,
        }

        # Retrieve (audit-instrumented when audit is set). Cross-encoder
        # scores against ORIGINAL query, not sub-queries — RRF merge already
        # handled per-sub-query coverage; CE judges final ordering against
        # the user's actual question.
        #
        # Phase 2 dispatcher (Plan 02-09): when retriever != "milvus", route
        # to the matching retrieve_<name> module. The chosen retriever shares
        # the audit dir via src.query.audit (contextvar-installed below) so
        # its sub-stages land alongside decompose / synthesis / etc. When
        # retriever == "milvus" (default), the existing Phase 1 path runs
        # unchanged — backward compatibility is absolute.
        retrieval_trace: dict[str, Any] = {}
        if retriever == "milvus":
            retrieved, retrieval_trace = _retrieve(
                sub_queries,
                query,
                collection,
                notebook,
                eff_top_k_retrieve,
                eff_top_k_rerank,
                audit=audit,
                use_mmr=policy.use_mmr,
                mmr_lambda=policy.mmr_lambda,
                mmr_pool_mult=policy.mmr_pool_mult,
                score_threshold=policy.score_threshold,
                max_children_per_parent=policy.max_children_per_parent,
                on_stage=on_stage,
                cancel_event=cancel_event,
                partition_names=partition_names,
            )
        else:
            # Phase 2 retriever (Plan 02-09 dispatcher). Install the active
            # QueryLogger via the src.query.audit contextvar facade so the
            # chosen retriever's sub-stages (paperqa_*, hipporag_*,
            # lazygraph_*, fused_*) land in the SAME audit dir as
            # decompose / synthesis. Reset in a finally so a downstream
            # exception can't leak the contextvar into the next request.
            from src.query import audit as _audit_mod  # noqa: PLC0415

            collections_for_retriever: list[str] = (
                partition_names if partition_names else [collection]
            )
            token = _audit_mod.set_active_logger(audit)
            try:
                if retriever == "paperqa":
                    from src.query.retrieve_paperqa import (  # noqa: PLC0415
                        retrieve_paperqa,
                    )

                    retrieved, retrieval_trace_any = retrieve_paperqa(
                        query,
                        collections_for_retriever,
                        on_stage=on_stage,
                        cancel_event=cancel_event,
                    )
                elif retriever == "hipporag":
                    from src.query.retrieve_hipporag import (  # noqa: PLC0415
                        retrieve_hipporag,
                    )

                    retrieved, retrieval_trace_any = retrieve_hipporag(
                        query,
                        collections_for_retriever,
                        on_stage=on_stage,
                        cancel_event=cancel_event,
                    )
                elif retriever == "lazygraph":
                    from src.query.retrieve_lazygraph import (  # noqa: PLC0415
                        retrieve_lazygraph,
                    )

                    retrieved, retrieval_trace_any = retrieve_lazygraph(
                        query,
                        collections_for_retriever,
                        on_stage=on_stage,
                        cancel_event=cancel_event,
                    )
                elif retriever == "fused":
                    from src.query.retrieve_fused import (  # noqa: PLC0415
                        retrieve_fused,
                    )

                    retrieved, retrieval_trace_any = retrieve_fused(
                        query,
                        collections_for_retriever,
                        on_stage=on_stage,
                        cancel_event=cancel_event,
                    )
                else:
                    raise ValueError(f"unknown retriever: {retriever!r}")
                retrieval_trace = (
                    retrieval_trace_any
                    if isinstance(retrieval_trace_any, dict)
                    else {}
                )
            finally:
                _audit_mod.reset_active_logger(token)

        trace["retrieval"] = retrieval_trace
        trace["retriever"] = retriever
        # Expose retrieved chunks on the trace so non-synthesizing callers
        # (e.g. CLI --raw) can render them without re-running retrieval.
        trace["retrieved_chunks"] = retrieved

        if not retrieved:
            trace["synthesis"] = None
            trace["total_latency_ms"] = int((time.perf_counter() - pipeline_t0) * 1000)
            audit.synthesis_skipped = True
            return None, trace

        # Persist parent texts so the audit dir contains everything synthesis saw.
        for idx, rc in enumerate(retrieved, 1):
            audit.write_chunk(idx, rc)
        audit.n_chunks_to_synth = len(retrieved)

        if not synthesize:
            # Raw-retrieval mode: skip synthesis and CRAG. Audit dir still
            # captures retrieval+rerank+MMR stages.
            trace["synthesis"] = None
            trace["total_latency_ms"] = int((time.perf_counter() - pipeline_t0) * 1000)
            audit.synthesis_skipped = True
            return None, trace

        # Generate (audit_dir flows through to queries.jsonl via _append_log)
        rag_response = generate(query, retrieved, model=gen_model, audit_dir=audit_dir_rel)
        audit.set_synthesis_latency(rag_response.latency_ms)
        synth_cost = compute_cost_usd(effective_gen_model, rag_response.usage)
        trace["synthesis"] = {
            "answer": rag_response.answer,
            "model": effective_gen_model,
            "latency_ms": rag_response.latency_ms,
            "n_citations": len(rag_response.citations),
        }
        audit.write_stage(
            "synthesis",
            {
                "model": effective_gen_model,
                "latency_ms": rag_response.latency_ms,
                "answer": rag_response.answer,
                "usage": dataclasses.asdict(rag_response.usage),
                "cost_usd": synth_cost,
                "parsed_citations": [
                    {
                        "source_file": c.source_file,
                        "page_number": c.page_number,
                        "modality": c.modality,
                    }
                    for c in rag_response.citations
                ],
            },
        )
        audit.accumulate_usage(
            "synthesis", rag_response.usage, synth_cost, rag_response.latency_ms
        )
        _notify_stage_and_maybe_cancel(
            on_stage,
            cancel_event,
            "synthesis",
            int(rag_response.latency_ms),
        )

        # CRAG-lite: grade + retry once if below threshold
        if use_crag:
            from src.query.grader import grade_groundedness, reformulate_query  # noqa: PLC0415

            contexts_for_grade = [rc.parent.text for rc in retrieved]
            score, reason, _grade_result = grade_groundedness(
                rag_response.answer, contexts_for_grade
            )
            grader_latency = _accumulate_crag_call(audit, "grader", _grade_result)
            triggered_retry = score < config.CRAG_THRESHOLD
            trace["crag"] = {"first_score": score, "first_reason": reason, "retried": False}
            audit.write_stage(
                "grader",
                {
                    "model": getattr(_grade_result, "model", config.JUDGE_MODEL),
                    "latency_ms": grader_latency,
                    "score": score,
                    "threshold": config.CRAG_THRESHOLD,
                    "reason": reason,
                    "triggered_retry": triggered_retry,
                    "usage": (
                        dataclasses.asdict(_grade_result.usage)
                        if _grade_result is not None
                        else {}
                    ),
                    "cost_usd": getattr(_grade_result, "cost_usd", None),
                },
            )

            if triggered_retry:
                new_q, _reformulate_result = reformulate_query(query, rag_response.answer)
                reformulate_latency = _accumulate_crag_call(
                    audit, "crag_reformulate", _reformulate_result
                )
                if new_q:
                    logger.info(
                        "CRAG-lite: groundedness %.2f < %.2f; retrying with reformulated query",
                        score,
                        config.CRAG_THRESHOLD,
                    )
                    retried_chunks, retry_retrieval_trace = _retrieve(
                        [new_q],
                        new_q,  # CRAG reformulated query is the new "original" for CE
                        collection,
                        notebook,
                        eff_top_k_retrieve,
                        eff_top_k_rerank,
                        audit=None,  # CRAG retry shares the same audit dir; do not overwrite
                        use_mmr=policy.use_mmr,
                        mmr_lambda=policy.mmr_lambda,
                        mmr_pool_mult=policy.mmr_pool_mult,
                        score_threshold=policy.score_threshold,
                        max_children_per_parent=policy.max_children_per_parent,
                    )
                    trace["crag"]["retry_query"] = new_q
                    trace["crag"]["retry_retrieval"] = retry_retrieval_trace
                    new_score = score
                    chose = "original"
                    retry_synthesis_latency = 0
                    retry_synthesis_cost: float | None = None
                    retry_grader_latency = 0
                    if retried_chunks:
                        retried_response = generate(
                            query, retried_chunks, model=gen_model, audit_dir=audit_dir_rel
                        )
                        retry_synthesis_latency = retried_response.latency_ms
                        retry_synthesis_cost = compute_cost_usd(
                            effective_gen_model, retried_response.usage
                        )
                        audit.accumulate_usage(
                            "crag_retry_synthesis",
                            retried_response.usage,
                            retry_synthesis_cost,
                            retried_response.latency_ms,
                        )
                        new_score, new_reason, _new_grade_result = grade_groundedness(
                            retried_response.answer,
                            [rc.parent.text for rc in retried_chunks],
                        )
                        retry_grader_latency = _accumulate_crag_call(
                            audit, "crag_retry_grader", _new_grade_result
                        )
                        trace["crag"]["retried"] = True
                        trace["crag"]["second_score"] = new_score
                        trace["crag"]["second_reason"] = new_reason
                        audit.crag_retried = True
                        if new_score > score:
                            rag_response = retried_response
                            retrieved = retried_chunks
                            chose = "retry"
                        audit.crag_chose = chose
                        trace["crag"]["chose"] = chose

                    audit.write_stage(
                        "crag_retry",
                        {
                            "model": effective_gen_model,
                            "retry_query": new_q,
                            "reformulate_latency_ms": reformulate_latency,
                            "retry_synthesis_latency_ms": retry_synthesis_latency,
                            "retry_synthesis_cost_usd": retry_synthesis_cost,
                            "retry_grader_latency_ms": retry_grader_latency,
                            "first_score": score,
                            "second_score": new_score,
                            "chose": chose,
                        },
                    )

        trace["total_latency_ms"] = int((time.perf_counter() - pipeline_t0) * 1000)
        return rag_response, trace
    finally:
        total_ms = int((time.perf_counter() - pipeline_t0) * 1000)
        from src.query.rerankers import get_reranker  # noqa: PLC0415

        try:
            reranker_id: dict[str, str | None] | None = get_reranker().identity()
        except Exception:
            reranker_id = None
        try:
            audit.finalize(total_latency_ms=total_ms, reranker_identity=reranker_id)
        except Exception as exc:
            logger.warning("audit.finalize failed: %s", exc)
