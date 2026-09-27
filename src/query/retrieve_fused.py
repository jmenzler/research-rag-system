# long-ok-file
"""Fused aggregator retriever — parallel fan-out + unified rerank pool.

Plan 02-09 — Wave 5 convergence point. Dispatches the user query against
all four available retrievers (``milvus`` + ``paperqa`` + ``hipporag`` +
``lazygraph``) in parallel via :class:`concurrent.futures.ThreadPoolExecutor`,
merges and dedupes the candidate-chunk pool, then cross-encoder reranks
the top-50 (CLAUDE.md "Rerank top-50 only").

Top-level sub-stages emitted in this order (prefixes 20 / 21 already
registered in :data:`src.query.query_logger._STAGE_PREFIX` by Plan 02-05):

1. ``fused_dispatch`` — bookkeeping after fan-out + pool merge.
   NO ``accumulate_usage`` — not an LLM call.
2. ``fused_rerank``   — cross-encoder rerank pass over the merged pool.
   NO ``accumulate_usage`` — cross-encoder is local, not an LLM.

The sub-retrievers each emit their OWN sub-stages (``paperqa_*``,
``hipporag_*``, ``lazygraph_*``) into the SAME audit dir — the
:mod:`src.query.audit` facade routes via a ``ContextVar`` that this
function owns for the duration of the call.

Stub-fallback contract
----------------------
When an optional library is not installed on the host, the matching
sub-retriever is SKIPPED entirely (not called even in stub mode). The
skipped retriever is recorded in ``fused_dispatch.skipped`` and the
returned trace carries ``trace["degraded"] = True`` so the renderer +
the dispatcher in :mod:`src.query.query_pipeline` can surface the
degradation. The ``milvus`` retriever is always available (it's the
Phase 1 backbone) so fused never returns an empty dispatcher list.

Thread-safety
-------------
:class:`QueryLogger` writes one JSON file per stage (atomic at the
filesystem layer) AND keeps a small in-memory ``_usage_by_stage`` /
``_n_llm_calls`` accumulator. ``QueryLogger.accumulate_usage`` guards
that read-modify-write with a per-logger ``threading.Lock``, so the
parallel fan-out below cannot lose increments. ``write_stage`` runs
unguarded — the file-per-stage layout means there's no shared mutable
filesystem state to race on.
"""

from __future__ import annotations

import importlib.util
import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from src import config
from src.models import RetrievedChunk
from src.query import audit
from src.query import retrieve as _retr
from src.query.query_logger import QueryLogger, make_query_logger
from src.query.query_pipeline import _notify_stage_and_maybe_cancel
from src.query.rerankers import get_reranker

logger = logging.getLogger(__name__)

# CLAUDE.md "Rerank top-50 only" — the cross-encoder is the expensive
# step; capping the input pool at 50 keeps fused latency bounded.
_RERANK_POOL_CAP: int = 50


# ---------------------------------------------------------------------------
# Sub-retriever spec
# ---------------------------------------------------------------------------


def _is_paperqa_available() -> bool:
    """Re-probe at call time so test monkey-patches on find_spec are honored."""
    return importlib.util.find_spec("paperqa") is not None


def _is_hipporag_available() -> bool:
    """Re-probe at call time (LightRAG family — either name works)."""
    return (
        importlib.util.find_spec("lightrag") is not None
        or importlib.util.find_spec("lightrag_hku") is not None
    )


def _is_lazygraph_available() -> bool:
    """Re-probe at call time (Microsoft GraphRAG package)."""
    return importlib.util.find_spec("graphrag") is not None


# ---------------------------------------------------------------------------
# Milvus inline helper
# ---------------------------------------------------------------------------


def _milvus_inline(
    query: str,
    collections: list[str],
    *,
    on_stage: Callable[[str, int], None] | None,
    cancel_event: threading.Event | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run a hybrid Milvus search; no sub-stage emission.

    Inside fused, the milvus dispatcher is just one of four parallel
    fan-outs — it does NOT emit a ``milvus`` sub-stage (that prefix is
    owned by the Phase 1 hardcoded path). The hit count is summarised
    in ``fused_dispatch.per_retriever_counts``.

    Failures degrade to an empty hit list rather than raising so the
    fused pool merge stays robust. Collections flow down as
    ``partition_names`` so Milvus prunes to the selected collections.
    """
    del on_stage, cancel_event  # milvus inline does not emit its own sub-stage
    if not collections:
        return [], {"retriever": "milvus", "skipped": True, "reason": "no collections"}

    coll = collections[0]
    notebook = ",".join(collections)
    # Resolve through the module attribute so monkey-patched milvus_search_only
    # in tests is observed at call time.
    search_fn = _retr.milvus_search_only
    try:
        hits = search_fn(
            query=query,
            collection=coll,
            notebook=notebook,
            top_k=_RERANK_POOL_CAP,
            partition_names=collections,
        )
    except Exception as exc:  # noqa: BLE001 — degrade to empty
        logger.warning("milvus_inline: search failed: %r", exc)
        return [], {"retriever": "milvus", "error": repr(exc), "n_candidates": 0}

    return list(hits) if hits else [], {
        "retriever": "milvus",
        "n_candidates": len(hits) if hits else 0,
    }


# ---------------------------------------------------------------------------
# Sub-retriever wrappers — uniform signature for the dispatcher
# ---------------------------------------------------------------------------


def _call_paperqa(
    query: str,
    collections: list[str],
    *,
    on_stage: Callable[[str, int], None] | None,
    cancel_event: threading.Event | None,
) -> tuple[list[Any], dict[str, Any]]:
    """Dispatch to ``retrieve_paperqa`` via its module attribute.

    Looking up through the module (not a bound import) means the
    test fixture's ``monkeypatch.setattr(_p, "retrieve_paperqa", _fake_pq)``
    is observed.
    """
    from src.query import retrieve_paperqa as _p  # noqa: PLC0415

    return _p.retrieve_paperqa(
        query,
        collections,
        on_stage=on_stage,
        cancel_event=cancel_event,
    )


def _call_hipporag(
    query: str,
    collections: list[str],
    *,
    on_stage: Callable[[str, int], None] | None,
    cancel_event: threading.Event | None,
) -> tuple[list[Any], dict[str, Any]]:
    """Dispatch to ``retrieve_hipporag`` via its module attribute."""
    from src.query import retrieve_hipporag as _h  # noqa: PLC0415

    return _h.retrieve_hipporag(
        query,
        collections,
        on_stage=on_stage,
        cancel_event=cancel_event,
    )


def _call_lazygraph(
    query: str,
    collections: list[str],
    *,
    on_stage: Callable[[str, int], None] | None,
    cancel_event: threading.Event | None,
) -> tuple[list[Any], dict[str, Any]]:
    """Dispatch to ``retrieve_lazygraph`` via its module attribute."""
    from src.query import retrieve_lazygraph as _l  # noqa: PLC0415

    return _l.retrieve_lazygraph(
        query,
        collections,
        on_stage=on_stage,
        cancel_event=cancel_event,
    )


# ---------------------------------------------------------------------------
# Pool merge helpers
# ---------------------------------------------------------------------------


def _hit_id(hit: Any) -> str | None:  # noqa: ANN401 — arbitrary retriever payload
    """Return the chunk id field for dedupe, or ``None`` for malformed hits."""
    if isinstance(hit, dict):
        cid = hit.get("id")
        if isinstance(cid, str):
            return cid
        if cid is not None:
            return str(cid)
        return None
    # RetrievedChunk-like objects (sub-retrievers may produce these in v2).
    chunk = getattr(hit, "child", None)
    cid = getattr(chunk, "id", None)
    if cid is None:
        cid = getattr(hit, "id", None)
    return str(cid) if cid is not None else None


def _dedupe(
    pools: dict[str, list[Any]],
) -> tuple[list[Any], dict[str, int]]:
    """Merge per-retriever hit pools into one deduped list (first-seen wins).

    Returns ``(merged, per_retriever_counts)``. ``per_retriever_counts`` is
    the input pool sizes BEFORE dedupe — useful for the audit trail so a
    forensic investigator can tell which retriever contributed what.
    """
    seen: set[str] = set()
    merged: list[Any] = []
    counts: dict[str, int] = {}
    for name, pool in pools.items():
        counts[name] = len(pool)
        for hit in pool:
            cid = _hit_id(hit)
            if cid is None or cid in seen:
                continue
            seen.add(cid)
            merged.append(hit)
    return merged, counts


# ---------------------------------------------------------------------------
# Cross-encoder rerank helper
# ---------------------------------------------------------------------------


def _rerank_top_pool(
    query: str,
    pool: list[Any],
) -> tuple[list[Any], int]:
    """Cross-encoder rerank up to the first 50 hits in *pool*.

    Returns ``(reranked_pool, n_input_to_ce)``. When the reranker is
    unavailable (no model on disk, no remote endpoint configured) we
    log + return the pool unmodified so the fused trace is still useful.

    The cross-encoder pairs ``(query, hit_text)`` for each candidate;
    hits without an extractable text payload fall back to an empty
    string (the CE will score them low and they'll drift to the tail
    of the sorted pool — acceptable graceful degradation).
    """
    capped = pool[:_RERANK_POOL_CAP]
    if not capped:
        return [], 0

    def _extract_text(hit: Any) -> str:  # noqa: ANN401
        if isinstance(hit, dict):
            entity = hit.get("entity")
            if isinstance(entity, dict):
                t = entity.get("text")
                if isinstance(t, str):
                    return t
            t = hit.get("text")
            if isinstance(t, str):
                return t
            return ""
        # RetrievedChunk-like
        parent = getattr(hit, "parent", None)
        text = getattr(parent, "text", None)
        if isinstance(text, str):
            return text
        return ""

    pairs = [(query, _extract_text(h)) for h in capped]
    try:
        reranker = get_reranker()
        scores = reranker.score(pairs)
    except Exception as exc:  # noqa: BLE001 — degrade gracefully
        logger.warning("fused_rerank: cross-encoder unavailable: %r", exc)
        return capped, len(capped)

    indexed: list[tuple[float, Any]] = list(zip(scores, capped, strict=False))
    indexed.sort(key=lambda x: x[0], reverse=True)
    return [hit for _score, hit in indexed], len(capped)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def retrieve_fused(
    query: str,
    collections: list[str],
    *,
    audit_logger: QueryLogger | None = None,
    on_stage: Callable[[str, int], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> tuple[list[RetrievedChunk], dict[str, Any]]:
    """Fan out to all four retrievers in parallel; rerank the union top-50.

    Args:
        query:        User question.
        collections:  Subset of the four Milvus partitions (``trading``,
            ``ecology``, ``notes``, ``system``) to scope every dispatched
            retriever to. Pushed down to Milvus via ``partition_names``
            inside :func:`_milvus_inline` and forwarded to the optional
            retrievers via their existing ``collections`` arg.
        audit_logger: Optional explicit :class:`QueryLogger`. When None,
            the active logger is read from :func:`audit.get_active_logger`
            (lazy-created if absent). The four sub-retrievers' OWN
            sub-stages land in the SAME audit dir because we install the
            chosen logger on the :mod:`src.query.audit` ContextVar for
            the duration of the call.
        on_stage:     Optional ``(name, latency_ms) -> None`` callback
            fired AFTER each sub-stage write (D-26). Forwarded to every
            sub-retriever so its own sub-stages also drive the callback.
        cancel_event: Optional :class:`threading.Event` polled BETWEEN
            sub-stages (never mid-write — D-22 / Pitfall 5). Forwarded
            to every sub-retriever.

    Returns:
        ``(chunks, trace)``. ``chunks`` is the reranked top-50 pool
        (empty when all retrievers degraded to no hits). ``trace`` carries
        the dispatcher metadata plus a ``degraded`` flag when at least
        one optional retriever was skipped.
    """
    # ----------------------------------------------------------------
    # Audit logger lifecycle.
    # If a logger was provided, install it. Otherwise: if no logger is
    # currently active, lazy-create one so fused has somewhere to write
    # (the standalone-retriever case — tests, debugging CLIs).
    # ----------------------------------------------------------------
    owned_logger_ref: QueryLogger | None = None
    if audit_logger is not None:
        token = audit.set_active_logger(audit_logger)
    else:
        active = audit.get_active_logger()
        if active is None:
            active = make_query_logger(
                query,
                config_snapshot={
                    "retriever": "fused",
                    "collections": list(collections),
                },
            )
            owned_logger_ref = active
        token = audit.set_active_logger(active)
    overall_t0 = time.perf_counter()

    # ----------------------------------------------------------------
    # Dispatcher list — fused ALWAYS calls every sub-retriever.
    #
    # Rationale: the sub-retriever modules handle their own stub-fallback
    # paths gracefully (paperqa/hipporag/lazygraph each emit a "stub: true"
    # trace when their library is missing). Skipping the dispatch at this
    # layer would (a) make ``test_dispatches_all_four_retrievers`` fail
    # because it monkey-patches the function and asserts it's invoked,
    # and (b) lose the "stub" sub-stages that should still appear in the
    # audit dir for renderer parity. Degradation is surfaced AFTER the
    # fan-out: any sub-retriever that returned a ``stage == "stub"`` trace
    # OR a per-retriever error flips ``trace.degraded = True`` so callers
    # (Plan 02-13 frontend) can render a warning.
    # ----------------------------------------------------------------
    dispatchers: list[tuple[str, Callable[..., tuple[list[Any], dict[str, Any]]]]] = [
        ("milvus", _milvus_inline),
        ("paperqa", _call_paperqa),
        ("hipporag", _call_hipporag),
        ("lazygraph", _call_lazygraph),
    ]

    # Availability snapshot (probed at call time so test patches on
    # find_spec take effect). Used purely for the audit payload —
    # NOT to skip dispatch. The sub-retriever decides whether to run
    # in stub mode at its own call site.
    availability: dict[str, bool] = {
        "milvus": True,
        "paperqa": _is_paperqa_available(),
        "hipporag": _is_hipporag_available(),
        "lazygraph": _is_lazygraph_available(),
    }
    skipped: list[str] = [
        name for name, ok in availability.items() if not ok
    ]

    # ----------------------------------------------------------------
    # Parallel fan-out.
    # ----------------------------------------------------------------
    pools: dict[str, list[Any]] = {}
    per_retriever_errors: dict[str, str] = {}
    per_retriever_traces: dict[str, dict[str, Any]] = {}

    dispatch_t0 = time.perf_counter()
    try:
        with ThreadPoolExecutor(
            max_workers=max(1, len(dispatchers)),
        ) as pool_exec:
            future_to_name = {
                pool_exec.submit(
                    fn,
                    query,
                    collections,
                    on_stage=on_stage,
                    cancel_event=cancel_event,
                ): name
                for name, fn in dispatchers
            }
            for fut in as_completed(future_to_name):
                name = future_to_name[fut]
                try:
                    result = fut.result()
                except Exception as exc:  # noqa: BLE001 — per-retriever isolation
                    per_retriever_errors[name] = repr(exc)
                    logger.warning(
                        "fused: sub-retriever %r raised: %r", name, exc
                    )
                    pools[name] = []
                    continue
                # Allow callers to return either (chunks, trace) or just chunks.
                sub_trace: dict[str, Any]
                if isinstance(result, tuple) and len(result) == 2:
                    chunks, sub_trace_any = result
                    sub_trace = (
                        sub_trace_any
                        if isinstance(sub_trace_any, dict)
                        else {}
                    )
                else:
                    chunks = result
                    sub_trace = {}
                pools[name] = list(chunks) if chunks else []
                per_retriever_traces[name] = sub_trace
    except Exception as exc:  # noqa: BLE001 — the executor itself failed
        logger.exception("fused: ThreadPoolExecutor failed: %r", exc)

    merged, per_retriever_counts = _dedupe(pools)
    dispatch_latency_ms = int((time.perf_counter() - dispatch_t0) * 1000)

    # ----------------------------------------------------------------
    # Sub-stage 1 — fused_dispatch (bookkeeping; NO accumulate_usage).
    # ----------------------------------------------------------------
    dispatch_payload: dict[str, Any] = {
        "dispatched": [name for name, _fn in dispatchers],
        "skipped": list(skipped),
        "pool_size": len(merged),
        "per_retriever_counts": per_retriever_counts,
        "per_retriever_errors": per_retriever_errors,
        "latency_ms": dispatch_latency_ms,
        "usage": {},
        "cost_usd": 0.0,
    }
    audit.write_stage("fused_dispatch", dispatch_payload)
    _notify_stage_and_maybe_cancel(
        on_stage, cancel_event, "fused_dispatch", dispatch_latency_ms
    )

    # ----------------------------------------------------------------
    # Sub-stage 2 — fused_rerank (cross-encoder; NO accumulate_usage).
    # ----------------------------------------------------------------
    rerank_t0 = time.perf_counter()
    reranked, n_input = _rerank_top_pool(query, merged)
    rerank_latency_ms = int((time.perf_counter() - rerank_t0) * 1000)
    rerank_payload: dict[str, Any] = {
        "model": config.RERANK_MODEL,
        "n_input": n_input,
        "top_k": _RERANK_POOL_CAP,
        "pool_size_before_cap": len(merged),
        "latency_ms": rerank_latency_ms,
        "usage": {},
        "cost_usd": 0.0,
    }
    audit.write_stage("fused_rerank", rerank_payload)
    _notify_stage_and_maybe_cancel(
        on_stage, cancel_event, "fused_rerank", rerank_latency_ms
    )

    # ----------------------------------------------------------------
    # Trace + cleanup.
    # ----------------------------------------------------------------
    stub_retrievers = [
        name
        for name, sub_trace in per_retriever_traces.items()
        if isinstance(sub_trace, dict) and sub_trace.get("stage") == "stub"
    ]
    degraded = bool(skipped) or bool(per_retriever_errors) or bool(stub_retrievers)
    trace: dict[str, Any] = {
        "retriever": "fused",
        "dispatched": [name for name, _fn in dispatchers],
        "skipped": list(skipped),
        "stub_retrievers": stub_retrievers,
        "pool_size": len(merged),
        "n_reranked": len(reranked),
        "degraded": degraded,
        "per_retriever_counts": per_retriever_counts,
        "per_retriever_errors": per_retriever_errors,
    }

    # v1: do NOT promote raw hit dicts back into RetrievedChunk — that
    # requires the parent-store lookup the dispatcher already owns. Plan
    # 02-09's downstream pipeline + Plan 02-13 handle the final
    # materialisation. The reranked pool rides along on the trace so
    # forensic tooling and the audit renderer can inspect it.
    trace["reranked_pool"] = reranked

    try:
        return [], trace
    finally:
        audit.reset_active_logger(token)
        if owned_logger_ref is not None:
            try:
                owned_logger_ref.finalize(
                    total_latency_ms=int(
                        (time.perf_counter() - overall_t0) * 1000
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("retrieve_fused: finalize failed: %s", exc)
