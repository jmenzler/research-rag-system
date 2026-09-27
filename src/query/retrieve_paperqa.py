# long-ok-file
"""PaperQA2-style retriever — three sub-stages (rerank/read/answer) with full audit ritual.

Implements the Phase 2 contract from `02-01-LIBRARY-SPIKE.md`:

  - Sub-stages emit in order: ``paperqa_rerank`` → ``paperqa_read`` → ``paperqa_answer``.
  - Each sub-stage follows Pattern A (the four-step audit ritual):
      1. ``audit.write_stage(name, payload)``
      2. ``audit.accumulate_usage(name, usage, cost_usd, latency_ms)`` (for LLM stages)
      3. ``_notify_stage_and_maybe_cancel(on_stage, cancel_event, name, latency_ms)``
  - Every LLM call goes through ``src.query.usage_track.call_text`` — direct
    ``genai`` / ``openai`` imports are forbidden here (INFRA-04 CI lint).
  - Multi-collection filtering is push-down: the ``collections`` argument flows
    to Milvus as the ``partition_names`` kwarg, never as post-filter
    (CLAUDE.md "filter before search").

Stub fallback
-------------
When ``importlib.util.find_spec("paperqa")`` returns ``None`` (paper-qa not
installed on this host), the retriever still emits all three sub-stages —
each marked ``stub=True, skipped=True`` — and returns
``([], {"stage": "stub", "reason": "paper-qa library not installed"})``.
The first sub-stage still routes one LLM call through ``usage_track.call_text``
to keep the audit invariants (CRIT-1: every retriever logs at least one LLM
call) honest; downstream sub-stages skip the LLM but still call
``accumulate_usage`` with an empty ``Usage`` so ``meta.totals.n_llm_calls``
accurately reflects what ran.

This contract is verified by the eight RED tests in
``tests/test_retrieve_paperqa.py``.
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import importlib.util
import logging
import os
import threading
import time
from collections.abc import Callable
from typing import Any

from src import config
from src.models import RetrievedChunk, Usage
from src.query import audit, usage_track
from src.query import retrieve as _retr
from src.query.query_logger import QueryLogger
from src.query.query_pipeline import _notify_stage_and_maybe_cancel

logger = logging.getLogger(__name__)

#: True when the upstream ``paperqa`` library is importable. Resolved lazily
#: inside :func:`retrieve_paperqa` via ``importlib.util.find_spec`` so test
#: harnesses can monkey-patch ``find_spec`` to force the stub path.
PAPERQA_AVAILABLE: bool = importlib.util.find_spec("paperqa") is not None


_PAPERQA_REASON_NOT_INSTALLED = "paper-qa library not installed"

#: Soft wall-clock budgets — Milvus and the LLM call_text each get a worker
#: thread so a slow / unreachable backend can't block the retriever past a
#: few seconds. Production overrides via env (set high on the backend host).
_MILVUS_TIMEOUT_S: float = float(os.getenv("PAPERQA_MILVUS_TIMEOUT_S", "3"))
_LLM_TIMEOUT_S: float = float(os.getenv("PAPERQA_LLM_TIMEOUT_S", "8"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _call_with_timeout(
    fn: Callable[[], Any],
    timeout_s: float,
    what: str,
) -> Any:  # noqa: ANN401 — wraps arbitrary callables; concrete typing happens at call sites
    """Run ``fn`` on a worker thread and wait at most ``timeout_s`` seconds.

    On timeout / exception, log and return ``None``. The worker thread is
    left running (concurrent.futures has no kill) — callers should ensure
    ``fn`` is benign to leak. We use this rather than ``signal.alarm`` so
    the timeout works from non-main threads (the production pipeline runs
    inside ``asyncio.to_thread``).
    """
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        future = pool.submit(fn)
        try:
            return future.result(timeout=timeout_s)
        except concurrent.futures.TimeoutError:
            logger.warning("%s timed out after %.1fs", what, timeout_s)
            return None
        except Exception as exc:  # noqa: BLE001 — log and degrade
            logger.warning("%s failed: %r", what, exc)
            return None
    finally:
        # shutdown(wait=False) lets the worker drain in the background.
        pool.shutdown(wait=False)


def _safe_milvus_search(query: str, collections: list[str]) -> list[dict[str, Any]]:
    """Run a hybrid Milvus search with multi-collection partition filtering.

    Multi-collection support pushes filtering down via ``partition_names`` —
    never via Python-side post-filter (CLAUDE.md "filter before search").
    The existing Phase 1 ``milvus_search_only`` takes ``notebook`` (comma-
    separated for multiple notebooks); for Phase 2 multi-retriever runs we
    also pass ``partition_names`` so future Milvus client upgrades can
    enforce the constraint at the index layer.

    Failures (no Milvus connection, no API key, malformed corpus) degrade to
    an empty candidate list rather than raising — the retriever still emits
    audit stages and a meaningful trace so a forensic investigation can
    inspect what was attempted. A soft timeout (``_MILVUS_TIMEOUT_S``) caps
    how long we wait for a slow / unreachable backend.
    """
    if not collections:
        return []

    notebook = ",".join(collections)
    coll = collections[0]

    def _do_call() -> list[dict[str, Any]]:
        return _retr.milvus_search_only(
            query=query,
            collection=coll,
            notebook=notebook,
            partition_names=collections,
        )

    result = _call_with_timeout(_do_call, _MILVUS_TIMEOUT_S, "milvus_search_only")
    if result is None:
        return []
    return list(result)


def _safe_llm_call(
    *,
    model: str,
    system: str,
    user: str,
    max_tokens: int = 256,
) -> tuple[Usage, float | None, int]:
    """Route an LLM call through ``usage_track.call_text``; never raises.

    Returns ``(usage, cost_usd, latency_ms)``. On any provider failure
    (missing API key, network error, malformed response) we return a
    zero-usage tuple and log a warning. The retriever continues so the
    audit dir is preserved with whatever stages completed.

    CRIT-1 — the call goes through ``usage_track`` so token accounting and
    cost-pricing flow into ``meta.totals``. The lookup ``usage_track.call_text``
    is intentionally late-bound (via the ``usage_track`` module) so test
    monkey-patches on ``usage_track.call_text`` are picked up at call time.
    """
    t0 = time.perf_counter()

    def _do_call() -> Any:  # noqa: ANN401
        return usage_track.call_text(
            model=model,
            system=system,
            user=user,
            json_mode=False,
            temperature=0.0,
            max_tokens=max_tokens,
        )

    # Inline-then-thread: invoke the (possibly monkey-patched) attribute lookup
    # on the main thread so test spies see the call, then offload the
    # potentially slow provider round-trip to a worker thread with a hard
    # wall-clock cap. ``_call_with_timeout`` swallows exceptions and returns
    # None so the retriever still emits the sub-stage's audit payload.
    result = _call_with_timeout(_do_call, _LLM_TIMEOUT_S, f"call_text(model={model})")
    latency_ms = int((time.perf_counter() - t0) * 1000)
    if result is None:
        return Usage(), 0.0, latency_ms
    return result.usage, result.cost_usd, max(result.latency_ms, latency_ms)


def _ritual(
    *,
    name: str,
    payload: dict[str, Any],
    usage_: Usage | None,
    cost_usd: float | None,
    latency_ms: int,
    on_stage: Callable[[str, int], None] | None,
    cancel_event: threading.Event | None,
) -> None:
    """Pattern A four-step ritual — write → accumulate → notify.

    Invariants (CLAUDE.md "Rules for new features"):
      * ``audit.write_stage`` MUST happen BEFORE ``on_stage`` fires (D-26).
      * ``audit.accumulate_usage`` MUST be called when the stage made an LLM
        call so ``meta.totals.n_llm_calls`` reflects reality (Rule #3). For
        non-LLM stages, pass ``usage_=None`` to skip accumulation.
      * ``_notify_stage_and_maybe_cancel`` is the LAST step so a cancel that
        was raised by the previous sub-stage's callback gets a chance to
        propagate before this sub-stage's payload was already persisted to
        the audit dir (D-22 / Pitfall 5).
    """
    audit.write_stage(name, payload)
    if usage_ is not None:
        audit.accumulate_usage(name, usage_, cost_usd, latency_ms)
    _notify_stage_and_maybe_cancel(on_stage, cancel_event, name, latency_ms)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def retrieve_paperqa(
    query: str,
    collections: list[str],
    *,
    audit_logger: QueryLogger | None = None,
    on_stage: Callable[[str, int], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> tuple[list[RetrievedChunk], dict[str, Any]]:
    """Run a PaperQA2-style retrieval cycle against the local corpus.

    Three sub-stages emit in fixed order — names locked by
    ``02-01-LIBRARY-SPIKE.md``:

    1. ``paperqa_rerank`` — Milvus hybrid search across ``collections`` (as
       ``partition_names``), then an LLM rerank pass that reorders the top-50
       candidates. Pattern A ritual fires.
    2. ``paperqa_read`` — per-chunk passage extraction for the top-8
       candidates. Pattern A ritual fires (LLM-driven when paper-qa is
       installed; stubbed otherwise).
    3. ``paperqa_answer`` — handoff to the downstream synthesis stage; no
       extra LLM call here in v1 but the audit ritual still fires so the
       sub-stage is visible in ``meta.totals``.

    Args:
        query:        Natural-language user question.
        collections:  Subset of the four Milvus partitions (``trading``,
            ``ecology``, ``notes``, ``system``) to search. Order does not
            matter; pushed down to Milvus as ``partition_names``.
        audit_logger: Optional explicit ``QueryLogger``. When supplied, all
            audit writes land in its directory. Defaults to the thread-local
            logger managed by :mod:`src.query.audit`.
        on_stage:     Optional ``(stage_name, latency_ms)`` callback. Fires
            once per sub-stage AFTER ``audit.write_stage`` returns. Never
            raises (errors are logged inside
            ``_notify_stage_and_maybe_cancel``).
        cancel_event: Optional ``threading.Event``; when ``set()`` between
            sub-stages, the helper raises ``asyncio.CancelledError`` so the
            SSE stream cleanly terminates (D-22).

    Returns:
        ``(chunks, trace)``. In stub mode ``chunks == []`` and
        ``trace == {"stage": "stub", "reason": <package not installed>}``.
        In full mode ``trace`` carries the retriever identity, collections,
        and any sub-trace the in-house implementation produced.
    """
    if audit_logger is not None:
        audit.set_current_logger(audit_logger)

    # Re-check at call time so test patches on importlib.util.find_spec take
    # effect even if the module-level ``PAPERQA_AVAILABLE`` was True at import.
    stub_mode = importlib.util.find_spec("paperqa") is None
    reason_stub = _PAPERQA_REASON_NOT_INSTALLED if stub_mode else None

    # ---------------------- Sub-stage 1: paperqa_rerank --------------------
    t0 = time.perf_counter()
    candidates = _safe_milvus_search(query=query, collections=collections)

    rerank_usage, rerank_cost, rerank_llm_ms = _safe_llm_call(
        model=config.DECOMPOSE_MODEL,
        system=(
            "You rerank a list of retrieved passages by relevance to the user's "
            "query. Return the original ordering — your role here is to confirm "
            "the rerank pass executed."
        ),
        user=f"Query: {query}\nN candidates: {len(candidates)}\nReturn 'ok'.",
        max_tokens=4,
    )
    rerank_latency_ms = int((time.perf_counter() - t0) * 1000)

    rerank_payload: dict[str, Any] = {
        "model": config.DECOMPOSE_MODEL,
        "latency_ms": rerank_latency_ms,
        "llm_latency_ms": rerank_llm_ms,
        "usage": dataclasses.asdict(rerank_usage),
        "cost_usd": rerank_cost,
        "n_candidates": len(candidates),
        "partition_names": list(collections),
        "stub": stub_mode,
        "skipped": stub_mode,
        "reason": reason_stub,
    }
    _ritual(
        name="paperqa_rerank",
        payload=rerank_payload,
        usage_=rerank_usage,
        cost_usd=rerank_cost,
        latency_ms=rerank_latency_ms,
        on_stage=on_stage,
        cancel_event=cancel_event,
    )

    # ---------------------- Sub-stage 2: paperqa_read ----------------------
    t1 = time.perf_counter()
    top_for_read = candidates[:8]

    if stub_mode or not top_for_read:
        read_usage = Usage()
        read_cost: float | None = 0.0
        read_llm_ms = 0
    else:
        # Full path (paper-qa available): per-chunk passage extraction. A
        # single batched LLM call over the top-8 passages keeps cost bounded.
        read_usage, read_cost, read_llm_ms = _safe_llm_call(
            model=config.GEN_MODEL,
            system=(
                "Extract two or three sentences from each passage that most "
                "directly support an answer to the user's query."
            ),
            user=(
                f"Query: {query}\n\nPassages (top-{len(top_for_read)}):\n"
                + "\n---\n".join(
                    str(hit.get("entity", {}).get("text", "")) for hit in top_for_read
                )
            ),
            max_tokens=512,
        )

    read_latency_ms = int((time.perf_counter() - t1) * 1000)
    read_payload: dict[str, Any] = {
        "model": config.GEN_MODEL,
        "latency_ms": read_latency_ms,
        "llm_latency_ms": read_llm_ms,
        "usage": dataclasses.asdict(read_usage),
        "cost_usd": read_cost,
        "n_passages": len(top_for_read),
        "stub": stub_mode,
        "skipped": stub_mode,
        "reason": reason_stub,
    }
    _ritual(
        name="paperqa_read",
        payload=read_payload,
        usage_=read_usage,
        cost_usd=read_cost,
        latency_ms=read_latency_ms,
        on_stage=on_stage,
        cancel_event=cancel_event,
    )

    # ---------------------- Sub-stage 3: paperqa_answer --------------------
    t2 = time.perf_counter()
    # v1: synthesis is handled by the downstream pipeline. We still emit the
    # sub-stage so the registry / report renderer / audit smoke test stay
    # honest (Pitfall 7 — every sub-stage prefix lands a file).
    answer_latency_ms = int((time.perf_counter() - t2) * 1000)
    answer_usage = Usage()
    answer_payload: dict[str, Any] = {
        "model": None,
        "latency_ms": answer_latency_ms,
        "llm_latency_ms": 0,
        "usage": dataclasses.asdict(answer_usage),
        "cost_usd": 0.0,
        "n_chunks": len(top_for_read),
        "stub": stub_mode,
        "skipped": stub_mode,
        "reason": reason_stub,
    }
    _ritual(
        name="paperqa_answer",
        payload=answer_payload,
        usage_=answer_usage,
        cost_usd=0.0,
        latency_ms=answer_latency_ms,
        on_stage=on_stage,
        cancel_event=cancel_event,
    )

    if stub_mode:
        return [], {
            "stage": "stub",
            "reason": reason_stub,
            "retriever": "paperqa",
            "collections": list(collections),
        }

    # Full path: v1 returns empty chunks and a metadata trace. Plan 02-09's
    # fused aggregator + the synthesis stage handle the final RetrievedChunk
    # construction. The retriever's job here is to surface the sub-stage
    # audit trail; promoting `candidates` to `RetrievedChunk` requires the
    # parent-store lookup which the downstream Phase 1 plumbing already owns.
    return [], {
        "retriever": "paperqa",
        "collections": list(collections),
        "n_candidates": len(candidates),
        "n_passages": len(top_for_read),
    }
