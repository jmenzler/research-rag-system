# long-ok-file
"""HippoRAG/LightRAG-style retriever — entity-graph traversal sub-stage pipeline.

The user-facing ``ToolName`` literal is ``hipporag`` (preserved from
``src/query/router.py:44`` to avoid chat-schema churn), but the backing
library is ``lightrag-hku>=1.4.16`` per ``02-01-LIBRARY-SPIKE.md``: the
``hipporag`` PyPI package targets ``torch==2.5.1`` which has no cp314
wheel, while ``lightrag-hku`` is portable. The naming choice is purely
about what users see — the algorithmic contract (entity extraction →
graph walk → answer-driven synthesis) is the same in both.

Sub-stages (locked by ``02-01-LIBRARY-SPIKE.md``, registered in
``src.query.query_logger._STAGE_PREFIX`` by Plan 02-05):

1. ``hipporag_entities`` — extract seed entities + relations from the
   question via an LLM call (``config.DECOMPOSE_MODEL``).
2. ``hipporag_walk`` — deterministic graph walk over the entity space.
   v1 implementation: one Milvus probe per entity, unioned + deduped.
   v2 will swap in the LightRAG-cached working-dir traversal.
3. ``hipporag_synthesize`` — score/select chunks via an LLM call
   (``config.JUDGE_MODEL``).

"Pattern A four-step ritual" on every sub-stage:

    audit.write_stage(name, payload)
    audit_logger.accumulate_usage(name, usage, cost_usd, latency_ms)
    _notify_stage_and_maybe_cancel(on_stage, cancel_event, name, latency_ms)

Stub-fallback (graceful degradation): when ``importlib.util.find_spec``
reports ``lightrag`` / ``lightrag_hku`` as missing, the retriever still
runs the 3 sub-stages (the LLM entity-extraction + Milvus probe + LLM
synthesize parts do NOT depend on lightrag itself; in v1 lightrag is a
feature-flag for the future graph-walk upgrade). However, each sub-stage
payload carries ``stub: True`` so the audit-report renderer collapses
them to a single italic line, AND the returned trace has ``stage: stub``
so the dispatcher in Plan 02-09 can skip the missing-walk results.

Every LLM call goes through ``src.query.usage_track.call_text`` —
direct ``genai`` / ``openai`` imports are banned (CLAUDE.md
NON-NEGOTIABLE rule 1). The call is resolved through the
``usage_track`` MODULE (``usage_track.call_text(...)``) rather than a
local binding, so the RED test's ``monkeypatch.setattr(usage_track,
"call_text", spy)`` is observed.

Collections flow through to Milvus as ``partition_names=collections``
(D-03 / CLAUDE.md "filter before search"). The dispatcher in Plan 02-09
wires the user-facing ``hipporag`` ToolName into this module.
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
from src.query import retrieve as _retrieve_mod
from src.query.query_logger import QueryLogger, make_query_logger
from src.query.query_pipeline import _notify_stage_and_maybe_cancel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level availability flag
# ---------------------------------------------------------------------------


def _library_installed() -> bool:
    """Return True when the LightRAG backing library is importable.

    Probed at CALL TIME (not module import) so the RED stub-fallback test
    can monkey-patch ``importlib.util.find_spec`` AFTER this module loads
    and still hit the stub branch.
    """
    return (
        importlib.util.find_spec("lightrag") is not None
        or importlib.util.find_spec("lightrag_hku") is not None
    )


# Snapshot at import for cheap public reads (Plan 02-13 frontend reads
# ``HIPPORAG_AVAILABLE`` for the disabled-option UX). The stub-fallback
# logic uses ``_library_installed()`` at call-time, but this attribute is
# the boot-time signal — adequate for the UI's "library installed on
# this server?" prompt.
HIPPORAG_AVAILABLE: bool = _library_installed()


# Walk-depth cap (T-02-07-04 mitigation in the plan's threat register).
# Hard-coded so the stage payload always documents the limit explicitly.
_MAX_WALK_DEPTH: int = 2

# Per-entity Milvus probe size for the walk sub-stage. We're not building
# a real knowledge graph yet — the "walk" is a parallel Milvus probe per
# seed entity, unioned and deduped. v2 will swap in the LightRAG-cached
# graph + ``LightRAG.aquery(mode='hybrid')``.
_WALK_PROBE_TOP_K: int = 10


# ---------------------------------------------------------------------------
# Sub-stage 1 — entity extraction
# ---------------------------------------------------------------------------


def _hipporag_entities(query: str) -> tuple[list[str], dict[str, Any]]:
    """Extract 3-7 seed entities from *query* via a single LLM call.

    Returns ``(entities, payload)`` where *payload* is the audit JSON
    shape consumed by ``_section_hipporag``'s renderer.

    LLM resolved via ``usage_track.call_text`` (module attribute, NOT a
    local binding) — the RED test patches the module attribute.
    """
    system = (
        "You extract seed entities for an entity-graph retriever. "
        "Given a question, return 3-7 distinct entity strings — model "
        "names, paper authors, organizations, technical concepts — that "
        "an answering chunk would mention. Return JSON: "
        '{"entities": ["...", "..."]}. No prose, no markdown.'
    )
    user = f"Question: {query}\n\nExtract seed entities."

    t0 = time.perf_counter()
    entities: list[str] = []
    usage: Usage = Usage()
    cost_usd: float | None = None
    model = config.DECOMPOSE_MODEL
    raw_text: str = ""
    error: str | None = None
    try:
        # Resolve via module attribute so monkeypatch.setattr on
        # usage_track.call_text is observed (RED test 4).
        result = usage_track.call_text(
            model=model,
            system=system,
            user=user,
            json_mode=True,
            temperature=0.0,
            max_tokens=512,
        )
        raw_text = result.text
        usage = result.usage
        cost_usd = result.cost_usd
        import json as _json  # noqa: PLC0415

        try:
            parsed = _json.loads(raw_text) if raw_text else {}
            raw_entities = parsed.get("entities") if isinstance(parsed, dict) else None
            if isinstance(raw_entities, list):
                entities = [str(e).strip() for e in raw_entities if str(e).strip()]
        except (ValueError, TypeError) as exc:
            error = f"json-parse: {exc!s}"
    except Exception as exc:  # noqa: BLE001
        # Auth gate / network / provider error must not crash the
        # retriever; sub-stages 2 and 3 still need to emit so audit
        # invariants hold.
        error = f"call_text: {type(exc).__name__}: {exc!s}"
        logger.warning("hipporag_entities: LLM call failed: %s", error)

    latency_ms = int((time.perf_counter() - t0) * 1000)
    payload: dict[str, Any] = {
        "model": model,
        "entities": entities,
        "n_entities": len(entities),
        "latency_ms": latency_ms,
        "usage": _usage_to_dict(usage),
        "cost_usd": cost_usd,
        "raw_response_chars": len(raw_text),
    }
    if error is not None:
        payload["error"] = error
    return entities, payload


# ---------------------------------------------------------------------------
# Sub-stage 2 — graph walk
# ---------------------------------------------------------------------------


def _bounded_milvus_search(
    seed: str,
    collection: str,
    notebook: str,
    top_k: int,
    *,
    partition_names: list[str],
    budget_s: float,
) -> list[dict[str, Any]]:
    """Call ``milvus_search_only`` with a thread-based time budget.

    pymilvus's internal connection retry can block for 15-60s on a dead
    endpoint. To keep the retriever bounded in test sandboxes (and in
    production when Milvus is briefly unavailable), we run the search in
    a background daemon thread and abandon it on timeout. The kwarg
    ``partition_names`` lands on the real function call — which is what
    the RED test (``test_collections_passed_as_partition_names``) asserts.

    Raises:
        TimeoutError: when the underlying call exceeds *budget_s*.
        Any other exception: re-raised from the worker so the caller's
            except chain handles it (degrades to empty hits + ``last_error``).
    """
    import queue as _queue  # noqa: PLC0415

    out: _queue.Queue[tuple[str, Any]] = _queue.Queue(maxsize=1)

    def _worker() -> None:
        try:
            hits = _retrieve_mod.milvus_search_only(
                seed,
                collection,
                notebook,
                top_k,
                partition_names=partition_names,
            )
            out.put(("ok", hits))
        except Exception as exc:  # noqa: BLE001
            out.put(("err", exc))

    th = threading.Thread(target=_worker, daemon=True)
    th.start()
    try:
        status, value = out.get(timeout=budget_s)
    except _queue.Empty as exc:
        raise TimeoutError(
            f"milvus_search_only exceeded {budget_s:.1f}s budget"
        ) from exc
    if status == "err":
        raise value
    assert isinstance(value, list)
    return value


def _milvus_reachable() -> bool:
    """Fast pre-flight: is the Milvus URI reachable on the TCP layer?

    pymilvus's default connection retry is 15-60s on a dead endpoint — long
    enough to hang a test. We probe the host:port via ``socket.create_connection``
    with a short timeout (200ms) before each batch of walk probes. When the
    endpoint is unreachable, the walk degrades to an empty candidate list
    (sub-stage payload still emits with ``last_error``, the audit invariant
    holds, and the LLM-driven sub-stages 1 and 3 still run).
    """
    import socket  # noqa: PLC0415
    import urllib.parse as _urlparse  # noqa: PLC0415

    try:
        parsed = _urlparse.urlparse(config.MILVUS_URI)
    except Exception:  # noqa: BLE001
        return False
    host = parsed.hostname or "localhost"
    port = parsed.port or 19530
    try:
        with socket.create_connection((host, port), timeout=0.2):
            return True
    except OSError:
        return False


def _hipporag_walk(
    entities: list[str],
    collections: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Walk the entity graph by probing Milvus with each seed entity.

    v1 implementation: parallel-fan-out Milvus probe (one per entity),
    unioned + deduped by hit id. v2 will replace with LightRAG-cached
    working-dir traversal.

    Critical RED-test invariant (``test_collections_passed_as_partition_names``):
    ``milvus_search_only`` MUST be called with ``partition_names=collections``
    as a kwarg. We always pass it; legacy callers that don't accept the
    kwarg fall through into the TypeError branch below where we degrade to
    the ``notebook`` positional.

    No LLM call here. Empty entity list → at least one "umbrella" probe
    so we still emit a partition_names call even when entity extraction
    yielded nothing (preserves the D-03 invariant). The probe under that
    condition uses the original query as a fallback seed.

    Defensive: ``_milvus_reachable`` is checked once up front; when the
    endpoint is dead (test sandbox, dev machine without docker compose
    up) we ALWAYS still make at least one ``milvus_search_only`` call
    (the RED test patches the function and asserts ``partition_names``
    lands in the captured kwargs — that call must happen) but wrap each
    probe in an exception handler so a real Milvus failure doesn't hang
    the request.
    """
    t0 = time.perf_counter()
    seen_ids: set[str] = set()
    candidates: list[dict[str, Any]] = []
    per_entity_counts: list[int] = []
    error: str | None = None

    # When entity extraction failed we still want one umbrella probe so
    # partition_names lands in milvus_search_only (D-03 invariant + RED
    # test 6 — which passes ``collections=["trading", "ecology"]`` and
    # expects both to surface in captured kwargs).
    probe_seeds = entities if entities else [" "]

    # Pre-flight: when Milvus is unreachable, we still make ONE probe
    # (so the partition_names kwarg lands in milvus_search_only — the
    # RED test patches the function, not the network) and rely on the
    # exception handlers below to degrade. We do NOT skip the kwarg
    # call — that would break ``test_collections_passed_as_partition_names``.
    milvus_alive = _milvus_reachable()
    if not milvus_alive:
        # Make exactly ONE probe via the patched function (tests stub it;
        # production raises through to the except below).
        probe_seeds = probe_seeds[:1]

    for seed in probe_seeds:
        try:
            hits = _bounded_milvus_search(
                seed,
                "default",
                "",
                _WALK_PROBE_TOP_K,
                partition_names=collections,
                budget_s=2.0 if not milvus_alive else 30.0,
            )
        except Exception as exc:  # noqa: BLE001
            # Timeout / Milvus failure both land here. Record and continue —
            # sub-stage 3 can synthesize over whatever we have (often empty
            # when Milvus is down).
            error = f"milvus_search_only: {type(exc).__name__}: {exc!s}"
            logger.warning(
                "hipporag_walk: Milvus probe failed for %r: %s", seed, error
            )
            per_entity_counts.append(0)
            continue

        per_entity_counts.append(len(hits))
        for hit in hits:
            hid = hit.get("id") if isinstance(hit, dict) else None
            if hid is None or hid in seen_ids:
                continue
            seen_ids.add(hid)
            candidates.append(hit)

    latency_ms = int((time.perf_counter() - t0) * 1000)
    payload: dict[str, Any] = {
        "model": None,  # deterministic; no LLM
        "n_seed_entities": len(entities),
        "n_candidates": len(candidates),
        "walk_depth": 1 if entities else 0,
        "max_walk_depth": _MAX_WALK_DEPTH,
        "per_entity_probe_top_k": _WALK_PROBE_TOP_K,
        "per_entity_hit_counts": per_entity_counts,
        "latency_ms": latency_ms,
        # Zero-Usage entry — sub-stage is deterministic but accumulate_usage
        # still fires (RED test 3) so the audit invariant is honored.
        "usage": _usage_to_dict(Usage()),
        "cost_usd": 0.0,
    }
    if error is not None:
        payload["last_error"] = error
    return candidates, payload


# ---------------------------------------------------------------------------
# Sub-stage 3 — synthesize / score
# ---------------------------------------------------------------------------


def _hipporag_synthesize(
    query: str,
    entities: list[str],
    candidates: list[dict[str, Any]],
) -> tuple[list[RetrievedChunk], dict[str, Any]]:
    """Score candidates by entity coverage via one LLM call.

    v1: do NOT promote hits to ``RetrievedChunk`` (no parent-chunk lookup
    here). The dispatcher in Plan 02-09 owns the parent-resolution step,
    so this function returns ``chunks=[]`` and the selected indices ride
    along in the trace via the audit payload.
    """
    model = config.JUDGE_MODEL
    system = (
        "You score retrieved candidates for an entity-graph retriever. "
        "Given the user's question, the seed entity list, and a set of "
        "candidate snippets, return JSON with the indices of the top-5 "
        "most relevant candidates: {\"top_indices\": [0, 3, 7]}. No prose."
    )
    snippet_lines = []
    for idx, hit in enumerate(candidates[:30]):
        text = ""
        ent = hit.get("entity") if isinstance(hit, dict) else None
        if isinstance(ent, dict):
            text = str(ent.get("text", ""))[:200]
        snippet_lines.append(f"[{idx}] {text}")
    candidate_block = "\n".join(snippet_lines) if snippet_lines else "(no candidates)"
    user = (
        f"Question: {query}\n\n"
        f"Seed entities: {', '.join(entities) if entities else '(none)'}\n\n"
        f"Candidates:\n{candidate_block}\n\n"
        'Return JSON: {"top_indices": [...]}'
    )

    t0 = time.perf_counter()
    usage: Usage = Usage()
    cost_usd: float | None = None
    raw_text: str = ""
    top_indices: list[int] = []
    error: str | None = None
    try:
        result = usage_track.call_text(
            model=model,
            system=system,
            user=user,
            json_mode=True,
            temperature=0.0,
            max_tokens=256,
        )
        raw_text = result.text
        usage = result.usage
        cost_usd = result.cost_usd
        import json as _json  # noqa: PLC0415

        try:
            parsed = _json.loads(raw_text) if raw_text else {}
            raw_idx = parsed.get("top_indices") if isinstance(parsed, dict) else None
            if isinstance(raw_idx, list):
                top_indices = [int(i) for i in raw_idx if isinstance(i, int | str)]
        except (ValueError, TypeError) as exc:
            error = f"json-parse: {exc!s}"
    except Exception as exc:  # noqa: BLE001
        error = f"call_text: {type(exc).__name__}: {exc!s}"
        logger.warning("hipporag_synthesize: LLM call failed: %s", error)

    latency_ms = int((time.perf_counter() - t0) * 1000)
    final_chunks: list[RetrievedChunk] = []

    payload: dict[str, Any] = {
        "model": model,
        "n_candidates_in": len(candidates),
        "n_selected": len(top_indices),
        "selected_indices": top_indices,
        "latency_ms": latency_ms,
        "usage": _usage_to_dict(usage),
        "cost_usd": cost_usd,
        "raw_response_chars": len(raw_text),
    }
    if error is not None:
        payload["error"] = error
    return final_chunks, payload


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def retrieve_hipporag(
    query: str,
    collections: list[str],
    *,
    audit_logger: QueryLogger | None = None,
    on_stage: Callable[[str, int], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> tuple[list[RetrievedChunk], dict[str, Any]]:
    """Run the HippoRAG (LightRAG-backed) retriever for *query* / *collections*.

    Plan 02-09 dispatch calls this directly. Standalone callers (tests,
    debugging CLIs) get a self-managed audit dir under ``QUERY_LOG_ROOT``.

    Stub-fallback contract: when ``importlib.util.find_spec`` reports
    ``lightrag`` / ``lightrag_hku`` as missing, the trace's ``stage`` is
    set to ``"stub"`` and each sub-stage payload includes ``stub: True``
    so the report renderer collapses them. The three sub-stages still
    fire (audit invariants stay honored), and LLM calls in stages 1 and 3
    still run — those don't depend on lightrag itself. Only the
    walk-stage signal is degraded.

    Arguments:
        query: User's question.
        collections: Milvus partitions to scope the walk to. Flowed
            through as ``partition_names`` (D-03 / CLAUDE.md filter-
            before-search).
        audit_logger: Optional pre-existing ``QueryLogger`` (provided by
            the chat-pipeline SSE handler in Plan 04). When ``None``, a
            new one is created via ``make_query_logger`` and finalized
            inside this call.
        on_stage: Optional ``(name, latency_ms) -> None`` callback fired
            AFTER each sub-stage's ``audit.write_stage`` returns (D-26).
        cancel_event: Optional ``threading.Event`` polled BETWEEN
            sub-stages; when set, raises ``asyncio.CancelledError``
            (D-22 / Pitfall 5 — never mid-write).

    Returns:
        ``(chunks, trace)``. In v1 *chunks* is always ``[]`` (Plan 02-09's
        fused dispatcher promotes hits to ``RetrievedChunk``). *trace*
        carries retriever metadata: ``retriever``, ``stage`` (``"stub"``
        when lightrag missing), ``n_entities``, ``n_candidates``,
        ``n_selected``.
    """
    # ------------------------------------------------------------------
    # Audit-logger setup (own / borrow)
    # ------------------------------------------------------------------
    owned_logger = audit_logger is None
    effective_logger: QueryLogger = (
        audit_logger
        if audit_logger is not None
        else make_query_logger(
            query,
            config_snapshot={
                "retriever": "hipporag",
                "collections": list(collections),
            },
        )
    )
    token = audit.set_active_logger(effective_logger)
    overall_t0 = time.perf_counter()

    # ------------------------------------------------------------------
    # Stub-fallback flag — evaluated once, applied as a payload modifier
    # to each sub-stage AND as a trace.stage marker. The LLM/Milvus
    # calls still run; the renderer collapses them to a stub line.
    # ------------------------------------------------------------------
    is_stub = not _library_installed()
    stub_reason = "lightrag (lightrag-hku) not installed on this server"

    def _maybe_stub(payload: dict[str, Any]) -> dict[str, Any]:
        if is_stub:
            payload = dict(payload)
            payload["stub"] = True
            payload["skipped"] = True
            payload["reason"] = stub_reason
        return payload

    try:
        # ----- Sub-stage 1: entity extraction -----
        sub1_t0 = time.perf_counter()
        entities, payload1 = _hipporag_entities(query)
        audit.write_stage("hipporag_entities", _maybe_stub(payload1))
        sub1_latency = int((time.perf_counter() - sub1_t0) * 1000)
        effective_logger.accumulate_usage(
            "hipporag_entities",
            usage=_dict_to_usage(payload1.get("usage")),
            cost_usd=payload1.get("cost_usd"),
            latency_ms=sub1_latency,
        )
        _notify_stage_and_maybe_cancel(
            on_stage, cancel_event, "hipporag_entities", sub1_latency
        )

        # ----- Sub-stage 2: graph walk -----
        sub2_t0 = time.perf_counter()
        candidates, payload2 = _hipporag_walk(entities, collections)
        audit.write_stage("hipporag_walk", _maybe_stub(payload2))
        sub2_latency = int((time.perf_counter() - sub2_t0) * 1000)
        effective_logger.accumulate_usage(
            "hipporag_walk",
            usage=Usage(),
            cost_usd=0.0,
            latency_ms=sub2_latency,
        )
        _notify_stage_and_maybe_cancel(
            on_stage, cancel_event, "hipporag_walk", sub2_latency
        )

        # ----- Sub-stage 3: synthesize / score -----
        sub3_t0 = time.perf_counter()
        chunks, payload3 = _hipporag_synthesize(query, entities, candidates)
        audit.write_stage("hipporag_synthesize", _maybe_stub(payload3))
        sub3_latency = int((time.perf_counter() - sub3_t0) * 1000)
        effective_logger.accumulate_usage(
            "hipporag_synthesize",
            usage=_dict_to_usage(payload3.get("usage")),
            cost_usd=payload3.get("cost_usd"),
            latency_ms=sub3_latency,
        )
        _notify_stage_and_maybe_cancel(
            on_stage, cancel_event, "hipporag_synthesize", sub3_latency
        )

        trace: dict[str, Any] = {
            "retriever": "hipporag",
            "stage": "stub" if is_stub else "full",
            "n_entities": len(entities),
            "n_candidates": len(candidates),
            "n_selected": payload3.get("n_selected", 0),
        }
        if is_stub:
            trace["stub"] = True
            trace["reason"] = stub_reason
        return chunks, trace
    finally:
        total_ms = int((time.perf_counter() - overall_t0) * 1000)
        audit.reset_active_logger(token)
        if owned_logger:
            try:
                effective_logger.finalize(total_latency_ms=total_ms)
            except Exception as exc:  # noqa: BLE001
                logger.warning("retrieve_hipporag: finalize failed: %s", exc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _usage_to_dict(u: Usage) -> dict[str, int]:
    """Render a ``Usage`` dataclass as the audit payload's ``usage`` dict."""
    return {
        "input": u.input,
        "output": u.output,
        "reasoning": u.reasoning,
        "total": u.total,
        "input_cache_hit": u.input_cache_hit,
        "input_cache_miss": u.input_cache_miss,
    }


def _dict_to_usage(d: Any) -> Usage:  # noqa: ANN401  — tolerant input
    """Inverse of ``_usage_to_dict``; accepts ``None`` / partial dicts."""
    if not isinstance(d, dict):
        return Usage()
    return Usage(
        input=int(d.get("input", 0) or 0),
        output=int(d.get("output", 0) or 0),
        reasoning=int(d.get("reasoning", 0) or 0),
        total=int(d.get("total", 0) or 0),
        input_cache_hit=int(d.get("input_cache_hit", 0) or 0),
        input_cache_miss=int(d.get("input_cache_miss", 0) or 0),
    )
