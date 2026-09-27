# long-ok-file
"""Hybrid retrieval pipeline: dense ANN + BM25 → RRF fusion → cross-encoder rerank.

Pipeline
--------
1. Embed query with Gemini Embedding 2 (768d MRL, task_type=RETRIEVAL_QUERY).
2. Hybrid search in Milvus via RRFRanker over:
   - Dense field  ``dense_embedding``  (COSINE, HNSW ef=max(2*top_k, 64))
   - Sparse field ``sparse_embedding`` (BM25, raw query string)
   Filtered to ``partition_names=[notebook]`` — never post-filter in Python.
3. Cross-encoder rerank top-50 candidates with ``BAAI/bge-reranker-v2-m3``.
4. Look up parent chunk text from ``parents/parents.sqlite``.
5. Return ``list[RetrievedChunk]`` (length ≤ top_k_rerank).

Score note
----------
``client.hybrid_search`` returns one fused RRF score per row (``hit['distance']``).
There are no separate per-modality scores at this API level.
``dense_score`` is set to the RRF-fused score; ``sparse_score`` is 0.0.
The fused score is the meaningful ranking signal — treat it as such.

CLI
---
    python -m src.retrieve \\
        --query "How is CEX hedge inventory rebalanced?" \\
        --collection notes \\
        --notebook example_topic
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from pathlib import Path
from typing import Any

import google.genai as genai
from pymilvus import AnnSearchRequest, RRFRanker

from src.config import (
    EMBED_DIM,
    EMBED_MODEL,
    EMBEDDING_PROVIDER,
    GEMINI_API_KEY,
    MAX_CHILDREN_PER_PARENT,
    MMR_LAMBDA,
    MMR_POOL_MULT,
    OPENROUTER_API_KEY,
    PARENTS_DB,
    RERANK_MIN_KEEP,
    RERANK_SCORE_THRESHOLD,
    RERANK_TOP_K,
    RETRIEVE_TOP_K,
    USE_MMR,
    validate_api_key,
)
from src.milvus_client import _MILVUS_LOCK, get_client
from src.models import ChildChunk, ParentChunk, RetrievedChunk
from src.query.mmr import mmr_select
from src.query.rerankers import get_reranker

logger = logging.getLogger(__name__)

# Reranker is now provided via ``src.query.rerankers.get_reranker()`` —
# the backend (local CrossEncoder vs remote vLLM HTTP) is selected by env var
# at process start. See rerankers.py for the protocol and implementations.
_get_reranker = get_reranker  # back-compat alias for the test suite


# ---------------------------------------------------------------------------
# Step 1 — query embedding
# ---------------------------------------------------------------------------


def _embed_query(query: str) -> list[float]:
    """Embed *query* using the configured EMBEDDING_PROVIDER.

    Returns an EMBED_DIM-dimensional float vector. Dispatches to Gemini or
    OpenRouter based on EMBEDDING_PROVIDER (must match what was used at
    ingest time — querying with a different embedder against an index built
    by another is meaningless).
    """
    if EMBEDDING_PROVIDER.startswith("gemini:"):
        return _embed_query_gemini(query)
    if EMBEDDING_PROVIDER.startswith("openrouter:"):
        return _embed_query_openrouter(query)
    raise RuntimeError(f"Unhandled EMBEDDING_PROVIDER: {EMBEDDING_PROVIDER}")


def _embed_query_gemini(query: str) -> list[float]:
    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.embed_content(
        model=EMBED_MODEL,
        contents=query,
        config=genai.types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=EMBED_DIM,
        ),
    )
    if not response.embeddings:
        raise RuntimeError("Gemini embed_content returned no embeddings for query.")
    raw_values = response.embeddings[0].values
    if raw_values is None:
        raise RuntimeError("Gemini embed_content returned embedding with None values.")
    if len(raw_values) != EMBED_DIM:
        raise RuntimeError(
            f"Expected {EMBED_DIM}-dim embedding, got {len(raw_values)}."
        )
    return list(raw_values)


def _embed_query_openrouter(query: str) -> list[float]:
    """Query embedding via OpenRouter's OpenAI-compat /v1/embeddings endpoint."""
    return _embed_queries_batch_openrouter([query])[0]


def _embed_queries_batch_openrouter(queries: list[str]) -> list[list[float]]:
    """Batch query embedding via OpenRouter — one HTTP call for N queries."""
    import openai as _openai  # noqa: PLC0415

    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set.")
    if not queries:
        return []
    client = _openai.OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
        # Hard timeout — OpenRouter occasionally hangs the connection silently
        # (observed with qwen3-embedding-8b after long idle periods).
        timeout=30.0,
        max_retries=2,
    )
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=queries,
        encoding_format="float",
    )
    if response.data is None or len(response.data) != len(queries):
        raise RuntimeError(
            f"OpenRouter returned {len(response.data) if response.data else 0} "
            f"embeddings for {len(queries)} queries (model={EMBED_MODEL})."
        )
    items = sorted(response.data, key=lambda d: d.index)
    out: list[list[float]] = []
    for item in items:
        vec = item.embedding
        if isinstance(vec, str):
            raise RuntimeError(
                f"Provider returned str embedding for model={EMBED_MODEL}; "
                f"expected float array."
            )
        if len(vec) != EMBED_DIM:
            raise RuntimeError(
                f"Provider returned dim={len(vec)}, expected {EMBED_DIM} "
                f"for model={EMBED_MODEL}."
            )
        out.append(list(vec))
    return out


def embed_queries_batch(queries: list[str]) -> list[list[float]]:
    """Public batch-embed helper. Returns one EMBED_DIM-vector per query.

    For Gemini: falls back to per-query calls (genai SDK doesn't expose a
    public batch shape that matches our query-mode usage).
    For OpenRouter: single HTTP call with input=[...], much faster for
    multi-sub-query evaluation harnesses.
    """
    if EMBEDDING_PROVIDER.startswith("openrouter:"):
        return _embed_queries_batch_openrouter(queries)
    return [_embed_query(q) for q in queries]


def milvus_search_only_with_vec(
    query: str,
    query_vec: list[float],
    collection: str,
    notebook: str,
    top_k: int = RETRIEVE_TOP_K,
) -> list[dict[str, Any]]:
    """Like ``milvus_search_only`` but takes a pre-computed query embedding.

    Use when you've batched embedding generation across multiple sub-queries
    upstream and want to skip the per-call embedder round-trip.
    """
    validate_api_key()
    return _milvus_hybrid_search(
        query=query,
        query_vec=query_vec,
        collection=collection,
        notebook=notebook,
        top_k=top_k,
    )


# ---------------------------------------------------------------------------
# Step 2 — Milvus hybrid search
# ---------------------------------------------------------------------------

_OUTPUT_FIELDS: list[str] = [
    "id",
    "text",
    "parent_chunk_id",
    "source_file",
    "notebook",
    "modality",
    "page_number",
    # Required by the optional MMR diversity filter (src.query.mmr); pulling
    # 4096 floats × 50 hits ≈ 800KB per query — small relative to embedder + LLM.
    "dense_embedding",
]


def _milvus_hybrid_search(
    query: str,
    query_vec: list[float],
    collection: str,
    notebook: str,
    top_k: int,
) -> list[dict[str, Any]]:
    """Run hybrid search in Milvus and return raw hit dicts.

    Uses two ``AnnSearchRequest`` objects fused with ``RRFRanker(k=60)``:
    - Dense:  ``dense_embedding`` field, COSINE metric, HNSW ef=max(2*top_k, 64).
    - Sparse: ``sparse_embedding`` field, BM25 metric, raw query string.

    Partition routing is enforced via ``partition_names=[notebook]``.

    Args:
        query:      Raw query string (fed to BM25 branch).
        query_vec:  768-dim dense embedding of the query.
        collection: Milvus collection name.
        notebook:   Partition key value (notebook tag).
        top_k:      Number of candidates to retrieve per modality.

    Returns:
        List of hit dicts (keys: ``id``, ``distance``, ``entity``).
    """
    client = get_client()

    # When is_partition_key=True, Milvus routes internally and blocks manual
    # partition_names on hybrid_search. Filter by notebook value via expr instead.
    # - Empty/None notebook => no filter (search across all partitions).
    # - Comma-separated notebook (e.g. "example_notebook,literature_review")
    #   => `notebook in [...]` IN-clause (still partition-pruned by Milvus).
    if not notebook:
        notebook_filter = ""
    elif "," in notebook:
        nbs = [n.strip() for n in notebook.split(",") if n.strip()]
        nbs_quoted = ", ".join(f'"{n}"' for n in nbs)
        notebook_filter = f"notebook in [{nbs_quoted}]"
    else:
        notebook_filter = f'notebook == "{notebook}"'

    # HNSW requires ef >= k. We pick max(2*top_k, 64) — wider exploration than
    # the strict minimum, which improves recall at negligible CPU cost on a
    # ~100k-row collection. Hardcoded ef=64 used to break for top_k > 64.
    ef = max(2 * top_k, 64)
    dense_req = AnnSearchRequest(
        data=[query_vec],
        anns_field="dense_embedding",
        param={"metric_type": "COSINE", "params": {"ef": ef}},
        limit=top_k,
        expr=notebook_filter,
    )

    sparse_req = AnnSearchRequest(
        data=[query],
        anns_field="sparse_embedding",
        param={"metric_type": "BM25"},
        limit=top_k,
        expr=notebook_filter,
    )

    # hybrid_search returns a list of lists; we query one query → take index 0.
    # _MILVUS_LOCK serializes concurrent calls from parallel eval pipeline
    # (pipeline_workers > 1). pymilvus thread-safety is undocumented; locking
    # is the conservative safe path. Milvus searches are fast so no throughput cost.
    with _MILVUS_LOCK:
        results = client.hybrid_search(
            collection_name=collection,
            reqs=[dense_req, sparse_req],
            ranker=RRFRanker(k=60),
            limit=top_k,
            output_fields=_OUTPUT_FIELDS,
        )

    if not results:
        logger.warning(
            "hybrid_search returned no results for collection=%s notebook=%s",
            collection,
            notebook,
        )
        return []

    # results is a list of hit-lists, one per query; we issued one query.
    hits = results[0]
    return [
        {
            "id": hit["id"],
            "distance": hit["distance"],
            "entity": hit["entity"],
        }
        for hit in hits
    ]


# ---------------------------------------------------------------------------
# Step 3 — cross-encoder rerank
# ---------------------------------------------------------------------------


def _rerank(
    query: str,
    hits: list[dict[str, Any]],
    top_k: int,
    score_threshold: float = RERANK_SCORE_THRESHOLD,
    max_children_per_parent: int = MAX_CHILDREN_PER_PARENT,
) -> list[tuple[dict[str, Any], float]]:
    """Rerank *hits*, drop low-score junk, dedup near-duplicate children of same parent.

    Pipeline (in order):
    1. Cross-encoder score every (query, hit.text) pair.
    2. Sort descending by score.
    3. Drop hits with score < ``score_threshold``.
    4. Greedy per-parent cap: take chunks in score order, but skip any whose
       ``parent_chunk_id`` already has ``max_children_per_parent`` entries selected.
       Caps redundancy WITHIN a parent (the parent is what gets sent to the LLM
       anyway), not across the whole document — academic questions often need
       3-6 chunks from the same paper for equation+assumption+result chains.
    5. Return the first ``top_k`` survivors.
    """
    reranker = get_reranker()

    pairs = [(query, hit["entity"]["text"]) for hit in hits]
    scores: list[float] = reranker.score(pairs)

    ranked = sorted(zip(hits, scores), key=lambda x: x[1], reverse=True)

    survivors: list[tuple[dict[str, Any], float]] = []
    per_parent_count: dict[str, int] = {}
    for hit, score in ranked:
        if score < score_threshold:
            break
        pid = hit["entity"]["parent_chunk_id"]
        if per_parent_count.get(pid, 0) >= max_children_per_parent:
            continue
        survivors.append((hit, score))
        per_parent_count[pid] = per_parent_count.get(pid, 0) + 1
        if len(survivors) >= top_k:
            break

    if not survivors and ranked and RERANK_MIN_KEEP > 0:
        # Floor: keep the best few when the threshold drops everything, so a valid
        # query never returns empty (low reranker scores / degraded fallback).
        for hit, score in ranked:
            pid = hit["entity"]["parent_chunk_id"]
            if per_parent_count.get(pid, 0) >= max_children_per_parent:
                continue
            survivors.append((hit, score))
            per_parent_count[pid] = per_parent_count.get(pid, 0) + 1
            if len(survivors) >= min(RERANK_MIN_KEEP, top_k):
                break
        logger.warning(
            "rerank floor: all %d candidates < threshold=%.2f; kept top %d (max=%.4f)",
            len(hits), score_threshold, len(survivors), ranked[0][1],
        )

    logger.info(
        "rerank: %d candidates → %d survivors (threshold=%.2f, max_children_per_parent=%d)",
        len(hits),
        len(survivors),
        score_threshold,
        max_children_per_parent,
    )
    return survivors


# ---------------------------------------------------------------------------
# Step 4 — parent chunk lookup
# ---------------------------------------------------------------------------


def _lookup_parents(
    parent_ids: list[str],
    db_path: str,
) -> dict[str, ParentChunk]:
    """Fetch ``ParentChunk`` rows from SQLite by ID.

    Args:
        parent_ids: List of ``parent_chunk_id`` values to look up.
        db_path:    Path to the SQLite database (relative or absolute).

    Returns:
        Dict mapping ``parent_chunk_id`` → ``ParentChunk``.
        Missing IDs are silently omitted (caller must handle).
    """
    if not parent_ids:
        return {}

    resolved = Path(db_path)
    if not resolved.is_absolute():
        # Resolve relative to project root (src/query/retrieve.py -> three levels up).
        resolved = Path(__file__).resolve().parent.parent.parent / db_path

    conn = sqlite3.connect(str(resolved))
    try:
        placeholders = ",".join("?" * len(parent_ids))
        rows = conn.execute(
            f"SELECT id, text, source_file, notebook, modality, page_number, image_path "
            f"FROM parents WHERE id IN ({placeholders})",
            parent_ids,
        ).fetchall()
    finally:
        conn.close()

    result: dict[str, ParentChunk] = {}
    for row in rows:
        pc = ParentChunk(
            id=row[0],
            text=row[1],
            source_file=row[2],
            notebook=row[3],
            modality=row[4],
            page_number=row[5],
            image_path=row[6],
        )
        result[pc.id] = pc
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def milvus_search_only(
    query: str,
    collection: str,
    notebook: str,
    top_k: int = RETRIEVE_TOP_K,
    *,
    partition_names: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Run hybrid Milvus search (dense + BM25 + RRF) for *query*; no rerank.

    Public wrapper around the internal ``_milvus_hybrid_search``. Used by the
    batched cross-encoder pipeline (B'') to collect candidates from multiple
    sub-queries before a single rerank pass.

    When ``partition_names`` is set, search runs once per collection in that
    list (each a Milvus collection name) with the notebook filter pinned to the
    same set via an IN-clause, and the per-collection hits are concatenated and
    re-sorted by distance. ``collection`` / ``notebook`` are ignored in that
    mode. This pushes the chat's selected-collections filter down to Milvus
    instead of post-filtering in Python.

    Returns raw hit dicts — caller is responsible for reranking and parent lookup.
    """
    validate_api_key()
    qvec = _embed_query(query)
    if partition_names:
        notebook_filter = ",".join(partition_names)
        merged: list[dict[str, Any]] = []
        for coll in partition_names:
            merged.extend(
                _milvus_hybrid_search(
                    query=query,
                    query_vec=qvec,
                    collection=coll,
                    notebook=notebook_filter,
                    top_k=top_k,
                )
            )
        merged.sort(key=lambda h: float(h["distance"]), reverse=True)
        return merged[:top_k]
    return _milvus_hybrid_search(
        query=query,
        query_vec=qvec,
        collection=collection,
        notebook=notebook,
        top_k=top_k,
    )


def batched_rerank_and_lookup(
    pairs: list[tuple[str, dict[str, Any]]],
    top_k: int = RERANK_TOP_K,
    score_threshold: float = RERANK_SCORE_THRESHOLD,
    max_children_per_parent: int = MAX_CHILDREN_PER_PARENT,
    *,
    use_mmr: bool | None = None,
    mmr_lambda: float | None = None,
    mmr_pool_mult: int | None = None,
) -> tuple[list[RetrievedChunk], dict[str, Any]]:
    """Rerank *pairs* in a single CrossEncoder call, then look up parents.

    Each *pair* is ``(sub_query, hit)``. The cross-encoder scores each chunk
    against ITS sub-query (preserving sub-query targeting), but all pairs go
    through ONE ``predict()`` call — much faster than N separate calls.

    The MMR-related kwargs (``use_mmr``, ``mmr_lambda``, ``mmr_pool_mult``)
    default to the module-level config when ``None``. The pipeline passes
    these per-query via ``RetrievalPolicy``.

    Returns:
        (retrieved_chunks, rerank_trace) where rerank_trace is a dict with:
        - ``per_pair_scores``: full distribution before threshold / cap
        - ``survivors``: full post-threshold / post-cap pool (pre-MMR)
        - ``mmr``: MMR pool/picks/dropped provenance (only when MMR fires)
    """
    if not pairs:
        return [], {}

    eff_use_mmr = USE_MMR if use_mmr is None else use_mmr
    eff_mmr_lambda = MMR_LAMBDA if mmr_lambda is None else mmr_lambda
    eff_mmr_pool_mult = MMR_POOL_MULT if mmr_pool_mult is None else mmr_pool_mult

    reranker = get_reranker()
    cross_inputs = [(sq, hit["entity"]["text"]) for sq, hit in pairs]
    scores: list[float] = reranker.score(cross_inputs)

    # Full distribution — written to 04_rerank.json for audit.
    per_pair_scores = [
        {
            "sub_query": sq,
            "child_id": hit["id"],
            "score": float(s),
        }
        for (sq, hit), s in zip(pairs, scores)
    ]

    # Dedup by Milvus child id, keeping highest score
    best_by_id: dict[str, tuple[dict[str, Any], float, str]] = {}
    for (sq, hit), score in zip(pairs, scores):
        cid = hit["id"]
        prev = best_by_id.get(cid)
        if prev is None or score > prev[1]:
            best_by_id[cid] = (hit, score, sq)

    ranked = sorted(
        best_by_id.values(), key=lambda x: x[1], reverse=True
    )

    # MMR reorder: collect a pool_mult× pool first, then let MMR pick top_k diverse.
    pool_size = top_k * eff_mmr_pool_mult if eff_use_mmr else top_k

    survivors: list[tuple[dict[str, Any], float, str]] = []
    per_parent: dict[str, int] = {}
    for hit, score, sq in ranked:
        if score < score_threshold:
            break
        pid = hit["entity"]["parent_chunk_id"]
        if per_parent.get(pid, 0) >= max_children_per_parent:
            continue
        survivors.append((hit, score, sq))
        per_parent[pid] = per_parent.get(pid, 0) + 1
        if len(survivors) >= pool_size:
            break

    if not survivors and ranked and RERANK_MIN_KEEP > 0:
        # Floor: keep the best few when the threshold drops everything, so a valid
        # query never returns empty (low reranker scores / degraded fallback).
        for hit, score, sq in ranked:
            pid = hit["entity"]["parent_chunk_id"]
            if per_parent.get(pid, 0) >= max_children_per_parent:
                continue
            survivors.append((hit, score, sq))
            per_parent[pid] = per_parent.get(pid, 0) + 1
            if len(survivors) >= min(RERANK_MIN_KEEP, pool_size):
                break
        logger.warning(
            "rerank floor: all %d pairs < threshold=%.2f; kept top %d (max=%.4f)",
            len(pairs), score_threshold, len(survivors), ranked[0][1],
        )

    # Snapshot the post-rerank, pre-MMR survivors. This is what the
    # rerank-stage recall@k reads — it must reflect the full pool the
    # cross-encoder produced, not the diversity-filtered subset, otherwise
    # the rerank/MMR delta is structurally zero.
    survivors_trace = [
        {
            "child_id": h["id"],
            "source_file": h["entity"]["source_file"],
            "page": h["entity"]["page_number"],
            "parent_chunk_id": h["entity"]["parent_chunk_id"],
            "rerank_score": float(s),
            "matched_sub_query": sq,
        }
        for h, s, sq in survivors
    ]

    mmr_trace: dict[str, Any] | None = None

    if eff_use_mmr and len(survivors) > top_k:
        mmr_pool = [(h, s) for h, s, _ in survivors]
        mmr_out = mmr_select(mmr_pool, top_k=top_k, lambda_=eff_mmr_lambda)
        chosen_ids = {h["id"] for h, _ in mmr_out}
        pool_ids = [h["id"] for h, _, _ in survivors]
        dropped_ids = [cid for cid in pool_ids if cid not in chosen_ids]

        # Resolve child_id → parent_chunk_id once so eval can compute final-stage
        # recall directly from mmr.picks without joining back to survivors.
        cid_to_pid = {h["id"]: h["entity"]["parent_chunk_id"] for h, _, _ in survivors}

        mmr_trace = {
            "lambda": eff_mmr_lambda,
            "top_k": top_k,
            "pool_size": len(survivors),
            "pool": [
                {
                    "child_id": h["id"],
                    "rerank_score": float(s),
                }
                for h, s, _ in survivors
            ],
            "picks": [
                {
                    "child_id": h["id"],
                    "parent_chunk_id": cid_to_pid.get(h["id"]),
                    "pick_order": i,
                    "rerank_score": float(s),
                }
                for i, (h, s) in enumerate(mmr_out)
            ],
            "dropped": dropped_ids,
        }

        survivors = [(h, s, sq) for h, s, sq in survivors if h["id"] in chosen_ids]
    else:
        survivors = survivors[:top_k]

    rerank_trace: dict[str, Any] = {
        "input_pair_count": len(pairs),
        "per_pair_scores": per_pair_scores,
        "config": {
            "score_threshold": score_threshold,
            "max_children_per_parent": max_children_per_parent,
            "mmr_pool_size": pool_size,
        },
        "survivors": survivors_trace,
    }
    if mmr_trace is not None:
        rerank_trace["mmr"] = mmr_trace

    parent_ids = [h["entity"]["parent_chunk_id"] for h, _, _ in survivors]
    parents = _lookup_parents(parent_ids=parent_ids, db_path=PARENTS_DB)

    chunks: list[RetrievedChunk] = []
    for hit, score, _sq in survivors:
        ent = hit["entity"]
        pid = ent["parent_chunk_id"]
        if pid not in parents:
            logger.warning("Parent %r missing in sqlite; skipping.", pid)
            continue
        child = ChildChunk(
            id=hit["id"],
            parent_id=pid,
            text=ent["text"],
            source_file=ent["source_file"],
            notebook=ent["notebook"],
            modality=ent["modality"],
            page_number=ent["page_number"],
        )
        chunks.append(
            RetrievedChunk(
                child=child,
                parent=parents[pid],
                dense_score=float(hit["distance"]),
                sparse_score=0.0,
                rerank_score=float(score),
            )
        )

    logger.info(
        "batched_rerank: %d pairs → %d unique → %d survivors "
        "(threshold=%.2f, max_children_per_parent=%d)",
        len(pairs),
        len(best_by_id),
        len(survivors),
        score_threshold,
        max_children_per_parent,
    )
    return chunks, rerank_trace


def hybrid_search_traced(
    query: str,
    collection: str,
    notebook: str,
    top_k_retrieve: int = RETRIEVE_TOP_K,
    top_k_rerank: int = RERANK_TOP_K,
) -> tuple[list[RetrievedChunk], dict[str, Any]]:
    """Same as :func:`hybrid_search` but also returns a trace dict.

    Trace fields (for JSONL logging):
        sub_query: the query passed in
        milvus_candidates: list of {id, source_file, page, rrf_score} (pre-rerank)
        rerank_survivors: list of {id, source_file, page, rerank_score} (post-cutoff/cap)
        retrieve_latency_ms, rerank_latency_ms
    """
    validate_api_key()

    trace: dict[str, Any] = {
        "sub_query": query,
        "milvus_candidates": [],
        "rerank_survivors": [],
        "retrieve_latency_ms": 0,
        "rerank_latency_ms": 0,
    }

    import time as _time  # noqa: PLC0415

    query_vec = _embed_query(query)

    t0 = _time.perf_counter()
    hits = _milvus_hybrid_search(
        query=query,
        query_vec=query_vec,
        collection=collection,
        notebook=notebook,
        top_k=top_k_retrieve,
    )
    trace["retrieve_latency_ms"] = int((_time.perf_counter() - t0) * 1000)

    if not hits:
        return [], trace

    trace["milvus_candidates"] = [
        {
            "id": h["id"],
            "source_file": h["entity"]["source_file"],
            "page": h["entity"]["page_number"],
            "rrf_score": float(h["distance"]),
        }
        for h in hits
    ]

    # When MMR is enabled, ask _rerank for an MMR_POOL_MULT× pool so MMR has
    # room to swap near-duplicates for more diverse candidates. The
    # cap+threshold logic inside _rerank still runs first; MMR reorders the
    # survivors.
    rerank_pool_size = top_k_rerank * MMR_POOL_MULT if USE_MMR else top_k_rerank

    t0 = _time.perf_counter()
    ranked = _rerank(query=query, hits=hits, top_k=rerank_pool_size)
    trace["rerank_latency_ms"] = int((_time.perf_counter() - t0) * 1000)

    if USE_MMR and len(ranked) > top_k_rerank:
        t0 = _time.perf_counter()
        ranked = mmr_select(ranked, top_k=top_k_rerank, lambda_=MMR_LAMBDA)
        trace["mmr_latency_ms"] = int((_time.perf_counter() - t0) * 1000)
        trace["mmr_lambda"] = MMR_LAMBDA
    else:
        ranked = ranked[:top_k_rerank]

    trace["rerank_survivors"] = [
        {
            "id": h["id"],
            "source_file": h["entity"]["source_file"],
            "page": h["entity"]["page_number"],
            "rerank_score": float(s),
        }
        for h, s in ranked
    ]

    parent_ids = [h["entity"]["parent_chunk_id"] for h, _ in ranked]
    parents = _lookup_parents(parent_ids=parent_ids, db_path=PARENTS_DB)

    retrieved: list[RetrievedChunk] = []
    for hit, rerank_score in ranked:
        entity = hit["entity"]
        parent_id = entity["parent_chunk_id"]
        if parent_id not in parents:
            logger.warning("Parent chunk %r not found in SQLite; skipping hit.", parent_id)
            continue
        child = ChildChunk(
            id=hit["id"],
            parent_id=parent_id,
            text=entity["text"],
            source_file=entity["source_file"],
            notebook=entity["notebook"],
            modality=entity["modality"],
            page_number=entity["page_number"],
        )
        retrieved.append(
            RetrievedChunk(
                child=child,
                parent=parents[parent_id],
                dense_score=float(hit["distance"]),
                sparse_score=0.0,
                rerank_score=float(rerank_score),
            )
        )

    return retrieved, trace


def hybrid_search(
    query: str,
    collection: str,
    notebook: str,
    top_k_retrieve: int = RETRIEVE_TOP_K,
    top_k_rerank: int = RERANK_TOP_K,
) -> list[RetrievedChunk]:
    """Retrieve top-*top_k_rerank* chunks relevant to *query* from *collection*/*notebook*.

    Pipeline: embed → hybrid Milvus search → cross-encoder rerank → parent lookup.

    Args:
        query:          Natural-language query string.
        collection:     Milvus collection name (e.g. ``"notes"``).
        notebook:       Partition key value (e.g. ``"example_topic"``).
        top_k_retrieve: Candidates to fetch from Milvus before reranking.
        top_k_rerank:   Final survivors returned after cross-encoder reranking.

    Returns:
        Ordered list of ``RetrievedChunk`` (descending rerank score), length ≤ *top_k_rerank*.
        Returns ``[]`` if Milvus contains no matching rows (logs a warning).

    Raises:
        RuntimeError: If the Gemini API key is missing or returns bad data.
        Exception:    Propagates any Milvus or SQLite errors (fail loudly).
    """
    validate_api_key()

    logger.debug(
        "hybrid_search: query=%r collection=%s notebook=%s top_k_retrieve=%d top_k_rerank=%d",
        query,
        collection,
        notebook,
        top_k_retrieve,
        top_k_rerank,
    )

    # 1. Embed
    query_vec = _embed_query(query)

    # 2. Hybrid search
    hits = _milvus_hybrid_search(
        query=query,
        query_vec=query_vec,
        collection=collection,
        notebook=notebook,
        top_k=top_k_retrieve,
    )

    if not hits:
        logger.warning("No hits from Milvus; returning empty list.")
        return []

    # 3. Rerank
    ranked = _rerank(query=query, hits=hits, top_k=top_k_rerank)

    # 4. Parent lookup
    parent_ids = [hit["entity"]["parent_chunk_id"] for hit, _ in ranked]
    parents = _lookup_parents(parent_ids=parent_ids, db_path=PARENTS_DB)

    # 5. Build RetrievedChunk objects
    retrieved: list[RetrievedChunk] = []
    for hit, rerank_score in ranked:
        entity = hit["entity"]
        child = ChildChunk(
            id=hit["id"],
            parent_id=entity["parent_chunk_id"],
            text=entity["text"],
            source_file=entity["source_file"],
            notebook=entity["notebook"],
            modality=entity["modality"],
            page_number=entity["page_number"],
        )

        parent_id = entity["parent_chunk_id"]
        if parent_id not in parents:
            logger.warning("Parent chunk %r not found in SQLite; skipping hit.", parent_id)
            continue

        parent = parents[parent_id]

        # dense_score = fused RRF score (the only score available from hybrid_search)
        # sparse_score = 0.0 (per-modality scores are not exposed by the API)
        retrieved.append(
            RetrievedChunk(
                child=child,
                parent=parent,
                dense_score=float(hit["distance"]),
                sparse_score=0.0,
                rerank_score=float(rerank_score),
            )
        )

    return retrieved


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hybrid RAG retrieval: dense + BM25 → rerank → parent lookup"
    )
    parser.add_argument("--query", required=True, help="Natural-language query string")
    parser.add_argument(
        "--collection",
        required=True,
        help="Milvus collection name (e.g. notes, trading, ecology)",
    )
    parser.add_argument(
        "--notebook",
        required=True,
        help="Notebook tag / partition key value (e.g. example_topic)",
    )
    parser.add_argument(
        "--top-k-retrieve",
        type=int,
        default=RETRIEVE_TOP_K,
        help=f"Candidates to fetch before reranking (default: {RETRIEVE_TOP_K})",
    )
    parser.add_argument(
        "--top-k-rerank",
        type=int,
        default=RERANK_TOP_K,
        help=f"Final results after reranking (default: {RERANK_TOP_K})",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for ``python -m src.retrieve``."""
    args = _parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s %(message)s",
    )

    results = hybrid_search(
        query=args.query,
        collection=args.collection,
        notebook=args.notebook,
        top_k_retrieve=args.top_k_retrieve,
        top_k_rerank=args.top_k_rerank,
    )

    if not results:
        print("No results returned.")
        sys.exit(0)

    print(f"\nTop {len(results)} result(s) for: {args.query!r}\n")
    for i, rc in enumerate(results, start=1):
        snippet = rc.child.text[:150].replace("\n", " ")
        print(
            f"[{i}] [{rc.child.modality}] {rc.child.source_file}:{rc.child.page_number}"
            f"  rerank={rc.rerank_score:.4f}  rrf={rc.dense_score:.4f}"
        )
        print(f"     {snippet}")
        print()


if __name__ == "__main__":
    main()
