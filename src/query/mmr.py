"""Maximal Marginal Relevance (MMR) diversity filter for reranked candidates.

Post-rerank reordering that trades a controlled amount of pointwise relevance for
coverage diversity. Used to mitigate the failure mode where the cross-encoder
collapses the top-K to near-duplicates of the same fact (e.g. five chunks all
quoting the same VPIN definition), starving the synthesizer of complementary
context.

Algorithm (Carbonell & Goldstein, 1998):
    For each iteration, pick the candidate that maximizes
        lambda_ * relevance(c) - (1 - lambda_) * max_sim(c, already_selected)
    where ``relevance`` is the rerank score (min-max normalized to [0, 1]) and
    ``sim`` is cosine similarity over dense_embedding vectors.

    lambda_ = 1.0  → pure relevance (output equals input order)
    lambda_ = 0.0  → pure diversity (rerank score ignored after first pick)
    lambda_ = 0.6  → relevance-heavy with meaningful diversity push (default)

Inputs are the survivors of ``_rerank``: ``list[tuple[hit_dict, rerank_score]]``
where ``hit_dict["entity"]["dense_embedding"]`` carries the 4096-d Qwen3
embedding (the caller must request this field — see ``_OUTPUT_FIELDS``).

This module has zero LLM/embedding cost: it operates on already-fetched vectors.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def mmr_select(
    candidates: list[tuple[dict[str, Any], float]],
    *,
    top_k: int,
    lambda_: float = 0.6,
) -> list[tuple[dict[str, Any], float]]:
    """Reorder *candidates* by MMR and return the top-K.

    Each candidate is a ``(hit, rerank_score)`` tuple where ``hit["entity"]``
    has a ``"dense_embedding"`` list of floats. Missing embeddings raise
    ``ValueError`` — caller must fetch them via Milvus output_fields.

    When ``len(candidates) <= top_k`` or ``lambda_ >= 0.999``, returns the
    input unchanged (no work to do). When ``lambda_ <= 0.001``, behaves as
    pure-diversity selection.
    """
    if not candidates:
        return []
    if top_k <= 0:
        return []
    if len(candidates) <= top_k:
        return candidates
    if lambda_ >= 0.999:
        return candidates[:top_k]

    # Validate that embeddings are present; build matrix.
    embeddings_raw = []
    for hit, _ in candidates:
        emb = hit.get("entity", {}).get("dense_embedding")
        if emb is None:
            raise ValueError(
                "mmr_select: candidate is missing dense_embedding; "
                "ensure 'dense_embedding' is in Milvus output_fields"
            )
        embeddings_raw.append(emb)
    embeddings = np.asarray(embeddings_raw, dtype=np.float64)

    # Unit-normalize so cosine similarity collapses to a dot product.
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    embeddings = embeddings / norms

    # Min-max normalize rerank scores to [0, 1] so lambda mixing is meaningful.
    rerank_scores = np.array([s for _, s in candidates], dtype=np.float64)
    score_span = float(rerank_scores.max() - rerank_scores.min())
    if score_span < 1e-9:
        relevance_norm = np.ones_like(rerank_scores)
    else:
        relevance_norm = (rerank_scores - rerank_scores.min()) / score_span

    n = len(candidates)
    selected: list[int] = [int(np.argmax(relevance_norm))]
    remaining = set(range(n)) - {selected[0]}

    while len(selected) < top_k and remaining:
        rem_list = list(remaining)
        sims = embeddings[rem_list] @ embeddings[selected].T
        max_sim = sims.max(axis=1)
        scores = lambda_ * relevance_norm[rem_list] - (1 - lambda_) * max_sim
        best_pos = int(np.argmax(scores))
        best_idx = rem_list[best_pos]
        selected.append(best_idx)
        remaining.discard(best_idx)

    result = [candidates[i] for i in selected]
    logger.info(
        "mmr_select: %d candidates → %d picks (lambda=%.2f)",
        n,
        len(result),
        lambda_,
    )
    return result
