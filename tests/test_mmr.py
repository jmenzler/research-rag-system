"""Tests for src.query.mmr.mmr_select.

MMR diversifies a reranked candidate list by trading some pointwise relevance
for coverage diversity. Tests below pin down the contract:

  * lambda=1.0 → output equals input order (no MMR work)
  * lambda=0.0 → first pick is best score, then maximally-far chunks
  * len(candidates) <= top_k → returned unchanged
  * missing dense_embedding → ValueError
  * unit-norm collapses dot product to cosine
  * empty/zero embedding handled (no division-by-zero crash)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from src.query.mmr import mmr_select


def _hit(child_id: str, embedding: list[float], text: str = "x") -> dict[str, Any]:
    return {
        "id": child_id,
        "distance": 0.5,
        "entity": {
            "text": text,
            "parent_chunk_id": f"p_{child_id}",
            "source_file": f"src_{child_id}",
            "notebook": "nb",
            "modality": "pdf",
            "page_number": 1,
            "dense_embedding": embedding,
        },
    }


def test_empty_input_returns_empty() -> None:
    assert mmr_select([], top_k=5, lambda_=0.5) == []


def test_top_k_zero_returns_empty() -> None:
    cands = [(_hit("a", [1.0, 0.0]), 0.9)]
    assert mmr_select(cands, top_k=0, lambda_=0.5) == []


def test_smaller_than_top_k_returns_unchanged() -> None:
    cands = [
        (_hit("a", [1.0, 0.0]), 0.9),
        (_hit("b", [0.0, 1.0]), 0.5),
    ]
    out = mmr_select(cands, top_k=5, lambda_=0.5)
    assert [c[0]["id"] for c in out] == ["a", "b"]


def test_lambda_one_preserves_relevance_order() -> None:
    # 4 chunks, scores 0.9 / 0.7 / 0.5 / 0.3, embeddings irrelevant when lambda=1
    cands = [
        (_hit("a", [1.0, 0.0]), 0.9),
        (_hit("b", [1.0, 0.0]), 0.7),  # near-duplicate of a in embedding space
        (_hit("c", [0.0, 1.0]), 0.5),
        (_hit("d", [0.0, -1.0]), 0.3),
    ]
    # lambda=1 means pure relevance — but mmr_select with lambda>=0.999
    # short-circuits to candidates[:top_k] (input order).
    out = mmr_select(cands, top_k=2, lambda_=1.0)
    assert [c[0]["id"] for c in out] == ["a", "b"]


def test_lambda_zero_picks_far_apart() -> None:
    # 4 candidates: a (best score, [1,0]), b (second best, ALSO near [1,0]),
    # c (third, orthogonal [0,1]), d (worst, opposite [-1,0]).
    # Lambda=0: first pick is highest-score (a), then pick maximally distant.
    cands = [
        (_hit("a", [1.0, 0.0]), 0.95),
        (_hit("b", [0.99, 0.01]), 0.90),  # near-duplicate of a
        (_hit("c", [0.0, 1.0]), 0.50),
        (_hit("d", [-1.0, 0.0]), 0.10),
    ]
    out = mmr_select(cands, top_k=2, lambda_=0.0)
    ids = [c[0]["id"] for c in out]
    assert ids[0] == "a"
    # Second pick is whichever is FURTHEST from a — that's d (cosine sim -1)
    # not b (cosine sim ~1).
    assert ids[1] == "d"


def test_lambda_balances_relevance_and_diversity() -> None:
    # Relevance scores forced to be informative: a=1.0, b=0.95 (near-dup of a),
    # c=0.6 (orthogonal). With lambda=0.5, picking c over b means diversity
    # penalty on b (-0.5 * ~1.0 = -0.5) outweighs the small relevance gap.
    cands = [
        (_hit("a", [1.0, 0.0]), 1.0),
        (_hit("b", [0.99, 0.01]), 0.95),
        (_hit("c", [0.0, 1.0]), 0.6),
    ]
    out = mmr_select(cands, top_k=2, lambda_=0.5)
    ids = [c[0]["id"] for c in out]
    assert ids[0] == "a"
    assert ids[1] == "c", "diversity should win: c is far from a, b is near a"


def test_missing_embedding_raises() -> None:
    bad_hit = {
        "id": "bad",
        "distance": 0.5,
        "entity": {
            "text": "x",
            "parent_chunk_id": "p",
            "source_file": "s",
            "notebook": "nb",
            "modality": "pdf",
            "page_number": 1,
            # no dense_embedding
        },
    }
    # Need len(candidates) > top_k so MMR actually runs (early-return otherwise).
    cands = [
        (_hit("good_a", [1.0, 0.0]), 0.9),
        (_hit("good_b", [0.0, 1.0]), 0.7),
        (bad_hit, 0.5),
    ]
    with pytest.raises(ValueError, match="dense_embedding"):
        mmr_select(cands, top_k=2, lambda_=0.5)


def test_zero_norm_embedding_does_not_crash() -> None:
    # Pathological case: zero embedding (shouldn't happen in practice, but
    # mmr should not divide by zero).
    cands = [
        (_hit("a", [1.0, 0.0]), 0.9),
        (_hit("b", [0.0, 0.0]), 0.5),  # zero vector
        (_hit("c", [0.0, 1.0]), 0.7),
    ]
    out = mmr_select(cands, top_k=2, lambda_=0.5)
    assert len(out) == 2


def test_constant_scores_does_not_crash() -> None:
    # All rerank scores identical → score_span == 0, normalization divides
    # by zero unless guarded.
    cands = [
        (_hit("a", [1.0, 0.0]), 0.5),
        (_hit("b", [0.0, 1.0]), 0.5),
        (_hit("c", [-1.0, 0.0]), 0.5),
    ]
    out = mmr_select(cands, top_k=2, lambda_=0.5)
    assert len(out) == 2


def test_output_is_subset_of_input() -> None:
    cands = [
        (_hit("a", list(np.random.rand(8))), 0.9),
        (_hit("b", list(np.random.rand(8))), 0.7),
        (_hit("c", list(np.random.rand(8))), 0.5),
        (_hit("d", list(np.random.rand(8))), 0.3),
    ]
    out = mmr_select(cands, top_k=3, lambda_=0.5)
    assert len(out) == 3
    out_ids = {c[0]["id"] for c in out}
    assert out_ids.issubset({"a", "b", "c", "d"})
