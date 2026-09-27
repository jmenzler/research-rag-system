"""Tests for the per-parent diversity cap in _rerank and batched_rerank_and_lookup.

The cap was changed from per-source_file (was: MAX_CHUNKS_PER_SOURCE) to
per-parent_chunk_id (now: MAX_CHILDREN_PER_PARENT). Old behavior killed recall
on academic questions where a single paper legitimately needs 3-6 chunks
(equation + assumption + result). The new cap dedups only near-duplicate children
of the same parent — the parent gets merged at lookup anyway.

These tests use a fake CrossEncoder so they don't touch the real BGE model.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from src.query import retrieve as r


def _hit(child_id: str, parent_id: str, source_file: str, text: str = "x") -> dict[str, Any]:
    return {
        "id": child_id,
        "distance": 0.5,
        "entity": {
            "text": text,
            "parent_chunk_id": parent_id,
            "source_file": source_file,
            "notebook": "nb",
            "modality": "pdf",
            "page_number": 1,
        },
    }


class _FakeReranker:
    """Implements the Reranker protocol — returns scores keyed by document text."""

    def __init__(self, scores_by_text: dict[str, float]) -> None:
        self._scores = scores_by_text

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [self._scores[text] for _q, text in pairs]


@pytest.fixture
def fake_reranker(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[dict[str, float]], None]:
    """Patch get_reranker (in retrieve module) to return a stub."""
    holder: dict[str, Any] = {}

    def install(scores_by_text: dict[str, float]) -> None:
        holder["enc"] = _FakeReranker(scores_by_text)
        monkeypatch.setattr(r, "get_reranker", lambda: holder["enc"])

    return install


# ---------- _rerank ----------


class TestRerank:
    def test_per_parent_cap_keeps_top_n_per_parent(
        self,
        fake_reranker: Callable[[dict[str, float]], None],
    ) -> None:
        """3 children of parent P, cap=2 -> only top-2 children survive."""
        hits = [
            _hit("c1", "P", "paper.txt", "t1"),
            _hit("c2", "P", "paper.txt", "t2"),
            _hit("c3", "P", "paper.txt", "t3"),
        ]
        fake_reranker({"t1": 0.9, "t2": 0.8, "t3": 0.7})

        survivors = r._rerank(
            query="q",
            hits=hits,
            top_k=15,
            score_threshold=0.0,
            max_children_per_parent=2,
        )

        assert [h["id"] for h, _ in survivors] == ["c1", "c2"]

    def test_per_parent_cap_does_not_cap_across_parents(
        self,
        fake_reranker: Callable[[dict[str, float]], None],
    ) -> None:
        """3 children of P1 + 3 children of P2, cap=2 -> 2 per parent = 4 total."""
        hits = [
            _hit("c1", "P1", "paper1.txt", "t1"),
            _hit("c2", "P1", "paper1.txt", "t2"),
            _hit("c3", "P1", "paper1.txt", "t3"),
            _hit("c4", "P2", "paper1.txt", "t4"),  # SAME source as P1
            _hit("c5", "P2", "paper1.txt", "t5"),
            _hit("c6", "P2", "paper1.txt", "t6"),
        ]
        fake_reranker({"t1": 0.95, "t2": 0.9, "t3": 0.8, "t4": 0.85, "t5": 0.75, "t6": 0.7})

        survivors = r._rerank(
            query="q",
            hits=hits,
            top_k=15,
            score_threshold=0.0,
            max_children_per_parent=2,
        )

        # Pre-fix this would have capped at 2 ANY-children-from-paper1.txt.
        # Post-fix: 2 children per parent_chunk_id => 4 survivors.
        ids = [h["id"] for h, _ in survivors]
        assert len(ids) == 4
        assert set(ids) == {"c1", "c2", "c4", "c5"}

    def test_score_threshold_drops_low_scoring(
        self,
        fake_reranker: Callable[[dict[str, float]], None],
    ) -> None:
        """Hits below threshold are dropped (break, not skip)."""
        hits = [
            _hit("c1", "P1", "p1.txt", "t1"),
            _hit("c2", "P2", "p2.txt", "t2"),
            _hit("c3", "P3", "p3.txt", "t3"),
        ]
        fake_reranker({"t1": 0.9, "t2": 0.5, "t3": 0.1})

        survivors = r._rerank(
            query="q",
            hits=hits,
            top_k=15,
            score_threshold=0.20,
            max_children_per_parent=2,
        )

        assert [h["id"] for h, _ in survivors] == ["c1", "c2"]

    def test_top_k_truncates_after_cap_and_threshold(
        self,
        fake_reranker: Callable[[dict[str, float]], None],
    ) -> None:
        """top_k limits final survivors regardless of scores."""
        hits = [_hit(f"c{i}", f"P{i}", f"p{i}.txt", f"t{i}") for i in range(10)]
        fake_reranker({f"t{i}": 0.9 - i * 0.01 for i in range(10)})

        survivors = r._rerank(
            query="q",
            hits=hits,
            top_k=3,
            score_threshold=0.0,
            max_children_per_parent=2,
        )

        assert len(survivors) == 3
        # Highest 3 scores
        assert [h["id"] for h, _ in survivors] == ["c0", "c1", "c2"]

    def test_returns_descending_score_order(
        self,
        fake_reranker: Callable[[dict[str, float]], None],
    ) -> None:
        """Survivors come out sorted descending by reranker score."""
        hits = [
            _hit("c_lo", "P1", "p.txt", "t_lo"),
            _hit("c_hi", "P2", "p.txt", "t_hi"),
            _hit("c_mid", "P3", "p.txt", "t_mid"),
        ]
        fake_reranker({"t_lo": 0.3, "t_hi": 0.9, "t_mid": 0.6})

        survivors = r._rerank(
            query="q",
            hits=hits,
            top_k=15,
            score_threshold=0.0,
            max_children_per_parent=2,
        )

        scores = [s for _h, s in survivors]
        assert scores == sorted(scores, reverse=True)
        assert [h["id"] for h, _ in survivors] == ["c_hi", "c_mid", "c_lo"]

    def test_q5_regression_six_children_one_parent_pair(
        self,
        fake_reranker: Callable[[dict[str, float]], None],
    ) -> None:
        """Regression: pre-fix Q5 lost recall because 6 expected children
        across a single source (paper 093) hit MAX_CHUNKS_PER_SOURCE=2.
        With per-parent cap and 3 distinct parents, recall is preserved.
        """
        # Simulate Q5: 6 children across 3 parents, all in paper 093.
        # Old code: caps at 2 per source_file -> at most 2 survivors.
        # New code: caps at 2 per parent_chunk_id -> up to 6 (2*3) survivors.
        hits = []
        scores_by_text = {}
        for parent_idx in range(3):
            for child_idx in range(2):
                cid = f"c_p{parent_idx}_{child_idx}"
                pid = f"P_093_{parent_idx}"
                txt = f"text_{cid}"
                hits.append(_hit(cid, pid, "093__paper.txt", txt))
                scores_by_text[txt] = 0.9 - (parent_idx * 0.01) - (child_idx * 0.001)
        fake_reranker(scores_by_text)

        survivors = r._rerank(
            query="q",
            hits=hits,
            top_k=15,
            score_threshold=0.0,
            max_children_per_parent=2,
        )

        survivor_parents = {h["entity"]["parent_chunk_id"] for h, _ in survivors}
        assert len(survivors) == 6
        assert survivor_parents == {"P_093_0", "P_093_1", "P_093_2"}


# ---------- batched_rerank_and_lookup ----------


class TestBatchedRerankAndLookup:
    def test_dedup_by_child_id_keeps_highest_score(
        self,
        fake_reranker: Callable[[dict[str, float]], None],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Same child surfaces from two sub-queries; keep the higher rerank score."""
        h1 = _hit("c_dup", "P1", "p.txt", "t1")
        # NB: text is the same for both — fake reranker keys on text.
        # We use a dummy approach: same hit dict, two different sub-queries.
        pairs = [
            ("sq_a", h1),
            ("sq_b", h1),
        ]
        fake_reranker({"t1": 0.7})
        # patch _lookup_parents to avoid sqlite
        monkeypatch.setattr(
            r,
            "_lookup_parents",
            lambda parent_ids, db_path: {
                pid: r.ParentChunk(
                    id=pid,
                    text="p",
                    source_file="p.txt",
                    notebook="nb",
                    modality="pdf",
                    page_number=1,
                    image_path=None,
                )
                for pid in parent_ids
            },
        )

        chunks, trace = r.batched_rerank_and_lookup(
            pairs,
            top_k=15,
            score_threshold=0.0,
            max_children_per_parent=2,
        )

        assert len(chunks) == 1
        assert chunks[0].child.id == "c_dup"
        # trace is now a dict; survivors list has 1 entry
        assert len(trace["survivors"]) == 1

    def test_per_parent_cap_in_batched(
        self,
        fake_reranker: Callable[[dict[str, float]], None],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """3 children of P, cap=2 in batched path -> 2 survivors."""
        pairs = [
            ("sq", _hit("c1", "P", "p.txt", "t1")),
            ("sq", _hit("c2", "P", "p.txt", "t2")),
            ("sq", _hit("c3", "P", "p.txt", "t3")),
        ]
        fake_reranker({"t1": 0.9, "t2": 0.8, "t3": 0.7})
        monkeypatch.setattr(
            r,
            "_lookup_parents",
            lambda parent_ids, db_path: {
                pid: r.ParentChunk(
                    id=pid,
                    text="p",
                    source_file="p.txt",
                    notebook="nb",
                    modality="pdf",
                    page_number=1,
                    image_path=None,
                )
                for pid in parent_ids
            },
        )

        chunks, _ = r.batched_rerank_and_lookup(
            pairs,
            top_k=15,
            score_threshold=0.0,
            max_children_per_parent=2,
        )

        assert [c.child.id for c in chunks] == ["c1", "c2"]

    def test_empty_pairs_returns_empty(self) -> None:
        chunks, trace = r.batched_rerank_and_lookup(
            [],
            top_k=15,
            score_threshold=0.0,
            max_children_per_parent=2,
        )
        assert chunks == []
        # trace is now a dict (empty when no pairs)
        assert trace == {}

    def test_threshold_drop_with_break_semantics(
        self,
        fake_reranker: Callable[[dict[str, float]], None],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Below-threshold hits stop the loop (break), even if a later hit
        would have been above threshold (it can't be — list is sorted desc).
        """
        pairs = [
            ("sq", _hit("c1", "P1", "p.txt", "t1")),
            ("sq", _hit("c2", "P2", "p.txt", "t2")),
        ]
        fake_reranker({"t1": 0.5, "t2": 0.1})
        monkeypatch.setattr(
            r,
            "_lookup_parents",
            lambda parent_ids, db_path: {
                pid: r.ParentChunk(
                    id=pid,
                    text="p",
                    source_file="p.txt",
                    notebook="nb",
                    modality="pdf",
                    page_number=1,
                    image_path=None,
                )
                for pid in parent_ids
            },
        )

        chunks, _ = r.batched_rerank_and_lookup(
            pairs,
            top_k=15,
            score_threshold=0.30,
            max_children_per_parent=2,
        )

        assert [c.child.id for c in chunks] == ["c1"]

    def test_mmr_stage_does_not_overwrite_survivors_trace(
        self,
        fake_reranker: Callable[[dict[str, float]], None],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Regression: rerank_trace.survivors must be the pre-MMR pool, not the
        post-MMR subset, so eval stage-decomposed recall@k can distinguish the
        rerank stage from the final stage. (Pre-fix the survivors_trace was
        built after MMR mutated `survivors`, collapsing the two stages.)
        """
        # Build a 6-pair set across 6 distinct parents so cap+threshold keep all,
        # then USE_MMR + top_k=2 + pool_mult=3 → pool_size=6, MMR picks 2.
        # Each hit gets a deterministic 4-d dense_embedding so mmr_select can run.
        n = 6
        pairs = []
        for i in range(n):
            h = _hit(f"c{i}", f"P{i}", "p.txt", f"t{i}")
            # Orthogonal-ish unit vectors in 4-d so MMR picks see real diversity.
            emb = [0.0, 0.0, 0.0, 0.0]
            emb[i % 4] = 1.0
            h["entity"]["dense_embedding"] = emb
            pairs.append(("sq", h))
        fake_reranker({f"t{i}": 0.9 - 0.05 * i for i in range(n)})
        monkeypatch.setattr(
            r,
            "_lookup_parents",
            lambda parent_ids, db_path: {
                pid: r.ParentChunk(
                    id=pid,
                    text="p",
                    source_file="p.txt",
                    notebook="nb",
                    modality="pdf",
                    page_number=1,
                    image_path=None,
                )
                for pid in parent_ids
            },
        )
        monkeypatch.setattr(r, "USE_MMR", True)
        monkeypatch.setattr(r, "MMR_LAMBDA", 0.5)
        monkeypatch.setattr(r, "MMR_POOL_MULT", 3)

        chunks, trace = r.batched_rerank_and_lookup(
            pairs,
            top_k=2,
            score_threshold=0.0,
            max_children_per_parent=2,
        )

        # After fix: survivors_trace is the full 6-element pool.
        # mmr.picks is the 2-element diversity-filtered subset.
        # Final chunks length is 2 (post-MMR).
        assert "mmr" in trace
        assert len(trace["survivors"]) == 6, (
            "regression: survivors must include the full pre-MMR pool so "
            "eval stage_recall can distinguish rerank from final"
        )
        assert len(trace["mmr"]["picks"]) == 2
        assert len(chunks) == 2
        # Each pick carries parent_chunk_id (so eval doesn't need to join via survivors).
        for pick in trace["mmr"]["picks"]:
            assert pick.get("parent_chunk_id"), "mmr.picks entries must carry parent_chunk_id"


# ---------- config defaults ----------


class TestConfigDefaults:
    def test_retrieve_top_k_is_100(self) -> None:
        """RETRIEVE_TOP_K production default. Bumped 50 → 100 on 2026-05-08
        per Akarsu corpus finding (Recall@5 0.83 → 0.89 going 50 → 100; we
        accept the diminishing-returns trade for the recall headroom)."""
        from src.config import RETRIEVE_TOP_K

        assert RETRIEVE_TOP_K == 100

    def test_max_children_per_parent_is_2(self) -> None:
        """Step 2: rename to MAX_CHILDREN_PER_PARENT, default 2."""
        from src.config import MAX_CHILDREN_PER_PARENT

        assert MAX_CHILDREN_PER_PARENT == 2

    def test_old_max_chunks_per_source_is_gone(self) -> None:
        """The old per-source cap symbol must not be importable."""
        with pytest.raises(ImportError):
            from src.config import MAX_CHUNKS_PER_SOURCE  # noqa: F401
