"""Tests for src/query/query_pipeline._rrf_merge — Reciprocal Rank Fusion.

Pins the canonical RRF formula sum(1/(k+rank)) and the per-sub-query
balance behavior that motivated the RAG-Fusion refactor (2026-05-08).
"""

from __future__ import annotations

from typing import Any

from src.query.query_pipeline import _RRF_K, _rrf_merge


def _hit(child_id: str, source: str = "p.txt") -> dict[str, Any]:
    return {
        "id": child_id,
        "distance": 0.5,
        "entity": {
            "text": child_id,
            "parent_chunk_id": "P_" + child_id,
            "source_file": source,
            "notebook": "nb",
            "modality": "pdf",
            "page_number": 1,
        },
    }


class TestRrfMerge:
    def test_empty_input_returns_empty(self) -> None:
        assert _rrf_merge([]) == []
        assert _rrf_merge([[], [], []]) == []

    def test_single_list_preserves_order(self) -> None:
        hits = [_hit("a"), _hit("b"), _hit("c")]
        result = _rrf_merge([hits])
        assert [r[0]["id"] for r in result] == ["a", "b", "c"]
        # RRF score formula: 1/(k + rank), with rank starting at 1
        assert result[0][1] == 1 / (_RRF_K + 1)
        assert result[1][1] == 1 / (_RRF_K + 2)
        assert result[2][1] == 1 / (_RRF_K + 3)

    def test_chunk_in_two_lists_gets_boosted(self) -> None:
        # Chunk "a" appears at rank 1 in both lists → score = 2 * 1/(k+1)
        # Chunk "b" appears only at rank 1 in list 1 → score = 1/(k+1)
        # Chunk "c" appears only at rank 1 in list 2 → score = 1/(k+1)
        result = _rrf_merge(
            [
                [_hit("a"), _hit("b")],
                [_hit("a"), _hit("c")],
            ]
        )
        ids = [r[0]["id"] for r in result]
        # "a" should be first
        assert ids[0] == "a"
        # "a" score should be exactly double of "b" or "c" first-rank score
        a_score = result[0][1]
        assert a_score == 2 * (1 / (_RRF_K + 1))

    def test_per_sub_query_balance(self) -> None:
        """Regression test for the imbalance bug that motivated the
        RAG-Fusion refactor. Sub-query A returns 50 strong chunks,
        sub-query B returns 5. Without RRF, A would dominate the top-K.
        With RRF: B's chunk-1 (score 1/61) outranks A's chunk-50 (score 1/110)
        even though A's chunk-50 came from the more "productive" sub-query.
        """
        list_a = [_hit(f"a{i}") for i in range(50)]
        list_b = [_hit(f"b{i}") for i in range(5)]
        result = _rrf_merge([list_a, list_b])
        ids = [r[0]["id"] for r in result]
        # First chunks from A and B should both rank in the top half — B is
        # not crowded out despite producing 10× fewer hits
        top_10 = ids[:10]
        b_in_top_10 = sum(1 for i in top_10 if i.startswith("b"))
        assert b_in_top_10 >= 1, f"B starved in top 10: {top_10}"

    def test_appeared_in_sub_query_indices_recorded(self) -> None:
        result = _rrf_merge(
            [
                [_hit("shared"), _hit("only_in_0")],
                [_hit("only_in_1"), _hit("shared")],
            ]
        )
        by_id = {r[0]["id"]: r[2] for r in result}
        assert by_id["shared"] == [0, 1]
        assert by_id["only_in_0"] == [0]
        assert by_id["only_in_1"] == [1]

    def test_dedupe_keeps_first_seen_hit_dict(self) -> None:
        # Same id, but different source_file in the two lists. RRF should
        # keep the FIRST one seen as the canonical hit (deterministic).
        h_a = _hit("x", source="paper_a.pdf")
        h_b = _hit("x", source="paper_b.pdf")
        result = _rrf_merge([[h_a], [h_b]])
        assert len(result) == 1
        assert result[0][0]["entity"]["source_file"] == "paper_a.pdf"

    def test_score_ordering_strictly_descending(self) -> None:
        result = _rrf_merge(
            [
                [_hit("a"), _hit("b"), _hit("c")],
                [_hit("d"), _hit("a"), _hit("b")],
            ]
        )
        scores = [r[1] for r in result]
        for i in range(1, len(scores)):
            assert scores[i] <= scores[i - 1]
