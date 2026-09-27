"""Tests for src/server/citations.py::validate_citations.

Covers CHAT-05 server-side citation validation pass (D-09, CRIT-7).
"""

from __future__ import annotations

from src.models import ChildChunk, ParentChunk, RetrievedChunk
from src.server.citations import validate_citations


def _chunk(idx: int) -> RetrievedChunk:
    parent = ParentChunk(
        id=f"parent-{idx}",
        text=f"parent text {idx}",
        source_file=f"source-{idx}.pdf",
        notebook="trading",
        modality="text",
        page_number=idx,
    )
    child = ChildChunk(
        id=f"child-{idx}",
        parent_id=f"parent-{idx}",
        text=f"child text {idx}",
        source_file=f"source-{idx}.pdf",
        notebook="trading",
        modality="text",
        page_number=idx,
    )
    return RetrievedChunk(
        child=child,
        parent=parent,
        dense_score=0.1 * idx,
        sparse_score=0.2 * idx,
        rerank_score=0.3 * idx,
    )


def test_resolved_markers_carry_scores() -> None:
    retrieved = [_chunk(1), _chunk(2), _chunk(3)]
    answer = "First fact [1]. Second fact [2]."
    out = validate_citations(answer, retrieved)
    assert len(out) == 2
    assert out[0]["marker"] == 1
    assert out[0]["resolved"] is True
    assert out[0]["child_id"] == "child-1"
    assert out[0]["parent_id"] == "parent-1"
    assert out[0]["paper_id"] == "source-1.pdf"
    assert out[0]["score_dense"] == 0.1
    assert out[0]["score_sparse"] == 0.2
    assert out[0]["score_rerank"] == 0.30000000000000004 or out[0]["score_rerank"] == 0.3
    assert out[1]["marker"] == 2
    assert out[1]["resolved"] is True


def test_unresolved_marker_emitted_with_resolved_false() -> None:
    retrieved = [_chunk(1), _chunk(2)]
    # marker 5 exceeds retrieved length (2) — must emit, NOT drop.
    answer = "Wrong cite [5]."
    out = validate_citations(answer, retrieved)
    assert len(out) == 1
    assert out[0]["marker"] == 5
    assert out[0]["resolved"] is False
    assert out[0]["child_id"] is None
    assert out[0]["parent_id"] is None
    assert out[0]["paper_id"] is None
    assert out[0]["score_dense"] is None


def test_duplicate_markers_deduped() -> None:
    retrieved = [_chunk(1)]
    answer = "First [1]. Also [1]. And [1]."
    out = validate_citations(answer, retrieved)
    assert len(out) == 1
    assert out[0]["marker"] == 1


def test_marker_order_preserved() -> None:
    retrieved = [_chunk(1), _chunk(2), _chunk(3)]
    answer = "Three [3]. Then one [1]. Then two [2]."
    out = validate_citations(answer, retrieved)
    assert [c["marker"] for c in out] == [3, 1, 2]


def test_false_positive_guard_skips_word_internal_brackets() -> None:
    retrieved = [_chunk(1)]
    # `arr[3]` is a python-style index expression; the negative-lookbehind
    # in _NUMERIC_BLOCK_RE must skip it (no leading word-char).
    answer = "Look at arr[3] then cite [1]."
    out = validate_citations(answer, retrieved)
    assert len(out) == 1
    assert out[0]["marker"] == 1


def test_comma_separated_markers_parsed() -> None:
    retrieved = [_chunk(1), _chunk(2)]
    answer = "Combined [1, 2]."
    out = validate_citations(answer, retrieved)
    assert len(out) == 2
    assert {c["marker"] for c in out} == {1, 2}


def test_empty_answer_returns_empty_list() -> None:
    out = validate_citations("", [_chunk(1)])
    assert out == []


def test_no_retrieved_marker_becomes_unresolved() -> None:
    out = validate_citations("Stuff [1].", [])
    assert len(out) == 1
    assert out[0]["resolved"] is False
