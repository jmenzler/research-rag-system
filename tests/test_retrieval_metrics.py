"""Unit tests for retrieval metrics — both chunk-level and source-level."""

from __future__ import annotations

import json
from pathlib import Path

from src.retrieval_metrics import (
    _dedupe_ranked,
    _norm_source,
    evaluate_from_logs,
    hit_at_k,
    mrr_at_k,
    recall_at_k,
    source_hit_at_k,
    source_mrr_at_k,
    source_recall_at_k,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def test_norm_source_keeps_pdf_basename() -> None:
    """Non-generic .pdf basename passed through verbatim."""
    assert _norm_source("a/b/c/file.pdf") == "file.pdf"


def test_norm_source_passes_through_basename() -> None:
    assert _norm_source("file.pdf") == "file.pdf"


def test_norm_source_flattened_doc_dir() -> None:
    """Post-flatten layout: sources/<coll>/<slug>/{nlm.txt,...} → <slug>."""
    p = "sources/trading/optimal_hf_market_making/nlm.txt"
    assert _norm_source(p) == "optimal_hf_market_making"


def test_norm_source_per_doc_dir_handles_source_pdf() -> None:
    p = "sources/trading/oracle_risk/source.pdf"
    assert _norm_source(p) == "oracle_risk"


def test_norm_source_strips_txt_suffix() -> None:
    """Bare leaf basename: drop trailing .txt."""
    p = "paper.txt"
    assert _norm_source(p) == "paper"


def test_norm_source_content_list_json_uses_parent() -> None:
    p = "sources/trading/optimal_hf_market_making/content_list.json"
    assert _norm_source(p) == "optimal_hf_market_making"


def test_dedupe_preserves_first_appearance_order() -> None:
    assert _dedupe_ranked(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]


def test_dedupe_empty() -> None:
    assert _dedupe_ranked([]) == []


# ---------------------------------------------------------------------------
# source-level metrics
# ---------------------------------------------------------------------------


def test_source_recall_full_match() -> None:
    retrieved = ["a.pdf", "b.pdf", "c.pdf"]
    ref = ["a.pdf", "c.pdf"]
    assert source_recall_at_k(retrieved, ref, k=15) == 1.0


def test_source_recall_partial() -> None:
    retrieved = ["a.pdf", "x.pdf"]
    ref = ["a.pdf", "b.pdf"]
    assert source_recall_at_k(retrieved, ref, k=15) == 0.5


def test_source_recall_zero() -> None:
    assert source_recall_at_k(["x.pdf"], ["a.pdf"], k=15) == 0.0


def test_source_recall_top_k_truncates() -> None:
    retrieved = ["x.pdf", "y.pdf", "a.pdf"]  # ref source is at rank 3
    ref = ["a.pdf"]
    assert source_recall_at_k(retrieved, ref, k=2) == 0.0
    assert source_recall_at_k(retrieved, ref, k=3) == 1.0


def test_source_recall_empty_reference_returns_zero() -> None:
    assert source_recall_at_k(["a.pdf"], [], k=15) == 0.0


def test_source_recall_normalizes_full_paths() -> None:
    retrieved = ["sources/coll_a/069__paper.pdf"]
    ref = ["sources/coll_b/069__paper.pdf"]
    assert source_recall_at_k(retrieved, ref, k=15) == 1.0


def test_source_hit_binary_one_match() -> None:
    assert source_hit_at_k(["a.pdf", "x.pdf"], ["a.pdf", "b.pdf"], k=15) == 1.0


def test_source_hit_zero_when_no_overlap() -> None:
    assert source_hit_at_k(["x.pdf"], ["a.pdf"], k=15) == 0.0


def test_source_hit_empty_reference() -> None:
    assert source_hit_at_k(["a.pdf"], [], k=15) == 0.0


def test_source_mrr_first_rank() -> None:
    assert source_mrr_at_k(["a.pdf", "b.pdf"], ["a.pdf"], k=15) == 1.0


def test_source_mrr_third_rank() -> None:
    retrieved = ["x.pdf", "y.pdf", "a.pdf", "b.pdf"]
    ref = ["a.pdf"]
    assert source_mrr_at_k(retrieved, ref, k=15) == 1.0 / 3.0


def test_source_mrr_no_match() -> None:
    assert source_mrr_at_k(["x.pdf"], ["a.pdf"], k=15) == 0.0


def test_source_mrr_outside_top_k() -> None:
    retrieved = ["x.pdf", "y.pdf", "a.pdf"]
    ref = ["a.pdf"]
    assert source_mrr_at_k(retrieved, ref, k=2) == 0.0


def test_source_metrics_with_basename_normalization() -> None:
    retrieved = ["sources/coll/069__paper.pdf", "sources/coll/038__other.pdf"]
    ref = ["sources/other_coll/069__paper.pdf"]
    assert source_hit_at_k(retrieved, ref, k=15) == 1.0
    assert source_recall_at_k(retrieved, ref, k=15) == 1.0
    assert source_mrr_at_k(retrieved, ref, k=15) == 1.0


# ---------------------------------------------------------------------------
# chunk-level metrics — regression coverage to confirm refactor didn't break them
# ---------------------------------------------------------------------------


def test_chunk_recall_full_match() -> None:
    assert recall_at_k(["a", "b", "c"], ["a", "c"], k=15) == 1.0


def test_chunk_hit_binary() -> None:
    assert hit_at_k(["a"], ["a"], k=15) == 1.0
    assert hit_at_k(["x"], ["a"], k=15) == 0.0


def test_chunk_mrr_rank_two() -> None:
    assert mrr_at_k(["x", "a"], ["a"], k=15) == 0.5


def test_chunk_metrics_empty_reference() -> None:
    assert recall_at_k(["a"], [], k=15) == 0.0
    assert hit_at_k(["a"], [], k=15) == 0.0
    assert mrr_at_k(["a"], [], k=15) == 0.0


# ---------------------------------------------------------------------------
# evaluate_from_logs — log + golden integration (no Milvus required when
# retrieved_sources is in the log AND no chunk IDs are emitted)
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_evaluate_source_only_log_skips_milvus(tmp_path: Path) -> None:
    """Logs with retrieved_sources and no retrieved_ids skip Milvus entirely.

    This is the Phase 2 case (graph retrievers don't have Milvus child IDs).
    """
    golden = tmp_path / "golden.jsonl"
    log = tmp_path / "queries.jsonl"
    _write_jsonl(
        golden,
        [
            {
                "question": "Q1",
                "reference_source_files": ["a.pdf", "b.pdf"],
            },
            {
                "question": "Q2",
                "reference_source_files": ["c.pdf"],
            },
        ],
    )
    _write_jsonl(
        log,
        [
            {
                "query": "Q1",
                "retrieved_ids": [],
                "retrieved_sources": ["a.pdf", "b.pdf", "x.pdf"],
            },
            {
                "query": "Q2",
                "retrieved_ids": [],
                "retrieved_sources": ["x.pdf", "c.pdf"],
            },
        ],
    )
    report = evaluate_from_logs(
        golden_path=golden,
        queries_log_path=log,
        collection="<unused>",
        k_values=(5,),
    )
    assert report["n_questions"] == 2
    assert report["source_aggregate"]["src_hit@5"] == 1.0
    assert report["source_aggregate"]["src_recall@5"] == 1.0
    assert report["aggregate"] == {}


def test_evaluate_partial_golden_emits_only_available_metrics(tmp_path: Path) -> None:
    """Golden with no reference fields produces no metrics (no crash)."""
    golden = tmp_path / "golden.jsonl"
    log = tmp_path / "queries.jsonl"
    _write_jsonl(golden, [{"question": "Q1"}])
    _write_jsonl(
        log,
        [{"query": "Q1", "retrieved_ids": [], "retrieved_sources": ["a.pdf"]}],
    )
    report = evaluate_from_logs(
        golden_path=golden,
        queries_log_path=log,
        collection="<unused>",
        k_values=(5,),
    )
    assert report["n_questions"] == 1
    assert report["source_aggregate"] == {}
    assert report["aggregate"] == {}
    assert report["per_question"][0]["scores"] == {}


def test_evaluate_normalizes_full_paths_in_log(tmp_path: Path) -> None:
    """Logs that emit full paths (defensive programming) still score correctly."""
    golden = tmp_path / "golden.jsonl"
    log = tmp_path / "queries.jsonl"
    _write_jsonl(golden, [{"question": "Q1", "reference_source_files": ["a.pdf"]}])
    _write_jsonl(
        log,
        [
            {
                "query": "Q1",
                "retrieved_ids": [],
                "retrieved_sources": ["sources/coll/a.pdf"],
            }
        ],
    )
    report = evaluate_from_logs(
        golden_path=golden,
        queries_log_path=log,
        collection="<unused>",
        k_values=(5,),
    )
    assert report["source_aggregate"]["src_hit@5"] == 1.0


def test_evaluate_missing_log_row_raises(tmp_path: Path) -> None:
    golden = tmp_path / "golden.jsonl"
    log = tmp_path / "queries.jsonl"
    _write_jsonl(golden, [{"question": "Q-missing", "reference_source_files": ["a.pdf"]}])
    log.write_text("")
    try:
        evaluate_from_logs(
            golden_path=golden,
            queries_log_path=log,
            collection="<unused>",
            k_values=(5,),
        )
    except ValueError as e:
        assert "No cached row" in str(e)
    else:
        raise AssertionError("expected ValueError for missing log row")
