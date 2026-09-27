# long-ok-file
"""Tests for src/eval/stage_recall.py — stage-decomposed chunk recall@k.

The module reads per-query audit directories and computes recall@k of golden
reference_chunk_ids at three pipeline stages: post-Milvus, post-rerank, post-MMR.

Tests construct fake audit directories on disk so no Milvus or LLM calls are needed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_milvus_json(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a realistic 03_milvus.json payload."""
    return {
        "sub_queries": [
            {
                "sub_query": "what is AS model?",
                "candidates": candidates,
            }
        ]
    }


def _fake_rerank_json(
    survivors: list[dict[str, Any]],
    mmr: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a realistic 04_rerank.json payload."""
    payload: dict[str, Any] = {
        "survivors": survivors,
        "config": {"score_threshold": 0.3, "max_children_per_parent": 1},
    }
    if mmr is not None:
        payload["mmr"] = mmr
    return payload


def _write_audit_dir(base: Path, milvus: dict | None, rerank: dict | None) -> Path:
    """Write a minimal fake audit directory and return its path."""
    base.mkdir(parents=True, exist_ok=True)
    if milvus is not None:
        (base / "03_milvus.json").write_text(json.dumps(milvus))
    if rerank is not None:
        (base / "04_rerank.json").write_text(json.dumps(rerank))
    return base


# Child IDs (Milvus stores these) and parent IDs (golden refs use these).
# They are in different namespaces because the chunk_id hash includes the kind.
CID_A = "aaaa111122223333"
CID_B = "bbbb111122223333"
CID_C = "cccc111122223333"
CID_D = "dddd111122223333"
PID_A = "a1a1a1a1a1a1a1a1"
PID_B = "b1b1b1b1b1b1b1b1"
PID_C = "c1c1c1c1c1c1c1c1"
PID_D = "d1d1d1d1d1d1d1d1"


# ---------------------------------------------------------------------------
# Slice 1 — audit file reading helpers
# ---------------------------------------------------------------------------


class TestReadMilvusChildIDs:
    def test_extracts_child_ids_from_candidates(self, tmp_path: Path) -> None:
        from src.eval.stage_recall import _read_milvus_child_ids

        audit_dir = _write_audit_dir(
            tmp_path,
            milvus=_fake_milvus_json(
                [
                    {"id": CID_A, "source_file": "a.pdf", "page": 1, "rrf_score": 0.9},
                    {"id": CID_B, "source_file": "b.pdf", "page": 2, "rrf_score": 0.8},
                ]
            ),
            rerank=None,
        )
        ids = _read_milvus_child_ids(str(audit_dir))
        assert ids == [CID_A, CID_B]

    def test_dedupes_across_sub_queries(self, tmp_path: Path) -> None:
        from src.eval.stage_recall import _read_milvus_child_ids

        milvus: dict[str, Any] = {
            "sub_queries": [
                {
                    "sub_query": "q1",
                    "candidates": [
                        {"id": CID_A, "source_file": "a.pdf", "page": 1, "rrf_score": 0.9},
                    ],
                },
                {
                    "sub_query": "q2",
                    "candidates": [
                        {"id": CID_A, "source_file": "a.pdf", "page": 1, "rrf_score": 0.85},
                        {"id": CID_B, "source_file": "b.pdf", "page": 2, "rrf_score": 0.8},
                    ],
                },
            ]
        }
        audit_dir = _write_audit_dir(tmp_path, milvus=milvus, rerank=None)
        ids = _read_milvus_child_ids(str(audit_dir))
        # CID_A appears in both sub-queries but should only appear once.
        assert ids == [CID_A, CID_B]

    def test_returns_empty_list_when_milvus_file_missing(self, tmp_path: Path) -> None:
        from src.eval.stage_recall import _read_milvus_child_ids

        audit_dir = _write_audit_dir(tmp_path, milvus=None, rerank=None)
        ids = _read_milvus_child_ids(str(audit_dir))
        assert ids == []

    def test_returns_empty_list_when_no_candidates(self, tmp_path: Path) -> None:
        from src.eval.stage_recall import _read_milvus_child_ids

        audit_dir = _write_audit_dir(
            tmp_path,
            milvus=_fake_milvus_json([]),
            rerank=None,
        )
        ids = _read_milvus_child_ids(str(audit_dir))
        assert ids == []


class TestReadRerankParentIDs:
    def test_extracts_parent_ids_from_survivors(self, tmp_path: Path) -> None:
        from src.eval.stage_recall import _read_rerank_parent_ids

        audit_dir = _write_audit_dir(
            tmp_path,
            milvus=None,
            rerank=_fake_rerank_json(
                [
                    {"child_id": CID_A, "parent_chunk_id": PID_A, "rerank_score": 0.9},
                    {"child_id": CID_B, "parent_chunk_id": PID_B, "rerank_score": 0.7},
                ]
            ),
        )
        pids = _read_rerank_parent_ids(str(audit_dir))
        assert pids == [PID_A, PID_B]

    def test_returns_empty_when_rerank_missing(self, tmp_path: Path) -> None:
        from src.eval.stage_recall import _read_rerank_parent_ids

        audit_dir = _write_audit_dir(tmp_path, milvus=None, rerank=None)
        pids = _read_rerank_parent_ids(str(audit_dir))
        assert pids == []


class TestReadFinalParentIDs:
    def test_uses_mmr_picks_when_mmr_present(self, tmp_path: Path) -> None:
        from src.eval.stage_recall import _read_final_parent_ids

        survivors = [
            {"child_id": CID_A, "parent_chunk_id": PID_A, "rerank_score": 0.9},
            {"child_id": CID_B, "parent_chunk_id": PID_B, "rerank_score": 0.8},
            {"child_id": CID_C, "parent_chunk_id": PID_C, "rerank_score": 0.7},
        ]
        mmr: dict[str, Any] = {
            "lambda": 0.7,
            "top_k": 2,
            "pool_size": 3,
            "pool": [
                {"child_id": CID_A, "rerank_score": 0.9},
                {"child_id": CID_B, "rerank_score": 0.8},
                {"child_id": CID_C, "rerank_score": 0.7},
            ],
            "picks": [
                {"child_id": CID_A, "pick_order": 0, "rerank_score": 0.9},
                {"child_id": CID_C, "pick_order": 1, "rerank_score": 0.7},
            ],
            "dropped": [CID_B],
        }
        rerank = _fake_rerank_json(survivors, mmr=mmr)
        audit_dir = _write_audit_dir(tmp_path, milvus=None, rerank=rerank)
        pids = _read_final_parent_ids(str(audit_dir))
        # MMR picked A and C, dropped B.
        assert pids == [PID_A, PID_C]

    def test_falls_back_to_survivors_when_no_mmr(self, tmp_path: Path) -> None:
        from src.eval.stage_recall import _read_final_parent_ids

        survivors = [
            {"child_id": CID_A, "parent_chunk_id": PID_A, "rerank_score": 0.9},
            {"child_id": CID_B, "parent_chunk_id": PID_B, "rerank_score": 0.8},
        ]
        audit_dir = _write_audit_dir(tmp_path, milvus=None, rerank=_fake_rerank_json(survivors))
        pids = _read_final_parent_ids(str(audit_dir))
        assert pids == [PID_A, PID_B]

    def test_returns_empty_when_rerank_missing(self, tmp_path: Path) -> None:
        from src.eval.stage_recall import _read_final_parent_ids

        audit_dir = _write_audit_dir(tmp_path, milvus=None, rerank=None)
        pids = _read_final_parent_ids(str(audit_dir))
        assert pids == []


# ---------------------------------------------------------------------------
# Slice 2 — compute_stage_recall (the main entry point)
# ---------------------------------------------------------------------------


class TestComputeStageRecall:
    def test_all_three_stages_when_full_data_present(self, tmp_path: Path) -> None:
        from src.eval.stage_recall import compute_stage_recall

        # Milvus returns child IDs for A, B, C, D
        milvus = _fake_milvus_json(
            [
                {"id": CID_A, "source_file": "a.pdf", "page": 1, "rrf_score": 0.95},
                {"id": CID_B, "source_file": "b.pdf", "page": 2, "rrf_score": 0.85},
                {"id": CID_C, "source_file": "c.pdf", "page": 3, "rrf_score": 0.75},
                {"id": CID_D, "source_file": "d.pdf", "page": 4, "rrf_score": 0.65},
            ]
        )

        # Rerank keeps survivors B, C, D (A dropped by threshold)
        survivors = [
            {"child_id": CID_B, "parent_chunk_id": PID_B, "rerank_score": 0.85},
            {"child_id": CID_C, "parent_chunk_id": PID_C, "rerank_score": 0.75},
            {"child_id": CID_D, "parent_chunk_id": PID_D, "rerank_score": 0.65},
        ]

        # MMR picks C, D (B dropped for diversity)
        mmr: dict[str, Any] = {
            "lambda": 0.7,
            "top_k": 2,
            "pool_size": 3,
            "picks": [
                {"child_id": CID_C, "pick_order": 0, "rerank_score": 0.75},
                {"child_id": CID_D, "pick_order": 1, "rerank_score": 0.65},
            ],
            "dropped": [CID_B],
        }

        rerank = _fake_rerank_json(survivors, mmr=mmr)
        audit_dir = _write_audit_dir(tmp_path, milvus=milvus, rerank=rerank)

        # Golden references: parent IDs A, B (both are "ground truth")
        golden_refs = [PID_A, PID_B]

        # child→parent mapping so milvus child IDs can be resolved
        c2p = {CID_A: PID_A, CID_B: PID_B, CID_C: PID_C, CID_D: PID_D}

        result = compute_stage_recall(
            str(audit_dir),
            golden_refs,
            child_to_parent=c2p,
            k=20,
        )

        # recall_at_retrieve: milvus has A,B,C,D → parents A,B,C,D → intersection {A,B} = 2/2 = 1.0
        assert result["recall_at_retrieve_k20"] == 1.0
        # recall_at_rerank: survivors have B,C,D → parents B,C,D → intersection {B} = 1/2 = 0.5
        assert result["recall_at_rerank_k20"] == 0.5
        # recall_at_final: MMR picks C,D → parents C,D → intersection {} = 0/2 = 0.0
        assert result["recall_at_final_k20"] == 0.0

    def test_returns_none_when_audit_dir_missing(self) -> None:
        from src.eval.stage_recall import compute_stage_recall

        result = compute_stage_recall(
            "/nonexistent/path/12345",
            [PID_A],
            child_to_parent={},
        )
        assert result is None

    def test_returns_none_when_audit_dir_is_empty_string(self) -> None:
        from src.eval.stage_recall import compute_stage_recall

        result = compute_stage_recall("", [PID_A], child_to_parent={})
        assert result is None

    def test_empty_golden_refs_returns_none(self, tmp_path: Path) -> None:
        """No ground-truth chunk IDs → recall is undefined (None), not 0.0.

        Scoring these rows 0.0 averaged spurious zeros into stage recall for any
        notebook lacking golden reference_chunk_ids (the eval roll-up only
        filters None).
        """
        from src.eval.stage_recall import compute_stage_recall

        milvus = _fake_milvus_json(
            [
                {"id": CID_A, "source_file": "a.pdf", "page": 1, "rrf_score": 0.9},
            ]
        )
        survivors = [
            {"child_id": CID_A, "parent_chunk_id": PID_A, "rerank_score": 0.9},
        ]
        audit_dir = _write_audit_dir(tmp_path, milvus=milvus, rerank=_fake_rerank_json(survivors))
        c2p = {CID_A: PID_A}

        result = compute_stage_recall(str(audit_dir), [], child_to_parent=c2p)
        assert result is None

    def test_rerank_stage_omitted_when_rerank_file_missing(self, tmp_path: Path) -> None:
        from src.eval.stage_recall import compute_stage_recall

        milvus = _fake_milvus_json(
            [
                {"id": CID_A, "source_file": "a.pdf", "page": 1, "rrf_score": 0.9},
            ]
        )
        audit_dir = _write_audit_dir(tmp_path, milvus=milvus, rerank=None)
        c2p = {CID_A: PID_A}

        result = compute_stage_recall(str(audit_dir), [PID_A], child_to_parent=c2p)
        assert result is not None
        assert result["recall_at_retrieve_k20"] == 1.0
        # Rerank file missing → None for rerank/final stages.
        assert result["recall_at_rerank_k20"] is None
        assert result["recall_at_final_k20"] is None

    def test_milvus_stage_omitted_when_milvus_file_missing(self, tmp_path: Path) -> None:
        from src.eval.stage_recall import compute_stage_recall

        survivors = [
            {"child_id": CID_A, "parent_chunk_id": PID_A, "rerank_score": 0.9},
        ]
        audit_dir = _write_audit_dir(tmp_path, milvus=None, rerank=_fake_rerank_json(survivors))

        result = compute_stage_recall(str(audit_dir), [PID_A])
        assert result is not None
        assert result["recall_at_retrieve_k20"] is None
        assert result["recall_at_rerank_k20"] == 1.0
        assert result["recall_at_final_k20"] == 1.0

    def test_child_ids_without_parent_map_entry_score_zero(self, tmp_path: Path) -> None:
        """Child IDs not in child_to_parent are silently dropped."""
        from src.eval.stage_recall import compute_stage_recall

        milvus = _fake_milvus_json(
            [
                {"id": CID_A, "source_file": "a.pdf", "page": 1, "rrf_score": 0.9},
                {"id": CID_B, "source_file": "b.pdf", "page": 2, "rrf_score": 0.8},
            ]
        )
        audit_dir = _write_audit_dir(tmp_path, milvus=milvus, rerank=None)
        # Only CID_A resolves to a parent; CID_B has no mapping.
        c2p = {CID_A: PID_A}

        result = compute_stage_recall(str(audit_dir), [PID_A], child_to_parent=c2p)
        assert result is not None
        # CID_A resolves to PID_A which is in golden → recall = 1/1 = 1.0
        assert result["recall_at_retrieve_k20"] == 1.0
