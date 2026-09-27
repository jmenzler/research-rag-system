# long-ok-file
"""Tests for src/eval/audit_aggregates.py — MMR activation + latency rollups.

The module reads per-query audit directories and computes summaries:
  - MMR activation rate (fraction of queries where MMR swapped at least one chunk)
  - MMR drop overlap with golden reference_chunk_ids
  - Per-stage latency percentiles (medians, p95)

Tests construct fake audit directories on disk. No LLM or Milvus calls.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_meta(base: str | Path, totals: dict[str, Any]) -> None:
    base = Path(base)
    base.mkdir(parents=True, exist_ok=True)
    (base / "meta.json").write_text(json.dumps({"totals": totals}))


def _write_rerank(
    base: str | Path,
    mmr: dict[str, Any] | None = None,
    survivors: list[dict[str, Any]] | None = None,
) -> None:
    base = Path(base)
    base.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {}
    if survivors is not None:
        payload["survivors"] = survivors
    if mmr is not None:
        payload["mmr"] = mmr
    (base / "04_rerank.json").write_text(json.dumps(payload))


def _make_audit_dirs(tmp_path: Path) -> tuple[str, str, str]:
    """Create three fake audit dirs and return their paths."""
    d1 = tmp_path / "dir1"
    d2 = tmp_path / "dir2"
    d3 = tmp_path / "dir3"
    return str(d1), str(d2), str(d3)


# ---------------------------------------------------------------------------
# Slice 1 — MMR activation rate
# ---------------------------------------------------------------------------


class TestMMRActivationRate:
    def test_all_active_when_all_have_nonempty_dropped(self, tmp_path: Path) -> None:
        from src.eval.audit_aggregates import mmr_activation_rate

        d1, d2, d3 = _make_audit_dirs(tmp_path)
        for d in (d1, d2, d3):
            _write_rerank(d, mmr={"dropped": ["chunk_x"]})

        rate = mmr_activation_rate([d1, d2, d3])
        assert rate == 1.0

    def test_none_active_when_no_mmr(self, tmp_path: Path) -> None:
        from src.eval.audit_aggregates import mmr_activation_rate

        d1, d2, d3 = _make_audit_dirs(tmp_path)
        for d in (d1, d2, d3):
            _write_rerank(d, mmr=None)

        rate = mmr_activation_rate([d1, d2, d3])
        assert rate == 0.0

    def test_none_active_when_dropped_is_empty(self, tmp_path: Path) -> None:
        from src.eval.audit_aggregates import mmr_activation_rate

        d1, d2, _ = _make_audit_dirs(tmp_path)
        _write_rerank(d1, mmr={"dropped": []})
        _write_rerank(d2, mmr={"dropped": []})

        rate = mmr_activation_rate([d1, d2])
        assert rate == 0.0

    def test_partial_activation(self, tmp_path: Path) -> None:
        from src.eval.audit_aggregates import mmr_activation_rate

        d1, d2, d3 = _make_audit_dirs(tmp_path)
        _write_rerank(d1, mmr={"dropped": ["x"]})
        _write_rerank(d2, mmr={"dropped": []})
        _write_rerank(d3, mmr=None)

        rate = mmr_activation_rate([d1, d2, d3])
        assert rate == pytest.approx(1 / 3)

    def test_skips_missing_rerank_file(self, tmp_path: Path) -> None:
        from src.eval.audit_aggregates import mmr_activation_rate

        d1, d2, _ = _make_audit_dirs(tmp_path)
        _write_rerank(d1, mmr={"dropped": ["x"]})
        # d2 has no 04_rerank.json at all

        rate = mmr_activation_rate([d1, d2])
        assert rate == 0.5  # 1 of 2 (d2 skipped, counts as not active)

    def test_returns_none_when_no_dirs(self) -> None:
        from src.eval.audit_aggregates import mmr_activation_rate

        assert mmr_activation_rate([]) is None


# ---------------------------------------------------------------------------
# Slice 2 — MMR drop overlap with golden
# ---------------------------------------------------------------------------


class TestMMRDropOverlap:
    def test_full_overlap(self, tmp_path: Path) -> None:
        from src.eval.audit_aggregates import mmr_drop_overlap

        d1, d2, _ = _make_audit_dirs(tmp_path)
        _write_rerank(d1, mmr={"dropped": ["pid_a", "pid_b"]})
        _write_rerank(d2, mmr={"dropped": ["pid_a"]})

        # Golden refs across all queries: pid_a, pid_c
        golden_refs = [{"reference_chunk_ids": ["pid_a", "pid_c"]}] * 2

        overlap = mmr_drop_overlap([d1, d2], golden_refs)
        # Dropped total: pid_a, pid_b, pid_a = 3 items
        # In golden: pid_a (2×), pid_b (0×) = 2 of 3
        assert overlap == pytest.approx(2 / 3)

    def test_no_overlap(self, tmp_path: Path) -> None:
        from src.eval.audit_aggregates import mmr_drop_overlap

        d1 = _make_audit_dirs(tmp_path)[0]
        _write_rerank(d1, mmr={"dropped": ["pid_z"]})

        golden_refs = [{"reference_chunk_ids": ["pid_a"]}]

        overlap = mmr_drop_overlap([d1], golden_refs)
        assert overlap == 0.0

    def test_no_mmr_anywhere_returns_none(self, tmp_path: Path) -> None:
        from src.eval.audit_aggregates import mmr_drop_overlap

        d1 = _make_audit_dirs(tmp_path)[0]
        _write_rerank(d1, mmr=None)

        overlap = mmr_drop_overlap([d1], [{"reference_chunk_ids": ["pid_a"]}])
        assert overlap is None

    def test_no_golden_refs_returns_none(self, tmp_path: Path) -> None:
        from src.eval.audit_aggregates import mmr_drop_overlap

        d1 = _make_audit_dirs(tmp_path)[0]
        _write_rerank(d1, mmr={"dropped": ["pid_a"]})

        overlap = mmr_drop_overlap([d1], [])
        assert overlap is None

    def test_resolves_dropped_child_ids_to_parents(self, tmp_path: Path) -> None:
        """dropped holds CHILD ids; goldens hold PARENT ids — resolve via survivors.

        Without resolution the child↔parent namespaces never intersect and the
        metric is permanently 0.0 (false reassurance).
        """
        from src.eval.audit_aggregates import mmr_drop_overlap

        d1 = _make_audit_dirs(tmp_path)[0]
        _write_rerank(
            d1,
            mmr={"dropped": ["child_1", "child_2"]},
            survivors=[
                {"child_id": "child_1", "parent_chunk_id": "pid_a"},
                {"child_id": "child_2", "parent_chunk_id": "pid_z"},
            ],
        )
        # Golden refs are PARENT ids. pid_a is a golden ref, pid_z is not.
        golden_refs = [{"reference_chunk_ids": ["pid_a", "pid_c"]}]

        overlap = mmr_drop_overlap([d1], golden_refs)
        # child_1→pid_a (in golden), child_2→pid_z (not) → 1 of 2.
        assert overlap == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Slice 3 — latency percentiles
# ---------------------------------------------------------------------------


class TestLatencyPercentiles:
    def test_median_and_p95(self, tmp_path: Path) -> None:
        from src.eval.audit_aggregates import latency_percentiles

        d1, d2, d3 = _make_audit_dirs(tmp_path)
        _write_meta(d1, {"synthesis_latency_ms": 100, "retrieval_latency_ms": 200})
        _write_meta(d2, {"synthesis_latency_ms": 200, "retrieval_latency_ms": 300})
        _write_meta(d3, {"synthesis_latency_ms": 300, "retrieval_latency_ms": 100})

        result = latency_percentiles([d1, d2, d3])

        # synthesis: [100, 200, 300] → p50=200, p95=290 (np default linear interp)
        assert result["latency_p50_ms"]["synthesis"] == 200.0
        assert result["latency_p95_ms"]["synthesis"] == 290.0
        # retrieval: [200, 300, 100] → sorted [100, 200, 300] → p50=200
        assert result["latency_p50_ms"]["retrieval"] == 200.0

    def test_skips_missing_meta(self, tmp_path: Path) -> None:
        from src.eval.audit_aggregates import latency_percentiles

        d1 = _make_audit_dirs(tmp_path)[0]
        d2 = _make_audit_dirs(tmp_path)[1]
        _write_meta(d1, {"synthesis_latency_ms": 100})
        # d2 has no meta.json

        result = latency_percentiles([d1, d2])
        # Only d1 contributes → p50 = p95 = 100
        assert result["latency_p50_ms"]["synthesis"] == 100.0

    def test_returns_empty_when_no_data(self, tmp_path: Path) -> None:
        from src.eval.audit_aggregates import latency_percentiles

        result = latency_percentiles([])
        assert result == {"latency_p50_ms": {}, "latency_p95_ms": {}}

    def test_includes_generic_stage_latencies_from_meta(self, tmp_path: Path) -> None:
        """meta.json totals may include stage-latency keys like rerank_latency_ms."""
        from src.eval.audit_aggregates import latency_percentiles

        d1 = _make_audit_dirs(tmp_path)[0]
        _write_meta(d1, {"synthesis_latency_ms": 500, "rerank_latency_ms": 80})

        result = latency_percentiles([d1])
        assert result["latency_p50_ms"]["synthesis"] == 500.0
        assert result["latency_p50_ms"]["rerank"] == 80.0
