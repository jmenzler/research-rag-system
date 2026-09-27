"""Tests for the per-query audit-trail writer (src/query/query_logger.py).

Covers directory structure, usage accumulation, meta.json content,
partial-crash resilience, chunk header format, and cost-null on unknown model.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.models import ChildChunk, ParentChunk, RetrievedChunk, Usage
from src.query.query_logger import QueryLogger

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_logger(tmp_path: Path) -> QueryLogger:
    """A QueryLogger pointed at a temp directory with no API key side-effects."""
    return QueryLogger(
        query="What is VPIN?",
        root=tmp_path / "logs" / "queries",
        config_snapshot={"gen_model": "deepseek-v4-pro", "use_mmr": False},
    )


def _fake_rc(
    *,
    child_id: str = "abc123",
    parent_id: str = "parentXYZ",
    rerank_score: float = 0.85,
) -> RetrievedChunk:
    child = ChildChunk(
        id=child_id,
        parent_id=parent_id,
        text="child text snippet",
        source_file="sources/trading/vpin_paper/source.pdf",
        notebook="trading",
        modality="text",
        page_number=5,
    )
    parent = ParentChunk(
        id=parent_id,
        text="Full parent context. This is what synthesis sees.",
        source_file="sources/trading/vpin_paper/source.pdf",
        notebook="trading",
        modality="text",
        page_number=5,
    )
    return RetrievedChunk(
        child=child,
        parent=parent,
        dense_score=0.91,
        sparse_score=0.0,
        rerank_score=rerank_score,
    )


def _fake_usage(
    *,
    input: int = 1000,
    output: int = 200,
    reasoning: int = 0,
    total: int = 1200,
    input_cache_hit: int = 0,
    input_cache_miss: int = 0,
) -> Usage:
    return Usage(
        input=input,
        output=output,
        reasoning=reasoning,
        total=total,
        input_cache_hit=input_cache_hit,
        input_cache_miss=input_cache_miss,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDirectoryStructure:
    def test_log_dir_created(self, tmp_logger: QueryLogger) -> None:
        assert tmp_logger.log_dir.exists()
        assert tmp_logger.log_dir.is_dir()

    def test_chunks_subdir_created(self, tmp_logger: QueryLogger) -> None:
        chunks_dir = tmp_logger.log_dir / "chunks"
        assert chunks_dir.exists()
        assert chunks_dir.is_dir()

    def test_dir_name_format(self, tmp_logger: QueryLogger) -> None:
        # Format after PR #7 Medium-3 fix:
        #   HHMMSSffffffZ_<8-char hex>_<pid>_<tid>
        # Microsecond precision + PID + TID to guarantee uniqueness even when
        # two threads start at the exact same microsecond on the same query.
        name = tmp_logger.log_dir.name
        parts = name.split("_")
        assert len(parts) == 4, (
            f"Expected 4 underscore-separated parts (time_hash_pid_tid), got {len(parts)}: {name!r}"
        )
        time_part, hash_part, pid_part, tid_part = parts
        assert time_part.endswith("Z")
        # HHMMSS + 6 microsecond digits + Z = 13 chars
        assert len(time_part) == 13, (
            f"Expected a 13-character time part, got {len(time_part)}: {time_part!r}"
        )
        assert len(hash_part) == 8
        assert pid_part.isdigit(), f"PID part must be numeric, got {pid_part!r}"
        assert tid_part.isdigit(), f"TID part must be numeric, got {tid_part!r}"

    def test_same_query_same_hash(self, tmp_path: Path) -> None:
        """Two loggers for the same query produce the same hash suffix."""
        q = "What is VPIN?"
        log1 = QueryLogger(q, root=tmp_path / "a")
        log2 = QueryLogger(q, root=tmp_path / "b")
        hash1 = log1.log_dir.name.split("_")[1]
        hash2 = log2.log_dir.name.split("_")[1]
        assert hash1 == hash2

    def test_concurrent_same_query_distinct_directories(self, tmp_path: Path) -> None:
        """Two concurrent threads with identical query text must not collide.

        Slice 4: Microsecond precision prevents same-second collisions. Even if
        two threads start in the same second, the microsecond suffix differs.
        """
        import threading

        q = "What is VPIN?"
        dirs: list[Path] = []
        barrier = threading.Barrier(2)

        def make_logger() -> None:
            barrier.wait()  # Force simultaneous construction
            ql = QueryLogger(q, root=tmp_path / "logs")
            dirs.append(ql.log_dir)

        t1 = threading.Thread(target=make_logger)
        t2 = threading.Thread(target=make_logger)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert len(dirs) == 2
        # With microsecond precision they are almost certainly distinct;
        # they MUST be distinct for write isolation.
        assert dirs[0] != dirs[1], (
            "Two concurrent QueryLoggers for the same query collided to the same "
            f"directory: {dirs[0]}. Use microsecond precision (%H%M%S%f) to fix."
        )


class TestWriteStage:
    def test_stage_file_written(self, tmp_logger: QueryLogger) -> None:
        tmp_logger.write_stage("decompose", {"model": "deepseek-v4-flash", "latency_ms": 123})
        stage_file = tmp_logger.log_dir / "01_decompose.json"
        assert stage_file.exists()
        data = json.loads(stage_file.read_text())
        assert data["model"] == "deepseek-v4-flash"
        assert data["latency_ms"] == 123

    def test_known_stage_prefix(self, tmp_logger: QueryLogger) -> None:
        for stage, prefix in [
            ("decompose", "01"),
            ("stepback", "02"),
            ("milvus", "03"),
            ("rerank", "04"),
            ("mmr", "05"),
            ("synthesis", "06"),
            ("grader", "07"),
            ("crag_retry", "08"),
        ]:
            tmp_logger.write_stage(stage, {"x": 1})
            assert (tmp_logger.log_dir / f"{prefix}_{stage}.json").exists()

    def test_unknown_stage_gets_99_prefix(self, tmp_logger: QueryLogger) -> None:
        tmp_logger.write_stage("future_stage", {"y": 2})
        assert (tmp_logger.log_dir / "99_future_stage.json").exists()

    def test_stage_json_is_sorted_keys(self, tmp_logger: QueryLogger) -> None:
        tmp_logger.write_stage("milvus", {"z": 1, "a": 2, "m": 3})
        raw = (tmp_logger.log_dir / "03_milvus.json").read_text()
        # sorted keys → "a" appears before "m" appears before "z"
        assert raw.index('"a"') < raw.index('"m"') < raw.index('"z"')


class TestAccumulateUsageAndFinalize:
    def test_totals_sum_across_stages(self, tmp_logger: QueryLogger, tmp_path: Path) -> None:
        usage_a = _fake_usage(input=500, output=100, total=600)
        usage_b = _fake_usage(input=800, output=300, total=1100)
        tmp_logger.accumulate_usage("decompose", usage_a, 0.0001)
        tmp_logger.accumulate_usage("synthesis", usage_b, 0.0020)
        tmp_logger.finalize(total_latency_ms=4500)

        meta = json.loads((tmp_logger.log_dir / "meta.json").read_text())
        assert meta["totals"]["tokens"]["input"] == 1300
        assert meta["totals"]["tokens"]["output"] == 400
        assert meta["totals"]["n_llm_calls"] == 2
        assert meta["totals"]["cost_usd"] == pytest.approx(0.0021)

    def test_meta_json_has_required_fields(self, tmp_logger: QueryLogger) -> None:
        tmp_logger.finalize(total_latency_ms=1000)
        meta = json.loads((tmp_logger.log_dir / "meta.json").read_text())
        for key in ("query", "query_hash", "ts_started", "ts_ended", "totals", "outcome"):
            assert key in meta, f"Missing key: {key}"

    def test_meta_json_includes_pid_and_tid(self, tmp_logger: QueryLogger) -> None:
        """pid and tid must be in meta.json as JSON fields, not just in the
        directory name. The dir name encodes them for collision avoidance, but
        downstream tooling reading only meta.json needs them too (e.g. to filter
        all queries from one parallel-eval worker process)."""
        import os
        import threading

        tmp_logger.finalize(total_latency_ms=1000)
        meta = json.loads((tmp_logger.log_dir / "meta.json").read_text())
        assert meta.get("pid") == os.getpid(), (
            f"meta.json pid expected={os.getpid()}, got={meta.get('pid')!r}"
        )
        assert meta.get("tid") == threading.get_ident(), (
            f"meta.json tid expected={threading.get_ident()}, got={meta.get('tid')!r}"
        )

    def test_outcome_defaults(self, tmp_logger: QueryLogger) -> None:
        tmp_logger.finalize(total_latency_ms=0)
        meta = json.loads((tmp_logger.log_dir / "meta.json").read_text())
        assert meta["outcome"]["synthesis_skipped"] is False
        assert meta["outcome"]["crag_retried"] is False
        assert meta["outcome"]["crag_chose"] is None

    def test_synthesis_skipped_flag(self, tmp_logger: QueryLogger) -> None:
        tmp_logger.synthesis_skipped = True
        tmp_logger.finalize(total_latency_ms=500)
        meta = json.loads((tmp_logger.log_dir / "meta.json").read_text())
        assert meta["outcome"]["synthesis_skipped"] is True

    def test_cost_none_when_all_costs_none(self, tmp_logger: QueryLogger) -> None:
        """If no stage has a known cost, totals.cost_usd should be None."""
        tmp_logger.accumulate_usage("synthesis", _fake_usage(), None)
        tmp_logger.finalize(total_latency_ms=100)
        meta = json.loads((tmp_logger.log_dir / "meta.json").read_text())
        assert meta["totals"]["cost_usd"] is None

    def test_finalize_is_idempotent(self, tmp_logger: QueryLogger) -> None:
        """Calling finalize twice should not raise; second call overwrites."""
        tmp_logger.finalize(total_latency_ms=100)
        tmp_logger.finalize(total_latency_ms=200)
        meta = json.loads((tmp_logger.log_dir / "meta.json").read_text())
        assert meta["totals"]["total_latency_ms"] == 200


class TestWriteChunk:
    def test_chunk_file_has_six_line_header(self, tmp_logger: QueryLogger) -> None:
        rc = _fake_rc()
        tmp_logger.write_chunk(1, rc)
        chunk_files = list((tmp_logger.log_dir / "chunks").iterdir())
        assert len(chunk_files) == 1
        content = chunk_files[0].read_text()
        lines = content.split("\n")
        # 6 header lines, blank line, then parent text
        assert "child_id:" in lines[0]
        assert "parent_chunk_id:" in lines[1]
        assert "source_file:" in lines[2]
        assert "page:" in lines[3]
        assert "rerank_score:" in lines[4]
        assert "modality:" in lines[5]
        assert lines[6] == ""  # blank separator

    def test_chunk_file_contains_parent_text(self, tmp_logger: QueryLogger) -> None:
        rc = _fake_rc()
        tmp_logger.write_chunk(1, rc)
        files = list((tmp_logger.log_dir / "chunks").iterdir())
        content = files[0].read_text()
        assert "Full parent context." in content

    def test_chunk_index_in_filename(self, tmp_logger: QueryLogger) -> None:
        rc = _fake_rc(child_id="chunk42")
        tmp_logger.write_chunk(7, rc)
        files = list((tmp_logger.log_dir / "chunks").iterdir())
        assert files[0].name.startswith("07_")


class TestPartialCrashResilience:
    def test_partial_dir_preserved_on_crash(self, tmp_path: Path) -> None:
        """Simulate a crash after writing decompose but before finalize."""
        audit = QueryLogger("crash test query", root=tmp_path / "logs")
        audit.write_stage("decompose", {"model": "x", "latency_ms": 10})
        # Simulate crash — do NOT call finalize.
        assert (audit.log_dir / "01_decompose.json").exists()
        # meta.json should NOT exist (finalize not called)
        assert not (audit.log_dir / "meta.json").exists()

    def test_finalize_still_writes_after_no_stages(self, tmp_path: Path) -> None:
        """finalize alone (no prior write_stage) still produces a valid meta.json."""
        audit = QueryLogger("bare finalize", root=tmp_path / "logs")
        audit.finalize(total_latency_ms=99)
        meta_path = audit.log_dir / "meta.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["totals"]["n_llm_calls"] == 0


class TestUnknownModelCostNull:
    def test_unknown_model_returns_none_not_crash(self, tmp_logger: QueryLogger) -> None:
        from src.config.models import compute_cost_usd

        result = compute_cost_usd("some-unknown-model-xyz", _fake_usage(input=1000, output=200))
        assert result is None  # should not raise

    def test_accumulate_none_cost_does_not_break_totals(self, tmp_logger: QueryLogger) -> None:
        tmp_logger.accumulate_usage("synthesis", _fake_usage(input=100, output=50), None)
        tmp_logger.finalize(total_latency_ms=500)
        meta = json.loads((tmp_logger.log_dir / "meta.json").read_text())
        # cost_usd is None when all stages report None
        assert meta["totals"]["cost_usd"] is None
        # tokens should still be populated
        assert meta["totals"]["tokens"]["input"] == 100
