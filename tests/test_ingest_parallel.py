"""Tests for the parallel ingest dispatcher in src.ingest.ingest.

Two boundaries are pinned here:

  1. ``_auto_workers(n_paths, requested)`` — the worker-count heuristic.
     Single-doc dev calls stay serial (workers=1); bulk runs scale up to
     a cap of 8. An explicit ``requested`` value always wins.

  2. ``_run_shard(paths, collection, notebook, worker_id)`` — the per-shard
     entry point. Owns its own clients, returns aggregate counts, and is
     called both inline (workers=1) and from a ProcessPoolExecutor child
     (workers>1). The dispatcher (``main``) glues these together; the
     contract tested here is what the worker function returns.

The shard worker invokes ``ingest_file`` for each path. We mock that
function so the test exercises the dispatch logic without touching
Milvus, OpenRouter, or sqlite. The ``_run_shard`` contract under test
is independent of which backend ``ingest_file`` writes to.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.ingest.ingest import _auto_workers, _resolve_paths, _run_shard

# ---------------------------------------------------------------------------
# _auto_workers — pure function, no fixtures needed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n_paths,requested,expected",
    [
        # Explicit override wins, capped by n_paths so we don't fork empty workers.
        (10, 4, 4),
        (10, 1, 1),
        (3, 8, 3),
        # Auto path (requested=0): cap at 8, scale by N//4, never below 1.
        (1, 0, 1),  # single dev doc → serial
        (3, 0, 1),  # tiny batch → serial
        (4, 0, 1),  # 4 // 4 == 1
        (8, 0, 2),
        (32, 0, 8),
        (100, 0, 8),  # capped
        (2400, 0, 8),  # large bulk → still 8
    ],
)
def test_auto_workers_heuristic(n_paths: int, requested: int, expected: int) -> None:
    assert _auto_workers(n_paths, requested) == expected


# ---------------------------------------------------------------------------
# _resolve_paths — sorted output is the round-robin precondition
# ---------------------------------------------------------------------------


def test_resolve_paths_returns_sorted(tmp_path: Path) -> None:
    """Round-robin sharding is deterministic only if path resolution is sorted."""
    for name in ("z.txt", "a.txt", "m.txt"):
        (tmp_path / name).write_text("x")
    resolved = _resolve_paths([str(tmp_path / "*.txt")])
    assert [p.name for p in resolved] == ["a.txt", "m.txt", "z.txt"]


def test_resolve_paths_combines_globs_and_literal_paths(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("x")
    (tmp_path / "b.txt").write_text("x")
    resolved = _resolve_paths([str(tmp_path / "*.txt"), str(tmp_path / "a.txt")])
    # 'a.txt' appears once via glob, once via literal — both kept; dedup not the
    # function's job (caller can shard duplicates harmlessly).
    assert sum(1 for p in resolved if p.name == "a.txt") == 2


# ---------------------------------------------------------------------------
# _run_shard — contract: returns aggregate counts; one ingest_file call per path
# ---------------------------------------------------------------------------


def _ingest_summary(n_parents: int = 3, n_children: int = 12) -> dict[str, Any]:
    return {"n_parents": n_parents, "n_children": n_children}


@patch("src.ingest.ingest._append_log")
@patch("src.ingest.ingest._open_parents_db")
@patch("src.ingest.ingest._get_encoder")
@patch("src.ingest.ingest._build_gemini_client")
@patch("src.ingest.ingest.ensure_partition")
@patch("src.ingest.ingest.ensure_collection")
@patch("src.ingest.ingest.config.validate_api_key")
@patch("src.ingest.ingest.ingest_file")
def test_run_shard_aggregates_counts(
    ingest_file_mock: MagicMock,
    _validate_mock: MagicMock,
    _ensure_col: MagicMock,
    _ensure_part: MagicMock,
    _build_gem: MagicMock,
    _enc: MagicMock,
    open_db: MagicMock,
    _append: MagicMock,
    tmp_path: Path,
) -> None:
    open_db.return_value = MagicMock()
    ingest_file_mock.side_effect = [
        _ingest_summary(2, 8),
        _ingest_summary(3, 12),
        {"skipped": True, "reason": "already_ingested"},
    ]
    paths = []
    for name in ("a.json", "b.json", "c.json"):
        p = tmp_path / name
        p.write_text("{}")
        paths.append(str(p))

    result = _run_shard(paths, collection="trading", notebook="trading", worker_id=1)

    # ingest_file called once per shard path.
    assert ingest_file_mock.call_count == 3
    # Aggregate counts only count successful (non-skip, non-fail) docs.
    assert result == {
        "n_parents": 5,
        "n_children": 20,
        "n_ok": 2,
        "n_fail": 0,
        "n_skip": 1,
    }


@patch("src.ingest.ingest._append_log")
@patch("src.ingest.ingest._open_parents_db")
@patch("src.ingest.ingest._get_encoder")
@patch("src.ingest.ingest._build_gemini_client")
@patch("src.ingest.ingest.ensure_partition")
@patch("src.ingest.ingest.ensure_collection")
@patch("src.ingest.ingest.config.validate_api_key")
@patch("src.ingest.ingest.ingest_file")
def test_run_shard_continues_after_per_doc_failure(
    ingest_file_mock: MagicMock,
    _validate_mock: MagicMock,
    _ensure_col: MagicMock,
    _ensure_part: MagicMock,
    _build_gem: MagicMock,
    _enc: MagicMock,
    open_db: MagicMock,
    _append: MagicMock,
    tmp_path: Path,
) -> None:
    """A single doc failure must not abort the rest of the shard.

    Worker isolation only helps if individual failures are absorbed at the
    per-doc loop. If we raise, the worker process dies and its remaining
    docs are silently lost.
    """
    open_db.return_value = MagicMock()
    ingest_file_mock.side_effect = [
        _ingest_summary(2, 8),
        RuntimeError("MinerU OOM on this doc"),
        _ingest_summary(1, 4),
    ]
    paths = []
    for name in ("a.json", "b.json", "c.json"):
        p = tmp_path / name
        p.write_text("{}")
        paths.append(str(p))

    result = _run_shard(paths, collection="trading", notebook="trading", worker_id=2)

    assert ingest_file_mock.call_count == 3
    assert result["n_ok"] == 2
    assert result["n_fail"] == 1
    assert result["n_skip"] == 0
    assert result["n_parents"] == 3
    assert result["n_children"] == 12
