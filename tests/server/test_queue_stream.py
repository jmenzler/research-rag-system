"""RED tests for queue SSE stream + stage-marker parse in src/server/api_ingest.py.

Covers INGEST-03 (stage-marker parse), INGEST-04 (crashed/cancelled terminal events),
INGEST-05 (done event shape), and security info-disclosure (no raw log_tail leak).

All tests skip at collection time until src.server.api_ingest exists.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from tests.server.conftest import FakeDiscoverQueue

pytest.importorskip("src.server.api_ingest")


# ---------------------------------------------------------------------------
# INGEST-03: _stage_from_log_tail parametrize
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "stage",
    ["fetching", "parsing", "chunking", "embedding", "unknown"],
)
def test_stage_from_log_tail(stage: str, log_tail_samples: dict[str, list[str]]) -> None:
    """INGEST-03: _stage_from_log_tail returns correct stage for each sample."""
    from src.server.api_ingest import _stage_from_log_tail  # noqa: PLC0415

    result = _stage_from_log_tail(log_tail_samples[stage])
    assert result == stage, (
        f"expected stage={stage!r}, got {result!r} for log_tail={log_tail_samples[stage]}"
    )


# ---------------------------------------------------------------------------
# INGEST-03: most-recent line wins
# ---------------------------------------------------------------------------


def test_stage_most_recent_wins() -> None:
    """INGEST-03: a tail with chunking line then embedding line returns 'embedding' (last wins)."""
    from src.server.api_ingest import _stage_from_log_tail  # noqa: PLC0415

    tail = [
        "Found 193 file(s) to process.",  # chunking marker
        "[007__1_strategy_layer_overview.txt]",  # chunking marker
        "  1 parents, 2 children — embedding…",  # embedding marker (later = wins)
    ]
    result = _stage_from_log_tail(tail)
    assert result == "embedding", f"expected 'embedding', got {result!r}"


# ---------------------------------------------------------------------------
# INGEST-03 regression: REAL pipeline log lines must not parse as "unknown"
# (these are verbatim from a live discover run — the markers the first cut missed)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("[2026-05-21 13:01:22]:(fetch_pipeline) INFO: Spider initialized", "fetching"),
        ("[2026-05-21 13:02:55]:(fetch_pipeline) INFO: Pipeline done: 6/7 ok", "fetching"),
        ("Processing pages:  96%|#########6| 26/27 [00:04<00:00,  7.25it/s]", "parsing"),
        ("MFR Predict:  84%|########3 | 1040/1243 [01:42<01:07,  2.99it/s]", "parsing"),
        ('INFO:     127.0.0.1:50928 - "POST /file_parse HTTP/1.1" 200 OK', "parsing"),
        ("  [progress] trading:6/10 pdf:1157/1157 (parse_pending=0)", "parsing"),
    ],
)
def test_stage_real_log_lines(line: str, expected: str) -> None:
    """Real-world fetch_pipeline + MinerU lines must map to a stage, never 'unknown'."""
    from src.server.api_ingest import _stage_from_log_tail  # noqa: PLC0415

    assert _stage_from_log_tail([line]) == expected


def test_meta_db_path_uses_citations_release() -> None:
    """D-02 regression: paper_meta path comes from DEFAULT_BASE/DEFAULT_RELEASE,
    not a hardcoded sources/v1 path that does not exist on the deploy host."""
    from pathlib import Path  # noqa: PLC0415

    from src.citations import DEFAULT_BASE, DEFAULT_RELEASE  # noqa: PLC0415
    from src.server.api_ingest import _META_DB_PATH  # noqa: PLC0415

    expected = Path(DEFAULT_BASE) / DEFAULT_RELEASE / "paper_meta.sqlite"
    assert _META_DB_PATH == expected
    assert "sources/v1" not in str(_META_DB_PATH)


# ---------------------------------------------------------------------------
# INGEST-04: crashed event payload
# ---------------------------------------------------------------------------


def test_crashed_event_payload(fake_queue: FakeDiscoverQueue) -> None:
    """INGEST-04/D-08: crashed state → SSE emits event='crashed' with note + corpus_ids."""
    from src.server import api_ingest as _ai  # noqa: PLC0415

    run_id = "crash001"
    fake_queue.script[run_id] = {
        "state": "crashed",
        "run_id": run_id,
        "result": {
            "status": "crashed",
            "queries": ["2305.12345"],
            "per_query": [],
            "note": "process exited without writing result.json",
        },
        "log_tail": [],
    }

    payload = _ai._build_terminal_event_payload(fake_queue.script[run_id], corpus_ids=[1])  # type: ignore[attr-defined]
    assert payload["note"] == "process exited without writing result.json"
    assert payload["corpus_ids"] == [1]


# ---------------------------------------------------------------------------
# INGEST-04: cancelled event payload
# ---------------------------------------------------------------------------


def test_cancelled_event_payload(fake_queue: FakeDiscoverQueue) -> None:
    """INGEST-04: cancelled state → event payload carries corpus_ids."""
    from src.server import api_ingest as _ai  # noqa: PLC0415

    run_id = "cancel001"
    fake_queue.script[run_id] = {
        "state": "cancelled",
        "run_id": run_id,
        "result": {
            "status": "cancelled",
            "queries": ["2305.12345"],
            "per_query": [],
        },
        "log_tail": [],
    }

    payload = _ai._build_terminal_event_payload(fake_queue.script[run_id], corpus_ids=[5, 6])  # type: ignore[attr-defined]
    assert payload["corpus_ids"] == [5, 6]


# ---------------------------------------------------------------------------
# INGEST-05: done event payload
# ---------------------------------------------------------------------------


def test_done_event_payload(fake_queue: FakeDiscoverQueue) -> None:
    """INGEST-05: done state → event payload carries corpus_ids + n_imported."""
    from src.server import api_ingest as _ai  # noqa: PLC0415

    run_id = "done001"
    fake_queue.script[run_id] = {
        "state": "done",
        "run_id": run_id,
        "result": {
            "status": "ok",
            "queries": ["2305.12345"],
            "per_query": [],
            "n_imported": 3,
        },
        "log_tail": [],
    }

    payload = _ai._build_done_event_payload(fake_queue.script[run_id], corpus_ids=[10, 20])  # type: ignore[attr-defined]
    assert payload["corpus_ids"] == [10, 20]
    assert payload["n_imported"] == 3


# ---------------------------------------------------------------------------
# Security info-disclosure: running frame exposes stage, not raw log_tail
# ---------------------------------------------------------------------------


def test_no_sensitive_log_leak(fake_queue: FakeDiscoverQueue) -> None:
    """Security: running SSE frame data has 'stage' field but does NOT dump full log_tail list."""
    from src.server import api_ingest as _ai  # noqa: PLC0415

    run_id = "run001"
    secret_tail = [
        "  [progress] trading:1/3 pdf:1/3 (parse_pending=2)",
        "internal secret line",
    ]
    fake_queue.script[run_id] = {
        "state": "running",
        "run_id": run_id,
        "pid": 12345,
        "elapsed_s": 42.1,
        "log_tail": secret_tail,
    }

    payload = _ai._build_running_event_payload(fake_queue.script[run_id], corpus_ids=[7])  # type: ignore[attr-defined]
    assert "stage" in payload, "running payload must include 'stage' field"
    assert "log_tail" not in payload, "running payload must NOT leak raw log_tail list"
    assert secret_tail not in payload.values(), (
        "raw log_tail list must not appear verbatim in payload"
    )
