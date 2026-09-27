"""Wave-0 RED tests for the D-26 on_stage callback contract.

Implementation lands in Plan 04 (additive `on_stage` + `cancel_event` kwargs to
run_query_pipeline).  Tests FAIL with TypeError until Plan 04 ships — that is
the desired RED state.

The genai.Client monkeypatch is copied verbatim from
tests/server/test_audit_smoke.py:115-142 (attributed inline).

Invariants enforced:
  - callback fires AFTER audit.write_stage returns (never before, never instead).
  - callback count == number of NN_*.json stage files on disk (CRIT-2 mitigation).
  - exceptions raised inside the callback do NOT crash the pipeline
    (caught + logged; partial trace beats no trace).
  - cancel_event.is_set() is checked BETWEEN stages and never mid-write
    (Pitfall 5 / D-22).
"""

from __future__ import annotations

import threading
import types
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest


def _install_genai_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verbatim copy of tests/server/test_audit_smoke.py:115-142.

    Replaces genai.Client so call_text runs without an API key.
    """
    fake_usage = types.SimpleNamespace(
        prompt_token_count=10,
        candidates_token_count=20,
        thoughts_token_count=0,
        total_token_count=30,
        cached_content_token_count=0,
    )
    fake_response = types.SimpleNamespace(text="stub", usage_metadata=fake_usage)
    fake_models = types.SimpleNamespace(
        generate_content=lambda **kwargs: fake_response,
    )
    fake_client = types.SimpleNamespace(models=fake_models)

    from google import genai as _genai_mod

    from src import config as _config

    monkeypatch.setattr(_genai_mod, "Client", lambda **kwargs: fake_client)
    monkeypatch.setattr(_config, "GEMINI_API_KEY", "fake-key-for-test")
    monkeypatch.setattr(_config, "validate_api_key", lambda: None)


def _stub_milvus_search(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the milvus search so the pipeline does not hit the real PC.

    ``milvus_search_only`` lives in ``src.query.retrieve`` (the test originally
    targeted a non-existent ``src.query.milvus_search`` module — fixed inline
    as a Rule-1 bug per Plan 01-04 execution). Returns an empty list so
    ``_retrieve`` short-circuits at ``batched_rerank_and_lookup([])``: still
    writes the milvus + rerank audit stages, no real Milvus call.

    Also stubs ``validate_api_key`` and the embed helper inside
    ``src.query.retrieve`` because they are imported by-name and so are not
    affected by patching ``src.config.validate_api_key`` alone.
    """
    try:
        import src.query.retrieve as _retr
    except ImportError:
        return

    def _fake_milvus(*_args: object, **_kwargs: object) -> list[dict[str, Any]]:
        return []

    def _fake_embed(_query: str) -> list[float]:
        return [0.0] * 4

    def _noop_validate() -> None:
        return None

    monkeypatch.setattr(_retr, "milvus_search_only", _fake_milvus)
    monkeypatch.setattr(_retr, "validate_api_key", _noop_validate)
    monkeypatch.setattr(_retr, "_embed_query", _fake_embed)


def _point_query_log_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Redirect query log writes into tmp_path so tests are sandboxed."""
    queries_root = tmp_path / "queries"
    monkeypatch.setenv("QUERY_LOG_ROOT", str(queries_root))
    try:
        import src.query.query_logger as _ql

        if hasattr(_ql, "QUERY_LOG_ROOT"):
            monkeypatch.setattr(_ql, "QUERY_LOG_ROOT", queries_root)
    except ImportError:
        pass
    return queries_root


def test_on_stage_invoked_once_per_write_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D-26 / CRIT-2: callback count equals number of audit.write_stage calls.

    Counted on disk via [0-9][0-9]_*.json files in the per-query directory.
    """
    _install_genai_stub(monkeypatch)
    _stub_milvus_search(monkeypatch)
    queries_root = _point_query_log_root(monkeypatch, tmp_path)

    from src.query.query_pipeline import run_query_pipeline

    mock = Mock()
    run_query_pipeline(
        query="test",
        collection="trading",
        notebook="trading",
        on_stage=mock,  # type: ignore[call-arg]
    )

    stage_files = list(queries_root.rglob("[0-9][0-9]_*.json"))
    n_stage_files = len(stage_files)
    assert n_stage_files > 0, "expected at least one NN_*.json stage file written; got 0"
    assert mock.call_count == n_stage_files, (
        f"D-26 invariant violated: callback fired {mock.call_count} times "
        f"but {n_stage_files} stage files written"
    )


def test_on_stage_exception_does_not_crash_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """01-RESEARCH.md §Pattern 2: callback exceptions are caught and logged."""
    _install_genai_stub(monkeypatch)
    _stub_milvus_search(monkeypatch)
    queries_root = _point_query_log_root(monkeypatch, tmp_path)

    from src.query.query_pipeline import run_query_pipeline

    boom = Mock(side_effect=RuntimeError("boom"))
    # Must NOT raise.
    result = run_query_pipeline(
        query="test",
        collection="trading",
        notebook="trading",
        on_stage=boom,  # type: ignore[call-arg]
    )
    assert result is not None  # pipeline returned (the response may be None)
    # All stage files still written, even though callback threw.
    stage_files = list(queries_root.rglob("[0-9][0-9]_*.json"))
    assert len(stage_files) > 0


def test_cancel_event_raises_between_stages_not_mid_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pitfall 5 / D-22: cancel_event.set() between stages raises CancelledError.

    Crucially: cancellation happens AFTER write_stage returns, never mid-write —
    so at least the first stage's *.json file exists on disk when cancellation hits.
    """
    import asyncio

    _install_genai_stub(monkeypatch)
    _stub_milvus_search(monkeypatch)
    queries_root = _point_query_log_root(monkeypatch, tmp_path)

    from src.query.query_pipeline import run_query_pipeline

    cancel_evt = threading.Event()

    def _on_stage(_name: str, _latency_ms: int) -> None:
        # Fire the cancellation flag after the first stage write returns.
        cancel_evt.set()

    with pytest.raises((asyncio.CancelledError, RuntimeError, Exception)) as exc_info:
        run_query_pipeline(
            query="test",
            collection="trading",
            notebook="trading",
            on_stage=_on_stage,  # type: ignore[call-arg]
            cancel_event=cancel_evt,  # type: ignore[call-arg]
        )

    # Verify the cancellation is "the right kind" — not a TypeError about kwargs.
    assert "Cancel" in type(exc_info.value).__name__ or isinstance(
        exc_info.value, asyncio.CancelledError
    ), f"unexpected exception type: {type(exc_info.value).__name__}"

    # At least one stage file MUST have been completed before cancellation hit
    # (cancel-between-stages invariant).
    stage_files = list(queries_root.rglob("[0-9][0-9]_*.json"))
    assert len(stage_files) >= 1, (
        "cancellation happened BEFORE first audit.write_stage finished — "
        "violates Pitfall 5 invariant"
    )
