"""RED tests for src/query/retrieve_paperqa.py (Plan 02-06).

Pins the D-06 sub-stage emission contract for the PaperQA2 retriever:
``paperqa_rerank``, ``paperqa_read``, ``paperqa_answer`` — emitted in
that order, each via ``audit.write_stage`` followed by
``accumulate_usage`` followed by the ``on_stage`` callback (D-26).

Every LLM call inside the retriever MUST go through
``src.query.usage_track.call_text`` (CLAUDE.md rule 1). Tests assert
exactly this by patching ``call_text`` and asserting it was called.

Tests use deferred imports so collection stays green before Plan 02-06
lands ``src.query.retrieve_paperqa``.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

PAPERQA_SUBSTAGES: tuple[str, str, str] = (
    "paperqa_rerank",
    "paperqa_read",
    "paperqa_answer",
)


@pytest.fixture
def audit_log_root_local(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Sandbox QUERY_LOG_ROOT into tmp_path."""
    root = tmp_path / "queries"
    monkeypatch.setenv("QUERY_LOG_ROOT", str(root))
    yield root


def test_emits_three_substages_in_order(
    audit_log_root_local: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RED — paperqa retriever emits paperqa_rerank, paperqa_read, paperqa_answer
    in that exact order via the on_stage callback (D-26 / D-06).
    """
    pytest.importorskip(
        "src.query.retrieve_paperqa",
        reason="Plan 02-06 has not landed yet",
    )
    from src.query.retrieve_paperqa import retrieve_paperqa  # noqa: PLC0415

    observed: list[str] = []

    def _on_stage(stage_name: str, _latency_ms: int) -> None:
        observed.append(stage_name)

    retrieve_paperqa(
        query="test",
        collections=["trading"],
        on_stage=_on_stage,
    )

    paperqa_events = [s for s in observed if s.startswith("paperqa_")]
    assert paperqa_events == list(PAPERQA_SUBSTAGES), (
        f"D-06: expected sub-stages {PAPERQA_SUBSTAGES} in order; got {paperqa_events}"
    )


def test_each_substage_calls_audit_write_stage(
    audit_log_root_local: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RED — each sub-stage writes a NN_*.json file via audit.write_stage
    BEFORE on_stage fires (CRIT-2; D-26 contract: callback fires AFTER write).
    """
    pytest.importorskip(
        "src.query.retrieve_paperqa",
        reason="Plan 02-06 has not landed yet",
    )
    from src.query.retrieve_paperqa import retrieve_paperqa  # noqa: PLC0415

    write_calls: list[str] = []
    from src.query import audit  # noqa: PLC0415

    real_write = audit.write_stage

    def _spy_write(name: str, payload: dict[str, Any]) -> None:
        write_calls.append(name)
        real_write(name, payload)

    monkeypatch.setattr(audit, "write_stage", _spy_write)
    retrieve_paperqa(query="test", collections=["trading"])

    for sub in PAPERQA_SUBSTAGES:
        assert sub in write_calls, (
            f"CRIT-2: sub-stage {sub!r} did not call audit.write_stage; observed={write_calls}"
        )


def test_each_substage_calls_accumulate_usage(
    audit_log_root_local: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RED — every sub-stage calls QueryLogger.accumulate_usage so
    meta.totals.cost_usd / n_llm_calls reflect the retriever (CRIT-2 mirror).
    """
    pytest.importorskip(
        "src.query.retrieve_paperqa",
        reason="Plan 02-06 has not landed yet",
    )
    from src.query.query_logger import QueryLogger  # noqa: PLC0415
    from src.query.retrieve_paperqa import retrieve_paperqa  # noqa: PLC0415

    accum_calls: list[str] = []
    real_accum = QueryLogger.accumulate_usage

    def _spy_accum(self: QueryLogger, stage: str, **kw: Any) -> None:  # noqa: ANN401
        accum_calls.append(stage)
        real_accum(self, stage, **kw)

    monkeypatch.setattr(QueryLogger, "accumulate_usage", _spy_accum)
    retrieve_paperqa(query="test", collections=["trading"])

    for sub in PAPERQA_SUBSTAGES:
        assert sub in accum_calls, (
            f"CRIT-2 (audit-usage mirror): {sub!r} did not call accumulate_usage; "
            f"observed={accum_calls}"
        )


def test_each_llm_call_goes_through_usage_track(
    audit_log_root_local: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RED — CLAUDE.md rule 1: every LLM call MUST go through call_text.

    Patches call_text to a Mock; runs the retriever; asserts the mock was
    called. (No raw genai.Client / openai.Client calls allowed.)
    """
    pytest.importorskip(
        "src.query.retrieve_paperqa",
        reason="Plan 02-06 has not landed yet",
    )
    from src.query import usage_track  # noqa: PLC0415
    from src.query.retrieve_paperqa import retrieve_paperqa  # noqa: PLC0415

    spy = MagicMock(wraps=usage_track.call_text)
    monkeypatch.setattr(usage_track, "call_text", spy)
    retrieve_paperqa(query="test", collections=["trading"])

    assert spy.called, (
        "CLAUDE.md rule 1: at least one LLM call must go through "
        "usage_track.call_text; retriever appears to bypass it."
    )


def test_on_stage_fires_after_write_stage_not_before(
    audit_log_root_local: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RED — D-26: on_stage callback fires AFTER audit.write_stage returns.

    Records the interleaving of write_stage(name) and on_stage(name); for
    each sub-stage, the write must appear before the callback.
    """
    pytest.importorskip(
        "src.query.retrieve_paperqa",
        reason="Plan 02-06 has not landed yet",
    )
    from src.query import audit  # noqa: PLC0415
    from src.query.retrieve_paperqa import retrieve_paperqa  # noqa: PLC0415

    events: list[tuple[str, str]] = []

    real_write = audit.write_stage

    def _spy_write(name: str, payload: dict[str, Any]) -> None:
        events.append(("write", name))
        real_write(name, payload)

    monkeypatch.setattr(audit, "write_stage", _spy_write)

    def _on_stage(stage_name: str, _latency_ms: int) -> None:
        events.append(("callback", stage_name))

    retrieve_paperqa(query="test", collections=["trading"], on_stage=_on_stage)

    for sub in PAPERQA_SUBSTAGES:
        write_idx = next(
            (i for i, (k, n) in enumerate(events) if k == "write" and n == sub),
            None,
        )
        cb_idx = next(
            (i for i, (k, n) in enumerate(events) if k == "callback" and n == sub),
            None,
        )
        assert write_idx is not None, f"no write_stage for {sub!r}: {events}"
        assert cb_idx is not None, f"no callback for {sub!r}: {events}"
        assert write_idx < cb_idx, (
            f"D-26: callback for {sub!r} fired BEFORE audit.write_stage. order={events}"
        )


def test_collections_passed_as_partition_names(
    audit_log_root_local: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RED — D-03 + CLAUDE.md "filter before search": collections flow through
    to the retriever as partition_names (NOT post-filter).
    """
    pytest.importorskip(
        "src.query.retrieve_paperqa",
        reason="Plan 02-06 has not landed yet",
    )
    from src.query.retrieve_paperqa import retrieve_paperqa  # noqa: PLC0415

    captured: dict[str, Any] = {}

    def _capture(*_a: object, **kw: object) -> list[Any]:
        captured.update(kw)
        return []

    import src.query.retrieve as _retr  # noqa: PLC0415

    monkeypatch.setattr(_retr, "milvus_search_only", _capture)
    monkeypatch.setattr(_retr, "validate_api_key", lambda: None)
    monkeypatch.setattr(_retr, "_embed_query", lambda _q: [0.0] * 4)

    retrieve_paperqa(query="test", collections=["trading", "ecology"])

    parts = captured.get("partition_names") or captured.get("partitions") or []
    assert "trading" in parts and "ecology" in parts, (
        f"D-03: expected partition_names containing trading + ecology; got {parts!r}"
    )


def test_cancel_event_raises_between_stages(
    audit_log_root_local: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RED — D-22 / Pitfall 5: cancel_event.set() between sub-stages raises
    CancelledError; never mid-write.
    """
    import asyncio  # noqa: PLC0415

    pytest.importorskip(
        "src.query.retrieve_paperqa",
        reason="Plan 02-06 has not landed yet",
    )
    from src.query.retrieve_paperqa import retrieve_paperqa  # noqa: PLC0415

    cancel_evt = threading.Event()

    def _on_stage(stage_name: str, _latency_ms: int) -> None:
        if stage_name == "paperqa_rerank":
            cancel_evt.set()

    with pytest.raises((asyncio.CancelledError, Exception)) as exc_info:
        retrieve_paperqa(
            query="test",
            collections=["trading"],
            on_stage=_on_stage,
            cancel_event=cancel_evt,
        )

    assert "Cancel" in type(exc_info.value).__name__ or isinstance(
        exc_info.value, asyncio.CancelledError
    ), f"unexpected exception type: {type(exc_info.value).__name__}"


def test_stub_fallback_when_paper_qa_not_installed(
    audit_log_root_local: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RED — A1: when paper-qa is not installed, retriever returns
    chunks=[] + trace={"stage":"stub", ...} without raising. All three
    sub-stages still emit write_stage with payload {"stub": True}.
    """
    pytest.importorskip(
        "src.query.retrieve_paperqa",
        reason="Plan 02-06 has not landed yet",
    )
    import importlib.util  # noqa: PLC0415

    from src.query.retrieve_paperqa import retrieve_paperqa  # noqa: PLC0415

    real_find_spec = importlib.util.find_spec

    def _fake_find_spec(name: str, *a: object, **kw: object) -> Any:  # noqa: ANN401
        if name == "paperqa":
            return None
        return real_find_spec(name, *a, **kw)  # type: ignore[arg-type]

    monkeypatch.setattr(importlib.util, "find_spec", _fake_find_spec)

    result = retrieve_paperqa(query="test", collections=["trading"])

    chunks, trace = result if isinstance(result, tuple) else (result, {})
    assert chunks == [], f"stub fallback: expected chunks=[]; got {chunks!r}"
    assert isinstance(trace, dict) and trace.get("stage") == "stub", (
        f"stub fallback: expected trace.stage=='stub'; got {trace!r}"
    )
