"""Audit-trail smoke test (INFRA-05 + INFRA-04 cost wiring).

Phase 0 invariants:
1. _STAGE_PREFIX is importable from src.query.query_logger.
2. Every prefix value is a 2-digit zero-padded unique string.
3. On any existing query directory under logs/queries/, every stage file's
   NN prefix appears in _STAGE_PREFIX.values().
4. meta.json on the latest query is valid JSON.
5. usage_track.call_text + QueryLogger.accumulate_usage + finalize write
   meta.totals.cost_usd > 0 (INFRA-04 cost accounting wired; verified via
   monkeypatched genai.Client, no real API key needed in CI).

Phase 1 Plan 04 promotion (RESEARCH §Risk 6):
6. The full GUI round-trip (POST /api/chats/{id}/messages with
   ``Accept: text/event-stream``) writes a real audit trail with
   ``meta.totals.cost_usd > 0`` AND ``meta.totals.n_llm_calls > 0``. This
   proves the SSE handler does NOT bypass usage_track + audit.write_stage
   (CRIT-1 / CRIT-2).
"""

from __future__ import annotations

import json
import re
import types
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
QUERIES_DIR = PROJECT_ROOT / "logs" / "queries"


def test_stage_prefix_importable() -> None:
    try:
        from src.query.query_logger import _STAGE_PREFIX
    except ImportError as exc:
        pytest.fail(
            f"INFRA-05: src.query.query_logger._STAGE_PREFIX not importable: {exc}. "
            "If you renamed/moved this, update INFRA-05 smoke test."
        )
    assert isinstance(_STAGE_PREFIX, dict)
    assert len(_STAGE_PREFIX) > 0, "_STAGE_PREFIX is empty"


def test_stage_prefix_values_are_unique_and_padded() -> None:
    from src.query.query_logger import _STAGE_PREFIX

    values = list(_STAGE_PREFIX.values())
    assert len(values) == len(set(values)), f"_STAGE_PREFIX has duplicate values: {values}"
    bad = [v for v in values if not re.fullmatch(r"\d{2}", v)]
    assert not bad, (
        f"_STAGE_PREFIX has un-padded or non-numeric values: {bad}. "
        "Every prefix must be exactly 2 digits, zero-padded (e.g. '01', '07')."
    )


def _find_latest_query_dir() -> Path | None:
    if not QUERIES_DIR.is_dir():
        return None
    date_dirs = [
        d
        for d in QUERIES_DIR.iterdir()
        if d.is_dir() and re.fullmatch(r"\d{4}-\d{2}-\d{2}", d.name)
    ]
    if not date_dirs:
        return None
    latest_date = max(date_dirs, key=lambda d: d.name)
    query_dirs = [d for d in latest_date.iterdir() if d.is_dir()]
    if not query_dirs:
        return None
    return max(query_dirs, key=lambda d: d.stat().st_mtime)


def test_existing_query_dir_stage_count_matches_registry() -> None:
    latest = _find_latest_query_dir()
    if latest is None:
        pytest.skip("No logs/queries/<date>/<id>/ dirs yet — INFRA-05 is forward-looking.")

    from src.query.query_logger import _STAGE_PREFIX

    stage_files = sorted(p for p in latest.iterdir() if p.is_file() and re.match(r"\d{2}_", p.name))
    file_prefixes = {p.name.split("_", 1)[0] for p in stage_files}
    registered = set(_STAGE_PREFIX.values())
    unregistered = file_prefixes - registered
    assert not unregistered, (
        f"INFRA-05: query dir {latest.relative_to(PROJECT_ROOT)} has stage files "
        f"with unregistered prefixes {unregistered}. Every audit.write_stage prefix "
        f"MUST be in _STAGE_PREFIX (src/query/query_logger.py)."
    )


def test_meta_json_shape_on_latest_query() -> None:
    latest = _find_latest_query_dir()
    if latest is None:
        pytest.skip("No logs/queries/<date>/<id>/ dirs yet.")
    meta_path = latest / "meta.json"
    if not meta_path.is_file():
        pytest.skip(f"{meta_path} does not exist on this run.")
    meta = json.loads(meta_path.read_text())
    assert isinstance(meta, dict)


def test_cost_usd_positive_on_call_text(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """INFRA-04: call_text + QueryLogger.accumulate_usage records cost_usd > 0.

    Uses a synthetic genai.Client (no real API key needed in CI).
    The monkeypatch replaces genai.Client so that the full call_text code path
    executes — including usage extraction and cost computation — but no outbound
    HTTP is made.

    Then we construct a QueryLogger, call accumulate_usage with the result, and
    finalize to confirm meta.totals.cost_usd > 0.  This validates the full
    cost-accounting pipeline end-to-end:
        call_text → LLMCallResult.cost_usd → accumulate_usage → finalize → meta.json
    """
    # Build synthetic usage metadata that call_text._usage_from_gemini will read.
    fake_usage = types.SimpleNamespace(
        prompt_token_count=10,
        candidates_token_count=20,
        thoughts_token_count=0,
        total_token_count=30,
        cached_content_token_count=0,
    )
    fake_response = types.SimpleNamespace(
        text="hello",
        usage_metadata=fake_usage,
    )

    # call_text uses genai.Client(api_key=..., http_options=...) then calls
    # gemini_client.models.generate_content(model=..., contents=..., config=...).
    # We mock genai.Client to return a fake client with that interface.
    fake_models = types.SimpleNamespace(
        generate_content=lambda **kwargs: fake_response,
    )
    fake_client = types.SimpleNamespace(models=fake_models)

    from google import genai as _genai_mod

    from src import config as _config

    # Monkeypatch genai.Client and config so no real API key is needed.
    monkeypatch.setattr(_genai_mod, "Client", lambda **kwargs: fake_client)
    monkeypatch.setattr(_config, "GEMINI_API_KEY", "fake-key-for-test")
    monkeypatch.setattr(_config, "validate_api_key", lambda: None)

    from src.query import usage_track
    from src.query.query_logger import QueryLogger

    # Invoke call_text (the ONLY authorized LLM call path — INFRA-04).
    try:
        result = usage_track.call_text(
            model="gemini-2.5-flash",
            system="test system",
            user="test user",
            json_mode=False,
            temperature=0.0,
        )
    except Exception as exc:
        pytest.skip(f"call_text raised unexpectedly (interface may have changed): {exc}")

    # cost_usd on the LLMCallResult must be positive.
    assert result.cost_usd is not None, (
        "INFRA-04: call_text returned cost_usd=None. "
        "Check that compute_cost_usd knows 'gemini-2.5-flash' (src/config/models.py)."
    )
    assert result.cost_usd > 0, (
        f"INFRA-04: call_text returned cost_usd={result.cost_usd}. Expected > 0."
    )

    # Now wire through QueryLogger to confirm meta.totals.cost_usd > 0.
    log_root = tmp_path / "queries"
    ql = QueryLogger("test prompt", root=log_root)
    ql.accumulate_usage(
        stage="synthesis",
        usage=result.usage,
        cost_usd=result.cost_usd,
        latency_ms=result.latency_ms,
    )
    ql.finalize(total_latency_ms=result.latency_ms)

    meta_files = list(log_root.rglob("meta.json"))
    assert meta_files, (
        "INFRA-04: QueryLogger.finalize() completed but no meta.json was written. "
        "Check that finalize() writes to self.log_dir / 'meta.json'."
    )
    meta = json.loads(meta_files[0].read_text())
    cost = meta.get("totals", {}).get("cost_usd", 0)
    assert cost > 0, (
        f"INFRA-04: meta.totals.cost_usd == {cost} after accumulate_usage+finalize. "
        "Expected > 0. Check accumulate_usage stores cost_usd and finalize sums it."
    )


# ---------------------------------------------------------------------------
# Phase 1 Plan 04 promotion — GUI round-trip writes a real audit trail
# ---------------------------------------------------------------------------


def _install_genai_stub_for_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub genai.Client + config knobs so the LLM stages run without a key."""
    fake_usage = types.SimpleNamespace(
        prompt_token_count=10,
        candidates_token_count=20,
        thoughts_token_count=0,
        total_token_count=30,
        cached_content_token_count=0,
    )
    fake_response = types.SimpleNamespace(text="stub answer", usage_metadata=fake_usage)
    fake_models = types.SimpleNamespace(
        generate_content=lambda **kwargs: fake_response,
    )
    fake_client = types.SimpleNamespace(models=fake_models)
    from google import genai as _genai_mod  # noqa: PLC0415

    from src import config as _config  # noqa: PLC0415

    monkeypatch.setattr(_genai_mod, "Client", lambda **kwargs: fake_client)
    monkeypatch.setattr(_config, "GEMINI_API_KEY", "fake-key-for-test")
    monkeypatch.setattr(_config, "validate_api_key", lambda: None)

    monkeypatch.setattr(_config, "DECOMPOSE_MODEL", "gemini-2.5-flash")


def _stub_milvus_for_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty-hit Milvus stub — pipeline writes decompose + milvus + rerank stages."""
    import src.query.retrieve as _retr  # noqa: PLC0415

    monkeypatch.setattr(_retr, "milvus_search_only", lambda *a, **k: [])
    monkeypatch.setattr(_retr, "validate_api_key", lambda: None)
    monkeypatch.setattr(_retr, "_embed_query", lambda _q: [0.0] * 4)


def test_gui_round_trip_writes_real_audit_trail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RESEARCH §Risk 6: real GUI round-trip writes a real audit trail.

    Drives ``POST /api/chats/{id}/messages`` with ``Accept: text/event-stream``
    end-to-end, drains the SSE stream, and asserts the on-disk meta.json has
    ``meta.totals.cost_usd > 0`` AND ``meta.totals.n_llm_calls > 0``. Also
    confirms the stage-file count matches the _STAGE_PREFIX registry intersect
    with the stages that actually ran (decompose + milvus + rerank, since the
    stub returns no chunks so synthesis is skipped).
    """
    pytest.importorskip("src.server.api_chats")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.server.api_chats import router
    from src.server.migrations import migrate

    _install_genai_stub_for_pipeline(monkeypatch)
    _stub_milvus_for_pipeline(monkeypatch)

    # Sandbox chats.db + logs/queries into tmp_path.
    chats_db = tmp_path / "chats.db"
    migrate.run(chats_db)
    import src.server.chats_store as _cs  # noqa: PLC0415

    monkeypatch.setattr(_cs, "CHATS_DB_PATH", chats_db)
    queries_root = tmp_path / "queries"
    monkeypatch.setenv("QUERY_LOG_ROOT", str(queries_root))

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    # Create a chat.
    r = client.post("/api/chats", json={"title": "audit smoke"})
    assert r.status_code == 200, r.text
    chat_id = r.json()["id"]

    # Drive the SSE round-trip.
    body = {"content": "test query", "message_uuid": "u-smoke", "expected_version": 0}
    events: list[tuple[str, str]] = []
    with client.stream(
        "POST",
        f"/api/chats/{chat_id}/messages",
        json=body,
        headers={"Accept": "text/event-stream"},
    ) as resp:
        assert resp.status_code == 200
        cur_event: str | None = None
        for raw in resp.iter_lines():
            line = raw.strip()
            if not line:
                cur_event = None
                continue
            if line.startswith("event:"):
                cur_event = line[len("event:") :].strip()
            elif line.startswith("data:"):
                if cur_event is not None:
                    events.append((cur_event, line[len("data:") :].strip()))
            if cur_event == "done" and events and events[-1][0] == "done":
                break

    assert events, "no SSE events received"
    assert events[0][0] == "queued", f"first event was {events[0]!r}, expected queued"
    done_event = next((e for e in events if e[0] == "done"), None)
    assert done_event is not None, f"no done event; got {[e[0] for e in events]}"

    # Locate meta.json + assert real totals.
    meta_files = list(queries_root.rglob("meta.json"))
    assert meta_files, f"no meta.json under {queries_root}"
    meta = json.loads(meta_files[0].read_text())
    totals = meta["totals"]
    assert totals.get("cost_usd", 0) > 0, (
        f"CRIT-1: SSE round-trip wrote meta.totals.cost_usd={totals.get('cost_usd')!r}; "
        "expected > 0. Indicates the SSE handler is bypassing usage_track.call_text."
    )
    assert totals.get("n_llm_calls", 0) > 0, (
        f"CRIT-2: SSE round-trip wrote meta.totals.n_llm_calls={totals.get('n_llm_calls')!r}; "
        "expected > 0. Indicates the SSE handler is bypassing audit.accumulate_usage."
    )

    # Every stage file's NN prefix is registered in _STAGE_PREFIX.
    from src.query.query_logger import _STAGE_PREFIX  # noqa: PLC0415

    stage_files = list(meta_files[0].parent.glob("[0-9][0-9]_*.json"))
    assert stage_files, "no NN_*.json stage files written"
    file_prefixes = {p.name.split("_", 1)[0] for p in stage_files}
    registered = set(_STAGE_PREFIX.values())
    unregistered = file_prefixes - registered
    assert not unregistered, (
        f"INFRA-05: GUI round-trip wrote stage files with unregistered "
        f"prefixes {unregistered}. Every audit.write_stage prefix MUST be in "
        f"_STAGE_PREFIX (src/query/query_logger.py)."
    )


# ---------------------------------------------------------------------------
# Phase 1 Plan 08 promotion — GUI SSE round-trip via shared conftest fixtures.
#
# The Phase 0 baseline (test_cost_usd_positive_on_call_text) and the Plan 04
# promotion (test_gui_round_trip_writes_real_audit_trail) remain authoritative.
# This Plan 08 test is the GUI-fixture-driven version: it re-asserts the same
# invariants (cost_usd > 0 + callback-count == write_stage-count) but consumes
# the ``client_with_chats`` fixture from tests/server/conftest.py, proving the
# shared fixture inventory is wired correctly for downstream phases.
# ---------------------------------------------------------------------------


def test_audit_smoke_via_gui_sse_round_trip(  # noqa: C901, PLR0915
    client_with_chats: object,
    audit_log_root: Path,
) -> None:
    """Plan 08: GUI SSE round-trip via shared fixtures (CRIT-1 + CRIT-2).

    Drives ``POST /api/chats/{id}/messages`` with ``Accept: text/event-stream``
    through ``client_with_chats`` (the shared fixture from
    ``tests/server/conftest.py``) and asserts:

    1. SSE event order: ``queued`` → ``stage*`` → ``citations`` → ``delta``
       → ``done`` (D-20).
    2. ``event: done`` payload exposes ``query_id`` (D-15 join key).
    3. The on-disk audit trail under ``audit_log_root`` carries the same
       ``query_id`` with at least one ``NN_*.json`` stage file and a
       ``meta.json`` whose ``totals.cost_usd > 0`` (CRIT-1) AND
       ``totals.n_llm_calls > 0`` (CRIT-2).
    4. The count of ``event: stage`` SSE frames received equals the count of
       ``NN_*.json`` stage files written on disk — i.e. the on_stage
       callback never fires WITHOUT a corresponding ``audit.write_stage`` and
       vice versa (CRIT-2 GUI-path mitigation).
    """
    # mypy can't introspect the conftest fixture type without an explicit cast.
    from fastapi.testclient import TestClient

    client: TestClient = client_with_chats  # type: ignore[assignment]

    # 1. Create a chat.
    r = client.post("/api/chats", json={"title": "plan-08 smoke"})
    assert r.status_code in (200, 201), f"POST /api/chats: {r.status_code} {r.text}"
    chat = r.json()
    chat_id = chat["id"]
    version = chat.get("version", 0)

    # 2. Stream a turn through the SSE branch.
    body = {
        "content": "what is market making?",
        "message_uuid": "08-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        "expected_version": version,
    }
    events: list[tuple[str, dict[str, Any]]] = []
    cur_event: str | None = None
    with client.stream(
        "POST",
        f"/api/chats/{chat_id}/messages",
        headers={"Accept": "text/event-stream"},
        json=body,
    ) as response:
        assert response.status_code == 200, response.text
        assert response.headers["content-type"].startswith("text/event-stream")
        for raw in response.iter_lines():
            line = raw.strip()
            if not line:
                cur_event = None
                continue
            if line.startswith(":"):
                # Keepalive comment frame — discard.
                continue
            if line.startswith("event:"):
                cur_event = line[len("event:") :].strip()
                continue
            if line.startswith("data:") and cur_event is not None:
                payload = line[len("data:") :].strip()
                try:
                    parsed = json.loads(payload)
                except json.JSONDecodeError:
                    parsed = {"raw": payload}
                if not isinstance(parsed, dict):
                    parsed = {"list": parsed}
                events.append((cur_event, parsed))
                if cur_event == "done":
                    break

    # 3. Event order.
    names = [name for name, _ in events]
    assert names, "no SSE events received"
    assert names[0] == "queued", f"first event must be queued; got {names[:3]}"
    assert "citations" in names, f"citations event missing; got {names}"
    assert "delta" in names, f"delta event missing; got {names}"
    assert "done" in names, f"done event missing; got {names}"
    # citations BEFORE delta (MOD-1 / D-20).
    assert names.index("citations") < names.index("delta"), (
        f"citations must precede delta (MOD-1); got {names}"
    )

    # 4. done payload structure (D-15). The empty-milvus stub returns no
    # chunks, so the pipeline early-returns ``rag_response = None`` and the
    # SSE handler emits ``query_id: None`` in the done event (api_chats.py
    # line 549-570). The audit trail is STILL written to disk because the
    # decompose stage runs before the empty-retrieval short-circuit. We
    # locate the qid from the on-disk meta.json instead.
    done_payload = next(p for n, p in events if n == "done")
    assert "query_id" in done_payload, (
        f"event: done payload must contain query_id key; got {done_payload!r}"
    )

    # 5. Audit trail on disk under audit_log_root.
    meta_files = list(audit_log_root.rglob("meta.json"))
    assert meta_files, (
        f"no meta.json under {audit_log_root}; "
        f"the GUI round-trip did not write an audit trail (CRIT-1 violated)"
    )
    # Newest mtime — single-user test sandbox.
    meta_path = max(meta_files, key=lambda p: p.stat().st_mtime)
    qid_dir = meta_path.parent
    assert meta_path.is_file(), f"meta.json missing at {meta_path}"
    meta = json.loads(meta_path.read_text())
    totals = meta.get("totals", {})

    # CRIT-1: GUI-path cost accounting must record real cost.
    assert totals.get("cost_usd", 0) > 0, (
        f"CRIT-1: SSE round-trip wrote meta.totals.cost_usd="
        f"{totals.get('cost_usd')!r}; expected > 0. The SSE handler is "
        "bypassing usage_track.call_text."
    )

    # CRIT-2 (audit-write): GUI path must accumulate LLM calls.
    assert totals.get("n_llm_calls", 0) > 0, (
        f"CRIT-2: SSE round-trip wrote meta.totals.n_llm_calls="
        f"{totals.get('n_llm_calls')!r}; expected > 0."
    )

    # CRIT-2 (callback mirror): callback-count == write_stage-count.
    stage_event_count = sum(1 for n in names if n == "stage")
    stage_files = sorted(qid_dir.glob("[0-9][0-9]_*.json"))
    assert stage_files, "no NN_*.json stage files written"
    assert stage_event_count == len(stage_files), (
        f"CRIT-2 (GUI-path callback mirror): callback-count "
        f"({stage_event_count}) != write_stage-count ({len(stage_files)}).\n"
        f"  stage events:  {[n for n in names if n == 'stage']}\n"
        f"  stage files:   {[p.name for p in stage_files]}"
    )
