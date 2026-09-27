"""RED tests for Phase 2 extensions to src/server/api_chats.py.

Covers CHAT-01 (multi-collection + retriever dispatch),
CHAT-07 (edit-last new query_id + 409),
CHAT-08 (regenerate-last preserves audit dir),
CHAT-10 (auto-name PATCH + fallback via call_text),
HIST-02 (search endpoint), HIST-03 (rename / archive / delete),
plus SSE substage emission for the new retrievers.

Tests use deferred imports; collection stays green before Plans
02-04 / 02-09 / 02-10 / 02-11 land the api_chats extensions.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from fastapi.testclient import TestClient

# --------------------------------------------------------------------------- #
# Shared fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def chats_db_path(tmp_path: Path) -> Path:
    """tmp_path-scoped chats.db with all migrations applied.

    Once Plan 02-03 ships 0003_chats_fts.sql, this returns a db with
    user_version >= 3.
    """
    from src.server.migrations import migrate

    db = tmp_path / "chats.db"
    migrate.run(db)
    return db


def _fake_genai_monkeypatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mirror tests/server/test_audit_smoke.py:115-149 genai stub."""
    import types

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

    from google import genai as _genai_mod

    from src import config as _config

    monkeypatch.setattr(_genai_mod, "Client", lambda **kwargs: fake_client)
    monkeypatch.setattr(_config, "GEMINI_API_KEY", "fake-key-for-test")
    monkeypatch.setattr(_config, "validate_api_key", lambda: None)

    from src.models import Usage
    from src.query import usage_track

    fake_result = usage_track.LLMCallResult(
        text="GLFT optimal spread",
        latency_ms=1,
        usage=Usage(input=10, output=3, total=13),
        raw_response=None,
        cost_usd=0.0001,
        model="test-stub",
    )
    monkeypatch.setattr(usage_track, "call_text", lambda *_a, **_kw: fake_result)


def _stub_milvus(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty-hit Milvus stub."""
    import src.query.retrieve as _retr

    monkeypatch.setattr(_retr, "milvus_search_only", lambda *_a, **_kw: [])
    monkeypatch.setattr(_retr, "validate_api_key", lambda: None)
    monkeypatch.setattr(_retr, "_embed_query", lambda _q: [0.0] * 4)


def _redirect_chats_db(monkeypatch: pytest.MonkeyPatch, chats_db_path: Path) -> None:
    import src.server.chats_store as _cs

    monkeypatch.setattr(_cs, "CHATS_DB_PATH", chats_db_path)


def _point_query_log_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    queries_root = tmp_path / "queries"
    monkeypatch.setenv("QUERY_LOG_ROOT", str(queries_root))
    return queries_root


def _drain_sse_done(
    client: TestClient, chat_id: str, payload: dict[str, Any]
) -> tuple[list[tuple[str, dict[str, Any]]], dict[str, Any] | None]:
    """Drain an SSE stream to completion; return (events, done_payload)."""
    events: list[tuple[str, dict[str, Any]]] = []
    done: dict[str, Any] | None = None
    cur_event: str | None = None
    with client.stream(
        "POST",
        f"/api/chats/{chat_id}/messages",
        json=payload,
        headers={"Accept": "text/event-stream"},
    ) as response:
        assert response.status_code == 200, response.text
        for raw in response.iter_lines():
            line = raw.strip()
            if not line:
                cur_event = None
                continue
            if line.startswith(":"):
                continue
            if line.startswith("event:"):
                cur_event = line[len("event:") :].strip()
            elif line.startswith("data:") and cur_event is not None:
                try:
                    parsed = json.loads(line[len("data:") :].strip())
                except json.JSONDecodeError:
                    parsed = {"raw": line[len("data:") :].strip()}
                if not isinstance(parsed, dict):
                    parsed = {"list": parsed}
                events.append((cur_event, parsed))
                if cur_event == "done":
                    done = parsed
                    break
    return events, done


@pytest.fixture
def sandboxed_client(
    chats_db_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[TestClient]:
    """TestClient wired to the api_chats router with all externals stubbed."""
    pytest.importorskip("src.server.api_chats")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.server.api_chats import router

    _fake_genai_monkeypatch(monkeypatch)
    _stub_milvus(monkeypatch)
    _redirect_chats_db(monkeypatch, chats_db_path)
    _point_query_log_root(monkeypatch, tmp_path)

    app = FastAPI()
    app.include_router(router)
    yield TestClient(app)


# --------------------------------------------------------------------------- #
# CHAT-01: create chat with retriever + collections
# --------------------------------------------------------------------------- #


def test_create_chat_with_retriever_and_collections(
    sandboxed_client: TestClient,
) -> None:
    """RED — POST /api/chats accepts retriever + collections fields (CHAT-01)."""
    r = sandboxed_client.post(
        "/api/chats",
        json={"title": "t", "retriever": "paperqa", "collections": ["trading", "ecology"]},
    )
    assert r.status_code in (200, 201), r.text
    body = r.json()
    assert body.get("retriever") == "paperqa", (
        f"CHAT-01: expected retriever='paperqa'; got {body!r}"
    )
    cols = body.get("collections")
    if isinstance(cols, str):
        cols = json.loads(cols)
    assert "trading" in cols and "ecology" in cols, (
        f"CHAT-01: expected collections=['trading','ecology']; got {cols!r}"
    )


def test_multi_collection_passes_partitions(
    sandboxed_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RED — D-03: chat's collections list flows to Milvus as partition_names."""
    captured: dict[str, Any] = {}

    def _capture(*_a: object, **kw: object) -> list[Any]:
        captured.update(kw)
        return []

    import src.query.retrieve as _retr

    monkeypatch.setattr(_retr, "milvus_search_only", _capture)

    r = sandboxed_client.post(
        "/api/chats",
        json={"retriever": "milvus", "collections": ["trading", "system"]},
    )
    chat_id = r.json()["id"]
    _drain_sse_done(
        sandboxed_client,
        chat_id,
        {"content": "hi", "message_uuid": "u1", "expected_version": 0},
    )

    parts = captured.get("partition_names") or captured.get("partitions") or []
    assert "trading" in parts and "system" in parts, (
        f"D-03: partition_names should carry chat.collections; got {parts!r}"
    )


# --------------------------------------------------------------------------- #
# CHAT-07: edit-last user turn
# --------------------------------------------------------------------------- #


def test_edit_last_produces_new_query_id(sandboxed_client: TestClient) -> None:
    """RED — POST /api/chats/{id}/edit-last starts a new SSE stream with a
    fresh query_id (CHAT-07).
    """
    chat_id = sandboxed_client.post("/api/chats", json={"title": "t"}).json()["id"]

    _, done1 = _drain_sse_done(
        sandboxed_client,
        chat_id,
        {"content": "first", "message_uuid": "u1", "expected_version": 0},
    )

    r = sandboxed_client.post(
        f"/api/chats/{chat_id}/edit-last",
        json={"content": "first-edited", "expected_version": 2},
        headers={"Accept": "text/event-stream"},
    )
    assert r.status_code == 200, r.text

    body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
    new_qid = body.get("query_id")
    if not new_qid and done1:
        new_qid = body.get("query_id")
    assert new_qid is not None, "CHAT-07: edit-last response missing query_id"
    assert done1 is not None and new_qid != done1.get("query_id"), (
        f"CHAT-07: edit-last must produce a NEW query_id; "
        f"old={done1.get('query_id')!r} new={new_qid!r}"
    )


def test_edit_last_409_on_stale_version(sandboxed_client: TestClient) -> None:
    """RED — Pattern C / D-17: edit-last with stale expected_version → 409."""
    chat_id = sandboxed_client.post("/api/chats", json={"title": "t"}).json()["id"]

    _drain_sse_done(
        sandboxed_client,
        chat_id,
        {"content": "hi", "message_uuid": "u1", "expected_version": 0},
    )

    r = sandboxed_client.post(
        f"/api/chats/{chat_id}/edit-last",
        json={"content": "edit", "expected_version": 0},
    )
    assert r.status_code == 409, r.text
    detail = r.json().get("detail", r.json())
    assert isinstance(detail.get("current_version"), int)


def test_edit_last_uses_fresh_message_uuid_not_replay(
    sandboxed_client: TestClient,
) -> None:
    """RED — Pitfall 4: the new assistant turn produced by edit-last has a
    DIFFERENT message_uuid than the original (so idempotency replay does
    not return the stale answer).
    """
    chat_id = sandboxed_client.post("/api/chats", json={"title": "t"}).json()["id"]

    _, first_done = _drain_sse_done(
        sandboxed_client,
        chat_id,
        {"content": "hi", "message_uuid": "u1", "expected_version": 0},
    )
    first_qid = first_done.get("query_id") if first_done else None

    r = sandboxed_client.post(
        f"/api/chats/{chat_id}/edit-last",
        json={"content": "hi-edited", "expected_version": 2},
    )
    assert r.status_code == 200, r.text
    body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
    new_qid = body.get("query_id")
    assert new_qid is not None and new_qid != first_qid, (
        "Pitfall 4: edit-last must not replay the prior assistant turn"
    )


# --------------------------------------------------------------------------- #
# CHAT-08: regenerate-last preserves audit dir
# --------------------------------------------------------------------------- #


def test_regenerate_preserves_audit_dir(sandboxed_client: TestClient, tmp_path: Path) -> None:
    """RED — CHAT-08 / D-20: regenerate-last discards the assistant row but
    the OLD logs/queries/<old_query_id>/ directory stays on disk.
    """
    chat_id = sandboxed_client.post("/api/chats", json={"title": "t"}).json()["id"]

    _, done = _drain_sse_done(
        sandboxed_client,
        chat_id,
        {"content": "hi", "message_uuid": "u1", "expected_version": 0},
    )
    assert done is not None
    old_qid = done.get("query_id")

    r = sandboxed_client.post(
        f"/api/chats/{chat_id}/regenerate-last",
        json={"expected_version": 2},
    )
    assert r.status_code == 200, r.text

    import os

    queries_root = Path(os.environ["QUERY_LOG_ROOT"])
    matching = [d for d in queries_root.rglob(f"*{(old_qid or '')[:8]}*")]
    assert matching, (
        f"CHAT-08: old audit dir for query_id={old_qid!r} should still exist under {queries_root}"
    )


# --------------------------------------------------------------------------- #
# CHAT-10: auto-name
# --------------------------------------------------------------------------- #


def test_auto_name_patches_on_first_done(sandboxed_client: TestClient, chats_db_path: Path) -> None:
    """RED — D-21: on first 'done' event for an Untitled chat, server PATCHes
    chats.title using routing_plan.title_hint.
    """
    r = sandboxed_client.post("/api/chats", json={"title": "Untitled chat"})
    chat_id = r.json()["id"]

    _drain_sse_done(
        sandboxed_client,
        chat_id,
        {"content": "what is GLFT?", "message_uuid": "u1", "expected_version": 0},
    )

    conn = sqlite3.connect(str(chats_db_path))
    try:
        title = conn.execute("SELECT title FROM chats WHERE id = ?", (chat_id,)).fetchone()[0]
    finally:
        conn.close()

    assert title and title != "Untitled chat", f"D-21: title should be auto-renamed; got {title!r}"


def test_auto_name_fallback_uses_usage_track(
    sandboxed_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RED — D-22: when routing_plan.title_hint is empty, the auto-name
    fallback routes through usage_track.call_text and records cost_usd > 0.
    """
    from unittest.mock import MagicMock

    from src.query import usage_track

    spy = MagicMock(wraps=usage_track.call_text)
    monkeypatch.setattr(usage_track, "call_text", spy)

    r = sandboxed_client.post("/api/chats", json={"title": "Untitled chat"})
    chat_id = r.json()["id"]
    _drain_sse_done(
        sandboxed_client,
        chat_id,
        {"content": "hi", "message_uuid": "u1", "expected_version": 0},
    )

    assert spy.call_count > 0, "D-22: auto-name fallback must use usage_track.call_text"


def test_auto_name_no_409_on_immediate_followup(
    sandboxed_client: TestClient,
) -> None:
    """RED — Pitfall 6: server-side auto-name PATCH must NOT bump chats.version
    in a way that races a client-side follow-up POST. Either (a) auto-name
    skips the version bump, or (b) the response carries the new version
    and the client's subsequent POST uses it.
    """
    chat_id = sandboxed_client.post("/api/chats", json={"title": "Untitled chat"}).json()["id"]

    _, done1 = _drain_sse_done(
        sandboxed_client,
        chat_id,
        {"content": "first", "message_uuid": "u1", "expected_version": 0},
    )

    fresh = sandboxed_client.get(f"/api/chats/{chat_id}")
    assert fresh.status_code == 200
    version = fresh.json().get("version", 2)

    r2 = sandboxed_client.post(
        f"/api/chats/{chat_id}/messages",
        json={"content": "follow", "message_uuid": "u2", "expected_version": version},
    )
    assert r2.status_code != 409, (
        f"Pitfall 6: immediate follow-up after auto-name should not 409; "
        f"got {r2.status_code} {r2.text}"
    )


# --------------------------------------------------------------------------- #
# HIST-03: PATCH / DELETE
# --------------------------------------------------------------------------- #


def test_patch_chat_rename(sandboxed_client: TestClient) -> None:
    """RED — HIST-03: PATCH /api/chats/{id} updates title."""
    chat_id = sandboxed_client.post("/api/chats", json={"title": "old"}).json()["id"]
    r = sandboxed_client.patch(
        f"/api/chats/{chat_id}",
        json={"title": "new", "expected_version": 0},
    )
    assert r.status_code == 200, r.text
    assert sandboxed_client.get(f"/api/chats/{chat_id}").json()["title"] == "new"


def test_patch_chat_archive_toggle(sandboxed_client: TestClient) -> None:
    """RED — HIST-03 + D-15: archived flag is a boolean toggle on PATCH."""
    chat_id = sandboxed_client.post("/api/chats", json={"title": "t"}).json()["id"]
    r = sandboxed_client.patch(
        f"/api/chats/{chat_id}",
        json={"archived": True, "expected_version": 0},
    )
    assert r.status_code == 200, r.text
    assert sandboxed_client.get(f"/api/chats/{chat_id}").json().get("archived") in (True, 1, "1")


def test_delete_chat_returns_204(sandboxed_client: TestClient) -> None:
    """RED — HIST-03: DELETE /api/chats/{id} returns 204."""
    chat_id = sandboxed_client.post("/api/chats", json={"title": "t"}).json()["id"]
    r = sandboxed_client.delete(f"/api/chats/{chat_id}")
    assert r.status_code == 204, r.text
    g = sandboxed_client.get(f"/api/chats/{chat_id}")
    assert g.status_code == 404


# --------------------------------------------------------------------------- #
# HIST-02: search
# --------------------------------------------------------------------------- #


def test_search_chats_endpoint_returns_chat_id_title_snippet(
    sandboxed_client: TestClient,
) -> None:
    """RED — HIST-02: GET /api/chats/search returns [{chat_id, title, snippet}]."""
    chat_id = sandboxed_client.post("/api/chats", json={"title": "GLFT market making"}).json()["id"]
    _drain_sse_done(
        sandboxed_client,
        chat_id,
        {
            "content": "the optimal bid ask spread for GLFT",
            "message_uuid": "u1",
            "expected_version": 0,
        },
    )

    r = sandboxed_client.get("/api/chats/search", params={"q": "GLFT"})
    assert r.status_code == 200, r.text
    rows = r.json()
    assert isinstance(rows, list) and rows, f"expected non-empty list; got {rows!r}"
    first = rows[0]
    assert {"chat_id", "title", "snippet"}.issubset(first.keys()), (
        f"row must have chat_id+title+snippet; got {first!r}"
    )


def test_search_chats_endpoint_400_on_fts5_syntax(
    sandboxed_client: TestClient,
) -> None:
    """RED — Pitfall 8: special FTS5 chars in q must NOT 500 (sanitize → 400 or
    sanitized 200; never 500).
    """
    r = sandboxed_client.get(
        "/api/chats/search",
        params={"q": '" '},
    )
    assert r.status_code != 500, (
        f"Pitfall 8: special FTS5 chars must not crash the server; got 500: {r.text}"
    )


# --------------------------------------------------------------------------- #
# D-06: SSE substage emission
# --------------------------------------------------------------------------- #


def test_sse_emits_paperqa_substages(
    sandboxed_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RED — D-06: when chat.retriever='paperqa', the SSE stream emits
    event: stage frames for paperqa_rerank, paperqa_read, paperqa_answer.
    """
    chat_id = sandboxed_client.post(
        "/api/chats", json={"title": "t", "retriever": "paperqa"}
    ).json()["id"]

    events, _ = _drain_sse_done(
        sandboxed_client,
        chat_id,
        {"content": "q", "message_uuid": "u1", "expected_version": 0},
    )

    stage_names = [(p.get("stage") or "") for n, p in events if n == "stage"]
    for sub in ("paperqa_rerank", "paperqa_read", "paperqa_answer"):
        assert sub in stage_names, f"D-06: SSE stage stream missing {sub!r}; got {stage_names}"


def test_sse_emits_fused_dispatches_all_four(
    sandboxed_client: TestClient,
) -> None:
    """RED — D-06: chat.retriever='fused' fans out — SSE stage stream
    carries fused_dispatch + fused_rerank AND at least one substage from
    each of the three optional retrievers.
    """
    chat_id = sandboxed_client.post("/api/chats", json={"title": "t", "retriever": "fused"}).json()[
        "id"
    ]

    events, _ = _drain_sse_done(
        sandboxed_client,
        chat_id,
        {"content": "q", "message_uuid": "u1", "expected_version": 0},
    )

    stage_names = [(p.get("stage") or "") for n, p in events if n == "stage"]

    assert "fused_dispatch" in stage_names, f"missing fused_dispatch: {stage_names}"
    assert "fused_rerank" in stage_names, f"missing fused_rerank: {stage_names}"
    assert any(s.startswith("paperqa_") for s in stage_names), (
        f"fused must dispatch paperqa; got {stage_names}"
    )
    assert any(s.startswith("hipporag_") for s in stage_names), (
        f"fused must dispatch hipporag; got {stage_names}"
    )
    assert any(s.startswith("lazygraph_") for s in stage_names), (
        f"fused must dispatch lazygraph; got {stage_names}"
    )


# --------------------------------------------------------------------------- #
# Plan 02-13 — GET /api/chats/retrievers (NewChatDialog availability map)
# --------------------------------------------------------------------------- #


def test_get_retriever_availability_shape(
    sandboxed_client: TestClient,
) -> None:
    """Plan 02-13 — GET /api/chats/retrievers returns a boolean map keyed on
    the five ToolName literals.

    milvus + fused are always True (milvus is built-in; fused dispatches the
    other 3 and gracefully degrades when any sub-retriever is stub'd).
    paperqa / hipporag / lazygraph mirror the import-time _AVAILABLE flags
    in their respective modules.
    """
    r = sandboxed_client.get("/api/chats/retrievers")
    assert r.status_code == 200, r.text
    body = r.json()
    assert isinstance(body, dict), f"expected dict; got {type(body).__name__}"
    assert set(body.keys()) == {
        "milvus",
        "paperqa",
        "hipporag",
        "lazygraph",
        "fused",
    }, f"unexpected keys: {sorted(body.keys())}"
    # Milvus + fused are always True.
    assert body["milvus"] is True, body
    assert body["fused"] is True, body
    # Optional retrievers are booleans (value depends on host install state).
    for key in ("paperqa", "hipporag", "lazygraph"):
        assert isinstance(body[key], bool), f"{key} should be bool; got {body[key]!r}"


def test_get_retriever_availability_reflects_module_flags(
    sandboxed_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Plan 02-13 — endpoint reads PAPERQA_AVAILABLE / HIPPORAG_AVAILABLE /
    LAZYGRAPH_AVAILABLE from the retriever modules at call time, so
    monkeypatching those flags flips the response.
    """
    import src.query.retrieve_hipporag as _hr
    import src.query.retrieve_lazygraph as _lg
    import src.query.retrieve_paperqa as _pq

    monkeypatch.setattr(_pq, "PAPERQA_AVAILABLE", False)
    monkeypatch.setattr(_hr, "HIPPORAG_AVAILABLE", True)
    monkeypatch.setattr(_lg, "LAZYGRAPH_AVAILABLE", False)

    r = sandboxed_client.get("/api/chats/retrievers")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["paperqa"] is False, body
    assert body["hipporag"] is True, body
    assert body["lazygraph"] is False, body
    # milvus + fused stay True regardless of the optional flags.
    assert body["milvus"] is True, body
    assert body["fused"] is True, body
