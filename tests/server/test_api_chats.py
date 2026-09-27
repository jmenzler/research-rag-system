# long-ok-file
"""Wave-0 RED tests for src/server/api_chats.py.

Implementation lands in Plans 02 (chats_store) / 03 (api_chats router) / 04 (callback).
These tests are stubs that fail predictably until the production module exists.
Collection succeeds because every import of `src.server.api_chats` is deferred
inside the test body (pytest.importorskip or local imports), so pytest --collect-only
emits the function list without ever resolving the missing module.

Analogs (cited verbatim from 01-PATTERNS.md §tests/server/test_api_chats.py):
  - tests/server/test_api_meta.py:10-22   TestClient + APIRouter pattern
  - tests/server/test_audit_smoke.py:99-189   genai.Client monkeypatch fixture
  - tests/server/test_migrations.py:26-44   tmp_path sqlite + migrate.run pattern
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    pass

# --------------------------------------------------------------------------- #
# Shared fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def chats_db_path(tmp_path: Path) -> Path:
    """tmp_path-scoped chats.db with all migrations applied.

    Mirrors tests/server/test_migrations.py:26-44 shape.
    Once Plan 02 ships 0002_chats.sql, this returns a db with user_version >= 2.
    """
    from src.server.migrations import migrate

    db = tmp_path / "chats.db"
    migrate.run(db)
    return db


def _fake_genai_monkeypatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replicate tests/server/test_audit_smoke.py:115-142 genai monkeypatch.

    Replaces the genai.Client constructor with a stub that returns a synthetic
    response carrying usage_metadata, so the full call_text code path executes
    without hitting Gemini's API.  No real API key needed in CI.
    """
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
    monkeypatch.setattr(_config, "DECOMPOSE_MODEL", "gemini-2.5-flash")


def _stub_milvus_for_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the Milvus + reranker entrypoints so pipeline runs without a server.

    Returns an empty hit list — `_retrieve` then short-circuits at
    `batched_rerank_and_lookup([])` (no real reranker call) and the pipeline
    early-returns from `if not retrieved`. Stage files for decompose + milvus
    + rerank still land on disk, which is enough for the SSE event-order
    assertion.
    """
    from typing import Any as _Any  # noqa: PLC0415

    import src.query.retrieve as _retr  # noqa: PLC0415

    def _fake_milvus(*_args: object, **_kwargs: object) -> list[dict[str, _Any]]:
        return []

    def _fake_embed(_query: str) -> list[float]:
        return [0.0] * 4

    monkeypatch.setattr(_retr, "milvus_search_only", _fake_milvus)
    monkeypatch.setattr(_retr, "validate_api_key", lambda: None)
    monkeypatch.setattr(_retr, "_embed_query", _fake_embed)


def _redirect_chats_db(monkeypatch: pytest.MonkeyPatch, chats_db_path: Path) -> None:
    """Point ``src.server.chats_store.CHATS_DB_PATH`` at the tmp_path db.

    ``open_chats_conn`` reads the module-level CHATS_DB_PATH from chats_store
    (NOT from app.py). The fixture creates the db at tmp_path/chats.db — this
    helper makes the production code actually use it.
    """
    import src.server.chats_store as _cs  # noqa: PLC0415

    monkeypatch.setattr(_cs, "CHATS_DB_PATH", chats_db_path)


def _point_query_log_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Redirect QueryLogger writes into tmp_path so tests are sandboxed."""
    queries_root = tmp_path / "queries"
    monkeypatch.setenv("QUERY_LOG_ROOT", str(queries_root))
    return queries_root


# --------------------------------------------------------------------------- #
# CHAT-02: persistence across reload
# --------------------------------------------------------------------------- #


def test_message_persists_across_reload(
    chats_db_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CHAT-02: user message survives a server reload.

    Wave 1+: POST a chat, POST a message via SSE, drain stream, GET chat back,
    assert the user content is in the messages list.
    """
    pytest.importorskip("src.server.api_chats")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.server.api_chats import router

    _fake_genai_monkeypatch(monkeypatch)
    _stub_milvus_for_tests(monkeypatch)
    _redirect_chats_db(monkeypatch, chats_db_path)
    _point_query_log_root(monkeypatch, tmp_path)

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    create = client.post("/api/chats", json={"title": "test"})
    assert create.status_code in (200, 201), create.text
    chat_id = create.json()["id"]

    # SSE round-trip — Accept: text/event-stream triggers the streaming branch.
    with client.stream(
        "POST",
        f"/api/chats/{chat_id}/messages",
        json={
            "content": "hello world",
            "message_uuid": "uuid-persist-1",
            "expected_version": 0,
        },
        headers={"Accept": "text/event-stream"},
    ) as response:
        assert response.status_code == 200
        for _ in response.iter_lines():
            pass

    fetched = client.get(f"/api/chats/{chat_id}")
    assert fetched.status_code == 200
    body = fetched.json()
    assert isinstance(body.get("messages"), list)
    assert len(body["messages"]) >= 2  # user + assistant
    user_msgs = [m for m in body["messages"] if m["role"] == "user"]
    assert any(m["content"] == "hello world" for m in user_msgs)


# --------------------------------------------------------------------------- #
# CHAT-03 (stream): SSE event order
# --------------------------------------------------------------------------- #


def test_sse_event_order(
    chats_db_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CHAT-03: SSE event taxonomy (D-20) emits events in fixed order.

    Sketch from 01-RESEARCH.md §Test Strategy lines 874-885.
    """
    pytest.importorskip("src.server.api_chats")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.server.api_chats import router

    _fake_genai_monkeypatch(monkeypatch)
    _stub_milvus_for_tests(monkeypatch)
    _redirect_chats_db(monkeypatch, chats_db_path)
    _point_query_log_root(monkeypatch, tmp_path)

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    chat_id = client.post("/api/chats", json={"title": "t"}).json()["id"]

    events: list[str] = []
    with client.stream(
        "POST",
        f"/api/chats/{chat_id}/messages",
        json={
            "content": "ping",
            "message_uuid": "uuid-order-1",
            "expected_version": 0,
        },
        headers={"Accept": "text/event-stream"},
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        for line in response.iter_lines():
            if line.startswith("event:"):
                events.append(line[len("event:") :].strip())

    assert events, "no SSE event lines observed"
    assert events[0] == "queued"
    assert "stage" in events
    assert "citations" in events
    assert "delta" in events
    assert events[-1] == "done"

    # The citations event MUST precede the first delta (MOD-1).
    cit_idx = events.index("citations")
    delta_idx = events.index("delta")
    assert cit_idx < delta_idx, f"citations must precede delta (MOD-1). order={events}"


# --------------------------------------------------------------------------- #
# CHAT-03 (stop): cancellation writes outcome=cancelled
# --------------------------------------------------------------------------- #


def test_cancel_writes_cancelled_outcome(
    chats_db_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """D-22: cancelling an SSE stream via DELETE writes meta.json with
    outcome.cancelled = True (no half-written stage files allowed).

    Strategy: rather than relying on TestClient's client-side abort signal
    propagating into the SSE generator's ``request.is_disconnected()`` poll
    (which is not reliable inside the in-process TestClient ASGI bridge), this
    test calls the explicit DELETE /stream endpoint while the SSE generator
    is yielding events. That hits the same code path (cancel_event.set →
    pipeline raises CancelledError → _record_cancellation_outcome).
    """
    pytest.importorskip("src.server.api_chats")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.server.api_chats import router

    _fake_genai_monkeypatch(monkeypatch)
    _stub_milvus_for_tests(monkeypatch)
    _redirect_chats_db(monkeypatch, chats_db_path)
    queries_root = _point_query_log_root(monkeypatch, tmp_path)

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    chat_id = client.post("/api/chats", json={"title": "t"}).json()["id"]

    message_uuid = "uuid-cancel-1"
    retrieval_started = threading.Event()
    cancellation_sent = threading.Event()
    cancel_statuses: list[int] = []

    def wait_for_cancellation(*_args: object, **_kwargs: object) -> list[dict[str, object]]:
        retrieval_started.set()
        if not cancellation_sent.wait(timeout=5.0):
            raise TimeoutError("Cancellation request did not complete")
        return []

    monkeypatch.setattr("src.query.retrieve.milvus_search_only", wait_for_cancellation)

    def _cancel_after_first_stage() -> None:
        if not retrieval_started.wait(timeout=5.0):
            return
        try:
            response = client.delete(f"/api/chats/{chat_id}/messages/{message_uuid}/stream")
            cancel_statuses.append(response.status_code)
        finally:
            cancellation_sent.set()

    canceller = threading.Thread(target=_cancel_after_first_stage, daemon=True)
    canceller.start()

    with client.stream(
        "POST",
        f"/api/chats/{chat_id}/messages",
        json={
            "content": "long query",
            "message_uuid": message_uuid,
            "expected_version": 0,
        },
        headers={"Accept": "text/event-stream"},
    ) as response:
        assert response.status_code == 200
        # Drain — the generator will exit (either via cancel or completion).
        for _ in response.iter_lines():
            pass

    canceller.join(timeout=5.0)
    assert cancel_statuses == [200]

    # Allow the post-cancel _record_cancellation_outcome IO to flush.
    deadline = time.monotonic() + 5.0
    meta_path: Path | None = None
    while time.monotonic() < deadline:
        candidates = list(queries_root.rglob("meta.json"))
        if candidates:
            meta_path = candidates[-1]
            # Re-read to make sure we see the post-cancel rewrite.
            meta = json.loads(meta_path.read_text())
            if meta.get("outcome", {}).get("cancelled") is True:
                break
        time.sleep(0.1)

    assert meta_path is not None, f"no meta.json written under {queries_root} within 5s of cancel"
    meta = json.loads(meta_path.read_text())
    assert "totals" in meta
    outcome = meta.get("outcome", {})
    assert outcome.get("cancelled") is True, (
        f"expected outcome.cancelled=True; got outcome={outcome!r}"
    )


# --------------------------------------------------------------------------- #
# CHAT-11 (lock): stale version → 409
# --------------------------------------------------------------------------- #


def test_stale_version_returns_409(
    chats_db_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CHAT-11 / D-17: POST with stale expected_version returns 409 + current_version."""
    pytest.importorskip("src.server.api_chats")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.server.api_chats import router

    _fake_genai_monkeypatch(monkeypatch)
    _stub_milvus_for_tests(monkeypatch)
    _redirect_chats_db(monkeypatch, chats_db_path)
    _point_query_log_root(monkeypatch, tmp_path)

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    chat_id = client.post("/api/chats", json={"title": "t"}).json()["id"]

    response = client.post(
        f"/api/chats/{chat_id}/messages",
        json={
            "content": "hi",
            "message_uuid": "u-1",
            "expected_version": 999,
        },
    )
    assert response.status_code == 409, response.text
    body = response.json()
    # FastAPI HTTPException(detail={...}) wraps the dict under "detail".
    detail = body.get("detail", body)
    assert isinstance(detail.get("current_version"), int)


# --------------------------------------------------------------------------- #
# CHAT-11 (idem): duplicate UUID idempotency
# --------------------------------------------------------------------------- #


def test_duplicate_uuid_idempotent(
    chats_db_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CHAT-11 / D-18: second POST with same message_uuid returns the existing row."""
    import sqlite3

    pytest.importorskip("src.server.api_chats")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.server.api_chats import router

    _fake_genai_monkeypatch(monkeypatch)
    _stub_milvus_for_tests(monkeypatch)
    _redirect_chats_db(monkeypatch, chats_db_path)
    _point_query_log_root(monkeypatch, tmp_path)

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    chat_id = client.post("/api/chats", json={"title": "t"}).json()["id"]

    def _drain_done_sse(payload: dict[str, object]) -> dict[str, object]:
        done_data: dict[str, object] | None = None
        with client.stream(
            "POST",
            f"/api/chats/{chat_id}/messages",
            json=payload,
            headers={"Accept": "text/event-stream"},
        ) as response:
            assert response.status_code == 200
            cur_event: str | None = None
            for raw in response.iter_lines():
                line = raw.strip()
                if not line:
                    cur_event = None
                    continue
                if line.startswith("event:"):
                    cur_event = line[len("event:") :].strip()
                elif line.startswith("data:") and cur_event == "done":
                    done_data = json.loads(line[len("data:") :].strip())
                    break
        assert done_data is not None, "no event: done seen"
        return done_data

    first = _drain_done_sse({"content": "hi", "message_uuid": "u-dup", "expected_version": 0})
    second = _drain_done_sse({"content": "hi", "message_uuid": "u-dup", "expected_version": 1})
    assert first["query_id"] == second["query_id"], (
        f"idempotent POST should reuse query_id; got {first} vs {second}"
    )

    # Only one row in messages with this uuid.
    conn = sqlite3.connect(str(chats_db_path))
    try:
        n = conn.execute(
            "SELECT COUNT(*) FROM messages WHERE message_uuid = ?", ("u-dup",)
        ).fetchone()[0]
        assert n == 1, f"expected 1 row for u-dup, got {n}"
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# CHAT-14: SSE keepalive ping within 20s
# --------------------------------------------------------------------------- #


def test_sse_emits_ping_within_20s(
    chats_db_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CHAT-14 / D-21 / MOD-16: server emits SSE comment-frame keepalives.

    sse-starlette ping_message_factory emits `:hb <iso>\\n\\n` comment frames.
    The test shortens the ping interval so it runs in <5s.

    We force a slow pipeline by monkeypatching the in-process milvus stub to
    sleep before returning, so the ping has time to fire BEFORE the pipeline
    completes (otherwise the generator finishes in <0.5s and the ping never
    has a quiet window to fire into).
    """
    import time as _time  # noqa: PLC0415

    pytest.importorskip("src.server.api_chats")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.server.api_chats import router

    _fake_genai_monkeypatch(monkeypatch)

    # Slow milvus stub — sleep 2s before returning empty.
    import src.query.retrieve as _retr  # noqa: PLC0415

    def _slow_fake_milvus(*_args: object, **_kwargs: object) -> list[dict[str, object]]:
        _time.sleep(2.0)
        return []

    monkeypatch.setattr(_retr, "milvus_search_only", _slow_fake_milvus)
    monkeypatch.setattr(_retr, "validate_api_key", lambda: None)
    monkeypatch.setattr(_retr, "_embed_query", lambda _q: [0.0] * 4)

    _redirect_chats_db(monkeypatch, chats_db_path)
    _point_query_log_root(monkeypatch, tmp_path)

    # Compress the ping interval to 1s so we can observe it quickly.
    import src.server.api_chats as _ac  # noqa: PLC0415

    monkeypatch.setattr(_ac, "SSE_PING_INTERVAL_SECONDS", 1)

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    chat_id = client.post("/api/chats", json={"title": "t"}).json()["id"]

    saw_ping = False
    deadline = time.monotonic() + 5.0
    with client.stream(
        "POST",
        f"/api/chats/{chat_id}/messages",
        json={
            "content": "slow query",
            "message_uuid": "uuid-ping-1",
            "expected_version": 0,
        },
        headers={"Accept": "text/event-stream"},
    ) as response:
        assert response.status_code == 200
        for line in response.iter_lines():
            if time.monotonic() > deadline:
                break
            # sse-starlette comment frame prefix: line begins with `:` (e.g. ":hb ...")
            if line.startswith(":"):
                saw_ping = True
                break

    assert saw_ping, "no SSE comment-frame keepalive observed within 5s"


# --------------------------------------------------------------------------- #
# Finding 4: non-SSE post_message must offload the blocking pipeline to a
# worker thread (FastAPI does NOT threadpool async-def handlers — running the
# pipeline inline would stall the single uvicorn event loop).
# --------------------------------------------------------------------------- #


def test_post_message_offloads_blocking_pipeline(
    chats_db_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Finding 4: the JSON (non-SSE) branch runs _post_message_sync via
    asyncio.to_thread, so the blocking pipeline does not stall the event loop.

    Spies on asyncio.to_thread (as referenced by the api_chats module) and
    asserts _post_message_sync was dispatched through it during a non-SSE POST.
    """
    import asyncio

    pytest.importorskip("src.server.api_chats")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.server import api_chats as _ac

    _redirect_chats_db(monkeypatch, chats_db_path)
    _point_query_log_root(monkeypatch, tmp_path)

    offloaded: list[object] = []
    real_to_thread = asyncio.to_thread

    async def _spy_to_thread(func: object, *args: object, **kwargs: object) -> object:
        offloaded.append(func)
        return await real_to_thread(func, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(_ac.asyncio, "to_thread", _spy_to_thread)

    # Replace the heavy sync body so the offloaded call returns immediately.
    monkeypatch.setattr(
        _ac,
        "_post_message_sync",
        lambda *a, **k: {"message_id": "stub", "content": "ok"},
    )

    app = FastAPI()
    app.include_router(_ac.router)
    client = TestClient(app)
    chat_id = client.post("/api/chats", json={"title": "t"}).json()["id"]

    resp = client.post(
        f"/api/chats/{chat_id}/messages",
        json={"content": "hi", "message_uuid": "offload-1", "expected_version": 0},
    )
    assert resp.status_code == 200, resp.text
    assert offloaded, (
        "post_message did not call asyncio.to_thread — the blocking pipeline "
        "runs inline on the event loop"
    )
    assert any(f is _ac._post_message_sync for f in offloaded), (
        "_post_message_sync was not the offloaded callable"
    )
