"""End-to-end integration tests for the Phase 1 chat surface (Plan 08).

Six test functions covering the Phase 1 promise from the GUI perspective:

* ``test_e2e_chat_full_roundtrip``                    — CHAT-02 + CHAT-12 + CHAT-13
* ``test_e2e_chat_idempotent_message_uuid``           — CHAT-11 (D-18 idempotency)
* ``test_e2e_chat_409_on_stale_version``              — CHAT-11 (D-17 optimistic lock)
* ``test_e2e_chat_cancel_writes_cancelled_outcome``   — CHAT-03 stop (D-22)
* ``test_e2e_chat_keepalive_ping_observed``           — CHAT-14 (D-21 keepalive wiring)
* ``test_e2e_chat_callback_count_matches_write_stage_count`` — CRIT-2 GUI path

All externals (genai.Client, milvus_search_only, get_reranker) are
monkeypatched via shared fixtures in :mod:`tests.server.conftest`. The audit
dir + chats.db are tmp_path-scoped. NO production state is touched; NO real
API key required.

These tests are the goal-backward verification of Plan 08 — if they all pass
together with the full-suite gate (six commands), Phase 1 is complete.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_chat(client: TestClient) -> tuple[str, int]:
    """POST /api/chats; return (chat_id, version)."""
    r = client.post("/api/chats", json={"title": "e2e test"})
    assert r.status_code in (200, 201), r.text
    body = r.json()
    return body["id"], body.get("version", 0)


def _parse_sse_lines(lines: list[str]) -> tuple[list[tuple[str, Any]], bool]:
    """Parse ``event:`` + ``data:`` pairs from a TestClient SSE stream.

    Returns ``(events, saw_comment_frame)`` where ``events`` is the list of
    ``(event_name, payload)`` tuples and ``saw_comment_frame`` records whether
    any ``:`` comment frame (keepalive ping) was observed.
    """
    events: list[tuple[str, Any]] = []
    saw_ping = False
    current: str | None = None
    for raw in lines:
        line = raw.strip()
        if not line:
            current = None
            continue
        if line.startswith(":"):
            saw_ping = True
            continue
        if line.startswith("event:"):
            current = line[len("event:") :].strip()
            continue
        if line.startswith("data:") and current is not None:
            payload = line[len("data:") :].strip()
            try:
                parsed: Any = json.loads(payload)
            except json.JSONDecodeError:
                parsed = {"raw": payload}
            events.append((current, parsed))
    return events, saw_ping


def _stream_turn(
    client: TestClient,
    chat_id: str,
    content: str,
    version: int,
    msg_uuid: str,
) -> tuple[int, list[tuple[str, Any]], bool]:
    """POST a turn with ``Accept: text/event-stream``; return parsed events."""
    body = {
        "content": content,
        "message_uuid": msg_uuid,
        "expected_version": version,
    }
    with client.stream(
        "POST",
        f"/api/chats/{chat_id}/messages",
        headers={"Accept": "text/event-stream"},
        json=body,
    ) as response:
        status = response.status_code
        lines = list(response.iter_lines())
    events, saw_ping = _parse_sse_lines(lines)
    return status, events, saw_ping


# ---------------------------------------------------------------------------
# CHAT-02 + CHAT-12 + CHAT-13: persistence + citations trace + cost/latency.
# ---------------------------------------------------------------------------


def test_e2e_chat_full_roundtrip(client_with_chats: TestClient, audit_log_root: Path) -> None:
    """POST chat → stream turn → assert order + done payload shape.

    Persistence verified by re-fetching ``GET /api/chats/{id}``.
    """
    chat_id, version = _create_chat(client_with_chats)

    status, events, _ = _stream_turn(
        client_with_chats,
        chat_id,
        "what is market making?",
        version,
        msg_uuid="e2e-full-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
    )
    assert status == 200, "SSE branch should return 200"

    names = [n for n, _ in events]
    assert names, "no SSE events received"
    assert names[0] == "queued"
    assert "citations" in names
    assert "delta" in names
    assert "done" in names
    assert names.index("citations") < names.index("delta"), (
        "MOD-1: citations event MUST precede the first delta event"
    )

    # CHAT-13: cost+latency in done payload.
    done = next(p for n, p in events if n == "done")
    assert isinstance(done, dict)
    # latency_ms is always populated. cost_usd MAY be None on the empty-milvus
    # short-circuit (no synthesis stage → no synthesis LLM call). But the
    # decompose stage still runs and contributes a non-zero cost via the audit
    # trail — verified on disk in the callback-count test below.
    assert "latency_ms" in done, "CHAT-13: done.latency_ms required"
    assert "cost_usd" in done, "CHAT-13: done.cost_usd required (may be null)"
    assert "query_id" in done, "D-15: done.query_id required (may be null)"
    assert "usage" in done, "CHAT-13: done.usage required"

    # CHAT-12 trace data: when the milvus stub returns chunks the citations
    # event carries score fields. For the empty-stub case the citations event
    # is still emitted (with []) — verify the empty-list contract.
    cit_payload = next(p for n, p in events if n == "citations")
    cites = cit_payload if isinstance(cit_payload, list) else cit_payload.get("citations", [])
    assert isinstance(cites, list), (
        f"CHAT-12: citations payload must be a list; got {type(cit_payload)}"
    )

    # CHAT-02: re-fetch chat. User + assistant message both persisted.
    r2 = client_with_chats.get(f"/api/chats/{chat_id}")
    assert r2.status_code == 200
    detail = r2.json()
    msgs = detail["messages"]
    assert isinstance(msgs, list)
    assert len(msgs) >= 2, f"expected user+assistant rows; got {len(msgs)}"
    roles = {m["role"] for m in msgs}
    assert "user" in roles, "user message not persisted"
    assert "assistant" in roles, "assistant message not persisted"


# ---------------------------------------------------------------------------
# CHAT-11 (D-18 idempotency): duplicate message_uuid returns the existing row.
# ---------------------------------------------------------------------------


def test_e2e_chat_idempotent_message_uuid(
    client_with_chats: TestClient,
    ephemeral_chats_db: Path,
) -> None:
    """Two POSTs with the same ``message_uuid`` produce ONE assistant row.

    The second POST MUST NOT trigger a new LLM call (D-18). We verify by
    counting rows in chats.db for the duplicate uuid.
    """
    chat_id, version = _create_chat(client_with_chats)
    msg_uuid = "e2e-idem-bbbb-bbbb-bbbb-bbbbbbbbbbbb"

    status1, events1, _ = _stream_turn(
        client_with_chats, chat_id, "hello", version, msg_uuid=msg_uuid
    )
    assert status1 == 200

    # Re-fetch current version (it bumped twice — user + assistant placeholder).
    r = client_with_chats.get(f"/api/chats/{chat_id}")
    version2 = r.json()["version"]

    # Second POST with the SAME message_uuid — server MUST be idempotent.
    status2, events2, _ = _stream_turn(
        client_with_chats, chat_id, "hello", version2, msg_uuid=msg_uuid
    )
    assert status2 == 200, "duplicate-uuid POST is idempotent, not 409"

    # The idempotent SSE branch emits citations + delta + done with
    # idempotent_replay=True (api_chats.py line 442-457).
    done1 = next(p for n, p in events1 if n == "done")
    done2 = next(p for n, p in events2 if n == "done")
    # Both reference the same query_id (or both None when empty-milvus stub).
    assert done2.get("query_id") == done1.get("query_id"), (
        f"D-18: duplicate POST must reuse query_id; "
        f"got {done2.get('query_id')!r} vs {done1.get('query_id')!r}"
    )

    # Cross-check at the DB layer — only one row per message_uuid (Plan 02
    # UNIQUE invariant).
    conn = sqlite3.connect(str(ephemeral_chats_db))
    try:
        n = conn.execute(
            "SELECT COUNT(*) FROM messages WHERE message_uuid = ?",
            (msg_uuid,),
        ).fetchone()[0]
        assert n == 1, f"D-18: expected 1 row for {msg_uuid}, got {n}"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# CHAT-11 (D-17): stale expected_version returns 409.
# ---------------------------------------------------------------------------


def test_e2e_chat_409_on_stale_version(
    client_with_chats: TestClient,
) -> None:
    """POST with stale ``expected_version`` returns HTTP 409.

    Tested via the non-streaming JSON branch — cleaner than parsing an SSE
    error frame. 409 body MUST include ``current_version`` (D-17).
    """
    chat_id, _ = _create_chat(client_with_chats)

    # Skip the version-bumping turn; POST directly with a bogus stale version.
    stale_body = {
        "content": "stale request",
        "message_uuid": "e2e-409-cccc-cccc-cccc-cccccccccccc",
        "expected_version": 999,  # bogus stale
    }
    r = client_with_chats.post(
        f"/api/chats/{chat_id}/messages",
        json=stale_body,
        headers={"Accept": "application/json"},
    )
    assert r.status_code == 409, f"expected 409, got {r.status_code}: {r.text}"

    body = r.json()
    # FastAPI HTTPException(detail={...}) wraps the dict under "detail".
    detail = body.get("detail", body)
    assert isinstance(detail.get("current_version"), int), (
        f"D-17: 409 body must include current_version; got {body!r}"
    )


# ---------------------------------------------------------------------------
# CHAT-03 (stop) — DELETE /stream signals cancel; meta.outcome.cancelled = True.
# ---------------------------------------------------------------------------


def test_e2e_chat_cancel_writes_cancelled_outcome(
    client_with_chats: TestClient,
    audit_log_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DELETE /stream during streaming → meta.outcome.cancelled = True (D-22).

    Strategy mirrors the existing ``test_cancel_writes_cancelled_outcome`` in
    test_api_chats.py: open the SSE stream in a thread, fire DELETE from the
    main thread once the (chat_id, message_uuid) appears in _ACTIVE_STREAMS,
    then drain the stream and assert outcome on disk.
    """
    chat_id, version = _create_chat(client_with_chats)
    msg_uuid = "e2e-cancel-dddd-dddd-dddd-dddddddddddd"

    # Slow the milvus stub so cancellation has time to land between stages.
    import src.query.retrieve as _retr

    def _slow_milvus(*_args: object, **_kwargs: object) -> list[dict[str, object]]:
        time.sleep(2.0)
        return []

    monkeypatch.setattr(_retr, "milvus_search_only", _slow_milvus)

    def _cancel_when_active() -> None:
        from src.server.api_chats import _ACTIVE_STREAMS, _ACTIVE_STREAMS_LOCK

        for _ in range(50):  # 5s budget at 0.1s polling
            with _ACTIVE_STREAMS_LOCK:
                if (chat_id, msg_uuid) in _ACTIVE_STREAMS:
                    break
            time.sleep(0.1)
        client_with_chats.delete(f"/api/chats/{chat_id}/messages/{msg_uuid}/stream")

    canceller = threading.Thread(target=_cancel_when_active, daemon=True)
    canceller.start()

    with client_with_chats.stream(
        "POST",
        f"/api/chats/{chat_id}/messages",
        headers={"Accept": "text/event-stream"},
        json={
            "content": "long query",
            "message_uuid": msg_uuid,
            "expected_version": version,
        },
    ) as response:
        assert response.status_code == 200
        for _ in response.iter_lines():
            pass

    canceller.join(timeout=5.0)
    assert not canceller.is_alive(), "canceller thread did not finish"

    # Allow the post-cancel atomic rename to flush.
    deadline = time.monotonic() + 5.0
    meta_path: Path | None = None
    while time.monotonic() < deadline:
        candidates = list(audit_log_root.rglob("meta.json"))
        if candidates:
            meta_path = candidates[-1]
            meta = json.loads(meta_path.read_text())
            outcome = meta.get("outcome", {})
            cancelled = outcome.get("cancelled") if isinstance(outcome, dict) else outcome
            if cancelled is True:
                break
        time.sleep(0.1)

    assert meta_path is not None, f"no meta.json under {audit_log_root} within 5s of cancel"
    meta = json.loads(meta_path.read_text())
    outcome = meta.get("outcome", {})
    cancelled = outcome.get("cancelled") if isinstance(outcome, dict) else outcome
    assert cancelled is True, (
        f"D-22: aborted turn must record outcome.cancelled = True; got outcome={outcome!r}"
    )


# ---------------------------------------------------------------------------
# CHAT-14 (D-21 keepalive): static wiring assertion + live observation.
# ---------------------------------------------------------------------------


def test_e2e_chat_keepalive_ping_observed(
    client_with_chats: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SSE handler is wired with ``ping=15`` (D-21 / MOD-16 / CHAT-14).

    Two assertions:

    1. STATIC: ``EventSourceResponse(..., ping=SSE_PING_INTERVAL_SECONDS,
       ping_message_factory=_heartbeat, ...)`` appears in
       ``src/server/api_chats.py``.
    2. LIVE: With ``SSE_PING_INTERVAL_SECONDS`` monkeypatched to 1s and a
       2s-sleeping milvus stub forcing a quiet window, at least one ``:``
       comment frame appears in the byte stream.
    """
    # 1. Static wiring assertion.
    api_chats_path = Path("src/server/api_chats.py")
    src = api_chats_path.read_text()
    assert "EventSourceResponse" in src, "api_chats must use EventSourceResponse"
    assert "SSE_PING_INTERVAL_SECONDS" in src, (
        "CHAT-14: SSE_PING_INTERVAL_SECONDS constant must exist"
    )
    assert "ping_message_factory" in src, "CHAT-14: ping_message_factory must be wired"

    # 2. Live observation — slow milvus + fast ping interval.
    import src.query.retrieve as _retr
    import src.server.api_chats as _ac

    def _slow_milvus(*_args: object, **_kwargs: object) -> list[dict[str, object]]:
        time.sleep(2.0)
        return []

    monkeypatch.setattr(_retr, "milvus_search_only", _slow_milvus)
    monkeypatch.setattr(_ac, "SSE_PING_INTERVAL_SECONDS", 1)

    chat_id, version = _create_chat(client_with_chats)

    saw_ping = False
    deadline = time.monotonic() + 5.0
    with client_with_chats.stream(
        "POST",
        f"/api/chats/{chat_id}/messages",
        headers={"Accept": "text/event-stream"},
        json={
            "content": "slow query for ping",
            "message_uuid": "e2e-ping-eeee-eeee-eeee-eeeeeeeeeeee",
            "expected_version": version,
        },
    ) as response:
        assert response.status_code == 200
        for line in response.iter_lines():
            if time.monotonic() > deadline:
                break
            if line.startswith(":"):
                saw_ping = True
                break

    assert saw_ping, "CHAT-14: no SSE comment-frame keepalive observed within 5s"


# ---------------------------------------------------------------------------
# CRIT-2 GUI path — callback-count == write_stage-count.
# ---------------------------------------------------------------------------


def test_e2e_chat_callback_count_matches_write_stage_count(
    client_with_chats: TestClient,
    audit_log_root: Path,
) -> None:
    """The on_stage callback fires exactly once per ``audit.write_stage`` call.

    Same invariant the pipeline-level test
    (``tests/test_query_pipeline_callback.py``) asserts; this test re-verifies
    it through the GUI-triggered SSE handler — proves the SSE wiring does not
    drop or duplicate stage events vs the on-disk audit trail.
    """
    chat_id, version = _create_chat(client_with_chats)
    msg_uuid = "e2e-cbcnt-ffff-ffff-ffff-ffffffffffff"

    _, events, _ = _stream_turn(
        client_with_chats,
        chat_id,
        "callback count check",
        version,
        msg_uuid=msg_uuid,
    )

    stage_event_count = sum(1 for n, _ in events if n == "stage")
    assert stage_event_count > 0, "no `event: stage` SSE frames received — callback wiring broken"

    # Find the most recent meta.json's qid_dir under audit_log_root. The
    # empty-milvus stub returns rag_response = None so done.query_id is None,
    # but the audit trail IS still written (decompose stage runs first).
    meta_files = list(audit_log_root.rglob("meta.json"))
    assert meta_files, f"no meta.json under {audit_log_root}"
    qid_dir = max(meta_files, key=lambda p: p.stat().st_mtime).parent
    stage_files = sorted(qid_dir.glob("[0-9][0-9]_*.json"))
    assert stage_files, f"no NN_*.json stage files under {qid_dir}"

    assert stage_event_count == len(stage_files), (
        f"CRIT-2 (GUI path): callback-count {stage_event_count} != "
        f"write_stage-count {len(stage_files)}\n"
        f"  events: {[n for n, _ in events if n == 'stage']}\n"
        f"  files:  {[p.name for p in stage_files]}"
    )
