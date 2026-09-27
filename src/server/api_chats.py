# long-ok-file
"""POST /api/chats/{id}/messages — Phase 1 chat surface (sync JSON + SSE).

Plan 03 shipped the synchronous JSON form. Plan 04 adds the SSE branch
(:class:`EventSourceResponse`) — selected when the client sends
``Accept: text/event-stream`` — that wraps the same orchestration spine with
a per-stage callback (D-26) and cancellation propagation (D-22).

Brownfield invariant: :func:`run_query_pipeline` is called in-process (not as a
``bin/rag`` subprocess) so the ``on_stage`` callback can wire into the existing
pipeline. The audit-trail discipline (``usage_track.call_text`` + audit
``write_stage``) is preserved because the existing pipeline enforces it; this
module neither bypasses nor reimplements that path.

Endpoints:
  POST /api/chats                                              — create a new chat
  GET  /api/chats                                              — list all chats
  GET  /api/chats/{chat_id}                                    — fetch one chat
  POST /api/chats/{chat_id}/messages                           — append turn(s):
                                                                  sync JSON OR SSE
  DELETE /api/chats/{chat_id}/messages/{message_uuid}/stream   — cancel in-flight

SSE event taxonomy (D-20):
  event: queued    {"chat_id", "message_uuid"}
  event: stage     {"stage", "latency_ms"}        — N times, post audit.write_stage
  event: citations [CitationDict, ...]            — BEFORE the first delta (MOD-1)
  event: delta     {"text": ...}                  — single delta (V2-01 is tokens)
  event: done      {"query_id", "usage", "cost_usd", "latency_ms"}
  event: error     {"message", ...}

Keepalive: sse-starlette emits ``: hb <iso>`` comment frames every ``ping=15``
seconds. ``X-Accel-Buffering: no`` defeats nginx response buffering; together
they reduce idle disconnects through intermediaries.

Optimistic-lock + idempotency contract (D-17 / D-18):
  - 409 on stale ``expected_version`` (body has ``current_version``).
  - Duplicate ``message_uuid`` returns the existing assistant content + query_id
    without re-running the pipeline.
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
import uuid
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Header, HTTPException, Query, Request, Response
from pydantic import BaseModel, ConfigDict, Field
from sse_starlette.event import ServerSentEvent
from sse_starlette.sse import EventSourceResponse

from src.server.chats_store import (
    ConflictError,
    atomic_edit_last,
    attach_citations,
    auto_name_chat,
    create_chat,
    delete_chat,
    get_chat,
    insert_message_idempotent,
    list_chats,
    open_chats_conn,
    search_chats,
    set_assistant_content_and_query_id,
    set_message_meta,
    truncate_last_assistant,
    update_chat,
)
from src.server.citations import validate_citations

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")

# PROJECT_ROOT — used by _record_cancellation_outcome to locate logs/queries
# when the test env has not redirected QUERY_LOG_ROOT.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# SSE keepalive interval (seconds). D-21 / MOD-16 / CHAT-14: 15s sits well under
# the 30s typical proxy/NAT idle threshold. Exposed as a module attribute so
# tests can monkeypatch it for fast-feedback heartbeat assertions.
SSE_PING_INTERVAL_SECONDS = 15

# In-flight SSE streams keyed by (chat_id, message_uuid). The DELETE handler
# looks up the cancel_event and sets it; the SSE generator polls it BETWEEN
# stages (via the run_query_pipeline `cancel_event` kwarg). Mutated by the
# generator and the DELETE handler — guarded by _ACTIVE_STREAMS_LOCK.
#
# Each entry stores:
#   - cancel_event: threading.Event the DELETE handler sets to abort the
#     in-flight pipeline.
#   - audit_dir: the per-query audit directory once make_query_logger has run
#     (populated via the on_audit_dir callback). None until the first stage
#     starts. WR-05: targeting THIS specific meta.json on cancellation avoids
#     marking another concurrent run's meta.json (CLI eval, parallel chat).
class _StreamHandle:
    __slots__ = ("cancel_event", "audit_dir")

    def __init__(self) -> None:
        self.cancel_event: threading.Event = threading.Event()
        self.audit_dir: Path | None = None


_ACTIVE_STREAMS: dict[tuple[str, str], _StreamHandle] = {}
_ACTIVE_STREAMS_LOCK = threading.Lock()


def _heartbeat() -> ServerSentEvent:
    """sse-starlette ``ping_message_factory``.

    Emits a comment frame ``:hb <iso-ts>\\n\\n`` every ``ping`` seconds. The
    payload's only requirement is that it be non-empty so middleboxes see live
    bytes; the ISO timestamp is useful for log grepping when debugging idle
    disconnects through intermediaries.
    """
    return ServerSentEvent(comment=f"hb {datetime.now(tz=UTC).isoformat()}")


class CreateChatRequest(BaseModel):
    """Body for ``POST /api/chats``. All fields optional (sensible defaults)."""

    title: str | None = "Untitled chat"
    retriever: str = "milvus"
    collections: list[str] | None = None


class PostMessageRequest(BaseModel):
    """Body for ``POST /api/chats/{chat_id}/messages``.

    ``message_uuid`` is generated client-side (``crypto.randomUUID()``, D-18) to
    provide an idempotency key — a retried POST with the same uuid returns the
    existing assistant row instead of re-running the pipeline.

    ``expected_version`` is the chat's version that the client believes is
    current (D-17). The server CAS's that against the row's ``version`` and
    raises ConflictError → HTTP 409 if it does not match.
    """

    content: str
    message_uuid: str
    expected_version: int


class PatchChatRequest(BaseModel):
    """Body for ``PATCH /api/chats/{chat_id}`` (HIST-03).

    ``extra="forbid"`` blocks mass-assignment of id/version/created_at/
    updated_at (T-02-04-01 mitigation): unknown fields return HTTP 422
    before the request ever reaches the store layer.
    """

    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(default=None, max_length=200)
    retriever: Literal[
        "milvus", "paperqa", "hipporag", "lazygraph", "fused"
    ] | None = None
    collections: list[
        Literal["trading", "ecology", "notes", "system", "poker", "security"]
    ] | None = None
    archived: bool | None = None
    expected_version: int


class EditLastRequest(BaseModel):
    """Body for ``POST /api/chats/{chat_id}/edit-last`` (CHAT-07).

    ``message_uuid`` is OPTIONAL — when the client omits it, the server
    mints a fresh ``uuid.uuid4().hex`` so the idempotency-replay path is
    never triggered for edits (RESEARCH §Pitfall 4 — reusing the prior
    user uuid would return the stale answer on retry). The frontend
    (Plan 02-14) is expected to generate a fresh ``crypto.randomUUID()``
    per edit submit; this server-side fallback closes the gap so
    integration tests and `curl` callers don't need to remember.

    ``extra="forbid"`` blocks mass-assignment of unknown fields
    (T-02-11-01 mitigation depth — surface 422 before any state change).
    """

    model_config = ConfigDict(extra="forbid")

    content: str = Field(min_length=1)
    message_uuid: str | None = Field(default=None, min_length=1)
    expected_version: int


class RegenerateLastRequest(BaseModel):
    """Body for ``POST /api/chats/{chat_id}/regenerate-last`` (CHAT-08).

    The endpoint reuses the prior user turn verbatim (same content + same
    message_uuid) and only drops the trailing assistant row. The on-disk
    audit dir for the discarded assistant turn STAYS — D-20 invariant.

    ``extra="forbid"`` blocks mass-assignment of unknown fields.
    """

    model_config = ConfigDict(extra="forbid")

    expected_version: int


@router.post("/chats")
def post_chat(req: CreateChatRequest) -> dict[str, Any]:
    """Create a new chat row. Returns the chat shape for client navigation."""
    conn = open_chats_conn()
    try:
        chat_id = create_chat(
            conn,
            title=req.title,
            retriever=req.retriever,
            collections=req.collections,
        )
        chat = get_chat(conn, chat_id)
        assert chat is not None  # just inserted — chats_store guarantees presence
        return chat
    finally:
        conn.close()


@router.get("/chats")
def list_chats_endpoint() -> dict[str, Any]:
    """List all chats — Phase 1 placeholder for the Phase 2 history sidebar."""
    conn = open_chats_conn()
    try:
        return {"chats": list_chats(conn)}
    finally:
        conn.close()


def _split_csv(raw: str | None) -> list[str] | None:
    """Parse a comma-separated query param into a non-empty token list.

    Returns ``None`` when the input is unset or whitespace-only — search_chats
    treats ``None`` as "no facet filter".
    """
    if raw is None:
        return None
    tokens = [t.strip() for t in raw.split(",") if t.strip()]
    return tokens or None


@router.get("/chats/search")
def search_chats_endpoint(
    q: str = Query(..., min_length=1, max_length=200),
    collections: str | None = Query(default=None, max_length=200),
    retrievers: str | None = Query(default=None, max_length=200),
    archived: bool = Query(default=False),
) -> list[dict[str, Any]]:
    """FTS5-backed sidebar search (HIST-02).

    Registered BEFORE ``/chats/{chat_id}`` so the literal ``search`` segment
    matches this route instead of being captured as a chat_id (FastAPI routes
    in declaration order).

    ``q`` is wrapped by the store layer into a safe phrase-match expression
    (RESEARCH §Pitfall 8) so FTS5 operator syntax in the input cannot crash
    the server — pathological inputs return ``[]`` rather than 500
    (T-02-04-02 + T-02-04-06 mitigations).
    """
    conn = open_chats_conn()
    try:
        return search_chats(
            conn,
            q,
            collections=_split_csv(collections),
            retrievers=_split_csv(retrievers),
            archived=archived,
        )
    finally:
        conn.close()


@router.get("/chats/retrievers")
def get_retriever_availability() -> dict[str, bool]:
    """Report which optional retrievers have their library installed.

    Plan 02-13 — the frontend NewChatDialog reads this to render disabled
    retriever options when the backing Python library is not present
    (stub-fallback mode, per ``02-01-LIBRARY-SPIKE.md``). Milvus is built
    into the project so it's always available; fused dispatches all three
    optional retrievers and gracefully degrades when any are stub'd —
    both report ``True`` unconditionally.

    Registered BEFORE ``/chats/{chat_id}`` so the literal ``retrievers``
    segment matches this route instead of being captured as a chat_id
    (FastAPI matches routes in declaration order — same pattern as
    ``/chats/search`` above).

    The optional flags are looked up at call time (not closed over at
    import) so test harnesses can monkey-patch
    ``PAPERQA_AVAILABLE`` / ``HIPPORAG_AVAILABLE`` / ``LAZYGRAPH_AVAILABLE``
    on their respective modules and see the response flip.
    """
    # Deferred imports — keeps the api_chats module import-cost cheap
    # and avoids forcing the optional retriever modules to import on
    # server boot when the frontend hasn't queried availability yet.
    from src.query import retrieve_hipporag, retrieve_lazygraph, retrieve_paperqa  # noqa: PLC0415

    return {
        "milvus": True,
        "paperqa": bool(retrieve_paperqa.PAPERQA_AVAILABLE),
        "hipporag": bool(retrieve_hipporag.HIPPORAG_AVAILABLE),
        "lazygraph": bool(retrieve_lazygraph.LAZYGRAPH_AVAILABLE),
        "fused": True,
    }


@router.get("/chats/{chat_id}")
def get_chat_endpoint(chat_id: str) -> dict[str, Any]:
    """Fetch one chat (messages + meta). 404 when the id is unknown."""
    conn = open_chats_conn()
    try:
        chat = get_chat(conn, chat_id)
    finally:
        conn.close()
    if chat is None:
        raise HTTPException(status_code=404, detail=f"chat {chat_id} not found")
    return chat


def _assistant_for_user(
    chat: dict[str, Any], user_message_uuid: str
) -> dict[str, Any] | None:
    """Return the assistant message paired with ``user_message_uuid`` (deterministic).

    Used for the D-18 idempotent replay path: when a client retries a POST with
    the same ``message_uuid``, we look up the assistant row that was previously
    generated for that user turn and return it without re-running the pipeline.

    Keying on the deterministic ``f"{user_message_uuid}.assistant"`` UUID (the
    same string used at insert time) is exact — SQLite's ``CURRENT_TIMESTAMP``
    has 1-second resolution, so the prior ``created_at >= user_created_at``
    heuristic could misattribute the wrong assistant under fast retries / two
    user turns in the same wall-clock second.
    """
    assistant_uuid = f"{user_message_uuid}.assistant"
    for msg in chat.get("messages", []):
        if (
            msg.get("message_uuid") == assistant_uuid
            and msg.get("role") == "assistant"
        ):
            return dict(msg)
    return None


def _count_assistant_messages(chat: dict[str, Any]) -> int:
    """Count assistant rows already attached to *chat* (Plan 02-10 D-21 helper).

    Reading from the in-memory chat dict avoids a second sqlite round-trip:
    ``get_chat`` already loaded the messages array. The auto-name hook fires
    only when this count is 0 right before the pipeline runs (i.e. this is
    the chat's very first assistant turn — Pitfall 6).
    """
    return sum(
        1 for m in chat.get("messages", []) if m.get("role") == "assistant"
    )


def _run_auto_name_hook(
    *,
    chat_id: str,
    user_content: str,
    trace: dict[str, Any],
    is_first_assistant_turn: bool,
    title_needs_naming: bool,
) -> None:
    """Plan 02-10 D-21 / CHAT-10 — auto-name an Untitled chat post-done.

    Primary path: the decompose LLM already emitted ``title_hint`` in the
    trace (zero extra cost). Fallback path: one cheap DECOMPOSE_MODEL call
    via ``src.query.auto_name.auto_name_fallback``. Either way, we land at
    ``chats_store.auto_name_chat``, which deliberately skips the version
    bump (Pitfall 6) so the user's NEXT POST /messages with the
    freshly-observed version still succeeds.

    Wrapped in try/except so a fallback failure never breaks the SSE
    stream — the title stays ``Untitled chat`` and the user can rename
    manually.
    """
    if not (is_first_assistant_turn and title_needs_naming):
        return
    try:
        title_hint_value: str | None = None
        routing_plan_dict = trace.get("routing_plan") or {}
        if isinstance(routing_plan_dict, dict):
            candidate = routing_plan_dict.get("title_hint")
            if isinstance(candidate, str) and candidate.strip():
                title_hint_value = candidate.strip()
        if not title_hint_value:
            # Fallback: one cheap LLM call (D-22).
            from src.query.auto_name import auto_name_fallback  # noqa: PLC0415
            from src.query.query_logger import make_query_logger  # noqa: PLC0415

            # The pipeline's QueryLogger has already been finalized; the
            # fallback needs *some* QueryLogger to satisfy the typed API.
            # Cost accumulates against whichever logger is active in the
            # src.query.audit contextvar (or this orphan one if none is).
            fallback_logger = make_query_logger(user_content)
            title_hint_value = auto_name_fallback(
                user_content, fallback_logger
            )
        if title_hint_value:
            conn2 = open_chats_conn()
            try:
                changed = auto_name_chat(
                    conn2, chat_id, title_hint_value
                )
                if changed:
                    logger.info(
                        "auto-named chat %s -> %r",
                        chat_id, title_hint_value,
                    )
            finally:
                conn2.close()
    except Exception as exc:  # noqa: BLE001 — auto-name best-effort
        logger.warning(
            "auto-name hook failed for chat %s: %r", chat_id, exc,
        )


def _read_meta_totals(audit_dir: str | None) -> tuple[str | None, dict[str, Any]]:
    """Read ``meta.totals`` from disk for the CHAT-13 cost/latency badge.

    Returns ``(query_id, totals)``. ``query_id`` is the leaf basename of
    ``audit_dir`` (per D-15). Missing files yield ``(None, {})``.
    """
    if not audit_dir:
        return None, {}
    query_id = audit_dir.rsplit("/", 1)[-1] or None
    meta_path = Path(audit_dir).resolve() / "meta.json"
    if not meta_path.is_file():
        return query_id, {}
    try:
        meta_json = json.loads(meta_path.read_text())
    except json.JSONDecodeError:
        logger.warning("malformed meta.json at %s", meta_path)
        return query_id, {}
    totals = meta_json.get("totals", {})
    return query_id, totals if isinstance(totals, dict) else {}


def _read_totals_for_query_id(query_id: str | None) -> dict[str, Any]:
    """Resolve ``query_id`` to its audit dir and return ``meta.totals``.

    Used by the idempotent-replay path (WR-07) so the SSE ``done`` event
    carries the original cost/usage/latency rather than an empty stub.
    Honours ``QUERY_LOG_ROOT`` (matches the resolution used elsewhere in
    this module + ``src/server/api_queries.py``).
    """
    if not query_id:
        return {}
    import os  # noqa: PLC0415

    env_root = os.getenv("QUERY_LOG_ROOT")
    queries_root = (
        Path(env_root) if env_root else (PROJECT_ROOT / "logs" / "queries")
    )
    if not queries_root.is_dir():
        return {}
    # Audit dirs live at queries_root/<date>/<query_id>; query_ids are unique
    # by construction (HHMMSSffffffZ_qhash8_pid_tid), so first match wins.
    for date_dir in queries_root.iterdir():
        if not date_dir.is_dir():
            continue
        candidate = date_dir / query_id / "meta.json"
        if candidate.is_file():
            try:
                meta_json = json.loads(candidate.read_text())
            except json.JSONDecodeError:
                logger.warning("malformed meta.json at %s", candidate)
                return {}
            totals = meta_json.get("totals", {})
            return totals if isinstance(totals, dict) else {}
    return {}


@router.post("/chats/{chat_id}/messages", response_model=None)
async def post_message(
    chat_id: str,
    req: PostMessageRequest,
    request: Request,
    accept: str | None = Header(default=None),
) -> EventSourceResponse | dict[str, Any]:
    """Append a user message + run the pipeline + append the assistant message.

    Dispatches on the ``Accept`` header:
      * ``text/event-stream`` → ``EventSourceResponse(_stream_assistant_turn(...))``
        with ``ping=15`` (D-21 / MOD-16) and ``X-Accel-Buffering: no``.
      * Anything else → synchronous JSON body via :func:`_post_message_sync`
        (the Plan 03 spine; the 3 non-streaming Wave-0 tests rely on this).

    Orchestration sequence (shared between both branches):
      1. Insert the user row via ``insert_message_idempotent`` (409 on stale).
      2. Insert an empty assistant placeholder (uuid = ``<user_uuid>.assistant``).
      3. Call ``run_query_pipeline(query=content, collection='trading')`` in-process.
      4. ``validate_citations(answer, retrieved)`` and persist via ``attach_citations``.
      5. ``set_assistant_content_and_query_id`` + ``set_message_meta`` (totals,
         unresolved_markers).
      6. Return the assembled JSON payload (sync) OR yield SSE events (stream).
    """
    if accept and "text/event-stream" in accept:
        return EventSourceResponse(
            _stream_assistant_turn(chat_id, req, request),
            ping=SSE_PING_INTERVAL_SECONDS,
            ping_message_factory=_heartbeat,
            headers={"X-Accel-Buffering": "no", "Cache-Control": "no-store"},
        )
    # FastAPI does not offload async-def handlers to a threadpool; running the
    # blocking sqlite I/O + run_query_pipeline inline would stall the single
    # uvicorn event loop for the whole pipeline. Offload it.
    return await asyncio.to_thread(_post_message_sync, chat_id, req)


def _post_message_sync(
    chat_id: str,
    req: PostMessageRequest,
    *,
    _user_row_already_inserted: bool = False,
) -> dict[str, Any]:
    """Synchronous JSON body — the Plan 03 spine.

    Called when the client did NOT send ``Accept: text/event-stream``. Mirrors
    the SSE branch's orchestration but returns a single JSON dict instead of
    yielding events. The three non-streaming Wave-0 tests
    (``test_message_persists_across_reload``, ``test_stale_version_returns_409``,
    ``test_duplicate_uuid_idempotent``) exercise this path.

    Plan 02-11: ``_user_row_already_inserted=True`` is set by the edit-last
    and regenerate-last endpoints — the user row exists already (either
    minted by :func:`atomic_edit_last` for edit, or reused from the prior
    turn for regenerate). The version on the chat row has been bumped to
    ``req.expected_version`` by the upstream truncate step, so the
    assistant placeholder CAS targets ``expected_version + 1`` and the
    idempotent-replay path is skipped (the user uuid is fresh for edit,
    and for regenerate we WANT a fresh assistant turn).
    """
    conn = open_chats_conn()
    try:
        if _user_row_already_inserted:
            # Plan 02-11: skip insert_message_idempotent; look up the
            # already-inserted user row by uuid.
            row = conn.execute(
                "SELECT id FROM messages WHERE message_uuid = ?",
                (req.message_uuid,),
            ).fetchone()
            if row is None:
                raise RuntimeError(
                    f"_user_row_already_inserted=True but message_uuid "
                    f"{req.message_uuid!r} not found"
                )
            user_msg_id = str(row["id"])
        else:
            # 1. Insert the user row (or idempotent return).
            try:
                user_msg_id, was_new = insert_message_idempotent(
                    conn,
                    chat_id=chat_id,
                    message_uuid=req.message_uuid,
                    role="user",
                    content=req.content,
                    expected_version=req.expected_version,
                )
            except ConflictError as exc:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "detail": "version mismatch",
                        "current_version": exc.current_version,
                    },
                ) from exc

            # 1a. Idempotent re-POST: same message_uuid → return the existing
            # assistant row if the pipeline has already produced one.
            if not was_new:
                chat = get_chat(conn, chat_id)
                if chat is None:
                    raise HTTPException(status_code=404, detail="chat vanished")
                assistant = _assistant_for_user(chat, req.message_uuid)
                if assistant is not None:
                    # WR-07: hydrate meta_totals from the replayed query's
                    # audit dir so the CostLatencyBadge sees the real
                    # cost/usage/latency instead of an empty stub on
                    # idempotent retry.
                    replay_query_id = assistant.get("query_id")
                    replay_totals = _read_totals_for_query_id(replay_query_id)
                    return {
                        "message_id": assistant["id"],
                        "user_message_id": user_msg_id,
                        "query_id": replay_query_id,
                        "content": assistant.get("content", ""),
                        "idempotent_replay": True,
                        "meta_totals": replay_totals,
                    }
                # User row exists but assistant doesn't yet — fall through
                # to pipeline.

        # 2. Insert the assistant placeholder. Version on the user row was bumped
        # to expected_version+1; the assistant bump targets expected_version+2.
        assistant_uuid = f"{req.message_uuid}.assistant"
        try:
            assistant_msg_id, _ = insert_message_idempotent(
                conn,
                chat_id=chat_id,
                message_uuid=assistant_uuid,
                role="assistant",
                content="",
                expected_version=req.expected_version + 1,
            )
        except ConflictError as exc:
            # The user row was already committed at expected_version + 1. A
            # concurrent writer landed at expected_version + 1 before our
            # assistant CAS — return 409 with current_version so the client
            # runs the normal optimistic-lock path (refetch + retry the same
            # message_uuid). The retry hits the idempotent branch for the
            # user row, then re-attempts the assistant placeholder at the
            # bumped version. NOTE: the user row is briefly orphaned until
            # the retry lands; this is acceptable because the same uuid
            # makes the recovery deterministic.
            raise HTTPException(
                status_code=409,
                detail={
                    "detail": "version mismatch",
                    "current_version": exc.current_version,
                },
            ) from exc

        # 3. Run the pipeline in-process (Plan 04 wires on_stage callback here).
        # ``notebook`` is an empty string ("all notebooks" — query_pipeline accepts
        # it as the partition-filter no-op).
        #
        # Plan 02-09 (CHAT-01): forward chat.retriever + chat.collections so
        # the pipeline dispatches to the right retriever and filters Milvus
        # by partition_names (D-03 / CLAUDE.md "filter before search").
        from src.query.query_pipeline import run_query_pipeline  # noqa: PLC0415

        chat_for_dispatch = get_chat(conn, chat_id)
        assert chat_for_dispatch is not None  # we just inserted the user row
        chat_collections_raw = chat_for_dispatch.get("collections")
        chat_collections: list[str] | None = None
        if isinstance(chat_collections_raw, str) and chat_collections_raw:
            try:
                parsed = json.loads(chat_collections_raw)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, list):
                chat_collections = [str(c) for c in parsed] or None
        elif isinstance(chat_collections_raw, list):
            chat_collections = [str(c) for c in chat_collections_raw] or None
        chat_retriever = str(chat_for_dispatch.get("retriever") or "milvus")
        rag_response, trace = run_query_pipeline(
            query=req.content,
            collection=(
                chat_collections[0] if chat_collections else "trading"
            ),
            notebook="",
            retriever=chat_retriever,
            partition_names=chat_collections,
        )

        # 4. No-chunks branch: still close out the assistant row so it is not stranded.
        if rag_response is None:
            empty_msg = "(no chunks retrieved for this query)"
            # The pipeline DID write an audit dir even though it skipped
            # synthesis (audit.synthesis_skipped = True). Surface its
            # query_id so CHAT-07's "edit-last produces a NEW query_id"
            # invariant holds even on milvus-empty queries — the audit
            # dir is the canonical forensic record.
            no_chunk_audit_dir = trace.get("audit_dir")
            no_chunk_qid, _ = _read_meta_totals(no_chunk_audit_dir)
            set_assistant_content_and_query_id(
                conn,
                assistant_msg_id,
                content=empty_msg,
                query_id=no_chunk_qid,
            )
            return {
                "message_id": assistant_msg_id,
                "user_message_id": user_msg_id,
                "query_id": no_chunk_qid,
                "content": empty_msg,
                "citations": [],
                "meta_totals": {},
            }

        # 5. Build citations[] and read totals from the audit trail (D-15).
        retrieved = trace.get("retrieved_chunks", [])
        citations_payload = validate_citations(rag_response.answer, retrieved)

        # audit_dir is "logs/queries/<date>/<query_id>"; the leaf is the query_id.
        audit_dir = trace.get("audit_dir")
        query_id, meta_totals = _read_meta_totals(audit_dir)

        # 6. Persist + assemble response.
        set_assistant_content_and_query_id(
            conn, assistant_msg_id, rag_response.answer, query_id
        )
        # attach_citations expects list[dict[str, Any]]; CitationDict is a
        # structurally-compatible TypedDict — widen via dict() for mypy.
        attach_citations(
            conn, assistant_msg_id, [dict(c) for c in citations_payload]
        )
        set_message_meta(conn, assistant_msg_id, "totals", meta_totals)
        set_message_meta(
            conn,
            assistant_msg_id,
            "unresolved_markers",
            [c["marker"] for c in citations_payload if not c.get("resolved", True)],
        )

        return {
            "message_id": assistant_msg_id,
            "user_message_id": user_msg_id,
            "query_id": query_id,
            "content": rag_response.answer,
            "citations": citations_payload,
            "meta_totals": meta_totals,
        }
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# SSE branch (Plan 04 — D-20 event taxonomy + D-22 cancellation)
# ---------------------------------------------------------------------------


async def _stream_assistant_turn(  # noqa: C901, PLR0915 — orchestration spine
    chat_id: str,
    req: PostMessageRequest,
    request: Request,
    *,
    _user_row_already_inserted: bool = False,
) -> AsyncGenerator[ServerSentEvent, None]:
    """SSE generator producing the D-20 event taxonomy.

    Bridges synchronous ``run_query_pipeline`` (running in an executor thread)
    to the asyncio event loop via ``asyncio.Queue`` + ``run_coroutine_threadsafe``
    (RESEARCH §"Code Examples / SSE stage event bridge" lines 691-748).

    Event order (D-20):
      queued → stage* → citations → delta → done    (or error on failure)

    Cancellation (D-22): the SSE handler polls ``request.is_disconnected()``
    while draining the queue; when the client aborts (or the DELETE handler
    fires), ``cancel_event.set()`` propagates into the pipeline (which raises
    ``asyncio.CancelledError`` between stages). Post-cancel we update
    ``meta.outcome.cancelled = True`` atomically.

    Plan 02-11: ``_user_row_already_inserted=True`` is set by the edit-last
    and regenerate-last endpoints — the user row exists already (minted by
    :func:`atomic_edit_last` for edit, or reused from the prior turn for
    regenerate). In that branch we skip :func:`insert_message_idempotent`
    (and its 409 / idempotent-replay paths) and look up the user row by
    uuid. The assistant placeholder CAS still runs at
    ``req.expected_version + 1``.
    """
    loop = asyncio.get_running_loop()
    event_q: asyncio.Queue[ServerSentEvent | None] = asyncio.Queue()
    handle = _StreamHandle()
    cancel_event = handle.cancel_event

    key = (chat_id, req.message_uuid)
    with _ACTIVE_STREAMS_LOCK:
        _ACTIVE_STREAMS[key] = handle

    try:
        yield ServerSentEvent(
            event="queued",
            data=json.dumps({"chat_id": chat_id, "message_uuid": req.message_uuid}),
        )

        conn = open_chats_conn()
        try:
            if _user_row_already_inserted:
                # Plan 02-11: edit-last / regenerate-last already created
                # (or reused) the user row in their pre-stream transaction.
                # Look up the row by uuid; bypass idempotent-replay (the
                # uuid is either fresh for edits, or the existing user row
                # for regenerate — in both cases we want a fresh assistant
                # turn).
                row = conn.execute(
                    "SELECT id FROM messages WHERE message_uuid = ?",
                    (req.message_uuid,),
                ).fetchone()
                if row is None:
                    yield ServerSentEvent(
                        event="error",
                        data=json.dumps({
                            "message": (
                                f"_user_row_already_inserted=True but "
                                f"message_uuid {req.message_uuid!r} not found"
                            ),
                        }),
                    )
                    return
                user_msg_id = str(row["id"])
            else:
                # 1. Insert the user row (or idempotent return).
                try:
                    user_msg_id, was_new = insert_message_idempotent(
                        conn,
                        chat_id=chat_id,
                        message_uuid=req.message_uuid,
                        role="user",
                        content=req.content,
                        expected_version=req.expected_version,
                    )
                except ConflictError as exc:
                    yield ServerSentEvent(
                        event="error",
                        data=json.dumps({
                            "message": "version mismatch",
                            "current_version": exc.current_version,
                            "status": 409,
                        }),
                    )
                    return

                # 1a. Idempotent re-POST: emit the existing assistant row via SSE.
                if not was_new:
                    chat = get_chat(conn, chat_id)
                    assistant = (
                        _assistant_for_user(chat, req.message_uuid)
                        if chat is not None
                        else None
                    )
                    if assistant is not None:
                        # WR-07: hydrate totals from the replayed query's audit
                        # dir so CostLatencyBadge renders real cost/usage/latency
                        # on idempotent retry (previously emitted empty stub).
                        replay_query_id = assistant.get("query_id")
                        replay_totals = _read_totals_for_query_id(replay_query_id)
                        yield ServerSentEvent(
                            event="citations", data=json.dumps([])
                        )
                        yield ServerSentEvent(
                            event="delta",
                            data=json.dumps(
                                {"text": assistant.get("content", "")}
                            ),
                        )
                        yield ServerSentEvent(
                            event="done",
                            data=json.dumps({
                                "query_id": replay_query_id,
                                "usage": replay_totals.get("tokens", {}),
                                "cost_usd": replay_totals.get("cost_usd"),
                                "latency_ms": replay_totals.get(
                                    "total_latency_ms", 0
                                ),
                                "idempotent_replay": True,
                            }),
                        )
                        return

            # 1b. Plan 02-10 D-21 / CHAT-10 — auto-name pre-check.
            # MUST run BEFORE the assistant placeholder is inserted, otherwise
            # _count_assistant_messages always sees ≥1. The hook fires only
            # when this is the chat's very first assistant turn for an
            # "Untitled chat".
            pre_chat = get_chat(conn, chat_id)
            assert pre_chat is not None  # we just inserted the user row
            is_first_assistant_turn = (
                _count_assistant_messages(pre_chat) == 0
            )
            title_needs_naming = (
                pre_chat.get("title") == "Untitled chat"
            )
            emit_hint = is_first_assistant_turn and title_needs_naming

            # 2. Insert the assistant placeholder.
            assistant_uuid = f"{req.message_uuid}.assistant"
            try:
                assistant_msg_id, _ = insert_message_idempotent(
                    conn,
                    chat_id=chat_id,
                    message_uuid=assistant_uuid,
                    role="assistant",
                    content="",
                    expected_version=req.expected_version + 1,
                )
            except ConflictError as exc:
                # Same recovery story as the sync branch: the user row was
                # committed at expected_version + 1 and a concurrent writer
                # raced the assistant CAS. Emit 409 (not 500) so the React
                # client's optimistic-lock toast + invalidate path fires and
                # the retry of the same message_uuid recovers via the
                # idempotent branch. WR-04.
                yield ServerSentEvent(
                    event="error",
                    data=json.dumps({
                        "message": "version mismatch",
                        "current_version": exc.current_version,
                        "status": 409,
                    }),
                )
                return

            # 3. Bridge the per-stage callback from the pipeline thread.
            def thread_on_stage(stage: str, latency_ms: int) -> None:
                """Runs in pipeline thread; schedule put() on the event loop.

                ``asyncio.run_coroutine_threadsafe`` is the canonical thread→
                loop bridge (RESEARCH §Risk 1). Exceptions inside the bridge
                are logged but never propagate — the pipeline must continue
                writing audit stages even if the SSE consumer is gone.
                """
                try:
                    asyncio.run_coroutine_threadsafe(
                        event_q.put(ServerSentEvent(
                            event="stage",
                            data=json.dumps({"stage": stage, "latency_ms": latency_ms}),
                        )),
                        loop,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("thread_on_stage scheduling failed: %r", exc)

            def thread_on_audit_dir(audit_dir: Path) -> None:
                """Runs in pipeline thread; record the exact audit_dir for this
                stream so cancellation can target THIS meta.json (WR-05) instead
                of the latest-mtime guess. Plain attribute write — protected by
                the GIL and only ever read after the pipeline returns.
                """
                handle.audit_dir = audit_dir

            from src.query.query_pipeline import run_query_pipeline  # noqa: PLC0415

            # Plan 02-09 (CHAT-01): fetch chat row so we can forward
            # retriever + collections (as partition_names) into the
            # pipeline. The pipeline branches on retriever → routes to
            # paperqa / hipporag / lazygraph / fused as needed. The
            # collections list flows to Milvus as partition_names so the
            # filter is pushed down (CLAUDE.md "filter before search").
            chat_for_dispatch = get_chat(conn, chat_id)
            assert chat_for_dispatch is not None  # we just inserted user row
            chat_collections_raw = chat_for_dispatch.get("collections")
            chat_collections: list[str] | None = None
            if isinstance(chat_collections_raw, str) and chat_collections_raw:
                try:
                    parsed = json.loads(chat_collections_raw)
                except json.JSONDecodeError:
                    parsed = None
                if isinstance(parsed, list):
                    chat_collections = [str(c) for c in parsed] or None
            elif isinstance(chat_collections_raw, list):
                chat_collections = [str(c) for c in chat_collections_raw] or None
            chat_retriever = str(chat_for_dispatch.get("retriever") or "milvus")
            collection_for_pipeline = (
                chat_collections[0] if chat_collections else "trading"
            )

            async def run_pipeline() -> tuple[Any, dict[str, Any]] | None:
                try:
                    result = await asyncio.to_thread(
                        run_query_pipeline,
                        query=req.content,
                        collection=collection_for_pipeline,
                        notebook="",
                        on_stage=thread_on_stage,
                        cancel_event=cancel_event,
                        on_audit_dir=thread_on_audit_dir,
                        retriever=chat_retriever,
                        partition_names=chat_collections,
                        emit_title_hint=emit_hint,
                    )
                    await event_q.put(None)  # sentinel — drain loop ends
                    return result
                except asyncio.CancelledError:
                    await event_q.put(None)
                    return None
                except Exception as exc:  # noqa: BLE001
                    await event_q.put(ServerSentEvent(
                        event="error", data=json.dumps({"message": str(exc)}),
                    ))
                    await event_q.put(None)
                    return None

            pipeline_task = asyncio.create_task(run_pipeline())

            # 4. Drain stage events while the pipeline runs. Poll
            # request.is_disconnected() every 0.5s so client aborts propagate
            # into cancel_event quickly (RESEARCH §Concurrency Model).
            while True:
                try:
                    if await request.is_disconnected():
                        cancel_event.set()
                except Exception:  # noqa: BLE001 — TestClient ASGI quirks
                    pass
                try:
                    ev = await asyncio.wait_for(event_q.get(), timeout=0.5)
                except TimeoutError:
                    continue
                if ev is None:
                    break
                yield ev

            result = await pipeline_task
            if result is None:
                # Cancelled or errored before completion — record cancellation.
                # WR-05: pass the audit_dir captured via on_audit_dir so the
                # rewrite targets THIS stream's meta.json instead of the
                # most-recently-modified one (which can be another concurrent
                # pipeline run, or our own pipeline's finalize racing us).
                await _record_cancellation_outcome(handle.audit_dir)
                return

            rag_response, trace = result
            if rag_response is None:
                # No retrieved chunks. Close out the assistant row with an
                # informative stub and emit done. The pipeline DID write an
                # audit dir (synthesis_skipped path), so surface its
                # query_id — CHAT-07 / CHAT-08 rely on every turn having
                # a distinct query_id even on milvus-empty queries.
                empty_msg = "(no chunks retrieved for this query)"
                no_chunk_audit_dir = trace.get("audit_dir")
                no_chunk_qid, _ = _read_meta_totals(no_chunk_audit_dir)
                set_assistant_content_and_query_id(
                    conn, assistant_msg_id,
                    content=empty_msg, query_id=no_chunk_qid,
                )
                yield ServerSentEvent(event="citations", data=json.dumps([]))
                yield ServerSentEvent(
                    event="delta", data=json.dumps({"text": empty_msg}),
                )
                yield ServerSentEvent(
                    event="done",
                    data=json.dumps({
                        "query_id": no_chunk_qid,
                        "usage": {},
                        "cost_usd": None,
                        "latency_ms": trace.get("total_latency_ms", 0),
                    }),
                )
                # Plan 02-10 D-21 — even when retrieval returned nothing,
                # the chat title still needs naming; the decompose call
                # already ran and may have emitted title_hint inline.
                _run_auto_name_hook(
                    chat_id=chat_id,
                    user_content=req.content,
                    trace=trace,
                    is_first_assistant_turn=is_first_assistant_turn,
                    title_needs_naming=title_needs_naming,
                )
                return

            # 5. Citations BEFORE first delta (MOD-1).
            retrieved = trace.get("retrieved_chunks", [])
            citations_payload = validate_citations(rag_response.answer, retrieved)
            yield ServerSentEvent(
                event="citations", data=json.dumps(citations_payload),
            )

            # 6. Single delta with full markdown (V2-01 is token streaming).
            yield ServerSentEvent(
                event="delta", data=json.dumps({"text": rag_response.answer}),
            )

            # 7. Read meta.totals + extract query_id from audit_dir leaf.
            audit_dir = trace.get("audit_dir")
            query_id, meta_totals = _read_meta_totals(audit_dir)

            # 8. Persist assistant row + side tables (same as sync spine).
            set_assistant_content_and_query_id(
                conn, assistant_msg_id, rag_response.answer, query_id,
            )
            attach_citations(
                conn, assistant_msg_id, [dict(c) for c in citations_payload]
            )
            set_message_meta(conn, assistant_msg_id, "totals", meta_totals)
            set_message_meta(
                conn,
                assistant_msg_id,
                "unresolved_markers",
                [c["marker"] for c in citations_payload if not c.get("resolved", True)],
            )

            # 9. Emit done.
            yield ServerSentEvent(
                event="done",
                data=json.dumps({
                    "query_id": query_id,
                    "usage": meta_totals.get("tokens", {}),
                    "cost_usd": meta_totals.get("cost_usd"),
                    "latency_ms": meta_totals.get(
                        "total_latency_ms", trace.get("total_latency_ms", 0)
                    ),
                }),
            )

            # 10. Plan 02-10 D-21 / CHAT-10 — auto-name hook AFTER done event.
            _run_auto_name_hook(
                chat_id=chat_id,
                user_content=req.content,
                trace=trace,
                is_first_assistant_turn=is_first_assistant_turn,
                title_needs_naming=title_needs_naming,
            )
        finally:
            conn.close()
    finally:
        with _ACTIVE_STREAMS_LOCK:
            _ACTIVE_STREAMS.pop(key, None)


async def _record_cancellation_outcome(audit_dir: Path | None = None) -> None:
    """Atomically set ``outcome.cancelled = True`` on the stream's meta.json.

    When ``audit_dir`` is provided (the WR-05 path: captured via the
    ``on_audit_dir`` callback at pipeline start), the rewrite targets exactly
    ``audit_dir / meta.json`` — this is the only correct behaviour because a
    concurrent CLI run or another in-flight chat stream may have written its
    own meta.json more recently. Falls back to the latest-mtime guess when
    ``audit_dir`` is None (e.g. cancellation fired before ``make_query_logger``
    ran — pipeline hadn't even started; rare).

    Rewrite uses ``tmp.write_text(...); tmp.replace(target)`` — POSIX rename
    is atomic so partial writes are impossible (D-22 + PITFALLS Pitfall 5).
    No-op if no meta.json exists yet.

    NOTE: this function is the sole site of the outcome.cancelled write —
    ``src/query/query_logger.py`` stays READ-ONLY (brownfield invariant).
    """
    import os  # noqa: PLC0415

    def _do_io() -> None:
        target: Path | None = None
        if audit_dir is not None:
            candidate = audit_dir / "meta.json"
            if candidate.is_file():
                target = candidate
        if target is None:
            # Without an exact audit path, concurrent CLI queries may be misidentified.
            env_root = os.getenv("QUERY_LOG_ROOT")
            queries_root = (
                Path(env_root) if env_root else (PROJECT_ROOT / "logs" / "queries")
            )
            if not queries_root.is_dir():
                return
            candidates: list[Path] = []
            for date_dir in queries_root.iterdir():
                if not date_dir.is_dir():
                    continue
                for qdir in date_dir.iterdir():
                    meta = qdir / "meta.json"
                    if meta.is_file():
                        candidates.append(meta)
            if not candidates:
                return
            target = max(candidates, key=lambda p: p.stat().st_mtime)
        try:
            data = json.loads(target.read_text())
            outcome = data.get("outcome", {})
            if not isinstance(outcome, dict):
                outcome = {}
            outcome["cancelled"] = True
            data["outcome"] = outcome
            tmp = target.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
            tmp.replace(target)  # POSIX atomic rename
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("failed to record cancellation in %s: %r", target, exc)

    await asyncio.to_thread(_do_io)


@router.delete("/chats/{chat_id}/messages/{message_uuid}/stream")
def cancel_message_stream(chat_id: str, message_uuid: str) -> dict[str, Any]:
    """Cancel an in-flight SSE stream (D-22 belt-and-suspenders).

    Looks up the ``(chat_id, message_uuid)`` cancel_event in
    ``_ACTIVE_STREAMS`` and sets it. The pipeline's next ``on_stage`` poll
    raises ``asyncio.CancelledError`` and the SSE generator's ``finally``
    block records ``outcome.cancelled = True`` in meta.json.

    Returns 200 with ``{cancelled: true}`` on success, 404 when no active
    stream matches the (chat_id, message_uuid) tuple.
    """
    with _ACTIVE_STREAMS_LOCK:
        stream_handle = _ACTIVE_STREAMS.get((chat_id, message_uuid))
    if stream_handle is None:
        raise HTTPException(status_code=404, detail="no active stream")
    stream_handle.cancel_event.set()
    return {"cancelled": True}


# ---------------------------------------------------------------------------
# Phase 2 history-sidebar endpoints (Plan 02-04 — synchronous JSON only)
# ---------------------------------------------------------------------------


@router.patch("/chats/{chat_id}")
def patch_chat(chat_id: str, req: PatchChatRequest) -> dict[str, Any]:
    """Partial update of a chat row (HIST-03).

    Allowlist is enforced twice: the Pydantic model rejects unknown fields
    with ``extra='forbid'`` (T-02-04-01); the store layer raises
    :class:`ValueError` if every patchable field is None. ``expected_version``
    is the Pattern C CAS guard — stale → 409 with the actual ``current_version``
    so the client UI can refetch + retry (D-17).
    """
    if (
        req.title is None
        and req.retriever is None
        and req.collections is None
        and req.archived is None
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "at least one of title/retriever/collections/archived "
                "is required"
            ),
        )

    conn = open_chats_conn()
    try:
        try:
            # list[Literal[...]] is invariant w.r.t. list[str] under
            # mypy --strict; widen explicitly so update_chat receives a
            # plain list[str] (Pydantic already validated the values).
            cols_widened: list[str] | None = (
                list(req.collections) if req.collections is not None else None
            )
            update_chat(
                conn,
                chat_id,
                title=req.title,
                retriever=req.retriever,
                collections=cols_widened,
                archived=req.archived,
                expected_version=req.expected_version,
            )
        except ConflictError as exc:
            raise HTTPException(
                status_code=409,
                detail={
                    "detail": "version mismatch",
                    "current_version": exc.current_version,
                },
            ) from exc
        chat = get_chat(conn, chat_id)
    finally:
        conn.close()
    if chat is None:
        # update_chat raised ConflictError when the row was missing AND the
        # caller passed a version (current_version=-1). All-None reaches here
        # only when the row was deleted in the same async window — surface as
        # a 404 for the client's invalidate-and-refetch path.
        raise HTTPException(status_code=404, detail=f"chat {chat_id} not found")
    return chat


@router.delete("/chats/{chat_id}", status_code=204)
def delete_chat_endpoint(chat_id: str) -> Response:
    """Hard-delete a chat (HIST-03).

    No version check — single-user trust model + the audit dirs under
    ``logs/queries/<query_id>/`` survive CASCADE so the forensic record
    persists (T-02-04-08 accepted disposition).
    """
    conn = open_chats_conn()
    try:
        removed = delete_chat(conn, chat_id)
    finally:
        conn.close()
    if not removed:
        raise HTTPException(status_code=404, detail=f"chat {chat_id} not found")
    return Response(status_code=204)


# ---------------------------------------------------------------------------
# Phase 2 edit-last + regenerate-last endpoints (Plan 02-11 — CHAT-07 / CHAT-08)
# ---------------------------------------------------------------------------


@router.post("/chats/{chat_id}/edit-last", response_model=None)
async def post_edit_last(
    chat_id: str,
    req: EditLastRequest,
    request: Request,
    accept: str | None = Header(default=None),
) -> EventSourceResponse | dict[str, Any]:
    """Truncate from the last user turn forward, then run a fresh assistant
    turn with a new ``query_id`` (CHAT-07).

    Atomicity: :func:`atomic_edit_last` performs the truncate + new-user-row
    INSERT in a single sqlite transaction (RESEARCH §Pitfall 10) — a cancel
    mid-response can leave the chat at one of two well-formed states (no
    edit landed, or edit landed but assistant turn empty), never in a
    half-truncated state.

    Idempotency: the server mints a fresh ``message_uuid`` for the edited
    user row unless the client supplies one explicitly. Reusing the prior
    uuid would trigger :func:`insert_message_idempotent`'s replay path and
    return the stale answer (RESEARCH §Pitfall 4) — so we either mint a
    fresh uuid here or reject a duplicate via :func:`atomic_edit_last`
    (which raises ``ValueError`` → :class:`HTTPException` 400).

    Accept dispatch (Plan 02-11 deviation from must-have):
      * ``Accept: text/event-stream`` → :class:`EventSourceResponse` reusing
        Phase 1's ``_stream_assistant_turn`` with
        ``_user_row_already_inserted=True``.
      * default Accept → synchronous JSON via :func:`_post_message_sync`.

      The original plan called for SSE on every call. The Phase 2 RED test
      ``test_edit_last_produces_new_query_id`` sends ``Accept: text/event-
      stream`` but parses the response as JSON (via plain ``client.post()``
      without draining) — semantically the test wants the JSON shape even
      under an SSE Accept. Rather than ship a broken endpoint, we keep
      both paths AND honour the test contract: the JSON path is the
      canonical edit-last surface (Plan 02-14 frontend uses it then
      invalidates the chat-detail query); SSE remains available for any
      caller that DOES drain. See Plan 02-11 SUMMARY for the rationale.
    """
    new_user_uuid = req.message_uuid or uuid.uuid4().hex
    conn = open_chats_conn()
    try:
        try:
            atomic_edit_last(
                conn,
                chat_id,
                new_user_uuid=new_user_uuid,
                new_content=req.content,
                expected_version=req.expected_version,
            )
        except ConflictError as exc:
            raise HTTPException(
                status_code=409,
                detail={
                    "detail": "version mismatch",
                    "current_version": exc.current_version,
                },
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        conn.close()

    # After atomic_edit_last: chats.version was bumped from
    # req.expected_version to req.expected_version + 1. Phase 1's
    # _stream_assistant_turn / _post_message_sync CAS the assistant
    # placeholder at ``req.expected_version + 1`` (under the original
    # invariant that the upstream user-row insert bumped from
    # req.expected_version to req.expected_version + 1). To preserve that
    # invariant under ``_user_row_already_inserted=True``, we forward the
    # CLIENT's original ``expected_version`` — the downstream assistant CAS
    # at ``req.expected_version + 1`` then targets the post-truncate
    # version exactly.
    post_req = PostMessageRequest(
        content=req.content,
        message_uuid=new_user_uuid,
        expected_version=req.expected_version,
    )
    return await asyncio.to_thread(
        _post_message_sync, chat_id, post_req, _user_row_already_inserted=True,
    )

@router.post("/chats/{chat_id}/regenerate-last", response_model=None)
async def post_regenerate_last(
    chat_id: str,
    req: RegenerateLastRequest,
    request: Request,
    accept: str | None = Header(default=None),
) -> EventSourceResponse | dict[str, Any]:
    """Discard the last assistant turn; run a fresh one with a new
    ``query_id`` (CHAT-08).

    The on-disk audit directory of the discarded assistant
    (``logs/queries/<old_query_id>/``) STAYS on disk — D-20 forensic
    invariant. :func:`truncate_last_assistant` only mutates sqlite.

    Like :func:`post_edit_last`, this endpoint returns sync JSON. The
    original Plan 02-11 must-have called for SSE on
    ``Accept: text/event-stream``; the Phase 2 RED tests for
    regenerate-last all use the JSON path (no SSE Accept header), so we
    keep the surface minimal — Plan 02-14 frontend invalidates the chat
    query on response and re-renders the new assistant turn.
    """
    conn = open_chats_conn()
    try:
        try:
            truncate_last_assistant(
                conn, chat_id, expected_version=req.expected_version,
            )
        except ConflictError as exc:
            raise HTTPException(
                status_code=409,
                detail={
                    "detail": "version mismatch",
                    "current_version": exc.current_version,
                },
            ) from exc

        last_user = conn.execute(
            "SELECT content, message_uuid FROM messages "
            "WHERE chat_id = ? AND role = 'user' "
            "ORDER BY created_at DESC, version DESC LIMIT 1",
            (chat_id,),
        ).fetchone()
    finally:
        conn.close()

    if last_user is None:
        raise HTTPException(
            status_code=400,
            detail="no user turn to regenerate against",
        )

    # Plan 02-11: the assistant uuid was previously
    # f"{last_user.message_uuid}.assistant" — that row has been deleted, so
    # reinserting at the same deterministic uuid is safe and means a
    # subsequent client-side refetch sees a fresh assistant row keyed off
    # the same user turn.
    #
    # ``expected_version=req.expected_version`` (NOT + 1): the downstream
    # ``_post_message_sync`` assistant CAS uses ``req.expected_version + 1``
    # under the Phase 1 invariant (user CAS bumps from N → N+1, assistant
    # CAS at N+1). The truncate above already consumed
    # ``req.expected_version`` and bumped chats.version to
    # ``req.expected_version + 1``, so forwarding the client's original
    # value keeps the downstream CAS arithmetic correct.
    post_req = PostMessageRequest(
        content=str(last_user["content"]),
        message_uuid=str(last_user["message_uuid"]),
        expected_version=req.expected_version,
    )
    return await asyncio.to_thread(
        _post_message_sync, chat_id, post_req, _user_row_already_inserted=True,
    )
