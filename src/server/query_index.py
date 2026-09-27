"""In-memory index over logs/queries/<date>/<id>/meta.json.

Read-only — never writes to the audit trail. TTL cache (30s). Retriever and
chat_id are NOT filled here; the endpoint layer sources them via the messages
join in chats.db.
"""

from __future__ import annotations

import base64
import json
import logging
import time
from typing import Any, TypedDict

logger = logging.getLogger(__name__)

# -- Row type -----------------------------------------------------------------


class QueryRow(TypedDict):
    """One query-log row. All keys exist; values may be None."""

    query_id: str
    query: str | None
    ts_started: str | None
    model: str | None
    tokens: int | None
    cost_usd: float | None
    latency_ms: int | None
    outcome: str
    retriever: str | None
    chat_id: str | None


# -- TTL cache ----------------------------------------------------------------

QUERY_INDEX_TTL_S: int = 30
_cache: list[QueryRow] | None = None
_cache_ts: float = 0.0

# -- Outcome ------------------------------------------------------------------

_OUTCOME_ANSWERED = "answered"
_OUTCOME_SKIPPED = "skipped"
_OUTCOME_CRAG_RETRY = "crag retry"
_OUTCOME_FAILED = "failed"


def _derive_outcome(outcome: dict[str, Any] | None) -> str:
    """Map meta.json outcome object to a facet string."""
    if outcome is None:
        return _OUTCOME_FAILED
    if outcome.get("crag_retried"):
        return _OUTCOME_CRAG_RETRY
    if outcome.get("synthesis_skipped"):
        return _OUTCOME_SKIPPED
    return _OUTCOME_ANSWERED


# -- Build --------------------------------------------------------------------


def build_query_index() -> list[QueryRow]:
    """Walk _queries_root() date dirs, parse meta.json, return rows sorted by ts_started DESC."""
    from src.server.api_queries import _queries_root  # lazy: break circular import

    root = _queries_root()
    rows: list[QueryRow] = []
    if not root.is_dir():
        return rows
    for date_dir in sorted(root.iterdir(), reverse=True):
        if not date_dir.is_dir():
            continue
        for query_dir in sorted(date_dir.iterdir(), reverse=True):
            if not query_dir.is_dir():
                continue
            query_id = query_dir.name
            meta_path = query_dir / "meta.json"
            if not meta_path.is_file():
                logger.warning("query_index: skipping %s (no meta.json)", query_dir)
                continue
            try:
                meta: dict[str, Any] = json.loads(meta_path.read_text())
            except json.JSONDecodeError:
                logger.warning("query_index: skipping %s (malformed meta.json)", query_dir)
                continue
            if not isinstance(meta, dict):
                logger.warning("query_index: skipping %s (meta.json not a dict)", query_dir)
                continue
            row = _parse_meta(query_id, meta)
            rows.append(row)
    rows.sort(key=lambda r: r.get("ts_started") or "", reverse=True)
    return rows


def _parse_meta(query_id: str, meta: dict[str, Any]) -> QueryRow:
    """Extract QueryRow fields from a meta.json dict.

    Nested fields: config.gen_model, totals.tokens.total, totals.cost_usd,
    totals.total_latency_ms. No retriever at top level.
    """
    config: dict[str, Any] = meta.get("config") or {}
    totals: dict[str, Any] = meta.get("totals") or {}
    tokens: dict[str, Any] = totals.get("tokens") or {}
    return QueryRow(
        query_id=query_id,
        query=meta.get("query"),
        ts_started=meta.get("ts_started"),
        model=config.get("gen_model"),
        tokens=tokens.get("total"),
        cost_usd=totals.get("cost_usd"),
        latency_ms=totals.get("total_latency_ms"),
        outcome=_derive_outcome(meta.get("outcome")),
        retriever=None,
        chat_id=None,
    )


# -- TTL cache ----------------------------------------------------------------


def get_query_index() -> list[QueryRow]:
    """TTL-cached build_query_index. Rebuilds every QUERY_INDEX_TTL_S seconds."""
    global _cache, _cache_ts  # noqa: PLW0603
    now = time.monotonic()
    if _cache is not None and (now - _cache_ts) < QUERY_INDEX_TTL_S:
        return _cache
    try:
        _cache = build_query_index()
        _cache_ts = now
    except Exception:
        logger.exception("query_index: build failed")
        if _cache is not None:
            return _cache
        return []
    return _cache


def invalidate_cache() -> None:
    """Force next get_query_index to rebuild."""
    global _cache  # noqa: PLW0603
    _cache = None


# -- Facets -------------------------------------------------------------------


def apply_query_facets(
    rows: list[QueryRow],
    retrievers: list[str] | None = None,
    outcomes: list[str] | None = None,
    cost_min: float | None = None,
    latency_min: int | None = None,
) -> list[QueryRow]:
    """Filter rows by retriever/outcome/cost_min/latency_min (ANDed)."""
    result = rows
    if retrievers:
        r_set = frozenset(retrievers)
        result = [r for r in result if r.get("retriever") in r_set]
    if outcomes:
        o_set = frozenset(outcomes)
        result = [r for r in result if r.get("outcome") in o_set]
    if cost_min is not None:
        result = [r for r in result if (r.get("cost_usd") or 0) >= cost_min]
    if latency_min is not None:
        result = [r for r in result if (r.get("latency_ms") or 0) >= latency_min]
    return result


# -- Cursor pagination --------------------------------------------------------

_SORT_DEFAULTS: dict[str, str] = {
    "ts": "ts_started",
    "cost": "cost_usd",
    "latencyMs": "latency_ms",
}

# Sort values are heterogeneous (str, float, int, None) plus tuple comparison
# in apply_cursor requires Any typing.
_AnySort = Any  # noqa: ANN401


def _get_sort_value(row: QueryRow, sort: str) -> _AnySort:
    """Look up the sort-column value for *row*."""
    col = _SORT_DEFAULTS.get(sort, "ts_started")
    return row.get(col)


def encode_cursor(sort_value: _AnySort, query_id: str) -> str:
    """Base64url-encode (sort_value, query_id) for opaque pagination."""
    payload = json.dumps([sort_value, query_id], separators=(",", ":"))
    return base64.urlsafe_b64encode(payload.encode()).decode().rstrip("=")


def decode_cursor(cursor: str) -> tuple[_AnySort, str]:  # noqa: ANN401
    """Decode a base64url cursor."""
    padded = cursor
    remainder = len(cursor) % 4
    if remainder:
        padded += "=" * (4 - remainder)
    raw = base64.urlsafe_b64decode(padded.encode()).decode()
    parts = json.loads(raw)
    return parts[0], str(parts[1])


def apply_cursor(
    rows: list[QueryRow],
    cursor: str | None,
    limit: int,
    sort: str = "ts",
    direction: str = "desc",
) -> tuple[list[QueryRow], str | None, int]:
    """Slice rows by cursor, clamp limit <= 200."""
    limit = min(limit, 200)
    total = len(rows)
    start = 0
    if cursor is not None:
        try:
            cursor_sort_val, cursor_qid = decode_cursor(cursor)
            for i, row in enumerate(rows):
                val = _get_sort_value(row, sort)
                qid: str = row["query_id"]
                if direction == "desc":
                    if (val, qid) < (cursor_sort_val, cursor_qid):
                        start = i
                        break
                else:
                    if (val, qid) > (cursor_sort_val, cursor_qid):
                        start = i
                        break
            else:
                start = 0
        except (ValueError, IndexError, KeyError):
            start = 0
    page = rows[start : start + limit]
    next_cursor: str | None = None
    if len(page) == limit and (start + limit) < total:
        last = page[-1]
        val = _get_sort_value(last, sort)
        next_cursor = encode_cursor(val, last["query_id"])
    return page, next_cursor, total
