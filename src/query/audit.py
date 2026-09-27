"""Module-level audit-trail seam for Phase 2 retriever sub-stages.

Phase 2 retriever modules (``retrieve_paperqa``, ``retrieve_hipporag``,
``retrieve_lazygraph``, ``retrieve_fused``) call the per-query audit logger
via this module rather than threading a ``QueryLogger`` through every helper:

    from src.query import audit
    audit.write_stage("paperqa_rerank", payload)
    audit.accumulate_usage("paperqa_rerank", usage, cost_usd, latency_ms)

The retriever entry-points push a ``QueryLogger`` onto a ``ContextVar`` for
the duration of one query and reset on exit. The SSE handler (later phase)
owns the lifecycle when the pipeline is driven from the chat endpoint; the
standalone-retriever code path lazily creates one via
``make_query_logger`` when nothing has been installed (e.g. in tests).

Tests monkey-patch the module attributes (``audit.write_stage = spy``); the
retrievers must call via ``from src.query import audit; audit.write_stage(...)``
so the patched attribute is picked up at call time.

Two APIs are exposed so plans that landed independently both keep working:

- ``set_current_logger(logger | None)`` — simple installer (Plan 02-06)
- ``set_active_logger(value) -> Token`` + ``reset_active_logger(token)`` —
  contextvars-style scoped installer (Plan 02-07 onward)
"""

from __future__ import annotations

import logging
from contextvars import ContextVar, Token
from typing import Any

from src.query.query_logger import QueryLogger, make_query_logger

_logger = logging.getLogger(__name__)

_active_logger: ContextVar[QueryLogger | None] = ContextVar(
    "audit_active_logger", default=None
)


def _current_logger() -> QueryLogger:
    """Return the active ``QueryLogger``, creating one on first use.

    Production code paths install their own logger via
    :func:`set_current_logger` / :func:`set_active_logger` so audit stages
    land in the same per-query directory as the rest of the pipeline.
    Standalone callers (tests, smoke probes) get a fresh logger keyed off
    ``QUERY_LOG_ROOT`` via :func:`make_query_logger`.
    """
    active = _active_logger.get()
    if active is None:
        active = make_query_logger(query="(standalone-retriever)")
        _active_logger.set(active)
    return active


def set_current_logger(logger_: QueryLogger | None) -> None:
    """Install (or clear) the active ``QueryLogger``.

    Simple, non-token API used by retrievers that don't need scoped
    restoration (Plan 02-06's paperqa). Pass ``None`` to clear.
    """
    _active_logger.set(logger_)


def set_active_logger(value: QueryLogger | None) -> Token[QueryLogger | None]:
    """Install *value* as the active audit logger.

    Returns the ``ContextVar.Token`` so callers can ``reset_active_logger``
    to restore prior state — matches contextvars semantics, used by Plan
    02-07 hipporag entry-point.
    """
    return _active_logger.set(value)


def reset_active_logger(token: Token[QueryLogger | None]) -> None:
    """Reverse a prior :func:`set_active_logger` call."""
    _active_logger.reset(token)


def get_active_logger() -> QueryLogger | None:
    """Return the currently-active ``QueryLogger``, or ``None``."""
    return _active_logger.get()


def write_stage(name: str, payload: dict[str, Any]) -> None:
    """Forward to ``QueryLogger.write_stage`` on the active logger.

    Side effect: ``<NN>_<name>.json`` is written under the current query's
    log directory (prefix resolved via ``_STAGE_PREFIX``).
    """
    _current_logger().write_stage(name, payload)


def accumulate_usage(
    stage: str,
    usage: Any,  # noqa: ANN401  — forwarded verbatim to QueryLogger; shape owned there
    cost_usd: float | None,
    latency_ms: int = 0,
) -> None:
    """Forward to ``QueryLogger.accumulate_usage`` on the active logger.

    Recording usage here keeps ``meta.totals.cost_usd`` / ``n_llm_calls``
    honest for sub-stages — without this call the stage would be invisible
    in the final ``meta.json`` summary.

    Non-stage args are passed by keyword so test spies that monkey-patch
    ``QueryLogger.accumulate_usage`` with a ``(self, stage, **kw)``
    signature still receive ``usage`` / ``cost_usd`` / ``latency_ms`` cleanly.
    """
    _current_logger().accumulate_usage(
        stage,
        usage=usage,
        cost_usd=cost_usd,
        latency_ms=latency_ms,
    )
