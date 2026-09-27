"""Post a digest of newly ingested arXiv papers to a Discord webhook."""
from __future__ import annotations

import logging
import os
from datetime import datetime

import httpx

import src.config  # noqa: F401 — ensures .env is loaded before os.getenv below
from src.monitor.filter import RelevanceResult

logger = logging.getLogger(__name__)

_ENV_VAR = "DISCORD_WEBHOOK_URL"
_TIMEOUT_S = 10.0

# Warn once at import time if the env var is absent.
_webhook_url: str | None = os.getenv(_ENV_VAR)
if not _webhook_url:
    logger.warning(
        "DISCORD_WEBHOOK_URL not set — Discord digest disabled. "
        "Set %s to enable.",
        _ENV_VAR,
    )


def post_digest(papers: list[RelevanceResult], date: datetime) -> None:
    """Post a paper digest to Discord. No-op if no papers or webhook not configured.

    Failures are logged but never raised — they must not block ingestion.
    """
    if not _webhook_url:
        return
    if not papers:
        return

    date_str = date.strftime("%Y-%m-%d")
    lines = [f"**New arXiv papers — {date_str}**\n"]
    for r in papers:
        lines.append(f"• **{r.paper.title}** ({r.paper.id})")
        if r.summary:
            lines.append(f"  {r.summary}")

    content = "\n".join(lines)
    # Discord message cap is 2000 chars; truncate gracefully.
    if len(content) > 1900:
        content = content[:1897] + "…"

    try:
        resp = httpx.post(
            _webhook_url,
            json={"content": content},
            timeout=_TIMEOUT_S,
        )
        resp.raise_for_status()
        logger.info("Discord digest posted: %d papers", len(papers))
    except httpx.HTTPError as exc:
        logger.error("Discord webhook failed: %s", exc)
