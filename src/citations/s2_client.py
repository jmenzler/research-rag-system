"""On-demand S2 Graph API abstract back-fill for niche hits.

Local ``abstracts.sqlite`` covers only the OA subset; the per-paper Graph API
has looser licensing and broader coverage (``tldr`` fallback when abstract is
null). Opt-in slow path — network-bound, never in the warm query envelope. PDF
download is out of scope here.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from typing import Any

import httpx

_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
# Graph API batch accepts up to 500 ids per request; serialise at <=1 req/sec.
_MAX_IDS = 500
_RATE_SLEEP_S = 1.0

Poster = Callable[[list[str], str | None], list[dict[str, Any] | None]]


def _http_post(ids: list[str], api_key: str | None) -> list[dict[str, Any] | None]:
    headers = {"x-api-key": api_key} if api_key else {}
    resp = httpx.post(
        _BATCH_URL,
        params={"fields": "abstract,tldr"},
        json={"ids": ids},
        headers=headers,
        timeout=30.0,
    )
    resp.raise_for_status()
    return list(resp.json())


def fetch_abstracts(
    corpus_ids: list[int],
    *,
    api_key: str | None = None,
    poster: Poster = _http_post,
) -> dict[int, str]:
    """Back-fill abstracts from the S2 Graph API, keyed by corpusid.

    Falls back to ``tldr.text`` when ``abstract`` is null. ``poster`` is injected
    so tests can run without a network round-trip.
    """
    ids = list({int(c) for c in corpus_ids})
    if not ids:
        return {}
    key = api_key if api_key is not None else os.environ.get("SEMANTIC_SCHOLAR_API_KEY")

    out: dict[int, str] = {}
    for start in range(0, len(ids), _MAX_IDS):
        chunk = ids[start : start + _MAX_IDS]
        if start:
            time.sleep(_RATE_SLEEP_S)
        records = poster([f"CorpusId:{cid}" for cid in chunk], key)
        # The batch endpoint preserves input order and emits null for misses.
        for cid, rec in zip(chunk, records, strict=True):
            if not rec:
                continue
            text = rec.get("abstract")
            if not text:
                tldr = rec.get("tldr")
                text = tldr.get("text") if isinstance(tldr, dict) else None
            if text:
                out[cid] = text
    return out
