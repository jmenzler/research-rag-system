"""Fetch and parse the arXiv Atom feed for q-fin.TR and cs.GT."""
from __future__ import annotations

import logging
import re
import time
import xml.etree.ElementTree as ET
from datetime import UTC, datetime, timedelta

import httpx

from src.monitor.paper import Paper

logger = logging.getLogger(__name__)

_ARXIV_URL = "https://export.arxiv.org/api/query"
# arXiv 429/503 are transient. Retry with backoff so a single rate-limit
# doesn't silently drop a poll — papers in the window would otherwise never
# be re-seen (the next poll's window has already moved past them).
_RETRY_STATUSES = frozenset({429, 503})
_MAX_ATTEMPTS = 4
_BACKOFF_S = (3.0, 8.0, 20.0)  # arXiv asks ≥3s between requests
_ATOM_NS = "http://www.w3.org/2005/Atom"
_ARXIV_NS = "http://arxiv.org/schemas/atom"
_ID_RE = re.compile(r"/abs/([^v]+)")

# Patterns covering common LaTeX inline math/markup.
_LATEX_RE = re.compile(
    r"\\(?:emph|textbf|textit|text|cite|ref|label|eqref)\{[^}]*\}"
    r"|\\[a-zA-Z]+\{[^}]*\}"
    r"|\$[^$]*\$"
    r"|\\[a-zA-Z]+"
)


def _strip_latex(text: str) -> str:
    cleaned = _LATEX_RE.sub("", text)
    return " ".join(cleaned.split())


def _parse_arxiv_id(entry_id: str) -> str:
    m = _ID_RE.search(entry_id)
    if not m:
        raise ValueError(f"Cannot extract arxiv_id from: {entry_id!r}")
    return m.group(1)


def _parse_published(text: str) -> datetime:
    text = text.strip()
    # Handles ISO 8601 with Z suffix: "2026-05-11T00:00:00Z"
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text)


def _tag(ns: str, local: str) -> str:
    return f"{{{ns}}}{local}"


def fetch_papers(*, window_hours: int = 96) -> list[Paper]:
    """Fetch arXiv q-fin.TR + cs.GT papers published within the last *window_hours*.

    The 96h default overlaps several polls so a rate-limited or weekend day is
    recovered on the next successful run (downstream ingest de-dups already-seen
    papers). 429/503 are retried with backoff rather than silently dropped.

    Rate limit: arXiv asks for ≥ 3s between requests. This function makes one
    request per call, so callers should not call it in a tight loop.
    """
    cutoff = datetime.now(UTC) - timedelta(hours=window_hours)

    params = {
        "search_query": "cat:q-fin.TR OR cat:cs.GT",
        "sortBy": "submittedDate",
        "sortOrder": "descending",
        "max_results": "100",
    }

    response: httpx.Response | None = None
    for attempt in range(_MAX_ATTEMPTS):
        try:
            response = httpx.get(_ARXIV_URL, params=params, timeout=60.0)
            response.raise_for_status()
            break
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status in _RETRY_STATUSES and attempt < _MAX_ATTEMPTS - 1:
                wait = _BACKOFF_S[attempt]
                logger.warning(
                    "arXiv API returned %s — retry %d/%d in %.0fs",
                    status, attempt + 1, _MAX_ATTEMPTS - 1, wait,
                )
                time.sleep(wait)
                continue
            logger.warning("arXiv API returned %s — skipping this poll", status)
            return []
    if response is None:  # all attempts exhausted on retryable status
        return []

    root = ET.fromstring(response.text)
    papers: list[Paper] = []

    for entry in root.findall(_tag(_ATOM_NS, "entry")):
        try:
            raw_id = (entry.findtext(_tag(_ATOM_NS, "id")) or "").strip()
            arxiv_id = _parse_arxiv_id(raw_id)

            published_text = entry.findtext(_tag(_ATOM_NS, "published")) or ""
            published = _parse_published(published_text)

            if published < cutoff:
                continue

            title = _strip_latex(entry.findtext(_tag(_ATOM_NS, "title")) or "")
            abstract = _strip_latex(entry.findtext(_tag(_ATOM_NS, "summary")) or "")
            abstract = abstract[:4000]

            authors = [
                name.text or ""
                for author in entry.findall(_tag(_ATOM_NS, "author"))
                for name in author.findall(_tag(_ATOM_NS, "name"))
            ]

            categories = [
                cat.get("term", "")
                for cat in entry.findall(_tag(_ATOM_NS, "category"))
                if cat.get("term")
            ]

            if not title or not abstract:
                logger.warning("Skipping entry with missing title/abstract: %s", arxiv_id)
                continue

            papers.append(
                Paper(
                    id=arxiv_id,
                    title=title,
                    abstract=abstract,
                    authors=authors,
                    published=published,
                    categories=categories,
                    url=f"https://arxiv.org/abs/{arxiv_id}",
                    pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
                )
            )
        except Exception:  # noqa: BLE001
            logger.warning("Skipping malformed arXiv entry", exc_info=True)

    logger.info("fetch_papers: %d papers within last %dh", len(papers), window_hours)
    return papers
