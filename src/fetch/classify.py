# long-ok-file
"""Source classification and URL utilities for notebook source tiers.

Extracted from ``scripts/fetch_sources.py``.
"""

from __future__ import annotations

import json
import re
import subprocess
from typing import Any
from urllib.parse import parse_qs, urlparse

# ---------------------------------------------------------------------------
# Regex constants
# ---------------------------------------------------------------------------

ARXIV_RE = re.compile(
    r"arxiv\.org/(?:abs|pdf|html)/([0-9]{4}\.[0-9]{4,6})(?:v\d+)?(?:\.pdf)?", re.I
)

# Recovers URL from title field when src["url"] is empty (SourceType.UNKNOWN).
URL_IN_TITLE_RE = re.compile(r"(https?://\S+)")

SLUG_RE = re.compile(r"[^a-z0-9]+")

# ---------------------------------------------------------------------------
# Tier constants
# ---------------------------------------------------------------------------

PAPER_HOSTS = (
    "sciencedirect.com", "springer.com", "wiley.com", "tandfonline.com",
    "nature.com", "ieee.org", "acm.org", "jstor.org", "oup.com",
    "cambridge.org", "worldscientific.com",
)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def slug(s: str) -> str:
    out = SLUG_RE.sub("_", s.lower()).strip("_")
    return out[:60] or "untitled"


def classify(src: dict[str, Any]) -> str:
    t = src["type"]
    if t in ("SourceType.PASTED_TEXT", "SourceType.MARKDOWN"):
        return "T6_nlm_native"
    url = _resolve_source_url(src).lower()
    host = urlparse(url).netloc
    path = urlparse(url).path

    if "arxiv.org" in host:
        return "T0_arxiv"
    if "researchgate.net" in host:
        return "T4_researchgate"
    if "ssrn.com" in host:
        return "T2_ssrn"
    if "doi.org" in host:
        return "T1_doi"
    if path.endswith(".pdf"):
        return "T3_generic_pdf"
    # If NLM marks it as PDF AND we have a public URL, treat it as a generic
    # PDF so the download path runs. Earlier order put T7 first which made us
    # skip the PDF download for any /download or ?type=printable URL whose
    # path didn't literally end in .pdf.
    if t == "SourceType.PDF":
        if url:
            return "T3_generic_pdf"
        return "T7_nlm_uploaded_pdf"
    if any(h in host for h in PAPER_HOSTS):
        return "T1_doi"
    return "T5_web_blog"


def list_sources(notebook_id: str) -> list[dict[str, Any]]:
    result = subprocess.run(
        ["notebooklm", "source", "list", "--notebook", notebook_id, "--json"],
        capture_output=True, text=True, check=True, timeout=120,
    )
    raw = json.loads(result.stdout)
    return raw["sources"] if isinstance(raw, dict) else raw  # type: ignore[no-any-return]


def resolve_pdf_url(src: dict[str, Any]) -> str | None:
    url = src.get("url") or ""
    m = ARXIV_RE.search(url)
    if m:
        return f"https://arxiv.org/pdf/{m.group(1)}.pdf"
    if url.lower().endswith(".pdf"):
        return url
    return url or None


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _resolve_source_url(src: dict[str, Any]) -> str:
    """Return the best URL for *src*, recovering from title for UNKNOWN sources.

    Returns empty string when no URL can be resolved (pasted text, NLM uploads).
    """
    url = (src.get("url") or "").strip()
    if not url and src.get("type") == "SourceType.UNKNOWN":
        m = URL_IN_TITLE_RE.search(src.get("title") or "")
        if m:
            url = m.group(1)
    return url


def canonical_url(src: dict[str, Any]) -> str:
    """Normalize a source URL into a dedup key.

    Returns "" for genuinely URL-less sources (pasted text, NLM-uploaded PDFs)
    that should never participate in dedup.
    """
    url = _resolve_source_url(src)
    if not url:
        return ""

    parsed = urlparse(url)
    scheme = parsed.scheme.lower() or "https"
    netloc = parsed.netloc.lower()
    path = parsed.path

    # arxiv → normalized to arxiv.org/abs/<id> (no scheme, no version, no .pdf extension)
    m = ARXIV_RE.search(f"{netloc}{path}")
    if m:
        return f"arxiv.org/abs/{m.group(1)}"

    # Drop fragment, strip trailing /
    path = path.rstrip("/") or "/"

    # ssrn → normalize abstract_id query param
    if "ssrn.com" in netloc:
        qs = parse_qs(parsed.query)
        if "abstract_id" in qs:
            return f"https://ssrn.com/abstract={qs['abstract_id'][0]}"
        # No abstract_id → just drop query entirely
        return f"{scheme}://{netloc}{path}"

    # researchgate → keep /publication/<digits>_<slug> only
    if "researchgate.net" in netloc:
        m = re.match(r"(/publication/\d+_[^/]+)", path)
        if m:
            path = m.group(1).lower()
        return f"{scheme}://{netloc}{path}"

    # doi.org → lowercase the DOI suffix (the path)
    if "doi.org" in netloc:
        path = path.lower()
        return f"{scheme}://{netloc}{path}"

    # Default: scheme + netloc + path, no query, no fragment, no trailing /
    return f"{scheme}://{netloc}{path}"


# ---------------------------------------------------------------------------
# Page metadata extraction
# ---------------------------------------------------------------------------

# Regex for <meta name="citation_*"> tags. Captures name and content attributes.
_META_CITATION_RE = re.compile(
    r'<meta\s+name="(citation_[^"]+)"\s+content="([^"]*)"', re.I
)
# Regex for <script type="application/ld+json"> blocks.
_LD_JSON_RE = re.compile(
    r'<script\s+type="application/ld\+json"[^>]*>(.*?)</script>', re.I | re.S
)


def _harvest_page_metadata(html: str) -> dict[str, Any]:
    """Extract citation metadata from HTML meta tags and JSON-LD.

    Returns a dict with keys ``doi``, ``title``, ``authors`` (list), ``published``.
    JSON-LD ``ScholarlyArticle`` blocks override ``<meta>`` tags on conflict.
    Returns empty dict when no metadata is found.
    """
    result: dict[str, Any] = {}
    # Map citation_* meta names to result keys
    _key_map = {
        "citation_doi": "doi",
        "citation_title": "title",
        "citation_publication_date": "published",
    }

    # --- Pass 1: <meta name="citation_*"> ---
    authors: list[str] = []
    for m in _META_CITATION_RE.finditer(html):
        name = m.group(1).lower()
        content = m.group(2).strip()
        if not content:
            continue
        if name == "citation_author":
            authors.append(content)
        elif name in _key_map:
            result[_key_map[name]] = content
    if authors:
        result["authors"] = authors

    # --- Pass 2: JSON-LD ScholarlyArticle ---
    for m in _LD_JSON_RE.finditer(html):
        raw = m.group(1).strip()
        if not raw:
            continue
        try:
            ld = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(ld, dict) or ld.get("@type") != "ScholarlyArticle":
            continue
        # JSON-LD author: can be a single string, dict with "name", or list of either
        ld_authors = ld.get("author")
        if ld_authors is not None:
            result["authors"] = _normalize_jsonld_authors(ld_authors)
        if ld.get("name"):
            result["title"] = ld["name"]
        if ld.get("datePublished"):
            result["published"] = ld["datePublished"]

    return result


def _normalize_jsonld_authors(author_val: object) -> list[str]:
    """Normalize JSON-LD author field to list of name strings."""
    if isinstance(author_val, str):
        return [author_val]
    if isinstance(author_val, dict):
        name = author_val.get("name")
        return [name] if isinstance(name, str) else []
    if isinstance(author_val, list):
        names: list[str] = []
        for a in author_val:
            if isinstance(a, str):
                names.append(a)
            elif isinstance(a, dict):
                n = a.get("name")
                if isinstance(n, str):
                    names.append(n)
        return names
    return []
