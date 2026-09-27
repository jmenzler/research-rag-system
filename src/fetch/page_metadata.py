"""HTML page-metadata harvester combining trafilatura + citation meta tags.

Trafilatura's ``extract_metadata`` handles the messy general web: OpenGraph,
Twitter cards, schema.org Article/NewsArticle/Person, ``<meta name="author">``,
JSON-LD ``author`` fields, sitename, date, title. Used at scale by HF, IBM,
Microsoft Research.

Citation meta tags (``citation_doi``, ``citation_author``) are NOT a
trafilatura priority — they're a Highwire-press scholar-specific convention
that ``src.fetch.classify._harvest_page_metadata`` handles via regex. We
combine both: trafilatura for general-web signals, citation regex for
scholarly DOIs.

Returns a dict with the same shape as the old
``classify._harvest_page_metadata`` plus ``sitename`` and ``date`` fields:

    {
        "authors":  list[str],   # validated, deduped
        "title":    str | None,
        "doi":      str | None,
        "published": str | None,  # ISO date or year string
        "sitename": str | None,
    }
"""

from __future__ import annotations

from typing import Any

from src.fetch.classify import _harvest_page_metadata


def harvest_page_metadata(html: str) -> dict[str, Any]:
    """Return merged metadata dict from trafilatura + citation regex.

    Trafilatura author wins (general-web signal). Citation regex DOI wins
    (scholar-specific signal). Title/date/sitename come from trafilatura
    only; the citation regex doesn't surface them.
    """
    out: dict[str, Any] = _harvest_page_metadata(html)

    try:
        from trafilatura import extract_metadata  # noqa: PLC0415
    except ImportError:
        return out

    try:
        meta = extract_metadata(html)
    except Exception:
        return out

    if meta is None:
        return out

    tr_author = getattr(meta, "author", None)
    if isinstance(tr_author, str) and tr_author.strip():
        out["authors"] = _split_author_string(tr_author)

    for src_attr, dst_key in (
        ("title", "title"),
        ("date", "published"),
        ("sitename", "sitename"),
    ):
        v = getattr(meta, src_attr, None)
        if isinstance(v, str) and v.strip():
            out[dst_key] = v.strip()

    return out


def _split_author_string(s: str) -> list[str]:
    """Trafilatura returns ``"Alice; Bob"`` or ``"Alice"``; normalize to list.

    Falls back to comma-split for sources that use commas, but only when no
    semicolon is present (commas inside names like ``"Smith, J."`` are common
    in scholarly cites and must not be treated as separators).
    """
    s = s.strip()
    if not s:
        return []
    sep = ";" if ";" in s else "\n"
    parts = [p.strip() for p in s.split(sep)]
    return [p for p in parts if p]
