# long-ok-file
"""Free-PDF discovery + cross-run URL cache for ``--retry-failed``.

The Spider doesn't read the URL cache itself — the cache is purely a
bookkeeping aid for ``--retry-failed`` to remember which URLs already
produced a usable artifact in a prior run. ``seed_url_cache_from_existing``
(idempotent walk over ``sources/*/meta.json``) populates it from disk;
``evict_url_cache_entries`` drops entries for URLs we want to re-fetch.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path
from urllib.parse import urljoin

# ---------------------------------------------------------------------------
# Free-PDF discovery from HTML
# ---------------------------------------------------------------------------

# Require arXiv context (arxiv.org/abs|pdf|html/ or an "arXiv:" label) before
# the NNNN.NNNNN[N] id. A bare token anywhere in body text is NOT enough — an
# incidental number would otherwise trigger a download of an unrelated paper.
_ARXIV_ID_RE = re.compile(
    r"(?:arxiv\.org/(?:abs|pdf|html)/|arxiv:\s*)([0-9]{4}\.[0-9]{4,6})(?:v\d+)?",
    re.I,
)
_PDF_LINK_RE = re.compile(r"""<a[^>]*href=["']([^"']*?\.pdf[^"']*)["']""", re.I)


def discover_free_pdf(html: str, base_url: str) -> str | None:
    """Find a likely free-PDF link in *html*.

    1. arXiv id in an arXiv-contextual reference — arxiv copies are always public.
    2. First in-page ``<a href="*.pdf">`` — the host's own copy.
    Returns the absolute URL or ``None``.
    """
    m = _ARXIV_ID_RE.search(html)
    if m:
        return f"https://arxiv.org/pdf/{m.group(1)}"
    pdf_match = _PDF_LINK_RE.search(html)
    if pdf_match:
        return str(urljoin(base_url, pdf_match.group(1)))
    return None


# ---------------------------------------------------------------------------
# Cross-run URL cache
# ---------------------------------------------------------------------------

URL_CACHE_PATH = Path("logs/_url_cache.json")


def _load_url_cache() -> dict[str, dict[str, str]]:
    if not URL_CACHE_PATH.exists():
        return {}
    try:
        return json.loads(URL_CACHE_PATH.read_text())  # type: ignore[no-any-return]
    except (json.JSONDecodeError, OSError):
        return {}


def _save_url_cache(cache: dict[str, dict[str, str]]) -> None:
    URL_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = URL_CACHE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache, indent=2))
    tmp.replace(URL_CACHE_PATH)


def seed_url_cache_from_existing(roots: list[Path] | None = None) -> int:
    """Walk ``sources/*/<NNN>__*/meta.json`` and pre-populate the URL cache from
    files already on disk. Idempotent — only adds new entries.

    Returns the number of cache entries added.
    """
    if roots is None:
        sources_dir = Path("sources")
        if not sources_dir.is_dir():
            return 0
        roots = [p for p in sources_dir.iterdir() if p.is_dir()]
    cache = _load_url_cache()
    added = 0
    for root in roots:
        for meta_p in root.glob("*/meta.json"):
            try:
                meta = json.loads(meta_p.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            url = meta.get("url")
            if not url:
                continue
            d = meta_p.parent
            for fname, kind in (("source.pdf", "pdf"), ("web.txt", "web"), ("pdf.txt", "pdf_txt")):
                f = d / fname
                if not f.exists():
                    continue
                if cache.get(url, {}).get(kind):
                    continue
                cache.setdefault(url, {})[kind] = str(f.resolve())
                added += 1
    if added:
        _save_url_cache(cache)
    return added


def evict_url_cache_entries(urls: Iterable[str]) -> int:
    """Remove cached entries for *urls* so the next fetch writes fresh files.

    Used by ``--retry-failed`` to drop stale entries before the spider runs.

    Returns the number of cache entries removed (0 if cache is missing or
    none of the URLs are present).
    """
    if not URL_CACHE_PATH.exists():
        return 0
    cache = _load_url_cache()
    removed = 0
    for url in urls:
        if cache.pop(url, None) is not None:
            removed += 1
    if removed:
        _save_url_cache(cache)
    return removed
