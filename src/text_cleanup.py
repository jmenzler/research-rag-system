"""Strip navigation / UI / footer cruft from NotebookLM-extracted web page text.

NLM's web extraction includes a lot of low-signal boilerplate:
- "Skip to main content"
- bare URLs on their own lines
- "Subscribe", "Download", "Buy Crypto", "Privacy Policy", "Terms of Service"
- repeated nav menus

These tokens add noise to BM25 (high-frequency, low-IDF) and dilute embedding
quality. Strip them deterministically before ingest.

Usage:

    from src.text_cleanup import clean_nlm_text
    cleaned = clean_nlm_text(raw_nlm_text)
"""
from __future__ import annotations

import re

# Lines (post-strip) matching these phrases are dropped wholesale.
NAV_PHRASES: tuple[str, ...] = (
    "skip to main content",
    "skip navigation",
    "skip to content",
    "subscribe",
    "sign in",
    "log in",
    "sign up",
    "log out",
    "privacy policy",
    "terms of service",
    "terms of use",
    "cookie policy",
    "cookies preferences",
    "menu",
    "share",
    "tweet",
    "facebook",
    "linkedin",
    "copy link",
    "buy crypto",
    "free trial",
    "download",
    "get started",
    "back to top",
    "view all",
    "see all",
    "load more",
    "show more",
    "show less",
    "read more",
    "click here",
    # Wikipedia / mediawiki edit links
    "bearbeiten",
    "quelltext bearbeiten",
    "bearbeiten | quelltext bearbeiten",
    "edit",
    "edit source",
    "view source",
    "talk",
    # Generic article chrome
    "published:",
    "last updated:",
    "reading time:",
    "min read",
    "written by",
    "posted on",
    "originally published",
    "originally posted",
    "this article was",
    "disclaimer:",
    "disclaimer",
    "all rights reserved",
    "copyright ©",
)

# Lines that match these regexes are dropped.
# Wikipedia-style edit links fused to paragraph text — strip the prefix
# instead of dropping the whole line.
_WIKI_EDIT_RE = re.compile(
    r"^\[(Bearbeiten|edit|Editar)(\s*\|\s*(Quelltext|source|fuente)\s*(bearbeiten|edit|editar))?\]",
    re.I,
)

NAV_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^https?://\S+$"),                      # bare URL on its own
    re.compile(r"^\W{0,4}$"),                           # punctuation-only
    re.compile(r"^[•‣◦⁃\-\*]\s*$"),  # lone bullet
    re.compile(r"^[☰-⛿✀-➿←-⇿]+$"),  # symbol-only
)


def _line_is_nav(line: str) -> bool:
    s = line.strip().lower()
    if not s:
        return False  # blank lines preserved (paragraph separators)
    if len(s) <= 60 and s in NAV_PHRASES:
        return True
    if any(p.match(line.strip()) for p in NAV_PATTERNS):
        return True
    # Short lines that contain a nav phrase as the bulk
    if len(s) <= 40:
        for ph in NAV_PHRASES:
            if ph in s:
                return True
    return False


def _collapse_blanks(text: str) -> str:
    """Collapse 3+ consecutive blank lines into 2 (one paragraph break)."""
    return re.sub(r"\n{3,}", "\n\n", text)


# Patterns that mark the start of trailing footer/CTA cruft on article pages.
# Case-insensitive, matched as start-of-line. The earliest match wins — anything
# from that line to end-of-document is dropped.
TAIL_CRUFT_PATTERNS: tuple[str, ...] = (
    r"^related\s+articles?\b",
    r"^related\s+reading\b",
    r"^related\s+posts?\b",
    r"^related\s+stories\b",
    r"^you\s+may\s+(also\s+)?like\b",
    r"^subscribe\s+(to|for)\b",
    r"^stay\s+up[- ]to[- ]date\b",
    r"^sign\s+up\s+(for|to)\s+(our\s+)?newsletter\b",
    r"^join\s+our\s+newsletter\b",
    r"^no\s+comments?\s+yet\b",
    r"^post\s+a\s+comment\b",
    r"^leave\s+a\s+(reply|comment)\b",
    r"^share\s+this\s+(article|post|story)\b",
    r"^about\s+the\s+author\b",
    r"^©\s*\d{4}",
    r"^all\s+rights\s+reserved\b",
)
_TAIL_RE = re.compile("|".join(TAIL_CRUFT_PATTERNS), re.I | re.M)


def tail_strip(text: str) -> str:
    """Drop trailing footer/CTA cruft from article text.

    Finds the earliest line that matches any of TAIL_CRUFT_PATTERNS and removes
    that line and everything after it. Conservative: only drops if the
    truncation leaves at least 80% of the original content intact (otherwise
    the match was probably a false positive in the article body).
    """
    if not text:
        return text
    m = _TAIL_RE.search(text)
    if not m:
        return text
    cut = text.rfind("\n", 0, m.start())
    if cut < 0:
        cut = m.start()
    if cut < int(0.8 * len(text)):
        # Truncation would drop more than 20% — likely a false positive
        return text
    return text[:cut].rstrip() + "\n"


def clean_nlm_text(text: str) -> str:
    """Strip nav cruft from NotebookLM-extracted text.

    - Drops nav-phrase lines (case-insensitive)
    - Drops bare-URL lines and punctuation-only lines
    - Collapses runs of blank lines

    Header (first line if it starts with `#`) is preserved verbatim.
    """
    if not text:
        return text

    lines = text.splitlines()
    out: list[str] = []
    for i, ln in enumerate(lines):
        # Preserve the first markdown header line as-is
        if i == 0 and ln.lstrip().startswith("#"):
            out.append(ln)
            continue
        # Strip Wikipedia edit-link prefix from line (fused to paragraph)
        ln = _WIKI_EDIT_RE.sub("", ln)
        if _line_is_nav(ln):
            continue
        out.append(ln)

    return _collapse_blanks("\n".join(out)).strip() + "\n"
