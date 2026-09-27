"""Rule-based author/year extraction + arxiv/DOI/quality enrichment + CLI backfill.

Public surface:

- ``extract_arxiv_id`` — derive arxiv ID from a URL or bare-id string
- ``extract_doi_from_pdf_text`` — regex over PDF first-page text
- ``quality_counts`` — tally element types from ``content_list.json``
- ``_detect_parser_versions`` — capture library + git versions at fetch time
- ``enrich_meta_inline`` — inline enrichment called from the Spider
- ``enrich_meta_json`` / ``__main__`` — idempotent backfill CLI

Idempotent: overwrites only the ``metadata`` key in ``meta.json``.
"""

from __future__ import annotations

import argparse
import functools
import json
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.cleanup import detect_fail_signature

# Hosts where individual authors aren't named, but the platform itself is
# the right citation. Each host maps to (authors_attribution, short_brand):
#   authors_attribution — full form used in the authors list (APA-style)
#   short_brand         — concise brand used inside the short_cite parens
_PLATFORM_ATTRIBUTION: dict[str, tuple[str, str]] = {
    "wikipedia.org":      ("Wikipedia contributors", "Wikipedia"),
    "wikimedia.org":      ("Wikipedia contributors", "Wikipedia"),
    "reddit.com":         ("Reddit",                 "Reddit"),
    "news.ycombinator.com": ("Hacker News",          "Hacker News"),
    "stackoverflow.com":  ("Stack Overflow",         "Stack Overflow"),
    "stackexchange.com":  ("Stack Exchange",         "Stack Exchange"),
}


def _platform_match(url: str | None) -> tuple[str, str] | None:
    """Return (authors_attribution, short_brand) for known platform hosts."""
    if not url:
        return None
    from urllib.parse import urlparse  # noqa: PLC0415
    host = urlparse(url).netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    if host in _PLATFORM_ATTRIBUTION:
        return _PLATFORM_ATTRIBUTION[host]
    for suffix, pair in _PLATFORM_ATTRIBUTION.items():
        if host.endswith("." + suffix):
            return pair
    return None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Compound surname particles that should attach to the last token.
_PARTICLES: set[str] = {
    "de", "van", "von", "der", "del", "della", "di", "da", "le", "la", "den",
}

# Words that signal "not an author line" (institutional / structural).
_AFFIL_TOKENS = re.compile(
    r"\b(Department|University|School of|Institute|Laboratory|"
    r"Faculty|College|Center|Centre|Corporation|Foundation|"
    r"Abstract|ABSTRACT|Keywords|JEL|Received|Forthcoming|"
    r"Initial Version|First version|Current version|This version|"
    r"Login|Password|Sign in|Sign up|Register|Subscribe|"
    r"Cookie|Privacy|Terms of (?:Use|Service)|Copyright|"
    r"Returning user|Forgot|Username|Account)\b",
    re.IGNORECASE,
)

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
# Footnote markers attached to or following names. Catches:
#   "Smith*"           — symbol after letter
#   "Smith 1"          — digit after a space (free-floating ref)
#   "Smith1"           — digit glued to letter (MinerU-style affiliation idx)
#   "Smith,1"          — digit glued after a comma
_FOOTNOTE_RE = re.compile(
    r"[\*†‡§¶∗\\]"
    r"|(?<=[A-Za-z])\d+(?=[\s,]|$)"
    r"|\b\d+(?=[\s,]|$)"
)
_HEADING_RE = re.compile(r"^#+\s+(.+?)\s*$")

# Year patterns
_ARXIV_URL_YEAR_RE = re.compile(
    r"arxiv\.org/(?:pdf|abs|html)/(\d{2})(\d{2})\.\d+",
)
_DATE_MONTH_RE = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December|"
    r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\b"
    r"[\s,.]+\d{0,2}[,\s]*\b(19|20)\d{2}\b",
    re.IGNORECASE,
)
_BARE_YEAR_RE = re.compile(r"\b((?:19|20)\d{2})\b")
_VERSION_YEAR_RE = re.compile(
    r"(?:First version|Initial Version|This version|Current version|Received)"
    r"[^\n]{0,80}\b((?:19|20)\d{2})\b",
    re.IGNORECASE,
)

_LOGIN_SCREEN_RE = re.compile(
    r"\b(login to your account|forgot password|keep me logged in|"
    r"sign in to (?:continue|view)|create (?:a |an )?account|"
    r"reset (?:your )?password|institutional login|new user|returning user)\b",
    re.IGNORECASE,
)

# Web-title patterns mis-classified as authors when a fail-page parser
# extracts the HTML <title>. Each rule rejects a distinct shape.
_WEBTITLE_PIPE_RE = re.compile(r"\|")
_WEBTITLE_TRAILING_BRAND_RE = re.compile(r" [-—] [A-Z][a-zA-Z0-9]{2,}\s*$")
_WEBTITLE_COLON_SUBTITLE_RE = re.compile(r"^[^,]*:")

# Nav / marketing junk that trips _looks_like_author_line otherwise.
# Imperative CTAs ("Jump to content"), arrow CTAs (→ Quant Tier List →),
# emoji-prefixed marketing ("🚀 FREE Guide ..."), single-token brand strings
# ("Products", "QuantStart") — none of which is plausibly an author block.
_NAV_PHRASE_RE = re.compile(
    r"^(jump to|skip to|click here|learn more|get started|read more|sign in|"
    r"sign up|log in|log out|subscribe|contact us|about us|home|menu|search)\b",
    re.IGNORECASE,
)
_TRAILING_ARROW_RE = re.compile(r"[←-⇿➠-➿»]\s*$")

# ---------------------------------------------------------------------------
# DOI regex patterns
# ---------------------------------------------------------------------------

_DOI_DOT_ORG_RE = re.compile(
    r"doi\.org/(10\.\d{4,}/[^\s<>\"{}|\\^`\[\]]+)", re.IGNORECASE,
)
_DOI_HTTPS_RE = re.compile(
    r"https?://doi\.org/(10\.\d{4,}/[^\s<>\"{}|\\^`\[\]]+)", re.IGNORECASE,
)
_DOI_LABEL_RE = re.compile(
    r"(?:DOI|doi)\s*:\s*(10\.\d{4,}/[^\s<>\"{}|\\^`\[\]]+)",
)

# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class DocMetadata:
    """Extracted metadata for one document."""

    authors: list[str]
    year: int | None
    short_cite: str | None
    confidence: str  # "high" | "medium" | "low" | "none"
    arxiv_id: str | None = None
    doi: str | None = None
    extraction_quality: dict[str, object] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Public API: standalone helpers
# ---------------------------------------------------------------------------


_ARXIV_ID_6DIGIT_RE = re.compile(
    r"arxiv\.org/(?:abs|pdf|html)/([0-9]{4}\.[0-9]{4,6})(?:v\d+)?(?:\.pdf)?", re.I,
)
_BARE_ARXIV_RE = re.compile(
    r"([0-9]{4}\.[0-9]{4,6})(?:v\d+)?(?:\.pdf)?$", re.I,
)


def extract_arxiv_id(url: str) -> str | None:
    """Extract arxiv ID from a URL or bare ID string.

    Tries an arxiv.org URL match first (4-6 digit suffixes — covers the modern
    schema introduced in 2007 and the 2025 expansion to 6-digit suffixes),
    then falls back to a bare ID like ``2403.12345v2``.
    """
    if not url:
        return None
    m = _ARXIV_ID_6DIGIT_RE.search(url)
    if m:
        return m.group(1)
    m = _BARE_ARXIV_RE.search(url)
    if m:
        return m.group(1)
    return None


def extract_doi_from_pdf_text(pdf_text: str) -> str | None:
    """Extract a DOI from PDF first-page text via regex.

    Handles three common forms:
      - ``https://doi.org/10.XXXX/...``
      - ``doi.org/10.XXXX/...``
      - ``DOI: 10.XXXX/...``
    """
    if not pdf_text:
        return None
    # Order matters: https://doi.org first so it captures the full URL form
    for pat in (_DOI_HTTPS_RE, _DOI_DOT_ORG_RE, _DOI_LABEL_RE):
        m = pat.search(pdf_text)
        if m:
            # Clean trailing punctuation that likely isn't part of the DOI
            doi = m.group(1).rstrip(".,;:)!?]}")
            return doi
    return None


def quality_counts(content_list_path: Path) -> dict[str, object]:
    """Return ``{n_pages, n_tables, n_equations, n_figures, n_lists, has_structured}``.

    Reads ``content_list.json`` (MinerU PDF or web extractor output) and tallies
    element types. Returns zeros with ``has_structured=False`` when the file is
    missing or unreadable.
    """
    empty: dict[str, object] = {
        "n_pages": 0,
        "n_tables": 0,
        "n_equations": 0,
        "n_figures": 0,
        "n_lists": 0,
        "has_structured": False,
    }
    if not content_list_path.exists():
        return empty

    try:
        raw = content_list_path.read_text(encoding="utf-8")
        elements = json.loads(raw)
    except (json.JSONDecodeError, OSError):
        return empty

    if not isinstance(elements, list):
        return empty

    counts: dict[str, int] = {}
    max_page = -1
    for el in elements:
        if not isinstance(el, dict):
            continue
        typ = el.get("type", "")
        if isinstance(typ, str):
            counts[typ] = counts.get(typ, 0) + 1
        pi = el.get("page_idx")
        if isinstance(pi, int) and pi > max_page:
            max_page = pi

    n_pages = (max_page + 1) if max_page >= 0 else 0

    return {
        "n_pages": n_pages,
        "n_tables": counts.get("table", 0),
        "n_equations": counts.get("equation", 0),
        "n_figures": counts.get("figure", 0),
        "n_lists": counts.get("list", 0),
        "has_structured": True,
    }


# ---------------------------------------------------------------------------
# Public API: _detect_parser_versions (used by orchestrator)
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _detect_parser_versions() -> dict[str, str]:
    """Snapshot library + repo versions at fetch time. Cached per-process.

    Returns dict with keys ``mineru``, ``trafilatura``, ``git_sha``.
    Values are version strings or ``"unknown"``.
    """
    return {
        "mineru": _safe_pkg_version("mineru") or "unknown",
        "trafilatura": _safe_pkg_version("trafilatura") or "unknown",
        "git_sha": _safe_git_sha() or "unknown",
    }


def _safe_pkg_version(pkg: str) -> str | None:
    """Return installed version of *pkg*, or None if not found."""
    try:
        from importlib.metadata import version

        return version(pkg)
    except Exception:
        return None


def _safe_git_sha() -> str | None:
    """Return short git SHA of HEAD, or None on failure."""
    try:
        import subprocess

        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if r.returncode == 0:
            return r.stdout.strip()
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _read_doc_text(doc_dir: Path) -> tuple[str, str | None]:
    """Return (text, source_filename) — first ~3000 chars of the best file.

    Skips login-screen-only files (paywalled pages where nlm.txt is just the
    login form) — returns "" so the extractor reports "none" instead of
    matching "Keep me logged in" as an author.
    """
    for fname in ("pdf.txt", "nlm.txt", "web.txt"):
        f = doc_dir / fname
        if f.exists() and f.stat().st_size > 0:
            try:
                text = f.read_text(encoding="utf-8", errors="ignore")[:3000]
            except Exception:
                continue
            if detect_fail_signature(text):
                continue
            head = text[:500]
            if _LOGIN_SCREEN_RE.search(head):
                continue
            return text, fname
    return "", None


def _strip_footnotes(s: str) -> str:
    """Remove footnote markers (\\*, †, numeric superscripts) from an author token.

    Iterates to a fixpoint so chained markers like ``"Smith1∗"`` get fully
    stripped: pass 1 removes the symbol, pass 2 removes the digit that's now
    adjacent to whitespace.
    """
    for _ in range(3):
        new = _FOOTNOTE_RE.sub("", s)
        if new == s:
            break
        s = new
    return s.strip(" ,.;:-—")


# Boundary between authors and affiliation list when MinerU flattens both
# onto a single line. Triggers on numbered affiliation refs like "1The",
# "2Singapore", "3Nanyang" — the digits are the affiliation markers, not
# co-author footnote markers (those have spaces / commas around them).
_INLINE_AFFIL_BOUNDARY_RE = re.compile(
    r"\s\d{1,2}(?:The|[A-Z][a-zA-Z]+\s+"
    r"(?:University|Universit[äaeé]t|Institute?|Laboratory|College|"
    r"Faculty|Department|School|Centre|Center|Corporation|Foundation))",
)


def _split_authors_before_affiliation(line: str) -> str | None:
    """When a line packs `<authors> <numbered-affiliations> <emails>` together,
    return just the author prefix. Returns ``None`` if no boundary detected.

    Example MinerU output::

        "Junzhe Jiang1∗ , Chang Yang1∗ , Bo Li1 1The Hong Kong Polytechnic
         University, ... author@example.com"

    The boundary is the first " <digit><Capitalized word>(University|...)"
    sequence. Anything from that point onwards is affiliation/email noise.
    """
    m = _INLINE_AFFIL_BOUNDARY_RE.search(line)
    if not m:
        return None
    prefix = line[: m.start()].strip()
    if len(prefix) < 4:
        return None
    return prefix


def _starts_with_symbol(line: str) -> bool:
    """True if the first char's Unicode category is a Symbol (emoji, arrow, etc).

    Cheaper and more complete than enumerating emoji ranges by hand. Author
    lines start with a letter; symbol-prefix is a strong nav/marketing tell.
    """
    if not line:
        return False
    cat = unicodedata.category(line[0])
    return cat.startswith("S")  # So, Sm, Sk, Sc — all Symbol categories


def _looks_like_author_line(line: str) -> bool:
    """Heuristic: is this line plausibly an author block?"""
    line = line.strip()
    if len(line) < 4 or len(line) > 400:
        return False
    if _AFFIL_TOKENS.search(line):
        return False
    if _EMAIL_RE.search(line):
        return False
    # Must have at least one capitalized word
    if not re.search(r"\b[A-Z][a-zA-Z]", line):
        return False
    # Reject lines that are just numbers / dates / single short tokens
    if re.fullmatch(r"[\d\s,.\-/]+", line):
        return False
    # Web-title rejection: HTML <title> patterns mis-extracted as authors.
    if _WEBTITLE_PIPE_RE.search(line):
        return False
    if _WEBTITLE_TRAILING_BRAND_RE.search(line):
        return False
    # Long ``Title: Subtitle`` headlines — but ``Smith, J.: A Novel ...``
    # passes (comma before the colon distinguishes a citation from a headline).
    if len(line) >= 30 and _WEBTITLE_COLON_SUBTITLE_RE.match(line):
        return False
    # Nav / marketing junk
    if _NAV_PHRASE_RE.match(line):
        return False
    if _TRAILING_ARROW_RE.search(line):
        return False
    if _starts_with_symbol(line):
        return False
    # Single-token lines are almost always brand names ("Products",
    # "QuantStart", "Mergify"). Real single-name authors are vanishingly
    # rare in modern papers; reject unconditionally.
    if " " not in line:
        return False
    return True


_SENTENCE_END_RE = re.compile(r"[.!?]\s*$")
# Lowercase verbs / connectors that mark prose fragments (PDF-text-scan leak).
# Hand-curated from the audit; covers the dominant Backed.fi / disclosure /
# abstract-as-author patterns.
_PROSE_TELLS_RE = re.compile(
    r"\b(is|are|was|were|the|this|that|these|those|which|who|whose|"
    r"according|depicts|present|presents|show|shows|study|studies|"
    r"argue|argues|propose|proposes|demonstrate|demonstrates|"
    r"investigate|investigates|find|finds|note|notes|consist|consists|"
    r"include|includes|contains|contain|address|addresses|focus|focuses)\b",
    re.IGNORECASE,
)
_LATEX_TOKEN_RE = re.compile(r"[\\${}^_]|\\mathbb|\\frac|\\sum")


def _looks_like_sentence_fragment(s: str) -> bool:
    """True if *s* looks like prose / non-name content, not a personal name.

    PDF text-scan extracts the first paragraph after a title when no real
    author block exists. Real human names have ≤4 words, no sentence
    punctuation, no lowercase verbs, no LaTeX markup, no weekday/month
    tokens (conference schedules), no embedded digits (table rows / room
    numbers).
    """
    words = s.split()
    if len(words) > 4:
        return True
    if _SENTENCE_END_RE.search(s):
        return True
    if _PROSE_TELLS_RE.search(s):
        return True
    if _LATEX_TOKEN_RE.search(s):
        return True
    tokens_lower = [t.lower() for t in re.findall(r"[A-Za-z]+", s)]
    if any(t in _WEEKDAY_NAMES or t in _MONTH_NAMES for t in tokens_lower):
        return True
    # Digits inside a name field — schedule rows ("20:00-22: - Room"),
    # affiliation indices ("Author1"), table coordinates. Stripped after
    # _strip_footnotes already handles trailing digit refs.
    if re.search(r"\d", s):
        return True
    return False


def _split_authors(line: str) -> list[str]:
    """Split an author line into individual author strings."""
    # Normalize "and" → ","
    line = re.sub(r"\s+and\s+", ", ", line, flags=re.IGNORECASE)
    line = re.sub(r"\s+with\s+", ", ", line, flags=re.IGNORECASE)
    parts = re.split(r"[,;]+", line)
    cleaned = [_strip_footnotes(p) for p in parts]
    # Filter empties + obvious non-name fragments
    out: list[str] = []
    for p in cleaned:
        if not p:
            continue
        # Reject pure affiliations / numerics
        if _AFFIL_TOKENS.search(p) or _EMAIL_RE.search(p):
            continue
        if re.fullmatch(r"[\d\W]+", p):
            continue
        # Reject single-token lower-case fragments
        if " " not in p and not p[0].isupper():
            continue
        # Reject sentence fragments (PDF text-scan leak — the #1 residual bug)
        if _looks_like_sentence_fragment(p):
            continue
        out.append(p)
    return out


def _extract_lastname(author: str) -> str | None:
    """Extract the surname from a full author string.

    Handles compound surnames (de Prado, van der Schaar, López de Prado).
    """
    tokens = author.split()
    if not tokens:
        return None
    # Strip trailing initials / single letters
    while len(tokens) > 1 and (
        len(tokens[-1].rstrip(".")) == 1 and tokens[-1].rstrip(".").isupper()
    ):
        tokens.pop()
    if not tokens:
        return None
    last = tokens[-1]
    # Walk backwards as long as the previous token is a particle.
    # Also absorb the capitalized token immediately preceding a particle run
    # (so "Marcos López de Prado" → "López de Prado", not just "de Prado").
    i = len(tokens) - 2
    consumed_particle = False
    while i >= 0 and tokens[i].lower() in _PARTICLES:
        last = f"{tokens[i]} {last}"
        i -= 1
        consumed_particle = True
    if (
        consumed_particle
        and i >= 0
        and tokens[i][:1].isupper()
        and tokens[i].lower() not in _PARTICLES
    ):
        last = f"{tokens[i]} {last}"
    # Title-case if all-caps
    if last.isupper():
        last = last.title()
    return last


def _extract_authors(text: str) -> list[str]:
    """Extract authors from the first lines after the title."""
    lines = text.splitlines()
    if not lines:
        return []

    # Find the first markdown heading (title), or use the first non-empty line
    title_idx = -1
    for i, line in enumerate(lines[:30]):
        if _HEADING_RE.match(line):
            title_idx = i
            break

    # Look at the next 1-15 lines for an author block
    start = title_idx + 1 if title_idx >= 0 else 0
    end = min(start + 20, len(lines))

    candidate_lines: list[str] = []
    for line in lines[start:end]:
        s = line.strip()
        if not s:
            if candidate_lines:
                break  # blank line ends author block
            continue
        # Skip another heading
        if _HEADING_RE.match(line):
            if candidate_lines:
                break
            continue

        # MinerU sometimes flattens authors+affiliations onto one line.
        # Try to peel off the leading author segment when this happens.
        if not _looks_like_author_line(s):
            trimmed = _split_authors_before_affiliation(s)
            if trimmed and _looks_like_author_line(trimmed):
                candidate_lines.append(trimmed)
                break

        if _looks_like_author_line(s):
            candidate_lines.append(s)
            # Stop after first plausible line — rare to span multi-line
            # unless it ends with a comma or "and"
            if not re.search(r"(?:,|\band\b)\s*$", s, re.IGNORECASE):
                break
        elif candidate_lines:
            # We hit affiliation etc. after gathering some lines
            break

    if not candidate_lines:
        return []

    full_line = " ".join(candidate_lines)
    # Strip institution suffixes (after trailing comma+capitalized institution)
    # Keep it simple: split on commas, drop trailing parts that match
    # affiliation tokens
    authors = _split_authors(full_line)
    # Reasonable cap (more than 8 = probably we leaked an affiliation list)
    return authors[:8]


def _extract_year(text: str, url: str | None) -> int | None:
    """Extract year, preferring URL > version line > date > standalone year."""
    # 1. arxiv URL
    if url:
        m = _ARXIV_URL_YEAR_RE.search(url)
        if m:
            return 2000 + int(m.group(1))

    # 2. "First version: ... 2018" / "Received 15 October 2000"
    m = _VERSION_YEAR_RE.search(text)
    if m:
        return int(m.group(1))

    # 3. Date pattern (Month YYYY)
    m = _DATE_MONTH_RE.search(text)
    if m:
        return int(m.group(0)[-4:])

    # 4. Standalone 4-digit year, most-frequent in first 800 chars
    matches = _BARE_YEAR_RE.findall(text[:800])
    if matches:
        # Pick most common; tie-breaker: latest
        counts = Counter(matches)
        best, _ = counts.most_common(1)[0]
        year = int(best)
        if 1980 <= year <= 2030:
            return year
    return None


def _format_short_cite(authors: list[str], year: int | None) -> str | None:
    """Format ``Lastname YYYY`` / ``A & B YYYY`` / ``A et al. YYYY``."""
    if not authors:
        return None
    lastnames = [_extract_lastname(a) for a in authors]
    lastnames = [ln for ln in lastnames if ln]
    if not lastnames:
        return None

    year_suffix = f" {year}" if year else ""
    if len(lastnames) == 1:
        return f"{lastnames[0]}{year_suffix}"
    if len(lastnames) == 2:
        return f"{lastnames[0]} & {lastnames[1]}{year_suffix}"
    return f"{lastnames[0]} et al.{year_suffix}"


def _format_platform_short_cite(
    meta: dict[str, Any], platform: str, year: int | None,
) -> str | None:
    """Build a content-rich citation for platform-attributed sources.

    Pattern: ``"<Page title> (<Platform> [YYYY])"``. The page title is the
    actual content signal — "Wikipedia 2026" is useless for retrieval, but
    "Kelly criterion (Wikipedia 2026)" identifies the article. Falls back
    to the platform name alone when no title is recoverable.
    """
    title = _harvest_title(meta)
    year_suffix = f" {year}" if year else ""
    if title:
        return f"{title} ({platform}{year_suffix})"
    return f"{platform}{year_suffix.lstrip()}" if year_suffix else platform


_SHORT_CITE_TITLE_MAX = 80


def _harvest_title(meta: dict[str, Any]) -> str | None:
    """Best-effort page title from page_metadata or top-level meta.

    Strips trailing " - <Brand>" / " | <Brand>" so the citation isn't doubly
    branded. Truncates to keep short_cite readable.
    """
    pm = meta.get("page_metadata") or {}
    title = pm.get("title") if isinstance(pm, dict) else None
    if not isinstance(title, str) or not title.strip():
        title = meta.get("title")
    if not isinstance(title, str) or not title.strip():
        return None
    cleaned = title.strip()
    cleaned = re.sub(r"\s*[-—|]\s*[^-—|]+\s*$", "", cleaned).strip()
    if not cleaned:
        return None
    if len(cleaned) > _SHORT_CITE_TITLE_MAX:
        cleaned = cleaned[: _SHORT_CITE_TITLE_MAX - 1].rstrip() + "…"
    return cleaned


def _has_page_metadata(meta: dict[str, Any]) -> bool:
    """True if any page_metadata signal was harvested for this source.

    Empty/missing page_metadata + empty body text == no content reached us.
    Don't attribute the host as 'author' in that case — fail-page handling
    has already removed the body; making up a sitename citation is noise.
    """
    pm = meta.get("page_metadata")
    if not isinstance(pm, dict):
        return False
    return any(pm.get(k) for k in ("title", "authors", "sitename", "published"))


# Month / weekday names — date or schedule fragments that leak into the
# sitename field on PDF-derived pages where structured signals are weak.
_MONTH_NAMES = frozenset({
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept",
    "oct", "nov", "dec",
})
_WEEKDAY_NAMES = frozenset({
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "mon", "tue", "wed", "thu", "fri", "sat", "sun",
})

# Words that signal "this is a fail-page, not real content". Used in
# sitename validation to avoid attributing CF / captcha stubs to the
# block service.
_FAIL_PAGE_BRAND_BLOCKLIST = frozenset({
    "cloudflare", "recaptcha", "captcha", "datadome", "perimeterx",
    "imperva", "akamai", "incapsula",
})

# Brand canonical casing. Many sources expose the same brand with
# different casing depending on whether it came from <title>, sitename,
# or URL-host fallback. Canonicalize on output.
_BRAND_CANONICAL = {
    "github": "GitHub",
    "mdpi": "MDPI",
    "researchgate": "ResearchGate",
    "arxiv": "arXiv",
    "pmc": "PMC",
    "quantstart": "QuantStart",
    "ideas/repec": "IDEAS/RePEc",
    "ssrn": "SSRN",
    "papers with code": "Papers with Code",
}


def _canonicalize_brand(name: str) -> str:
    """Return the canonical casing for a known brand; else *name* unchanged."""
    return _BRAND_CANONICAL.get(name.lower(), name)


# Lowercase function/connector words. A real brand has at most one of these
# ("Bank of America"); prose phrases have multiple ("With the collaboration of").
_FUNCTION_WORDS = frozenset({
    "the", "of", "with", "in", "on", "for", "by", "as", "an", "a", "and", "or",
    "to", "at", "from", "via", "per", "into", "onto", "upon",
})


def _valid_sitename(name: str | None) -> str | None:
    """Filter clearly bad sitename candidates (month names, all-caps prose,
    sentence fragments). Returns the cleaned name or None to reject."""
    if not name:
        return None
    s = name.strip()
    if not s:
        return None
    # Month / weekday / fail-page brand anywhere → not a real sitename.
    # Date strings, schedule fragments, and CF/captcha block notices leak
    # into the sitename field on PDF-derived sources where structured
    # signals are weak.
    tokens = re.findall(r"[A-Za-z]+", s)
    tokens_lower = [t.lower() for t in tokens]
    if any(
        t in _MONTH_NAMES or t in _WEEKDAY_NAMES or t in _FAIL_PAGE_BRAND_BLOCKLIST
        for t in tokens_lower
    ):
        return None
    # Sentence punctuation or >4 words → prose, not a brand
    if _SENTENCE_END_RE.search(s):
        return None
    if len(s.split()) > 4:
        return None
    # All-caps run of 2+ words → PDF cover-page text ("VERSION OF DECEMBER")
    if s.isupper() and " " in s:
        return None
    # ≥2 lowercase function words → prose ("With the collaboration of")
    function_count = sum(1 for t in tokens_lower if t in _FUNCTION_WORDS)
    if function_count >= 2:
        return None
    return _canonicalize_brand(s)


def _sitename_fallback(meta: dict[str, Any], url: str | None) -> str | None:
    """Return the sitename as a single-author attribution.

    Priority: ``page_metadata.sitename`` (trafilatura) > tail of page title
    (after a `` - `` / `` | `` separator) > URL host stripped of TLD. Used
    when no human author is recoverable so corporate/blog content still has
    a meaningful citation handle. Each candidate is validated via
    :func:`_valid_sitename` and case-canonicalized.
    """
    pm = meta.get("page_metadata") or {}
    if isinstance(pm, dict):
        cand = _valid_sitename(pm.get("sitename") if isinstance(pm.get("sitename"), str) else None)
        if cand:
            return cand
        title = pm.get("title")
        if isinstance(title, str):
            m = re.search(r"\s+[-—|]\s+([^-—|]{2,40})\s*$", title)
            if m:
                cand = _valid_sitename(m.group(1))
                if cand:
                    return cand

    title_top = meta.get("title")
    if isinstance(title_top, str):
        m = re.search(r"\s+[-—|]\s+([^-—|]{2,40})\s*$", title_top)
        if m:
            cand = _valid_sitename(m.group(1))
            if cand:
                return cand

    if url:
        from urllib.parse import urlparse  # noqa: PLC0415
        host = urlparse(url).netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        base = host.rsplit(".", 1)[0] if "." in host else host
        if base:
            derived = base.replace("-", " ").replace(".", " ").title()
            return _canonicalize_brand(derived)
    return None


def _year_from_page_metadata(meta: dict[str, Any]) -> int | None:
    """Pull a 4-digit year out of ``page_metadata.published`` if present.

    Trafilatura stores ISO-date strings like ``"2025-03-14"`` or sometimes
    just a year. Used as a fallback when text-based year extraction fails
    (typical for web sources without dated headers in the body).
    """
    pm = meta.get("page_metadata") or {}
    if not isinstance(pm, dict):
        return None
    pub = pm.get("published")
    if not isinstance(pub, str):
        return None
    m = re.search(r"\b(19|20)\d{2}\b", pub)
    if not m:
        return None
    y = int(m.group(0))
    if 1980 <= y <= 2030:
        return y
    return None


# ---------------------------------------------------------------------------
# Public API: extract_metadata
# ---------------------------------------------------------------------------


def _authors_from_page_metadata(meta: dict[str, Any]) -> list[str] | None:
    """Return validated authors from ``meta['page_metadata']`` or None.

    Structured HTML tags (``<meta name="citation_author">``, JSON-LD
    ScholarlyArticle) outrank text-scan extraction when each entry passes
    :func:`_looks_like_author_line`. Returning None signals "fall back to
    text scan" — used both when no page_metadata is present and when any
    entry fails validation (a half-bogus list is no better than no signal).
    """
    pm = meta.get("page_metadata")
    if not isinstance(pm, dict):
        return None
    authors = pm.get("authors")
    if not isinstance(authors, list) or not authors:
        return None
    cleaned: list[str] = []
    for a in authors:
        if not isinstance(a, str) or not _looks_like_author_line(a):
            return None
        cleaned.append(a.strip())
    return cleaned


def extract_metadata(doc_dir: Path, meta: dict[str, Any]) -> DocMetadata:
    """Extract metadata for a single doc dir.

    Signal priority:
      1. ``meta["page_metadata"]["authors"]`` (trafilatura + citation tags)
      2. text-scan of ``pdf.txt`` / ``nlm.txt`` — academic-format sources
         have a "title + author block" structure that the rule extractor
         reliably catches.
      3. ``web.txt`` — text-scan is DISABLED. Generic web pages have no
         author block; falling through to text-scan extracts page titles
         or nav text. When ``page_metadata.authors`` is absent for a web
         source, we return ``[]`` rather than ship garbage.

    Year extraction always runs against whatever text is available.
    """
    text, source_fname = _read_doc_text(doc_dir)
    url = meta.get("url") if isinstance(meta.get("url"), str) else None

    platform = _platform_match(url)
    sitename_author: str | None = None
    if platform is not None:
        authors_attribution, _short_brand = platform
        authors: list[str] = [authors_attribution]
        page_authors = None
    else:
        page_authors = _authors_from_page_metadata(meta)
        if page_authors is not None:
            authors = page_authors
        elif source_fname == "pdf.txt":
            # Real PDFs have a title + author block. nlm.txt and web.txt are
            # NotebookLM / trafilatura dumps of arbitrary pages — text-scan
            # there pulls nav text on aggregator/blog sources, even when the
            # URL looks academic. Page_metadata is the only trustworthy
            # author signal for non-PDF sources.
            authors = _extract_authors(text)
        elif text or _has_page_metadata(meta):
            # Sitename fallback: corporate/blog content without a human byline
            # ("Quant Blueprint", "Mergify", "QuantStart") — the brand IS the
            # author. Better signal than empty for retrieval and citation.
            # Skipped when the source completely failed to fetch (no text and
            # no page_metadata) so we don't attribute fail-pages to the host.
            sitename_author = _sitename_fallback(meta, url)
            authors = [sitename_author] if sitename_author else []
        else:
            authors = []
    year = _extract_year(text, url) or _year_from_page_metadata(meta)
    if platform is not None:
        _, short_brand = platform
        short_cite = _format_platform_short_cite(meta, short_brand, year)
    elif sitename_author is not None:
        short_cite = _format_platform_short_cite(meta, sitename_author, year)
    else:
        short_cite = _format_short_cite(authors, year)

    if page_authors is not None or (
        authors and year and platform is None and sitename_author is None
    ):
        confidence = "high"
    elif authors or year:
        # Platform / sitename fallback caps confidence at medium — the brand
        # is a known attribution, not an individual author.
        confidence = "medium"
    elif text:
        confidence = "low"
    else:
        confidence = "none"

    return DocMetadata(
        authors=authors,
        year=year,
        short_cite=short_cite,
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# Public API: enrich_meta_inline
# ---------------------------------------------------------------------------


_CONFIDENCE_RANK = {"none": 0, "low": 1, "medium": 2, "high": 3}


def _merge_with_guard(
    old_md: dict[str, Any],
    new_md: dict[str, object],
) -> dict[str, object]:
    """Non-regression guard for the rule-based author/year fields.

    When the existing ``metadata.confidence`` is strictly higher than the new
    extraction's confidence, preserve the old ``authors``/``year``/
    ``short_cite``/``confidence``. New fields owned by this PR
    (``arxiv_id``, ``doi``, ``extraction_quality``) always come from the new
    block. Used by both inline and CLI paths to avoid silently downgrading
    sources that had hand-tuned high-confidence extractions.
    """
    old_conf = _CONFIDENCE_RANK.get(str(old_md.get("confidence") or ""), 0)
    new_conf = _CONFIDENCE_RANK.get(str(new_md.get("confidence") or ""), 0)
    if old_conf > new_conf and old_md.get("authors"):
        new_md["authors"] = old_md["authors"]
        new_md["year"] = old_md.get("year")
        new_md["short_cite"] = old_md.get("short_cite")
        new_md["confidence"] = old_md.get("confidence")
    return new_md


_QUALITY_KEYS = (
    "n_pages", "n_tables", "n_equations", "n_figures", "n_lists", "has_structured",
)


def _build_metadata_block(meta: dict[str, Any], doc_dir: Path) -> dict[str, object]:
    """Compute the canonical ``metadata`` block for *meta*, before any merge.

    Single source of truth for the seven owned fields:
    ``authors``, ``year``, ``short_cite``, ``confidence``, ``arxiv_id``, ``doi``,
    ``extraction_quality``. Used by both the inline (Spider) and CLI backfill
    paths so they can never drift.
    """
    md = extract_metadata(doc_dir, meta)
    arxiv_id = extract_arxiv_id(meta.get("url", ""))
    doi = _extract_doi_from_meta(meta, doc_dir, arxiv_id=arxiv_id)
    qc = quality_counts(doc_dir / "content_list.json")
    return {
        "authors": md.authors,
        "year": md.year,
        "short_cite": md.short_cite,
        "confidence": md.confidence,
        "arxiv_id": arxiv_id,
        "doi": doi,
        "extraction_quality": {k: qc[k] for k in _QUALITY_KEYS},
    }


def enrich_meta_inline(meta: dict[str, Any], doc_dir: Path) -> None:
    """Enrich *meta* dict inline with full metadata block. Never raises.

    Called from ``FetchPipelineSpider.on_scraped_item``. When *meta* already
    carries a higher-confidence ``metadata`` block, the rule-based author
    fields are preserved (see :func:`_merge_with_guard`).
    """
    try:
        new_md = _build_metadata_block(meta, doc_dir)
        old_md = meta.get("metadata") or {}
        if isinstance(old_md, dict):
            new_md = _merge_with_guard(old_md, new_md)
        meta["metadata"] = new_md
    except Exception as exc:
        print(
            f"[metadata] enrich_meta_inline failed for {doc_dir}: {exc}",
            file=sys.stderr,
        )


def _extract_doi_from_meta(
    meta: dict[str, Any],
    doc_dir: Path,
    *,
    arxiv_id: str | None = None,
) -> str | None:
    """Resolve a DOI for *meta*. Priority order:

    1. ``page_metadata.doi`` from harvested HTML meta tags.
    2. Regex scan of ``pdf.txt`` / ``nlm.txt`` / ``web.txt`` (in that order).
       Only ~20% of sources have ``pdf.txt`` — the rest only have NotebookLM
       fulltext or trafilatura output, so the fallback must read all three.
       Full file content (no 5KB cap) — DOIs commonly appear in footers /
       bibliographies, well past the first page.
    3. arXiv-derived DOI: when *arxiv_id* is provided, the canonical DOI is
       ``10.48550/arXiv.<id>`` (registered with DataCite by arXiv since 2022;
       resolves on doi.org).
    """
    pm = meta.get("page_metadata")
    if isinstance(pm, dict):
        doi_val = pm.get("doi")
        if isinstance(doi_val, str):
            return doi_val

    for fname in ("pdf.txt", "nlm.txt", "web.txt"):
        f = doc_dir / fname
        if not f.exists():
            continue
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        doi = extract_doi_from_pdf_text(text)
        if doi:
            return doi

    if arxiv_id:
        return f"10.48550/arXiv.{arxiv_id}"
    return None


# ---------------------------------------------------------------------------
# Public API: enrich_meta_json (idempotent backfill CLI helper)
# ---------------------------------------------------------------------------


def _merged_block_for(
    meta: dict[str, Any], doc_dir: Path, *, force: bool = False,
) -> tuple[dict[str, object], dict[str, Any]]:
    """Return ``(new_md, old_md)`` for *meta*: the merged block ready to write,
    plus the old block for comparison. Preserves non-owned fields (``title``,
    ``host``, ``source_kind``) and applies the non-regression guard.

    When *force* is True, skip ``_merge_with_guard`` — needed for one-time
    cleanup of bogus ``confidence="medium"`` entries that would otherwise
    survive a re-extraction at lower confidence.
    """
    old_md = meta.get("metadata", {}) if isinstance(meta.get("metadata"), dict) else {}
    new_md = _build_metadata_block(meta, doc_dir)
    if not force:
        new_md = _merge_with_guard(old_md, new_md)
    for key in ("title", "host", "source_kind"):
        if key in old_md:
            new_md[key] = old_md[key]
    return new_md, old_md


def enrich_meta_json(meta_path: Path, *, force: bool = False) -> bool:
    """Enrich a single ``meta.json`` file in place.  Idempotent.

    Returns ``True`` if the file was modified, ``False`` if it was already
    up-to-date. *force* bypasses the non-regression guard (cleanup-only).
    """
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"  ERROR reading {meta_path}: {exc}", file=sys.stderr)
        return False

    new_md, old_md = _merged_block_for(meta, meta_path.parent, force=force)
    if _metadata_deep_equal(old_md, new_md):
        return False

    meta["metadata"] = new_md
    meta_path.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return True


def _metadata_deep_equal(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Compare two metadata dicts for equality.  Lists are compared element-wise."""
    if set(a.keys()) != set(b.keys()):
        return False
    for key in a:
        va = a[key]
        vb = b.get(key)
        if isinstance(va, list) and isinstance(vb, list):
            if va != vb:
                return False
        elif isinstance(va, dict) and isinstance(vb, dict):
            if not _metadata_deep_equal(va, vb):
                return False
        elif va != vb:
            return False
    return True


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


def _walk_source_dirs(sources_root: Path) -> list[Path]:
    """Yield all doc dirs (``sources/<slug>/<doc_dir>/``) that have a meta.json."""
    dirs: list[Path] = []
    for slug_dir in sorted(sources_root.iterdir()):
        if not slug_dir.is_dir() or slug_dir.name.startswith("_"):
            continue
        for d in sorted(slug_dir.iterdir()):
            if d.is_dir() and (d / "meta.json").exists():
                dirs.append(d)
    return dirs


def _compute_diff_output(
    old_meta: dict[str, Any], meta_path: Path, *, force: bool = False,
) -> None:
    """Print a human-readable diff of what would change in *meta_path*.

    Reuses :func:`_merged_block_for` so the dry-run reflects exactly what
    :func:`enrich_meta_json` would write.
    """
    new_md, old_md = _merged_block_for(old_meta, meta_path.parent, force=force)

    changed_keys = [
        key for key in sorted(set(old_md.keys()) | set(new_md.keys()))
        if old_md.get(key) != new_md.get(key)
    ]
    if not changed_keys:
        return

    short_slug = str(meta_path.relative_to(meta_path.parents[2]))
    final_conf = str(new_md.get("confidence") or "")
    print(f"  [{final_conf:6s}] {short_slug:60s}")
    for key in changed_keys:
        old_val = old_md.get(key)
        new_val = new_md.get(key)
        if isinstance(new_val, dict):
            print(f"    {key}: (new block)")
        else:
            print(f"    {key}: {old_val!r} -> {new_val!r}")


# ---------------------------------------------------------------------------
# __main__ CLI
# ---------------------------------------------------------------------------


def _main() -> int:
    p = argparse.ArgumentParser(
        description="Rule-based metadata extraction + enrichment. "
        "Writes full metadata blocks to meta.json.",
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--slug", help="Single notebook slug under sources/")
    g.add_argument("--all", action="store_true", help="Process every slug under sources/")
    g.add_argument("--doc", help="Single doc directory path")
    p.add_argument("--dry-run", action="store_true", help="Print results, don't write")
    p.add_argument(
        "--force",
        action="store_true",
        help="Bypass the non-regression guard (cleanup-only; overwrites old high-confidence rows)",
    )
    p.add_argument(
        "--sources-root",
        default=None,
        help="Override sources root directory (default: ROOT/sources)",
    )

    args = p.parse_args()

    # Resolve sources root
    if args.sources_root:
        sources_root = Path(args.sources_root).resolve()
    else:
        # Find the project root (2 levels up from src/fetch/metadata.py)
        module_dir = Path(__file__).resolve().parent  # src/fetch/
        root = module_dir.parent.parent  # rag-system/
        sources_root = root / "sources"

    if not sources_root.exists():
        print(f"ERROR: sources root not found: {sources_root}", file=sys.stderr)
        return 1

    # --doc mode
    if args.doc:
        doc_dir = Path(args.doc).resolve()
        if not doc_dir.is_dir():
            print(f"ERROR: not a directory: {doc_dir}", file=sys.stderr)
            return 1
        meta_path = doc_dir / "meta.json"
        if not meta_path.exists():
            print(f"ERROR: no meta.json in {doc_dir}", file=sys.stderr)
            return 1

        if args.dry_run:
            old = json.loads(meta_path.read_text(encoding="utf-8"))
            _compute_diff_output(old, meta_path, force=args.force)
            return 0

        changed = enrich_meta_json(meta_path, force=args.force)
        print(
            f"{'Updated' if changed else 'Already up-to-date'}: "
            f"{meta_path.relative_to(sources_root.parent)}"
        )
        return 0

    # --slug or --all mode
    if args.all:
        doc_dirs = _walk_source_dirs(sources_root)
    else:
        slug_dir = sources_root / args.slug
        if not slug_dir.is_dir():
            print(f"ERROR: slug dir not found: {slug_dir}", file=sys.stderr)
            return 1
        doc_dirs = [
            d for d in sorted(slug_dir.iterdir())
            if d.is_dir() and (d / "meta.json").exists()
        ]

    if not doc_dirs:
        print("No doc dirs found.", file=sys.stderr)
        return 0

    for d in doc_dirs:
        meta_path = d / "meta.json"

        if args.dry_run:
            try:
                old = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception as exc:
                print(f"  ERROR reading {meta_path}: {exc}", file=sys.stderr)
                continue
            _compute_diff_output(old, meta_path, force=args.force)
            continue

        try:
            changed = enrich_meta_json(meta_path, force=args.force)
        except Exception as exc:
            print(f"  ERROR {meta_path}: {exc}", file=sys.stderr)
            continue

        if changed:
            short = str(meta_path.relative_to(sources_root.parent))
            print(f"  updated: {short}")

    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
