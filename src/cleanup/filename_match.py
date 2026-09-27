"""Filename-vs-content overlap scoring — Rule 1's canonical-picking signal.

When a paper's content appears under multiple filenames (cross-partition dupes
or same-partition mis-saves), the right canonical is the filename whose
descriptive tokens best match the actual content's title/body.

This replaces the older filename-vs-filename overlap heuristic, which broke
when a paper was saved once correctly and once under a corrupted filename
(e.g. `BTO/018 backtesting_frameworks_pytrade` vs the real
`literature_review/018 a_hybrid_har_lstm_garch` — same content, only one
filename matches).
"""

from __future__ import annotations

import re
from pathlib import Path

# Filename-token cleanup heuristics
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_MIN_TOK_LEN = 4
_URL_FILENAME_RE = re.compile(r"^https?[_-]")

# Stopwords + URL/site-noise to strip from filenames before scoring.
STOPWORDS: frozenset[str] = frozenset({
    # English stopwords
    "and", "the", "for", "with", "from", "into", "onto", "what", "when", "where",
    "how", "why", "this", "that", "these", "those", "have", "has", "had", "are",
    "was", "were", "been", "being", "will", "would", "could", "should", "your",
    "you", "use", "uses", "using", "used", "via", "but", "not", "all", "any",
    "can", "may", "might", "more", "most", "much", "some", "such", "than",
    "then", "they", "them", "their", "there", "here", "about", "between",
    "across", "above", "below", "after", "before", "during", "since", "until",
    "while", "also", "yet", "out", "off",
    # site/extraction noise — matching these in body wouldn't tell us anything
    "com", "org", "net", "io", "wiki", "wikipedia", "github", "reddit", "papers",
    "arxiv", "pdf", "html", "edu", "blog", "post", "page", "site", "review",
    "guide", "intro", "introduction", "based", "vol", "ssrn", "doi",
})


def descriptive_tokens(path_str: str) -> list[str]:
    """Extract descriptive tokens from a filename, stripping `NNN__` prefix and stopwords.

    Returns [] for URL-derived filenames (start with `http` / `https`) and
    arxiv-only filenames — these have no descriptive content to score against.
    """
    stem = Path(path_str).stem
    if "__" in stem:
        stem = stem.split("__", 1)[-1]
    if _URL_FILENAME_RE.match(stem):
        return []
    toks = _TOKEN_RE.findall(stem.lower())
    return [t for t in toks if len(t) >= _MIN_TOK_LEN and t not in STOPWORDS]


def filename_to_content_overlap(path_str: str, content: str, head_chars: int = 5000) -> float:
    """Ratio of descriptive filename tokens that appear in `content`'s first N chars.

    head_chars=5000 is enough for a paper title + abstract or article header to
    establish topic match without scanning whole-document body.

    Returns 1.0 for URL/arxiv-only filenames (no signal → don't penalize).
    Returns 0.0 if filename has descriptive tokens but none appear in content.
    """
    toks = descriptive_tokens(path_str)
    if not toks:
        return 1.0
    head = content[:head_chars].lower()
    return sum(1 for t in toks if t in head) / len(toks)


def filename_match_ratio(path_str: str, content: str) -> float:
    """Strict variant: search the whole content (not just head). Used by audit.

    Lower ratio = filename promises X but content doesn't deliver — likely a
    corrupted-name save where filename and content disagree.
    """
    toks = descriptive_tokens(path_str)
    if not toks:
        return 1.0
    lc = content.lower()
    return sum(1 for t in toks if t in lc) / len(toks)
