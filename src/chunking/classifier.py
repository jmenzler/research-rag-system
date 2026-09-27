"""Section-role classifier — high-precision regex; `body` fallback.

The classifier decides which `SectionProfile` to apply to a section. False
positives are worse than misses: a section that fails to match falls through
to the safe `body` profile (matches today's chunker behavior).
"""

from __future__ import annotations

import re

ROLE_PATTERNS: list[tuple[str, str]] = [
    ("references",  r"^(references?|bibliograph(y|ies))$"),
    ("acknowledg",  r"^(acknowledg(e?ments?))$"),
    ("toc",         r"^(table\s*of\s*contents?|contents?|toc|outline|index)$"),
    ("abstract",    r"^abstract$"),
    ("intro",       r"^(introduction|intro)$"),
    ("related",     r"^(related\s*work|prior\s*work|background|literature\s*review)$"),
    ("method",      r"^(methods?|methodology|approach|materials?\s*and\s*methods?|"
                    r"implementation|system\s*design|architecture|model|"
                    r"experimental?\s*setup|setup)$"),
    ("results",     r"^(results?|findings?|analysis|experiments?|evaluation|"
                    r"validation|performance|empirical|empirics?)$"),
    ("discussion",  r"^discussion$"),
    ("limitations", r"^limitations?$"),
    ("future",      r"^future\s*work$"),
    ("conclusion",  r"^(conclusions?|concluding\s*remarks?|summary)$"),
]
_COMPILED = [(role, re.compile(pat, re.I)) for role, pat in ROLE_PATTERNS]
_NUMBERING_RE = re.compile(r"^(\d+(\.\d+)*)\s*[.\)]?\s+")

# Drop these sections wholesale — never useful for retrieval.
DROP_ROLES = {"references", "acknowledg", "toc"}


def normalize_heading(text: str) -> str:
    """Lowercase, strip leading section numbering, strip trailing punctuation."""
    t = text.strip().lower()
    t = _NUMBERING_RE.sub("", t)
    t = t.rstrip(":.,;!?")
    return t


def classify_section_role(heading_text: str) -> str:
    """Return role tag, or 'body' when no high-precision pattern matches."""
    norm = normalize_heading(heading_text)
    for role, pat in _COMPILED:
        if pat.match(norm):
            return role
    return "body"


# Lines like "title.... 8" — dotted leaders followed by a trailing page number.
_TOC_DOTTED_LINE_RE = re.compile(r"\.{3,}\s*\d+\s*$")
# Lines like "1.1 Foo .... 12" — numeric prefix AND a trailing page number,
# both on the same line. A bare "1.1 Foo" without a trailing number is a
# section heading or contract clause, not a ToC entry.
_TOC_NUMBERED_LINE_RE = re.compile(r"^\d+(\.\d+)*\s+\S.*\s+\d+\s*$")


def looks_like_toc(text: str) -> bool:
    """Structural ToC detector.

    Triggers when either signal exceeds threshold across non-empty lines:
      * >=40% of lines end with dotted leaders + page number
      * >=60% of lines start with a numbered prefix AND end with a page
        number on the same line (catches dotless ToCs like
        ``5.1 Technology  10`` while excluding legal-contract clauses
        and academic numbered subsections, which lack the trailing number)
    A short section (<5 non-empty lines) is never flagged — too noisy for
    legitimate numbered-step or bullet content.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 5:
        return False
    dotted = sum(1 for ln in lines if _TOC_DOTTED_LINE_RE.search(ln))
    numbered = sum(1 for ln in lines if _TOC_NUMBERED_LINE_RE.match(ln))
    return dotted / len(lines) >= 0.40 or numbered / len(lines) >= 0.60
