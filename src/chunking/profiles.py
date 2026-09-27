"""Parent and child token budgets by document section role."""

from __future__ import annotations

from dataclasses import dataclass

from src.config import (
    CHILD_MIN_TOKENS as _CHILD_MIN_TOKENS,
)
from src.config import (
    CHUNK_TEXT_BUDGET_CHARS as _CHUNK_TEXT_BUDGET_CHARS,
)
from src.config import (
    MERGE_UNDER_TOKENS as _MERGE_UNDER_TOKENS,
)


@dataclass(frozen=True)
class SectionProfile:
    """Token budgets for one section role."""

    parent_tokens: int
    child_tokens: int
    child_overlap: int


SECTION_PROFILES: dict[str, SectionProfile] = {
    "abstract":    SectionProfile(400,  400,  0),    # atomic
    "intro":       SectionProfile(1000, 512,  75),
    "related":     SectionProfile(1000, 512,  50),
    "method":      SectionProfile(1200, 768,  150),  # procedural integrity
    "results":     SectionProfile(1000, 512,  50),   # table-inclusive
    "discussion":  SectionProfile(1000, 640,  100),
    "limitations": SectionProfile(800,  512,  50),
    "future":      SectionProfile(600,  400,  25),
    "conclusion":  SectionProfile(400,  400,  0),    # atomic
    "body":        SectionProfile(1000, 512,  50),   # safe default
}


# Child chunks below this token count are dropped — empirically catches
# orphan author lines, isolated dataset names, page footers.
CHILD_MIN_TOKENS: int = _CHILD_MIN_TOKENS

# Chunks (parents or children) below this are candidates for merging with
# neighbours.  Never used as a drop threshold — dropping loses information.
MERGE_UNDER_TOKENS: int = _MERGE_UNDER_TOKENS

# Hard char budget for any single emitted chunk (after breadcrumb prepend).
# Tracks Milvus VARCHAR(text) - safety margin; configurable via env.
TEXT_FIELD_MAX_CHARS: int = _CHUNK_TEXT_BUDGET_CHARS
