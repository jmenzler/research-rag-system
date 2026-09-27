"""Dataclass models for the structural chunker."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Section:
    """A run of elements bounded by L1/L2 heading boundaries."""

    role: str
    breadcrumb: str           # full heading path joined by ' > '
    page_idx: int             # page_idx of the section's first body element
    elements: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ProtoChunk:
    """A single chunk produced by the structural chunker.

    `child_idx is None` marks atomic chunks (parent-only, e.g. tables).
    """

    parent_idx: int
    child_idx: int | None
    role: str
    breadcrumb: str
    page_idx: int
    modality: str             # text | table | equation
    text: str                 # final embed text (with breadcrumb prepend)
    raw_text: str             # text without breadcrumb prepend (for parent storage)
    n_tokens: int
