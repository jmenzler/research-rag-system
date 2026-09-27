"""Common Paper dataclass shared across all monitor paper sources."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass
class Paper:
    id: str
    title: str
    abstract: str
    authors: list[str] = field(default_factory=list)
    published: datetime = field(default_factory=lambda: datetime.now(UTC))
    categories: list[str] = field(default_factory=list)
    url: str = ""
    pdf_url: str = ""
