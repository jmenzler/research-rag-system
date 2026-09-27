"""Citation graph configuration for user-supplied Semantic Scholar datasets."""

from __future__ import annotations

import os

TARGET_FIELDS: tuple[str, ...] = (
    "Computer Science",
    "Mathematics",
    "Economics",
    "Business",
    "Physics",
    "Engineering",
)

DEFAULT_RELEASE: str = "2026-05-12"
DEFAULT_BASE: str = os.getenv("CITATIONS_DATA_DIR", "data/citations")
