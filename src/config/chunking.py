"""Chunk sizing, overlap, and Milvus text field budget."""
from __future__ import annotations

import os

PARENT_TOKENS: int = 1000
CHILD_TOKENS: int = 512
CHILD_OVERLAP: int = 50
CHILD_MIN_TOKENS: int = 50
MERGE_UNDER_TOKENS: int = int(os.getenv("MERGE_UNDER_TOKENS", "150"))

MILVUS_TEXT_MAX_CHARS: int = int(os.getenv("MILVUS_TEXT_MAX_CHARS", "4096"))

CHUNK_TEXT_BUDGET_CHARS: int = int(
    os.getenv("CHUNK_TEXT_BUDGET_CHARS", str(MILVUS_TEXT_MAX_CHARS - 300))
)
