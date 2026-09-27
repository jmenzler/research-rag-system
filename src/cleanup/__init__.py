"""Corpus cleanup primitives — single source of truth for extraction quality gates.

Used by both extraction (`src/fetch/`) and corpus audit
(`src/filename_content_match_audit.py`, `src/build_cleanup_manifest.py`).

Modules:
- `signatures`     — fail-page literal tokens
- `quality`        — text-quality gates (paragraph floor, min length, fail-page detection)
- `filename_match` — descriptive-token extraction, filename-vs-content overlap (Rule 1)
- `manifest`       — corpus cleanup manifest builder (Rule 1 + Rule 6)
- `retry`          — multi-fetcher retry orchestration with quality-gated stop-early
"""

from src.cleanup.quality import (
    MIN_LONGEST_PARAGRAPH,
    MIN_TEXT_LEN,
    PDF_MAGIC,
    detect_fail_signature,
    longest_paragraph_chars,
    pdf_is_openable,
)
from src.cleanup.signatures import (
    CLOUDFLARE_SIGNATURES,
    FAIL_PAGE_SIGNATURES,
    is_cloudflare_signature,
)

__all__ = [
    "CLOUDFLARE_SIGNATURES",
    "FAIL_PAGE_SIGNATURES",
    "MIN_LONGEST_PARAGRAPH",
    "MIN_TEXT_LEN",
    "PDF_MAGIC",
    "detect_fail_signature",
    "is_cloudflare_signature",
    "longest_paragraph_chars",
    "pdf_is_openable",
]
