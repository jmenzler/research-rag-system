"""PDF document text extractors.

Public API:
    parse_pdf(path)  → str   MinerU pipeline + VLM fallback

Submodules:
    src.pdf_parsers.mineru         — active parser (PDF → text + content_list.json sidecar)
    src.pdf_parsers.mineru_daemon  — long-running MinerU server + client
"""
from __future__ import annotations

from src.pdf_parsers.mineru import parse_pdf

__all__ = ["parse_pdf"]
