"""Create a doc dir for a paper and ingest it into the corpus.

PDF path (preferred): download source.pdf → MinerU → content_list.json → ingest.
Abstract fallback: write synthetic content_list.json from title+abstract.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

import httpx
import tiktoken
from google import genai

from src import config
from src.ingest.chunker import _get_encoder
from src.ingest.ingest import ingest_file
from src.ingest.sources import is_ingested, upsert
from src.ingest.storage import _open_parents_db
from src.milvus_client import ensure_collection, ensure_partition
from src.monitor.paper import Paper
from src.pdf_parsers.mineru import parse_pdf

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SOURCES_DIR = _PROJECT_ROOT / "sources" / "trading"
_COLLECTION = "trading"
_NOTEBOOK = "arxiv_monitor"


def _short_cite(authors: list[str], year: int) -> str:
    if not authors:
        return str(year)
    surnames = [a.split()[-1] for a in authors if a.strip()]
    if not surnames:
        return str(year)
    if len(surnames) == 1:
        return f"{surnames[0]} {year}"
    if len(surnames) == 2:
        return f"{surnames[0]} & {surnames[1]} {year}"
    return f"{surnames[0]} et al. {year}"


def _write_abstract_stub(doc_dir: Path, paper: Paper) -> Path:
    content = [
        {
            "type": "text",
            "text": f"{paper.title}\n\n{paper.abstract}",
            "page_idx": 0,
        }
    ]
    path = doc_dir / "content_list.json"
    path.write_text(json.dumps(content, ensure_ascii=False), encoding="utf-8")
    return path


def _write_meta(doc_dir: Path, paper: Paper, extraction_quality: str) -> None:
    year = paper.published.year
    meta = {
        "title": paper.title,
        "authors": paper.authors,
        "year": year,
        "arxiv_id": paper.id,
        "doi": None,
        "short_cite": _short_cite(paper.authors, year),
        "confidence": "high",
        "extraction_quality": extraction_quality,
    }
    (doc_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _download_pdf(url: str, dest: Path) -> bool:
    """Download PDF to *dest*. Returns True on success."""
    try:
        resp = httpx.get(url, follow_redirects=True, timeout=60.0)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        return True
    except Exception:
        logger.warning("PDF download failed: %s", url, exc_info=True)
        return False


def _parse_pdf_to_sidecar(pdf_path: Path) -> bool:
    """Run MinerU on *pdf_path*; writes content_list.json to same dir. Returns True on success."""
    try:
        parse_pdf(pdf_path)
        sidecar = pdf_path.parent / "content_list.json"
        if not sidecar.exists():
            logger.warning("MinerU ran but content_list.json not found for %s", pdf_path)
            return False
        return True
    except Exception:
        logger.warning("MinerU parse failed for %s", pdf_path, exc_info=True)
        return False


def ingest_paper(paper: Paper) -> bool:
    """Ingest one paper into the trading corpus.

    Downloads the full PDF if available; falls back to abstract stub.
    Returns True if ingested, False if skipped (already present or ingest failed).
    """
    db_conn = _open_parents_db()
    try:
        if is_ingested(db_conn, paper.id):
            logger.debug("skip %s — already ingested", paper.id)
            return False

        doc_dir = _SOURCES_DIR / paper.id
        doc_dir.mkdir(parents=True, exist_ok=True)

        # --- attempt full PDF ingestion ---
        extraction_quality = "abstract_only"
        if paper.pdf_url:
            pdf_path = doc_dir / "source.pdf"
            if _download_pdf(paper.pdf_url, pdf_path) and _parse_pdf_to_sidecar(pdf_path):
                extraction_quality = "full_pdf"
            else:
                logger.info("PDF path failed for %s — falling back to abstract stub", paper.id)

        # write abstract stub only when MinerU didn't produce a sidecar
        content_list_path = doc_dir / "content_list.json"
        if not content_list_path.exists():
            _write_abstract_stub(doc_dir, paper)

        _write_meta(doc_dir, paper, extraction_quality)

        config.validate_api_key()
        ensure_collection(_COLLECTION)
        ensure_partition(_COLLECTION, _NOTEBOOK)

        gemini_client = genai.Client(api_key=config.GEMINI_API_KEY)
        enc: tiktoken.Encoding = _get_encoder()
        ingest_conn: sqlite3.Connection = _open_parents_db()
        try:
            ingest_file(
                path=content_list_path,
                notebook=_NOTEBOOK,
                collection=_COLLECTION,
                gemini_client=gemini_client,
                db_conn=ingest_conn,
                enc=enc,
            )
        finally:
            ingest_conn.close()

        upsert(
            db_conn,
            canonical_id=paper.id,
            id_type="arxiv",
            status="ingested",
            title=paper.title,
            authors=paper.authors,
            year=paper.published.year,
            url=paper.url,
            doi=None,
            arxiv_id=paper.id,
            source_type="arxiv",
            notebook=_NOTEBOOK,
            file_hash=None,
        )
        logger.info(
            "ingested %s (%s) — %s", paper.id, extraction_quality, paper.title[:80]
        )
        return True

    except Exception:
        logger.error("ingest_paper failed for %s", paper.id, exc_info=True)
        return False
    finally:
        db_conn.close()
