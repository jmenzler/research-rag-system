"""Extension-gate tests for the ingest dispatch.

Covers the regression where `_TEXT_EXTENSIONS = {".pdf"}` blocked
web-scraped sources (`.txt`/`.html`) that ship with a `content_list.json`
sidecar but no source.pdf — the structural chunker only needs the JSON.

Boundary the tests pin:
- `.pdf`/`.html`/`.md`/`.txt` are accepted at the dispatcher gate
- non-PDF inputs WITHOUT a structural sidecar are skipped cleanly
  (no MinerU call on text — that produces garbage)
- non-PDF inputs WITH a structural sidecar route through the structural
  path (path used only for `source_file` label and `path.parent`)
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from src.ingest.ingest import _TEXT_EXTENSIONS, ingest_file


def _make_db(tmp_path: Path) -> sqlite3.Connection:
    """Open a parents-store sqlite with the schema the ingest code expects."""
    from src.ingest.sources import init_schema  # noqa: PLC0415

    conn = sqlite3.connect(tmp_path / "parents.sqlite")
    init_schema(conn)
    return conn


def _write_minimal_sidecar(dir_: Path) -> Path:
    """Write a tiny content_list.json that chunk_sidecar will accept."""
    sidecar = dir_ / "content_list.json"
    sidecar.write_text(
        json.dumps(
            [
                {"type": "text", "text": "Test Paper Title", "text_level": 1, "page_idx": 0},
                {"type": "text", "text": "Some body content for the paper.", "page_idx": 0},
            ]
        )
    )
    return sidecar


def test_text_extensions_includes_non_pdf() -> None:
    """The gate accepts the structural-sidecar-bearing extensions, not just .pdf."""
    assert ".pdf" in _TEXT_EXTENSIONS
    assert ".txt" in _TEXT_EXTENSIONS
    assert ".html" in _TEXT_EXTENSIONS
    assert ".md" in _TEXT_EXTENSIONS


def test_unsupported_extension_skipped(tmp_path: Path) -> None:
    """Truly unsupported extensions still skip cleanly."""
    src = tmp_path / "doc.xyz"
    src.write_text("nope")
    conn = _make_db(tmp_path)

    summary = ingest_file(
        path=src,
        notebook="t",
        collection="c",
        gemini_client=MagicMock(),
        db_conn=conn,
        enc=MagicMock(),
    )
    assert summary["skipped"] is True
    assert "unsupported extension" in str(summary["reason"])


def test_non_pdf_without_sidecar_skipped(tmp_path: Path) -> None:
    """`.txt`/`.html` without content_list.json must not call MinerU."""
    src = tmp_path / "web.txt"
    src.write_text("scraped page body")
    conn = _make_db(tmp_path)

    with patch("src.ingest.ingest._parse_text_document") as mock_parse:
        summary = ingest_file(
            path=src,
            notebook="t",
            collection="c",
            gemini_client=MagicMock(),
            db_conn=conn,
            enc=MagicMock(),
        )
    mock_parse.assert_not_called()
    assert summary["skipped"] is False  # extension check passed
    # Reached _ingest_text_file but the inner branch returned (0, 0).
    assert summary["n_parents"] == 0
    assert summary["n_children"] == 0


def test_non_pdf_with_sidecar_routes_structural(tmp_path: Path) -> None:
    """`.txt` next to a content_list.json takes the structural path."""
    src = tmp_path / "nlm.txt"
    src.write_text("byproduct text — chunker reads sidecar instead")
    _write_minimal_sidecar(tmp_path)
    conn = _make_db(tmp_path)

    captured: dict[str, Any] = {}

    def fake_chunk_sidecar(
        sidecar: Path, _enc: object, *, meta: object = None
    ) -> tuple[str, list[Any]]:
        from src.chunking.models import ProtoChunk

        captured["sidecar"] = sidecar
        captured["meta"] = meta
        proto = ProtoChunk(
            parent_idx=0,
            child_idx=0,
            role="body",
            breadcrumb="Test Paper Title",
            page_idx=0,
            modality="text",
            text="Some body content for the paper.",
            raw_text="Some body content for the paper.",
            n_tokens=6,
        )
        return ("Test Paper Title", [proto])

    with (
        patch("src.ingest.ingest.chunk_sidecar", side_effect=fake_chunk_sidecar),
        patch("src.ingest.ingest._parse_text_document") as mock_parse,
        patch("src.ingest.ingest.existing_child_ids", return_value=set()),
        patch("src.ingest.ingest._embed_text_batch", return_value=[[0.1] * 4096]),
        patch("src.ingest.ingest._insert_children", return_value=None),
        patch("src.ingest.ingest._store_parent", return_value=None),
    ):
        summary = ingest_file(
            path=src,
            notebook="t",
            collection="c",
            gemini_client=MagicMock(),
            db_conn=conn,
            enc=MagicMock(),
        )

    mock_parse.assert_not_called()
    assert summary["skipped"] is False
    assert summary["n_parents"] == 1
    assert summary["n_children"] == 1
    assert captured["sidecar"] == tmp_path / "content_list.json"
