"""Unit tests for the two-pass parse flow in src/pdf_parsers/cli.py.

Defines the contract for the new `--pass {auto,pipeline-only,vlm}` flag and
the `pdf.needs_vlm` marker file used to hand off broken-font PDFs from the
pipeline pass to the VLM pass.

These tests mock both `parse_pipeline_only` and `parse_vlm_only` so they run
in <1s and don't touch GPU.

Marker contract:
- Pipeline-only pass: on broken fonts, write `pdf.needs_vlm` next to the PDF
  and DO NOT write `pdf.txt`. On clean output, write `pdf.txt` and DO NOT
  write the marker.
- VLM pass: only process PDFs whose sibling has a `pdf.needs_vlm` marker.
  On success, write `pdf.txt` and DELETE the marker. PDFs without the marker
  are not touched.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

if TYPE_CHECKING:
    from types import ModuleType


@pytest.fixture()
def parse_pdfs_mod() -> ModuleType:
    """Import src/pdf_parsers/cli.py as a module."""
    from src.pdf_parsers import cli as parse_pdfs

    return parse_pdfs


def _make_pdf(tmp_path: Path, name: str = "source.pdf") -> Path:
    """Create a fake PDF (just a file with the right name; we mock the parser)."""
    pdf_dir = tmp_path / "001__fake_doc"
    pdf_dir.mkdir(parents=True)
    pdf = pdf_dir / name
    pdf.write_bytes(b"%PDF-1.4 fake")
    return pdf


# ---- Pipeline-only pass ----------------------------------------------------


def test_pipeline_only_writes_pdftxt_on_clean_output(
    parse_pdfs_mod: ModuleType,
    tmp_path: Path,
) -> None:
    pdf = _make_pdf(tmp_path)
    out = pdf.parent / "pdf.txt"

    with patch("src.pdf_parsers.mineru.parse_pipeline_only", return_value="clean text"):
        ok, _, _ = parse_pdfs_mod.parse_one(pdf, out, pass_mode="pipeline-only")

    assert ok
    assert out.read_text() == "clean text"
    assert not (pdf.parent / "pdf.needs_vlm").exists()


def test_pipeline_only_writes_marker_on_broken_fonts(
    parse_pdfs_mod: ModuleType,
    tmp_path: Path,
) -> None:
    pdf = _make_pdf(tmp_path)
    out = pdf.parent / "pdf.txt"
    marker = pdf.parent / "pdf.needs_vlm"

    with patch("src.pdf_parsers.mineru.parse_pipeline_only", return_value="text with � glyph"):
        ok, _, _ = parse_pdfs_mod.parse_one(pdf, out, pass_mode="pipeline-only")

    # Pipeline-only deferred to VLM: not a failure, but pdf.txt must NOT be
    # written (otherwise the VLM pass's "skip if exists" logic would kick in).
    assert ok
    assert not out.exists()
    assert marker.exists()


def test_pipeline_only_overwrites_stale_marker_on_clean_output(
    parse_pdfs_mod: ModuleType,
    tmp_path: Path,
) -> None:
    """A re-parse of a now-clean PDF removes a leftover marker from a prior run."""
    pdf = _make_pdf(tmp_path)
    out = pdf.parent / "pdf.txt"
    marker = pdf.parent / "pdf.needs_vlm"
    marker.write_text("")  # leftover from earlier run

    with patch("src.pdf_parsers.mineru.parse_pipeline_only", return_value="now clean"):
        ok, _, _ = parse_pdfs_mod.parse_one(pdf, out, pass_mode="pipeline-only", force=True)

    assert ok
    assert out.read_text() == "now clean"
    assert not marker.exists()


# ---- VLM pass --------------------------------------------------------------


def test_vlm_pass_only_processes_marked_pdfs(
    parse_pdfs_mod: ModuleType,
    tmp_path: Path,
) -> None:
    """VLM pass must skip PDFs without a `pdf.needs_vlm` marker."""
    pdf = _make_pdf(tmp_path)
    out = pdf.parent / "pdf.txt"
    # No marker present.

    with patch("src.pdf_parsers.mineru.parse_vlm_only") as mock_vlm:
        ok, status, _ = parse_pdfs_mod.parse_one(pdf, out, pass_mode="vlm")

    mock_vlm.assert_not_called()
    # Skipping is a non-failure, non-write outcome.
    assert ok
    assert "skip" in status.lower() or "no marker" in status.lower()
    assert not out.exists()


def test_vlm_pass_processes_marked_pdf_and_deletes_marker(
    parse_pdfs_mod: ModuleType,
    tmp_path: Path,
) -> None:
    pdf = _make_pdf(tmp_path)
    out = pdf.parent / "pdf.txt"
    marker = pdf.parent / "pdf.needs_vlm"
    marker.write_text("")

    with patch("src.pdf_parsers.mineru.parse_vlm_only", return_value="vlm output"):
        ok, _, _ = parse_pdfs_mod.parse_one(pdf, out, pass_mode="vlm")

    assert ok
    assert out.read_text() == "vlm output"
    assert not marker.exists()


def test_vlm_pass_keeps_marker_on_failure(
    parse_pdfs_mod: ModuleType,
    tmp_path: Path,
) -> None:
    """If VLM raises, marker must remain so a retry picks it up."""
    pdf = _make_pdf(tmp_path)
    out = pdf.parent / "pdf.txt"
    marker = pdf.parent / "pdf.needs_vlm"
    marker.write_text("")

    with patch("src.pdf_parsers.mineru.parse_vlm_only", side_effect=RuntimeError("oom")):
        ok, _, _ = parse_pdfs_mod.parse_one(pdf, out, pass_mode="vlm")

    assert not ok
    assert marker.exists()
    assert not out.exists()


# ---- auto pass (current behaviour preserved) -------------------------------


def test_auto_pass_falls_back_to_vlm_within_one_call(
    parse_pdfs_mod: ModuleType,
    tmp_path: Path,
) -> None:
    """Default `pass_mode='auto'` must keep the existing parse_pdf() behaviour:
    pipeline first, VLM if broken fonts, all in one parse_one call. No marker
    file is involved in the auto path."""
    pdf = _make_pdf(tmp_path)
    out = pdf.parent / "pdf.txt"
    marker = pdf.parent / "pdf.needs_vlm"

    with patch("src.pdf_parsers.mineru.parse_pdf", return_value="auto-resolved text"):
        ok, _, _ = parse_pdfs_mod.parse_one(pdf, out, pass_mode="auto")

    assert ok
    assert out.read_text() == "auto-resolved text"
    assert not marker.exists()
