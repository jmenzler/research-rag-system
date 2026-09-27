"""Unit tests for scripts/refetch_metadata.py — candidate identification only.

The HTTP loop is integration-level (touches the network); not unit-tested.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def test_is_weak_empty_authors() -> None:
    from refetch_metadata import _is_weak_attribution

    assert _is_weak_attribution({"authors": []}) is True
    assert _is_weak_attribution({}) is True


def test_is_weak_single_token_brand() -> None:
    from refetch_metadata import _is_weak_attribution

    assert _is_weak_attribution({"authors": ["QuantStart"]}) is True
    assert _is_weak_attribution({"authors": ["Mergify"]}) is True
    assert _is_weak_attribution({"authors": ["GitHub"]}) is True


def test_is_weak_two_word_title_case() -> None:
    """'Dean Markwick' could be real OR sitename — refetch confirms."""
    from refetch_metadata import _is_weak_attribution

    assert _is_weak_attribution({"authors": ["Dean Markwick"]}) is True


def test_is_strong_multi_author() -> None:
    from refetch_metadata import _is_weak_attribution

    md = {"authors": ["Alice Smith", "Bob Jones", "Carol Lee"]}
    assert _is_weak_attribution(md) is False


def test_is_candidate_skips_pdf_source(tmp_path: Path) -> None:
    """A doc dir with pdf.txt is NOT a refetch candidate."""
    from refetch_metadata import _is_candidate

    doc = tmp_path / "pdfdoc"
    doc.mkdir()
    (doc / "pdf.txt").write_text("body", encoding="utf-8")
    meta = {"url": "https://example.com/paper.pdf", "metadata": {"authors": []}}
    assert _is_candidate(meta, doc) is False


def test_is_candidate_skips_platform_blocklist(tmp_path: Path) -> None:
    """Wikipedia / Reddit / HN won't benefit from refetch — already attributed."""
    from refetch_metadata import _is_candidate

    doc = tmp_path / "wiki"
    doc.mkdir()
    (doc / "web.txt").write_text("body", encoding="utf-8")
    meta = {
        "url": "https://en.wikipedia.org/wiki/Kelly_criterion",
        "metadata": {"authors": []},
    }
    assert _is_candidate(meta, doc) is False


def test_is_candidate_skips_already_harvested(tmp_path: Path) -> None:
    """Source with page_metadata already present is skipped."""
    from refetch_metadata import _is_candidate

    doc = tmp_path / "harvested"
    doc.mkdir()
    (doc / "web.txt").write_text("body", encoding="utf-8")
    meta = {
        "url": "https://example.com/x",
        "metadata": {"authors": []},
        "page_metadata": {"authors": ["A"]},
    }
    assert _is_candidate(meta, doc) is False


def test_is_candidate_picks_web_source_with_weak_attribution(tmp_path: Path) -> None:
    """Web source + weak attribution + no page_metadata = refetch candidate."""
    from refetch_metadata import _is_candidate

    doc = tmp_path / "blog"
    doc.mkdir()
    (doc / "web.txt").write_text("body", encoding="utf-8")
    meta = {
        "url": "https://someblog.example.com/post",
        "metadata": {"authors": ["Someblog"]},
    }
    assert _is_candidate(meta, doc) is True
