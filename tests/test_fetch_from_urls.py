"""Unit tests for src.fetch.from_urls.

Exercises the URL-file parser, title derivation, and src-dict builder. The
spider invocation in ``_run_spider`` is integration-level (writes to disk,
calls into Scrapling) and is exercised end-to-end via the smoke-test
ingestion path rather than here.
"""

from __future__ import annotations

from pathlib import Path

from src.fetch.classify import classify
from src.fetch.from_urls import _derive_title, build_sources, parse_url_file


def test_parse_url_file_basic(tmp_path: Path) -> None:
    """Strips blank lines and ``#`` comments; preserves order."""
    f = tmp_path / "urls.txt"
    f.write_text(
        "# arxiv methods papers\n"
        "\n"
        "https://arxiv.org/abs/2508.02435\n"
        "https://arxiv.org/pdf/2510.14278\n"
        "# trailing comment\n",
        encoding="utf-8",
    )
    pairs = parse_url_file(f)
    assert pairs == [
        ("", "https://arxiv.org/abs/2508.02435"),
        ("", "https://arxiv.org/pdf/2510.14278"),
    ]


def test_parse_url_file_title_override(tmp_path: Path) -> None:
    """``title || url`` form yields the override verbatim."""
    f = tmp_path / "urls.txt"
    f.write_text(
        "Custom Title || https://example.com/post\n  spaced || https://example.com/other  \n",
        encoding="utf-8",
    )
    pairs = parse_url_file(f)
    assert pairs == [
        ("Custom Title", "https://example.com/post"),
        ("spaced", "https://example.com/other"),
    ]


def test_parse_url_file_ignores_empty_url(tmp_path: Path) -> None:
    """``title ||`` with no URL is skipped."""
    f = tmp_path / "urls.txt"
    f.write_text("Title ||\nhttps://example.com\n", encoding="utf-8")
    pairs = parse_url_file(f)
    assert pairs == [("", "https://example.com")]


def test_derive_title_arxiv_id() -> None:
    """Arxiv URLs surface the ID as the initial title."""
    assert _derive_title("https://arxiv.org/abs/2508.02435", "") == "arxiv:2508.02435"
    assert _derive_title("https://arxiv.org/pdf/2510.14278v1", "") == "arxiv:2510.14278"


def test_derive_title_override_wins() -> None:
    """Non-empty override takes precedence over auto-derived title."""
    assert _derive_title("https://arxiv.org/abs/2508.02435", "Custom") == "Custom"


def test_derive_title_generic_pdf() -> None:
    """Generic PDF URLs use the filename stem (without ``.pdf``)."""
    assert _derive_title("https://example.com/papers/foo.pdf", "") == "foo"


def test_derive_title_generic_web() -> None:
    """Web URLs use the last path segment."""
    assert _derive_title("https://example.com/blog/post-slug", "") == "post-slug"


def test_derive_title_no_path() -> None:
    """URLs without a path fall back to the netloc."""
    assert _derive_title("https://example.com", "") == "example.com"


def test_build_sources_shape() -> None:
    """Each src dict has the fields classify() and the spider read."""
    pairs = [
        ("", "https://arxiv.org/abs/2508.02435"),
        ("Custom", "https://example.com/post"),
    ]
    sources = build_sources(pairs)
    assert len(sources) == 2
    assert sources[0]["index"] == 0
    assert sources[1]["index"] == 1
    assert sources[0]["url"] == "https://arxiv.org/abs/2508.02435"
    assert sources[1]["title"] == "Custom"
    for src in sources:
        assert src["type"] == "SourceType.UNKNOWN"
        assert "title" in src
        assert "url" in src


def test_build_sources_classify_routing() -> None:
    """classify() routes built sources to the correct tier from the URL host."""
    pairs = [
        ("", "https://arxiv.org/abs/2508.02435"),
        ("", "https://example.com/paper.pdf"),
        ("", "https://example.com/blog/post"),
    ]
    sources = build_sources(pairs)
    tiers = [classify(s) for s in sources]
    assert tiers[0] == "T0_arxiv"
    assert tiers[1] == "T3_generic_pdf"
    assert tiers[2] == "T5_web_blog"
