"""Regression tests for inline image handling in the Scrapling spider pipeline.

Verifies that ``extract_web_text`` produces inline ``![caption](img:N)`` references
— not a bottom ``## Images`` section, no raw URLs, no leftover markers.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from src.fetch.html_preprocess import collect_images, extract_web_text

_WIKI_URL = "https://encyclopedia.example/calculus"


def _image_article(figures: list[str]) -> str:
    sections = []
    for index, figure in enumerate(figures):
        paragraph = (
            f"Section {index + 1} examines a different part of the illustrated method. "
            "The accompanying diagram connects the quantities discussed in the text. "
            "Readers can compare the labels against the explanation and follow each "
            "step independently. The article keeps figures near their discussion so "
            "that the relationship between the evidence and the argument stays clear. "
        ) * 4
        sections.append(f"<p>{paragraph}</p>{figure}")
    return (
        "<!DOCTYPE html><html><head><title>Illustrated methods</title></head><body>"
        "<main><article><div class='mw-parser-output'><h1>Illustrated methods</h1>"
        + "".join(sections)
        + "</div></article></main></body></html>"
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def wiki_html() -> str:
    captions = [
        "Gottfried Wilhelm Leibniz",
        "Augustin-Louis Cauchy (1789–1857)",
        "Bernhard Riemann (1826–1866)",
        "Henri Lebesgue (1875–1941)",
        "Leonhard Euler",
    ]
    figures = [
        f'<figure><img src="/images/portrait-{index}.png" width="400" '
        f'alt="Portrait {index}"><figcaption>{caption}</figcaption></figure>'
        for index, caption in enumerate(captions)
    ]
    figures.extend(
        f'<span class="mw-default-size"><a href="/images/Integral_{index}.svg">'
        f'<img src="/images/Integral_{index}.svg" width="400" '
        f'alt="Integral diagram {index}"></a></span>'
        for index in [1, 2, 3, 4, 5, 2]
    )
    figures.append(
        '<img class="mwe-math-fallback-image-inline" '
        'src="/images/mwe-math-fallback.svg" alt="formula">'
        '<img src="data:image/png;base64,AA==" alt="embedded image">'
    )
    return _image_article(figures)


@pytest.fixture(scope="module")
def wiki_text_and_registry(wiki_html: str) -> tuple[str, dict[int, dict[str, str]]]:
    """Run the full extraction pipeline on the Wikipedia page."""
    text, registry = extract_web_text(
        wiki_html.encode(),
        base_url=_WIKI_URL,
    )
    return text, registry


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCollectImages:
    """Unit tests for ``collect_images`` — registry construction."""

    def test_registry_not_empty(self, wiki_html: str) -> None:
        """At least 10 content images are found on the Integralrechnung page."""
        registry = collect_images(wiki_html, _WIKI_URL)
        assert len(registry) >= 10, f"Expected ≥10 images, got {len(registry)}"

    def test_skips_math_fallback_images(self, wiki_html: str) -> None:
        """Wikipedia math fallback images must not appear in the registry."""
        registry = collect_images(wiki_html, _WIKI_URL)
        for n, info in registry.items():
            assert "mwe-math-fallback" not in info["url"], (
                f"IMG_{n} URL contains mwe-math-fallback: {info['url']}"
            )

    def test_skips_data_uris(self, wiki_html: str) -> None:
        """data: URIs must not appear in the registry."""
        registry = collect_images(wiki_html, _WIKI_URL)
        for n, info in registry.items():
            assert not info["url"].startswith("data:"), f"IMG_{n} is a data: URI"

    def test_registry_has_required_fields(self, wiki_html: str) -> None:
        """Every registry entry must have 'url' and 'label' keys."""
        registry = collect_images(wiki_html, _WIKI_URL)
        for n, info in registry.items():
            assert "url" in info, f"IMG_{n} missing 'url'"
            assert "label" in info, f"IMG_{n} missing 'label'"
            assert isinstance(info["url"], str)
            assert isinstance(info["label"], str)

    def test_captions_from_figcaption(self, wiki_html: str) -> None:
        """Wikipedia figcaption text must produce non-generic labels."""
        registry = collect_images(wiki_html, _WIKI_URL)
        labels = {info["label"] for info in registry.values()}
        # These captions are from <figcaption> elements on the page
        assert "Gottfried Wilhelm Leibniz" in labels
        assert "Augustin-Louis Cauchy (1789–1857)" in labels
        assert "Bernhard Riemann (1826–1866)" in labels
        assert "Henri Lebesgue (1875–1941)" in labels


class TestExtractText:
    """Integration tests for ``extract_web_text`` — inline image references."""

    def test_all_registry_images_have_inline_refs(
        self,
        wiki_text_and_registry: tuple[str, dict[int, dict[str, str]]],
    ) -> None:
        """Every image in the registry must appear as ``![*](img:N)`` in text.

        Duplicate URLs (same image used twice on a page) are deduplicated —
        only the first registry index gets an inline ref.
        """
        text, registry = wiki_text_and_registry
        img_refs_in_text = {int(m.group(1)) for m in re.finditer(r"!\[[^\]]*\]\(img:(\d+)\)", text)}
        # Identify duplicate URLs (only the first index for each URL is expected)
        seen_urls: set[str] = set()
        duplicate_indices: set[int] = set()
        for n, info in registry.items():
            if info["url"] in seen_urls:
                duplicate_indices.add(n)
            else:
                seen_urls.add(info["url"])
        missing = set(registry.keys()) - img_refs_in_text - duplicate_indices
        assert not missing, (
            f"Registry entries without inline refs: {missing}\n"
            f"Duplicates (expected): {duplicate_indices}\n"
            f"Registry: {list(registry.keys())}\n"
            f"Inline refs found: {sorted(img_refs_in_text)}"
        )

    def test_no_images_section(
        self, wiki_text_and_registry: tuple[str, dict[int, dict[str, str]]]
    ) -> None:
        """The legacy ``## Images`` appendix section must NOT be present."""
        text, _registry = wiki_text_and_registry
        assert "## Images" not in text, (
            "Found '## Images' section — images must be inline, not appended"
        )

    def test_no_raw_image_urls(
        self, wiki_text_and_registry: tuple[str, dict[int, dict[str, str]]]
    ) -> None:
        """No ``![](http...)`` raw URLs must leak into the embedding text."""
        text, _registry = wiki_text_and_registry
        raw_urls = re.findall(r"!\[[^\]]*\]\(http[^)]+\)", text)
        assert len(raw_urls) == 0, f"Found {len(raw_urls)} raw URL image refs: {raw_urls[:3]}"

    def test_no_leftover_markers(
        self, wiki_text_and_registry: tuple[str, dict[int, dict[str, str]]]
    ) -> None:
        """No ``__IMG_N__`` markers must remain in the final text."""
        text, _registry = wiki_text_and_registry
        markers = re.findall(r"__IMG_\d+__", text)
        assert len(markers) == 0, f"Found {len(markers)} leftover markers: {markers[:5]}"

    def test_images_inline_not_only_at_end(
        self,
        wiki_text_and_registry: tuple[str, dict[int, dict[str, str]]],
    ) -> None:
        """Image references must appear throughout the text, not just at the end."""
        text, _registry = wiki_text_and_registry
        refs = list(re.finditer(r"!\[[^\]]*\]\(img:\d+\)", text))
        assert len(refs) >= 10

        # At least one image ref must appear in the first 30 % of the text
        # (proving images aren't all dumped at the bottom).
        early_cutoff = len(text) * 3 // 10
        early_refs = [r for r in refs if r.start() < early_cutoff]
        assert len(early_refs) >= 1, (
            "All image refs are in the bottom 70 % of the text — "
            "expected at least one in the first 30 %"
        )

    def test_caption_labels_appear_in_text(
        self,
        wiki_text_and_registry: tuple[str, dict[int, dict[str, str]]],
    ) -> None:
        """Captions extracted from figcaption must appear as image labels."""
        text, _registry = wiki_text_and_registry
        assert "![Gottfried Wilhelm Leibniz](img:" in text
        assert "![Augustin-Louis Cauchy (1789–1857)](img:" in text
        assert "![Bernhard Riemann (1826–1866)](img:" in text
        assert "![Henri Lebesgue (1875–1941)](img:" in text
        assert "![Leonhard Euler](img:" in text

    def test_graph_images_are_inline(
        self,
        wiki_text_and_registry: tuple[str, dict[int, dict[str, str]]],
    ) -> None:
        """Integral graph SVGs (span-wrapped) must appear inline.

        Integral_2.svg appears twice on the page — only the first occurrence
        (lowest index) gets an inline ref.
        """
        text, registry = wiki_text_and_registry
        seen_urls: set[str] = set()
        for n, info in registry.items():
            if "Integral_" not in info["url"]:
                continue
            # Skip duplicate URLs (only the first index is inlined)
            if info["url"] in seen_urls:
                continue
            seen_urls.add(info["url"])
            label = info["label"]
            assert f"![{label}](img:" in text, (
                f"Graph image '{label}' (IMG_{n}) not found inline in text"
            )

    def test_figures_json_roundtrip(
        self,
        wiki_text_and_registry: tuple[str, dict[int, dict[str, str]]],
        tmp_path: Path,
    ) -> None:
        """The registry can be written as figures.json and read back."""
        _text, registry = wiki_text_and_registry
        fig_path = tmp_path / "figures.json"
        fig_path.write_text(json.dumps(registry, indent=2))
        reread = json.loads(fig_path.read_text())
        assert len(reread) == len(registry)
        # Keys are stringified ints in JSON
        for key in registry:
            assert str(key) in reread
            assert reread[str(key)]["url"] == registry[key]["url"]
            assert reread[str(key)]["label"] == registry[key]["label"]


class TestNonWikipedia:
    """Smoke tests on non-Wikipedia pages to catch regressions in the
    all-marker approach."""

    @pytest.mark.parametrize(
        "label,url,min_images",
        [
            ("blog", "https://blog.example/illustrated-methods", 10),
        ],
    )
    def test_blog_images_inline(self, label: str, url: str, min_images: int) -> None:
        """Blog pages with <figure> images produce inline refs."""
        figures = [
            f'<figure><img src="/images/diagram-{index}.png" width="400" '
            f'alt="Diagram {index}"><figcaption>Method {index}</figcaption></figure>'
            for index in range(10)
        ]
        html = _image_article(figures)
        text, registry = extract_web_text(html.encode(), base_url=url)
        assert len(registry) >= min_images, (
            f"Expected ≥{min_images} images on {label}, got {len(registry)}"
        )
        img_refs = re.findall(r"!\[[^\]]*\]\(img:\d+\)", text)
        assert len(img_refs) == len(registry), (
            f"Registry has {len(registry)} entries but text has {len(img_refs)} "
            f"inline refs on {label}"
        )
        assert "## Images" not in text
        assert len(re.findall(r"__IMG_\d+__", text)) == 0
        assert len(re.findall(r"!\[[^\]]*\]\(http[^)]+\)", text)) == 0


# ---------------------------------------------------------------------------
# HTML fixture tests — citation + image survival
# ---------------------------------------------------------------------------


_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "html"


def _load_fixture(name: str) -> str:
    return (_FIXTURES / name).read_text()


class TestWikipediaCitations:
    """Wikipedia-style ``<sup>`` bracket citations survive extraction."""

    @pytest.fixture(scope="class")
    def result(self) -> tuple[str, dict[int, dict[str, str]]]:
        html = _load_fixture("wikipedia_citations.html")
        return extract_web_text(html.encode(), base_url="https://en.wikipedia.org/")

    def test_numeric_citations_survive(self, result: tuple[str, dict[int, dict[str, str]]]) -> None:
        """All ``[N]`` superscript citations must appear inline."""
        text, _registry = result
        found = {int(m.group(1)) for m in re.finditer(r"\[(\d+)\]", text)}
        expected = {1, 2, 3, 4, 5, 7, 8, 9}  # [6] is in figcaption reference
        missing = expected - found
        assert not missing, f"Missing numeric citations: {missing}"

    def test_citations_are_inline_not_block(
        self, result: tuple[str, dict[int, dict[str, str]]]
    ) -> None:
        """Citations like ``.[1]`` must be fused to sentence text, not on own line."""
        text, _registry = result
        assert re.search(r"\.\[1\]", text), "Citation [1] not fused to preceding sentence"

    def test_multiple_citations(self, result: tuple[str, dict[int, dict[str, str]]]) -> None:
        """Adjacent citations ``.[2][3]`` must both survive."""
        text, _registry = result
        assert "[2][3]" in text, "Adjacent citations [2][3] missing"

    def test_citation_needed_survives(self, result: tuple[str, dict[int, dict[str, str]]]) -> None:
        """Editorial ``[citation needed]`` tags must survive."""
        text, _registry = result
        assert "[citation needed]" in text.lower()

    def test_original_research_survives(
        self, result: tuple[str, dict[int, dict[str, str]]]
    ) -> None:
        """Editorial ``[original research?]`` tags must survive."""
        text, _registry = result
        assert "[original research?]" in text.lower()

    def test_references_section_absent(self, result: tuple[str, dict[int, dict[str, str]]]) -> None:
        """The ``## References`` section must not survive trafilatura extraction."""
        text, _registry = result
        assert "References" not in text, "Reference section leaked into extracted text"

    def test_no_raw_image_urls(self, result: tuple[str, dict[int, dict[str, str]]]) -> None:
        text, _registry = result
        raw = re.findall(r"!\[[^\]]*\]\(http[^)]+\)", text)
        assert len(raw) == 0, f"Raw URLs leaked: {raw}"


class TestAcademicCitations:
    """Academic author-year citations and images with citation labels."""

    @pytest.fixture(scope="class")
    def result(self) -> tuple[str, dict[int, dict[str, str]]]:
        html = _load_fixture("academic_citations.html")
        return extract_web_text(html.encode(), base_url="https://example.com/")

    def test_year_bracket_citations(self, result: tuple[str, dict[int, dict[str, str]]]) -> None:
        """``[2024]`` style year citations must survive."""
        text, _registry = result
        assert "[2024]" in text
        assert "Brown and Wilson [2018]" in text

    def test_paren_citations(self, result: tuple[str, dict[int, dict[str, str]]]) -> None:
        """``(Jones, 2020)`` parenthetical citations must survive."""
        text, _registry = result
        assert "(Jones, 2020)" in text

    def test_multiple_adjacent_citations(
        self, result: tuple[str, dict[int, dict[str, str]]]
    ) -> None:
        """Adjacent ``[Lee, 2019][Park, 2021][Kim, 2023]`` must all survive."""
        text, _registry = result
        assert "[Lee, 2019]" in text
        assert "[Park, 2021]" in text
        assert "[Kim, 2023]" in text

    def test_images_have_escaped_alt_text(
        self,
        result: tuple[str, dict[int, dict[str, str]]],
    ) -> None:
        """Image alt text containing ``[YYYY]`` must escape brackets so
        markdown parsing doesn't break."""
        text, _registry = result
        # img:N refs must be findable with a simple regex
        img_refs = re.findall(r"img:(\d+)", text)
        assert len(img_refs) >= 1, "No img:N references found"
        # Brackets in alt text must be escaped
        assert "&#91;2024&#93;" in text, "Brackets in citation labels not escaped"
        # Raw brackets must NOT appear inside alt text (would break markdown)
        raw_bracket_in_alt = re.findall(r"!\[.*\[\d{4}\].*\]\(img:", text)
        assert len(raw_bracket_in_alt) == 0, (
            f"Unescaped citation brackets in image alt text: {raw_bracket_in_alt}"
        )

    def test_no_raw_image_urls(self, result: tuple[str, dict[int, dict[str, str]]]) -> None:
        text, _registry = result
        raw = re.findall(r"!\[[^\]]*\]\(http[^)]+\)", text)
        assert len(raw) == 0, f"Raw URLs leaked: {raw}"

    def test_all_registry_entries_referenced(
        self,
        result: tuple[str, dict[int, dict[str, str]]],
    ) -> None:
        text, registry = result
        img_ns = {int(m.group(1)) for m in re.finditer(r"img:(\d+)", text)}
        # Both images (same label, different URLs) must be referenced
        missing = set(registry.keys()) - img_ns
        assert not missing, f"Registry entries not referenced: {missing}"


class TestMixedContent:
    """Market microstructure page with mixed citation styles + images."""

    @pytest.fixture(scope="class")
    def result(self) -> tuple[str, dict[int, dict[str, str]]]:
        html = _load_fixture("mixed_content.html")
        return extract_web_text(html.encode(), base_url="https://example.com/")

    def test_paren_citations(self, result: tuple[str, dict[int, dict[str, str]]]) -> None:
        """``(Author, YYYY)`` citations must survive."""
        text, _registry = result
        assert "(Easley et al., 2012)" in text
        assert "(Menkveld, 2013)" in text
        assert "(Biais et al., 2015)" in text

    def test_numeric_citations(self, result: tuple[str, dict[int, dict[str, str]]]) -> None:
        """``[1]`` numeric citations must survive."""
        text, _registry = result
        assert "[1]" in text
        assert "[2]" in text

    def test_images_are_referenced(self, result: tuple[str, dict[int, dict[str, str]]]) -> None:
        text, registry = result
        img_ns = {int(m.group(1)) for m in re.finditer(r"img:(\d+)", text)}
        assert img_ns == set(registry.keys()), (
            f"Mismatch: registry={list(registry.keys())} text={sorted(img_ns)}"
        )

    def test_no_raw_image_urls(self, result: tuple[str, dict[int, dict[str, str]]]) -> None:
        text, _registry = result
        raw = re.findall(r"!\[[^\]]*\]\(http[^)]+\)", text)
        assert len(raw) == 0, f"Raw URLs leaked: {raw}"
