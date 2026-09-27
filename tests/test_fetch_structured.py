"""Unit tests for ``src.fetch.structured.extract_structured``.

Hand-rolled HTML fixtures, no network.
"""

from __future__ import annotations

import textwrap

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _html(heading: str | None = None, body: str = "") -> str:
    """Minimal HTML document with enough content to pass trafilatura's boilerplate filter."""
    title = heading or "Test Document"
    return textwrap.dedent(f"""\
    <!DOCTYPE html>
    <html><head><title>{title}</title></head><body>
    <article>
    {body}
    <p>Additional content to ensure trafilatura does not filter this as boilerplate
    because it requires a minimum amount of substantive text before it considers
    a document worth extracting.</p>
    </article>
    </body></html>""")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEmptyHtml:
    def test_returns_none_on_empty_html(self) -> None:
        from src.fetch.structured import extract_structured

        assert extract_structured("") is None
        assert extract_structured("<html></html>") is None


class TestHeadings:
    def test_h1_h2_h3_become_text_with_levels(self) -> None:
        from src.fetch.structured import extract_structured

        body = (
            "<h1>Main Title</h1>"
            "<p>Intro paragraph with enough text here to pass boilerplate filters "
            "and ensure trafilatura considers this document substantive.</p>"
            "<h2>First Section</h2>"
            "<p>Section one content with more descriptive text to fill things out "
            "so the extractor has enough material to work with properly.</p>"
            "<h3>Sub Section</h3>"
            "<p>Sub section content needs to also have enough words and substance "
            "so that trafilatura does not throw this paragraph away as noise.</p>"
            "<h2>Second Section</h2>"
            "<p>Second section with meaningful content that discusses important "
            "topics and provides sufficient text for the extraction algorithms.</p>"
        )
        elements = extract_structured(_html(body=body))
        assert elements is not None

        levels = [(e.get("text_level"), e.get("text")) for e in elements if e.get("text_level")]
        assert levels[0] == (1, "Main Title")
        # First h2
        assert (2, "First Section") in levels
        # h3
        assert (3, "Sub Section") in levels


class TestParagraphs:
    def test_paragraphs_become_text_elements(self) -> None:
        from src.fetch.structured import extract_structured

        body = (
            "<p>First paragraph with meaningful text that discusses the topic "
            "in detail and provides enough substance for the content extractor "
            "to recognize this as a valid piece of writing worth keeping.</p>"
            "<p>Second paragraph that continues the discussion with additional "
            "information and context to flesh out the document further.</p>"
        )
        elements = extract_structured(_html(body=body))
        assert elements is not None

        texts = [e for e in elements if e["type"] == "text" and "text_level" not in e]
        assert len(texts) >= 2


class TestLists:
    def test_lists_become_list_elements(self) -> None:
        from src.fetch.structured import extract_structured

        body = (
            "<p>Context paragraph with enough text to establish the document as "
            "meaningful content that trafilatura should extract properly.</p>"
            "<ul><li>First item</li><li>Second item</li><li>Third item</li></ul>"
        )
        elements = extract_structured(_html(body=body))
        assert elements is not None

        lists = [e for e in elements if e["type"] == "list"]
        assert len(lists) >= 1
        text = lists[0]["text"]
        assert "- First item" in text
        assert "- Second item" in text
        assert "- Third item" in text


class TestTables:
    def test_table_emerges_with_html_body(self) -> None:
        from src.fetch.structured import extract_structured

        body = (
            "<p>Context paragraph with sufficient text content to ensure that "
            "trafilatura treats this document as a substantive extract rather "
            "than boilerplate to discard during the filtering process.</p>"
            "<table>"
            "<tr><th>Name</th><th>Value</th></tr>"
            "<tr><td>Alpha</td><td>1.0</td></tr>"
            "<tr><td>Beta</td><td>2.5</td></tr>"
            "</table>"
        )
        elements = extract_structured(_html(body=body))
        assert elements is not None

        tables = [e for e in elements if e["type"] == "table"]
        assert len(tables) >= 1
        table = tables[0]
        assert "table_body" in table
        assert "<table>" in table["table_body"]
        assert "<td>Alpha</td>" in table["table_body"]

    def test_table_caption_detection(self) -> None:
        from src.fetch.structured import extract_structured

        body = (
            "<p>Context paragraph with enough meaningful text for extraction.</p>"
            "<p>Table 3: Performance Results</p>"
            "<table>"
            "<tr><td>Metric</td><td>Score</td></tr>"
            "<tr><td>Accuracy</td><td>0.95</td></tr>"
            "</table>"
        )
        elements = extract_structured(_html(body=body))
        assert elements is not None

        tables = [e for e in elements if e["type"] == "table"]
        assert len(tables) >= 1
        assert tables[0]["table_caption"] == "Table 3: Performance Results"

    def test_table_no_preceding_caption(self) -> None:
        from src.fetch.structured import extract_structured

        body = (
            "<p>Context paragraph with enough meaningful text for extraction.</p>"
            "<table>"
            "<tr><td>Metric</td><td>Score</td></tr>"
            "</table>"
        )
        elements = extract_structured(_html(body=body))
        assert elements is not None

        tables = [e for e in elements if e["type"] == "table"]
        assert len(tables) >= 1
        assert tables[0]["table_caption"] == ""

    def test_body_hash_matches_chunker_path(self) -> None:
        from src.chunking.chunker import _table_body_hash
        from src.fetch.structured import extract_structured

        body = (
            "<p>Context paragraph with enough meaningful text for extraction "
            "so that trafilatura considers this document substantive enough "
            "to process and include in its output without filtering it out.</p>"
            "<table>"
            "<tr><td>Name</td><td>Value</td></tr>"
            "<tr><td>Alpha</td><td>1.0</td></tr>"
            "</table>"
        )
        elements = extract_structured(_html(body=body))
        assert elements is not None

        tables = [e for e in elements if e["type"] == "table"]
        assert len(tables) >= 1
        h = _table_body_hash(tables[0])
        assert isinstance(h, str) and len(h) == 64


class TestNestedFormatting:
    def test_handles_nested_formatting_in_cells(self) -> None:
        from src.fetch.structured import extract_structured

        body = (
            "<p>Context paragraph with enough meaningful text for extraction "
            "to ensure that the document passes all of the boilerplate filters "
            "that trafilatura applies before it begins extracting content.</p>"
            "<table>"
            "<tr><td>Simple cell</td><td>Cell with <em>emphasis</em> inside</td></tr>"
            "<tr><td>Another <strong>bold</strong> word</td><td>Plain</td></tr>"
            "</table>"
        )
        elements = extract_structured(_html(body=body))
        assert elements is not None

        tables = [e for e in elements if e["type"] == "table"]
        assert len(tables) >= 1
        html_body = tables[0]["table_body"]
        assert "<em>" not in html_body
        assert "<strong>" not in html_body
        assert "emphasis" in html_body

    def test_inline_formatting_flattened_in_paragraph(self) -> None:
        from src.fetch.structured import extract_structured

        body = (
            "<p>A paragraph with <strong>bold text</strong> and "
            "<em>italic text</em> inside it that provides a meaningful "
            "amount of content for proper extraction handling.</p>"
            "<p>Another paragraph with <code>inline code</code> as well "
            "for additional testing of the formatting flattening behavior.</p>"
        )
        elements = extract_structured(_html(body=body))
        assert elements is not None

        texts = [e for e in elements if e["type"] == "text" and "text_level" not in e]
        # Find the paragraph with bold/italic
        bold_para = next((t for t in texts if "bold text" in t["text"]), None)
        assert bold_para is not None
        assert "<strong>" not in bold_para["text"]


class TestTitleSynthesis:
    def test_synthesizes_title_from_page_metadata(self) -> None:
        from src.fetch.structured import extract_structured

        body = (
            "<p>Document body without any heading elements at all but with "
            "enough substantive content to pass the minimum text threshold "
            "that trafilatura uses to distinguish real articles from noise "
            "and navigational boilerplate on web pages.</p>"
        )
        elements = extract_structured(
            _html(body=body),
            page_metadata={"title": "Synthesized Title"},
        )
        assert elements is not None

        titles = [e for e in elements if e.get("text_level") == 1]
        assert len(titles) == 1
        assert titles[0]["text"] == "Synthesized Title"

    def test_no_synthesis_when_h1_exists(self) -> None:
        from src.fetch.structured import extract_structured

        body = (
            "<h1>Real Title</h1>"
            "<p>Paragraph with enough content to ensure proper extraction "
            "from trafilatura by providing a substantive amount of text that "
            "will not be classified as boilerplate content.</p>"
        )
        elements = extract_structured(
            _html(body=body),
            page_metadata={"title": "Should Not Appear"},
        )
        assert elements is not None

        titles = [e for e in elements if e.get("text_level") == 1]
        assert len(titles) == 1
        assert titles[0]["text"] == "Real Title"
