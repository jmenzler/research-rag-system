"""Unit tests for ``src.chunking.render.content_list_to_markdown``."""

from __future__ import annotations

import textwrap


class TestRender:
    def test_render_text_with_levels(self) -> None:
        from src.chunking.render import content_list_to_markdown

        elements = [
            {"type": "text", "text_level": 1, "text": "Title"},
            {"type": "text", "text_level": 2, "text": "Section A"},
            {"type": "text", "text": "A plain paragraph."},
            {"type": "text", "text_level": 3, "text": "Subsection"},
        ]
        md = content_list_to_markdown(elements)
        assert md.startswith("# Title")
        assert "## Section A" in md
        assert "A plain paragraph." in md
        assert "### Subsection" in md

    def test_render_table_uses_html_table_to_markdown(self) -> None:
        from src.chunking.render import content_list_to_markdown

        elements = [
            {
                "type": "table",
                "table_body": (
                    "<table><tr><td>A</td><td>B</td></tr><tr><td>1</td><td>2</td></tr></table>"
                ),
                "table_caption": "Table 1: Results",
            },
        ]
        md = content_list_to_markdown(elements)
        assert "**Table 1: Results**" in md
        assert "| A | B |" in md
        assert "| 1 | 2 |" in md

    def test_render_list(self) -> None:
        from src.chunking.render import content_list_to_markdown

        elements = [
            {"type": "list", "text": "- Item one\n- Item two"},
        ]
        md = content_list_to_markdown(elements)
        assert "- Item one" in md
        assert "- Item two" in md

    def test_round_trip(self) -> None:
        from src.chunking.render import content_list_to_markdown
        from src.fetch.structured import extract_structured

        html = textwrap.dedent("""\
        <!DOCTYPE html>
        <html><head><title>Round Trip Test</title></head><body>
        <article>
        <h1>Pipeline Architecture</h1>
        <p>This document describes the data processing pipeline in detail
        covering the key design decisions and tradeoffs made during the
        development of the extraction and chunking system.</p>
        <h2>Data Flow</h2>
        <p>Sources enter through the fetch layer where they are classified
        and dispatched to the appropriate parser based on content type and
        available metadata signatures detected during initial inspection.</p>
        <table>
        <tr><td>Stage</td><td>Latency</td></tr>
        <tr><td>Fetch</td><td>2s</td></tr>
        <tr><td>Chunk</td><td>0.5s</td></tr>
        </table>
        </article>
        </body></html>""")

        elements = extract_structured(html)
        assert elements is not None

        md = content_list_to_markdown(elements)
        assert len(md) > 0
        assert "Pipeline Architecture" in md
        assert "Data Flow" in md
        assert "|" in md  # table marker

    def test_render_handles_empty_input(self) -> None:
        from src.chunking.render import content_list_to_markdown

        assert content_list_to_markdown([]) == ""

    def test_render_code_block(self) -> None:
        from src.chunking.render import content_list_to_markdown

        elements = [
            {"type": "code", "text": "def foo():\n    return 42"},
        ]
        md = content_list_to_markdown(elements)
        assert "```" in md
        assert "def foo():" in md
