# long-ok-file
"""End-to-end ingest tests with real chunker + sqlite, mocked I/O boundaries.

These tests pin the WIRING that the unit tests can't catch alone:

  * ``ingest_file`` plumbing: chunker → ParentChunk store → embed batch →
    Milvus insert call sequence.
  * ``_resolve_doc_representative`` — a content_list.json path swaps to
    source.pdf for the source_file label, but the structural chunker still
    reads from path.parent / content_list.json.
  * Idempotency cache: re-ingesting the same file in the same notebook
    short-circuits before the embed call (cache hit).
  * Source-of-truth check: the ParentChunk rows landing in sqlite have the
    correct notebook/source_file/modality threaded through from caller args.

Boundaries mocked:
  * ``_embed_text_batch`` (returns deterministic zero vectors, dim=4096)
  * ``get_client`` (Milvus client — captures insert calls without network)
  * ``_build_gemini_client`` (returns a MagicMock; no API key needed)

Boundaries NOT mocked:
  * Real ``cl100k_base`` tiktoken encoder.
  * Real ``src.chunking`` structural chunker.
  * Real sqlite at tmp_path/parents.sqlite.
  * Real ``_strip_tail_sections`` / ``_load_source_meta``.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import tiktoken

from src.ingest.chunker import _get_encoder
from src.ingest.ingest import ingest_file
from src.ingest.storage import _open_parents_db

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_db(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[sqlite3.Connection]:
    """Open an isolated parents.sqlite and point config.PARENTS_DB at it."""
    db_path = tmp_path / "parents.sqlite"
    monkeypatch.setattr("src.config.PARENTS_DB", str(db_path))
    conn = _open_parents_db()
    yield conn
    conn.close()


@pytest.fixture(scope="module")
def enc() -> tiktoken.Encoding:
    """Real tiktoken encoder. Module-scoped — load is hot."""
    return _get_encoder()


@pytest.fixture
def doc_dir(tmp_path: Path) -> Path:
    """A doc directory with a realistic content_list.json + source.pdf stub.

    The structural chunker reads content_list.json; source.pdf is just a
    placeholder so _resolve_doc_representative finds something to label
    source_file with.
    """
    d = tmp_path / "ondemand_test_doc"
    d.mkdir()
    # source.pdf is opaque content (the structural chunker doesn't read it,
    # but file_hash needs SOMETHING to digest, and _resolve_doc_representative
    # swaps to it for the source_file label).
    (d / "source.pdf").write_bytes(b"PDF-stub-bytes-deterministic")

    # Minimal content_list.json — title + 3 body paragraphs across 2 pages.
    # Enough material to produce >=1 parent + >=1 child but small enough
    # to keep the test fast.
    content_list = [
        {"type": "text", "text": "Test Doc Title", "text_level": 1, "page_idx": 0},
        {
            "type": "text",
            "text": (
                "First body paragraph with enough words to register as substantive "
                "content under the structural chunker. This needs to be long enough "
                "that the chunker doesn't skip it as too-short, and contains some "
                "domain-shaped vocabulary like avellaneda stoikov reservation price "
                "to make it look plausible to the test."
            ),
            "page_idx": 0,
        },
        {
            "type": "text",
            "text": (
                "Second body paragraph extending the first. Adds more tokens so "
                "we definitely get past the parent-merge floor and emit a real "
                "parent chunk into sqlite. Inventory risk drives the spread."
            ),
            "page_idx": 1,
        },
    ]
    (d / "content_list.json").write_text(json.dumps(content_list))

    # meta.json is best-effort — supplies short_cite for doc-prefix.
    (d / "meta.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "short_cite": "Test 2026",
                    "year": 2026,
                    "authors": ["Test Author"],
                    "confidence": "high",
                },
            }
        )
    )

    return d


# ---------------------------------------------------------------------------
# Patches — applied uniformly across E2E tests
# ---------------------------------------------------------------------------


def _patch_io(test_fn: Callable[..., None]) -> Callable[..., None]:
    """Decorator stack for the three I/O boundaries — keeps tests focused on logic.

    The ordering of @patch decorators is reverse-application: the bottom one
    fills the first positional arg.

    IMPORTANT: patch where the symbol is *used*, not where it's defined.
    ``ingest.py`` imports ``_embed_text_batch`` and ``_insert_children`` into
    its own namespace at top, so we must patch ``src.ingest.ingest._embed_text_batch``
    (not ``src.ingest.embed._embed_text_batch``) to actually intercept the call.
    Same goes for ``_insert_children`` (the Milvus side-effect).

    ``existing_child_ids`` is stubbed via ``new=`` (no injected arg) to return an
    empty set — the pre-embed gate then treats every chunk as new, so these
    tests exercise the full embed+insert path without a live Milvus.
    """
    return patch("src.ingest.ingest.existing_child_ids", new=lambda _collection, _ids: set())(
        patch("src.ingest.ingest._build_gemini_client")(
            patch("src.ingest.ingest._embed_text_batch")(
                patch("src.ingest.ingest._insert_children")(
                    test_fn,
                )
            )
        )
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@_patch_io
def test_ingest_file_e2e_writes_parents_to_sqlite(
    insert_children_mock: MagicMock,
    embed_mock: MagicMock,
    build_gemini_mock: MagicMock,
    doc_dir: Path,
    isolated_db: sqlite3.Connection,
    enc: tiktoken.Encoding,
) -> None:
    """End-to-end: ingest_file → real chunker → real sqlite → mocked Milvus.

    Asserts the parent rows actually landed in sqlite with the right notebook,
    source_file, and modality.
    """
    # Mocks: embed returns deterministic zero vectors at dim=4096.
    embed_mock.side_effect = lambda _client, texts: [[0.0] * 4096 for _ in texts]
    build_gemini_mock.return_value = MagicMock()
    # insert_children_mock just absorbs the call — we'll inspect its call_args.

    summary = ingest_file(
        path=doc_dir / "content_list.json",
        notebook="trading",
        collection="trading",
        gemini_client=MagicMock(),
        db_conn=isolated_db,
        enc=enc,
    )

    assert summary["skipped"] is False, f"unexpectedly skipped: {summary!r}"
    n_parents = summary["n_parents"]
    n_children = summary["n_children"]
    assert isinstance(n_parents, int) and n_parents >= 1
    assert isinstance(n_children, int) and n_children >= 1

    # sqlite truth check — every parent claimed in the summary must exist.
    rows = isolated_db.execute(
        "SELECT id, notebook, source_file, modality FROM parents",
    ).fetchall()
    assert len(rows) == n_parents, f"summary claimed {n_parents} parents, sqlite has {len(rows)}"
    for _id, nb, src, mod in rows:
        assert nb == "trading"
        assert src.endswith("source.pdf"), (
            f"source_file should reflect _resolve_doc_representative swap: got {src}"
        )
        assert mod in ("pdf", "text")  # structural chunker per-element modality

    # _insert_children was called at least once.
    assert insert_children_mock.called
    # Args: (collection, children, vectors) positionally.
    args = insert_children_mock.call_args.args
    assert args[0] == "trading", f"collection arg wrong: {args[0]!r}"
    children = args[1]
    vectors = args[2]
    assert len(children) == n_children, (
        f"insert called with {len(children)} children, summary said {n_children}"
    )
    assert len(vectors) == len(children), "vectors/children length mismatch"
    assert all(len(v) == 4096 for v in vectors), "embed vectors at wrong dim"


@_patch_io
def test_ingest_file_e2e_idempotent_on_second_call(
    insert_children_mock: MagicMock,
    embed_mock: MagicMock,
    build_gemini_mock: MagicMock,
    doc_dir: Path,
    isolated_db: sqlite3.Connection,
    enc: tiktoken.Encoding,
) -> None:
    """Second call with same file + notebook is a no-op (cache hit).

    Critical for parallel ingest restarts and for re-running ingest after a
    partial failure — we should never double-embed.
    """
    embed_mock.side_effect = lambda _client, texts: [[0.0] * 4096 for _ in texts]
    build_gemini_mock.return_value = MagicMock()
    # insert_children_mock just absorbs the call — we'll inspect its call_args.

    # First ingest: embed + insert called.
    first = ingest_file(
        path=doc_dir / "content_list.json",
        notebook="trading",
        collection="trading",
        gemini_client=MagicMock(),
        db_conn=isolated_db,
        enc=enc,
    )
    assert first["skipped"] is False
    embed_calls_after_first = embed_mock.call_count
    insert_calls_after_first = insert_children_mock.call_count
    assert embed_calls_after_first >= 1
    assert insert_calls_after_first >= 1

    # Second ingest: file_hash already in sources → must short-circuit.
    second = ingest_file(
        path=doc_dir / "content_list.json",
        notebook="trading",
        collection="trading",
        gemini_client=MagicMock(),
        db_conn=isolated_db,
        enc=enc,
    )
    assert second["skipped"] is True
    assert second["reason"] == "already_ingested"
    assert second["n_parents"] == 0
    assert second["n_children"] == 0
    # Critical: no additional embed or insert calls.
    assert embed_mock.call_count == embed_calls_after_first, (
        "second ingest re-embedded — idempotency broken"
    )
    assert insert_children_mock.call_count == insert_calls_after_first, (
        "second ingest re-inserted — idempotency broken"
    )


@_patch_io
def test_ingest_file_e2e_resolves_content_list_to_source_pdf(
    insert_children_mock: MagicMock,
    embed_mock: MagicMock,
    build_gemini_mock: MagicMock,
    doc_dir: Path,
    isolated_db: sqlite3.Connection,
    enc: tiktoken.Encoding,
) -> None:
    """A content_list.json input must produce source_file ending in source.pdf.

    Pre-Session-22 layout used the content_list.json path itself as
    source_file — that produced meaningless 'content_list.json' labels at
    retrieval time. _resolve_doc_representative is the fix; this test pins it.
    """
    embed_mock.side_effect = lambda _client, texts: [[0.0] * 4096 for _ in texts]
    build_gemini_mock.return_value = MagicMock()
    # insert_children_mock just absorbs the call — we'll inspect its call_args.

    ingest_file(
        path=doc_dir / "content_list.json",
        notebook="trading",
        collection="trading",
        gemini_client=MagicMock(),
        db_conn=isolated_db,
        enc=enc,
    )

    # _insert_children gets a list of ChildChunk objects; each carries source_file.
    children = insert_children_mock.call_args.args[1]
    assert len(children) >= 1, "no children passed to insert"
    for child in children:
        assert child.source_file.endswith("source.pdf"), (
            f"expected source.pdf swap, got {child.source_file}"
        )
        assert "content_list.json" not in child.source_file


@_patch_io
def test_ingest_file_e2e_applies_ctx_sidecar_on_structural_path(
    insert_children_mock: MagicMock,
    embed_mock: MagicMock,
    build_gemini_mock: MagicMock,
    doc_dir: Path,
    isolated_db: sqlite3.Connection,
    enc: tiktoken.Encoding,
) -> None:
    """The structural ingest path must consume the content_list.json.ctx.json
    sidecar and prepend each child's context summary to its embed text.

    Keys are derived by running the SAME chunker contextualize_corpus uses, so
    the test pins the round-trip alignment that Finding 1 closes: a sidecar
    written from chunk_sidecar+iter_child_keys lands on the matching child.
    """
    from src.chunking import chunk_sidecar
    from src.ingest.chunker import _load_source_meta, iter_child_keys

    embed_mock.side_effect = lambda _client, texts: [[0.0] * 4096 for _ in texts]
    build_gemini_mock.return_value = MagicMock()

    content_list = doc_dir / "content_list.json"
    meta = _load_source_meta(content_list.parent)
    _title, protos = chunk_sidecar(content_list, enc, meta=meta)
    keys = iter_child_keys(protos)
    assert keys, "fixture produced no children — can't test ctx application"

    # Contextualize the FIRST child only; the rest stay plain.
    target_pidx, target_cidx, _raw = keys[0]
    ctx_sentence = "CTX-MARKER: this chunk explains the reservation price."
    (content_list.parent / "content_list.json.ctx.json").write_text(
        json.dumps(
            {
                "chunks": [
                    {"parent_idx": target_pidx, "child_idx": target_cidx, "context": ctx_sentence},
                ],
            }
        )
    )

    ingest_file(
        path=content_list,
        notebook="trading",
        collection="trading",
        gemini_client=MagicMock(),
        db_conn=isolated_db,
        enc=enc,
    )

    children = insert_children_mock.call_args.args[1]
    with_marker = [c for c in children if c.text.startswith(ctx_sentence)]
    assert len(with_marker) == 1, f"expected exactly one ctx-prefixed child, got {len(with_marker)}"
    # Untouched children must NOT carry the marker.
    assert sum(1 for c in children if ctx_sentence in c.text) == 1


@_patch_io
def test_ingest_file_e2e_threads_notebook_through_to_milvus_rows(
    insert_children_mock: MagicMock,
    embed_mock: MagicMock,
    build_gemini_mock: MagicMock,
    doc_dir: Path,
    isolated_db: sqlite3.Connection,
    enc: tiktoken.Encoding,
) -> None:
    """The notebook arg threads all the way to Milvus row metadata.

    Partition routing depends on this. If the notebook field gets dropped or
    mis-set, the chunk lands in the wrong partition and queries scoped to
    the right partition can't find it.
    """
    embed_mock.side_effect = lambda _client, texts: [[0.0] * 4096 for _ in texts]
    build_gemini_mock.return_value = MagicMock()
    # insert_children_mock just absorbs the call — we'll inspect its call_args.

    ingest_file(
        path=doc_dir / "content_list.json",
        notebook="ecology",
        collection="ecology",
        gemini_client=MagicMock(),
        db_conn=isolated_db,
        enc=enc,
    )

    children = insert_children_mock.call_args.args[1]
    assert len(children) >= 1
    assert all(child.notebook == "ecology" for child in children), (
        "notebook arg didn't thread to ChildChunk records"
    )

    # And the parents in sqlite agree.
    parent_notebooks = {
        row[0] for row in isolated_db.execute("SELECT notebook FROM parents").fetchall()
    }
    assert parent_notebooks == {"ecology"}, f"parents have wrong notebook: {parent_notebooks}"
