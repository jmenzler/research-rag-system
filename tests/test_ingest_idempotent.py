# long-ok-file
"""Idempotent ingest: pre-embed existence gate + upsert writes.

Re-running an unchanged source must cost 0 embed tokens and add 0 new rows.
Two collaborators make that true:

  * ``storage.existing_child_ids`` — queries Milvus for which child ids already
    exist (batched), tolerating a missing collection on first ingest.
  * ``ingest._filter_unembedded`` — drops children already present unless
    ``force`` re-embeds everything; returns the skip count for logging.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.ingest.ingest import _filter_unembedded
from src.ingest.storage import existing_child_ids
from src.models import ChildChunk


def _child(idx: int) -> ChildChunk:
    return ChildChunk(
        id=f"c{idx:04x}",
        parent_id="p0000",
        text=f"child {idx} text",
        source_file="/tmp/doc.pdf",
        notebook="trading",
        modality="pdf",
        page_number=0,
    )


# ---------------------------------------------------------------------------
# existing_child_ids — present ids / missing collection / batching
# ---------------------------------------------------------------------------


@patch("src.ingest.storage.get_client")
def test_existing_child_ids_returns_present_ids(get_client_mock: MagicMock) -> None:
    fake = MagicMock()
    # Of the three queried, only two exist in the collection.
    fake.query.return_value = [{"id": "c0000"}, {"id": "c0002"}]
    get_client_mock.return_value = fake

    present = existing_child_ids("trading", ["c0000", "c0001", "c0002"])

    assert present == {"c0000", "c0002"}
    kwargs = fake.query.call_args.kwargs
    assert kwargs["collection_name"] == "trading"
    assert kwargs["ids"] == ["c0000", "c0001", "c0002"]
    assert kwargs["output_fields"] == ["id"]


@patch("src.ingest.storage.get_client")
def test_existing_child_ids_empty_input_skips_query(get_client_mock: MagicMock) -> None:
    fake = MagicMock()
    get_client_mock.return_value = fake

    assert existing_child_ids("trading", []) == set()
    fake.query.assert_not_called()


@patch("src.ingest.storage.get_client")
def test_existing_child_ids_missing_collection_returns_empty(
    get_client_mock: MagicMock,
) -> None:
    """First ingest: the collection doesn't exist yet → no ids present, no raise."""
    fake = MagicMock()
    fake.has_collection.return_value = False
    get_client_mock.return_value = fake

    assert existing_child_ids("brand_new", ["c0000"]) == set()
    fake.query.assert_not_called()


@patch("src.ingest.storage.get_client")
def test_existing_child_ids_keeps_hits_on_midstream_failure(
    get_client_mock: MagicMock,
) -> None:
    """A transient failure on a later batch must NOT discard earlier confirmed
    hits — losing them forces a full re-embed of an already-ingested doc."""
    fake = MagicMock()
    fake.has_collection.return_value = True

    calls = {"n": 0}

    def _query(**kw: object) -> list[dict[str, str]]:
        calls["n"] += 1
        if calls["n"] == 1:
            ids = kw["ids"]
            assert isinstance(ids, list)
            return [{"id": i} for i in ids]
        raise RuntimeError("transient query failure")

    fake.query.side_effect = _query
    get_client_mock.return_value = fake

    ids = [f"c{i:05x}" for i in range(5000)]  # 2 batches (>4096)
    present = existing_child_ids("trading", ids)

    # First batch (4096 ids) confirmed; second batch raised but its hits aren't lost.
    assert present == set(ids[:4096])
    assert calls["n"] == 2


@patch("src.ingest.storage.get_client")
def test_existing_child_ids_batches_large_input(get_client_mock: MagicMock) -> None:
    """An id list larger than the per-call batch must be split across queries."""
    fake = MagicMock()
    # Echo back whatever ids each batch asked for so the union equals the input.
    fake.query.side_effect = lambda **kw: [{"id": i} for i in kw["ids"]]
    get_client_mock.return_value = fake

    ids = [f"c{i:05x}" for i in range(10_000)]
    present = existing_child_ids("trading", ids)

    assert present == set(ids)
    assert fake.query.call_count > 1, "large id list must be batched into multiple queries"
    # No single call may exceed the batch cap.
    for call in fake.query.call_args_list:
        assert len(call.kwargs["ids"]) <= 4096


# ---------------------------------------------------------------------------
# _filter_unembedded — filters existing, respects force, reports skip count
# ---------------------------------------------------------------------------


@patch("src.ingest.ingest.existing_child_ids")
def test_filter_unembedded_drops_existing(existing_mock: MagicMock) -> None:
    children = [_child(i) for i in range(4)]
    existing_mock.return_value = {"c0001", "c0003"}

    new, n_skipped = _filter_unembedded("trading", children, force=False)

    assert [c.id for c in new] == ["c0000", "c0002"]
    assert n_skipped == 2


@patch("src.ingest.ingest.existing_child_ids")
def test_filter_unembedded_force_returns_all(existing_mock: MagicMock) -> None:
    """force=True bypasses the existence check entirely (re-embed escape hatch)."""
    children = [_child(i) for i in range(3)]

    new, n_skipped = _filter_unembedded("trading", children, force=True)

    assert new == children
    assert n_skipped == 0
    existing_mock.assert_not_called()


@patch("src.ingest.ingest.existing_child_ids")
def test_filter_unembedded_all_present_returns_empty(existing_mock: MagicMock) -> None:
    """Re-run of an unchanged source: every id present → nothing to embed."""
    children = [_child(i) for i in range(3)]
    existing_mock.return_value = {c.id for c in children}

    new, n_skipped = _filter_unembedded("trading", children, force=False)

    assert new == []
    assert n_skipped == 3


@patch("src.ingest.ingest.existing_child_ids")
def test_filter_unembedded_empty_children(existing_mock: MagicMock) -> None:
    new, n_skipped = _filter_unembedded("trading", [], force=False)
    assert new == []
    assert n_skipped == 0
