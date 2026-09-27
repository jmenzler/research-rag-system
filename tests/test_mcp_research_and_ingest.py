"""Tests for the MCP `research_and_ingest` tool surface.

Asserts that:
- Single-string query POSTs `{"query": "x", ...}`.
- List-of-strings POSTs `{"query": ["a", "b"], ...}` (no rewriting).
- The other params propagate correctly.

httpx.Client is stubbed at the module level; no network.
"""

from __future__ import annotations

from typing import Any

import pytest

from src import mcp_server as mcp_mod


class _StubResponse:
    def __init__(self, status_code: int = 200, payload: dict[str, Any] | None = None) -> None:
        self.status_code = status_code
        self._payload = payload or {"run_id": "abc123", "state": "queued"}
        self.text = ""

    def json(self) -> dict[str, Any]:
        return self._payload


class _StubClient:
    def __init__(self, **_: object) -> None:
        self.posts: list[tuple[str, dict[str, Any]]] = []

    def __enter__(self) -> _StubClient:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def post(self, url: str, json: dict[str, Any]) -> _StubResponse:
        self.posts.append((url, json))
        return _StubResponse()


@pytest.fixture()
def stub_client(monkeypatch: pytest.MonkeyPatch) -> _StubClient:
    stub = _StubClient()

    def factory(**kwargs: object) -> _StubClient:
        del kwargs
        return stub

    monkeypatch.setattr(mcp_mod.httpx, "Client", factory)
    return stub


def test_single_string_query_posts_string(stub_client: _StubClient) -> None:
    mcp_mod.research_and_ingest(
        query="what is GLFT",
        collection="trading",
    )
    assert len(stub_client.posts) == 1
    _, body = stub_client.posts[0]
    assert body["query"] == "what is GLFT"
    assert body["collection"] == "trading"


def test_list_query_posts_list(stub_client: _StubClient) -> None:
    mcp_mod.research_and_ingest(
        query=["a", "b", "c"],
        collection="trading",
        partition="research_briefs",
        mode="deep",
    )
    assert len(stub_client.posts) == 1
    _, body = stub_client.posts[0]
    assert body["query"] == ["a", "b", "c"]
    assert body["partition"] == "research_briefs"
    assert body["mode"] == "deep"


def test_default_params_present(stub_client: _StubClient) -> None:
    mcp_mod.research_and_ingest(query="x")
    _, body = stub_client.posts[0]
    assert body["collection"] == "trading"
    assert body["partition"] == "research_briefs"
    assert body["mode"] == "deep"
    assert body["timeout"] == 1800
    assert body["keep"] is False


def test_tool_signature_advertises_union() -> None:
    """The MCP tool's `query` parameter accepts str OR list[str].

    This is what makes the bulk-mode discoverable to MCP clients via the tool
    schema rather than being a runtime-only Python convention.
    """
    import typing

    hints = typing.get_type_hints(mcp_mod.research_and_ingest)
    annot = hints["query"]
    args = typing.get_args(annot)
    assert list in {typing.get_origin(a) or a for a in args}, (
        f"`query` should be Union[str, list[str]]; got {annot!r}"
    )
    assert str in args, f"`query` should still accept bare str; got {annot!r}"
