"""Tests that src/server/app.py wires meta router onto the existing api.app."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.server import api as api_module
from src.server import app as app_module


def test_composer_reexports_same_app_instance() -> None:
    """Brownfield invariant: composer does NOT create a second FastAPI app."""
    assert app_module.app is api_module.app


def test_composer_includes_meta_router() -> None:
    client = TestClient(app_module.app)
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_composer_preserves_existing_routes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Existing /notebooks must still respond — brownfield routes intact.

    Mock the Milvus-backed implementation so the route check doesn't depend on
    a reachable Milvus (the real client blocks on gRPC channel-ready without a
    bounded timeout, which hangs CI). api.py binds ``list_notebooks`` into its
    own namespace at import time, so the patch target is ``api_module``.
    """
    monkeypatch.setattr(
        api_module,
        "list_notebooks",
        lambda collection=None: {"notebooks": []},
    )
    client = TestClient(app_module.app)
    r = client.get("/notebooks")
    assert r.status_code != 404


def test_composer_project_root_resolves_to_repo_root() -> None:
    """PROJECT_ROOT must point at the repo root (parents[2] from src/server/)."""
    from src.server.app import PROJECT_ROOT

    # The repo root contains pyproject.toml
    assert (PROJECT_ROOT / "pyproject.toml").is_file()
