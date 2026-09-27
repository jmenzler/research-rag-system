"""Tests for src/server/static_bundle.py — SPA mount, redirect, cache headers."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.server import static_bundle


def _make_static(tmp_path: Path) -> Path:
    static = tmp_path / "static"
    (static / "assets").mkdir(parents=True)
    (static / "index.html").write_text("<!doctype html><html><body>shell</body></html>")
    (static / "assets" / "sample.js").write_text("console.log('hi')")
    return static


@pytest.fixture
def app_with_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    static = _make_static(tmp_path)
    monkeypatch.setattr(static_bundle, "STATIC_DIR", static)
    app = FastAPI()

    @app.get("/api/health")
    def health() -> dict[str, bool]:
        return {"ok": True}

    static_bundle.register(app)
    return app


def test_root_redirects_to_app(app_with_bundle: FastAPI) -> None:
    client = TestClient(app_with_bundle, follow_redirects=False)
    r = client.get("/")
    assert r.status_code == 302
    assert r.headers["location"] == "/app"


def test_app_returns_index_html(app_with_bundle: FastAPI) -> None:
    client = TestClient(app_with_bundle)
    r = client.get("/app")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "shell" in r.text


def test_deep_link_returns_same_index_html(app_with_bundle: FastAPI) -> None:
    client = TestClient(app_with_bundle)
    r = client.get("/app/chats/abc123/deep/link")
    assert r.status_code == 200
    assert "shell" in r.text  # SPA serves the same index.html


def test_index_html_no_store(app_with_bundle: FastAPI) -> None:
    client = TestClient(app_with_bundle)
    r = client.get("/app")
    assert r.headers["cache-control"].lower() == "no-store"


def test_assets_immutable_cache(app_with_bundle: FastAPI) -> None:
    client = TestClient(app_with_bundle)
    r = client.get("/assets/sample.js")
    assert r.status_code == 200
    assert "immutable" in r.headers["cache-control"]
    assert "max-age=31536000" in r.headers["cache-control"]


def test_api_routes_not_shadowed(app_with_bundle: FastAPI) -> None:
    client = TestClient(app_with_bundle)
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"ok": True}


def test_missing_bundle_returns_503_not_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    empty = tmp_path / "no_static"  # does not exist
    monkeypatch.setattr(static_bundle, "STATIC_DIR", empty)
    app = FastAPI()
    static_bundle.register(app)  # MUST NOT raise
    client = TestClient(app, follow_redirects=False)
    r = client.get("/app")
    assert r.status_code == 503
    assert "not built" in r.json()["error"]
    assert "--outDir ../src/server/static" in r.json()["error"]
