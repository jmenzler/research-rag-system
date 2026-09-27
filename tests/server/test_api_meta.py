"""Tests for src/server/api_meta.py — /api/health + /api/version shape contract."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.server.api_meta import router


def _make_client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_health_returns_ok_sha_started_at() -> None:
    r = _make_client().get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert isinstance(body["sha"], str) and body["sha"]
    assert isinstance(body["started_at"], str) and "T" in body["started_at"]


def test_version_returns_sha_and_built_at_keys() -> None:
    r = _make_client().get("/api/version")
    assert r.status_code == 200
    body = r.json()
    assert "sha" in body and "built_at" in body
    assert isinstance(body["sha"], str)
    # built_at is null in dev, string in prod — either is valid here
    assert body["built_at"] is None or isinstance(body["built_at"], str)


def test_health_started_at_is_stable_across_requests() -> None:
    client = _make_client()
    first = client.get("/api/health").json()["started_at"]
    second = client.get("/api/health").json()["started_at"]
    assert first == second  # captured at module import, not per-request
