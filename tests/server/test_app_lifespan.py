"""Tests that lifespan in src/server/app.py runs migrations on startup."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient


def test_lifespan_calls_migrate_run_on_startup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.server import app as app_module

    calls: list[Path] = []

    def fake_run(db_path: Path) -> None:
        calls.append(db_path)

    monkeypatch.setattr(app_module.migrate, "run", fake_run)
    monkeypatch.setattr(app_module, "CHATS_DB_PATH", tmp_path / "chats.db")
    with TestClient(app_module.app):
        pass
    assert calls == [tmp_path / "chats.db"]


def test_lifespan_propagates_migration_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    from collections.abc import AsyncIterator
    from contextlib import asynccontextmanager

    from fastapi import FastAPI

    from src.server import app as app_module

    @asynccontextmanager
    async def failing_lifespan(_: FastAPI) -> AsyncIterator[None]:
        raise RuntimeError("simulated migration failure")
        yield  # unreachable

    original_lifespan = app_module.app.router.lifespan_context
    app_module.app.router.lifespan_context = failing_lifespan
    try:
        with pytest.raises(RuntimeError, match="simulated migration failure"):
            with TestClient(app_module.app):
                pass
    finally:
        app_module.app.router.lifespan_context = original_lifespan


def test_lifespan_is_attached_to_same_app_instance() -> None:
    from src.server import api as api_module
    from src.server import app as app_module

    assert app_module.app is api_module.app
    assert app_module.app.router.lifespan_context is not None
