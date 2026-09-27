"""Tests that __main__.py points uvicorn at the new composer module."""

from __future__ import annotations

from pathlib import Path


def test_main_targets_composer_not_api() -> None:
    main_path = Path(__file__).resolve().parents[2] / "src" / "server" / "__main__.py"
    content = main_path.read_text()
    assert 'uvicorn.run("src.server.app:app"' in content
    assert 'uvicorn.run("src.server.api:app"' not in content


def test_main_preserves_scheduler_and_queue_boot() -> None:
    main_path = Path(__file__).resolve().parents[2] / "src" / "server" / "__main__.py"
    content = main_path.read_text()
    # Brownfield: setter functions from api still imported, scheduler+queue still booted
    assert "from src.server.api import set_discover_queue, set_monitor_scheduler" in content
    assert "scheduler.start()" in content
    assert "queue.start()" in content


def test_composer_target_import_resolves() -> None:
    """Smoke: the string 'src.server.app:app' that uvicorn will use must resolve."""
    from src.server.app import app

    assert app is not None
