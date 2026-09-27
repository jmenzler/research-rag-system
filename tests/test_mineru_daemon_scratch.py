"""Unit tests for the mineru-api scratch-root relocation + wipe contract.

mineru-api's per-parse scratch (PDF copy + every extracted image) is relocated
out of the repo and wiped on each cold spawn so it can't grow unbounded (it once
hit 14G). These tests pin that behaviour without spawning the real daemon — the
lifecycle suite (slow-marked) covers the spawn path.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.pdf_parsers import mineru_daemon


def test_prepare_scratch_wipes_when_not_keep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "scratch"
    root.mkdir()
    (root / "stale_task").mkdir()
    monkeypatch.setattr(mineru_daemon, "_SCRATCH_ROOT", root)
    monkeypatch.setattr(mineru_daemon, "_KEEP_SCRATCH", False)

    returned = mineru_daemon._prepare_scratch_root()

    assert returned == root
    assert root.exists()
    assert not (root / "stale_task").exists()


def test_prepare_scratch_keeps_when_keep(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "scratch"
    root.mkdir()
    (root / "stale_task").mkdir()
    monkeypatch.setattr(mineru_daemon, "_SCRATCH_ROOT", root)
    monkeypatch.setattr(mineru_daemon, "_KEEP_SCRATCH", True)

    mineru_daemon._prepare_scratch_root()

    assert (root / "stale_task").exists()


def test_prepare_scratch_creates_missing_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "does" / "not" / "exist"
    monkeypatch.setattr(mineru_daemon, "_SCRATCH_ROOT", root)
    monkeypatch.setattr(mineru_daemon, "_KEEP_SCRATCH", False)

    mineru_daemon._prepare_scratch_root()

    assert root.is_dir()


def test_default_scratch_root_outside_repo() -> None:
    """Guard against regressing to the in-repo ./output default that caused the
    14G accumulation."""
    repo = Path(__file__).resolve().parent.parent
    assert repo not in mineru_daemon._SCRATCH_ROOT.parents
    assert mineru_daemon._SCRATCH_ROOT.name == "mineru-scratch"
