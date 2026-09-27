"""DiscoverQueue full-PDF upgrade job: enqueue dedup + command build."""

from __future__ import annotations

import time
from pathlib import Path

from src.server.discover_queue import DefaultSpawner, DiscoverQueue, QueueEntry


def _queue(tmp_path: Path) -> DiscoverQueue:
    return DiscoverQueue(state_dir=tmp_path / "state", log_dir=tmp_path / "logs")


def test_enqueue_upgrade_writes_upgrade_entry(tmp_path: Path) -> None:
    q = _queue(tmp_path)
    entry = q.enqueue_upgrade(collection="trading")
    assert entry is not None
    assert entry.kind == "upgrade"
    assert entry.collection == "trading"
    on_disk = list((tmp_path / "state" / "queued").glob("*.json"))
    assert len(on_disk) == 1


def test_enqueue_upgrade_is_deduped(tmp_path: Path) -> None:
    q = _queue(tmp_path)
    first = q.enqueue_upgrade()
    second = q.enqueue_upgrade()
    assert first is not None
    assert second is None  # an upgrade job is already queued
    assert len(list((tmp_path / "state" / "queued").glob("*.json"))) == 1


def test_build_cmd_upgrade(tmp_path: Path) -> None:
    entry = QueueEntry(
        run_id="u",
        counter=1,
        kind="upgrade",
        queries=["upgrade-abstract-only"],
        collection="trading",
        partition="arxiv_monitor",
        mode="fast",
        timeout=1800,
        keep=False,
        enqueued_at=time.time(),
        pid=None,
        log_path=str(tmp_path / "l.log"),
        result_path=str(tmp_path / "r.json"),
    )
    cmd = DefaultSpawner()._build_cmd(entry, tmp_path / "r.json")
    assert "scripts.upgrade_abstract_only" in cmd
    assert "--collection" in cmd
    assert "scripts.discover_via_deep_research" not in cmd


def test_build_cmd_discover_unchanged(tmp_path: Path) -> None:
    entry = QueueEntry(
        run_id="d",
        counter=1,
        queries=["q"],
        collection="trading",
        partition="research_briefs",
        mode="deep",
        timeout=1800,
        keep=False,
        enqueued_at=time.time(),
        pid=None,
        log_path=str(tmp_path / "l.log"),
        result_path=str(tmp_path / "r.json"),
    )
    cmd = DefaultSpawner()._build_cmd(entry, tmp_path / "r.json")
    assert "scripts.discover_via_deep_research" in cmd
    assert "scripts.upgrade_abstract_only" not in cmd
