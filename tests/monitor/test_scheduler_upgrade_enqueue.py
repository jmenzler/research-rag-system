"""MonitorScheduler routes full-PDF upgrades to the DiscoverQueue after a poll."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.monitor.scheduler import MonitorScheduler


def test_enqueue_upgrade_calls_queue_when_present() -> None:
    queue = MagicMock()
    with patch("src.server.api.get_discover_queue", return_value=queue):
        MonitorScheduler._enqueue_full_pdf_upgrade()
    queue.enqueue_upgrade.assert_called_once_with(collection="trading")


def test_enqueue_upgrade_noop_when_queue_not_ready() -> None:
    # No queue wired yet (early boot) — must not raise.
    with patch("src.server.api.get_discover_queue", return_value=None):
        MonitorScheduler._enqueue_full_pdf_upgrade()


def test_enqueue_upgrade_swallows_queue_errors() -> None:
    queue = MagicMock()
    queue.enqueue_upgrade.side_effect = RuntimeError("queue full")
    with patch("src.server.api.get_discover_queue", return_value=queue):
        MonitorScheduler._enqueue_full_pdf_upgrade()  # must not propagate
