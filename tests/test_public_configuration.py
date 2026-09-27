from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from prompts.contextualize.audiences import DEFAULT_AUDIENCE
from src.ingest.contextualize import _build_system_prompt, _read_prompt
from src.query import rerankers
from src.server import __main__ as server_main

if TYPE_CHECKING:
    from tests.conftest import _FakeBroker


def test_contextualization_loads_exported_prompts() -> None:
    _read_prompt.cache_clear()
    prompt = _build_system_prompt(DEFAULT_AUDIENCE)
    assert DEFAULT_AUDIENCE["audience"] in prompt
    assert "{glossary}" not in prompt
    assert "{examples}" not in prompt


def test_default_reranker_uses_loopback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RERANK_BACKEND", raising=False)
    monkeypatch.delenv("RERANK_REMOTE_URL", raising=False)
    instance = rerankers._make_reranker()
    assert instance.endpoint == "http://127.0.0.1:8090"


def test_default_server_binds_loopback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RAG_SERVER_HOST", raising=False)
    with (
        patch.object(server_main, "DiscoverQueue"),
        patch.object(server_main, "MonitorScheduler"),
        patch.object(server_main.uvicorn, "run") as run,
        patch.object(server_main, "set_discover_queue"),
        patch.object(server_main, "set_monitor_scheduler"),
        patch.object(server_main, "set_ingest_queue"),
    ):
        server_main.main()
    assert run.call_args.kwargs["host"] == "127.0.0.1"


def test_missing_model_fails_before_gpu_claim(
    fake_broker: _FakeBroker, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import reranker_daemon

    monkeypatch.setattr(reranker_daemon, "MODEL_PATH", "")
    manager = reranker_daemon.RerankerManager()
    with patch.object(reranker_daemon.subprocess, "Popen") as popen:
        with pytest.raises(RuntimeError, match="RERANKER_MODEL_PATH"):
            manager._spawn()
    assert not fake_broker.acquire_calls
    popen.assert_not_called()
