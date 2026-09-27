# long-ok-file
from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from src.pipeline.queues import FetchItem

if TYPE_CHECKING:
    from scrapling.spiders import Request

    from src.fetch.spider import FetchPipelineSpider


class _FakeQueue:
    def __init__(self) -> None:
        self.events: list[tuple[Any, ...]] = []

    def put(self, ev: tuple[Any, ...]) -> None:
        self.events.append(ev)


def _items() -> list[FetchItem]:
    return [
        FetchItem(
            nb_tag="trading",
            nb_id="nb1",
            src={"id": f"s{i}", "url": f"https://example.com/{i}", "index": i},
            no_cffi=False,
        )
        for i in range(3)
    ]


def test_transport_drop_is_reconciled_as_fetch_fail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A source the spider never yields (on_error path) still emits fetch_fail.

    Without reconciliation the coordinator's fetch_done_count stalls below
    fetch_total forever (the original deadlock).
    """

    class _FakeSpider:
        def __init__(self) -> None:
            self._sources: list[dict[str, Any]] = []

        async def stream(self) -> AsyncGenerator[dict[str, Any], None]:
            for i in range(2):  # s2 is "lost" to on_error, never yielded
                yield {"nb_tag": "trading", "source_id": f"s{i}", "ok": True}

    import src.fetch.spider as spider_mod

    monkeypatch.setattr(spider_mod, "FetchPipelineSpider", _FakeSpider)

    from src.pipeline.runner import spider_fetch_all

    q = _FakeQueue()
    spider_fetch_all(_items(), q, tmp_path)  # type: ignore[arg-type]

    kinds = {(e[0], e[2]) for e in q.events}
    assert ("fetch_ok", "s0") in kinds
    assert ("fetch_ok", "s1") in kinds
    assert ("fetch_fail", "s2") in kinds
    fail = next(e for e in q.events if e[0] == "fetch_fail" and e[2] == "s2")
    assert fail[3] == "spider_no_result"
    assert len(q.events) == 3


# ---------------------------------------------------------------------------
# on_error escalation: a transport-level fetch error must route back into the
# http -> stealth -> stealth_max chain rather than silently dropping the source.
# ---------------------------------------------------------------------------


class _FakeScheduler:
    def __init__(self) -> None:
        self.enqueued: list[Any] = []

    async def enqueue(self, request: Request) -> bool:
        self.enqueued.append(request)
        return True


class _FakeEngine:
    def __init__(self) -> None:
        self.scheduler = _FakeScheduler()


def _spider(tmp_path: Path) -> FetchPipelineSpider:
    import os

    from src.fetch.spider import FetchPipelineSpider

    os.environ["SPIDER_OUT_DIR"] = str(tmp_path)
    sp = FetchPipelineSpider()
    sp._stats = {"ok": 0, "stealth_rescue": 0, "stealth_max_rescue": 0, "fail": 0}
    sp._out_root = tmp_path
    return sp


def _req(sid: str, meta: dict[str, Any]) -> Request:
    from scrapling.spiders import Request

    return Request("https://example.com/p", sid=sid, meta=meta)


def test_on_error_escalates_http_to_stealth(tmp_path: Path) -> None:
    import anyio

    sp = _spider(tmp_path)
    sp._engine = _FakeEngine()
    meta = {"title": "t", "original_url": "https://example.com/p", "nb_tag": "trading"}

    anyio.run(sp.on_error, _req("http", meta), ConnectionError("refused"))

    assert len(sp._engine.scheduler.enqueued) == 1
    req = sp._engine.scheduler.enqueued[0]
    assert req.sid == "stealth"
    assert req.meta["_sid"] == "stealth"
    assert sp._stats["fail"] == 0  # not counted as terminal yet


def test_on_error_exhausted_writes_failure_artifact(tmp_path: Path) -> None:
    import anyio

    sp = _spider(tmp_path)
    sp._engine = _FakeEngine()
    meta = {
        "title": "My Paper",
        "original_url": "https://example.com/p",
        "nb_tag": "trading",
        "_sid": "stealth_max",
        "_term_retries": 3,  # terminal retries already exhausted
        "index": 7,
        "source_id": "s7",
        "tier": "T5_web_blog",
    }

    anyio.run(sp.on_error, _req("stealth_max", meta), TimeoutError("read timeout"))

    assert sp._engine.scheduler.enqueued == []  # no further re-enqueue
    assert sp._stats["fail"] == 1
    out_dir = tmp_path / "trading" / "my_paper"
    assert (out_dir / "_failure_body.txt").exists()
    written = json.loads((out_dir / "meta.json").read_text())
    assert written["fetch"]["web"]["ok"] is False
    assert written["fetch"]["web"]["reason"].startswith("transport_err:")
