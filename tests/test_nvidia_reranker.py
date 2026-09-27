from __future__ import annotations

import math
from typing import Any

from src.query.rerankers import (
    FallbackReranker,
    NvidiaRemoteReranker,
    _sigmoid,
)


class _FakeResp:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


class _FakeSession:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        self.calls: list[dict[str, Any]] = []

    def post(self, url: str, **kwargs: object) -> _FakeResp:
        self.calls.append({"url": url, **kwargs})
        return _FakeResp(self._payload)


def test_sigmoid_bounds() -> None:
    assert _sigmoid(0.0) == 0.5
    assert _sigmoid(-100.0) == 0.0
    assert _sigmoid(100.0) == 1.0
    assert abs(_sigmoid(2.0) - 1.0 / (1.0 + math.exp(-2.0))) < 1e-9


def test_nvidia_score_maps_logits_to_original_order() -> None:
    rr = NvidiaRemoteReranker(
        endpoint="https://x/reranking",
        model_name="m",
        api_key="k",
    )
    # NIM returns rankings sorted by logit; index points back at input position.
    rr._session = _FakeSession(
        {  # type: ignore[assignment]
            "rankings": [{"index": 1, "logit": 2.0}, {"index": 0, "logit": -2.0}],
        }
    )
    scores = rr.score([("q", "docA"), ("q", "docB")])
    assert scores[0] == _sigmoid(-2.0)  # docA
    assert scores[1] == _sigmoid(2.0)  # docB
    assert scores[1] > scores[0]


def test_nvidia_empty_pairs() -> None:
    rr = NvidiaRemoteReranker(endpoint="https://x", model_name="m", api_key="k")
    assert rr.score([]) == []


def test_fallback_used_when_primary_raises() -> None:
    class _Boom:
        def score(self, pairs: list[tuple[str, str]]) -> list[float]:
            raise RuntimeError("rate limited")

        def identity(self) -> dict[str, str | None]:
            return {"backend": "remote_nvidia", "model": "m", "endpoint": "e"}

    class _Ok:
        def score(self, pairs: list[tuple[str, str]]) -> list[float]:
            return [0.7] * len(pairs)

        def identity(self) -> dict[str, str | None]:
            return {"backend": "remote_llamacpp", "model": "q", "endpoint": "e2"}

    fb = FallbackReranker(primary=_Boom(), secondary=_Ok())
    assert fb.score([("q", "d"), ("q", "e")]) == [0.7, 0.7]
    assert fb.identity()["backend"] == "remote_nvidia+fallback"
