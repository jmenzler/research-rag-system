"""Tests for the pluggable reranker abstraction in src/query/rerankers.py.

Covers the factory's env-var dispatch, the singleton behavior, the local-vs-remote
implementations conform to the Reranker protocol, and the remote backend's HTTP
shape against vLLM /v1/score.

Slice 2 (parallelism PR): also tests per-instance threading.Lock on each backend
so concurrent score() calls from the parallel eval pipeline are safe.
"""

from __future__ import annotations

import threading
import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.query import rerankers as rr


@pytest.fixture(autouse=True)
def _reset_singleton() -> None:
    """Drop the singleton between tests so env changes take effect."""
    rr.reset_reranker()


class TestFactory:
    def test_default_is_remote_llamacpp(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("RERANK_BACKEND", raising=False)
        monkeypatch.delenv("RERANK_REMOTE_URL", raising=False)
        monkeypatch.delenv("RERANK_REMOTE_MODEL", raising=False)
        instance = rr._make_reranker()
        assert isinstance(instance, rr.RemoteLlamaCppReranker)
        assert instance.endpoint == "http://127.0.0.1:8090"
        assert instance.model_name == "Qwen3-Reranker-4B"

    def test_explicit_local(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RERANK_BACKEND", "local")
        instance = rr._make_reranker()
        assert isinstance(instance, rr.LocalCrossEncoderReranker)

    def test_explicit_remote_vllm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RERANK_BACKEND", "remote_vllm")
        monkeypatch.setenv("RERANK_REMOTE_URL", "http://test:9999")
        monkeypatch.setenv("RERANK_REMOTE_MODEL", "TestModel")
        instance = rr._make_reranker()
        assert isinstance(instance, rr.RemoteVLLMReranker)
        assert instance.endpoint == "http://test:9999"
        assert instance.model_name == "TestModel"

    def test_unknown_backend_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RERANK_BACKEND", "moonshot")
        with pytest.raises(RuntimeError, match="Unknown RERANK_BACKEND"):
            rr._make_reranker()

    def test_explicit_remote_llamacpp(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RERANK_BACKEND", "remote_llamacpp")
        monkeypatch.setenv("RERANK_REMOTE_URL", "http://test:7777")
        monkeypatch.setenv("RERANK_REMOTE_MODEL", "TestRR")
        instance = rr._make_reranker()
        assert isinstance(instance, rr.RemoteLlamaCppReranker)
        assert instance.endpoint == "http://test:7777"
        assert instance.model_name == "TestRR"

    def test_singleton_caches(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RERANK_BACKEND", "remote_vllm")
        a = rr.get_reranker()
        b = rr.get_reranker()
        assert a is b

    def test_reset_clears_singleton(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RERANK_BACKEND", "remote_vllm")
        a = rr.get_reranker()
        rr.reset_reranker()
        b = rr.get_reranker()
        assert a is not b


class TestLocalCrossEncoder:
    def test_lazy_load(self) -> None:
        backend = rr.LocalCrossEncoderReranker(model_id="dummy")
        assert backend._model is None  # not loaded just by construction

    def test_score_empty_skips_load(self) -> None:
        """Empty input returns empty list without invoking the model."""
        backend = rr.LocalCrossEncoderReranker(model_id="dummy")
        assert backend.score([]) == []
        assert backend._model is None

    def test_score_calls_predict_with_pairs(self) -> None:
        backend = rr.LocalCrossEncoderReranker(model_id="dummy")

        # Stub the underlying CrossEncoder
        fake_model = MagicMock()
        fake_model.predict.return_value = MagicMock(tolist=lambda: [0.9, 0.1])
        backend._model = fake_model

        pairs = [("q", "d1"), ("q", "d2")]
        out = backend.score(pairs)

        fake_model.predict.assert_called_once_with(pairs)
        assert out == [0.9, 0.1]


class TestRemoteVLLM:
    def test_score_one_posts_correct_payload(self) -> None:
        backend = rr.RemoteVLLMReranker(
            endpoint="http://h:8089",
            model_name="MyModel",
        )

        with patch.object(backend._session, "post") as mock_post:
            mock_post.return_value.json.return_value = {
                "data": [{"index": 0, "score": 0.42}],
            }
            mock_post.return_value.raise_for_status = MagicMock()

            score = backend._score_one("hello", "world")

            assert score == 0.42
            mock_post.assert_called_once()
            call_kwargs = mock_post.call_args.kwargs
            assert call_kwargs["json"] == {
                "model": "MyModel",
                "text_1": "hello",
                "text_2": "world",
            }
            assert call_kwargs["timeout"] == 60.0
            assert mock_post.call_args.args[0] == "http://h:8089/v1/score"

    def test_score_empty(self) -> None:
        backend = rr.RemoteVLLMReranker(endpoint="http://h:8089", model_name="M")
        assert backend.score([]) == []

    def test_score_preserves_order_under_concurrency(self) -> None:
        """Thread pool may complete out of order; final list must match input order."""
        backend = rr.RemoteVLLMReranker(
            endpoint="http://h:8089",
            model_name="M",
            max_workers=4,
        )
        # Simulate scoring: doc text -> score
        responses = {"a": 0.1, "b": 0.5, "c": 0.9, "d": 0.3}

        def fake_post(url: str, json: dict[str, Any], timeout: float) -> MagicMock:
            mock = MagicMock()
            mock.raise_for_status = MagicMock()
            mock.json.return_value = {
                "data": [{"index": 0, "score": responses[json["text_2"]]}],
            }
            return mock

        with patch.object(backend._session, "post", side_effect=fake_post):
            pairs = [("q", "a"), ("q", "b"), ("q", "c"), ("q", "d")]
            scores = backend.score(pairs)

        assert scores == [0.1, 0.5, 0.9, 0.3]

    def test_endpoint_trailing_slash_stripped(self) -> None:
        backend = rr.RemoteVLLMReranker(
            endpoint="http://h:8089/",
            model_name="M",
        )
        assert backend.endpoint == "http://h:8089"

    def test_raise_for_status_on_http_error(self) -> None:
        backend = rr.RemoteVLLMReranker(endpoint="http://h:8089", model_name="M")
        with patch.object(backend._session, "post") as mock_post:
            mock_post.return_value.raise_for_status.side_effect = RuntimeError("500")
            with pytest.raises(RuntimeError, match="500"):
                backend._score_one("q", "d")


class TestProtocolConformance:
    """Both implementations satisfy the Reranker protocol."""

    def test_local_satisfies_protocol(self) -> None:
        backend: rr.Reranker = rr.LocalCrossEncoderReranker(model_id="dummy")
        # Empty-input must work without loading
        assert backend.score([]) == []

    def test_remote_satisfies_protocol(self) -> None:
        backend: rr.Reranker = rr.RemoteVLLMReranker(
            endpoint="http://h:8089",
            model_name="M",
        )
        assert backend.score([]) == []


class TestIdentityMethod:
    """Each backend's .identity() returns the expected fields."""

    def test_local_identity(self) -> None:
        backend = rr.LocalCrossEncoderReranker(model_id="BAAI/bge-reranker-v2-m3")
        ident = backend.identity()
        assert ident["backend"] == "local"
        assert ident["model"] == "BAAI/bge-reranker-v2-m3"
        assert ident["endpoint"] is None

    def test_remote_vllm_identity(self) -> None:
        backend = rr.RemoteVLLMReranker(endpoint="http://pc:8089", model_name="Qwen3-Reranker-0.6B")
        ident = backend.identity()
        assert ident["backend"] == "remote_vllm"
        assert ident["model"] == "Qwen3-Reranker-0.6B"
        assert ident["endpoint"] == "http://pc:8089"

    def test_remote_llamacpp_identity(self) -> None:
        backend = rr.RemoteLlamaCppReranker(
            endpoint="http://pc:8090", model_name="Qwen3-Reranker-4B"
        )
        ident = backend.identity()
        assert ident["backend"] == "remote_llamacpp"
        assert ident["model"] == "Qwen3-Reranker-4B"
        assert ident["endpoint"] == "http://pc:8090"

    def test_identity_keys_present(self) -> None:
        """All backends expose backend/model/endpoint regardless of type."""
        backends = [
            rr.LocalCrossEncoderReranker(model_id="m"),
            rr.RemoteVLLMReranker(endpoint="http://h:1", model_name="m"),
            rr.RemoteLlamaCppReranker(endpoint="http://h:2", model_name="m"),
        ]
        for backend in backends:
            ident = backend.identity()
            assert "backend" in ident
            assert "model" in ident
            assert "endpoint" in ident


# ---------------------------------------------------------------------------
# Slice 2 — per-instance lock presence
# ---------------------------------------------------------------------------


class TestRerankerLockPresence:
    """Every backend __init__ must set self._lock (threading.Lock or compatible)."""

    def test_local_cross_encoder_has_lock(self) -> None:
        backend = rr.LocalCrossEncoderReranker(model_id="BAAI/bge-reranker-v2-m3")
        assert hasattr(backend, "_lock"), "LocalCrossEncoderReranker is missing self._lock"
        assert hasattr(backend._lock, "acquire") and hasattr(backend._lock, "release")

    def test_remote_vllm_has_lock(self) -> None:
        backend = rr.RemoteVLLMReranker(endpoint="http://localhost:8089", model_name="m")
        assert hasattr(backend, "_lock"), "RemoteVLLMReranker is missing self._lock"
        assert hasattr(backend._lock, "acquire") and hasattr(backend._lock, "release")

    def test_remote_llamacpp_has_lock(self) -> None:
        backend = rr.RemoteLlamaCppReranker(endpoint="http://localhost:8090", model_name="m")
        assert hasattr(backend, "_lock"), "RemoteLlamaCppReranker is missing self._lock"
        assert hasattr(backend._lock, "acquire") and hasattr(backend._lock, "release")


# ---------------------------------------------------------------------------
# Slice 2 — concurrent score() calls (mocked HTTP)
# ---------------------------------------------------------------------------


_SAMPLE_PAIRS: list[tuple[str, str]] = [
    ("What is VPIN?", "VPIN measures order-flow imbalance."),
    ("What is MMR?", "MMR maximises relevance while minimising redundancy."),
    ("Explain hedging", "Hedging reduces directional risk via offsetting positions."),
]


class TestRemoteVLLMRerankerConcurrency:
    """4 concurrent threads call score() on one shared RemoteVLLMReranker."""

    N_THREADS = 4

    def _mock_response(self) -> MagicMock:
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {"data": [{"index": 0, "score": 0.75}]}
        return resp

    def test_concurrent_score_no_exception(self) -> None:
        backend = rr.RemoteVLLMReranker(endpoint="http://h:8089", model_name="M")
        backend._session.post = lambda url, **kw: self._mock_response()  # type: ignore[method-assign]

        errors: list[Exception] = []
        barrier = threading.Barrier(self.N_THREADS)

        def worker() -> None:
            try:
                barrier.wait()
                backend.score(_SAMPLE_PAIRS)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(self.N_THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Concurrent score() raised exceptions: {errors}"

    def test_concurrent_score_returns_correct_length(self) -> None:
        backend = rr.RemoteVLLMReranker(endpoint="http://h:8089", model_name="M")
        backend._session.post = lambda url, **kw: self._mock_response()  # type: ignore[method-assign]

        results: list[list[float]] = []

        def worker() -> None:
            results.append(backend.score(_SAMPLE_PAIRS))

        threads = [threading.Thread(target=worker) for _ in range(self.N_THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == self.N_THREADS
        for scores in results:
            assert len(scores) == len(_SAMPLE_PAIRS)


class TestRemoteLlamaCppRerankerConcurrency:
    """4 concurrent threads call score() on one shared RemoteLlamaCppReranker."""

    N_THREADS = 4

    def _mock_response(self, n_docs: int) -> MagicMock:
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {
            "results": [{"index": i, "relevance_score": 0.8} for i in range(n_docs)]
        }
        return resp

    def test_concurrent_score_no_exception(self) -> None:
        backend = rr.RemoteLlamaCppReranker(endpoint="http://h:8090", model_name="M")

        def fake_post(url: str, json: dict[str, Any], **kw: object) -> MagicMock:
            time.sleep(0.002)
            return self._mock_response(len(json.get("documents", [])))

        backend._session.post = fake_post  # type: ignore[method-assign]

        errors: list[Exception] = []
        barrier = threading.Barrier(self.N_THREADS)

        def worker() -> None:
            try:
                barrier.wait()
                backend.score(_SAMPLE_PAIRS)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(self.N_THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Concurrent score() raised: {errors}"

    def test_concurrent_score_returns_correct_length(self) -> None:
        backend = rr.RemoteLlamaCppReranker(endpoint="http://h:8090", model_name="M")

        def fake_post(url: str, json: dict[str, Any], **kw: object) -> MagicMock:
            return self._mock_response(len(json.get("documents", [])))

        backend._session.post = fake_post  # type: ignore[method-assign]

        results: list[list[float]] = []

        def worker() -> None:
            results.append(backend.score(_SAMPLE_PAIRS))

        threads = [threading.Thread(target=worker) for _ in range(self.N_THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == self.N_THREADS
        for scores in results:
            assert len(scores) == len(_SAMPLE_PAIRS)


class TestLocalCrossEncoderConcurrency:
    """LocalCrossEncoderReranker with mocked model — no GPU needed."""

    N_THREADS = 4

    def test_concurrent_score_no_exception(self) -> None:
        import numpy as np  # type: ignore[import-untyped]

        backend = rr.LocalCrossEncoderReranker(model_id="dummy")
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.9, 0.7, 0.5])
        backend._model = mock_model

        errors: list[Exception] = []
        barrier = threading.Barrier(self.N_THREADS)

        def worker() -> None:
            try:
                barrier.wait()
                backend.score(_SAMPLE_PAIRS)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(self.N_THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Concurrent score() raised: {errors}"
