"""Reranker backends — pluggable scoring against (query, document) pairs.

Selectable via environment variable ``RERANK_BACKEND``:

- ``local`` (default) — sentence-transformers CrossEncoder loaded on this machine.
  Picks MPS > CUDA > CPU. Used by ``BAAI/bge-reranker-v2-m3``.
- ``remote_vllm`` — HTTP POST to a vLLM ``/v1/score`` server. Used for models that
  don't fit locally or are deployed elsewhere (e.g. Qwen3-Reranker-* on the PC).
- ``remote_llamacpp`` — POST to a llama.cpp ``/v1/rerank`` server (Qwen3 GGUF on PC).
- ``remote_nvidia`` — hosted NVIDIA NIM reranking endpoint, with automatic
  fallback to the local ``remote_llamacpp`` Qwen3 reranker on error/rate-limit.

Wiring it in:

    from src.query.rerankers import get_reranker
    scores = get_reranker().score([(q, d) for d in docs])

Adding a new backend: implement the ``Reranker`` protocol, register it in
``_BACKENDS``, add the relevant env vars to ``src/config/retrieval.py``.

Backend selection happens once per process (singleton). To swap, restart.
"""

from __future__ import annotations

import logging
import math
import os
import threading
from typing import Any, Protocol

import requests  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


def _sigmoid(x: float) -> float:
    """Squash an unbounded logit into (0, 1), saturating at the extremes."""
    if x <= -60.0:
        return 0.0
    if x >= 60.0:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


class Reranker(Protocol):
    """Anything that can score a batch of (query, document) text pairs."""

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Return one float per pair, higher = more relevant."""
        ...

    def identity(self) -> dict[str, str | None]:
        """Return backend identification for audit logging.

        Keys: ``backend``, ``model``, ``endpoint`` (None when not applicable).
        """
        ...


# ---------------------------------------------------------------------------
# Local: sentence-transformers CrossEncoder
# ---------------------------------------------------------------------------


class LocalCrossEncoderReranker:
    """Wraps a sentence-transformers CrossEncoder loaded in-process.

    Lazy-loads on first ``score`` call so importing this module is cheap.
    """

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self._model: Any = None  # sentence_transformers.CrossEncoder
        # Serializes concurrent score() calls — CrossEncoder is not thread-safe
        # under MPS/CUDA (single GPU stream), and the load step is also guarded.
        self._lock = threading.Lock()

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch  # noqa: PLC0415
        from sentence_transformers import CrossEncoder  # noqa: PLC0415

        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        logger.info("Loading local reranker %s on device=%s …", self.model_id, device)
        self._model = CrossEncoder(self.model_id, device=device)
        logger.info("Local reranker loaded.")

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        with self._lock:
            self._load()
            return list(self._model.predict(pairs).tolist())

    def identity(self) -> dict[str, str | None]:
        return {"backend": "local", "model": self.model_id, "endpoint": None}


# ---------------------------------------------------------------------------
# Remote: vLLM /v1/score endpoint
# ---------------------------------------------------------------------------


class RemoteVLLMReranker:
    """POST to a vLLM ``/v1/score`` server, one HTTP call per (query, doc) pair.

    vLLM's ``/v1/score`` accepts a single ``text_1`` and ``text_2`` per request
    (it does not batch multiple pairs in one call). For N pairs, we make N serial
    HTTP calls. ``requests.Session`` is not thread-safe, so we do NOT spawn an
    internal ThreadPoolExecutor — that would race against ``self._session``
    even with the outer ``self._lock`` in place. vLLM batches concurrent
    requests internally at the server level; serial client calls are correct
    and sufficient for our eval workload.

    For production serving, prefer ``/v1/rerank`` which accepts multiple
    documents per call — but that requires reorganising the caller because
    rerank takes ``(query, [doc1, doc2, ...])`` not ``[(q1, d1), …]``.
    """

    def __init__(
        self,
        endpoint: str,
        model_name: str,
        timeout_s: float = 60.0,
        max_workers: int = 8,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model_name = model_name
        self.timeout_s = timeout_s
        self.max_workers = max_workers
        # Reuse a session for HTTP keepalive — meaningful when looping calls.
        self._session = requests.Session()
        # Protects the shared requests.Session — Session is not thread-safe.
        self._lock = threading.Lock()

    def _score_one(self, query: str, doc: str) -> float:
        resp = self._session.post(
            f"{self.endpoint}/v1/score",
            json={
                "model": self.model_name,
                "text_1": query,
                "text_2": doc,
            },
            timeout=self.timeout_s,
        )
        resp.raise_for_status()
        body = resp.json()
        # Schema: {"data": [{"index": 0, "score": <float>}], ...}
        return float(body["data"][0]["score"])

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []

        with self._lock:
            # Serial loop — requests.Session is not thread-safe (documented).
            # Spawning concurrent workers that share self._session would create
            # a data race even though the outer lock serialises pipeline-worker
            # callers. vLLM batches concurrency at the server level; serial
            # client calls are correct and sufficient for our eval workload.
            # Matches RemoteLlamaCppReranker.score.
            results: list[float] = []
            for q, d in pairs:
                results.append(self._score_one(q, d))
            return results

    def identity(self) -> dict[str, str | None]:
        return {
            "backend": "remote_vllm",
            "model": self.model_name,
            "endpoint": self.endpoint,
        }


# ---------------------------------------------------------------------------
# Remote: llama.cpp llama-server /v1/rerank endpoint
# ---------------------------------------------------------------------------


class RemoteLlamaCppReranker:
    """POST to a llama.cpp ``llama-server /v1/rerank`` endpoint.

    llama.cpp's /v1/rerank takes ``{query, documents: [...]}`` — N documents in
    ONE HTTP call (unlike vLLM's /v1/score which is one pair per call). Since
    the GPU saturates at ~50 docs per call anyway, serial calls grouped by
    query are optimal — concurrent calls only serialize on the single GPU.

    Used for Qwen3-Reranker GGUF (Q8_0) on the backend host with CC 7.5.
    """

    def __init__(
        self,
        endpoint: str,
        model_name: str,
        timeout_s: float = 120.0,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model_name = model_name
        self.timeout_s = timeout_s
        self._session = requests.Session()
        # Protects the shared requests.Session and the serial GPU call sequence.
        # The GPU is single-stream anyway, so this lock costs nothing in throughput.
        self._lock = threading.Lock()

    def _rerank_one_query(self, query: str, docs: list[str]) -> list[float]:
        resp = self._session.post(
            f"{self.endpoint}/v1/rerank",
            json={
                "model": self.model_name,
                "query": query,
                "documents": docs,
            },
            timeout=self.timeout_s,
        )
        resp.raise_for_status()
        body = resp.json()
        # Schema: {"results": [{"index": int, "relevance_score": float}, ...]}
        scored = [0.0] * len(docs)
        for r in body["results"]:
            scored[r["index"]] = float(r["relevance_score"])
        return scored

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []

        # Group docs by query — typically all pairs share one query, but
        # batched_rerank_and_lookup feeds (sub_query, hit) pairs from N sub-queries.
        from collections import defaultdict  # noqa: PLC0415

        with self._lock:
            groups: dict[str, list[tuple[int, str]]] = defaultdict(list)
            for i, (q, d) in enumerate(pairs):
                groups[q].append((i, d))

            results: list[float] = [0.0] * len(pairs)
            # Serial calls — GPU is single-stream, parallelism only adds overhead.
            for q, items in groups.items():
                docs = [d for _i, d in items]
                scores = self._rerank_one_query(q, docs)
                for (orig_idx, _), s in zip(items, scores):
                    results[orig_idx] = s
            return results

    def identity(self) -> dict[str, str | None]:
        return {
            "backend": "remote_llamacpp",
            "model": self.model_name,
            "endpoint": self.endpoint,
        }


# ---------------------------------------------------------------------------
# Remote: NVIDIA hosted NIM reranking endpoint
# ---------------------------------------------------------------------------


class NvidiaRemoteReranker:
    """POST to NVIDIA's hosted NIM reranking endpoint (nv-rerank-qa-mistral-4b).

    Request: ``{query: {text}, passages: [{text}, ...]}`` — N passages per call.
    The NIM returns a relevance ``logit`` per passage (unbounded, ~+17..-21); we
    squash it through a sigmoid so scores land in (0, 1), matching the 0..1 scale
    the other backends emit and keeping ``RERANK_SCORE_THRESHOLD`` meaningful.
    """

    def __init__(
        self,
        endpoint: str,
        model_name: str,
        api_key: str,
        timeout_s: float = 30.0,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model_name = model_name
        self._api_key = api_key
        self.timeout_s = timeout_s
        self._session = requests.Session()
        self._lock = threading.Lock()

    def _rerank_one_query(self, query: str, docs: list[str]) -> list[float]:
        resp = self._session.post(
            self.endpoint,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Accept": "application/json",
            },
            json={
                "model": self.model_name,
                "query": {"text": query},
                "passages": [{"text": d} for d in docs],
            },
            timeout=self.timeout_s,
        )
        resp.raise_for_status()
        body = resp.json()
        # Schema: {"rankings": [{"index": int, "logit": float}, ...]}
        scored = [0.0] * len(docs)
        for r in body["rankings"]:
            scored[r["index"]] = _sigmoid(float(r["logit"]))
        return scored

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []

        from collections import defaultdict  # noqa: PLC0415

        with self._lock:
            groups: dict[str, list[tuple[int, str]]] = defaultdict(list)
            for i, (q, d) in enumerate(pairs):
                groups[q].append((i, d))

            results: list[float] = [0.0] * len(pairs)
            for q, items in groups.items():
                docs = [d for _i, d in items]
                scores = self._rerank_one_query(q, docs)
                for (orig_idx, _), s in zip(items, scores):
                    results[orig_idx] = s
            return results

    def identity(self) -> dict[str, str | None]:
        return {
            "backend": "remote_nvidia",
            "model": self.model_name,
            "endpoint": self.endpoint,
        }


class IdentityReranker:
    """Neutral fallback — uniform score per pair (no reranking), so an NVIDIA
    failure degrades to retrieval-order top-k instead of the retired llamacpp
    fallback's ~0 scores that the threshold turned into zero survivors."""

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [1.0] * len(pairs)

    def identity(self) -> dict[str, str | None]:
        return {"backend": "identity", "model": None, "endpoint": None}


class FallbackReranker:
    """Try a primary reranker; on any failure, fall back to a secondary.

    Keeps retrieval working when the hosted NVIDIA NIM rate-limits or errors —
    falls back to the local Qwen3 reranker on the PC GPU.
    """

    def __init__(self, primary: Reranker, secondary: Reranker) -> None:
        self._primary = primary
        self._secondary = secondary

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        try:
            return self._primary.score(pairs)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Primary reranker %s failed (%s); falling back to %s",
                self._primary.identity().get("backend"), e,
                self._secondary.identity().get("backend"),
            )
            return self._secondary.score(pairs)

    def identity(self) -> dict[str, str | None]:
        prim = self._primary.identity()
        return {
            "backend": f"{prim['backend']}+fallback",
            "model": prim["model"],
            "endpoint": prim["endpoint"],
        }


# ---------------------------------------------------------------------------
# Singleton + factory
# ---------------------------------------------------------------------------


_singleton: Reranker | None = None


def _make_reranker() -> Reranker:
    """Build the configured reranker. Reads env at construction time only."""
    backend = os.getenv("RERANK_BACKEND", "remote_llamacpp").lower()

    if backend == "local":
        from src.config import RERANK_MODEL  # noqa: PLC0415

        return LocalCrossEncoderReranker(model_id=RERANK_MODEL)

    if backend == "remote_vllm":
        endpoint = os.getenv("RERANK_REMOTE_URL", "http://127.0.0.1:8089")
        model_name = os.getenv("RERANK_REMOTE_MODEL", "Qwen3-Reranker-0.6B")
        max_workers = int(os.getenv("RERANK_REMOTE_WORKERS", "8"))
        logger.info(
            "Using remote_vllm reranker: endpoint=%s model=%s workers=%d",
            endpoint, model_name, max_workers,
        )
        return RemoteVLLMReranker(
            endpoint=endpoint, model_name=model_name, max_workers=max_workers,
        )

    if backend == "remote_llamacpp":
        endpoint = os.getenv("RERANK_REMOTE_URL", "http://127.0.0.1:8090")
        model_name = os.getenv("RERANK_REMOTE_MODEL", "Qwen3-Reranker-4B")
        logger.info(
            "Using remote_llamacpp reranker: endpoint=%s model=%s",
            endpoint, model_name,
        )
        return RemoteLlamaCppReranker(endpoint=endpoint, model_name=model_name)

    if backend == "remote_nvidia":
        api_key = os.getenv("NVIDIA_API_KEY")
        if not api_key:
            raise RuntimeError(
                "RERANK_BACKEND=remote_nvidia requires NVIDIA_API_KEY in the env."
            )
        nv_url = os.getenv(
            "RERANK_NVIDIA_URL",
            "https://ai.api.nvidia.com/v1/retrieval/nvidia/reranking",
        )
        nv_model = os.getenv("RERANK_NVIDIA_MODEL", "nvidia/rerank-qa-mistral-4b")
        logger.info(
            "Using remote_nvidia reranker: model=%s endpoint=%s (fallback=identity)",
            nv_model, nv_url,
        )
        return FallbackReranker(
            primary=NvidiaRemoteReranker(
                endpoint=nv_url, model_name=nv_model, api_key=api_key,
            ),
            secondary=IdentityReranker(),
        )

    raise RuntimeError(
        f"Unknown RERANK_BACKEND={backend!r}. Expected one of: "
        f"local, remote_vllm, remote_llamacpp, remote_nvidia."
    )


def get_reranker() -> Reranker:
    """Return the process-wide reranker singleton, building on first call."""
    global _singleton
    if _singleton is None:
        _singleton = _make_reranker()
    return _singleton


def reset_reranker() -> None:
    """Drop the singleton — only for tests that need to rebuild between cases."""
    global _singleton
    _singleton = None
