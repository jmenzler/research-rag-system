"""End-to-end integration tests for src.query.query_pipeline.run_query_pipeline.

These tests stub external dependencies (router LLM, Milvus, cross-encoder,
sqlite parents, synthesis LLM, grader LLM) but exercise the actual function
bodies of run_query_pipeline → _retrieve → _rrf_merge → batched_rerank_and_lookup
→ generate. The point is to verify pipeline COMPOSITION:

  - stage order: decompose → milvus(per-sq) → RRF → CE → MMR → synth → CRAG
  - audit dir captures every stage in expected files
  - cross-encoder is called against ORIGINAL query, not sub-queries
  - queries.jsonl ends up with audit_dir field
  - RRF merge actually balances per-sub-query lists
  - CRAG retry path triggers + records 07_grader/08_crag_retry stages

Unit tests for each component still live in their own files; this file
catches integration-only regressions (wrong wiring, missing audit writes).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.models import ParentChunk, Usage
from src.query.query_pipeline import run_query_pipeline
from src.query.usage_track import LLMCallResult

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def _fake_router_result(use_mmr: bool | None = None) -> LLMCallResult:
    """Router LLM returns a 3-sub-query plan with optional use_mmr override."""
    payload = {
        "intent": "multi_hop",
        "entities": ["AS", "QRH", "inventory"],
        "fire": ["milvus"],
        "tool_payloads": {
            "milvus": [
                "Avellaneda-Stoikov reservation price formula inventory risk",
                "quadratic rough Heston model multi-asset market making",
                "portfolio derivatives delta exposure dimensionality reduction",
            ],
        },
        "use_mmr": use_mmr,
        "use_mmr_reason": "test",
    }
    return LLMCallResult(
        text=json.dumps(payload),
        latency_ms=1,
        usage=Usage(),
        raw_response=None,
        cost_usd=0.0,
        model="fake-router",
    )


def _fake_hit(  # noqa: ANN401  — generic test stub
    child_id: str,
    parent_id: str,
    source_file: str,
    distance: float = 0.5,
) -> dict[str, Any]:
    """Build a Milvus-hit dict matching the schema retrieve.py reads."""
    return {
        "id": child_id,
        "distance": distance,
        "entity": {
            "text": f"chunk text for {child_id}",
            "parent_chunk_id": parent_id,
            "source_file": source_file,
            "notebook": "trading",
            "modality": "pdf",
            "page_number": 1,
            "dense_embedding": [1.0 if i == hash(child_id) % 4 else 0.0 for i in range(4)],
        },
    }


def _fake_milvus_search_per_sq(
    sub_query: str,
    *,
    sub_query_idx: int,
    n_per_sq: int = 30,
) -> list[dict[str, Any]]:
    """Make per-sub-query hit lists with controlled overlap.

    sub_q 0 returns chunks shared/00, shared/01, only_a/00..n
    sub_q 1 returns chunks shared/00, shared/02, only_b/00..n
    sub_q 2 returns chunks shared/01, shared/02, only_c/00..n

    Shared chunks should bubble to top via RRF; per-sub-query unique
    chunks fill in below.
    """
    hits: list[dict[str, Any]] = []
    if sub_query_idx == 0:
        hits.append(_fake_hit("shared00", "P_shared00", "shared.pdf", 0.95))
        hits.append(_fake_hit("shared01", "P_shared01", "shared.pdf", 0.90))
        for i in range(n_per_sq - 2):
            hits.append(_fake_hit(f"only_a_{i:02d}", f"P_a_{i:02d}", "paper_a.pdf", 0.5 - i * 0.01))
    elif sub_query_idx == 1:
        hits.append(_fake_hit("shared00", "P_shared00", "shared.pdf", 0.93))
        hits.append(_fake_hit("shared02", "P_shared02", "shared.pdf", 0.88))
        for i in range(n_per_sq - 2):
            hits.append(_fake_hit(f"only_b_{i:02d}", f"P_b_{i:02d}", "paper_b.pdf", 0.5 - i * 0.01))
    else:
        hits.append(_fake_hit("shared01", "P_shared01", "shared.pdf", 0.92))
        hits.append(_fake_hit("shared02", "P_shared02", "shared.pdf", 0.91))
        for i in range(n_per_sq - 2):
            hits.append(_fake_hit(f"only_c_{i:02d}", f"P_c_{i:02d}", "paper_c.pdf", 0.5 - i * 0.01))
    return hits


class _FakeReranker:
    """Cross-encoder stub: returns pseudo-scores from text content."""

    def __init__(self) -> None:
        self.calls_seen: list[list[tuple[str, str]]] = []

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        self.calls_seen.append(list(pairs))
        # Score = 1.0 for shared* (boost RRF winners), 0.5 for others. Both
        # above the 0.20 threshold. No randomness so test is deterministic.
        return [1.0 if "shared" in text else 0.5 for _q, text in pairs]

    def identity(self) -> dict[str, str | None]:
        return {"backend": "fake", "model": "fake-rr", "endpoint": None}


def _fake_lookup_parents(
    parent_ids: list[str],
    db_path: object,  # noqa: ARG001 — kwargs match production signature
) -> dict[str, ParentChunk]:
    return {
        pid: ParentChunk(
            id=pid,
            text=f"parent text for {pid}",
            source_file="x.pdf",
            notebook="trading",
            modality="pdf",
            page_number=1,
        )
        for pid in parent_ids
    }


def _fake_synthesize_answer(
    query: str,
    retrieved: list[Any],  # noqa: ARG001
    *,
    model: object = None,  # noqa: ARG001
) -> tuple[str, int, Usage]:
    return (f"Synthesized answer for: {query[:30]}\n\nSources:\n[1] x.pdf", 100, Usage())


# ---------------------------------------------------------------------------
# Composition assertions
# ---------------------------------------------------------------------------


@pytest.fixture
def _fake_reranker() -> _FakeReranker:
    return _FakeReranker()


@pytest.fixture
def _patched_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    _fake_reranker: _FakeReranker,
) -> dict[str, Any]:
    """Patch all external deps so run_query_pipeline runs end-to-end against fakes.
    Returns dict of patches/state for tests to inspect.
    """
    # Module-vs-function name collision: `src.query.__init__` does
    # `from src.query.generate import generate`, which shadows the module
    # attribute. Use importlib to fetch modules unambiguously.
    import importlib

    gen_mod = importlib.import_module("src.query.generate")
    retrieve_mod = importlib.import_module("src.query.retrieve")
    router_mod = importlib.import_module("src.query.router")
    rerankers_module = importlib.import_module("src.query.rerankers")
    query_logger_mod = importlib.import_module("src.query.query_logger")

    # query_pipeline computes audit_dir_rel = audit.dir_path.relative_to(
    # _PROJECT_ROOT). Patch _PROJECT_ROOT so audit dir lives under tmp_path.
    monkeypatch.setattr(query_logger_mod, "_PROJECT_ROOT", tmp_path)

    # Isolate audit writes + queries.jsonl to tmp_path
    monkeypatch.setenv("QUERY_LOG_ROOT", str(tmp_path / "query_logs"))
    queries_log = tmp_path / "queries.jsonl"
    monkeypatch.setattr(gen_mod, "_QUERIES_LOG", queries_log)
    monkeypatch.setattr(gen_mod, "_LOGS_DIR", tmp_path)

    # Router LLM — patch call_text (the new router uses call_text, not _generate_text_via_provider)
    router_mock = MagicMock(return_value=_fake_router_result())
    monkeypatch.setattr(router_mod, "call_text", router_mock)

    # Milvus search — index by sub_query_idx so per-sub-query lists are distinct
    sq_call_count = {"i": 0}

    def fake_milvus(query: str, collection: str, notebook: str, top_k: int) -> list[dict[str, Any]]:
        idx = sq_call_count["i"]
        sq_call_count["i"] += 1
        return _fake_milvus_search_per_sq(query, sub_query_idx=idx, n_per_sq=30)

    monkeypatch.setattr(retrieve_mod, "milvus_search_only", fake_milvus)
    monkeypatch.setattr(retrieve_mod, "_lookup_parents", _fake_lookup_parents)

    # Reranker (singleton — patch the getter at every import site)
    monkeypatch.setattr(retrieve_mod, "get_reranker", lambda: _fake_reranker)
    monkeypatch.setattr(rerankers_module, "get_reranker", lambda: _fake_reranker)

    # Synthesis (function-level patch)
    monkeypatch.setattr(gen_mod, "synthesize_answer", _fake_synthesize_answer)

    return {
        "router_mock": router_mock,
        "queries_log": queries_log,
        "tmp_path": tmp_path,
        "gen_mod": gen_mod,
        "retrieve_mod": retrieve_mod,
        "router_mod": router_mod,
    }


def _find_audit_dir(tmp_path: Path) -> Path:
    """Find the single per-query audit dir created under tmp_path/query_logs/<date>/."""
    root = tmp_path / "query_logs"
    date_dirs = [d for d in root.iterdir() if d.is_dir()]
    assert len(date_dirs) == 1, f"expected exactly one date dir, got {date_dirs}"
    qdirs = [d for d in date_dirs[0].iterdir() if d.is_dir()]
    assert len(qdirs) == 1, f"expected exactly one query dir, got {qdirs}"
    return qdirs[0]


class TestPipelineComposition:
    def test_stage_order_and_audit_files(
        self, _patched_pipeline: dict[str, Any], _fake_reranker: _FakeReranker
    ) -> None:
        """Happy path: 3 sub-queries → RRF → CE → MMR → synth. All stage files written."""
        response, trace = run_query_pipeline(
            "How does AS extend to multi-asset under QRH?",
            collection="trading",
            notebook="trading",
            use_crag=False,
        )

        assert response is not None
        audit_dir = _find_audit_dir(_patched_pipeline["tmp_path"])

        # Every expected stage file exists
        files = {p.name for p in audit_dir.iterdir() if p.is_file()}
        assert "01_decompose.json" in files
        assert "03_milvus.json" in files
        assert "04_rerank.json" in files
        assert "06_synthesis.json" in files
        assert "meta.json" in files

        # Trace shape
        assert "decomposition" in trace
        assert "retrieval" in trace
        assert "synthesis" in trace
        assert "rrf" in trace["retrieval"]
        assert trace["retrieval"]["rrf"]["n_unique_after_merge"] > 0
        assert "policy" in trace

    def test_milvus_stage_records_rrf_merge_provenance(
        self, _patched_pipeline: dict[str, Any]
    ) -> None:
        run_query_pipeline(
            "test query",
            collection="trading",
            notebook="trading",
            use_crag=False,
        )
        audit_dir = _find_audit_dir(_patched_pipeline["tmp_path"])
        milvus_payload = json.loads((audit_dir / "03_milvus.json").read_text())

        # Per-sub-query candidates preserved
        assert "sub_queries" in milvus_payload
        assert len(milvus_payload["sub_queries"]) == 3

        # RRF merge metadata captured
        rrf = milvus_payload.get("rrf_merge")
        assert rrf is not None
        assert rrf["k_constant"] == 60
        assert rrf["n_sub_queries"] == 3
        assert rrf["n_unique_after_merge"] > 0
        assert "top_post_rrf" in rrf

        # Shared chunks should rank highest after RRF (appeared in 2+ lists)
        top_3_ids = [c["id"] for c in rrf["top_post_rrf"][:3]]
        assert any(cid.startswith("shared") for cid in top_3_ids), (
            f"RRF should boost shared chunks; top-3 was {top_3_ids}"
        )

    def test_cross_encoder_called_with_original_query_not_sub_queries(
        self, _patched_pipeline: dict[str, Any], _fake_reranker: _FakeReranker
    ) -> None:
        """The whole point of the RAG-Fusion refactor: CE judges against the
        user's actual question, not the LLM's paraphrased sub-queries."""
        original_query = "How does AS extend to multi-asset under QRH?"
        run_query_pipeline(
            original_query,
            collection="trading",
            notebook="trading",
            use_crag=False,
        )
        # Reranker must have been called exactly once
        assert len(_fake_reranker.calls_seen) == 1
        pairs_seen = _fake_reranker.calls_seen[0]
        # EVERY pair must use the original query, never a sub-query
        for q, _text in pairs_seen:
            assert q == original_query, f"CE got sub-query {q!r}, expected original"

    def test_queries_jsonl_records_audit_dir(self, _patched_pipeline: dict[str, Any]) -> None:
        run_query_pipeline(
            "test query",
            collection="trading",
            notebook="trading",
            use_crag=False,
        )
        log = _patched_pipeline["queries_log"]
        assert log.is_file()
        rec = json.loads(log.read_text().strip().split("\n")[-1])
        assert "audit_dir" in rec
        assert rec["audit_dir"]
        # audit_dir is recorded relative to _PROJECT_ROOT (= tmp_path in test).
        # Eval resolves it the same way to read stage_recall.
        full_audit_path = _patched_pipeline["tmp_path"] / rec["audit_dir"]
        assert (full_audit_path / "01_decompose.json").is_file(), (
            f"resolved audit_dir does not exist: {full_audit_path}"
        )

    def test_router_use_mmr_override_propagates_to_policy(
        self, monkeypatch: pytest.MonkeyPatch, _patched_pipeline: dict[str, Any]
    ) -> None:
        """Router emits use_mmr=False → resolved policy.use_mmr=False → trace records it."""
        monkeypatch.setattr(
            _patched_pipeline["router_mod"],
            "call_text",
            MagicMock(return_value=_fake_router_result(use_mmr=False)),
        )
        _, trace = run_query_pipeline(
            "test query",
            collection="trading",
            notebook="trading",
            use_crag=False,
        )
        assert trace["policy"]["use_mmr"] is False
        assert trace["policy"]["use_mmr_source"] == "router_llm"

    def test_router_use_mmr_null_falls_through_to_baseline(
        self, monkeypatch: pytest.MonkeyPatch, _patched_pipeline: dict[str, Any]
    ) -> None:
        # Default fake router returns use_mmr=None
        _, trace = run_query_pipeline(
            "test query",
            collection="trading",
            notebook="trading",
            use_crag=False,
        )
        assert trace["policy"]["use_mmr_source"] == "intent_baseline"

    def test_crag_retry_path_records_grader_stage(
        self, monkeypatch: pytest.MonkeyPatch, _patched_pipeline: dict[str, Any]
    ) -> None:
        """When grader returns score < threshold, pipeline reformulates + retries
        and records 07_grader stage. CRAG_THRESHOLD is 0.7 in production config."""
        import importlib

        grader_mod = importlib.import_module("src.query.grader")
        monkeypatch.setattr(
            grader_mod,
            "grade_groundedness",
            lambda answer, contexts: (0.3, "fake-low-score", None),
        )
        monkeypatch.setattr(
            grader_mod,
            "reformulate_query",
            lambda q, a: ("reformulated query for retry", None),
        )
        _, trace = run_query_pipeline(
            "test query",
            collection="trading",
            notebook="trading",
            use_crag=True,
        )
        assert "crag" in trace
        assert trace["crag"]["first_score"] == 0.3
        assert trace["crag"].get("retried") is True

    def test_no_chunks_returns_none_response_without_crashing(
        self, monkeypatch: pytest.MonkeyPatch, _patched_pipeline: dict[str, Any]
    ) -> None:
        """Empty Milvus result → trace says synthesis None, no crash."""
        monkeypatch.setattr(
            _patched_pipeline["retrieve_mod"],
            "milvus_search_only",
            lambda **kwargs: [],
        )
        response, trace = run_query_pipeline(
            "no-results query",
            collection="trading",
            notebook="trading",
            use_crag=False,
        )
        assert response is None
        assert trace["synthesis"] is None
