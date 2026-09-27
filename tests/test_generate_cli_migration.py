"""Tests for the CLI → run_query_pipeline migration (PR #31).

Background: src/query/generate.py::_cli used to run its own inline pipeline
(legacy decompose_query + step_back_query), bypassing src/query/router.py.
Result: REWRITER_EMIT_HYDE=true had no effect on CLI/MCP — the live audit
trail showed router:null, intent:null, hyde_doc:null even with env knobs on.

This file tests the migrated _cli: it must call run_query_pipeline with the
correct kwargs derived from argv, so the router fires and REWRITER_EMIT_*
flags actually propagate through to the LLM call.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from src.models import RAGResponse, Usage

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Never

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_rag_response() -> RAGResponse:
    return RAGResponse(
        query="fake query",
        answer="fake answer",
        citations=[],
        retrieved=[],
        latency_ms=10,
        usage=Usage(),
    )


@pytest.fixture(autouse=True)
def _no_real_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCliCallsRunQueryPipeline:
    """The migrated _cli must delegate the entire retrieval+synthesis pipeline
    to run_query_pipeline (which uses route_query / router.py). This is the
    bug-fix at the core of PR #31.
    """

    def test_cli_synthesize_mode_invokes_run_query_pipeline(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Default CLI flow (synthesize) must call run_query_pipeline."""
        import importlib

        gen_mod = importlib.import_module("src.query.generate")
        qp_mod = importlib.import_module("src.query.query_pipeline")

        run_query_pipeline_mock = MagicMock(
            return_value=(_fake_rag_response(), {"retrieved_chunks": []})
        )
        # Patch the symbol where _cli looks it up.
        monkeypatch.setattr(qp_mod, "run_query_pipeline", run_query_pipeline_mock)
        monkeypatch.setattr(gen_mod, "run_query_pipeline", run_query_pipeline_mock, raising=False)

        monkeypatch.setattr(
            sys,
            "argv",
            ["src.query.generate", "--query", "test query", "--collection", "trading"],
        )

        gen_mod._cli()

        assert run_query_pipeline_mock.call_count == 1, (
            "Migrated _cli must call run_query_pipeline exactly once; "
            f"instead got call_count={run_query_pipeline_mock.call_count}"
        )
        # Verify the query and collection were forwarded
        call_args = run_query_pipeline_mock.call_args
        # query is first positional or 'query' kwarg
        forwarded_query = call_args.kwargs.get("query") or (
            call_args.args[0] if call_args.args else None
        )
        forwarded_collection = call_args.kwargs.get("collection") or (
            call_args.args[1] if len(call_args.args) > 1 else None
        )
        assert forwarded_query == "test query"
        assert forwarded_collection == "trading"

    def test_cli_raw_mode_passes_synthesize_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """--raw on the CLI must propagate synthesize=False to run_query_pipeline."""
        import importlib

        gen_mod = importlib.import_module("src.query.generate")
        qp_mod = importlib.import_module("src.query.query_pipeline")

        run_query_pipeline_mock = MagicMock(return_value=(None, {"retrieved_chunks": []}))
        monkeypatch.setattr(qp_mod, "run_query_pipeline", run_query_pipeline_mock)
        monkeypatch.setattr(gen_mod, "run_query_pipeline", run_query_pipeline_mock, raising=False)

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "src.query.generate",
                "--query",
                "test query",
                "--collection",
                "trading",
                "--raw",
            ],
        )

        gen_mod._cli()

        kwargs = run_query_pipeline_mock.call_args.kwargs
        assert kwargs.get("synthesize") is False, (
            "Migrated _cli with --raw must pass synthesize=False to run_query_pipeline; "
            f"got kwargs={kwargs}"
        )

    def test_cli_raw_does_not_disable_decompose(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """--raw must NOT couple to decompose=False. Regression for live-smoke
        finding (post-PR #31) that --raw silently bypassed route_query
        (router=false in 01_decompose.json) and prevented REWRITER_EMIT_* env
        knobs from firing. Only --no-decompose should disable decompose."""
        import importlib

        gen_mod = importlib.import_module("src.query.generate")
        qp_mod = importlib.import_module("src.query.query_pipeline")

        run_query_pipeline_mock = MagicMock(return_value=(None, {"retrieved_chunks": []}))
        monkeypatch.setattr(qp_mod, "run_query_pipeline", run_query_pipeline_mock)
        monkeypatch.setattr(gen_mod, "run_query_pipeline", run_query_pipeline_mock, raising=False)

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "src.query.generate",
                "--query",
                "test query",
                "--collection",
                "trading",
                "--raw",
            ],
        )

        gen_mod._cli()

        kwargs = run_query_pipeline_mock.call_args.kwargs
        assert kwargs.get("decompose") is True, (
            "--raw alone must NOT disable decompose (router-bypass regression). "
            "Only --no-decompose should set decompose=False. "
            f"got kwargs={kwargs}"
        )

    def test_cli_no_decompose_propagates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """--no-decompose must propagate decompose=False to run_query_pipeline."""
        import importlib

        gen_mod = importlib.import_module("src.query.generate")
        qp_mod = importlib.import_module("src.query.query_pipeline")

        run_query_pipeline_mock = MagicMock(
            return_value=(_fake_rag_response(), {"retrieved_chunks": []})
        )
        monkeypatch.setattr(qp_mod, "run_query_pipeline", run_query_pipeline_mock)
        monkeypatch.setattr(gen_mod, "run_query_pipeline", run_query_pipeline_mock, raising=False)

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "src.query.generate",
                "--query",
                "test query",
                "--collection",
                "trading",
                "--no-decompose",
            ],
        )

        gen_mod._cli()

        kwargs = run_query_pipeline_mock.call_args.kwargs
        assert kwargs.get("decompose") is False

    def test_cli_stepback_and_crag_flags_propagate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """--stepback / --crag must propagate use_stepback / use_crag."""
        import importlib

        gen_mod = importlib.import_module("src.query.generate")
        qp_mod = importlib.import_module("src.query.query_pipeline")

        run_query_pipeline_mock = MagicMock(
            return_value=(_fake_rag_response(), {"retrieved_chunks": []})
        )
        monkeypatch.setattr(qp_mod, "run_query_pipeline", run_query_pipeline_mock)
        monkeypatch.setattr(gen_mod, "run_query_pipeline", run_query_pipeline_mock, raising=False)

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "src.query.generate",
                "--query",
                "test query",
                "--collection",
                "trading",
                "--stepback",
                "--crag",
            ],
        )

        gen_mod._cli()

        kwargs = run_query_pipeline_mock.call_args.kwargs
        assert kwargs.get("use_stepback") is True
        assert kwargs.get("use_crag") is True

    def test_cli_top_k_overrides_propagate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """--top-k-retrieve / --top-k-rerank must propagate to run_query_pipeline."""
        import importlib

        gen_mod = importlib.import_module("src.query.generate")
        qp_mod = importlib.import_module("src.query.query_pipeline")

        run_query_pipeline_mock = MagicMock(
            return_value=(_fake_rag_response(), {"retrieved_chunks": []})
        )
        monkeypatch.setattr(qp_mod, "run_query_pipeline", run_query_pipeline_mock)
        monkeypatch.setattr(gen_mod, "run_query_pipeline", run_query_pipeline_mock, raising=False)

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "src.query.generate",
                "--query",
                "test query",
                "--collection",
                "trading",
                "--top-k-retrieve",
                "77",
                "--top-k-rerank",
                "11",
            ],
        )

        gen_mod._cli()

        kwargs = run_query_pipeline_mock.call_args.kwargs
        assert kwargs.get("top_k_retrieve") == 77
        assert kwargs.get("top_k_rerank") == 11


class TestNoLegacyStepBackCall:
    """The legacy step_back_query call must NOT be invoked from _cli anymore.
    Stepback now flows through route_query inside run_query_pipeline.
    """

    def test_cli_does_not_call_legacy_step_back_query(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import importlib

        gen_mod = importlib.import_module("src.query.generate")
        qp_mod = importlib.import_module("src.query.query_pipeline")
        decompose_mod = importlib.import_module("src.query.decompose")

        run_query_pipeline_mock = MagicMock(
            return_value=(_fake_rag_response(), {"retrieved_chunks": []})
        )
        monkeypatch.setattr(qp_mod, "run_query_pipeline", run_query_pipeline_mock)
        monkeypatch.setattr(gen_mod, "run_query_pipeline", run_query_pipeline_mock, raising=False)

        step_back_mock = MagicMock()
        monkeypatch.setattr(decompose_mod, "step_back_query", step_back_mock)

        decompose_query_mock = MagicMock()
        monkeypatch.setattr(decompose_mod, "decompose_query", decompose_query_mock)

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "src.query.generate",
                "--query",
                "test query",
                "--collection",
                "trading",
                "--stepback",
            ],
        )

        gen_mod._cli()

        assert step_back_mock.call_count == 0, (
            "Migrated _cli must NOT call legacy step_back_query; "
            "stepback comes from route_query via run_query_pipeline."
        )
        assert decompose_query_mock.call_count == 0, (
            "Migrated _cli must NOT call legacy decompose_query; "
            "decomposition comes from route_query via run_query_pipeline."
        )


class TestRunQueryPipelineSynthesizeFlag:
    """run_query_pipeline must accept a synthesize=False kwarg that skips the
    synthesis + CRAG blocks but still runs retrieval and the audit trail.
    """

    def test_synthesize_false_returns_none_response(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """run_query_pipeline(..., synthesize=False) returns (None, trace)
        without calling the synthesis LLM, but retrieval still runs.
        """
        import importlib

        retrieve_mod = importlib.import_module("src.query.retrieve")
        gen_mod = importlib.import_module("src.query.generate")
        ql_mod = importlib.import_module("src.query.query_logger")
        qp_mod = importlib.import_module("src.query.query_pipeline")

        # Stub Milvus → 0 hits (we just want to verify synthesize is skipped).
        monkeypatch.setattr(retrieve_mod, "milvus_search_only", lambda *a, **k: [])
        # If generate.generate is called, fail the test.
        called = {"generate": False}

        def _fail_if_called(*_a: object, **_kw: object) -> Never:
            called["generate"] = True
            raise RuntimeError("synthesize=False must skip generate()")

        monkeypatch.setattr(gen_mod, "generate", _fail_if_called)

        # Isolate audit dir
        monkeypatch.setenv("QUERY_LOG_ROOT", str(tmp_path / "logs"))
        monkeypatch.setattr(ql_mod, "_PROJECT_ROOT", tmp_path)

        response, _trace = qp_mod.run_query_pipeline(
            "test query",
            collection="trading",
            notebook="trading",
            use_router=False,  # avoid router LLM
            use_crag=False,
            synthesize=False,
        )

        assert response is None
        assert called["generate"] is False, "synthesize=False must skip generate()"


class TestTraceIncludesRetrievedChunks:
    """The trace returned by run_query_pipeline must include retrieved_chunks
    so the CLI can render them in --raw mode without re-running retrieval.
    """

    def test_trace_contains_retrieved_chunks_key(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        import importlib

        retrieve_mod = importlib.import_module("src.query.retrieve")
        ql_mod = importlib.import_module("src.query.query_logger")
        qp_mod = importlib.import_module("src.query.query_pipeline")

        monkeypatch.setattr(retrieve_mod, "milvus_search_only", lambda *a, **k: [])
        monkeypatch.setenv("QUERY_LOG_ROOT", str(tmp_path / "logs"))
        monkeypatch.setattr(ql_mod, "_PROJECT_ROOT", tmp_path)

        _, trace = qp_mod.run_query_pipeline(
            "test query",
            collection="trading",
            notebook="trading",
            use_router=False,
            use_crag=False,
            synthesize=False,
        )

        assert "retrieved_chunks" in trace, (
            "trace must contain 'retrieved_chunks' so callers (e.g. CLI --raw) "
            "can render them without re-running retrieval."
        )
        assert isinstance(trace["retrieved_chunks"], list)
