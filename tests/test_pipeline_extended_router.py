"""Pipeline integration tests for the extended router (steps 10-12).

Step 10: Pipeline wires plan.hyde_doc as extra sub-query; skips legacy
         step_back_query when plan.stepback is non-null.
Step 11: Audit payload includes new router fields.
Step 12: Report renderer shows hyde_doc / stepback when present; omits when null.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.models import Usage
from src.query.router import RoutingPlan, ToolCall
from src.query.usage_track import LLMCallResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_plan(
    *,
    intent: str = "multi_hop",
    sub_queries: list[str] | None = None,
    hyde_doc: str | None = None,
    stepback: str | None = None,
) -> RoutingPlan:
    return RoutingPlan(
        user_query="test query",
        intent=intent,  # type: ignore[arg-type]
        entities=["A", "B"],
        nodes=[ToolCall(tool="milvus", sub_queries=sub_queries or ["q1", "q2"])],
        use_mmr=True,
        use_mmr_reason="test",
        hyde_doc=hyde_doc,
        stepback=stepback,
        disambiguation=[],
        filters={},
    )


def _fake_llm_result(text: str = "") -> LLMCallResult:
    return LLMCallResult(
        text=text,
        latency_ms=1,
        usage=Usage(input=0, output=0, reasoning=0, total=0),
        raw_response=None,
        cost_usd=0.0,
        model="fake",
    )


# ---------------------------------------------------------------------------
# Step 10: Pipeline sub-query wiring
# ---------------------------------------------------------------------------


class TestPipelineSubQueryWiring:
    def test_hyde_doc_appended_to_subqueries(self) -> None:
        """When plan.hyde_doc is non-null, it is appended to the milvus sub-queries."""
        from src.query.query_pipeline import _extract_sub_queries_from_plan

        plan = _make_plan(sub_queries=["q1", "q2"], hyde_doc="hypothetical passage text")
        sub_queries = _extract_sub_queries_from_plan(plan)

        assert "hypothetical passage text" in sub_queries
        assert "q1" in sub_queries
        assert "q2" in sub_queries
        assert len(sub_queries) == 3

    def test_legacy_stepback_skipped_when_router_emits(self) -> None:
        """When plan.stepback is non-null, legacy step_back_query is NOT called."""
        from src.query.query_pipeline import _should_call_legacy_stepback

        plan = _make_plan(stepback="What is multi-hop?")
        assert _should_call_legacy_stepback(plan) is False

    def test_legacy_stepback_called_when_router_returns_null(self) -> None:
        """When plan.stepback is None, legacy step_back_query IS called (if use_stepback=True)."""
        from src.query.query_pipeline import _should_call_legacy_stepback

        plan = _make_plan(stepback=None)
        # With stepback=None, the legacy path should be available
        assert _should_call_legacy_stepback(plan) is True


# ---------------------------------------------------------------------------
# Step 11: Audit payload extension
# ---------------------------------------------------------------------------


class TestAuditPayloadExtension:
    def test_decompose_audit_payload_includes_new_fields(self) -> None:
        """The decompose audit payload written to 01_decompose.json includes new fields."""
        from src.query.query_pipeline import _build_decompose_payload

        plan = _make_plan(
            intent="multi_hop",
            sub_queries=["q1", "q2"],
            hyde_doc="hypothetical passage",
            stepback="foundational question",
        )
        payload = _build_decompose_payload(
            plan=plan,
            sub_queries=["q1", "q2", "hypothetical passage"],
            latency_ms=100,
        )

        assert payload["hyde_doc"] == "hypothetical passage"
        assert payload["stepback"] == "foundational question"
        assert payload["disambiguation"] == []
        assert payload["filters"] == {}

    def test_decompose_audit_payload_populates_usage_and_cost_from_llm_result(self) -> None:
        """Per CLAUDE.md NON-NEGOTIABLE: usage and cost must reach the audit payload.

        When an LLMCallResult is passed to _build_decompose_payload, the resulting
        payload must contain the real usage dict + cost_usd, not hardcoded None.
        Otherwise the router stage is invisible in meta.totals (CLAUDE.md rule 3).
        """
        from src.query.query_pipeline import _build_decompose_payload

        plan = _make_plan(sub_queries=["q1"])
        usage = Usage(input=100, output=50, reasoning=0, total=150)
        llm_result = LLMCallResult(
            text="{}",
            latency_ms=123,
            usage=usage,
            raw_response=None,
            cost_usd=0.0042,
            model="deepseek-v4-flash",
        )

        payload = _build_decompose_payload(
            plan=plan,
            sub_queries=["q1"],
            latency_ms=123,
            llm_result=llm_result,
        )

        assert payload["usage"] is not None
        assert payload["usage"]["input"] == 100
        assert payload["usage"]["output"] == 50
        assert payload["usage"]["total"] == 150
        assert payload["cost_usd"] == 0.0042


class TestFiltersDeferralTODO:
    """Per PR #30 review (HIGH #3): filters are emitted + logged but NOT wired
    into Milvus expr yet. The TODO marker pins that decision in source so the
    follow-up is discoverable.
    """

    def test_filters_todo_marker_present_in_query_pipeline(self) -> None:
        """The TODO(router-filters) marker must remain in query_pipeline.py
        until the Milvus wiring follow-up lands.
        """
        from pathlib import Path

        src = (Path(__file__).parent.parent / "src" / "query" / "query_pipeline.py").read_text(
            encoding="utf-8"
        )
        assert "TODO(router-filters)" in src, (
            "TODO(router-filters) marker missing. Per PR #30 review: filters must "
            "remain unwired to Milvus in this PR; the marker tracks the follow-up."
        )


class TestRouterUsageAccumulation:
    """Per CLAUDE.md NON-NEGOTIABLE rule 3: every LLM stage MUST call
    accumulate_usage(stage, usage, cost_usd, latency_ms) or it is invisible
    in meta.totals.

    The router stage was previously hardcoding `usage: None, cost_usd: None`
    in its decompose payload and NOT calling accumulate_usage. This test
    pins the fix in place.
    """

    def test_router_call_accumulates_usage_into_audit(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """When route_query returns an LLMCallResult, the pipeline must call
        audit.accumulate_usage("decompose", result.usage, result.cost_usd, result.latency_ms).
        """
        from src.query import query_logger as ql_mod
        from src.query import query_pipeline as qp_mod
        from src.query import retrieve as retrieve_mod
        from src.query import router as router_mod

        # Fake router LLM result with non-zero usage/cost so we can assert propagation
        usage = Usage(input=100, output=50, reasoning=0, total=150)
        fake_router_result = LLMCallResult(
            text=json.dumps(
                {
                    "intent": "multi_hop",
                    "tool_payloads": {"milvus": ["q1"]},
                }
            ),
            latency_ms=123,
            usage=usage,
            raw_response=None,
            cost_usd=0.0042,
            model="deepseek-v4-flash",
        )
        monkeypatch.setattr(router_mod, "call_text", MagicMock(return_value=fake_router_result))

        # Stub Milvus → empty so pipeline exits after the decompose stage cleanly
        monkeypatch.setattr(retrieve_mod, "milvus_search_only", lambda *_a, **_kw: [])

        # Isolate audit writes
        monkeypatch.setenv("QUERY_LOG_ROOT", str(tmp_path / "logs"))
        monkeypatch.setattr(ql_mod, "_PROJECT_ROOT", tmp_path)

        # Capture accumulate_usage calls
        accumulate_calls: list[tuple[str, Usage, float | None, int]] = []
        original_init = ql_mod.QueryLogger.__init__

        def _patched_init(self: ql_mod.QueryLogger, *args: object, **kwargs: object) -> None:
            original_init(self, *args, **kwargs)
            original_accumulate = self.accumulate_usage

            def _capture(stage: str, u: Usage, cost: float | None, lat: int = 0) -> None:
                accumulate_calls.append((stage, u, cost, lat))
                original_accumulate(stage, u, cost, lat)

            self.accumulate_usage = _capture  # type: ignore[method-assign]

        monkeypatch.setattr(ql_mod.QueryLogger, "__init__", _patched_init)

        # Run pipeline — downstream is stubbed, so the decompose stage is the
        # only LLM stage exercised.
        _, trace = qp_mod.run_query_pipeline(
            "test query",
            collection="trading",
            notebook="trading",
            use_crag=False,
        )

        # Assert: accumulate_usage("decompose", ...) was called with the router's usage
        decompose_calls = [c for c in accumulate_calls if c[0] == "decompose"]
        assert len(decompose_calls) == 1, (
            f"Expected one accumulate_usage('decompose', ...) call. Got: {accumulate_calls}"
        )
        _, captured_usage, captured_cost, captured_lat = decompose_calls[0]
        assert captured_usage == usage
        assert captured_cost == 0.0042
        assert captured_lat == 123

        # Assert: the trace decompose_payload has real usage + cost, not None
        decompose_payload = trace["decomposition"]
        assert decompose_payload["usage"] is not None
        assert decompose_payload["cost_usd"] == 0.0042


# ---------------------------------------------------------------------------
# Step 12: Report renderer
# ---------------------------------------------------------------------------


class TestReportRenderer:
    def _make_log_dir(
        self,
        hyde_doc: str | None = None,
        stepback: str | None = None,
    ) -> Path:
        """Create a minimal log directory with meta.json + 01_decompose.json."""
        d = Path(tempfile.mkdtemp())
        meta = {
            "query": "test query",
            "ts_started": "2026-05-13T00:00:00Z",
            "outcome": {},
            "totals": {},
            "config": {},
            "pid": 1,
            "tid": 1,
            "host": "test",
        }
        (d / "meta.json").write_text(json.dumps(meta))

        decompose = {
            "router": True,
            "intent": "multi_hop",
            "entities": ["A"],
            "sub_queries": ["q1"],
            "parsed_subqueries": ["q1"],
            "use_mmr": True,
            "use_mmr_reason": "test",
            "latency_ms": 100,
            "model": "deepseek-v4-flash",
            "usage": None,
            "cost_usd": None,
            "hyde_doc": hyde_doc,
            "stepback": stepback,
            "disambiguation": [],
            "filters": {},
        }
        (d / "01_decompose.json").write_text(json.dumps(decompose))
        return d

    def test_report_render_shows_hyde_when_present(self) -> None:
        """When hyde_doc is present in decompose stage, render includes it."""
        from src.query.report_render import render_report

        log_dir = self._make_log_dir(hyde_doc="A hypothetical passage about X", stepback=None)
        report = render_report(log_dir)
        assert "hypothetical" in report.lower() or "hyde" in report.lower()

    def test_report_render_shows_stepback_when_present(self) -> None:
        """When stepback is present in decompose stage, render includes it."""
        from src.query.report_render import render_report

        log_dir = self._make_log_dir(hyde_doc=None, stepback="What is multi-hop reasoning?")
        report = render_report(log_dir)
        assert "multi-hop reasoning" in report or "stepback" in report.lower()

    def test_report_render_omits_hyde_when_null(self) -> None:
        """When hyde_doc is None, the report does not add a HyDE section."""
        from src.query.report_render import render_report

        log_dir = self._make_log_dir(hyde_doc=None, stepback=None)
        report = render_report(log_dir)
        # With no hyde_doc, there should be no "HyDE" header in the report
        assert "## HyDE" not in report
        assert "## Hyde" not in report


# ---------------------------------------------------------------------------
# Belt-and-suspenders fixture
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_real_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
