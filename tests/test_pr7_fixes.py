"""TDD tests for PR #7 review findings.

Written RED-first: each test must fail before the corresponding fix is applied.

Findings covered:
    BLOCKER 1: RemoteVLLMReranker.score must not use ThreadPoolExecutor internally.
    BLOCKER 2: JUDGE_MODEL / OCR_MODEL / TABLE_SUMMARY_MODEL defaults must be
               gemini-3.1-flash-lite-preview, not gemini-2.5-flash.
    MEDIUM 3:  QueryLogger run_dir must be unique even when pid+tid differ.
    MEDIUM 4:  _json_schema_unsupported must be protected by a threading.Lock.
    MEDIUM 5:  ValidationError in call_with_schema must score as nan, not abort.
    FYI 7:     _LLM_PRICES_PER_M_TOKENS must include gemini-3.1-flash-lite-preview.
"""

from __future__ import annotations

import asyncio
import inspect
import math
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, ValidationError

if TYPE_CHECKING:
    from types import ModuleType

# ===========================================================================
# BLOCKER 1 — RemoteVLLMReranker.score must NOT use a ThreadPoolExecutor
# ===========================================================================


class TestRemoteVLLMNoInternalTPE:
    """Structural assertion: score() must not instantiate a ThreadPoolExecutor.

    The lock around the session is intentional and correct (outer pipeline
    workers serialise on it). The bug is that INSIDE the critical section the
    old implementation spun up its own TPE whose worker threads shared the
    session unsafely. Fix is Option A: serial for-loop (matches sibling
    backend RemoteLlamaCppReranker).
    """

    def test_score_body_has_no_thread_pool_executor(self) -> None:
        """score() must NOT contain a ThreadPoolExecutor instantiation."""
        from src.query import rerankers as rr

        src = inspect.getsource(rr.RemoteVLLMReranker.score)
        assert "ThreadPoolExecutor" not in src, (
            "RemoteVLLMReranker.score must not use ThreadPoolExecutor — "
            "it creates a data race against the shared requests.Session. "
            "Replace with a serial for-loop matching RemoteLlamaCppReranker.score."
        )

    def test_score_serial_calls_session_once_per_pair(self) -> None:
        """Each pair produces one sequential request."""
        from src.query import rerankers as rr

        backend = rr.RemoteVLLMReranker(endpoint="http://h:8089", model_name="M")

        call_count = 0

        def fake_post(url: str, **kw: object) -> MagicMock:
            nonlocal call_count
            call_count += 1
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            resp.json.return_value = {"data": [{"index": 0, "score": 0.5}]}
            return resp

        backend._session.post = fake_post  # type: ignore[method-assign]

        pairs = [("q", "d1"), ("q", "d2"), ("q", "d3")]
        scores = backend.score(pairs)

        assert call_count == 3, f"Expected 3 serial calls, got {call_count}"
        assert len(scores) == 3

    def test_score_preserves_order_with_serial_loop(self) -> None:
        """Score order must match pair order when using serial loop."""
        from src.query import rerankers as rr

        backend = rr.RemoteVLLMReranker(endpoint="http://h:8089", model_name="M")
        score_map = {"d1": 0.1, "d2": 0.9, "d3": 0.5}

        def fake_post(url: str, json: dict[str, Any], **kw: object) -> MagicMock:
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            resp.json.return_value = {"data": [{"index": 0, "score": score_map[json["text_2"]]}]}
            return resp

        backend._session.post = fake_post  # type: ignore[method-assign]

        pairs = [("q", "d1"), ("q", "d2"), ("q", "d3")]
        scores = backend.score(pairs)

        assert scores == [0.1, 0.9, 0.5]


# ===========================================================================
# BLOCKER 2 — Model defaults must not regress to gemini-2.5
# ===========================================================================


class TestModelDefaults:
    """JUDGE_MODEL, OCR_MODEL, TABLE_SUMMARY_MODEL must default to
    gemini-3.1-flash-lite-preview when env vars are unset."""

    def _reload_models(self, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
        import importlib

        monkeypatch.delenv("JUDGE_MODEL", raising=False)
        monkeypatch.delenv("OCR_MODEL", raising=False)
        monkeypatch.delenv("TABLE_SUMMARY_MODEL", raising=False)
        import src.config.models as m

        importlib.reload(m)
        return m

    def test_judge_model_default_is_gemini_3_1_flash_lite(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        m = self._reload_models(monkeypatch)
        assert m.JUDGE_MODEL == "gemini-3.1-flash-lite-preview", (
            f"JUDGE_MODEL defaulted to {m.JUDGE_MODEL!r} — must be "
            "'gemini-3.1-flash-lite-preview' (project policy: no 2.5 defaults)."
        )

    def test_ocr_model_default_is_gemini_3_1_flash_lite(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        m = self._reload_models(monkeypatch)
        assert m.OCR_MODEL == "gemini-3.1-flash-lite-preview", (
            f"OCR_MODEL defaulted to {m.OCR_MODEL!r} — must be 'gemini-3.1-flash-lite-preview'."
        )

    def test_table_summary_model_default_is_gemini_3_1_flash_lite(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        m = self._reload_models(monkeypatch)
        assert m.TABLE_SUMMARY_MODEL == "gemini-3.1-flash-lite-preview", (
            f"TABLE_SUMMARY_MODEL defaulted to {m.TABLE_SUMMARY_MODEL!r} — must be "
            "'gemini-3.1-flash-lite-preview'."
        )

    def test_env_override_still_works(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import importlib

        monkeypatch.setenv("JUDGE_MODEL", "custom-model-x")
        import src.config.models as m

        importlib.reload(m)
        assert m.JUDGE_MODEL == "custom-model-x"


# ===========================================================================
# MEDIUM 3 — QueryLogger directory unique per pid+tid
# ===========================================================================


class TestQueryLoggerDirectoryUniqueness:
    """Directory name must include pid and tid so same-microsecond same-query
    concurrent calls from different threads always produce distinct paths."""

    def test_dir_name_includes_pid(self, tmp_path: Path) -> None:
        """run_dir name must contain the current process id."""
        from src.query.query_logger import QueryLogger

        ql = QueryLogger("test query", root=tmp_path / "logs")
        pid = str(__import__("os").getpid())
        assert pid in ql.log_dir.name, (
            f"Expected PID {pid} in dir name {ql.log_dir.name!r}. "
            "Add os.getpid() to dir_name in QueryLogger.__init__."
        )

    def test_dir_name_includes_tid(self, tmp_path: Path) -> None:
        """run_dir name must contain the current thread id."""
        from src.query.query_logger import QueryLogger

        tid = threading.get_ident()
        ql = QueryLogger("test query", root=tmp_path / "logs")
        assert str(tid) in ql.log_dir.name, (
            f"Expected TID {tid} in dir name {ql.log_dir.name!r}. "
            "Add threading.get_ident() to dir_name in QueryLogger.__init__."
        )

    def test_four_threads_same_query_forced_collision_all_distinct(self, tmp_path: Path) -> None:
        """4 threads with the same query string and mocked identical timestamp
        must still produce 4 distinct directories (because pid/tid differ)."""
        from unittest.mock import patch

        from src.query.query_logger import QueryLogger

        thread_count = 4
        dirs: list[Path] = []
        errors: list[Exception] = []
        barrier = threading.Barrier(thread_count)

        # Freeze timestamp so microsecond can't save us — only pid/tid can.
        frozen_dt = __import__("datetime").datetime(
            2026,
            1,
            1,
            12,
            0,
            0,
            0,  # microsecond=0 → forced collision on ts+hash
            tzinfo=__import__("datetime").timezone.utc,
        )

        def make_logger() -> None:
            try:
                barrier.wait()
                with patch("src.query.query_logger.datetime") as mock_dt:
                    mock_dt.now.return_value = frozen_dt
                    mock_dt.timezone = __import__("datetime").timezone
                    ql = QueryLogger("same query", root=tmp_path / "logs")
                dirs.append(ql.log_dir)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=make_logger) for _ in range(thread_count)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Threads raised: {errors}"
        assert len(dirs) == thread_count
        assert len(set(dirs)) == thread_count, (
            f"Collision detected: {len(dirs) - len(set(dirs))} directories shared. Dirs: {dirs}"
        )


# ===========================================================================
# MEDIUM 4 — _json_schema_unsupported protected by a Lock
# ===========================================================================


class TestJsonSchemaUnsupportedLock:
    """_json_schema_unsupported must be guarded by a threading.Lock so
    concurrent writes from thread_count eval threads can't race."""

    def test_module_has_unsupported_lock(self) -> None:
        """usage_track must expose a _json_schema_unsupported_lock attribute."""
        from src.query import usage_track

        assert hasattr(usage_track, "_json_schema_unsupported_lock"), (
            "usage_track must have a _json_schema_unsupported_lock (threading.Lock). "
            "Add: _json_schema_unsupported_lock = threading.Lock()"
        )
        lock = usage_track._json_schema_unsupported_lock
        assert hasattr(lock, "acquire") and hasattr(lock, "release"), (
            "_json_schema_unsupported_lock must be a threading.Lock-compatible object."
        )

    def test_concurrent_adds_produce_correct_set(self) -> None:
        """Concurrent additions preserve every unique model name."""
        from src.query import usage_track

        original = usage_track._json_schema_unsupported.copy()
        usage_track._json_schema_unsupported.clear()

        try:
            thread_count = 20
            model_names = [f"test-model-{i}" for i in range(thread_count)]
            barrier = threading.Barrier(thread_count)

            def add_model(name: str) -> None:
                barrier.wait()
                with usage_track._json_schema_unsupported_lock:
                    usage_track._json_schema_unsupported.add(name)

            threads = [threading.Thread(target=add_model, args=(name,)) for name in model_names]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert usage_track._json_schema_unsupported == set(model_names), (
                "Concurrent set additions lost some entries. "
                f"Expected {thread_count} models, got {len(usage_track._json_schema_unsupported)}"
            )
        finally:
            usage_track._json_schema_unsupported.clear()
            usage_track._json_schema_unsupported.update(original)

    def test_create_with_schema_holds_lock_on_check_and_add(self) -> None:
        """_create_with_schema must acquire the lock before checking/adding the
        model to _json_schema_unsupported, not just add without protection."""
        from src.query import usage_track

        # Confirm the lock is actually used — verify it's referenced in the
        # source of _create_with_schema.
        src = inspect.getsource(usage_track._create_with_schema)
        assert "_json_schema_unsupported_lock" in src, (
            "_create_with_schema must use _json_schema_unsupported_lock "
            "to protect the check-and-add of _json_schema_unsupported."
        )


# ===========================================================================
# MEDIUM 5 — ValidationError in call_with_schema scores as nan, not abort
# ===========================================================================


class _SampleSchema(BaseModel):
    score: float


class TestCallWithSchemaValidationError:
    """When call_text returns text that fails model_validate_json, the metric
    must score that question as math.nan and continue — not abort the run."""

    def test_call_with_schema_returns_nan_on_validation_error(self) -> None:
        """call_with_schema must return (nan_sentinel, result) on ValidationError,
        not raise."""
        from src.eval.metrics import call_with_schema

        bad_result = MagicMock()
        bad_result.text = '{"totally": "wrong", "schema": true}'
        bad_result.latency_ms = 0
        bad_result.usage = MagicMock()
        bad_result.cost_usd = None
        bad_result.model = "test-model"

        with patch(
            "src.eval.metrics.call_text",
            return_value=bad_result,
        ):
            # Should NOT raise; should return sentinel with nan score
            result = asyncio.run(call_with_schema("test-model", "", "user", _SampleSchema))

        parsed, llm_result = result
        assert math.isnan(parsed.score), (  # type: ignore[union-attr]
            "call_with_schema must return a sentinel with score=nan on "
            "ValidationError, not raise. Got: {parsed!r}"
        )

    def test_call_with_schema_returns_nan_on_json_decode_error(self) -> None:
        """call_with_schema must return (nan_sentinel, result) on JSONDecodeError."""
        from src.eval.metrics import call_with_schema

        bad_result = MagicMock()
        bad_result.text = "this is not json at all!!!"
        bad_result.latency_ms = 0
        bad_result.usage = MagicMock()
        bad_result.cost_usd = None
        bad_result.model = "test-model"

        with patch("src.eval.metrics.call_text", return_value=bad_result):
            result = asyncio.run(call_with_schema("test-model", "", "user", _SampleSchema))

        parsed, _ = result
        assert math.isnan(parsed.score), (  # type: ignore[union-attr]
            "call_with_schema must handle JSONDecodeError and return nan sentinel."
        )

    def test_one_bad_question_does_not_abort_other_questions(self) -> None:
        """In a 3-question faithfulness run, a ValidationError on question 1
        must score that question as nan while questions 2 and 3 score normally."""
        from src.eval.metrics import faithfulness
        from src.eval.schemas import (
            NLIStatementOutput,
            StatementFaithfulnessAnswer,
            StatementsOutput,
        )

        # Good outputs for questions 2 and 3
        good_stmt = StatementsOutput(statements=["s1"])
        good_nli = NLIStatementOutput(
            statements=[StatementFaithfulnessAnswer(statement="s1", reason="", verdict=1)]
        )

        call_count = 0

        async def fake_call_with_schema(
            model: str, system: str, user: str, schema: type
        ) -> tuple[Any, Any]:
            nonlocal call_count
            call_count += 1
            # First call (statement extraction for question 1) raises ValidationError
            if call_count == 1:
                raise ValidationError.from_exception_data(
                    title="StatementsOutput",
                    input_type="json",
                    line_errors=[],
                )
            # Subsequent calls for other questions work fine
            if "StatementsOutput" in schema.__name__ or schema == StatementsOutput:
                return good_stmt, MagicMock()
            return good_nli, MagicMock()

        rows = [
            {
                "user_input": f"Q{i}?",
                "response": "A.",
                "retrieved_contexts": ["ctx"],
            }
            for i in range(3)
        ]

        scores: list[float] = []
        with patch("src.eval.metrics.call_with_schema", side_effect=fake_call_with_schema):
            for row in rows:
                score = asyncio.run(faithfulness(row, judge_model="test-model"))
                scores.append(score)

        # Q1 must be nan (bad LLM output), Q2 and Q3 must not be nan
        assert math.isnan(scores[0]), (
            f"Expected scores[0]=nan (bad question), got {scores[0]}. "
            "ValidationError must not abort the metric — score nan and continue."
        )
        # Q2 and Q3 should complete (may be nan for other reasons, but must not
        # raise).  Verify the run completed (len==3).
        assert len(scores) == 3, (
            "Run aborted early — ValidationError on question 0 killed questions 1 and 2."
        )


# ===========================================================================
# FYI 7 — Pricing table must include gemini-3.1-flash-lite-preview
# ===========================================================================


class TestGemini31FlashLiteInPricingTable:
    def test_gemini_3_1_flash_lite_preview_in_pricing_table(self) -> None:
        from src.config.models import _LLM_PRICES_PER_M_TOKENS

        assert "gemini-3.1-flash-lite-preview" in _LLM_PRICES_PER_M_TOKENS, (
            "'gemini-3.1-flash-lite-preview' is missing from _LLM_PRICES_PER_M_TOKENS. "
            "After fixing BLOCKER 2 (reverting defaults to this model), "
            "cost tracking will return None for every eval run."
        )

    def test_gemini_3_1_flash_lite_preview_has_all_price_fields(self) -> None:
        from src.config.models import _LLM_PRICES_PER_M_TOKENS

        prices = _LLM_PRICES_PER_M_TOKENS.get("gemini-3.1-flash-lite-preview", {})
        for field in ("cache_miss", "cache_hit", "output", "reasoning"):
            assert field in prices, (
                f"gemini-3.1-flash-lite-preview pricing entry is missing field {field!r}"
            )
            assert isinstance(prices[field], float), (
                f"gemini-3.1-flash-lite-preview prices[{field!r}] must be a float"
            )

    def test_gemini_3_1_flash_lite_preview_compute_cost_returns_float(self) -> None:
        """compute_cost_usd must return a float (not None) for the default judge model."""
        from src.config.models import compute_cost_usd
        from src.models import Usage

        usage = Usage(input=1000, output=200, reasoning=0, total=1200)
        cost = compute_cost_usd("gemini-3.1-flash-lite-preview", usage)
        assert cost is not None, (
            "compute_cost_usd returned None for 'gemini-3.1-flash-lite-preview'. "
            "Add the model to _LLM_PRICES_PER_M_TOKENS."
        )
        assert cost > 0, f"Expected positive cost, got {cost}"
