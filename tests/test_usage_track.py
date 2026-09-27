"""Tests for src/query/usage_track.py — schema enforcement parameter.

Slice 1 (TDD): adds a `schema: type[BaseModel] | None = None` kwarg to
`call_text()`. When set:
- Gemini: `response_schema=schema` is passed in `GenerateContentConfig`,
  along with `response_mime_type="application/json"`.
- OpenAI-compat: `response_format={"type":"json_schema","json_schema":{...},
  "strict":True}` is passed.
- DeepSeek may not support `json_schema`; on `BadRequestError` we fall back
  to `{"type":"json_object"}` (cached per-process, per-model).

The point: no parse-and-retry loop on the caller side. Provider rejects
malformed output → caller raises immediately.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from pydantic import BaseModel


class _DemoSchema(BaseModel):
    """Minimal schema used by these tests."""

    answer: str
    score: int


# ---------------------------------------------------------------------------
# Gemini path: response_schema lands in GenerateContentConfig
# ---------------------------------------------------------------------------


class TestGeminiSchema:
    def _mock_gemini_response(self) -> MagicMock:
        resp = MagicMock()
        resp.text = '{"answer":"hi","score":1}'
        resp.usage_metadata = MagicMock(
            prompt_token_count=10,
            candidates_token_count=5,
            thoughts_token_count=0,
            total_token_count=15,
            cached_content_token_count=0,
        )
        return resp

    def test_schema_passes_response_schema_to_gemini(self) -> None:
        from src.query import usage_track

        with (
            patch.object(usage_track.config, "validate_api_key"),
            patch.object(usage_track.config, "GEMINI_API_KEY", "test-key"),
            patch.object(usage_track.genai, "Client") as mock_client_cls,
            patch.object(usage_track.types, "GenerateContentConfig") as mock_cfg,
        ):
            client = mock_client_cls.return_value
            client.models.generate_content.return_value = self._mock_gemini_response()

            usage_track.call_text(
                model="gemini-3-flash-preview",
                system="sys",
                user="usr",
                json_mode=False,
                temperature=0.0,
                schema=_DemoSchema,
            )

            assert mock_cfg.called, "GenerateContentConfig must be invoked"
            kwargs = mock_cfg.call_args.kwargs
            assert kwargs.get("response_schema") is _DemoSchema, (
                f"Gemini path must pass response_schema=_DemoSchema; got {kwargs!r}"
            )
            assert kwargs.get("response_mime_type") == "application/json", (
                "Schema mode forces JSON mime type"
            )

    def test_no_schema_does_not_pass_response_schema(self) -> None:
        from src.query import usage_track

        with (
            patch.object(usage_track.config, "validate_api_key"),
            patch.object(usage_track.config, "GEMINI_API_KEY", "test-key"),
            patch.object(usage_track.genai, "Client") as mock_client_cls,
            patch.object(usage_track.types, "GenerateContentConfig") as mock_cfg,
        ):
            client = mock_client_cls.return_value
            client.models.generate_content.return_value = self._mock_gemini_response()

            usage_track.call_text(
                model="gemini-3-flash-preview",
                system="sys",
                user="usr",
                json_mode=False,
                temperature=0.0,
            )

            assert mock_cfg.called
            kwargs = mock_cfg.call_args.kwargs
            assert "response_schema" not in kwargs or kwargs["response_schema"] is None


# ---------------------------------------------------------------------------
# OpenAI-compat path: response_format json_schema
# ---------------------------------------------------------------------------


class TestOpenAISchema:
    def _mock_oai_response(self) -> MagicMock:
        resp = MagicMock()
        msg = MagicMock()
        msg.content = '{"answer":"hi","score":1}'
        choice = MagicMock(message=msg)
        resp.choices = [choice]
        resp.usage = MagicMock(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            completion_tokens_details=None,
            prompt_cache_hit_tokens=0,
            prompt_cache_miss_tokens=0,
        )
        return resp

    def test_schema_passes_json_schema_response_format_for_deepseek(self) -> None:
        from src.query import usage_track

        # Reset the per-process unsupported set so a previous test doesn't
        # bias us to json_object.
        usage_track._json_schema_unsupported.clear()

        with patch.object(usage_track, "_openai_compat_client") as mock_make_client:
            client = MagicMock()
            client.chat.completions.create.return_value = self._mock_oai_response()
            mock_make_client.return_value = client

            usage_track.call_text(
                model="deepseek-chat",
                system="sys",
                user="usr",
                json_mode=False,
                temperature=0.0,
                schema=_DemoSchema,
            )

            kwargs = client.chat.completions.create.call_args.kwargs
            rf = kwargs.get("response_format")
            assert rf is not None, "response_format must be set when schema is provided"
            assert rf["type"] == "json_schema", (
                f"Expected response_format type=json_schema; got {rf!r}"
            )
            assert rf["json_schema"]["name"] == "_DemoSchema"
            assert rf["json_schema"]["strict"] is True
            assert "schema" in rf["json_schema"]

    def test_falls_back_to_json_object_on_400(self) -> None:
        """If the provider rejects json_schema, fall back to json_object once
        and cache the model in `_json_schema_unsupported`. Never retry on
        parse failure — that's a separate (caller) concern."""
        import openai as _openai_module

        from src.query import usage_track

        usage_track._json_schema_unsupported.clear()

        with patch.object(usage_track, "_openai_compat_client") as mock_make_client:
            client = MagicMock()

            err = _openai_module.BadRequestError(
                message="json_schema unsupported",
                response=MagicMock(status_code=400, request=MagicMock()),
                body=None,
            )
            ok_response = self._mock_oai_response()
            client.chat.completions.create.side_effect = [err, ok_response]
            mock_make_client.return_value = client

            usage_track.call_text(
                model="deepseek-chat",
                system="sys",
                user="usr",
                json_mode=False,
                temperature=0.0,
                schema=_DemoSchema,
            )

            calls = client.chat.completions.create.call_args_list
            assert len(calls) == 2, "Must retry exactly once on BadRequestError"
            second_rf = calls[1].kwargs["response_format"]
            assert second_rf == {"type": "json_object"}, (
                f"Fallback must use json_object; got {second_rf!r}"
            )
            assert "deepseek-chat" in usage_track._json_schema_unsupported

    def test_cached_unsupported_skips_retry(self) -> None:
        """If a model is already known-unsupported, send json_object directly
        — no wasted 400 round trip."""
        from src.query import usage_track

        usage_track._json_schema_unsupported.clear()
        usage_track._json_schema_unsupported.add("deepseek-chat")

        try:
            with patch.object(usage_track, "_openai_compat_client") as mock_make_client:
                client = MagicMock()
                client.chat.completions.create.return_value = self._mock_oai_response()
                mock_make_client.return_value = client

                usage_track.call_text(
                    model="deepseek-chat",
                    system="sys",
                    user="usr",
                    json_mode=False,
                    temperature=0.0,
                    schema=_DemoSchema,
                )

                calls = client.chat.completions.create.call_args_list
                assert len(calls) == 1, "Cached unsupported must skip the json_schema attempt"
                assert calls[0].kwargs["response_format"] == {"type": "json_object"}
        finally:
            usage_track._json_schema_unsupported.clear()

    def test_no_schema_no_response_format(self) -> None:
        """No schema, no json_mode → no response_format at all."""
        from src.query import usage_track

        with patch.object(usage_track, "_openai_compat_client") as mock_make_client:
            client = MagicMock()
            client.chat.completions.create.return_value = self._mock_oai_response()
            mock_make_client.return_value = client

            usage_track.call_text(
                model="deepseek-chat",
                system="sys",
                user="usr",
                json_mode=False,
                temperature=0.0,
            )

            kwargs = client.chat.completions.create.call_args.kwargs
            assert "response_format" not in kwargs or kwargs["response_format"] is None
