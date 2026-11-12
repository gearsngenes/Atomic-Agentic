from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any

import pytest
from google.genai import errors as genai_errors
from google.genai import types as _real_genai_types

gemini_module = importlib.import_module("atomic_agentic.llm.gemini_engine")

from atomic_agentic.exceptions import LLMEngineError
from atomic_agentic.llm import GeminiEngine
from atomic_agentic.utils.core import run_coro_sync


class FakeGenAIClient:
    instances: list["FakeGenAIClient"] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.models = SimpleNamespace(generate_content=self._generate_content)
        self.aio = SimpleNamespace(
            models=SimpleNamespace(generate_content=self._generate_content_async)
        )
        self.files = SimpleNamespace(
            upload=self._upload_file,
            delete=self._delete_file,
        )
        self.generate_calls: list[dict[str, Any]] = []
        self.aio_generate_calls: list[dict[str, Any]] = []
        self.deleted_files: list[str] = []
        FakeGenAIClient.instances.append(self)

    def _generate_content(self, **kwargs: Any) -> Any:
        self.generate_calls.append(kwargs)
        return SimpleNamespace(text=" gemini text ", usage_metadata=None)

    async def _generate_content_async(self, **kwargs: Any) -> Any:
        self.aio_generate_calls.append(kwargs)
        return SimpleNamespace(text=" gemini async text ", usage_metadata=None)

    def _upload_file(self, *, file: str) -> Any:
        return SimpleNamespace(
            name="gemini_file",
            uri="gs://fake/gemini_file",
            mime_type="text/plain",
        )

    def _delete_file(self, *, name: str) -> None:
        self.deleted_files.append(name)


class FakeGenAI:
    Client = FakeGenAIClient

    class types:
        GenerateContentConfig = _real_genai_types.GenerateContentConfig
        Content = _real_genai_types.Content
        Part = _real_genai_types.Part
        FileData = _real_genai_types.FileData
        ThinkingConfig = _real_genai_types.ThinkingConfig


def _make_engine(monkeypatch: pytest.MonkeyPatch, **kwargs: Any) -> GeminiEngine:
    FakeGenAIClient.instances.clear()
    monkeypatch.setattr(gemini_module, "genai", FakeGenAI)
    return GeminiEngine(model="gemini-2.5-flash", **kwargs)


class TestGeminiEngine:
    def test_missing_genai_sdk_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(gemini_module, "genai", None)
        with pytest.raises(LLMEngineError, match="google-genai"):
            GeminiEngine(model="gemini_test")

    def test_constructor_builds_client_from_kwargs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        FakeGenAIClient.instances.clear()
        monkeypatch.setattr(gemini_module, "genai", FakeGenAI)

        engine = GeminiEngine(
            model="gemini-2.5-flash",
            timeout_seconds=7.0,
            api_key="secret",
        )
        fake = FakeGenAIClient.instances[-1]

        assert engine.name == "gemini_gemini_2_5_flash"
        assert fake.kwargs["api_key"] == "secret"
        assert fake.kwargs["http_options"] == {"timeout": 7000}

    def test_constructor_client_injection_skips_construction(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        FakeGenAIClient.instances.clear()
        monkeypatch.setattr(gemini_module, "genai", FakeGenAI)

        injected = FakeGenAIClient()
        count_before = len(FakeGenAIClient.instances)
        engine = GeminiEngine(model="gemini-2.5-flash", client=injected)

        assert len(FakeGenAIClient.instances) == count_before
        assert engine._client is injected

    def test_build_content_list_maps_roles_and_skips_system(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)
        messages = [
            {"role": "system", "content": "Sys."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        contents = engine._build_content_list(messages, {})

        assert len(contents) == 2
        assert contents[0].role == "user"
        assert contents[0].parts[0].text == "Hello"
        assert contents[1].role == "model"
        assert contents[1].parts[0].text == "Hi"

    def test_build_content_list_appends_attachments_to_last_user_turn(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)
        file_obj = SimpleNamespace(name="f1", uri="gs://fake/f1", mime_type="text/plain")
        attachments = {
            "file.txt": {"uploaded": True, "file_obj": file_obj, "mime": "text/plain"},
            "note.txt": {"inlined": True, "inlined_text": "Inline!"},
        }
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        contents = engine._build_content_list(messages, attachments)

        assert len(contents) == 2
        first_parts = contents[0].parts
        assert len(first_parts) == 3
        assert first_parts[0].text == "Hello"
        assert first_parts[1].file_data.file_uri == "gs://fake/f1"
        assert first_parts[2].text == "Inline!"
        assert contents[1].role == "model"

    def test_build_content_list_attachments_land_in_last_user_turn_not_first(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)
        file_obj = SimpleNamespace(name="f1", uri="gs://fake/f1", mime_type="text/plain")
        attachments = {"file.txt": {"uploaded": True, "file_obj": file_obj, "mime": "text/plain"}}
        messages = [
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Reply"},
            {"role": "user", "content": "Second"},
        ]

        contents = engine._build_content_list(messages, attachments)

        assert len(contents[0].parts) == 1  # first user turn untouched
        assert len(contents[2].parts) == 2  # last user turn gets the attachment
        assert contents[2].parts[1].file_data.file_uri == "gs://fake/f1"

    def test_build_content_list_standalone_attachment_when_no_user_turn(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)
        file_obj = SimpleNamespace(name="f1", uri="gs://fake/f1", mime_type="text/plain")
        attachments = {"file.txt": {"uploaded": True, "file_obj": file_obj, "mime": "text/plain"}}
        messages = [{"role": "assistant", "content": "Hello"}]

        contents = engine._build_content_list(messages, attachments)

        # No user turn exists — a standalone trailing user Content is appended.
        assert contents[-1].role == "user"
        assert contents[-1].parts[0].file_data.file_uri == "gs://fake/f1"

    def test_call_provider_uses_sync_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch, temperature=0.4)
        messages = [
            {"role": "system", "content": "Sys."},
            {"role": "user", "content": "Hello"},
        ]
        payload = engine._build_provider_payload(messages, {})

        assert payload["system_instruction"] == "Sys."
        assert len(payload["contents"]) == 1
        assert payload["contents"][0].role == "user"

        response = engine._call_provider(payload)
        fake = FakeGenAIClient.instances[-1]
        call = fake.generate_calls[-1]

        assert call["model"] == "gemini-2.5-flash"
        assert call["config"].temperature == 0.4
        assert call["config"].system_instruction == "Sys."
        assert engine._extract_result(response, requested_structured=False) == " gemini text "

    def test_call_provider_async_uses_aio_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)
        payload = engine._build_provider_payload(
            [{"role": "user", "content": "Hi"}], {}
        )
        response = run_coro_sync(engine._call_provider_async(payload))
        fake = FakeGenAIClient.instances[-1]

        assert fake.aio_generate_calls
        assert not fake.generate_calls
        assert engine._extract_result(response, requested_structured=False) == " gemini async text "

    def test_build_generate_config_pairs_mime_type_and_json_schema(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)
        schema = {"type": "object", "properties": {"a": {"type": "string"}}}

        cfg = engine._build_generate_config(None, schema)

        assert cfg.response_mime_type == "application/json"
        assert cfg.response_json_schema == schema

    def test_build_generate_config_omits_both_when_no_output_structure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)

        cfg = engine._build_generate_config(None)

        assert cfg.response_mime_type is None
        assert cfg.response_json_schema is None

    def test_extract_result_not_requested_returns_text_unchanged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)
        response = SimpleNamespace(text=" gemini text ")

        assert engine._extract_result(response, requested_structured=False) == " gemini text "

    def test_extract_result_requested_parses_json(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)
        response = SimpleNamespace(text='{"a": 1}')

        assert engine._extract_result(response, requested_structured=True) == {"a": 1}

    def test_extract_result_requested_json_decode_error_falls_back(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)
        response = SimpleNamespace(text="not json")

        assert engine._extract_result(response, requested_structured=True) == "not json"

    def test_extract_result_requested_none_text_falls_back(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # json.loads(None) raises TypeError, caught by the same except clause
        # as JSONDecodeError -- proves the documented (JSONDecodeError,
        # TypeError) fallback branch, distinct from every other engine's
        # single-exception fallback.
        engine = _make_engine(monkeypatch)
        response = SimpleNamespace(text=None)

        assert engine._extract_result(response, requested_structured=True) is None

    def test_temperature_none_omits_from_config(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch, temperature=None)
        cfg = engine._build_generate_config(None)
        assert cfg.temperature is None

    def test_thinking_config_passed_to_generate_config(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch, thinking_config={"thinking_budget": 1024})
        cfg = engine._build_generate_config(None)
        assert cfg.thinking_config.thinking_budget == 1024

    def test_gemini_to_dict_includes_non_secret_config(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(gemini_module, "genai", FakeGenAI)
        engine = GeminiEngine(
            model="gemini-2.5-flash",
            api_key="secret",
            temperature=0.2,
            max_output_tokens=512,
            thinking_config={"thinking_budget": 100},
        )
        data = engine.to_dict()

        assert data["type"] == "GeminiEngine"
        assert data["model"] == "gemini-2.5-flash"
        assert data["temperature"] == 0.2
        assert data["max_output_tokens"] == 512
        assert data["thinking_config"] == {"thinking_budget": 100}
        assert "secret" not in str(data)

    def test_collect_system_joins_multiple_messages(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)
        messages = [
            {"role": "system", "content": "Part one."},
            {"role": "user", "content": "Hello"},
            {"role": "system", "content": "Part two."},
        ]
        result = engine._collect_system(messages)
        assert result == "Part one.\n\nPart two."

    def test_collect_system_returns_none_when_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)
        messages = [{"role": "user", "content": "Hello"}]
        assert engine._collect_system(messages) is None

    def test_should_retry_server_error_always_retries(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch, max_retries=2)
        exc = genai_errors.ServerError(500, {"message": "srv"}, None)
        assert engine._should_retry(exc, attempt=1) is True

    def test_should_retry_client_error_only_on_429(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch, max_retries=2)
        rate_limited = genai_errors.ClientError(429, {"message": "rl"}, None)
        bad_request = genai_errors.ClientError(400, {"message": "bad"}, None)

        assert engine._should_retry(rate_limited, attempt=1) is True
        assert engine._should_retry(bad_request, attempt=1) is False

    def test_should_retry_respects_max_retries(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch, max_retries=2)
        exc = genai_errors.ServerError(500, {"message": "srv"}, None)
        assert engine._should_retry(exc, attempt=3) is False


class TestGeminiAttachments:
    def test_validate_attachment_path_rejects_illegal_ext(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        engine = _make_engine(monkeypatch)
        path = tmp_path / "payload.zip"
        path.write_bytes(b"data")

        with pytest.raises(LLMEngineError, match="not permitted"):
            engine._validate_attachment_path(str(path))

    def test_validate_attachment_path_passes_for_txt(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        engine = _make_engine(monkeypatch)
        path = tmp_path / "notes.txt"
        path.write_text("hello")
        engine._validate_attachment_path(str(path))  # no exception

    def test_prepare_attachment_uploads_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        engine = _make_engine(monkeypatch)
        path = tmp_path / "doc.txt"
        path.write_text("content")

        meta = engine._prepare_attachment(str(path))

        assert meta["uploaded"] is True
        assert meta["kind"] == "file"
        assert meta["resource_name"] == "gemini_file"

    def test_on_detach_with_resource_name_calls_delete(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)
        fake = FakeGenAIClient.instances[-1]

        engine._on_detach({"resource_name": "files/abc123"})

        assert "files/abc123" in fake.deleted_files

    def test_on_detach_without_resource_name_is_noop(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)
        fake = FakeGenAIClient.instances[-1]

        engine._on_detach({})
        engine._on_detach({"resource_name": None})

        assert fake.deleted_files == []


class TestGeminiTokenUsage:
    def test_extract_token_usage_happy_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)
        usage = SimpleNamespace(
            prompt_token_count=10,
            total_token_count=25,
            candidates_token_count=15,
            thoughts_token_count=None,
            tool_use_prompt_token_count=None,
            cached_content_token_count=None,
        )
        response = SimpleNamespace(usage_metadata=usage)

        result = engine._extract_token_usage(response)

        assert result.input_tokens == 10
        assert result.total_tokens == 25
        assert result.generated_tokens == 15
        assert result.response_tokens == 15

    def test_extract_token_usage_response_tokens_falls_back_when_candidates_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)
        usage = SimpleNamespace(
            prompt_token_count=10,
            total_token_count=25,
            candidates_token_count=None,
            thoughts_token_count=None,
            tool_use_prompt_token_count=None,
            cached_content_token_count=None,
        )
        response = SimpleNamespace(usage_metadata=usage)

        result = engine._extract_token_usage(response)

        assert result.response_tokens == result.generated_tokens == 15

    def test_extract_token_usage_none_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)
        response = SimpleNamespace(usage_metadata=None)

        with pytest.raises(LLMEngineError, match="usage_metadata"):
            engine._extract_token_usage(response)

    def test_extract_token_usage_missing_prompt_token_count_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)
        usage = SimpleNamespace(
            prompt_token_count=None,
            total_token_count=25,
            candidates_token_count=15,
            thoughts_token_count=None,
            tool_use_prompt_token_count=None,
            cached_content_token_count=None,
        )
        with pytest.raises(LLMEngineError, match="prompt_token_count"):
            engine._extract_token_usage(SimpleNamespace(usage_metadata=usage))

    def test_extract_token_usage_negative_generated_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)
        usage = SimpleNamespace(
            prompt_token_count=30,
            total_token_count=10,
            candidates_token_count=None,
            thoughts_token_count=None,
            tool_use_prompt_token_count=None,
            cached_content_token_count=None,
        )
        with pytest.raises(LLMEngineError, match="less than"):
            engine._extract_token_usage(SimpleNamespace(usage_metadata=usage))
