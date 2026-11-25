from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("google.genai")
from google.genai import types as _real_genai_types

gemini_module = importlib.import_module("atomic_agentic.llm.gemini_engine")

from atomic_agentic.exceptions import LLMEngineError
from atomic_agentic.llm import GeminiEngine
from atomic_agentic.models.results import GeminiTokenUsage
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

    def test_constructor_builds_client_from_kwargs_unchanged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        FakeGenAIClient.instances.clear()
        monkeypatch.setattr(gemini_module, "genai", FakeGenAI)

        engine = GeminiEngine(model="gemini-2.5-flash", api_key="secret")
        fake = FakeGenAIClient.instances[-1]

        assert engine.name == "GeminiEngine"
        assert fake.kwargs == {"api_key": "secret"}

    def test_http_options_pass_through_untouched(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        FakeGenAIClient.instances.clear()
        monkeypatch.setattr(gemini_module, "genai", FakeGenAI)
        caller_options = {"api_version": "v1beta", "timeout": 120_000}

        GeminiEngine(model="gemini-2.5-flash", http_options=caller_options)

        assert FakeGenAIClient.instances[-1].kwargs["http_options"] is caller_options

    def test_client_with_client_kwargs_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(gemini_module, "genai", FakeGenAI)

        with pytest.raises(LLMEngineError, match="either 'client' or client kwargs"):
            GeminiEngine(
                model="gemini-2.5-flash", client=FakeGenAIClient(), api_key="secret"
            )

    def test_runtime_parameters_start_empty_so_provider_defaults_apply(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)

        cfg = engine._build_provider_payload([{"role": "user", "content": "Hi"}])["config"]

        assert engine.runtime_parameters == {}
        assert cfg.temperature is None

    @pytest.mark.parametrize(
        "key",
        ["model", "contents", "system_instruction",
         "response_mime_type", "response_json_schema", "stream"],
    )
    def test_reserved_runtime_keys_raise(
        self, monkeypatch: pytest.MonkeyPatch, key: str
    ) -> None:
        engine = _make_engine(monkeypatch)

        with pytest.raises(LLMEngineError, match=key):
            engine.update_runtime_parameters(**{key: "x"})

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
        }
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        contents = engine._build_content_list(messages, attachments)

        assert len(contents) == 2
        first_parts = contents[0].parts
        assert len(first_parts) == 2
        assert first_parts[0].text == "Hello"
        assert first_parts[1].file_data.file_uri == "gs://fake/f1"
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

    def test_blank_user_turn_without_attachments_is_rejected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)

        with pytest.raises(LLMEngineError, match=r"^GeminiEngine\.invoke: .*blank content"):
            engine.invoke({"messages": [{"role": "user", "content": ""}]})

    def test_call_provider_uses_sync_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)
        engine.update_runtime_parameters(temperature=0.4)
        messages = [
            {"role": "system", "content": "Sys."},
            {"role": "user", "content": "Hello"},
        ]
        payload = engine._build_provider_payload(messages)

        assert payload["config"].system_instruction == "Sys."
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
            [{"role": "user", "content": "Hi"}]
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

        cfg = engine._build_provider_payload(
            [{"role": "user", "content": "Hi"}], schema
        )["config"]

        assert cfg.response_mime_type == "application/json"
        assert cfg.response_json_schema == schema

    def test_build_generate_config_omits_both_when_no_output_structure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)

        cfg = engine._build_provider_payload([{"role": "user", "content": "Hi"}])["config"]

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

    @pytest.mark.parametrize("missing", [None, ""])
    def test_extract_result_no_text(
        self, monkeypatch: pytest.MonkeyPatch, missing: Any
    ) -> None:
        engine = _make_engine(monkeypatch)
        response = SimpleNamespace(text=missing)

        assert engine._extract_result(response, requested_structured=False) == ""
        assert engine._extract_result(response, requested_structured=True) == ""

    def test_extract_result_null_text_structured_is_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)

        assert engine._extract_result(SimpleNamespace(text="null"), True) is None

    def test_removed_runtime_key_is_omitted_from_config(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)
        engine.update_runtime_parameters(temperature=0.4)
        engine.update_runtime_parameters(temperature=None)

        cfg = engine._build_provider_payload([{"role": "user", "content": "Hi"}])["config"]

        assert cfg.temperature is None

    def test_thinking_config_dict_passes_through_to_generate_config(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)
        engine.update_runtime_parameters(thinking_config={"thinking_budget": 1024})

        cfg = engine._build_provider_payload([{"role": "user", "content": "Hi"}])["config"]

        assert cfg.thinking_config.thinking_budget == 1024

    def test_payload_is_the_sdk_call_kwargs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)

        payload = engine._build_provider_payload([{"role": "user", "content": "Hi"}])

        assert set(payload) == {"model", "contents", "config"}
        assert payload["model"] == "gemini-2.5-flash"

    def test_engine_owned_config_fields_win_over_runtime_parameters(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)
        engine._runtime_parameters.update(
            system_instruction="smuggled", response_mime_type="text/plain"
        )
        schema = {"type": "object"}

        cfg = engine._build_provider_payload(
            [{"role": "system", "content": "Real."}, {"role": "user", "content": "Hi"}],
            schema,
        )["config"]

        assert cfg.system_instruction == "Real."
        assert cfg.response_mime_type == "application/json"

    def test_payload_snapshots_runtime_parameters_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)
        engine.update_runtime_parameters(temperature=0.1)
        payload = engine._build_provider_payload([{"role": "user", "content": "Hi"}])

        engine.update_runtime_parameters(temperature=0.9)
        engine._call_provider(payload)

        assert FakeGenAIClient.instances[-1].generate_calls[-1]["config"].temperature == 0.1

    def test_unknown_runtime_key_fails_at_update(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)
        engine.update_runtime_parameters(temperature=0.2)

        with pytest.raises(LLMEngineError, match="unknown keys") as excinfo:
            engine.update_runtime_parameters(not_a_real_field=1, temperature=0.9, also_bad=2)

        message = str(excinfo.value)
        assert "['also_bad', 'not_a_real_field']" in message
        assert "temperature" in message  # listed as a valid key
        assert "system_instruction" not in message.split("valid keys:")[1]
        assert engine.runtime_parameters == {"temperature": 0.2}

    def test_known_runtime_keys_are_accepted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)

        engine.update_runtime_parameters(
            temperature=0.2, max_output_tokens=64, thinking_config={"thinking_budget": 0}
        )

        assert set(engine.runtime_parameters) == {
            "temperature", "max_output_tokens", "thinking_config"
        }

    def test_unknown_key_smuggled_past_the_check_fails_in_the_sdk_config(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)
        engine._runtime_parameters["not_a_real_field"] = 1

        with pytest.raises(Exception, match="not_a_real_field"):
            engine._build_provider_payload([{"role": "user", "content": "Hi"}])

    def test_gemini_to_dict_includes_non_secret_config(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(gemini_module, "genai", FakeGenAI)
        engine = GeminiEngine(model="gemini-2.5-flash", api_key="secret")
        engine.update_runtime_parameters(
            temperature=0.2, max_output_tokens=512,
            thinking_config={"thinking_budget": 100},
        )
        data = engine.to_dict()

        assert data["type"] == "GeminiEngine"
        assert data["model"] == "gemini-2.5-flash"
        assert data["runtime_parameters"] == {
            "temperature": 0.2,
            "max_output_tokens": 512,
            "thinking_config": {"thinking_budget": 100},
        }
        for removed in ("max_retries", "retry_backoff_base", "attachments",
                        "timeout_seconds", "temperature"):
            assert removed not in data
        assert "secret" not in str(data)

    def test_system_messages_are_joined_into_the_system_instruction(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)
        messages = [
            {"role": "system", "content": "Part one."},
            {"role": "user", "content": "Hello"},
            {"role": "system", "content": "Part two."},
        ]

        cfg = engine._build_provider_payload(messages)["config"]

        assert cfg.system_instruction == "Part one.\n\nPart two."

    def test_no_system_message_leaves_the_system_instruction_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)

        cfg = engine._build_provider_payload([{"role": "user", "content": "Hello"}])["config"]

        assert cfg.system_instruction is None


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

    def test_attach_rejects_blank_and_non_string_paths(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)

        with pytest.raises(LLMEngineError, match="non-empty string"):
            engine.attach("")
        with pytest.raises(LLMEngineError, match="non-empty string"):
            engine.attach(123)  # type: ignore[arg-type]

    def test_attach_stores_metadata_and_caches_the_same_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        engine = _make_engine(monkeypatch)
        path = tmp_path / "doc.txt"
        path.write_text("content")

        first = engine.attach(str(path))
        second = engine.attach(str(path))

        assert first == second
        assert first is not second
        assert engine.attachments == {str(path): first}

    def test_attachments_property_returns_a_copy(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        engine = _make_engine(monkeypatch)
        path = tmp_path / "doc.txt"
        path.write_text("content")
        engine.attach(str(path))

        engine.attachments.clear()  # type: ignore[attr-defined]
        engine.attachments[str(path)]["kind"] = "changed"  # type: ignore[index]
        engine.attach(str(path))["kind"] = "changed"  # type: ignore[index]

        assert engine.attachments[str(path)]["kind"] == "file"

    def test_payload_reads_the_engine_attachments(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        engine = _make_engine(monkeypatch)
        path = tmp_path / "doc.txt"
        path.write_text("content")
        engine.attach(str(path))

        payload = engine._build_provider_payload([{"role": "user", "content": "Q?"}])

        assert len(payload["contents"][-1].parts) == 2

    def test_detach_with_resource_name_deletes_the_file(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)
        fake = FakeGenAIClient.instances[-1]
        engine._attachments["doc.txt"] = {"resource_name": "files/abc123"}

        assert engine.detach("doc.txt") is True

        assert "files/abc123" in fake.deleted_files
        assert engine.attachments == {}

    def test_detach_without_resource_name_deletes_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)
        fake = FakeGenAIClient.instances[-1]
        engine._attachments["a.txt"] = {}
        engine._attachments["b.txt"] = {"resource_name": None}

        assert engine.detach("a.txt") is True
        assert engine.detach("b.txt") is True

        assert fake.deleted_files == []

    def test_detach_returns_false_for_unknown_path_and_swallows_delete_errors(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)

        def _boom(*, name: str) -> None:
            raise RuntimeError("delete failed")

        engine._client.files.delete = _boom  # type: ignore[attr-defined]
        engine._attachments["doc.txt"] = {"resource_name": "files/abc123"}

        assert engine.detach("never_attached.txt") is False
        assert engine.detach("doc.txt") is True

    def test_clear_attachments_detaches_everything(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)
        fake = FakeGenAIClient.instances[-1]
        engine._attachments["a.txt"] = {"resource_name": "files/a"}
        engine._attachments["b.txt"] = {}

        engine.clear_attachments()

        assert engine.attachments == {}
        assert fake.deleted_files == ["files/a"]


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

    def test_extract_token_usage_none_returns_zeros_when_not_required(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)
        engine.require_usage = False

        result = engine._extract_token_usage(SimpleNamespace(usage_metadata=None))

        assert isinstance(result, GeminiTokenUsage)
        assert (result.input_tokens, result.generated_tokens,
                result.total_tokens, result.response_tokens) == (0, 0, 0, 0)

    def test_require_usage_rejects_a_non_bool(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)

        with pytest.raises(LLMEngineError, match="require_usage"):
            engine.require_usage = 0  # type: ignore[assignment]

    def test_partial_usage_still_raises_when_not_required(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)
        engine.require_usage = False
        usage = SimpleNamespace(
            prompt_token_count=None, total_token_count=5, candidates_token_count=None,
            thoughts_token_count=None, tool_use_prompt_token_count=None,
            cached_content_token_count=None,
        )

        with pytest.raises(LLMEngineError, match="prompt_token_count"):
            engine._extract_token_usage(SimpleNamespace(usage_metadata=usage))

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
