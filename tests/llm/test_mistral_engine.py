from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any

import pytest

mistral_module = importlib.import_module("atomic_agentic.llm.mistral_engine")

from atomic_agentic.exceptions import LLMEngineError
from atomic_agentic.llm import MistralEngine
from atomic_agentic.models.results import MistralTokenUsage
from atomic_agentic.utils.core import run_coro_sync


class FakeMistralClient:
    instances: list["FakeMistralClient"] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.chat = SimpleNamespace(
            complete=self._complete,
            complete_async=self._complete_async,
        )
        self.files = SimpleNamespace(
            upload=self._upload,
            get_signed_url=self._get_signed_url,
            delete=self._delete,
        )
        self.complete_calls: list[dict[str, Any]] = []
        self.aio_complete_calls: list[dict[str, Any]] = []
        self.deleted_files: list[str] = []
        FakeMistralClient.instances.append(self)

    def _complete(self, **kwargs: Any) -> Any:
        self.complete_calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=" mistral text "))]
        )

    async def _complete_async(self, **kwargs: Any) -> Any:
        self.aio_complete_calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=" mistral async text "))]
        )

    def _upload(self, **kwargs: Any) -> Any:
        return SimpleNamespace(id="mistral_file")

    def _get_signed_url(self, *, file_id: str, **kwargs: Any) -> Any:
        return SimpleNamespace(url=f"https://signed.example/{file_id}")

    def _delete(self, *, file_id: str, **kwargs: Any) -> None:
        self.deleted_files.append(file_id)


def _make_engine(monkeypatch: pytest.MonkeyPatch, **kwargs: Any) -> MistralEngine:
    FakeMistralClient.instances.clear()
    monkeypatch.setattr(mistral_module, "Mistral", FakeMistralClient)
    return MistralEngine(model="mistral-small-latest", **kwargs)


class TestMistralValidatingProperties:
    def test_strict_setter_rejects_non_bool(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)
        with pytest.raises(LLMEngineError, match="strict"):
            engine.strict = "not-a-bool"  # type: ignore[assignment]

    def test_strict_defaults_true_and_stays_out_of_the_runtime_dict(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)
        assert engine.strict is True

        engine.strict = False

        assert "strict" not in engine.runtime_parameters


class TestMistralEngine:
    def test_missing_mistral_sdk_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(mistral_module, "Mistral", None)
        with pytest.raises(LLMEngineError, match="mistralai"):
            MistralEngine(model="mistral_test")

    def test_constructor_builds_client_from_kwargs_unchanged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch, api_key="secret", timeout_ms=5000)
        fake = FakeMistralClient.instances[-1]

        assert engine.name == "MistralEngine"
        assert fake.kwargs == {"api_key": "secret", "timeout_ms": 5000}

    def test_no_timeout_is_injected_into_the_client(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _make_engine(monkeypatch)

        assert "timeout_ms" not in FakeMistralClient.instances[-1].kwargs

    def test_client_with_client_kwargs_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(mistral_module, "Mistral", FakeMistralClient)

        with pytest.raises(LLMEngineError, match="either 'client' or client kwargs"):
            MistralEngine(
                model="mistral-small-latest", client=FakeMistralClient(), api_key="secret"
            )

    @pytest.mark.parametrize("key", ["model", "messages", "response_format", "stream"])
    def test_reserved_runtime_keys_raise(
        self, monkeypatch: pytest.MonkeyPatch, key: str
    ) -> None:
        engine = _make_engine(monkeypatch)

        with pytest.raises(LLMEngineError, match=key):
            engine.update_runtime_parameters(**{key: "x"})

    def test_constructor_client_injection_skips_construction(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(mistral_module, "Mistral", FakeMistralClient)
        injected = FakeMistralClient()
        FakeMistralClient.instances.clear()

        engine = MistralEngine(model="mistral-small-latest", client=injected)

        assert FakeMistralClient.instances == []
        assert engine._client is injected

    def test_call_provider_async_uses_aio_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)
        engine.update_runtime_parameters(temperature=0.3)
        payload = engine._build_provider_payload([{"role": "user", "content": "hello"}])

        result = run_coro_sync(engine._call_provider_async(payload))

        fake = FakeMistralClient.instances[-1]
        assert fake.aio_complete_calls
        call = fake.aio_complete_calls[-1]
        assert call["model"] == "mistral-small-latest"
        assert call["temperature"] == 0.3
        assert engine._extract_result(result, requested_structured=False) == " mistral async text "

    def test_removed_runtime_key_is_omitted_from_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)
        engine.update_runtime_parameters(temperature=0.3)
        engine.update_runtime_parameters(temperature=None)
        engine._call_provider(
            engine._build_provider_payload([{"role": "user", "content": "hi"}])
        )

        fake = FakeMistralClient.instances[-1]
        assert "temperature" not in fake.complete_calls[-1]

    def test_no_temperature_is_sent_by_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)
        engine._call_provider(
            engine._build_provider_payload([{"role": "user", "content": "hi"}])
        )

        assert "temperature" not in FakeMistralClient.instances[-1].complete_calls[-1]

    def test_payload_snapshots_runtime_parameters_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)
        engine.update_runtime_parameters(temperature=0.1)
        payload = engine._build_provider_payload([{"role": "user", "content": "hi"}])

        engine.update_runtime_parameters(temperature=0.9)
        engine._call_provider(payload)

        assert FakeMistralClient.instances[-1].complete_calls[-1]["temperature"] == 0.1

    def test_mistral_ensure_user_parts_converts_the_last_user_message(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)
        messages = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "hello"},
        ]

        index = engine._ensure_user_parts(messages)

        assert index == 2
        assert messages[2]["content"] == [{"type": "text", "text": "hello"}]
        assert messages[0]["content"] == "first"

    def test_mistral_ensure_user_parts_empty_content_becomes_empty_parts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)
        messages = [{"role": "user", "content": ""}]

        assert engine._ensure_user_parts(messages) == 0
        assert messages[0]["content"] == []

    def test_prepare_attachment_failure_names_the_cause(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        engine = _make_engine(monkeypatch)

        with pytest.raises(LLMEngineError, match=r"_prepare_attachment failed for .*: .*gone\.txt"):
            engine._prepare_attachment(str(tmp_path / "gone.txt"))

    def test_mistral_payload_and_extract_text(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine = _make_engine(monkeypatch)
        engine.update_runtime_parameters(temperature=0.6)

        engine._attachments.update({
            "notes.txt": {"kind": "text", "inlined_text": "Inline notes"},
            "doc.pdf": {"kind": "pdf", "signed_url": "https://doc"},
            "image.png": {"kind": "image", "signed_url": "https://image"},
        })

        payload = engine._build_provider_payload([{"role": "user", "content": "Question"}])

        user_parts = payload["messages"][0]["content"]

        assert user_parts[0] == {"type": "text", "text": "Question"}
        assert any(part["type"] == "text" and "Inline notes" in part["text"] for part in user_parts)
        assert {"type": "document_url", "document_url": "https://doc"} in user_parts
        assert {"type": "image_url", "image_url": "https://image"} in user_parts

        response = engine._call_provider(payload)
        fake = FakeMistralClient.instances[-1]

        # ``_extract_result`` does not strip; the base ``extract`` does.
        assert engine._extract_result(response, requested_structured=False) == " mistral text "
        assert fake.complete_calls[-1]["model"] == "mistral-small-latest"
        assert fake.complete_calls[-1]["temperature"] == 0.6

    def test_mistral_extract_text_from_dict_chunk_list(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)
        response = self._response_with_content(
            [{"text": "hello "}, {"type": "reference"}, {"text": "world"}]
        )

        assert engine._extract_result(response, requested_structured=False) == "hello world"

    def test_mistral_extract_text_from_sdk_chunk_objects(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        models = pytest.importorskip("mistralai.client.models")
        TextChunk, ImageURLChunk = models.TextChunk, models.ImageURLChunk
        engine = _make_engine(monkeypatch)
        response = self._response_with_content([
            TextChunk(text="hello "),
            ImageURLChunk(image_url="https://example.com/x.png"),
            TextChunk(text="world"),
        ])

        assert engine._extract_result(response, requested_structured=False) == "hello world"
        assert engine._extract_result(
            self._response_with_content([TextChunk(text='{"a": 1}')]),
            requested_structured=True,
        ) == {"a": 1}

    @pytest.mark.parametrize("content", [None, "", []])
    def test_extract_result_no_text(
        self, monkeypatch: pytest.MonkeyPatch, content: Any
    ) -> None:
        engine = _make_engine(monkeypatch)
        response = self._response_with_content(content)

        assert engine._extract_result(response, requested_structured=False) == ""
        assert engine._extract_result(response, requested_structured=True) == ""

    def test_extract_result_null_text_structured_is_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)

        assert engine._extract_result(self._response_with_content("null"), True) is None

    @staticmethod
    def _response_with_content(content: Any) -> Any:
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )

    def test_build_payload_response_format_uses_live_strict_property_true(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)
        schema = {"type": "object", "properties": {"a": {"type": "string"}}}

        payload = engine._build_provider_payload(
            [{"role": "user", "content": "hi"}], schema
        )

        assert payload["response_format"] == {
            "type": "json_schema",
            "json_schema": {"name": "output_structure", "schema": schema, "strict": True},
        }

    def test_build_payload_response_format_reflects_strict_false(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)
        engine.strict = False
        schema = {"type": "object", "properties": {"a": {"type": "string"}}}

        payload = engine._build_provider_payload(
            [{"role": "user", "content": "hi"}], schema
        )

        # Proves `strict` is read fresh at call time, not baked at construction.
        assert payload["response_format"]["json_schema"]["strict"] is False

    def test_build_payload_omits_response_format_when_not_structured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)

        payload = engine._build_provider_payload([{"role": "user", "content": "hi"}])

        assert "response_format" not in payload

    def test_call_provider_forwards_response_format_when_present(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)
        schema = {"type": "object"}
        payload = engine._build_provider_payload(
            [{"role": "user", "content": "hi"}], schema
        )

        engine._call_provider(payload)

        fake = FakeMistralClient.instances[-1]
        assert fake.complete_calls[-1]["response_format"] == {
            "type": "json_schema",
            "json_schema": {"name": "output_structure", "schema": schema, "strict": True},
        }

    def test_payload_is_the_sdk_call_kwargs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)
        engine.update_runtime_parameters(temperature=0.3)

        payload = engine._build_provider_payload([{"role": "user", "content": "hi"}])

        assert payload == {
            "temperature": 0.3,
            "model": "mistral-small-latest",
            "messages": [{"role": "user", "content": "hi"}],
        }

    def test_payload_without_attachments_adds_no_user_turn(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)

        payload = engine._build_provider_payload([{"role": "system", "content": "be brief"}])

        assert payload["messages"] == [{"role": "system", "content": "be brief"}]

    def test_blank_user_turn_with_attachment_carries_only_the_attachment_part(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)
        engine._attachments["doc.pdf"] = {"kind": "pdf", "signed_url": "https://doc"}
        messages = engine._normalize_messages([{"role": "user", "content": ""}])

        payload = engine._build_provider_payload(messages)

        assert payload["messages"] == [
            {"role": "user", "content": [{"type": "document_url", "document_url": "https://doc"}]}
        ]

    def test_blank_user_turn_without_attachments_is_rejected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)

        with pytest.raises(LLMEngineError, match=r"^MistralEngine\.invoke: .*blank content"):
            engine.invoke({"messages": [{"role": "user", "content": ""}]})

    def test_engine_owned_fields_win_over_runtime_parameters(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)
        engine._runtime_parameters.update(model="smuggled", messages="smuggled")

        payload = engine._build_provider_payload([{"role": "user", "content": "hi"}])

        assert payload["model"] == "mistral-small-latest"
        assert isinstance(payload["messages"], list)

    def test_extract_result_requested_parses_json(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"a": 1}'))]
        )

        assert engine._extract_result(response, requested_structured=True) == {"a": 1}

    def test_extract_result_requested_parse_failure_falls_back(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="not json"))]
        )

        assert engine._extract_result(response, requested_structured=True) == "not json"

    def test_mistral_to_dict_includes_non_secret_config(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        engine = _make_engine(monkeypatch)
        engine.update_runtime_parameters(temperature=0.7)
        engine.strict = False
        data = engine.to_dict()

        assert data["type"] == "MistralEngine"
        assert data["model"] == "mistral-small-latest"
        assert data["runtime_parameters"] == {"temperature": 0.7}
        assert data["strict"] is False
        for removed in ("inline_cutoff_chars", "max_retries", "attachments",
                        "timeout_seconds", "temperature"):
            assert removed not in data

    def test_get_model_data(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine = _make_engine(monkeypatch)
        model_data = engine._get_model_data()

        assert model_data.provider == "mistral"
        assert model_data.model_name == "mistral-small-latest"


class TestMistralAttachments:
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

    def test_prepare_attachment_pdf_uploads_and_signs(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        engine = _make_engine(monkeypatch)
        path = tmp_path / "report.pdf"
        path.write_bytes(b"%PDF data")

        meta = engine._prepare_attachment(str(path))

        assert meta["kind"] == "pdf"
        assert meta["uploaded"] is True
        assert meta["file_id"] == "mistral_file"
        assert "signed.example/mistral_file" in meta["signed_url"]

    def test_prepare_attachment_sign_failure_deletes_the_upload(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        engine = _make_engine(monkeypatch)
        fake = FakeMistralClient.instances[-1]

        def boom(**kwargs: Any) -> Any:
            raise RuntimeError("sign failed")

        fake.files.get_signed_url = boom
        path = tmp_path / "report.pdf"
        path.write_bytes(b"%PDF data")

        with pytest.raises(LLMEngineError) as exc:
            engine._prepare_attachment(str(path))

        assert "sign failed" in str(exc.value) or "sign failed" in str(exc.value.__cause__)
        assert fake.deleted_files == ["mistral_file"]

    def test_prepare_attachment_sign_failure_ignores_delete_errors(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        engine = _make_engine(monkeypatch)
        fake = FakeMistralClient.instances[-1]

        def sign_boom(**kwargs: Any) -> Any:
            raise RuntimeError("sign failed")

        def delete_boom(**kwargs: Any) -> Any:
            raise RuntimeError("delete failed")

        fake.files.get_signed_url = sign_boom
        fake.files.delete = delete_boom
        path = tmp_path / "report.pdf"
        path.write_bytes(b"%PDF data")

        with pytest.raises(LLMEngineError) as exc:
            engine._prepare_attachment(str(path))

        assert "delete failed" not in str(exc.value)
        assert "sign failed" in str(exc.value) or "sign failed" in str(exc.value.__cause__)

    def test_prepare_attachment_image_uploads_and_signs(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        engine = _make_engine(monkeypatch)
        path = tmp_path / "photo.png"
        path.write_bytes(b"\x89PNG data")

        meta = engine._prepare_attachment(str(path))

        assert meta["kind"] == "image"
        assert meta["uploaded"] is True
        assert meta["file_id"] == "mistral_file"

    def test_prepare_attachment_text_inlines(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        engine = _make_engine(monkeypatch)
        path = tmp_path / "notes.txt"
        path.write_text("Hello from text!")

        meta = engine._prepare_attachment(str(path))

        assert meta["kind"] == "text"
        assert meta["inlined"] is True
        assert "Hello from text!" in meta["inlined_text"]

    def test_prepare_attachment_text_inlines_whole(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        engine = _make_engine(monkeypatch)
        path = tmp_path / "long.txt"
        path.write_text("X" * 300_000)

        meta = engine._prepare_attachment(str(path))

        assert meta["inlined_text"] == "X" * 300_000

    def test_payload_appends_every_text_attachment_whole(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)
        engine._attachments["a.txt"] = {"kind": "text", "inlined_text": "A" * 250_000}
        engine._attachments["b.txt"] = {"kind": "text", "inlined_text": "B" * 250_000}

        payload = engine._build_provider_payload([{"role": "user", "content": "Q?"}])

        texts = [p["text"] for p in payload["messages"][0]["content"]]
        assert any(t.endswith("A" * 250_000) for t in texts)
        assert any(t.endswith("B" * 250_000) for t in texts)
        assert not any("cutoff" in t or "truncated" in t for t in texts)

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
        path = tmp_path / "notes.txt"
        path.write_text("hello")

        first = engine.attach(str(path))
        second = engine.attach(str(path))

        assert first == second
        assert first is not second
        assert engine.attachments == {str(path): first}

    def test_attachments_property_returns_a_copy(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        engine = _make_engine(monkeypatch)
        path = tmp_path / "notes.txt"
        path.write_text("hello")
        engine.attach(str(path))

        engine.attachments.clear()  # type: ignore[attr-defined]
        engine.attachments[str(path)]["kind"] = "changed"  # type: ignore[index]
        engine.attach(str(path))["kind"] = "changed"  # type: ignore[index]

        assert engine.attachments[str(path)]["kind"] == "text"

    def test_detach_with_file_id_deletes_the_file(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)
        fake = FakeMistralClient.instances[-1]
        engine._attachments["doc.pdf"] = {"file_id": "abc123"}

        assert engine.detach("doc.pdf") is True

        assert "abc123" in fake.deleted_files
        assert engine.attachments == {}

    def test_detach_without_file_id_deletes_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)
        fake = FakeMistralClient.instances[-1]
        engine._attachments["notes.txt"] = {"kind": "text"}

        assert engine.detach("notes.txt") is True

        assert fake.deleted_files == []

    def test_detach_returns_false_for_unknown_path_and_swallows_delete_errors(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)

        def _boom(*, file_id: str, **kwargs: Any) -> None:
            raise RuntimeError("delete failed")

        engine._client.files.delete = _boom  # type: ignore[attr-defined]
        engine._attachments["doc.pdf"] = {"file_id": "abc123"}

        assert engine.detach("never_attached.txt") is False
        assert engine.detach("doc.pdf") is True

    def test_clear_attachments_detaches_everything(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)
        fake = FakeMistralClient.instances[-1]
        engine._attachments["a.pdf"] = {"file_id": "file_a"}
        engine._attachments["b.txt"] = {"kind": "text"}

        engine.clear_attachments()

        assert engine.attachments == {}
        assert fake.deleted_files == ["file_a"]


_NO_DETAILS = object()  # sentinel: "attribute genuinely absent", not "set to None"


class TestMistralTokenUsage:
    def _make_usage(
        self,
        *,
        prompt_tokens: int | None = 10,
        completion_tokens: int | None = 5,
        total_tokens: int | None = 15,
        prompt_tokens_details: Any = _NO_DETAILS,
    ) -> Any:
        # B2 regression fixture: the real mistralai.UsageInfo model never
        # declares prompt_tokens_details as an attribute at all (verified
        # empirically against the installed SDK) — by default this fixture
        # must NOT set the attribute either, unlike the old fixture which
        # always set it (defaulting to None) and masked the real
        # AttributeError crash.
        kwargs: dict[str, Any] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }
        if prompt_tokens_details is not _NO_DETAILS:
            kwargs["prompt_tokens_details"] = prompt_tokens_details
        return SimpleNamespace(**kwargs)

    def test_happy_path_with_no_prompt_tokens_details_attribute(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # B2 regression: usage genuinely lacks prompt_tokens_details as an
        # attribute (matching the real mistralai.UsageInfo shape) — must not
        # crash with AttributeError.
        engine = _make_engine(monkeypatch)
        response = SimpleNamespace(usage=self._make_usage())
        assert not hasattr(response.usage, "prompt_tokens_details")

        result = engine._extract_token_usage(response)

        assert result.input_tokens == 10
        assert result.generated_tokens == 5
        assert result.total_tokens == 15
        assert result.response_tokens == 5
        assert result.cached_tokens is None

    def test_usage_none_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine = _make_engine(monkeypatch)
        with pytest.raises(LLMEngineError, match="usage"):
            engine._extract_token_usage(SimpleNamespace(usage=None))

    def test_usage_none_returns_zeros_when_not_required(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)
        engine.require_usage = False

        result = engine._extract_token_usage(SimpleNamespace(usage=None))

        assert isinstance(result, MistralTokenUsage)
        assert (result.input_tokens, result.generated_tokens,
                result.total_tokens, result.response_tokens) == (0, 0, 0, 0)

    def test_require_usage_rejects_a_non_bool(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine = _make_engine(monkeypatch)

        with pytest.raises(LLMEngineError, match="require_usage"):
            engine.require_usage = "no"  # type: ignore[assignment]

    def test_partial_usage_still_raises_when_not_required(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)
        engine.require_usage = False
        usage = SimpleNamespace(prompt_tokens=None, completion_tokens=1, total_tokens=1)

        with pytest.raises(LLMEngineError, match="prompt_tokens"):
            engine._extract_token_usage(SimpleNamespace(usage=usage))

    def test_missing_prompt_tokens_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine = _make_engine(monkeypatch)
        with pytest.raises(LLMEngineError, match="prompt_tokens"):
            engine._extract_token_usage(
                SimpleNamespace(usage=self._make_usage(prompt_tokens=None))
            )

    def test_missing_completion_tokens_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine = _make_engine(monkeypatch)
        with pytest.raises(LLMEngineError, match="completion_tokens"):
            engine._extract_token_usage(
                SimpleNamespace(usage=self._make_usage(completion_tokens=None))
            )

    def test_missing_total_tokens_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine = _make_engine(monkeypatch)
        with pytest.raises(LLMEngineError, match="total_tokens"):
            engine._extract_token_usage(
                SimpleNamespace(usage=self._make_usage(total_tokens=None))
            )

    def test_cached_tokens_from_mapping_details(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine = _make_engine(monkeypatch)
        details = {"cached_tokens": 3}
        response = SimpleNamespace(usage=self._make_usage(prompt_tokens_details=details))

        result = engine._extract_token_usage(response)

        assert result.cached_tokens == 3

    def test_cached_tokens_from_namespace_details(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine = _make_engine(monkeypatch)
        details = SimpleNamespace(cached_tokens=7)
        response = SimpleNamespace(usage=self._make_usage(prompt_tokens_details=details))

        result = engine._extract_token_usage(response)

        assert result.cached_tokens == 7
