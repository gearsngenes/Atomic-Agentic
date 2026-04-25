from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Mapping

import pytest

from atomic_agentic.a2a.PyA2AtomicHost import PyA2AtomicHost, PYA2A_RESULT_KEY
from atomic_agentic.core.Invokable import AtomicInvokable
from atomic_agentic.core.Parameters import ParamSpec


class EchoInvokable(AtomicInvokable):
    def invoke(self, inputs: Mapping[str, Any]) -> dict[str, Any]:
        return self.filter_inputs(inputs)


def make_param(name: str, index: int) -> ParamSpec:
    return ParamSpec(
        name=name,
        index=index,
        kind=ParamSpec.POSITIONAL_OR_KEYWORD,
        type="Any",
    )


def make_invokable(name: str = "echo") -> EchoInvokable:
    return EchoInvokable(
        name=name,
        description="Echo invokable.",
        parameters=[make_param("value", 0)],
        return_type="dict[str, Any]",
    )


def make_message(content: Any) -> SimpleNamespace:
    return SimpleNamespace(
        content=content,
        message_id="message_1",
        conversation_id="conversation_1",
    )


def make_text_content(text: str = "hello") -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def make_function_call_content(
    name: str,
    parameters: list[SimpleNamespace] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        type="function_call",
        name=name,
        parameters=parameters or [],
    )


def make_param_content(name: str, value: Any) -> SimpleNamespace:
    return SimpleNamespace(name=name, value=value)


class TestPyA2AtomicHostConstruction:
    def test_valid_construction(self) -> None:
        host = PyA2AtomicHost(
            [make_invokable("echo")],
            name="test_host",
            description="Test host.",
        )

        assert host.host == "localhost"
        assert host.port == 5000
        assert host.url == "http://localhost:5000"
        assert host.version == "1.0.0"
        assert host.invokable_names == ["echo"]

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"name": "", "description": "Test host."},
            {"name": "test_host", "description": ""},
            {"name": "test_host", "description": "Test host.", "version": ""},
            {"name": "test_host", "description": "Test host.", "host": ""},
            {"name": "test_host", "description": "Test host.", "port": 0},
        ],
    )
    def test_invalid_construction_values_raise(self, kwargs: dict[str, Any]) -> None:
        with pytest.raises(ValueError):
            PyA2AtomicHost([make_invokable("echo")], **kwargs)

    def test_invokables_must_be_list(self) -> None:
        with pytest.raises(TypeError, match="invokables"):
            PyA2AtomicHost(
                (make_invokable("echo"),),  # type: ignore[arg-type]
                name="test_host",
                description="Test host.",
            )

    def test_invokables_must_be_atomic_invokables(self) -> None:
        with pytest.raises(TypeError, match="AtomicInvokable"):
            PyA2AtomicHost(
                ["not_invokable"],  # type: ignore[list-item]
                name="test_host",
                description="Test host.",
            )

    def test_duplicate_invokable_names_raise(self) -> None:
        with pytest.raises(ValueError, match="Duplicate"):
            PyA2AtomicHost(
                [make_invokable("echo"), make_invokable("echo")],
                name="test_host",
                description="Test host.",
            )


class TestPyA2AtomicHostRegistry:
    def test_register_remove_and_clear(self) -> None:
        host = PyA2AtomicHost(
            [],
            name="test_host",
            description="Test host.",
        )

        assert host.register(make_invokable("echo")) == "echo"
        assert host.invokable_names == ["echo"]

        assert host.remove("echo") is True
        assert host.remove("echo") is False

        host.register(make_invokable("adder"))
        host.clear_invokables()
        assert host.invokable_names == []

    def test_register_rejects_duplicate_name(self) -> None:
        host = PyA2AtomicHost(
            [make_invokable("echo")],
            name="test_host",
            description="Test host.",
        )

        with pytest.raises(ValueError, match="Duplicate"):
            host.register(make_invokable("echo"))

    def test_register_rejects_non_invokable(self) -> None:
        host = PyA2AtomicHost(
            [],
            name="test_host",
            description="Test host.",
        )

        with pytest.raises(TypeError, match="AtomicInvokable"):
            host.register("not_invokable")  # type: ignore[arg-type]

    def test_remove_rejects_blank_name(self) -> None:
        host = PyA2AtomicHost(
            [],
            name="test_host",
            description="Test host.",
        )

        with pytest.raises(ValueError, match="name"):
            host.remove("   ")


class TestPyA2AtomicHostPayloads:
    def test_list_invokables_payload(self) -> None:
        host = PyA2AtomicHost(
            [make_invokable("echo")],
            name="test_host",
            description="Test host.",
        )

        payload = host._list_invokables_payload()

        assert list(payload) == ["echo"]
        assert payload["echo"]["name"] == "echo"
        assert payload["echo"]["description"] == "Echo invokable."

    def test_get_invokable_metadata_payload(self) -> None:
        host = PyA2AtomicHost(
            [make_invokable("echo")],
            name="test_host",
            description="Test host.",
        )

        metadata = host._get_invokable_metadata_payload("echo")

        assert metadata["name"] == "echo"
        assert metadata["return_type"] == "dict[str, Any]"
        assert metadata["invokable_type"] == "EchoInvokable"

    def test_get_unknown_invokable_metadata_raises(self) -> None:
        host = PyA2AtomicHost(
            [],
            name="test_host",
            description="Test host.",
        )

        with pytest.raises(KeyError, match="Unknown invokable"):
            host._get_invokable_metadata_payload("missing")

    def test_invoke_registered_invokable(self) -> None:
        host = PyA2AtomicHost(
            [make_invokable("echo")],
            name="test_host",
            description="Test host.",
        )

        assert host._invoke_registered_invokable("echo", {"value": 123}) == {"value": 123}

    def test_invoke_unknown_registered_invokable_raises(self) -> None:
        host = PyA2AtomicHost(
            [],
            name="test_host",
            description="Test host.",
        )

        with pytest.raises(KeyError, match="Unknown invokable"):
            host._invoke_registered_invokable("missing", {})


class TestPyA2AtomicHostMessageHandling:
    def test_text_message_returns_guidance_text(self) -> None:
        host = PyA2AtomicHost(
            [],
            name="test_host",
            description="Test host.",
        )
        response = host.handle_message(make_message(make_text_content()))

        assert response.content.type == "text"
        assert "function calls" in response.content.text

    def test_unsupported_content_type_returns_text_response(self) -> None:
        host = PyA2AtomicHost(
            [],
            name="test_host",
            description="Test host.",
        )
        message = make_message(SimpleNamespace(type="unknown"))

        response = host.handle_message(message)

        assert response.content.type == "text"
        assert "Unsupported content type" in response.content.text

    def test_handle_list_invokables_function_call(self) -> None:
        host = PyA2AtomicHost(
            [make_invokable("echo")],
            name="test_host",
            description="Test host.",
        )
        message = make_message(
            make_function_call_content(PyA2AtomicHost.LIST_INVOKABLES_FUNCTION)
        )

        response = host.handle_message(message)

        assert response.content.type == "function_response"
        assert response.content.name == PyA2AtomicHost.LIST_INVOKABLES_FUNCTION
        assert response.content.response["echo"]["name"] == "echo"

    def test_handle_get_invokable_metadata_function_call(self) -> None:
        host = PyA2AtomicHost(
            [make_invokable("echo")],
            name="test_host",
            description="Test host.",
        )
        message = make_message(
            make_function_call_content(
                PyA2AtomicHost.GET_INVOKABLE_METADATA_FUNCTION,
                [make_param_content("name", "echo")],
            )
        )

        response = host.handle_message(message)

        assert response.content.type == "function_response"
        assert response.content.response["name"] == "echo"

    def test_handle_direct_invokable_function_call(self) -> None:
        host = PyA2AtomicHost(
            [make_invokable("echo")],
            name="test_host",
            description="Test host.",
        )
        message = make_message(
            make_function_call_content(
                "echo",
                [make_param_content("value", 123)],
            )
        )

        response = host.handle_message(message)

        assert response.content.type == "function_response"
        assert response.content.response[PYA2A_RESULT_KEY] == {"value": 123}

    def test_handle_unknown_invokable_returns_error_payload(self) -> None:
        host = PyA2AtomicHost(
            [],
            name="test_host",
            description="Test host.",
        )
        message = make_message(make_function_call_content("missing"))

        response = host.handle_message(message)

        assert response.content.type == "function_response"
        assert response.content.response["error_type"] == "KeyError"

    def test_handle_duplicate_parameters_returns_error_payload(self) -> None:
        host = PyA2AtomicHost(
            [],
            name="test_host",
            description="Test host.",
        )
        message = make_message(
            make_function_call_content(
                "echo",
                [
                    make_param_content("value", 1),
                    make_param_content("value", 2),
                ],
            )
        )

        response = host.handle_message(message)

        assert response.content.type == "function_response"
        assert response.content.response["error_type"] == "ValueError"


class TestPyA2AtomicHostSerialization:
    def test_to_dict(self) -> None:
        host = PyA2AtomicHost(
            [make_invokable("echo")],
            name="test_host",
            description="Test host.",
            version="1.2.3",
            host="127.0.0.1",
            port=9999,
        )

        data = host.to_dict()

        assert data["type"] == "PyA2AtomicHost"
        assert data["name"] == "test_host"
        assert data["description"] == "Test host."
        assert data["version"] == "1.2.3"
        assert data["host"] == "127.0.0.1"
        assert data["port"] == 9999
        assert data["url"] == "http://127.0.0.1:9999"
        assert data["invokable_names"] == ["echo"]
        assert data["invokable_count"] == 1
