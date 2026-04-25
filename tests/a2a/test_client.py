from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any

import pytest

client_module = importlib.import_module("atomic_agentic.a2a.PyA2AtomicClient")

from atomic_agentic.a2a.PyA2AtomicClient import PyA2AtomicClient
from atomic_agentic.a2a.PyA2AtomicHost import PYA2A_RESULT_KEY


class FakeA2AClient:
    def __init__(self, url: str, headers: dict[str, str] | None = None) -> None:
        self.url = url
        self.headers = headers
        self.sent_messages: list[Any] = []
        self.agent_card = SimpleNamespace(name="fake_agent")
        self.next_payload: dict[str, Any] = {}

    def get_agent_card(self) -> Any:
        return self.agent_card

    def send_message(self, message: Any) -> Any:
        self.sent_messages.append(message)
        return SimpleNamespace(
            content=SimpleNamespace(
                type="function_response",
                response=dict(self.next_payload),
            )
        )


class FakeA2AClientFactory:
    def __init__(self) -> None:
        self.instances: list[FakeA2AClient] = []

    def __call__(
        self,
        url: str,
        headers: dict[str, str] | None = None,
    ) -> FakeA2AClient:
        client = FakeA2AClient(url, headers=headers)
        self.instances.append(client)
        return client


@pytest.fixture
def fake_client_factory(monkeypatch: pytest.MonkeyPatch) -> FakeA2AClientFactory:
    factory = FakeA2AClientFactory()
    monkeypatch.setattr(client_module, "A2AClient", factory)
    return factory


def latest_fake_client(factory: FakeA2AClientFactory) -> FakeA2AClient:
    return factory.instances[-1]


class TestPyA2AtomicClientConstruction:
    def test_valid_construction_fetches_agent_card(
        self,
        fake_client_factory: FakeA2AClientFactory,
    ) -> None:
        client = PyA2AtomicClient("http://example.test/a2a")

        fake = latest_fake_client(fake_client_factory)

        assert client.url == "http://example.test/a2a"
        assert client.agent_card.name == "fake_agent"
        assert fake.url == "http://example.test/a2a"
        assert fake.headers is None

    @pytest.mark.parametrize("url", ["", "   "])
    def test_invalid_url_raises(self, url: str) -> None:
        with pytest.raises(ValueError, match="url"):
            PyA2AtomicClient(url)

    def test_headers_are_normalized_and_hidden_in_to_dict(
        self,
        fake_client_factory: FakeA2AClientFactory,
    ) -> None:
        client = PyA2AtomicClient(
            "http://example.test/a2a",
            headers={"Authorization": "Bearer secret", "X-Test": "yes"},
        )

        fake = latest_fake_client(fake_client_factory)

        assert client.headers == {"Authorization": "Bearer secret", "X-Test": "yes"}
        assert fake.headers == {"Authorization": "Bearer secret", "X-Test": "yes"}

        data = client.to_dict()
        assert data["type"] == "PyA2AtomicClient"
        assert data["url"] == "http://example.test/a2a"
        assert data["has_headers"] is True
        assert data["header_keys"] == ["Authorization", "X-Test"]
        assert data["agent_name"] == "fake_agent"
        assert "Bearer secret" not in str(data)

    def test_headers_are_immutable(
        self,
        fake_client_factory: FakeA2AClientFactory,
    ) -> None:
        client = PyA2AtomicClient(
            "http://example.test/a2a",
            headers={"Authorization": "Bearer secret"},
        )

        with pytest.raises(TypeError):
            client.headers["Authorization"] = "changed"  # type: ignore[index]

    def test_headers_setter_rebuilds_client(
        self,
        fake_client_factory: FakeA2AClientFactory,
    ) -> None:
        client = PyA2AtomicClient("http://example.test/a2a")

        client.headers = {"X-Test": "yes"}

        assert len(fake_client_factory.instances) == 2
        assert client.headers == {"X-Test": "yes"}
        assert latest_fake_client(fake_client_factory).headers == {"X-Test": "yes"}

    @pytest.mark.parametrize(
        "headers",
        [
            "bad",
            {"ok": 1},
            {1: "ok"},
        ],
    )
    def test_invalid_headers_raise(self, headers: object) -> None:
        with pytest.raises(ValueError, match="headers"):
            PyA2AtomicClient(
                "http://example.test/a2a",
                headers=headers,  # type: ignore[arg-type]
            )


class TestPyA2AtomicClientFunctionCalls:
    def test_send_function_call_builds_message_and_returns_payload(
        self,
        fake_client_factory: FakeA2AClientFactory,
    ) -> None:
        client = PyA2AtomicClient("http://example.test/a2a")
        fake = latest_fake_client(fake_client_factory)
        fake.next_payload = {"ok": True}

        payload = client._send_function_call(
            function_name="echo",
            parameters={"value": 123},
        )

        assert payload == {"ok": True}
        assert len(fake.sent_messages) == 1

        message = fake.sent_messages[0]
        role_value = getattr(message.role, "value", message.role)

        assert str(role_value).lower().endswith("user")
        assert message.content.name == "echo"
        assert [(p.name, p.value) for p in message.content.parameters] == [
            ("value", 123),
        ]

    def test_send_function_call_strips_function_name(
        self,
        fake_client_factory: FakeA2AClientFactory,
    ) -> None:
        client = PyA2AtomicClient("http://example.test/a2a")
        fake = latest_fake_client(fake_client_factory)
        fake.next_payload = {"ok": True}

        client._send_function_call(
            function_name="  echo  ",
            parameters={},
        )

        message = fake.sent_messages[0]
        assert message.content.name == "echo"

    def test_send_function_call_rejects_blank_name(
        self,
        fake_client_factory: FakeA2AClientFactory,
    ) -> None:
        client = PyA2AtomicClient("http://example.test/a2a")

        with pytest.raises(ValueError, match="function_name"):
            client._send_function_call(function_name="   ", parameters={})

    def test_send_function_call_rejects_non_mapping_parameters(
        self,
        fake_client_factory: FakeA2AClientFactory,
    ) -> None:
        client = PyA2AtomicClient("http://example.test/a2a")

        with pytest.raises(TypeError, match="parameters"):
            client._send_function_call(
                function_name="echo",
                parameters=["bad"],  # type: ignore[arg-type]
            )

    def test_send_function_call_rejects_non_function_response(
        self,
        fake_client_factory: FakeA2AClientFactory,
    ) -> None:
        client = PyA2AtomicClient("http://example.test/a2a")
        fake = latest_fake_client(fake_client_factory)

        def send_message(_: Any) -> Any:
            return SimpleNamespace(content=SimpleNamespace(type="text", response={}))

        fake.send_message = send_message  # type: ignore[method-assign]

        with pytest.raises(RuntimeError, match="expected function_response"):
            client._send_function_call(function_name="echo", parameters={})

    def test_send_function_call_rejects_non_mapping_payload(
        self,
        fake_client_factory: FakeA2AClientFactory,
    ) -> None:
        client = PyA2AtomicClient("http://example.test/a2a")
        fake = latest_fake_client(fake_client_factory)

        def send_message(_: Any) -> Any:
            return SimpleNamespace(
                content=SimpleNamespace(type="function_response", response=["bad"])
            )

        fake.send_message = send_message  # type: ignore[method-assign]

        with pytest.raises(RuntimeError, match="must be a mapping"):
            client._send_function_call(function_name="echo", parameters={})

    def test_send_function_call_wraps_send_message_errors(
        self,
        fake_client_factory: FakeA2AClientFactory,
    ) -> None:
        client = PyA2AtomicClient("http://example.test/a2a")
        fake = latest_fake_client(fake_client_factory)

        def send_message(_: Any) -> Any:
            raise RuntimeError("network boom")

        fake.send_message = send_message  # type: ignore[method-assign]

        with pytest.raises(RuntimeError, match="send_message failed"):
            client._send_function_call(function_name="echo", parameters={})


class TestPyA2AtomicClientPublicAPI:
    def test_list_invokables_validates_payload(
        self,
        fake_client_factory: FakeA2AClientFactory,
    ) -> None:
        client = PyA2AtomicClient("http://example.test/a2a")
        fake = latest_fake_client(fake_client_factory)
        fake.next_payload = {
            "echo": {
                "name": "echo",
                "description": "Echo.",
            }
        }

        assert client.list_invokables() == {
            "echo": {
                "name": "echo",
                "description": "Echo.",
            }
        }

    def test_list_invokables_rejects_non_string_name(
        self,
        fake_client_factory: FakeA2AClientFactory,
    ) -> None:
        client = PyA2AtomicClient("http://example.test/a2a")
        fake = latest_fake_client(fake_client_factory)
        fake.next_payload = {
            123: {
                "name": "echo",
                "description": "Echo.",
            }
        }

        with pytest.raises(RuntimeError, match="non-string"):
            client.list_invokables()

    def test_list_invokables_rejects_non_mapping_metadata(
        self,
        fake_client_factory: FakeA2AClientFactory,
    ) -> None:
        client = PyA2AtomicClient("http://example.test/a2a")
        fake = latest_fake_client(fake_client_factory)
        fake.next_payload = {"echo": "bad"}

        with pytest.raises(RuntimeError, match="non-mapping metadata"):
            client.list_invokables()

    def test_get_invokable_metadata_validates_required_keys(
        self,
        fake_client_factory: FakeA2AClientFactory,
    ) -> None:
        client = PyA2AtomicClient("http://example.test/a2a")
        fake = latest_fake_client(fake_client_factory)
        fake.next_payload = {
            "name": "echo",
            "description": "Echo.",
            "parameters": [],
            "return_type": "dict[str, Any]",
            "filter_extraneous_inputs": True,
            "invokable_type": "EchoInvokable",
        }

        metadata = client.get_invokable_metadata("echo")

        assert metadata["name"] == "echo"
        assert metadata["invokable_type"] == "EchoInvokable"

    def test_get_invokable_metadata_rejects_blank_name(
        self,
        fake_client_factory: FakeA2AClientFactory,
    ) -> None:
        client = PyA2AtomicClient("http://example.test/a2a")

        with pytest.raises(ValueError, match="remote_name"):
            client.get_invokable_metadata("   ")

    def test_get_invokable_metadata_rejects_missing_required_keys(
        self,
        fake_client_factory: FakeA2AClientFactory,
    ) -> None:
        client = PyA2AtomicClient("http://example.test/a2a")
        fake = latest_fake_client(fake_client_factory)
        fake.next_payload = {"name": "echo"}

        with pytest.raises(RuntimeError, match="missing required key"):
            client.get_invokable_metadata("echo")

    def test_call_invokable_returns_result_key(
        self,
        fake_client_factory: FakeA2AClientFactory,
    ) -> None:
        client = PyA2AtomicClient("http://example.test/a2a")
        fake = latest_fake_client(fake_client_factory)
        fake.next_payload = {PYA2A_RESULT_KEY: {"value": 123}}

        assert client.call_invokable("echo", {"value": 123}) == {"value": 123}

    def test_call_invokable_strips_remote_name(
        self,
        fake_client_factory: FakeA2AClientFactory,
    ) -> None:
        client = PyA2AtomicClient("http://example.test/a2a")
        fake = latest_fake_client(fake_client_factory)
        fake.next_payload = {PYA2A_RESULT_KEY: {"value": 123}}

        result = client.call_invokable("  echo  ", {"value": 123})

        assert result == {"value": 123}
        assert fake.sent_messages[0].content.name == "echo"

    def test_call_invokable_rejects_blank_name(
        self,
        fake_client_factory: FakeA2AClientFactory,
    ) -> None:
        client = PyA2AtomicClient("http://example.test/a2a")

        with pytest.raises(ValueError, match="remote_name"):
            client.call_invokable("   ", {})

    def test_call_invokable_rejects_non_mapping_inputs(
        self,
        fake_client_factory: FakeA2AClientFactory,
    ) -> None:
        client = PyA2AtomicClient("http://example.test/a2a")

        with pytest.raises(TypeError, match="inputs"):
            client.call_invokable("echo", ["bad"])  # type: ignore[arg-type]

    def test_call_invokable_requires_result_key(
        self,
        fake_client_factory: FakeA2AClientFactory,
    ) -> None:
        client = PyA2AtomicClient("http://example.test/a2a")
        fake = latest_fake_client(fake_client_factory)
        fake.next_payload = {"wrong": 123}

        with pytest.raises(RuntimeError, match="required result key"):
            client.call_invokable("echo", {})

    def test_error_payload_raises_runtime_error(
        self,
        fake_client_factory: FakeA2AClientFactory,
    ) -> None:
        client = PyA2AtomicClient("http://example.test/a2a")
        fake = latest_fake_client(fake_client_factory)
        fake.next_payload = {
            "error": "boom",
            "error_type": "ValueError",
        }

        with pytest.raises(RuntimeError, match="ValueError"):
            client.list_invokables()

    def test_error_payload_defaults_error_type(
        self,
        fake_client_factory: FakeA2AClientFactory,
    ) -> None:
        client = PyA2AtomicClient("http://example.test/a2a")
        fake = latest_fake_client(fake_client_factory)
        fake.next_payload = {
            "error": "boom",
            "error_type": "",
        }

        with pytest.raises(RuntimeError, match="RemoteError"):
            client.list_invokables()