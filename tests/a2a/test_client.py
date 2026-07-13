from __future__ import annotations

import asyncio
import importlib
from types import SimpleNamespace
from typing import Any

import pytest

client_module = importlib.import_module("atomic_agentic.a2a.PyA2AtomicClient")

from atomic_agentic.a2a.PyA2AtomicClient import PyA2AtomicClient
from atomic_agentic.a2a.PyA2AtomicHost import PYA2A_RESULT_KEY
from atomic_agentic.exceptions import PyA2AtomicConnectionError, RemoteInvocationError


class FakeA2AClient:
    def __init__(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        timeout: float = 600,
        google_a2a_compatible: bool = False,
    ) -> None:
        self.endpoint_url = url
        self.headers = headers
        self.timeout = timeout
        self._use_google_a2a = google_a2a_compatible
        self.sent_messages: list[Any] = []
        self.agent_card = SimpleNamespace(name="fake_agent")
        self.next_payload: dict[str, Any] = {}
        self.fetch_card_call_count: int = 0

    def get_agent_card(self) -> Any:
        return self.agent_card

    def _fetch_agent_card(self) -> Any:
        self.fetch_card_call_count += 1
        return self.agent_card

    def use_google_a2a_format(self, use_google_format: bool = True) -> None:
        self._use_google_a2a = use_google_format

    def is_using_google_a2a_format(self) -> bool:
        return self._use_google_a2a

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
        timeout: float = 600,
        google_a2a_compatible: bool = False,
    ) -> FakeA2AClient:
        client = FakeA2AClient(
            url,
            headers=headers,
            timeout=timeout,
            google_a2a_compatible=google_a2a_compatible,
        )
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
    def test_scalar_header_values_are_normalized(
        self,
        fake_client_factory: FakeA2AClientFactory,
    ) -> None:
        client = PyA2AtomicClient(
            "http://example.test/a2a",
            headers={
                "X-Int": 1,
                "X-Float": 1.5,
                "X-Bool": True,
                "X-Bytes": b"abc",
                "X-ByteArray": bytearray(b"xyz"),
            },  # type: ignore[arg-type]
        )

        expected = {
            "X-Int": "1",
            "X-Float": "1.5",
            "X-Bool": "true",
            "X-Bytes": "abc",
            "X-ByteArray": "xyz",
        }

        fake = latest_fake_client(fake_client_factory)

        assert client.headers == expected
        assert fake.headers == expected

    def test_valid_construction_fetches_agent_card(
        self,
        fake_client_factory: FakeA2AClientFactory,
    ) -> None:
        client = PyA2AtomicClient("http://example.test/a2a")

        fake = latest_fake_client(fake_client_factory)

        assert client.url == "http://example.test/a2a"
        assert client.agent_card.name == "fake_agent"
        assert fake.endpoint_url == "http://example.test/a2a"
        assert fake.headers is None

    @pytest.mark.parametrize("url", ["", "   "])
    def test_invalid_url_raises(self, url: str) -> None:
        with pytest.raises(ValueError, match="url"):
            PyA2AtomicClient(url)

    def test_url_reflects_live_client_endpoint(
        self,
        fake_client_factory: FakeA2AClientFactory,
    ) -> None:
        """python_a2a's A2AClient rewrites its own endpoint_url after a
        successful call (trying URL variants); `url` must read that live
        value, not a copy cached at construction time."""
        client = PyA2AtomicClient("http://example.test/a2a")
        fake = latest_fake_client(fake_client_factory)

        fake.endpoint_url = "http://example.test/a2a/tasks/send"

        assert client.url == "http://example.test/a2a/tasks/send"

    def test_timeout_and_google_a2a_compatible_defaults(
        self,
        fake_client_factory: FakeA2AClientFactory,
    ) -> None:
        client = PyA2AtomicClient("http://example.test/a2a")

        assert client.timeout == 600
        assert client.google_a2a_compatible is False

    def test_timeout_and_google_a2a_compatible_forwarded(
        self,
        fake_client_factory: FakeA2AClientFactory,
    ) -> None:
        client = PyA2AtomicClient(
            "http://example.test/a2a",
            timeout=45,
            google_a2a_compatible=True,
        )
        fake = latest_fake_client(fake_client_factory)

        assert client.timeout == 45
        assert client.google_a2a_compatible is True
        assert fake.timeout == 45
        assert fake.is_using_google_a2a_format() is True

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
        assert data["timeout"] == 600
        assert data["google_a2a_compatible"] is False
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

    def test_headers_setter_updates_client_headers_without_rebuild(
        self,
        fake_client_factory: FakeA2AClientFactory,
    ) -> None:
        client = PyA2AtomicClient("http://example.test/a2a")

        client.headers = {"X-Test": "yes"}

        assert len(fake_client_factory.instances) == 1
        assert client.headers == {"X-Test": "yes"}
        assert latest_fake_client(fake_client_factory).headers == {"X-Test": "yes"}

    @pytest.mark.parametrize(
        ("headers", "match"),
        [
            ("bad", "headers must be a mapping"),
            ({1: "ok"}, "header names must be strings"),
            ({"bad": None}, "must not be None"),
            ({"bad": {"nested": "value"}}, "must not be a mapping"),
            ({"bad": ["value"]}, "must not be a collection"),
            ({"bad": object()}, "must be str, int, float, bool, bytes, or bytearray"),
            ({"bad": b"\xff"}, "ASCII-decodable"),
            ({"bad\n": "value"}, "forbidden character"),
            ({"bad": "line\nbreak"}, "forbidden character"),
        ],
    )
    def test_invalid_headers_raise(self, headers: object, match: str) -> None:
        with pytest.raises(ValueError, match=match):
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

        with pytest.raises(PyA2AtomicConnectionError, match="expected function_response"):
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

        with pytest.raises(PyA2AtomicConnectionError, match="must be a mapping"):
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

        with pytest.raises(PyA2AtomicConnectionError, match="send_message failed"):
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

        with pytest.raises(PyA2AtomicConnectionError, match="non-string"):
            client.list_invokables()

    def test_list_invokables_rejects_non_mapping_metadata(
        self,
        fake_client_factory: FakeA2AClientFactory,
    ) -> None:
        client = PyA2AtomicClient("http://example.test/a2a")
        fake = latest_fake_client(fake_client_factory)
        fake.next_payload = {"echo": "bad"}

        with pytest.raises(PyA2AtomicConnectionError, match="non-mapping metadata"):
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

        with pytest.raises(PyA2AtomicConnectionError, match="missing required key"):
            client.get_invokable_metadata("echo")

    def test_call_invokable_returns_raw_payload(
        self,
        fake_client_factory: FakeA2AClientFactory,
    ) -> None:
        client = PyA2AtomicClient("http://example.test/a2a")
        fake = latest_fake_client(fake_client_factory)
        fake.next_payload = {PYA2A_RESULT_KEY: {"value": 123}}

        assert client.call_invokable("echo", {"value": 123}) == {PYA2A_RESULT_KEY: {"value": 123}}

    def test_call_invokable_strips_remote_name(
        self,
        fake_client_factory: FakeA2AClientFactory,
    ) -> None:
        client = PyA2AtomicClient("http://example.test/a2a")
        fake = latest_fake_client(fake_client_factory)
        fake.next_payload = {PYA2A_RESULT_KEY: {"value": 123}}

        result = client.call_invokable("  echo  ", {"value": 123})

        assert result == {PYA2A_RESULT_KEY: {"value": 123}}
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

    def test_call_invokable_returns_payload_as_is_without_extraction(
        self,
        fake_client_factory: FakeA2AClientFactory,
    ) -> None:
        client = PyA2AtomicClient("http://example.test/a2a")
        fake = latest_fake_client(fake_client_factory)
        fake.next_payload = {"some_key": 123}

        assert client.call_invokable("echo", {}) == {"some_key": 123}

    def test_error_payload_raises_remote_invocation_error(
        self,
        fake_client_factory: FakeA2AClientFactory,
    ) -> None:
        client = PyA2AtomicClient("http://example.test/a2a")
        fake = latest_fake_client(fake_client_factory)
        fake.next_payload = {
            "error": "boom",
            "error_type": "ValueError",
        }

        with pytest.raises(RemoteInvocationError) as exc_info:
            client.list_invokables()

        assert str(exc_info.value) == "boom"
        assert exc_info.value.error_type == "ValueError"
        assert exc_info.value.function_name == "list_invokables"

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

        with pytest.raises(RemoteInvocationError) as exc_info:
            client.list_invokables()

        assert exc_info.value.error_type == "RemoteError"


class TestPyA2AtomicClientRefresh:
    def test_refresh_with_nothing_provided_raises(
        self,
        fake_client_factory: FakeA2AClientFactory,
    ) -> None:
        client = PyA2AtomicClient(
            "http://example.test/a2a",
            headers={"X-Old": "yes"},
        )

        with pytest.raises(ValueError, match="requires at least one of"):
            client.refresh()

    def test_refresh_timeout_only_updates_timeout_and_refetches(
        self,
        fake_client_factory: FakeA2AClientFactory,
    ) -> None:
        client = PyA2AtomicClient(
            "http://example.test/a2a",
            headers={"X-Old": "yes"},
        )
        fake = latest_fake_client(fake_client_factory)

        client.refresh(timeout=45)

        assert client.timeout == 45
        assert fake.timeout == 45
        assert client.headers == {"X-Old": "yes"}
        assert len(fake_client_factory.instances) == 1
        assert fake.fetch_card_call_count == 1

    def test_refresh_google_a2a_compatible_only_updates_flag_and_refetches(
        self,
        fake_client_factory: FakeA2AClientFactory,
    ) -> None:
        client = PyA2AtomicClient("http://example.test/a2a")
        fake = latest_fake_client(fake_client_factory)

        client.refresh(google_a2a_compatible=True)

        assert client.google_a2a_compatible is True
        assert fake.is_using_google_a2a_format() is True
        assert len(fake_client_factory.instances) == 1
        assert fake.fetch_card_call_count == 1

    def test_refresh_agent_card_fetch_failure_raises_connection_error(
        self,
        fake_client_factory: FakeA2AClientFactory,
    ) -> None:
        client = PyA2AtomicClient("http://example.test/a2a")
        fake = latest_fake_client(fake_client_factory)

        def failing_fetch() -> Any:
            raise ValueError("boom")

        fake._fetch_agent_card = failing_fetch  # type: ignore[method-assign]

        with pytest.raises(PyA2AtomicConnectionError, match="failed to refresh agent card"):
            client.refresh(timeout=45)

    def test_refresh_agent_card_fetch_failure_rolls_back_all_mutations(
        self,
        fake_client_factory: FakeA2AClientFactory,
    ) -> None:
        """A failed refresh() must leave headers/timeout/google_a2a_compatible
        exactly as they were — not silently applied despite the raise."""
        client = PyA2AtomicClient(
            "http://example.test/a2a",
            headers={"X-Old": "yes"},
            timeout=30,
            google_a2a_compatible=False,
        )
        fake = latest_fake_client(fake_client_factory)

        def failing_fetch() -> Any:
            raise ValueError("boom")

        fake._fetch_agent_card = failing_fetch  # type: ignore[method-assign]

        with pytest.raises(PyA2AtomicConnectionError):
            client.refresh(
                headers={"X-New": "yes"},
                timeout=99,
                google_a2a_compatible=True,
            )

        assert client.headers == {"X-Old": "yes"}
        assert fake.headers == {"X-Old": "yes"}
        assert client.timeout == 30
        assert fake.timeout == 30
        assert client.google_a2a_compatible is False
        assert fake.is_using_google_a2a_format() is False

    def test_refresh_with_headers_updates_and_refetches(
        self,
        fake_client_factory: FakeA2AClientFactory,
    ) -> None:
        client = PyA2AtomicClient("http://example.test/a2a")
        fake = latest_fake_client(fake_client_factory)

        client.refresh(headers={"X-New": "yes"})

        assert client.headers == {"X-New": "yes"}
        assert fake.headers == {"X-New": "yes"}
        assert len(fake_client_factory.instances) == 1
        assert fake.fetch_card_call_count == 1

    def test_refresh_updates_agent_card(
        self,
        fake_client_factory: FakeA2AClientFactory,
    ) -> None:
        client = PyA2AtomicClient("http://example.test/a2a")
        fake = latest_fake_client(fake_client_factory)
        fake.agent_card = SimpleNamespace(name="updated_agent")

        client.refresh(timeout=600)

        assert client.agent_card.name == "updated_agent"


class TestPyA2AtomicClientAsyncCreate:
    def test_async_create_constructs_and_fetches_card(
        self,
        fake_client_factory: FakeA2AClientFactory,
    ) -> None:
        client = asyncio.run(PyA2AtomicClient.async_create("http://example.test/a2a"))
        fake = latest_fake_client(fake_client_factory)

        assert isinstance(client, PyA2AtomicClient)
        assert client.url == "http://example.test/a2a"
        assert client.agent_card.name == "fake_agent"
        assert len(fake_client_factory.instances) == 1
        assert fake.endpoint_url == "http://example.test/a2a"

    def test_async_create_forwards_timeout_and_google_a2a_compatible(
        self,
        fake_client_factory: FakeA2AClientFactory,
    ) -> None:
        client = asyncio.run(
            PyA2AtomicClient.async_create(
                "http://example.test/a2a",
                timeout=45,
                google_a2a_compatible=True,
            )
        )
        fake = latest_fake_client(fake_client_factory)

        assert client.timeout == 45
        assert client.google_a2a_compatible is True
        assert fake.timeout == 45
        assert fake.is_using_google_a2a_format() is True