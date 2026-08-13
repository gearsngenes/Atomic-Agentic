from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

from atomic_agentic.a2a.PyA2AtomicClient import PyA2AtomicClient
from atomic_agentic.constants.a2a import PYA2A_RESULT_KEY
from atomic_agentic.constants.core import NO_VAL
from atomic_agentic.exceptions import RemoteInvocationError, ToolDefinitionError, ToolInvocationError
from atomic_agentic.models.parameters import ParamSpec
from atomic_agentic.models.results.tools import PyA2AtomicToolResult
from atomic_agentic.tools.a2a import PyA2AtomicTool

def param_dict(
    name: str,
    index: int,
    *,
    kind: str = ParamSpec.POSITIONAL_OR_KEYWORD,
    type_: str = "Any",
    default: Any = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "name": name,
        "index": index,
        "kind": kind,
        "type": type_,
    }
    if default is not None:
        kwargs["default"] = default
    return ParamSpec(**kwargs).to_dict()


def remote_metadata(
    *,
    description: str = "Remote echo invokable.",
    return_type: str = "dict[str, Any]",
    parameters: list[Mapping[str, Any]] | None = None,
    extra_description: str = "",
) -> dict[str, Any]:
    return {
        "name": "echo",
        "description": description,
        "parameters": list(
            parameters
            if parameters is not None
            else [
                param_dict("value", 0, type_="Any"),
                param_dict("tag", 1, type_="str", default="default"),
            ]
        ),
        "return_type": return_type,
        "invokable_type": "EchoInvokable",
        "extra_description": extra_description,
    }


class FakePyA2AtomicClient(PyA2AtomicClient):
    """Fake A2A client that bypasses network setup but satisfies isinstance."""

    def __init__(
        self,
        *,
        url: str = "http://example.test/a2a",
        headers: Mapping[str, str] | None = None,
        agent_name: str = "remote_agent",
        agent_description: str = "Remote agent card description.",
        metadata: Mapping[str, Any] | None = None,
        result: Any | None = None,
        raw_payload: dict[str, Any] | None = None,
        call_error: Exception | None = None,
    ) -> None:
        self._url = url
        self._headers = dict(headers) if headers is not None else None
        self._agent_card = SimpleNamespace(
            name=agent_name,
            description=agent_description,
        )
        self.metadata: dict[str, Any] = dict(metadata or remote_metadata())
        self.result = result if result is not None else {"ok": True}
        self.raw_payload = raw_payload
        self.call_error = call_error
        self.metadata_calls: list[str] = []
        self.calls: list[tuple[str, dict[str, Any]]] = []

    @property
    def url(self) -> str:
        return self._url

    @property
    def headers(self) -> Mapping[str, str] | None:
        return self._headers

    @headers.setter
    def headers(self, value: Mapping[str, str] | None) -> None:
        self._headers = dict(value) if value is not None else None

    @property
    def agent_card(self) -> Any:
        return self._agent_card

    def get_invokable_metadata(self, remote_name: str) -> dict[str, Any]:
        self.metadata_calls.append(remote_name)
        return dict(self.metadata)

    def call_invokable(self, remote_name: str, inputs: Mapping[str, Any]) -> dict[str, Any]:
        self.calls.append((remote_name, dict(inputs)))
        if self.call_error is not None:
            raise self.call_error
        if self.raw_payload is not None:
            return dict(self.raw_payload)
        return {"__py_a2a_result__": self.result}

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "FakePyA2AtomicClient",
            "url": self.url,
            "has_headers": self.headers is not None,
            "agent_name": self.agent_card.name,
        }


def make_tool(
    *,
    client: FakePyA2AtomicClient | None = None,
    remote_name: str = "echo",
    name: str | None = None,
    namespace: str | None = None,
    description: str | None = None,
) -> PyA2AtomicTool:
    return PyA2AtomicTool(
        remote_name=remote_name,
        name=name,
        namespace=namespace,
        description=description,
        client=client or FakePyA2AtomicClient(),
    )


class TestPyA2AtomicToolConstruction:
    def test_valid_construction_with_fake_client(self) -> None:
        client = FakePyA2AtomicClient()
        tool = make_tool(client=client)

        assert tool.client is client
        assert tool.remote_name == "echo"
        assert tool.url == "http://example.test/a2a"
        assert tool.headers is None
        assert tool.agent_card.name == "remote_agent"
        assert tool.name == "echo"
        assert tool.namespace == "remote_agent"
        assert tool.description == "Remote echo invokable."
        assert tool.full_name == "PyA2AtomicTool.remote_agent.echo"
        assert client.metadata_calls == ["echo"]

    def test_remote_name_is_stripped(self) -> None:
        tool = make_tool(remote_name="  echo  ")

        assert tool.remote_name == "echo"

    def test_explicit_identity_overrides_defaults(self) -> None:
        tool = make_tool(
            name="local_echo",
            namespace="a2a_tools",
            description="Local description.",
        )

        assert tool.name == "local_echo"
        assert tool.namespace == "a2a_tools"
        assert tool.description == "Local description."
        assert tool.full_name == "PyA2AtomicTool.a2a_tools.local_echo"

    def test_namespace_falls_back_to_pya2a_when_agent_card_name_missing(self) -> None:
        client = FakePyA2AtomicClient(agent_name="")

        tool = make_tool(client=client)

        assert tool.namespace == "pya2a"

    def test_description_uses_agent_card_when_metadata_description_missing(self) -> None:
        client = FakePyA2AtomicClient(
            metadata=remote_metadata(description=""),
            agent_description="Agent card fallback.",
        )

        tool = make_tool(client=client)

        assert tool.description == "Agent card fallback."

    def test_description_uses_stub_when_metadata_and_agent_card_description_missing(self) -> None:
        client = FakePyA2AtomicClient(
            metadata=remote_metadata(description=""),
            agent_description="",
        )

        tool = make_tool(client=client)

        assert tool.description == "PyA2Atomic tool 'echo'"

    def test_missing_remote_name_raises(self) -> None:
        with pytest.raises(ToolDefinitionError, match="remote_name"):
            make_tool(remote_name="   ")

    def test_non_client_raises(self) -> None:
        with pytest.raises(TypeError, match="PyA2AtomicClient"):
            PyA2AtomicTool(remote_name="echo", client=object())  # type: ignore[arg-type]

    def test_client_plus_raw_transport_settings_raises(self) -> None:
        with pytest.raises(ValueError, match="either client or raw transport"):
            PyA2AtomicTool(
                remote_name="echo",
                client=FakePyA2AtomicClient(),
                url="http://example.test",
            )

    def test_url_required_when_client_not_provided(self) -> None:
        with pytest.raises(ValueError, match="url is required"):
            PyA2AtomicTool(remote_name="echo")


class TestPyA2AtomicToolSignatureAndMetadata:
    def test_parameters_and_return_type_come_from_remote_metadata(self) -> None:
        tool = make_tool()

        assert [(p.name, p.kind, p.type, p.default) for p in tool.parameters] == [
            ("value", ParamSpec.POSITIONAL_OR_KEYWORD, ("Any",), NO_VAL),
            ("tag", ParamSpec.POSITIONAL_OR_KEYWORD, ("str",), "default"),
        ]
        assert tool.return_type == "dict[str, Any]"

    def test_remote_metadata_property_returns_copy(self) -> None:
        tool = make_tool()

        snapshot = tool.remote_metadata
        snapshot["description"] = "mutated"

        assert tool.remote_metadata["description"] == "Remote echo invokable."

class TestPyA2AtomicToolExtraDescription:
    def test_extra_description_empty_by_default(self) -> None:
        tool = make_tool()

        assert tool._extra_description() == ""

    def test_surfaces_remote_extra_description_verbatim(self) -> None:
        client = FakePyA2AtomicClient(
            metadata=remote_metadata(extra_description="Output schema: [result]")
        )
        tool = make_tool(client=client)

        assert tool._extra_description() == "Output schema: [result]"

    def test_extra_description_defaults_to_empty_when_key_missing_from_legacy_host(self) -> None:
        legacy_metadata = remote_metadata()
        del legacy_metadata["extra_description"]
        client = FakePyA2AtomicClient(metadata=legacy_metadata)
        tool = make_tool(client=client)

        assert tool._extra_description() == ""


class TestPyA2AtomicToolInvocation:
    def test_invoke_forwards_remote_name_and_kwargs_to_client(self) -> None:
        client = FakePyA2AtomicClient(result={"value": 123})
        tool = make_tool(client=client)

        result = tool.invoke({"value": 123})

        assert result.result == {"value": 123}
        assert client.calls == [("echo", {"value": 123, "tag": "default"})]

    def test_to_arg_kwarg_returns_empty_args_and_copy_of_kwargs(self) -> None:
        tool = make_tool()
        inputs = {"value": 123}

        args, kwargs = tool.to_arg_kwarg(inputs)

        assert args == ()
        assert kwargs == {"value": 123}
        assert kwargs is not inputs

    def test_execute_rejects_positional_args(self) -> None:
        tool = make_tool()

        with pytest.raises(ToolInvocationError, match="do not accept positional"):
            tool.execute((1,), {})

    def test_execute_wraps_client_exception(self) -> None:
        client = FakePyA2AtomicClient(call_error=RuntimeError("remote boom"))
        tool = make_tool(client=client)

        with pytest.raises(ToolInvocationError, match="invocation failed"):
            tool.invoke({"value": "hello"})

    def test_async_execute_forwards_to_client_in_thread(self) -> None:
        client = FakePyA2AtomicClient(result={"async": True})
        tool = make_tool(client=client)

        result = asyncio.run(tool.async_invoke({"value": "hello"}))

        assert result.result == {"async": True}
        assert client.calls == [("echo", {"value": "hello", "tag": "default"})]

    def test_async_execute_rejects_positional_args(self) -> None:
        tool = make_tool()

        with pytest.raises(ToolInvocationError, match="do not accept positional"):
            asyncio.run(tool.async_execute((1,), {}))

    def test_async_execute_wraps_client_exception(self) -> None:
        client = FakePyA2AtomicClient(call_error=RuntimeError("remote boom"))
        tool = make_tool(client=client)

        with pytest.raises(ToolInvocationError, match="async invocation failed"):
            asyncio.run(tool.async_invoke({"value": "hello"}))


class TestPyA2AtomicToolMakeResult:
    def test_invoke_returns_pya2a_tool_result(self) -> None:
        client = FakePyA2AtomicClient(result={"ok": True})
        tool = make_tool(client=client)

        result = tool.invoke({"value": 1})

        assert isinstance(result, PyA2AtomicToolResult)

    def test_result_carries_url_from_client(self) -> None:
        client = FakePyA2AtomicClient(url="http://myhost.test/a2a")
        tool = make_tool(client=client)

        result = tool.invoke({"value": 1})

        assert result.url == "http://myhost.test/a2a"

    def test_result_carries_remote_name(self) -> None:
        tool = make_tool()

        result = tool.invoke({"value": 1})

        assert result.remote_name == "echo"

    def test_result_carries_invokable_type_from_metadata(self) -> None:
        tool = make_tool()

        result = tool.invoke({"value": 1})

        assert result.invokable_type == "EchoInvokable"

    def test_to_dict_includes_transport_fields(self) -> None:
        client = FakePyA2AtomicClient(url="http://myhost.test/a2a")
        tool = make_tool(client=client)

        data = tool.invoke({"value": 1}).to_dict()

        assert data["url"] == "http://myhost.test/a2a"
        assert data["remote_name"] == "echo"
        assert data["invokable_type"] == "EchoInvokable"

    def test_async_invoke_also_returns_pya2a_tool_result(self) -> None:
        client = FakePyA2AtomicClient(result={"async": True})
        tool = make_tool(client=client)

        result = asyncio.run(tool.async_invoke({"value": "hello"}))

        assert isinstance(result, PyA2AtomicToolResult)
        assert result.remote_name == "echo"
        assert result.invokable_type == "EchoInvokable"


class TestPyA2AtomicToolErrorHandling:
    def test_error_payload_raises_remote_invocation_error(self) -> None:
        client = FakePyA2AtomicClient(
            raw_payload={"error": "boom", "error_type": "ToolInvocationError", "function_name": "echo"}
        )
        tool = make_tool(client=client)

        with pytest.raises(RemoteInvocationError):
            tool.invoke({"value": 1})

    def test_error_payload_carries_error_type(self) -> None:
        client = FakePyA2AtomicClient(
            raw_payload={"error": "something failed", "error_type": "AgentError", "function_name": "echo"}
        )
        tool = make_tool(client=client)

        with pytest.raises(RemoteInvocationError) as exc_info:
            tool.invoke({"value": 1})

        assert exc_info.value.error_type == "AgentError"

    def test_error_payload_carries_function_name(self) -> None:
        client = FakePyA2AtomicClient(
            raw_payload={"error": "something failed", "error_type": "AgentError", "function_name": "echo"}
        )
        tool = make_tool(client=client)

        with pytest.raises(RemoteInvocationError) as exc_info:
            tool.invoke({"value": 1})

        assert exc_info.value.function_name == "echo"

    def test_missing_error_type_falls_back_to_remote_error(self) -> None:
        client = FakePyA2AtomicClient(raw_payload={"error": "unknown failure"})
        tool = make_tool(client=client)

        with pytest.raises(RemoteInvocationError) as exc_info:
            tool.invoke({"value": 1})

        assert exc_info.value.error_type == "RemoteError"

    def test_invalid_error_type_falls_back_to_remote_error(self) -> None:
        client = FakePyA2AtomicClient(raw_payload={"error": "bad type", "error_type": 42})
        tool = make_tool(client=client)

        with pytest.raises(RemoteInvocationError) as exc_info:
            tool.invoke({"value": 1})

        assert exc_info.value.error_type == "RemoteError"

    def test_missing_result_key_raises_tool_invocation_error(self) -> None:
        client = FakePyA2AtomicClient(raw_payload={"some_other_key": "data"})
        tool = make_tool(client=client)

        with pytest.raises(ToolInvocationError, match="missing required result key"):
            tool.invoke({"value": 1})

    def test_async_error_payload_raises_remote_invocation_error(self) -> None:
        client = FakePyA2AtomicClient(
            raw_payload={"error": "async boom", "error_type": "RuntimeError", "function_name": "echo"}
        )
        tool = make_tool(client=client)

        with pytest.raises(RemoteInvocationError) as exc_info:
            asyncio.run(tool.async_invoke({"value": 1}))

        assert exc_info.value.error_type == "RuntimeError"


class TestPyA2AtomicToolHeadersAndSerialization:
    def test_headers_has_no_setter(self) -> None:
        client = FakePyA2AtomicClient()
        tool = make_tool(client=client)

        with pytest.raises(AttributeError):
            tool.headers = {"Authorization": "Bearer token"}  # type: ignore[misc]

    def test_url_headers_and_agent_card_proxy_through_client(self) -> None:
        client = FakePyA2AtomicClient(
            url="http://remote.test",
            headers={"X-Test": "yes"},
            agent_name="agent_one",
        )
        tool = make_tool(client=client)

        assert tool.url == "http://remote.test"
        assert tool.headers == {"X-Test": "yes"}
        assert tool.agent_card.name == "agent_one"

    def test_to_dict_includes_remote_name_and_client_snapshot(self) -> None:
        tool = make_tool()

        data = tool.to_dict()

        assert data["type"] == "PyA2AtomicTool"
        assert data["name"] == "echo"
        assert data["namespace"] == "remote_agent"
        assert data["remote_name"] == "echo"
        assert data["client"]["type"] == "FakePyA2AtomicClient"
        assert data["client"]["url"] == "http://example.test/a2a"
