from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

from atomic_agentic.a2a.PyA2AtomicClient import PyA2AtomicClient
from atomic_agentic.core.Exceptions import ToolDefinitionError, ToolInvocationError
from atomic_agentic.core.Parameters import ParamSpec
from atomic_agentic.tools.a2a import PyA2AtomicTool
from atomic_agentic.core.sentinels import NO_VAL

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
        "filter_extraneous_inputs": True,
        "invokable_type": "EchoInvokable",
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

    def call_invokable(self, remote_name: str, inputs: Mapping[str, Any]) -> Any:
        self.calls.append((remote_name, dict(inputs)))
        if self.call_error is not None:
            raise self.call_error
        return self.result

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
            ("value", ParamSpec.POSITIONAL_OR_KEYWORD, "Any", NO_VAL),
            ("tag", ParamSpec.POSITIONAL_OR_KEYWORD, "str", "default"),
        ]
        assert tool.return_type == "dict[str, Any]"

    def test_remote_metadata_property_returns_copy(self) -> None:
        tool = make_tool()

        snapshot = tool.remote_metadata
        snapshot["description"] = "mutated"

        assert tool.remote_metadata["description"] == "Remote echo invokable."

    def test_invalid_remote_parameters_type_raises_on_refresh(self) -> None:
        client = FakePyA2AtomicClient()
        tool = make_tool(client=client)

        client.metadata = {
            **remote_metadata(),
            "parameters": "bad",
        }

        with pytest.raises(ToolDefinitionError, match="'parameters' must be a list"):
            tool.refresh()

    def test_invalid_remote_parameter_item_raises_on_refresh(self) -> None:
        client = FakePyA2AtomicClient()
        tool = make_tool(client=client)

        client.metadata = {
            **remote_metadata(),
            "parameters": ["bad"],
        }

        with pytest.raises(ToolDefinitionError, match="parameters\\[0\\]"):
            tool.refresh()

    def test_invalid_remote_return_type_raises_on_refresh(self) -> None:
        client = FakePyA2AtomicClient()
        tool = make_tool(client=client)

        client.metadata = {
            **remote_metadata(),
            "return_type": 123,
        }

        with pytest.raises(ToolDefinitionError, match="'return_type' must be a str"):
            tool.refresh()


class TestPyA2AtomicToolInvocation:
    def test_invoke_forwards_remote_name_and_kwargs_to_client(self) -> None:
        client = FakePyA2AtomicClient(result={"value": 123})
        tool = make_tool(client=client)

        result = tool.invoke({"value": 123})

        assert result == {"value": 123}
        assert client.calls == [("echo", {"value": 123})]

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

    def test_async_execute_forwards_to_client_in_thread(self) -> None:
        client = FakePyA2AtomicClient(result={"async": True})
        tool = make_tool(client=client)

        result = asyncio.run(tool.async_invoke({"value": "hello"}))

        assert result == {"async": True}
        assert client.calls == [("echo", {"value": "hello"})]

    def test_async_execute_rejects_positional_args(self) -> None:
        tool = make_tool()

        with pytest.raises(ToolInvocationError, match="do not accept positional"):
            asyncio.run(tool.async_execute((1,), {}))

    def test_async_execute_wraps_client_exception(self) -> None:
        client = FakePyA2AtomicClient(call_error=RuntimeError("remote boom"))
        tool = make_tool(client=client)

        with pytest.raises(ToolInvocationError, match="async invocation failed"):
            asyncio.run(tool.async_invoke({"value": "hello"}))


class TestPyA2AtomicToolRefreshAndSerialization:
    def test_headers_setter_updates_client_headers_and_refreshes_metadata(self) -> None:
        client = FakePyA2AtomicClient()
        tool = make_tool(client=client)

        client.metadata = {
            **remote_metadata(return_type="str"),
            "parameters": [param_dict("value", 0, type_="str")],
        }

        tool.headers = {"Authorization": "Bearer token"}

        assert client.headers == {"Authorization": "Bearer token"}
        assert tool.return_type == "str"
        assert [(p.name, p.type) for p in tool.parameters] == [("value", "str")]

    def test_refresh_with_headers_updates_client_headers(self) -> None:
        client = FakePyA2AtomicClient()
        tool = make_tool(client=client)

        tool.refresh(headers={"X-Test": "yes"})

        assert client.headers == {"X-Test": "yes"}

    def test_refresh_rebuilds_schema_but_keeps_local_identity_and_description(self) -> None:
        client = FakePyA2AtomicClient()
        tool = make_tool(
            client=client,
            name="local_echo",
            namespace="a2a_tools",
            description="Original local description.",
        )

        client.metadata = {
            **remote_metadata(description="Fresh remote description.", return_type="str"),
            "parameters": [param_dict("value", 0, type_="str")],
        }

        tool.refresh()

        assert tool.name == "local_echo"
        assert tool.namespace == "a2a_tools"
        assert tool.description == "Original local description."
        assert tool.return_type == "str"
        assert [(p.name, p.type) for p in tool.parameters] == [("value", "str")]

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
