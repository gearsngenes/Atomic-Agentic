from __future__ import annotations

import importlib
import warnings
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("python_a2a", reason="requires the python-a2a extra")

client_module = importlib.import_module("atomic_agentic.a2a.PyA2AtomicClient")

from atomic_agentic.a2a.PyA2AtomicClient import PyA2AtomicClient
from atomic_agentic.a2a.PyA2AtomicHost import PyA2AtomicHost
from atomic_agentic.models.parameters import ParamSpec
from atomic_agentic.tools import Tool
from atomic_agentic.tools.python_a2a import PyA2AtomicTool


class _FakeA2AClient:
    def __init__(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        timeout: float = 600,
        google_a2a_compatible: bool = False,
    ) -> None:
        self.endpoint_url = url
        self.headers = headers
        self.agent_card = SimpleNamespace(name="fake_agent")

    def get_agent_card(self) -> Any:
        return self.agent_card


class _MetadataClient(PyA2AtomicClient):
    def get_invokable_metadata(self, remote_name: str) -> dict[str, Any]:
        return {
            "name": remote_name,
            "description": "Remote echo.",
            "parameters": [
                ParamSpec(
                    name="value",
                    index=0,
                    kind=ParamSpec.POSITIONAL_OR_KEYWORD,
                    type="Any",
                ).to_dict()
            ],
            "return_type": "Any",
            "invokable_type": "EchoInvokable",
            "extra_description": "",
        }


@pytest.fixture(autouse=True)
def fake_a2a_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(client_module, "A2AClient", _FakeA2AClient)


def _echo(value: str) -> str:
    return value


class TestPyA2AtomicDeprecation:
    def test_client_construction_warns_once_at_the_caller(self) -> None:
        with pytest.warns(DeprecationWarning, match=r"2\.0\.0.*A2AClientHub") as record:
            PyA2AtomicClient("http://example.test/a2a")

        deprecations = [w for w in record if issubclass(w.category, DeprecationWarning)]
        assert len(deprecations) == 1
        assert deprecations[0].filename == __file__

    def test_host_construction_warns_once_at_the_caller(self) -> None:
        tool = Tool(function=_echo, name="echo", namespace="tests")

        with pytest.warns(DeprecationWarning, match=r"2\.0\.0.*A2AtomicExecutor") as record:
            PyA2AtomicHost([tool], name="host", description="Host.")

        deprecations = [w for w in record if issubclass(w.category, DeprecationWarning)]
        assert len(deprecations) == 1
        assert deprecations[0].filename == __file__

    def test_tool_built_from_a_client_does_not_warn_again(self) -> None:
        with pytest.warns(DeprecationWarning):
            client = _MetadataClient("http://example.test/a2a")

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            PyA2AtomicTool(remote_name="echo", client=client)

        assert not [w for w in record if "2.0.0" in str(w.message)]
