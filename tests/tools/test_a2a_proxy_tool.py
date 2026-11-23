"""Unit-level `A2AProxyTool` identity, description and extra-description behavior.

Server-backed behavior lives in `tests/a2a/test_proxy_tool.py`; this module uses
a fake hub so nothing here needs a network.
"""
from __future__ import annotations

from typing import Any, Mapping

import pytest

from atomic_agentic.a2a.a2a_client_hub import A2AClientHub
from atomic_agentic.exceptions import ToolDefinitionError
from atomic_agentic.models.a2a_sdk import A2AtomicSkillMetadata
from atomic_agentic.models.parameters import ParamSpec
from atomic_agentic.models.results.tools import A2AProxyToolResult
from atomic_agentic.tools.a2a_sdk import A2AProxyTool


class _FakeHub(A2AClientHub):
    """Skips A2AClientHub.__init__: fixed agent card and skills, no connection."""

    def __init__(
        self,
        *,
        skills: dict[str, A2AtomicSkillMetadata] | None = None,
        card_description: str = "Fake agent.",
        hub_name: str | None = None,
    ) -> None:
        self._skills = {"add": _skill()} if skills is None else skills
        self._card = type("FakeAgentCard", (), {"name": "fake", "description": card_description})()
        self._hub_name = hub_name

    @property
    def agent_card(self) -> Any:
        return self._card

    @property
    def transport_mode(self) -> str:
        return "JSONRPC"

    @property
    def location(self) -> str:
        return "http://example.test/a2a-sdk"

    @property
    def name(self) -> str | None:
        return self._hub_name

    @property
    def persistent(self) -> bool:
        return False

    def list_skills(self) -> Mapping[str, A2AtomicSkillMetadata]:
        return dict(self._skills)


def _skill(
    remote_name: str = "add",
    description: str = "Remote skill.",
    extra_description: str = "",
) -> A2AtomicSkillMetadata:
    return A2AtomicSkillMetadata(
        remote_name=remote_name,
        description=description,
        extra_description=extra_description,
        params=(
            ParamSpec(name="a", index=0, kind=ParamSpec.POSITIONAL_OR_KEYWORD, type="int"),
            ParamSpec(name="b", index=1, kind=ParamSpec.POSITIONAL_OR_KEYWORD, type="int"),
        ),
        return_type="int",
    )


class TestNaming:
    def test_named_hub_prefixes_skill_name(self) -> None:
        hub = _FakeHub(hub_name="calc")

        tool = A2AProxyTool(hub, "add")

        assert tool.client_hub is hub
        assert tool.name == "calc_add"
        assert tool.remote_name == "add"

    def test_unnamed_hub_uses_raw_skill_name(self) -> None:
        assert A2AProxyTool(_FakeHub(), "add").name == "add"

    def test_generic_mode_naming(self) -> None:
        named = A2AProxyTool(_FakeHub(hub_name="calc"))
        unnamed = A2AProxyTool(_FakeHub())

        assert named.name == "calc_send_parts"
        assert unnamed.name == "send_parts"
        assert named.remote_name is None

    def test_unknown_skill_raises(self) -> None:
        with pytest.raises(ToolDefinitionError):
            A2AProxyTool(_FakeHub(), "missing")

    def test_blank_name_raises(self) -> None:
        with pytest.raises(ValueError):
            A2AProxyTool(_FakeHub(), "  ")

    def test_constructor_takes_only_hub_and_name(self) -> None:
        with pytest.raises(TypeError):
            A2AProxyTool(_FakeHub(), "add", "local_add")  # type: ignore[call-arg]
        with pytest.raises(TypeError):
            A2AProxyTool(_FakeHub(), "add", description="x")  # type: ignore[call-arg]

    def test_passthrough_properties_and_skill_id_are_gone(self) -> None:
        tool = A2AProxyTool(_FakeHub(), "add")

        for attr in ("skill_id", "base_url", "persistent", "transport_mode"):
            assert not hasattr(tool, attr)


class TestExecutePayload:
    def test_skill_mode_payload_keys(self) -> None:
        hub = _FakeHub(hub_name="calc")
        hub.call_atomic_skill = lambda skill, inputs: 3  # type: ignore[method-assign]
        tool = A2AProxyTool(hub, "add")

        payload = tool.execute((), {"a": 1, "b": 2})

        assert payload == {
            "transport_mode": "JSONRPC",
            "location": "http://example.test/a2a-sdk",
            "persistent": False,
            "remote_name": "add",
            "result": 3,
        }

    def test_result_class_validates_payload(self) -> None:
        hub = _FakeHub()
        hub.call_atomic_skill = lambda skill, inputs: 3  # type: ignore[method-assign]

        result = A2AProxyTool(hub, "add").invoke({"a": 1, "b": 2})

        assert isinstance(result, A2AProxyToolResult)
        assert result.location == "http://example.test/a2a-sdk"
        assert result.remote_name == "add"


class TestDescriptionFallback:
    def test_skill_mode_uses_skill_description_over_card(self) -> None:
        tool = A2AProxyTool(_FakeHub(), "add")

        assert tool._description == "Remote skill."

    def test_skill_mode_uses_card_description_when_skill_description_blank(self) -> None:
        hub = _FakeHub(skills={"add": _skill(description="")}, card_description="Card text.")

        assert A2AProxyTool(hub, "add")._description == "Card text."

    def test_skill_mode_uses_stub_when_both_blank(self) -> None:
        hub = _FakeHub(skills={"add": _skill(description="")}, card_description="")

        assert A2AProxyTool(hub, "add")._description.startswith("A2A atomic skill proxy tool 'add'")

    def test_description_is_replaceable_through_setter(self) -> None:
        tool = A2AProxyTool(_FakeHub(), "add")

        tool.description = "Explicit."

        assert tool._description == "Explicit."

    def test_generic_mode_uses_card_then_default_text(self) -> None:
        assert A2AProxyTool(_FakeHub(card_description="Card text."))._description == "Card text."
        assert A2AProxyTool(_FakeHub(card_description=""))._description.startswith(
            "A2A generic send-parts proxy tool"
        )


class TestExtraDescription:
    def test_skill_mode_surfaces_remote_extra_description(self) -> None:
        hub = _FakeHub(skills={"add": _skill(extra_description="Output schema: [result]")})

        tool = A2AProxyTool(hub, "add")

        assert tool._extra_description() == "Output schema: [result]"
        assert "Output schema: [result]" in tool.description

    def test_empty_by_default(self) -> None:
        assert A2AProxyTool(_FakeHub(), "add")._extra_description() == ""

    def test_generic_mode_is_empty(self) -> None:
        assert A2AProxyTool(_FakeHub())._extra_description() == ""
