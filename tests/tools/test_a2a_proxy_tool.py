"""Unit-level `A2AProxyTool` identity, description and extra-description behavior.

Server-backed behavior lives in `tests/a2a/test_proxy_tool.py`; this module uses
a fake hub so nothing here needs a network.
"""
from __future__ import annotations

from typing import Any, Mapping

from atomic_agentic.a2a.a2a_client_hub import A2AClientHub
from atomic_agentic.models.a2a_sdk import A2AtomicSkillMetadata
from atomic_agentic.models.parameters import ParamSpec
from atomic_agentic.tools.a2a_sdk import A2AProxyTool


class _FakeHub(A2AClientHub):
    """Skips A2AClientHub.__init__: fixed agent card and skills, no connection."""

    def __init__(
        self,
        *,
        skills: dict[str, A2AtomicSkillMetadata] | None = None,
        card_description: str = "Fake agent.",
    ) -> None:
        self._skills = {"add": _skill()} if skills is None else skills
        self._card = type("FakeAgentCard", (), {"name": "fake", "description": card_description})()

    @property
    def agent_card(self) -> Any:
        return self._card

    @property
    def transport_mode(self) -> str:
        return "JSONRPC"

    @property
    def base_url(self) -> str:
        return "http://example.test/a2a-sdk"

    @property
    def persistent(self) -> bool:
        return False

    def get_atomic_skills(self) -> Mapping[str, A2AtomicSkillMetadata]:
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


class TestPositionalOrder:
    def test_positional_order_is_client_hub_skill_id_name_description(self) -> None:
        hub = _FakeHub()

        tool = A2AProxyTool(hub, "add", "local_add", "Local description.")

        assert tool.client_hub is hub
        assert tool.skill_id == "add"
        assert tool.name == "local_add"
        assert tool._description == "Local description."


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

    def test_explicit_description_wins(self) -> None:
        tool = A2AProxyTool(_FakeHub(), "add", description="Explicit.")

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
