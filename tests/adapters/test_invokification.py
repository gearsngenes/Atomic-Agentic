from __future__ import annotations

from typing import Any

import pytest

from atomic_agentic.adapters import batch_to_invokable, to_invokable
from atomic_agentic.exceptions import ToolDefinitionError
from atomic_agentic.tools.a2a_sdk import A2AProxyTool
from atomic_agentic.tools.mcp import MCPProxyTool

from .test_toolify import FakeA2AClientHub, FakeMCPClientHub, a2a_sdk_skill_metadata
from atomic_agentic.invokables.prompt_template import PromptTemplate
from atomic_agentic.tools.base import Tool


class TestToInvokable:
    def test_tool_is_returned_unchanged(self) -> None:
        tool = Tool(function=lambda x: x, name="orig", description="d")

        out = to_invokable(tool, name="other", description="other")

        assert out is tool
        assert out.name == "orig"

    def test_prompt_template_is_returned_unchanged(self) -> None:
        tpl = PromptTemplate(template="Hi {a}.", name="tpl", description="d")

        assert to_invokable(tpl, name="other", description="other") is tpl

    def test_str_becomes_prompt_template(self) -> None:
        out = to_invokable("Hello {who}.", name="greet", description="a greeting")

        assert isinstance(out, PromptTemplate)
        assert out.name == "greet"
        assert [p.name for p in out.parameters] == ["who"]
        assert out.invoke({"who": "there"}).result == "Hello there."

    def test_str_is_used_verbatim(self) -> None:
        out = to_invokable("  padded {x}\n", name="greet", description="d")

        assert out.template == "  padded {x}\n"

    @pytest.mark.parametrize("blank", ["", "   "])
    def test_blank_str_raises_value_error(self, blank: str) -> None:
        with pytest.raises(ValueError):
            to_invokable(blank, name="greet", description="d")

    def test_function_becomes_tool_with_given_name(self) -> None:
        def add(a: int, b: int) -> int:
            return a + b

        out = to_invokable(add, name="adder", description="adds")

        assert isinstance(out, Tool)
        assert out.name == "adder"

    def test_lambda_with_explicit_name(self) -> None:
        out = to_invokable(lambda x: x, name="ident", description="identity")

        assert isinstance(out, Tool)
        assert out.name == "ident"

    @pytest.mark.parametrize("bad", [None, 5, {"a": 1}])
    def test_other_types_raise_type_error_naming_name_and_type(self, bad: Any) -> None:
        with pytest.raises(TypeError, match=r"slot_name.*" + type(bad).__name__):
            to_invokable(bad, name="slot_name", description="d")


def _documented(a: int) -> int:
    """Documented function."""
    return a


class _NoNameCallable:
    def __call__(self, value: int = 0) -> int:
        return value


class TestToInvokableOptionalNames:
    def test_callable_without_name_or_description_infers_both(self) -> None:
        out = to_invokable(_documented)

        assert isinstance(out, Tool)
        assert out.name == "_documented"
        assert out.description == "Documented function."

    def test_callable_with_only_name_infers_description(self) -> None:
        out = to_invokable(_documented, name="renamed")

        assert out.name == "renamed"
        assert out.description == "Documented function."

    def test_invokable_without_name_is_returned_unchanged(self) -> None:
        tpl = PromptTemplate(template="Hi {a}.", name="tpl", description="d")

        assert to_invokable(tpl) is tpl

    @pytest.mark.parametrize(
        "kwargs, missing",
        [
            ({"description": "d"}, "name"),
            ({"name": "n"}, "description"),
            ({}, "name, description"),
        ],
    )
    def test_str_missing_name_or_description_raises(self, kwargs: dict, missing: str) -> None:
        with pytest.raises(TypeError, match=missing):
            to_invokable("Hello {who}.", **kwargs)

    def test_unsupported_type_without_name_raises(self) -> None:
        with pytest.raises(TypeError, match="unsupported component type"):
            to_invokable(42)  # type: ignore[arg-type]


class TestBatchToInvokable:
    @pytest.mark.parametrize("empty", [[], None])
    def test_empty_batch_returns_empty_list(self, empty: Any) -> None:
        assert batch_to_invokable(empty) == []

    def test_mixed_batch_preserves_order_and_expands_hub_in_place(self) -> None:
        tool = Tool(function=lambda x: x, name="orig", description="d")
        tpl = PromptTemplate(template="Hi {a}.", name="tpl", description="d")

        out = batch_to_invokable([tool, _documented, tpl, FakeMCPClientHub()])

        assert out[0] is tool
        assert out[2] is tpl
        assert [t.name for t in out] == ["orig", "_documented", "tpl", "search", "summarize"]
        assert isinstance(out[1], Tool)

    def test_mcp_hub_expands_in_list_tools_order(self) -> None:
        out = batch_to_invokable([FakeMCPClientHub()])

        assert all(isinstance(t, MCPProxyTool) for t in out)
        assert [t.remote_name for t in out] == ["search", "summarize"]

    def test_a2a_hub_expands_skills_then_one_generic_tool(self) -> None:
        hub = FakeA2AClientHub(
            skills={
                "add": a2a_sdk_skill_metadata(remote_name="add"),
                "multiply": a2a_sdk_skill_metadata(remote_name="multiply"),
            }
        )

        out = batch_to_invokable([hub])

        assert all(isinstance(t, A2AProxyTool) for t in out)
        assert [t.skill_id for t in out] == ["add", "multiply", None]

    def test_zero_skill_a2a_hub_yields_one_generic_tool(self) -> None:
        out = batch_to_invokable([FakeA2AClientHub(skills={})])

        assert len(out) == 1
        assert out[0].skill_id is None

    def test_str_item_raises_naming_its_index(self) -> None:
        with pytest.raises(TypeError, match="index 1"):
            batch_to_invokable([_documented, "Hello {who}."])

    def test_unsupported_item_raises_naming_its_index(self) -> None:
        with pytest.raises(TypeError, match=r"unsupported component type.*index 2"):
            batch_to_invokable([_documented, _documented, 42])

    def test_callable_without_name_propagates_tool_definition_error(self) -> None:
        with pytest.raises(ToolDefinitionError):
            batch_to_invokable([_NoNameCallable()])
