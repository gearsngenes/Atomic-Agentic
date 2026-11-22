from __future__ import annotations

from typing import Any

import pytest

from atomic_agentic.adapters import to_invokable
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
