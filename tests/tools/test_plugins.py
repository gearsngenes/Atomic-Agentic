from __future__ import annotations

import math

import pytest

from atomic_agentic.core.Exceptions import ToolInvocationError
from atomic_agentic.tools.Plugins import CONSOLE_TOOLS, MATH_TOOLS, PARSER_TOOLS
from atomic_agentic.tools.base import Tool


ALL_PLUGIN_TOOLS = [*MATH_TOOLS, *CONSOLE_TOOLS, *PARSER_TOOLS]


def tool_by_full_name(full_name: str) -> Tool:
    matches = [tool for tool in ALL_PLUGIN_TOOLS if tool.full_name == full_name]
    assert len(matches) == 1, f"Expected exactly one plugin tool named {full_name!r}"
    return matches[0]


class TestPluginBundles:
    def test_plugin_bundles_are_non_empty_lists(self) -> None:
        assert isinstance(MATH_TOOLS, list)
        assert isinstance(CONSOLE_TOOLS, list)
        assert isinstance(PARSER_TOOLS, list)

        assert MATH_TOOLS
        assert CONSOLE_TOOLS
        assert PARSER_TOOLS

    def test_every_plugin_item_is_a_tool(self) -> None:
        assert all(isinstance(tool, Tool) for tool in ALL_PLUGIN_TOOLS)

    def test_plugin_tool_full_names_are_unique(self) -> None:
        full_names = [tool.full_name for tool in ALL_PLUGIN_TOOLS]

        assert len(full_names) == len(set(full_names))

    def test_plugin_tools_have_basic_metadata(self) -> None:
        for tool in ALL_PLUGIN_TOOLS:
            assert tool.name
            assert tool.namespace
            assert tool.full_name
            assert tool.description
            assert isinstance(tool.parameters, list)
            assert isinstance(tool.return_type, str)


class TestMathPluginSmoke:
    def test_math_add_invokes_correctly(self) -> None:
        tool = tool_by_full_name("Tool.Math.add")

        assert tool.invoke({"a": 2, "b": 3}) == 5

    def test_math_divide_by_zero_returns_inf(self) -> None:
        tool = tool_by_full_name("Tool.Math.divide")

        result = tool.invoke({"a": 2, "b": 0})

        assert math.isinf(result)

    def test_math_sqrt_negative_wraps_value_error(self) -> None:
        tool = tool_by_full_name("Tool.Math.sqrt")

        with pytest.raises(ToolInvocationError, match="invocation failed"):
            tool.invoke({"x": -1})

    def test_math_mean_empty_returns_zero(self) -> None:
        tool = tool_by_full_name("Tool.Math.mean")

        assert tool.invoke({"nums": []}) == 0.0


class TestParserPluginSmoke:
    def test_parser_json_loads_parses_object(self) -> None:
        tool = tool_by_full_name("Tool.Parser.json_loads")

        assert tool.invoke({"s": '{"a": 1}'}) == {"a": 1}

    def test_parser_split_splits_string(self) -> None:
        tool = tool_by_full_name("Tool.Parser.split")

        assert tool.invoke({"s": "a,b,c", "sep": ","}) == ["a", "b", "c"]

    def test_parser_join_joins_strings(self) -> None:
        tool = tool_by_full_name("Tool.Parser.join")

        assert tool.invoke({"lst": ["a", "b", "c"], "sep": "-"}) == "a-b-c"

    def test_parser_extract_json_string_finds_object(self) -> None:
        tool = tool_by_full_name("Tool.Parser.extract_json_string")

        assert tool.invoke({"s": "prefix {\"a\": 1} suffix"}) == '{"a": 1}'

    def test_parser_safe_eval_literal_smoke(self) -> None:
        tool = tool_by_full_name("Tool.Parser.safe_eval")

        assert tool.invoke({"s": "[1, 2, 3]"}) == [1, 2, 3]


class TestConsolePluginSmoke:
    def test_console_print_tool_can_print_value(self, capsys: pytest.CaptureFixture[str]) -> None:
        tool = tool_by_full_name("Tool.Console.print")

        result = tool.invoke({"value": "hello"})

        captured = capsys.readouterr()
        assert result is None
        assert captured.out == "hello\n"

    def test_console_log_tool_invokes_without_error(self) -> None:
        tool = tool_by_full_name("Tool.Console.log")

        assert tool.invoke({"message": "hello", "level": "INFO"}) is None

    def test_console_user_input_exists_but_is_not_invoked(self) -> None:
        tool = tool_by_full_name("Tool.Console.user_input")

        assert tool.name == "user_input"
        assert tool.namespace == "Console"
