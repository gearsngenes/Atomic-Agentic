from __future__ import annotations

import pytest
import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

from conftest import (
    ROLE_TEMPLATE,
    EchoLLMEngine,
    ScriptedLLMEngine,
    BadRepr,
    ScriptedRunState,
    ScriptedToolAgent,
    BadInitializeToolAgent,
    PendingPreparedToolAgent,
    make_agent,
    register_math_tools,
    prepared_slot,
    executed_slot,
    make_state,
    make_llm_result,
    make_llm_record,
    make_tool_result,
    make_planact_agent,
    make_react_agent,
    react_step_json,
    add,
    multiply,
    join_text,
    fail_tool,
    package_tool_result,
)

from atomic_agentic.agents.toolagent import ToolAgent, extract_dependencies, return_tool
from atomic_agentic.agents.planact import PlanActAgent
from atomic_agentic.agents.react import ReActAgent
from atomic_agentic.models.agents.runstates import ToolAgentRunState
from atomic_agentic.models.agents.records import AgentRecord, ToolAgentRecord
from atomic_agentic.models.agents.records import LLMRecord
from atomic_agentic.models.results.agents import ToolAgentResult
from atomic_agentic.models.agents.blackboard_models import BlackboardSlot, ConstantSpec
from atomic_agentic.exceptions import (
    AgentError,
    ToolAgentError,
    ToolInvocationError,
    ToolRegistrationError,
)
from atomic_agentic.models.agents.prompts import PromptConfig
from atomic_agentic.constants.core import NO_VAL
from atomic_agentic.engines.LLMEngines import LLMEngine
from atomic_agentic.models.results import LLMModelData, LLMResult, TokenUsage, ToolResult
from atomic_agentic.tools import Tool
from atomic_agentic.core.Invokable import AtomicInvokable

class TestToolAgentConstruction:
    def test_valid_construction_auto_registers_return_tool(self) -> None:
        agent = make_agent()

        assert agent.has_tool(return_tool.full_name)
        assert agent.get_tool(return_tool.full_name) is return_tool

    def test_tool_instructions_property_returns_template(self) -> None:
        agent = make_agent()
        assert agent.tool_instructions == ROLE_TEMPLATE

    def test_build_context_populates_tools_limit_constants(self) -> None:
        agent = make_agent(tool_calls_limit=3)
        register_math_tools(agent)

        context, _ = agent._build_context({})

        assert ToolAgent.TOOLS_FIELD in context
        assert ToolAgent.LIMIT_FIELD in context
        assert ToolAgent.CONSTANTS_FIELD in context
        assert "3" in context[ToolAgent.LIMIT_FIELD]
        assert agent._tool_prompt_key not in context

    def test_build_context_unlimited_when_no_limit(self) -> None:
        agent = make_agent()
        context, _ = agent._build_context({})
        assert context[ToolAgent.LIMIT_FIELD] == "unlimited"

    def test_custom_prompt_key_stored_and_accessible(self) -> None:
        agent = ScriptedToolAgent(prompt_key="custom_key")
        assert agent._tool_prompt_key == "custom_key"
        assert "custom_key" in agent.system_prompts

    def test_tool_instructions_accepts_prompt_config_directly(self) -> None:
        config = PromptConfig(template=ROLE_TEMPLATE, description="pre-built config")
        agent = ScriptedToolAgent(tool_instructions=config)
        assert agent.tool_instructions == ROLE_TEMPLATE

    def test_update_prompt_rejects_tool_prompt_key(self) -> None:
        agent = make_agent()
        replacement = PromptConfig(template=ROLE_TEMPLATE, description="replacement")
        with pytest.raises(AgentError):
            agent.update_prompt(agent._tool_prompt_key, replacement)

    def test_update_prompt_accepts_other_key(self) -> None:
        agent = make_agent()
        config = PromptConfig(template="hello", description="other")
        agent.update_prompt("other_key", config)
        assert "other_key" in agent.system_prompts

    def test_tool_prompt_missing_tools_placeholder_raises(self) -> None:
        config = PromptConfig(
            template="Limit: {TOOL_CALLS_LIMIT} Constants: {CONSTANTS}",
            description="missing TOOLS",
        )
        with pytest.raises(ToolAgentError, match="TOOLS"):
            ScriptedToolAgent._validate_tool_prompt_template(config)

    def test_tool_prompt_missing_tool_calls_limit_placeholder_raises(self) -> None:
        config = PromptConfig(
            template="Tools: {TOOLS} Constants: {CONSTANTS}",
            description="missing TOOL_CALLS_LIMIT",
        )
        with pytest.raises(ToolAgentError, match="TOOL_CALLS_LIMIT"):
            ScriptedToolAgent._validate_tool_prompt_template(config)

    def test_tool_prompt_missing_constants_placeholder_raises(self) -> None:
        config = PromptConfig(
            template="Tools: {TOOLS} Limit: {TOOL_CALLS_LIMIT}",
            description="missing CONSTANTS",
        )
        with pytest.raises(ToolAgentError, match="CONSTANTS"):
            ScriptedToolAgent._validate_tool_prompt_template(config)

    def test_tool_prompt_extra_simple_placeholder_is_allowed(self) -> None:
        config = PromptConfig(
            template="Tools: {TOOLS} Limit: {TOOL_CALLS_LIMIT} Constants: {CONSTANTS} Extra: {EXTRA}",
            description="extra field",
        )
        ScriptedToolAgent._validate_tool_prompt_template(config)  # must not raise

    def test_constructor_positional_placeholder_raises_tool_agent_error(self) -> None:
        with pytest.raises(ToolAgentError):
            ScriptedToolAgent(
                tool_instructions="Tools: {TOOLS} Limit: {TOOL_CALLS_LIMIT} Constants: {CONSTANTS} {}"
            )

    def test_constructor_field_expression_raises_tool_agent_error(self) -> None:
        with pytest.raises(ToolAgentError):
            ScriptedToolAgent(
                tool_instructions="Tools: {TOOLS.name} Limit: {TOOL_CALLS_LIMIT} Constants: {CONSTANTS}"
            )

    @pytest.mark.parametrize("value", [None, 0, 1, 5])
    def test_tool_calls_limit_accepts_none_and_non_negative_int(
        self,
        value: int | None,
    ) -> None:
        agent = make_agent(tool_calls_limit=value)

        assert agent.tool_calls_limit == value

    @pytest.mark.parametrize("value", [-1, "1", 1.5, True])
    def test_tool_calls_limit_rejects_negative_or_non_int(self, value: Any) -> None:
        with pytest.raises(ToolAgentError, match="tool_calls_limit"):
            make_agent(tool_calls_limit=value)  # type: ignore[arg-type]

    def test_peek_at_cache_is_frozen(self) -> None:
        agent = make_agent(peek_at_cache=True)

        with pytest.raises(AttributeError):
            agent.peek_at_cache = False  # type: ignore[misc]

    def test_peek_at_cache_construction_rejects_non_bool(self) -> None:
        with pytest.raises(ToolAgentError, match="peek_at_cache"):
            make_agent(peek_at_cache=1)  # type: ignore[arg-type]

    def test_blackboard_preview_limit_is_frozen(self) -> None:
        agent = make_agent(blackboard_preview_limit=10)

        with pytest.raises(AttributeError):
            agent.blackboard_preview_limit = 20  # type: ignore[misc]

    def test_blackboard_preview_limit_construction_rejects_zero(self) -> None:
        with pytest.raises(ToolAgentError, match="blackboard_preview_limit"):
            make_agent(blackboard_preview_limit=0)

    def test_blackboard_preview_limit_construction_rejects_negative(self) -> None:
        with pytest.raises(ToolAgentError, match="blackboard_preview_limit"):
            make_agent(blackboard_preview_limit=-1)

    def test_blackboard_preview_limit_construction_rejects_non_int(self) -> None:
        with pytest.raises(ToolAgentError, match="blackboard_preview_limit"):
            make_agent(blackboard_preview_limit="10")  # type: ignore[arg-type]

    def test_preview_limit_removed(self) -> None:
        agent = make_agent()

        with pytest.raises(AttributeError):
            _ = agent.preview_limit  # type: ignore[attr-defined]

        with pytest.raises(TypeError):
            ScriptedToolAgent(preview_limit=10)  # type: ignore[call-arg]

    def test_fail_fast_defaults_to_true(self) -> None:
        agent = make_agent()
        assert agent.fail_fast is True

    def test_fail_fast_false_accepted(self) -> None:
        agent = make_agent(fail_fast=False)
        assert agent.fail_fast is False

    def test_fail_fast_property_is_readable(self) -> None:
        for value in (True, False):
            agent = make_agent(fail_fast=value)
            assert agent.fail_fast is value

    def test_fail_fast_is_not_directly_settable(self) -> None:
        agent = make_agent()
        with pytest.raises(AttributeError):
            agent.fail_fast = False  # type: ignore[misc]

    @pytest.mark.parametrize("value", [1, "True", None, 0])
    def test_fail_fast_rejects_non_bool(self, value: Any) -> None:
        with pytest.raises(ToolAgentError, match="fail_fast"):
            make_agent(fail_fast=value)  # type: ignore[arg-type]

    def test_generation_retries_defaults_to_zero(self) -> None:
        agent = make_agent()
        assert agent.generation_retries == 0

    def test_generation_retries_accepted(self) -> None:
        agent = ScriptedToolAgent(generation_retries=3)
        assert agent.generation_retries == 3

    def test_generation_retries_property_is_not_directly_settable(self) -> None:
        agent = make_agent()
        with pytest.raises(AttributeError):
            agent.generation_retries = 1  # type: ignore[misc]

    @pytest.mark.parametrize("value", [-1, "1", 1.5, True, None])
    def test_generation_retries_rejects_invalid(self, value: Any) -> None:
        with pytest.raises(ToolAgentError, match="generation_retries"):
            ScriptedToolAgent(generation_retries=value)  # type: ignore[arg-type]


class TestToolAgentNamespace:
    def test_namespace_is_required(self) -> None:
        with pytest.raises(TypeError):
            PlanActAgent(
                name="a",
                description="d",
                llm_engine=EchoLLMEngine(),
            )

    def test_plan_act_agent_namespace_explicit(self) -> None:
        agent = PlanActAgent(
            name="a",
            namespace="planner_ns",
            description="d",
            llm_engine=EchoLLMEngine(),
        )
        assert agent.namespace == "planner_ns"

    def test_react_agent_namespace_explicit(self) -> None:
        agent = ReActAgent(
            name="a",
            namespace="react_ns",
            description="d",
            llm_engine=EchoLLMEngine(),
            tool_calls_limit=5,
        )
        assert agent.namespace == "react_ns"


class TestToolAgentPostInvokeRouting:
    def test_scripted_tool_agent_forwards_post_routing_to_base_agent(self) -> None:
        agent = make_agent(post_invoke=package_tool_result)
        agent.set_script(
            [[{"tool": return_tool.full_name, "args": {"val": 5}}]]
        )

        result = agent.invoke({"prompt": "run", "label": "scripted"})

        assert result.result == {"label": "scripted", "result": 5}
        assert agent.post_result_key == "result"

    def test_planact_agent_supports_post_invoke_passthrough(self) -> None:
        agent = make_planact_agent(
            [
                json.dumps(
                    [
                        {
                            "step": 0,
                            "tool": return_tool.full_name,
                            "args": {"val": 5},
                        }
                    ]
                )
            ],
            post_invoke=package_tool_result,
        )

        result = agent.invoke({"prompt": "run plan", "label": "planact"})

        assert result.result == {"label": "planact", "result": 5}

    def test_react_agent_supports_post_invoke_passthrough(self) -> None:
        agent = make_react_agent(
            [
                react_step_json(
                    tool=return_tool.full_name,
                    args={"val": 7},
                    duration=0,
                    description="Return the final literal value for the test task.",
                )
            ],
            tool_calls_limit=1,
            post_invoke=package_tool_result,
        )

        result = agent.invoke({"prompt": "run react", "label": "react"})

        assert result.result == {"label": "react", "result": 7}


class TestToolRegistration:
    def test_register_callable_adds_tool(self) -> None:
        agent = make_agent()

        key = agent.register(add)

        assert key == "Tool.tests.add"
        assert agent.has_tool(key)
        assert agent.get_tool(key).invoke({"x": 1, "y": 2}).result == 3

    def test_register_tool_instance_adds_tool(self) -> None:
        agent = make_agent()
        tool = Tool(
            function=add,
            name="adder",
            namespace="tests",
            description="Add values.",
        )

        key = agent.register(tool)

        assert key == "Tool.tests.adder"
        assert agent.get_tool(key) is tool

    def test_register_duplicate_raises_by_default(self) -> None:
        agent = make_agent()
        agent.register(add)

        with pytest.raises(ToolRegistrationError, match="already registered"):
            agent.register(add)

    def test_register_duplicate_skip_returns_existing_key(self) -> None:
        agent = make_agent()
        first = agent.register(add)
        second = agent.register(add, name_collision_mode="skip")

        assert second == first
        assert agent.get_tool(first).function is add

    def test_register_duplicate_replace_replaces_tool(self) -> None:
        agent = make_agent()
        first = agent.register(add, name="calc")
        second = agent.register(multiply, name="calc", name_collision_mode="replace")

        assert second == first
        assert agent.get_tool(first).invoke({"x": 3, "y": 4}).result == 12

    def test_register_invalid_collision_mode_raises(self) -> None:
        agent = make_agent()

        with pytest.raises(ToolRegistrationError, match="name_collision_mode"):
            agent.register(add, name_collision_mode="bad")

    def test_get_tool_unknown_raises(self) -> None:
        agent = make_agent()

        with pytest.raises(ToolAgentError, match="unknown tool"):
            agent.get_tool("Tool.tests.missing")

    def test_remove_tool_returns_true_then_false(self) -> None:
        agent = make_agent()
        key = agent.register(add)

        assert agent.remove_tool(key) is True
        assert agent.remove_tool(key) is False

    def test_clear_tools_removes_all_tools(self) -> None:
        agent = make_agent()
        register_math_tools(agent)

        agent.clear_tools()

        assert agent.list_tools() == {}

    def test_batch_register_callables(self) -> None:
        agent = make_agent()

        keys = agent.batch_register(tools=[add, multiply])

        assert keys == ["Tool.tests.add", "Tool.tests.multiply"]
        assert agent.has_tool("Tool.tests.add")
        assert agent.has_tool("Tool.tests.multiply")

    def test_batch_register_empty_tools_no_client_raises(self) -> None:
        agent = make_agent()

        with pytest.raises(ValueError, match="tools list is empty"):
            agent.batch_register([])

    def test_actions_context_lists_registered_tools(self) -> None:
        agent = make_agent()
        key = agent.register(add)

        context = agent.actions_context()

        assert key in context

    def test_register_atomic_invokable_stores_directly(self) -> None:
        """AtomicInvokable registers without wrapping under its own full_name."""
        agent = make_agent()
        tool = Tool(function=add, name="adder", namespace="myns", description="Add.")
        key = agent.register(tool)
        assert key == "Tool.myns.adder"
        assert agent.get_tool(key) is tool

    def test_register_atomic_invokable_name_override_raises(self) -> None:
        agent = make_agent()
        tool = Tool(function=add, name="adder", namespace="myns", description="Add.")
        with pytest.raises(ToolRegistrationError, match="name and description overrides"):
            agent.register(tool, name="other")

    def test_register_atomic_invokable_description_override_raises(self) -> None:
        agent = make_agent()
        tool = Tool(function=add, name="adder", namespace="myns", description="Add.")
        with pytest.raises(ToolRegistrationError, match="name and description overrides"):
            agent.register(tool, description="other")

    def test_register_callable_uses_self_name_as_namespace(self) -> None:
        """Callable registration uses agent.name as the tool namespace."""
        agent = make_agent()
        key = agent.register(add)
        assert key == f"Tool.{agent.name}.add"

    def test_register_unsupported_type_raises(self) -> None:
        agent = make_agent()
        with pytest.raises(ToolRegistrationError, match="unsupported component type"):
            agent.register(42)  # type: ignore[arg-type]

    def test_register_non_invokable_non_callable_raises(self) -> None:
        agent = make_agent()
        with pytest.raises(ToolRegistrationError, match="unsupported component type"):
            agent.register("not_callable")  # type: ignore[arg-type]

    def test_list_tools_return_type_is_atomic_invokable_dict(self) -> None:
        """list_tools returns dict[str, AtomicInvokable], not dict[str, Tool]."""
        agent = make_agent()
        agent.register(add)
        result = agent.list_tools()
        assert isinstance(result, dict)
        for v in result.values():
            assert isinstance(v, AtomicInvokable)

    def test_get_tool_returns_atomic_invokable(self) -> None:
        agent = make_agent()
        key = agent.register(add)
        result = agent.get_tool(key)
        assert isinstance(result, AtomicInvokable)

    def test_batch_register_tools_none_client_none_raises(self) -> None:
        agent = make_agent()
        with pytest.raises(ValueError, match="at least one of"):
            agent.batch_register()

    def test_batch_register_tools_empty_no_client_raises(self) -> None:
        agent = make_agent()
        with pytest.raises(ValueError):
            agent.batch_register(tools=[])

    def test_batch_register_remote_names_without_client_raises(self) -> None:
        agent = make_agent()
        with pytest.raises(ValueError, match="remote_names requires a client"):
            agent.batch_register(tools=[add], remote_names=["foo"])

    def test_batch_register_remote_names_not_found_raises(self) -> None:
        """remote_names entries absent from the client's list raise ToolRegistrationError."""
        class _StubClient:
            def list_invokables(self) -> list[str]:
                return ["Tool.tests.foo"]

        agent = make_agent()
        stub = _StubClient()
        with pytest.raises(ToolRegistrationError, match="not found on client"):
            agent.batch_register(client=stub, remote_names=["Tool.tests.foo", "Tool.tests.bar"])

    def test_batch_register_intraset_duplicate_raises(self) -> None:
        """Duplicate full_name in incoming batch always raises regardless of mode."""
        agent = make_agent()
        t1 = Tool(function=add, name="adder", namespace="myns", description="Add.")
        t2 = Tool(function=multiply, name="adder", namespace="myns", description="Dup.")
        with pytest.raises(ToolRegistrationError, match="duplicate full_name"):
            agent.batch_register(tools=[t1, t2])

    def test_batch_register_callable_intraset_duplicate_raises(self) -> None:
        """Duplicate callable full_name in incoming batch raises."""
        agent = make_agent()
        with pytest.raises(ToolRegistrationError, match="duplicate full_name"):
            agent.batch_register(tools=[add, add])

    def test_batch_register_mixed_invokables_and_callables(self) -> None:
        """batch_register handles a mixed list of AtomicInvokable and Callable."""
        agent = make_agent()
        tool = Tool(function=multiply, name="mult", namespace="myns", description="Mult.")
        keys = agent.batch_register(tools=[add, tool])
        assert f"Tool.{agent.name}.add" in keys
        assert "Tool.myns.mult" in keys

    def test_batch_register_skip_mode_excludes_skipped_from_return(self) -> None:
        """Skipped tools under skip mode are excluded from the returned list."""
        agent = make_agent()
        agent.register(add)
        keys = agent.batch_register(tools=[add, multiply], name_collision_mode="skip")
        assert f"Tool.{agent.name}.multiply" in keys
        assert f"Tool.{agent.name}.add" not in keys

    def test_batch_register_replace_mode_overwrites_existing(self) -> None:
        """Replace mode overwrites existing toolbox entries."""
        agent = make_agent()
        agent.register(add)
        keys = agent.batch_register(tools=[add], name_collision_mode="replace")
        assert f"Tool.{agent.name}.add" in keys


class TestConstantRegistration:
    def test_register_constant_adds_normalized_spec(self) -> None:
        agent = make_agent()

        name = agent.register_constant(
            " USER_CONTEXT ",
            {"user": "Ada"},
            "Current user context.",
            inline_limit=12,
        )

        assert name == "USER_CONTEXT"
        assert agent.has_constant("USER_CONTEXT") is True
        spec = agent.get_constant("USER_CONTEXT")
        assert isinstance(spec, ConstantSpec)
        assert spec.name == "USER_CONTEXT"
        assert spec.value == {"user": "Ada"}
        assert spec.description == "Current user context."
        assert spec.inline_limit == 12
        assert spec.type == "dict"

    def test_constants_property_returns_shallow_copy(self) -> None:
        agent = make_agent()
        agent.register_constant("VALUE", 1)

        constants = agent.constants
        constants.clear()

        assert len(agent.constants) == 1
        assert agent.has_constant("VALUE") is True

    def test_register_constant_duplicate_raises(self) -> None:
        agent = make_agent()
        agent.register_constant("VALUE", 1)

        with pytest.raises(ToolAgentError, match="constant already registered"):
            agent.register_constant("VALUE", 2)

    @pytest.mark.parametrize(
        "name",
        ["", " ", "1VALUE", "bad-name", "bad name"],
    )
    def test_register_constant_invalid_name_raises(self, name: str) -> None:
        agent = make_agent()

        with pytest.raises(ToolAgentError, match="invalid constant spec"):
            agent.register_constant(name, 1)

    def test_batch_register_constants_adds_all_specs_atomically(self) -> None:
        agent = make_agent()

        names = agent.batch_register_constants(
            A=(1,),
            B=("two", "Second value."),
            C=((1, 2), "Coordinates.", 8),
        )

        assert names == ["A", "B", "C"]
        assert agent.get_constant("A").value == 1
        assert agent.get_constant("B").description == "Second value."
        assert agent.get_constant("C").value == (1, 2)
        assert agent.get_constant("C").inline_limit == 8

    def test_batch_register_constants_rejects_empty_batch(self) -> None:
        agent = make_agent()

        with pytest.raises(ToolAgentError, match="expects at least one constant"):
            agent.batch_register_constants()

    @pytest.mark.parametrize(
        "payload",
        [1, [1], (1, 2, 3, 4)],
    )
    def test_batch_register_constants_rejects_malformed_payload(self, payload: Any) -> None:
        agent = make_agent()

        with pytest.raises(ToolAgentError, match="constant 'VALUE'"):
            agent.batch_register_constants(VALUE=payload)  # type: ignore[arg-type]

    def test_batch_register_constants_rejects_duplicates_before_mutating(self) -> None:
        agent = make_agent()

        with pytest.raises(ToolAgentError, match="duplicate constant name in batch"):
            agent.batch_register_constants(**{" VALUE ": (1,), "VALUE": (2,)})

        assert agent.constants == []

    def test_batch_register_constants_rejects_existing_constant_before_mutating(self) -> None:
        agent = make_agent()
        agent.register_constant("EXISTING", 1)

        with pytest.raises(ToolAgentError, match="constant already registered"):
            agent.batch_register_constants(NEW=(2,), EXISTING=(3,))

        assert agent.has_constant("NEW") is False
        assert agent.get_constant("EXISTING").value == 1

    def test_remove_and_clear_constants(self) -> None:
        agent = make_agent()
        agent.batch_register_constants(A=(1,), B=(2,))

        assert agent.remove_constant("A") is True
        assert agent.remove_constant("A") is False
        assert agent.has_constant("A") is False
        assert agent.has_constant("B") is True

        agent.clear_constants()

        assert agent.constants == []

    def test_get_constant_unknown_or_invalid_name_raises(self) -> None:
        agent = make_agent()

        with pytest.raises(ToolAgentError, match="constant name"):
            agent.get_constant(" ")

        with pytest.raises(ToolAgentError, match="unknown constant"):
            agent.get_constant("MISSING")

    def test_constants_context_hides_values_and_renders_metadata(self) -> None:
        agent = make_agent()
        agent.register_constant("SECRET", "super-secret-value", "Sensitive value.")
        agent.register_constant("UNLABELED", 3)

        context = agent.constants_context()

        assert "SECRET" in context
        assert "Type: str" in context
        assert "Description: Sensitive value." in context
        assert "UNLABELED" in context
        assert "Description: No description provided." in context
        assert "super-secret-value" not in context

    def test_constants_context_includes_registered_constants(self) -> None:
        agent = make_agent()
        agent.register_constant("THRESHOLD", 0.9, "Decision threshold.")

        context, _ = agent._build_context({})
        constants_block = context[ToolAgent.CONSTANTS_FIELD]

        assert "THRESHOLD" in constants_block
        assert "Type: float" in constants_block
        assert "Decision threshold." in constants_block
        assert "0.9" not in constants_block

    def test_resolve_exact_constant_placeholder_preserves_type(self) -> None:
        agent = make_agent()
        value = {"items": [1, 2]}
        agent.register_constant("PAYLOAD", value)
        state = make_state()

        resolved = agent._resolve_placeholders(
            {"payload": "<<__k.PAYLOAD__>>"},
            state=state,
        )

        assert resolved == {"payload": value}
        assert resolved["payload"] is value

    def test_resolve_inline_constant_placeholder_renders_repr_with_inline_limit(self) -> None:
        agent = make_agent()
        agent.register_constant("LONG_TEXT", "abcdef", inline_limit=5)
        state = make_state()

        resolved = agent._resolve_placeholders(
            {"message": "Value is <<__k.LONG_TEXT__>>."},
            state=state,
        )

        assert resolved == {"message": "Value is 'abcd."}

    def test_resolve_unknown_constant_placeholder_raises(self) -> None:
        agent = make_agent()
        state = make_state()

        with pytest.raises(ToolAgentError, match="unknown constant reference"):
            agent._resolve_placeholders("<<__k.MISSING__>>", state=state)


class TestPlaceholderResolution:
    def test_full_step_placeholder_preserves_type(self) -> None:
        agent = make_agent()
        state = make_state(running=[executed_slot(0, [1, 2, 3])])

        assert agent._resolve_placeholders("<<__s0__>>", state=state) == [1, 2, 3]

    def test_full_cache_placeholder_preserves_type(self) -> None:
        agent = make_agent()
        state = make_state(cache=[executed_slot(0, {"cached": 10})])

        assert agent._resolve_placeholders("<<__c0__>>", state=state) == {"cached": 10}

    def test_inline_step_placeholder_uses_repr(self) -> None:
        agent = make_agent()
        state = make_state(running=[executed_slot(0, ["a", "b"])])

        assert agent._resolve_placeholders("result=<<__s0__>>", state=state) == "result=['a', 'b']"

    def test_inline_cache_placeholder_uses_repr(self) -> None:
        agent = make_agent()
        state = make_state(cache=[executed_slot(0, {"cached": 10})])

        assert agent._resolve_placeholders("cache=<<__c0__>>", state=state) == "cache={'cached': 10}"

    def test_nested_dict_list_tuple_set_resolution(self) -> None:
        agent = make_agent()
        state = make_state(
            cache=[executed_slot(0, "cached")],
            running=[
                executed_slot(0, 5),
                executed_slot(1, ("a", "b")),
            ],
        )

        resolved = agent._resolve_placeholders(
            {
                "a": "<<__s0__>>",
                "b": ["<<__c0__>>", "inline <<__s1__>>"],
                "c": ("<<__s0__>>",),
                "d": {"<<__c0__>>"},
            },
            state=state,
        )

        assert resolved == {
            "a": 5,
            "b": ["cached", "inline ('a', 'b')"],
            "c": (5,),
            "d": {"cached"},
        }

    def test_placeholder_in_dict_key_resolves(self) -> None:
        agent = make_agent()
        state = make_state(running=[executed_slot(0, "dynamic_key")])

        resolved = agent._resolve_placeholders({"<<__s0__>>": "value"}, state=state)

        assert resolved == {"dynamic_key": "value"}

    def test_multiple_placeholders_in_one_string(self) -> None:
        agent = make_agent()
        state = make_state(
            running=[
                executed_slot(0, 1),
                executed_slot(1, 2),
            ]
        )

        assert agent._resolve_placeholders("<<__s0__>> + <<__s1__>>", state=state) == "1 + 2"

    def test_out_of_range_step_placeholder_raises(self) -> None:
        agent = make_agent()
        state = make_state(running=[])

        with pytest.raises(ToolAgentError, match="Step reference 0 out of range"):
            agent._resolve_placeholders("<<__s0__>>", state=state)

    def test_out_of_range_cache_placeholder_raises(self) -> None:
        agent = make_agent()
        state = make_state(cache=[])

        with pytest.raises(ToolAgentError, match="Cache reference 0 out of range"):
            agent._resolve_placeholders("<<__c0__>>", state=state)

    def test_unexecuted_step_placeholder_raises(self) -> None:
        agent = make_agent()
        state = make_state(running=[BlackboardSlot(step=0)])

        with pytest.raises(ToolAgentError, match="Referenced step 0 is not executed"):
            agent._resolve_placeholders("<<__s0__>>", state=state)

    def test_unexecuted_cache_placeholder_raises(self) -> None:
        agent = make_agent()
        state = make_state(cache=[BlackboardSlot(step=0)])

        with pytest.raises(ToolAgentError, match="Referenced cache 0 is not.*executed"):
            agent._resolve_placeholders("<<__c0__>>", state=state)

    def test_non_string_scalar_is_returned_unchanged(self) -> None:
        agent = make_agent()
        state = make_state()

        assert agent._resolve_placeholders(123, state=state) == 123
        assert agent._resolve_placeholders(None, state=state) is None

    def test_extract_dependencies_finds_nested_placeholders(self) -> None:
        obj = {"a": "<<__s0__>>", "b": ["prefix <<__s1__>>"], "<<__s2__>>": "key"}

        assert extract_dependencies(obj, ToolAgent.STEP_REF_PATTERN) == {0, 1, 2}
        assert extract_dependencies(obj, ToolAgent.CACHE_REF_PATTERN) == set()


class TestExecutePreparedBatch:
    def test_executes_single_non_return_tool_and_stores_result(self) -> None:
        agent = make_agent()
        keys = register_math_tools(agent)

        slot = prepared_slot(0, keys["add"], {"x": 2, "y": 3})
        state = make_state(running=[slot], prepared_steps=[0])

        updated = agent._execute_prepared_batch(state)

        assert updated.running_blackboard[0].result.result == 5
        assert updated.running_blackboard[0].status == "executed"
        assert updated.running_blackboard[0].is_executed() is True
        assert updated.executed_steps == {0}
        assert updated.tool_calls_used == 1
        assert updated.prepared_steps == []

    def test_executes_multiple_non_return_tools(self) -> None:
        agent = make_agent()
        keys = register_math_tools(agent)

        state = make_state(
            running=[
                prepared_slot(0, keys["add"], {"x": 2, "y": 3}),
                prepared_slot(1, keys["multiply"], {"x": 4, "y": 5}),
            ],
            prepared_steps=[0, 1],
        )

        updated = agent._execute_prepared_batch(state)

        assert updated.running_blackboard[0].result.result == 5
        assert updated.running_blackboard[0].status == "executed"
        assert updated.running_blackboard[1].result.result == 20
        assert updated.running_blackboard[1].status == "executed"
        assert updated.tool_calls_used == 2

    def test_executes_return_tool_sets_done_and_return_value(self) -> None:
        agent = make_agent()

        state = make_state(
            running=[prepared_slot(0, return_tool.full_name, {"val": 123})],
            prepared_steps=[0],
        )

        updated = agent._execute_prepared_batch(state)

        assert updated.is_done is True
        assert updated.return_value == 123
        assert updated.running_blackboard[0].status == "executed"
        assert updated.tool_calls_used == 0

    def test_return_tool_does_not_increment_tool_calls_used(self) -> None:
        agent = make_agent()

        state = make_state(
            running=[prepared_slot(0, return_tool.full_name, {"val": "done"})],
            prepared_steps=[0],
            tool_calls_used=3,
        )

        updated = agent._execute_prepared_batch(state)

        assert updated.tool_calls_used == 3

    def test_empty_prepared_steps_raises(self) -> None:
        agent = make_agent()
        state = make_state(running=[])

        with pytest.raises(ToolAgentError, match="no prepared steps"):
            agent._execute_prepared_batch(state)

    def test_duplicate_prepared_steps_raises(self) -> None:
        agent = make_agent()
        keys = register_math_tools(agent)
        state = make_state(
            running=[prepared_slot(0, keys["add"], {"x": 1, "y": 2})],
            prepared_steps=[0, 0],
        )

        with pytest.raises(ToolAgentError, match="duplicates"):
            agent._execute_prepared_batch(state)

    def test_out_of_range_prepared_step_raises(self) -> None:
        agent = make_agent()
        state = make_state(running=[], prepared_steps=[0])

        with pytest.raises(ToolAgentError, match="out of range"):
            agent._execute_prepared_batch(state)

    def test_step_mismatch_raises(self) -> None:
        agent = make_agent()
        keys = register_math_tools(agent)
        slot = prepared_slot(99, keys["add"], {"x": 1, "y": 2})
        state = make_state(running=[slot], prepared_steps=[0])

        with pytest.raises(ToolAgentError, match="step mismatch"):
            agent._execute_prepared_batch(state)

    def test_already_executed_step_raises(self) -> None:
        agent = make_agent()
        state = make_state(running=[executed_slot(0, 3)], prepared_steps=[0])

        with pytest.raises(ToolAgentError, match="already executed"):
            agent._execute_prepared_batch(state)

    def test_unprepared_slot_raises(self) -> None:
        agent = make_agent()
        state = make_state(running=[BlackboardSlot(step=0)], prepared_steps=[0])

        with pytest.raises(ToolAgentError, match="not prepared"):
            agent._execute_prepared_batch(state)

    def test_invalid_tool_name_raises(self) -> None:
        agent = make_agent()
        slot = prepared_slot(0, "", {})
        state = make_state(running=[slot], prepared_steps=[0])

        with pytest.raises(ToolAgentError, match="invalid tool name"):
            agent._execute_prepared_batch(state)

    def test_unknown_tool_raises(self) -> None:
        agent = make_agent()
        slot = prepared_slot(0, "Tool.tests.missing", {})
        state = make_state(running=[slot], prepared_steps=[0])

        with pytest.raises(ToolAgentError, match="unknown tool"):
            agent._execute_prepared_batch(state)

    def test_multiple_return_tools_in_same_batch_raises(self) -> None:
        agent = make_agent()
        state = make_state(
            running=[
                prepared_slot(0, return_tool.full_name, {"val": 1}),
                prepared_slot(1, return_tool.full_name, {"val": 2}),
            ],
            prepared_steps=[0, 1],
        )

        with pytest.raises(ToolAgentError, match="multiple return"):
            agent._execute_prepared_batch(state)

    def test_tool_calls_limit_exceeded_raises(self) -> None:
        agent = make_agent(tool_calls_limit=0)
        keys = register_math_tools(agent)
        state = make_state(
            running=[prepared_slot(0, keys["add"], {"x": 1, "y": 2})],
            prepared_steps=[0],
        )

        with pytest.raises(ToolAgentError, match="tool_calls_limit exceeded"):
            agent._execute_prepared_batch(state)

    def test_tool_failure_records_error_and_raises(self) -> None:
        agent = make_agent()
        keys = register_math_tools(agent)
        slot = prepared_slot(0, keys["fail_tool"], {})
        state = make_state(running=[slot], prepared_steps=[0])

        with pytest.raises((ToolInvocationError, ToolAgentError)):
            agent._execute_prepared_batch(state)

        assert state.running_blackboard[0].error is not NO_VAL
        assert state.running_blackboard[0].status == "failed"
        assert state.running_blackboard[0].is_failed() is True

    def test_prepared_steps_cleared_after_success(self) -> None:
        agent = make_agent()
        keys = register_math_tools(agent)
        state = make_state(
            running=[prepared_slot(0, keys["add"], {"x": 1, "y": 2})],
            prepared_steps=[0],
        )

        updated = agent._execute_prepared_batch(state)

        assert updated.prepared_steps == []

    def test_fail_fast_false_marks_all_failures_without_raising(self) -> None:
        agent = make_agent(fail_fast=False)
        keys = register_math_tools(agent)
        slot = prepared_slot(0, keys["fail_tool"], {})
        state = make_state(running=[slot], prepared_steps=[0])

        agent._execute_prepared_batch(state)  # must not raise

        assert state.running_blackboard[0].status == "failed"
        assert state.running_blackboard[0].error is not NO_VAL
        assert state.running_blackboard[0].is_failed() is True

    def test_fail_fast_false_mixed_batch_executes_successes_marks_failures(self) -> None:
        agent = make_agent(fail_fast=False)
        keys = register_math_tools(agent)
        state = make_state(
            running=[
                prepared_slot(0, keys["add"], {"x": 1, "y": 2}),
                prepared_slot(1, keys["fail_tool"], {}),
            ],
            prepared_steps=[0, 1],
        )

        agent._execute_prepared_batch(state)  # must not raise

        assert state.running_blackboard[0].status == "executed"
        assert state.running_blackboard[0].result.result == 3
        assert state.running_blackboard[1].status == "failed"
        assert state.running_blackboard[1].error is not NO_VAL

    def test_fail_fast_false_executed_steps_excludes_failed_slots(self) -> None:
        agent = make_agent(fail_fast=False)
        keys = register_math_tools(agent)
        state = make_state(
            running=[
                prepared_slot(0, keys["add"], {"x": 1, "y": 2}),
                prepared_slot(1, keys["fail_tool"], {}),
            ],
            prepared_steps=[0, 1],
        )

        updated = agent._execute_prepared_batch(state)

        assert 0 in updated.executed_steps
        assert 1 not in updated.executed_steps

    def test_fail_fast_false_prepared_steps_cleared_after_partial_failure(self) -> None:
        agent = make_agent(fail_fast=False)
        keys = register_math_tools(agent)
        state = make_state(
            running=[prepared_slot(0, keys["fail_tool"], {})],
            prepared_steps=[0],
        )

        updated = agent._execute_prepared_batch(state)

        assert updated.prepared_steps == []


class TestAsyncExecutePreparedBatch:
    """Async analog of TestExecutePreparedBatch covering _async_execute_prepared_batch."""

    def test_executes_single_non_return_tool_and_stores_result(self) -> None:
        agent = make_agent()
        keys = register_math_tools(agent)

        slot = prepared_slot(0, keys["add"], {"x": 2, "y": 3})
        state = make_state(running=[slot], prepared_steps=[0])

        updated = asyncio.run(agent._async_execute_prepared_batch(state))

        assert updated.running_blackboard[0].result.result == 5
        assert updated.running_blackboard[0].status == "executed"
        assert updated.executed_steps == {0}
        assert updated.tool_calls_used == 1
        assert updated.prepared_steps == []

    def test_executes_return_tool_sets_done_and_return_value(self) -> None:
        agent = make_agent()

        state = make_state(
            running=[prepared_slot(0, return_tool.full_name, {"val": 123})],
            prepared_steps=[0],
        )

        updated = asyncio.run(agent._async_execute_prepared_batch(state))

        assert updated.is_done is True
        assert updated.return_value == 123
        assert updated.tool_calls_used == 0

    def test_empty_prepared_steps_raises(self) -> None:
        agent = make_agent()
        state = make_state(running=[])

        with pytest.raises(ToolAgentError, match="no prepared steps"):
            asyncio.run(agent._async_execute_prepared_batch(state))

    def test_duplicate_prepared_steps_raises(self) -> None:
        agent = make_agent()
        keys = register_math_tools(agent)
        state = make_state(
            running=[prepared_slot(0, keys["add"], {"x": 1, "y": 2})],
            prepared_steps=[0, 0],
        )

        with pytest.raises(ToolAgentError, match="duplicates"):
            asyncio.run(agent._async_execute_prepared_batch(state))

    def test_out_of_range_prepared_step_raises(self) -> None:
        agent = make_agent()
        state = make_state(running=[], prepared_steps=[0])

        with pytest.raises(ToolAgentError, match="out of range"):
            asyncio.run(agent._async_execute_prepared_batch(state))

    def test_step_mismatch_raises(self) -> None:
        agent = make_agent()
        keys = register_math_tools(agent)
        slot = prepared_slot(99, keys["add"], {"x": 1, "y": 2})
        state = make_state(running=[slot], prepared_steps=[0])

        with pytest.raises(ToolAgentError, match="step mismatch"):
            asyncio.run(agent._async_execute_prepared_batch(state))

    def test_already_executed_step_raises(self) -> None:
        agent = make_agent()
        state = make_state(running=[executed_slot(0, 3)], prepared_steps=[0])

        with pytest.raises(ToolAgentError, match="already executed"):
            asyncio.run(agent._async_execute_prepared_batch(state))

    def test_unprepared_slot_raises(self) -> None:
        agent = make_agent()
        state = make_state(running=[BlackboardSlot(step=0)], prepared_steps=[0])

        with pytest.raises(ToolAgentError, match="not prepared"):
            asyncio.run(agent._async_execute_prepared_batch(state))

    def test_invalid_tool_name_raises(self) -> None:
        agent = make_agent()
        slot = prepared_slot(0, "", {})
        state = make_state(running=[slot], prepared_steps=[0])

        with pytest.raises(ToolAgentError, match="invalid tool name"):
            asyncio.run(agent._async_execute_prepared_batch(state))

    def test_multiple_return_tools_in_same_batch_raises(self) -> None:
        agent = make_agent()
        state = make_state(
            running=[
                prepared_slot(0, return_tool.full_name, {"val": 1}),
                prepared_slot(1, return_tool.full_name, {"val": 2}),
            ],
            prepared_steps=[0, 1],
        )

        with pytest.raises(ToolAgentError, match="multiple return"):
            asyncio.run(agent._async_execute_prepared_batch(state))

    def test_tool_calls_limit_exceeded_raises(self) -> None:
        agent = make_agent(tool_calls_limit=0)
        keys = register_math_tools(agent)
        state = make_state(
            running=[prepared_slot(0, keys["add"], {"x": 1, "y": 2})],
            prepared_steps=[0],
        )

        with pytest.raises(ToolAgentError, match="tool_calls_limit exceeded"):
            asyncio.run(agent._async_execute_prepared_batch(state))

    def test_non_invocation_exception_is_wrapped_as_tool_agent_error(self) -> None:
        agent = make_agent()
        keys = register_math_tools(agent)
        # Missing required "y" raises a raw TypeError out of Tool.async_invoke,
        # before it can be wrapped as a ToolInvocationError.
        slot = prepared_slot(0, keys["add"], {"x": 1})
        state = make_state(running=[slot], prepared_steps=[0])

        with pytest.raises(ToolAgentError, match="tool call failed at step 0"):
            asyncio.run(agent._async_execute_prepared_batch(state))

        assert isinstance(state.running_blackboard[0].error, ToolAgentError)
        assert state.running_blackboard[0].status == "failed"

    def test_fail_fast_false_marks_all_failures_without_raising(self) -> None:
        agent = make_agent(fail_fast=False)
        keys = register_math_tools(agent)
        slot = prepared_slot(0, keys["fail_tool"], {})
        state = make_state(running=[slot], prepared_steps=[0])

        asyncio.run(agent._async_execute_prepared_batch(state))  # must not raise

        assert state.running_blackboard[0].status == "failed"
        assert state.running_blackboard[0].error is not NO_VAL
        assert state.running_blackboard[0].is_failed() is True

    def test_fail_fast_false_mixed_batch_executes_successes_marks_failures(self) -> None:
        agent = make_agent(fail_fast=False)
        keys = register_math_tools(agent)
        state = make_state(
            running=[
                prepared_slot(0, keys["add"], {"x": 1, "y": 2}),
                prepared_slot(1, keys["fail_tool"], {}),
            ],
            prepared_steps=[0, 1],
        )

        asyncio.run(agent._async_execute_prepared_batch(state))  # must not raise

        assert state.running_blackboard[0].status == "executed"
        assert state.running_blackboard[0].result.result == 3
        assert state.running_blackboard[1].status == "failed"
        assert state.running_blackboard[1].error is not NO_VAL

    def test_fail_fast_false_executed_steps_excludes_failed_slots(self) -> None:
        agent = make_agent(fail_fast=False)
        keys = register_math_tools(agent)
        state = make_state(
            running=[
                prepared_slot(0, keys["add"], {"x": 1, "y": 2}),
                prepared_slot(1, keys["fail_tool"], {}),
            ],
            prepared_steps=[0, 1],
        )

        updated = asyncio.run(agent._async_execute_prepared_batch(state))

        assert 0 in updated.executed_steps
        assert 1 not in updated.executed_steps


class TestScriptedInvokeLoop:
    def test_scripted_invoke_runs_tools_placeholders_and_return(self) -> None:
        agent = make_agent()
        keys = register_math_tools(agent)
        agent.set_script(
            [
                [{"tool": keys["add"], "args": {"x": 2, "y": 3}}],
                [{"tool": keys["multiply"], "args": {"x": "<<__s0__>>", "y": 10}}],
                [{"tool": return_tool.full_name, "args": {"val": "<<__s1__>>"}}],
            ]
        )

        result = agent.invoke({"prompt": "run"})

        assert result.result == 50

    def test_context_disabled_does_not_populate_cache_for_llm(self) -> None:
        agent = make_agent(context_enabled=False)
        keys = register_math_tools(agent)
        agent.set_script(
            [
                [{"tool": keys["add"], "args": {"x": 2, "y": 3}}],
                [{"tool": return_tool.full_name, "args": {"val": "<<__s0__>>"}}],
            ]
        )

        assert agent.invoke({"prompt": "run"}).result == 5
        # update_blackboard always runs; context_enabled only controls cache_blackboard for the LLM.
        assert len(agent.blackboard) == 2
        assert len(agent.records) == 1

    def test_context_enabled_stores_tool_agent_turn_with_blackboard_span(self) -> None:
        agent = make_agent(context_enabled=True)
        keys = register_math_tools(agent)
        agent.set_script(
            [
                [{"tool": keys["add"], "args": {"x": 2, "y": 3}}],
                [{"tool": return_tool.full_name, "args": {"val": "<<__s0__>>"}}],
            ]
        )

        assert agent.invoke({"prompt": "run"}).result == 5

        assert len(agent.records) == 1
        turn = agent.records[0]
        assert isinstance(turn, ToolAgentRecord)
        assert turn.user_prompt == "run"
        assert turn.generated_response == 5
        assert turn.final_result.result == 5
        assert turn.blackboard_start == 0
        assert turn.blackboard_end == len(agent.blackboard)

    def test_context_enabled_persists_executed_blackboard(self) -> None:
        agent = make_agent(context_enabled=True)
        keys = register_math_tools(agent)
        agent.set_script(
            [
                [{"tool": keys["add"], "args": {"x": 2, "y": 3}}],
                [{"tool": return_tool.full_name, "args": {"val": "<<__s0__>>"}}],
            ]
        )

        assert agent.invoke({"prompt": "run"}).result == 5

        board = agent.blackboard
        assert len(board) == 2
        assert board[0].tool == keys["add"]
        assert board[0].result.result == 5
        assert board[0].status == "executed"
        assert board[1].tool == return_tool.full_name
        assert board[1].result.result == 5
        assert board[1].status == "executed"

    def test_context_enabled_rewrites_step_placeholders_to_cache_placeholders(self) -> None:
        agent = make_agent(context_enabled=True)
        keys = register_math_tools(agent)
        agent.set_script(
            [
                [{"tool": keys["add"], "args": {"x": 2, "y": 3}}],
                [{"tool": keys["multiply"], "args": {"x": "<<__s0__>>", "y": 10}}],
                [{"tool": return_tool.full_name, "args": {"val": "<<__s1__>>"}}],
            ]
        )

        assert agent.invoke({"prompt": "run"}).result == 50

        board = agent.blackboard
        assert board[1].args == {"x": "<<__c0__>>", "y": 10}
        assert board[2].args == {"val": "<<__c1__>>"}

    def test_clear_memory_clears_agent_history_and_blackboard(self) -> None:
        agent = make_agent(context_enabled=True)
        keys = register_math_tools(agent)
        agent.set_script(
            [
                [{"tool": keys["add"], "args": {"x": 2, "y": 3}}],
                [{"tool": return_tool.full_name, "args": {"val": "<<__s0__>>"}}],
            ]
        )

        agent.invoke({"prompt": "run"})

        assert agent.blackboard
        assert agent.records

        agent.clear_memory()

        assert agent.blackboard == []
        assert agent.records == []

    def test_prepare_empty_batch_raises(self) -> None:
        agent = make_agent()
        agent.set_script([[]])

        with pytest.raises(ToolAgentError, match="empty batch"):
            agent.invoke({"prompt": "run"})

    def test_pending_prepared_steps_before_prepare_raises(self) -> None:
        agent = PendingPreparedToolAgent(
            script=[[{"tool": return_tool.full_name, "args": {"val": 1}}]]
        )

        with pytest.raises(ToolAgentError, match="prepared_steps is non-empty"):
            agent.invoke({"prompt": "run"})

    def test_cached_placeholder_can_be_used_on_later_invoke_when_context_enabled(self) -> None:
        agent = make_agent(context_enabled=True)
        keys = register_math_tools(agent)

        agent.set_script(
            [
                [{"tool": keys["add"], "args": {"x": 2, "y": 3}}],
                [{"tool": return_tool.full_name, "args": {"val": "<<__s0__>>"}}],
            ]
        )
        assert agent.invoke({"prompt": "first"}).result == 5

        agent.set_script(
            [
                [{"tool": return_tool.full_name, "args": {"val": "<<__c0__>>"}}],
            ]
        )
        assert agent.invoke({"prompt": "second"}).result == 5


class TestBlackboardPersistenceAndDisplay:
    def test_blackboard_property_returns_copied_slots(self) -> None:
        agent = make_agent(context_enabled=True)
        keys = register_math_tools(agent)
        agent.set_script(
            [
                [{"tool": keys["add"], "args": {"x": 1, "y": 2}}],
                [{"tool": return_tool.full_name, "args": {"val": "<<__s0__>>"}}],
            ]
        )
        agent.invoke({"prompt": "run"})

        snapshot = agent.blackboard
        snapshot[0].result = make_tool_result(999)

        assert agent.blackboard[0].result.result == 3

    def test_blackboard_serialized_without_peek_hides_results_and_resolved_args(self) -> None:
        agent = make_agent(context_enabled=True)
        keys = register_math_tools(agent)
        agent.set_script(
            [
                [{"tool": keys["add"], "args": {"x": 1, "y": 2}}],
                [{"tool": return_tool.full_name, "args": {"val": "<<__s0__>>"}}],
            ]
        )
        agent.invoke({"prompt": "run"})

        serialized = agent.blackboard_serialized(peek=False)

        assert isinstance(serialized, list)
        assert serialized[0]["tool"] == keys["add"]
        assert serialized[0]["status"] == "executed"
        assert "result" not in serialized[0]
        assert "resolved_args" not in serialized[0]
        assert "run_id" in serialized[0]
        assert isinstance(serialized[0]["run_id"], str)

    def test_blackboard_serialized_with_peek_includes_preview_and_resolved_args(self) -> None:
        agent = make_agent(context_enabled=True)
        keys = register_math_tools(agent)
        agent.set_script(
            [
                [{"tool": keys["add"], "args": {"x": 1, "y": 2}}],
                [{"tool": return_tool.full_name, "args": {"val": "<<__s0__>>"}}],
            ]
        )
        agent.invoke({"prompt": "run"})

        serialized = agent.blackboard_serialized(peek=True)

        assert isinstance(serialized, list)
        # result field is now a preview string (repr of the caller-facing value)
        assert serialized[0]["result"] == repr(3)
        assert serialized[0]["resolved_args"] == {"x": 1, "y": 2}
        assert serialized[0]["status"] == "executed"
        assert "run_id" in serialized[0]
        assert isinstance(serialized[0]["run_id"], str)

    def test_blackboard_serialized_peek_applies_preview_limit(self) -> None:
        agent = make_agent(context_enabled=True, blackboard_preview_limit=5)
        keys = register_math_tools(agent)
        agent.set_script(
            [
                [{"tool": keys["join_text"], "args": {"prefix": "long", "value": "abcdefghij"}}],
                [{"tool": return_tool.full_name, "args": {"val": "<<__s0__>>"}}],
            ]
        )
        agent.invoke({"prompt": "run"})

        peek_serialized = agent.blackboard_serialized(peek=True)
        hidden_serialized = agent.blackboard_serialized(peek=False)

        # result preview is sliced to blackboard_preview_limit chars then "..." appended
        assert peek_serialized[0]["result"].endswith("...")
        assert len(peek_serialized[0]["result"]) == 5 + len("...")
        # result is absent in the non-peek view
        assert "result" not in hidden_serialized[0]
        # run_id is present in both views for executed slots
        assert "run_id" in peek_serialized[0]
        assert "run_id" in hidden_serialized[0]

    def test_blackboard_serialized_unexecuted_slot_has_no_run_id(self) -> None:
        agent = make_agent(context_enabled=True)
        keys = register_math_tools(agent)
        agent.set_script(
            [
                [{"tool": keys["add"], "args": {"x": 1, "y": 2}}],
                [{"tool": return_tool.full_name, "args": {"val": "<<__s0__>>"}}],
            ]
        )
        # Script is set but agent is NOT invoked — slots remain in planned/empty state.
        serialized = agent.blackboard_serialized(peek=False)
        for slot_dict in serialized:
            assert "run_id" not in slot_dict

    def test_to_dict_includes_tool_agent_diagnostics(self) -> None:
        agent = make_agent(context_enabled=True)
        keys = register_math_tools(agent)
        agent.set_script(
            [
                [{"tool": keys["add"], "args": {"x": 1, "y": 2}}],
                [{"tool": return_tool.full_name, "args": {"val": "<<__s0__>>"}}],
            ]
        )
        agent.invoke({"prompt": "run"})

        snapshot = agent.to_dict()

        assert snapshot["tool_calls_limit"] == agent.tool_calls_limit
        assert snapshot["peek_at_cache"] == agent.peek_at_cache
        assert snapshot["blackboard_preview_limit"] == agent.blackboard_preview_limit
        assert set(snapshot["tools"]) == set(agent._toolbox)
        assert snapshot["blackboard"] == agent.blackboard_serialized(peek=False)

    def test_rendered_history_with_peek_at_cache_includes_cached_step_results(self) -> None:
        agent = make_agent(context_enabled=True, peek_at_cache=True, blackboard_preview_limit=10)
        keys = register_math_tools(agent)
        agent.set_script(
            [
                [
                    {
                        "tool": keys["join_text"],
                        "args": {
                            "prefix": "long",
                            "value": "abcdefghijklmnopqrstuvwxyz",
                        },
                    }
                ],
                [{"tool": return_tool.full_name, "args": {"val": "<<__s0__>>"}}],
            ]
        )

        result = agent.invoke({"prompt": "run"})

        assert result.result == "long:abcdefghijklmnopqrstuvwxyz"
        content = agent.render_turn(agent.records[0])[1]["content"]
        assert "CACHED STEPS" in content
        assert "result" in content
        assert "long:abcd" in content
        assert "long:abcdefghijklmnopqrstuvwxyz" in content

    def test_rendered_history_without_peek_at_cache_hides_cached_step_results(self) -> None:
        agent = make_agent(context_enabled=True, peek_at_cache=False)
        keys = register_math_tools(agent)
        agent.set_script(
            [
                [{"tool": keys["add"], "args": {"x": 1, "y": 2}}],
                [{"tool": return_tool.full_name, "args": {"val": "<<__s0__>>"}}],
            ]
        )

        assert agent.invoke({"prompt": "run"}).result == 3

        content = agent.render_turn(agent.records[0])[1]["content"]
        assert "CACHED STEPS" in content
        assert "'args'" in content
        assert "'result'" not in content

    def test_rendered_history_blackboard_preview_limit_truncates_results_only(self) -> None:
        agent = make_agent(
            context_enabled=True,
            peek_at_cache=True,
            blackboard_preview_limit=10,
        )
        keys = register_math_tools(agent)
        agent.set_script(
            [
                [
                    {
                        "tool": keys["join_text"],
                        "args": {
                            "prefix": "long",
                            "value": "abcdefghijklmnopqrstuvwxyz",
                        },
                    }
                ],
                [{"tool": return_tool.full_name, "args": {"val": "<<__s0__>>"}}],
            ]
        )

        assert agent.invoke({"prompt": "run"}).result == "long:abcdefghijklmnopqrstuvwxyz"

        content = agent.render_turn(agent.records[0])[1]["content"]
        assert "abcdefghijklmnopqrstuvwxyz" in content
        assert "'long:abcd..." in content

    def test_response_preview_limit_truncates_response_not_cached_args(self) -> None:
        agent = make_agent(
            context_enabled=True,
            peek_at_cache=True,
            response_preview_limit=10,
            blackboard_preview_limit=None,
        )
        keys = register_math_tools(agent)
        agent.set_script(
            [
                [
                    {
                        "tool": keys["join_text"],
                        "args": {
                            "prefix": "long",
                            "value": "abcdefghijklmnopqrstuvwxyz",
                        },
                    }
                ],
                [{"tool": return_tool.full_name, "args": {"val": "<<__s0__>>"}}],
            ]
        )

        assert agent.invoke({"prompt": "run"}).result == "long:abcdefghijklmnopqrstuvwxyz"

        content = agent.render_turn(agent.records[0])[1]["content"]
        response_section = content.split("CACHED STEPS", maxsplit=1)[0]
        cached_section = content.split("CACHED STEPS", maxsplit=1)[1]
        assert "RESPONSE:\nlong:abcde..." in response_section
        assert "abcdefghijklmnopqrstuvwxyz" in cached_section
        assert "'long:abcdefghijklmnopqrstuvwxyz'" in cached_section

    def test_context_disabled_invoke_stores_blackboard(self) -> None:
        agent = make_agent(context_enabled=False)
        keys = register_math_tools(agent)
        agent.set_script(
            [
                [{"tool": keys["add"], "args": {"x": 1, "y": 2}}],
                [{"tool": return_tool.full_name, "args": {"val": "<<__s0__>>"}}],
            ]
        )

        assert agent.invoke({"prompt": "run"}).result == 3

        board = agent.blackboard
        assert len(board) == 2
        assert board[0].tool == keys["add"]
        assert board[0].status == "executed"
        assert board[1].tool == return_tool.full_name
        assert board[1].status == "executed"


class TestToolAgentRecordRendering:
    def test_render_turn_returns_single_user_assistant_pair(self) -> None:
        agent = make_agent(context_enabled=True, peek_at_cache=True)
        keys = register_math_tools(agent)
        agent.set_script(
            [
                [{"tool": keys["add"], "args": {"x": 1, "y": 2}}],
                [{"tool": return_tool.full_name, "args": {"val": "<<__s0__>>"}}],
            ]
        )

        assert agent.invoke({"prompt": "run"}).result == 3

        rendered = agent.render_turn(agent.records[0])

        assert len(rendered) == 2
        assert [message["role"] for message in rendered] == ["user", "assistant"]
        assert rendered[0]["content"] == "run"
        assert rendered[1]["content"].startswith("RESPONSE:")
        assert "CACHED STEPS [0, 1] PRODUCED" in rendered[1]["content"]
        assert "'run_id'" in rendered[1]["content"]

    def test_render_turn_uses_blackboard_span_only_for_that_turn(self) -> None:
        agent = make_agent(context_enabled=True, peek_at_cache=True)
        keys = register_math_tools(agent)

        agent.set_script(
            [
                [{"tool": keys["add"], "args": {"x": 1, "y": 2}}],
                [{"tool": return_tool.full_name, "args": {"val": "<<__s0__>>"}}],
            ]
        )
        assert agent.invoke({"prompt": "first"}).result == 3

        agent.set_script(
            [
                [{"tool": keys["multiply"], "args": {"x": 4, "y": 5}}],
                [{"tool": return_tool.full_name, "args": {"val": "<<__s0__>>"}}],
            ]
        )
        assert agent.invoke({"prompt": "second"}).result == 20

        first_rendered = agent.render_turn(agent.records[0])[1]["content"]
        second_rendered = agent.render_turn(agent.records[1])[1]["content"]

        assert "CACHED STEPS [0, 1] PRODUCED" in first_rendered
        assert "CACHED STEPS [2, 3] PRODUCED" in second_rendered
        assert "Tool.tests.add" in first_rendered
        assert "Tool.tests.multiply" not in first_rendered
        assert "Tool.tests.multiply" in second_rendered
        assert "Tool.tests.add" not in second_rendered

    def test_render_turn_raises_for_non_tool_agent_turn(self) -> None:
        agent = make_agent()
        turn = AgentRecord(
            user_prompt="run",
            generated_response="raw",
        )

        with pytest.raises(ToolAgentError, match="ToolAgentRecord"):
            agent.render_turn(turn)


class TestRenderTurnWithFailedSlots:
    """
    Tests for render_turn behaviour when FAILED slots are present in the
    blackboard span (introduced by fail_fast=False in a16 Pass 1).
    """

    def _make_agent_with_failed_slot(
        self,
        *,
        peek_at_cache: bool = False,
        blackboard_preview_limit: int | None = None,
    ) -> ScriptedToolAgent:
        """
        Agent with step 0 = fail_tool (FAILED), step 1 = add (EXECUTED),
        return uses step 1's result.  context_enabled=True so the span is
        persisted and render_turn has something to render.
        """
        agent = make_agent(
            context_enabled=True,
            fail_fast=False,
            peek_at_cache=peek_at_cache,
            blackboard_preview_limit=blackboard_preview_limit,
        )
        keys = register_math_tools(agent)
        agent.set_script([
            [
                {"tool": "Tool.tests.fail_tool", "args": {}},
                {"tool": keys["add"], "args": {"x": 3, "y": 4}},
            ],
            [{"tool": return_tool.full_name, "args": {"val": "<<__s1__>>"}}],
        ])
        result = agent.invoke({"prompt": "run"})
        assert result.result == 7
        return agent

    def test_render_turn_peek_at_cache_with_failed_slot_does_not_crash(self) -> None:
        """B1 fix: peek_at_cache=True must not crash when a FAILED slot is in the span."""
        agent = self._make_agent_with_failed_slot(peek_at_cache=True)
        # Must not raise — FAILED slots' slot.result = NO_VAL must never be
        # passed to _preview_blackboard_result.
        rendered = agent.render_turn(agent.records[0])
        assert rendered is not None
        assert len(rendered) == 2

    def test_render_turn_mixed_span_shows_cached_and_failed_sections(self) -> None:
        """Mixed span: both CACHED STEPS and FAILED STEPS sections appear."""
        agent = self._make_agent_with_failed_slot()
        content = agent.render_turn(agent.records[0])[1]["content"]
        assert "CACHED STEPS" in content
        assert "FAILED STEPS" in content
        assert "RESPONSE:" in content
        # RESPONSE must come first.
        assert content.index("RESPONSE:") < content.index("CACHED STEPS")
        assert content.index("CACHED STEPS") < content.index("FAILED STEPS")

    def test_render_turn_failed_entries_omit_args_include_tool_and_error(self) -> None:
        """FAILED STEPS entries contain tool + error but NOT args."""
        agent = self._make_agent_with_failed_slot()
        content = agent.render_turn(agent.records[0])[1]["content"]
        failed_section = content.split("FAILED STEPS")[1]
        assert "fail_tool" in failed_section
        assert "'error'" in failed_section
        # Args key must not appear in the failed section.
        assert "'args'" not in failed_section

    def test_render_turn_failed_error_truncated_by_preview_limit(self) -> None:
        """Error strings in FAILED entries are truncated by blackboard_preview_limit."""
        agent = self._make_agent_with_failed_slot(blackboard_preview_limit=10)
        content = agent.render_turn(agent.records[0])[1]["content"]
        failed_section = content.split("FAILED STEPS")[1]
        # The stored error is ToolInvocationError wrapping RuntimeError("intentional failure").
        # str(slot.error) = "Tool.tests.fail_tool: invocation failed: intentional failure" (57 chars).
        # After truncation to blackboard_preview_limit=10: "Tool.tests".
        assert "invocation failed" not in failed_section  # confirms truncation cut off the tail
        assert "Tool.test" in failed_section              # a prefix from the first 10 chars is present

    def test_render_turn_all_non_return_steps_failed_return_in_cached_section(self) -> None:
        """When all non-return steps fail, only return slot appears in CACHED STEPS."""
        agent = make_agent(context_enabled=True, fail_fast=False)
        register_math_tools(agent)
        agent.set_script([
            [{"tool": "Tool.tests.fail_tool", "args": {}}],          # step 0: FAILED
            [{"tool": return_tool.full_name, "args": {"val": 99}}],  # step 1: EXECUTED
        ])
        result = agent.invoke({"prompt": "run"})
        assert result.result == 99
        content = agent.render_turn(agent.records[0])[1]["content"]
        # Return slot executed → CACHED section present.
        assert "CACHED STEPS" in content
        # Failed steps → FAILED section present.
        assert "FAILED STEPS" in content
        # fail_tool must not appear in the CACHED section (it appears in FAILED).
        cached_section = content.split("CACHED STEPS")[1].split("FAILED STEPS")[0]
        assert "fail_tool" not in cached_section


class TestParsingHelpers:
    def test_extract_from_json_string_extracts_json_array_from_plain_text(self) -> None:
        agent = make_agent()

        value = agent._extract_from_json_string(
            'Plan:\n[{"step": 0, "tool": "Tool.tests.add", "args": {"x": 1, "y": 2}}]\nDone.'
        )

        assert value == [{"step": 0, "tool": "Tool.tests.add", "args": {"x": 1, "y": 2}}]

    def test_extract_from_json_string_extracts_json_array_from_markdown_fence(self) -> None:
        agent = make_agent()

        value = agent._extract_from_json_string(
            '```json\n[{"step": 0, "tool": "Tool.tests.add", "args": {"x": 1, "y": 2}}]\n```'
        )

        assert value == [{"step": 0, "tool": "Tool.tests.add", "args": {"x": 1, "y": 2}}]

    def test_extract_from_json_string_extracts_json_object_from_plain_text(self) -> None:
        agent = make_agent()

        value = agent._extract_from_json_string(
            'Before {"step": 0, "tool": "Tool.tests.add", "args": {}} after'
        )

        assert value == {"step": 0, "tool": "Tool.tests.add", "args": {}}

    def test_extract_from_json_string_extracts_json_object_from_markdown_fence(self) -> None:
        agent = make_agent()

        value = agent._extract_from_json_string(
            '```json\n{"step": 0, "tool": "Tool.tests.add", "args": {}}\n```'
        )

        assert value == {"step": 0, "tool": "Tool.tests.add", "args": {}}

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("[]", []),
            ("{}", {}),
        ],
    )
    def test_extract_from_json_string_accepts_empty_json_array_or_object(
        self,
        raw: str,
        expected: Any,
    ) -> None:
        agent = make_agent()

        assert agent._extract_from_json_string(raw) == expected

    @pytest.mark.parametrize("raw", ["", "   ", "not json"])
    def test_extract_from_json_string_rejects_invalid_text(self, raw: str) -> None:
        agent = make_agent()

        with pytest.raises(json.JSONDecodeError):
            agent._extract_from_json_string(raw)

    def test_extract_from_json_string_skips_unparseable_candidates(self) -> None:
        agent = make_agent()

        value = agent._extract_from_json_string("garbage {not valid json then [1, 2, 3]")

        assert value == [1, 2, 3]


class TestToolAgentAsyncBaseLoop:
    def test_async_scripted_invoke_runs_tools_placeholders_and_return(self) -> None:
        agent = make_agent()
        keys = register_math_tools(agent)
        agent.set_script(
            [
                [{"tool": keys["add"], "args": {"x": 2, "y": 3}}],
                [{"tool": keys["multiply"], "args": {"x": "<<__s0__>>", "y": 10}}],
                [{"tool": return_tool.full_name, "args": {"val": "<<__s1__>>"}}],
            ]
        )

        result = asyncio.run(agent.async_invoke({"prompt": "run"}))

        assert result.result == 50

    def test_async_context_enabled_persists_blackboard(self) -> None:
        agent = make_agent(context_enabled=True)
        keys = register_math_tools(agent)
        agent.set_script(
            [
                [{"tool": keys["add"], "args": {"x": 2, "y": 3}}],
                [{"tool": return_tool.full_name, "args": {"val": "<<__s0__>>"}}],
            ]
        )

        result = asyncio.run(agent.async_invoke({"prompt": "run"}))

        assert result.result == 5
        assert len(agent.blackboard) == 2
        assert agent.blackboard[0].result.result == 5
        assert agent.blackboard[1].result.result == 5

    def test_async_execute_prepared_batch_records_tool_error(self) -> None:
        agent = make_agent()
        keys = register_math_tools(agent)
        slot = prepared_slot(0, keys["fail_tool"], {})
        state = make_state(running=[slot], prepared_steps=[0])

        with pytest.raises((ToolInvocationError, ToolAgentError)):
            asyncio.run(agent._async_execute_prepared_batch(state))

        assert state.running_blackboard[0].error is not NO_VAL
        assert state.running_blackboard[0].status == "failed"
        assert state.running_blackboard[0].is_failed() is True

    def test_async_prepare_empty_batch_raises(self) -> None:
        agent = make_agent()
        agent.set_script([[]])

        with pytest.raises(ToolAgentError, match="empty batch"):
            asyncio.run(agent.async_invoke({"prompt": "run"}))

class TestToolAgentRecordMetadataContract:
    def test_render_turn_with_none_span_returns_base_user_assistant_pair(self) -> None:
        agent = make_agent()
        turn = ToolAgentRecord(
            user_prompt="run",
            generated_response="raw response",
            blackboard_start=None,
            blackboard_end=None,
        )

        rendered = agent.render_turn(turn)

        assert rendered == [
            {"role": "user", "content": "run"},
            {"role": "assistant", "content": "raw response"},
        ]

    def test_render_turn_with_empty_span_returns_base_user_assistant_pair(self) -> None:
        agent = make_agent()
        turn = ToolAgentRecord(
            user_prompt="run",
            generated_response="raw response",
            blackboard_start=0,
            blackboard_end=0,
        )

        rendered = agent.render_turn(turn)

        assert rendered == [
            {"role": "user", "content": "run"},
            {"role": "assistant", "content": "raw response"},
        ]

    def test_render_turn_rejects_span_beyond_current_blackboard(self) -> None:
        agent = make_agent()
        turn = ToolAgentRecord(
            user_prompt="run",
            generated_response="raw response",
            blackboard_start=0,
            blackboard_end=1,
        )

        with pytest.raises(ToolAgentError, match="Invalid blackboard span"):
            agent.render_turn(turn)

    def test_invoke_returns_tool_agent_result(self) -> None:
        """ToolAgent.invoke returns a ToolAgentResult, not a plain AgentResult."""
        agent = make_agent()
        agent.set_script(
            [[{"tool": return_tool.full_name, "args": {"val": 42}}]]
        )

        result = agent.invoke({"prompt": "run"})

        assert isinstance(result, ToolAgentResult)
        assert result.result == 42
        assert result.exception_records == ()

    def test_invoke_tool_usage_records_call_counts(self) -> None:
        """tool_usage on the returned ToolAgentResult reflects non-return call counts."""
        agent = make_agent()
        keys = register_math_tools(agent)
        agent.set_script(
            [
                [{"tool": keys["add"], "args": {"x": 1, "y": 2}}],
                [{"tool": keys["add"], "args": {"x": 3, "y": 4}}],
                [{"tool": return_tool.full_name, "args": {"val": 0}}],
            ]
        )

        result = agent.invoke({"prompt": "run"})

        assert isinstance(result, ToolAgentResult)
        usage = {r.tool_name: r.call_count for r in result.tool_usage}
        assert usage.get(keys["add"]) == 2

    def test_invoke_tool_usage_empty_when_only_return_tool_called(self) -> None:
        """tool_usage is empty when only the return tool was executed."""
        agent = make_agent()
        agent.set_script(
            [[{"tool": return_tool.full_name, "args": {"val": "done"}}]]
        )

        result = agent.invoke({"prompt": "run"})

        assert isinstance(result, ToolAgentResult)
        assert result.tool_usage == ()

    def test_invoke_result_has_empty_exception_records_on_clean_run(self) -> None:
        agent = make_agent()
        agent.set_script(
            [[{"tool": return_tool.full_name, "args": {"val": 42}}]]
        )

        result = agent.invoke({"prompt": "run"})

        assert isinstance(result, ToolAgentResult)
        assert result.exception_records == ()

    def test_invoke_result_with_fail_fast_false_and_failure_has_exception_records(self) -> None:
        agent = make_agent(fail_fast=False)
        keys = register_math_tools(agent)
        agent.set_script(
            [
                [{"tool": keys["fail_tool"], "args": {}}],
                [{"tool": return_tool.full_name, "args": {"val": 42}}],
            ]
        )

        result = agent.invoke({"prompt": "run"})

        assert result.result == 42
        assert len(result.exception_records) == 1
        blackboard_index, error = result.exception_records[0]
        assert isinstance(blackboard_index, int)
        assert isinstance(error, Exception)

    def test_blackboard_span_is_integer_when_context_disabled(self) -> None:
        agent = make_agent(context_enabled=False)
        agent.set_script(
            [[{"tool": return_tool.full_name, "args": {"val": 1}}]]
        )

        agent.invoke({"prompt": "run"})

        record = agent.records[0]
        assert isinstance(record.blackboard_start, int)
        assert isinstance(record.blackboard_end, int)


class TestAsyncHookDispatch:
    """Verify that async hook overrides are wired correctly for each agent type."""

    def test_toolagent_base_has_concrete_async_hooks(self) -> None:
        agent = make_agent()
        assert callable(getattr(agent, "_ainitialize_run_state", None))
        assert callable(getattr(agent, "_aprepare_next_batch", None))

    def test_planact_overrides_ainitialize_run_state(self) -> None:
        # PlanActAgent._ainitialize_run_state must be its own override, not the base default.
        agent = make_planact_agent(
            [
                json.dumps([
                    {"tool": "Tool.tests.add", "args": {"x": 1, "y": 2}},
                    {"tool": return_tool.full_name, "args": {"val": "<<__s0__>>"}},
                ])
            ]
        )
        assert type(agent)._ainitialize_run_state is not ToolAgent._ainitialize_run_state

    def test_react_overrides_aprepare_next_batch(self) -> None:
        # ReActAgent._aprepare_next_batch must be its own override, not the base default.
        agent = make_react_agent(
            [
                react_step_json(
                    tool="Tool.tests.add",
                    args={"x": 1, "y": 2},
                    description="Add the two numbers for the test.",
                )
            ],
            tool_calls_limit=1,
        )
        assert type(agent)._aprepare_next_batch is not ToolAgent._aprepare_next_batch

    def test_scripted_toolagent_async_invoke_uses_base_hook_defaults(self) -> None:
        # ScriptedToolAgent inherits the asyncio.to_thread defaults — full async_invoke works.
        agent = make_agent()
        keys = register_math_tools(agent)
        agent.set_script(
            [
                [{"tool": keys["add"], "args": {"x": 2, "y": 3}}],
                [{"tool": return_tool.full_name, "args": {"val": "<<__s0__>>"}}],
            ]
        )

        result = asyncio.run(agent.async_invoke({"prompt": "run"}))

        assert result.result == 5

    def test_planact_async_invoke_uses_agenerate_plan(self) -> None:
        # Full async_invoke via PlanActAgent exercises _ainitialize_run_state → _agenerate_plan.
        agent = make_planact_agent(
            [
                json.dumps([
                    {"tool": "Tool.tests.add", "args": {"x": 4, "y": 6}},
                    {"tool": return_tool.full_name, "args": {"val": "<<__s0__>>"}},
                ])
            ]
        )

        result = asyncio.run(agent.async_invoke({"prompt": "run plan"}))

        assert result.result == 10

    def test_react_async_invoke_uses_agenerate_next_step(self) -> None:
        # Full async_invoke via ReActAgent exercises _aprepare_next_batch → _agenerate_next_step.
        agent = make_react_agent(
            [
                react_step_json(
                    step=0,
                    tool="Tool.tests.add",
                    args={"x": 3, "y": 7},
                    description="Add the two numbers for the current calculation.",
                ),
                react_step_json(
                    step=1,
                    tool=return_tool.full_name,
                    args={"val": "<<__s0__>>"},
                    description="Return the addition result as the final answer.",
                ),
            ],
            tool_calls_limit=1,
        )

        result = asyncio.run(agent.async_invoke({"prompt": "run react"}))

        assert result.result == 10


# ── TestCascadeFailedPropagation ───────────────────────────────────────────────

class TestToolAgentToDictFastFail:
    """Tests that to_dict() includes the fail_fast and generation_retries keys."""

    def test_to_dict_includes_fail_fast_true(self) -> None:
        agent = make_agent(fail_fast=True)
        d = agent.to_dict()
        assert "fail_fast" in d
        assert d["fail_fast"] is True

    def test_to_dict_includes_fail_fast_false(self) -> None:
        agent = make_agent(fail_fast=False)
        d = agent.to_dict()
        assert "fail_fast" in d
        assert d["fail_fast"] is False

    def test_to_dict_includes_generation_retries_default(self) -> None:
        agent = make_agent()
        d = agent.to_dict()
        assert "generation_retries" in d
        assert d["generation_retries"] == 0

    def test_to_dict_generation_retries_reflects_construction_value(self) -> None:
        agent = ScriptedToolAgent(generation_retries=2)
        assert agent.to_dict()["generation_retries"] == 2


# ── TestReActGenerationRetry ──────────────────────────────────────────────────

class TestExecutePreparedBatchEarlyValidation:
    """I1: sync _execute_prepared_batch pre-validates tool existence before the gather."""

    def test_unknown_tool_raises_before_any_execution(self) -> None:
        """Unknown tool in batch raises ToolAgentError before any tool runs."""
        agent = make_agent()
        keys = register_math_tools(agent)

        state = make_state(
            running=[
                prepared_slot(0, keys["add"], {"x": 1, "y": 2}),
                prepared_slot(1, "Tool.tests.no_such_tool", {}),
            ],
            prepared_steps=[0, 1],
        )

        with pytest.raises(ToolAgentError, match="no_such_tool"):
            agent._execute_prepared_batch(state)

        # Neither slot should have executed — early exit before the gather.
        assert state.running_blackboard[0].is_empty() or state.running_blackboard[0].is_prepared()
        assert state.running_blackboard[1].is_empty() or state.running_blackboard[1].is_prepared()

    def test_known_tools_proceed_normally(self) -> None:
        """Batch with all valid tools executes without raising."""
        agent = make_agent()
        keys = register_math_tools(agent)

        state = make_state(
            running=[
                prepared_slot(0, keys["add"], {"x": 3, "y": 4}),
            ],
            prepared_steps=[0],
        )

        updated = agent._execute_prepared_batch(state)
        assert updated.running_blackboard[0].result.result == 7


# ── TestAwaitFieldKeyInErrors ────────────────────────────────────────────────

