from __future__ import annotations

import asyncio
from typing import Any

import pytest

from atomic_agentic.agents.scriptact import ScriptActAgent
from atomic_agentic.agents.tools import attr_call_tool, builtin_call_tool
from atomic_agentic.constants.agents import (
    ATTR_CALL_ALIAS,
    PY_BUILTIN_ALIAS,
    RETURN_ALIAS,
    RHS_ASSIGN_ALIAS,
)
from atomic_agentic.exceptions import ToolAgentError, ToolInvocationError, ToolRegistrationError
from atomic_agentic.tools import Tool
from ..fake_engines import FakeLLMEngine


def add(a: int, b: int) -> int:
    """Returns a + b."""
    return a + b


def log_message(message: str) -> None:
    """Pretends to log a message; returns nothing meaningful."""
    return None


def fail_tool(x: int) -> int:
    """Always raises -- exercises a real tool-execution failure."""
    raise RuntimeError("simulated tool failure")


class _Node:
    """Exposes a method whose own keyword argument is named `obj` -- an
    attribute/method-call dispatcher must not let its own internal
    parameter names collide with a real target method's argument names."""

    def __init__(self) -> None:
        self.children: list[Any] = []

    def attach(self, obj: Any) -> Any:
        self.children.append(obj)
        return obj


class _NoNameCallable:
    """A callable with no ``__name__`` -- register_tool/register_tools must
    wrap the resulting AttributeError as a ToolRegistrationError, not let it
    escape raw."""

    def __call__(self, value: int = 0) -> int:
        return value


def _make_agent(engine: FakeLLMEngine, *, tools: list[Any] | None = None, **kwargs: Any) -> ScriptActAgent:
    return ScriptActAgent(
        name="tests", namespace="tests", description="ScriptActAgent under test.",
        llm_engine=engine, tools=tools, **kwargs,
    )


class TestScriptActAgentConstruction:
    def test_defaults(self) -> None:
        agent = _make_agent(FakeLLMEngine(responses=[]))

        assert agent.regeneration_limit == 5
        assert agent.tool_calls_limit is None
        assert agent.replanning_limit == 2
        assert agent.fail_fast is False
        assert agent.tool_instructions is None

    def test_tool_instructions_forwarded(self) -> None:
        agent = _make_agent(FakeLLMEngine(responses=[]), tool_instructions="Be terse.")
        assert agent.tool_instructions == "Be terse."

    def test_regeneration_limit_rejects_negative(self) -> None:
        with pytest.raises(ToolAgentError):
            _make_agent(FakeLLMEngine(responses=[]), regeneration_limit=-1)

    def test_regeneration_limit_rejects_none(self) -> None:
        with pytest.raises(ToolAgentError):
            _make_agent(FakeLLMEngine(responses=[]), regeneration_limit=None)  # type: ignore[arg-type]

    def test_both_budget_limits_default_legal(self) -> None:
        agent = _make_agent(
            FakeLLMEngine(responses=[]), tool_calls_limit=None, replanning_limit=0,
        )

        assert agent.tool_calls_limit is None
        assert agent.replanning_limit == 0

    def test_tool_calls_limit_rejects_negative(self) -> None:
        with pytest.raises(ToolAgentError):
            _make_agent(FakeLLMEngine(responses=[]), tool_calls_limit=-1)

    def test_replanning_limit_rejects_negative(self) -> None:
        with pytest.raises(ToolAgentError):
            _make_agent(FakeLLMEngine(responses=[]), replanning_limit=-1)

    def test_replanning_limit_rejects_none(self) -> None:
        with pytest.raises(ToolAgentError):
            _make_agent(FakeLLMEngine(responses=[]), replanning_limit=None)  # type: ignore[arg-type]

    def test_fail_fast_rejects_non_bool(self) -> None:
        with pytest.raises(ToolAgentError):
            _make_agent(FakeLLMEngine(responses=[]), fail_fast="yes")  # type: ignore[arg-type]

    def test_fail_fast_round_trips(self) -> None:
        agent = _make_agent(FakeLLMEngine(responses=[]), fail_fast=True)
        assert agent.fail_fast is True

    def test_registering_tool_colliding_with_a_real_builtin_raises(self) -> None:
        agent = _make_agent(FakeLLMEngine(responses=[]))

        with pytest.raises(ToolRegistrationError):
            agent.register_tool(add, alias="len")

    @pytest.mark.parametrize(
        "sentinel", [RHS_ASSIGN_ALIAS, RETURN_ALIAS, PY_BUILTIN_ALIAS, ATTR_CALL_ALIAS],
    )
    def test_registering_tool_under_reserved_sentinel_raises(self, sentinel: str) -> None:
        agent = _make_agent(FakeLLMEngine(responses=[]))

        with pytest.raises(ToolRegistrationError):
            agent.register_tool(add, alias=sentinel)

    def test_constants_registered_at_construction(self) -> None:
        agent = _make_agent(
            FakeLLMEngine(responses=[]),
            constants=[1, "two"],
            constant_aliases=["one", None],
            constant_descriptions=["first constant", None],
        )

        assert agent.has_constant("one")
        assert agent.get_constant("one").value == 1
        assert agent.get_constant("one").description == "first constant"
        assert len(agent.constants) == 2

    def test_tool_concurrency_limit_defaults_to_none_and_round_trips(self) -> None:
        agent = _make_agent(FakeLLMEngine(responses=[]))
        assert agent.tool_concurrency_limit is None

        agent2 = _make_agent(FakeLLMEngine(responses=[]), tool_concurrency_limit=2)
        assert agent2.tool_concurrency_limit == 2

    def test_tool_concurrency_limit_rejects_zero(self) -> None:
        with pytest.raises(ToolAgentError):
            _make_agent(FakeLLMEngine(responses=[]), tool_concurrency_limit=0)

    def test_to_dict_includes_fail_fast_and_replanning_limit(self) -> None:
        agent = _make_agent(FakeLLMEngine(responses=[]), fail_fast=True, replanning_limit=3)

        d = agent.to_dict()

        assert d["fail_fast"] is True
        assert d["replanning_limit"] == 3


class TestScriptActAgentRegistrationEdgeCases:
    def test_unknown_collision_policy_raises(self) -> None:
        agent = _make_agent(FakeLLMEngine(responses=[]))

        with pytest.raises(ToolRegistrationError):
            agent.register_tool(add, name_collision_policy="bogus")  # type: ignore[arg-type]

    def test_non_identifier_alias_raises(self) -> None:
        agent = _make_agent(FakeLLMEngine(responses=[]))

        with pytest.raises(ToolRegistrationError):
            agent.register_tool(add, alias="not valid!")

    def test_register_tool_accepts_an_already_wrapped_invokable_directly(self) -> None:
        agent = _make_agent(FakeLLMEngine(responses=[]))
        wrapped = Tool(function=add, name="add_tool", namespace="tests", description="Adds.")

        agent.register_tool(wrapped)

        assert agent.get_tool("add_tool") is wrapped

    def test_register_tool_wraps_toolify_failure_as_registration_error(self) -> None:
        agent = _make_agent(FakeLLMEngine(responses=[]))

        with pytest.raises(ToolRegistrationError):
            agent.register_tool(_NoNameCallable())

    def test_register_tool_rejects_an_unsupported_component_type(self) -> None:
        agent = _make_agent(FakeLLMEngine(responses=[]))

        with pytest.raises(ToolRegistrationError):
            agent.register_tool(123)  # type: ignore[arg-type]

    def test_register_tool_duplicate_raises_by_default(self) -> None:
        agent = _make_agent(FakeLLMEngine(responses=[]), tools=[add])

        with pytest.raises(ToolRegistrationError):
            agent.register_tool(add)

    def test_register_tool_duplicate_with_skip_policy_returns_false(self) -> None:
        agent = _make_agent(FakeLLMEngine(responses=[]), tools=[add])

        result = agent.register_tool(add, name_collision_policy="skip")

        assert result is False

    def test_register_tool_duplicate_with_replace_policy_overwrites(self) -> None:
        agent = _make_agent(FakeLLMEngine(responses=[]), tools=[add])
        wrapped = Tool(function=add, name="add", namespace="tests", description="A different add.")

        result = agent.register_tool(wrapped, name_collision_policy="replace")

        assert result is True
        assert agent.get_tool("add") is wrapped

    def test_register_tools_alias_length_mismatch_raises(self) -> None:
        agent = _make_agent(FakeLLMEngine(responses=[]))

        with pytest.raises(ValueError):
            agent.register_tools([add], aliases=["a", "b"])

    def test_register_tools_intra_batch_duplicate_raises(self) -> None:
        agent = _make_agent(FakeLLMEngine(responses=[]))

        with pytest.raises(ToolRegistrationError):
            agent.register_tools([add, add])

    def test_register_tools_collision_against_existing_toolbox_skip(self) -> None:
        agent = _make_agent(FakeLLMEngine(responses=[]), tools=[add])
        original = agent.get_tool("add")

        result = agent.register_tools([add], name_collision_policy="skip")

        assert result is False
        assert agent.get_tool("add") is original


class TestScriptActAgentToolAccessors:
    def test_get_tool_raises_for_unknown_id(self) -> None:
        agent = _make_agent(FakeLLMEngine(responses=[]))

        with pytest.raises(ToolAgentError):
            agent.get_tool("nonexistent")

    def test_list_has_remove_clear_tools_roundtrip(self) -> None:
        agent = _make_agent(FakeLLMEngine(responses=[]), tools=[add])

        assert agent.has_tool("add")
        assert "add" in agent.list_tools()

        assert agent.remove_tool("add") is True
        assert agent.remove_tool("add") is False
        assert not agent.has_tool("add")

        agent.register_tool(add)
        agent.clear_tools()
        assert agent.list_tools() == {}


class TestScriptActAgentConstantAccessors:
    def test_register_constant_auto_names_without_an_alias(self) -> None:
        agent = _make_agent(FakeLLMEngine(responses=[]))

        agent.register_constant(1)
        agent.register_constant(2)

        assert agent.has_constant("K_0")
        assert agent.has_constant("K_1")

    def test_register_constant_duplicate_raises_by_default(self) -> None:
        agent = _make_agent(FakeLLMEngine(responses=[]))
        agent.register_constant(1, alias="x")

        with pytest.raises(ToolAgentError):
            agent.register_constant(2, alias="x")

    def test_register_constant_duplicate_with_suffix_renames_instead_of_colliding(self) -> None:
        agent = _make_agent(FakeLLMEngine(responses=[]))
        agent.register_constant(1, alias="x")

        result = agent.register_constant(2, alias="x", name_collision_policy="suffix")

        assert result is True
        assert agent.get_constant("x").value == 1
        assert agent.get_constant("x_0").value == 2

    def test_get_constant_raises_for_unknown_alias(self) -> None:
        agent = _make_agent(FakeLLMEngine(responses=[]))

        with pytest.raises(ToolAgentError):
            agent.get_constant("nonexistent")

    def test_remove_constant_roundtrip(self) -> None:
        agent = _make_agent(FakeLLMEngine(responses=[]))
        agent.register_constant(1, alias="x")

        assert agent.remove_constant("x") is True
        assert agent.remove_constant("x") is False
        assert not agent.has_constant("x")

    def test_update_constant_description_replaces_only_the_description(self) -> None:
        agent = _make_agent(FakeLLMEngine(responses=[]))
        agent.register_constant(1, alias="x", description="old")

        agent.update_constant_description("x", "new")

        spec = agent.get_constant("x")
        assert spec.description == "new"
        assert spec.value == 1

    def test_clear_constants(self) -> None:
        agent = _make_agent(FakeLLMEngine(responses=[]))
        agent.register_constant(1, alias="x")

        agent.clear_constants()

        assert agent.constants == {}


class TestScriptActAgentPromptContextRendering:
    def test_actions_context_renders_alias_distinct_from_the_tool_name(self) -> None:
        agent = _make_agent(FakeLLMEngine(responses=[]))
        agent.register_tool(add, alias="plus")

        rendered = agent.actions_context()

        assert rendered.startswith("plus(")
        assert "def add" not in rendered

    def test_actions_context_renders_a_multiline_tool_description(self) -> None:
        def multiline_tool(x: int) -> int:
            """First line.

            Second paragraph.
            """
            return x

        agent = _make_agent(FakeLLMEngine(responses=[]), tools=[multiline_tool])

        rendered = agent.actions_context()

        assert "First line." in rendered
        assert "Second paragraph." in rendered

    def test_constants_context_renders_the_wire_name_and_type(self) -> None:
        agent = _make_agent(FakeLLMEngine(responses=[]))
        agent.register_constant(5, alias="count", description="A count.")

        rendered = agent.constants_context()

        assert "K_COUNT: int" in rendered
        assert "A count." in rendered

    def test_system_message_includes_excluded_builtins(self) -> None:
        agent = _make_agent(FakeLLMEngine(responses=[]))

        task = agent._initialize_task(turns=[], prompt="x", inputs={})
        messages = agent.render_task(task)

        system_msg = next(m["content"] for m in messages if m["role"] == "system")
        assert "eval" in system_msg


class TestScriptActAgentOneShotLifecycle:
    def test_registered_tool_call_and_return(self) -> None:
        engine = FakeLLMEngine(responses=["result = add(a=4, b=6)\nreturn result"])
        agent = _make_agent(engine, tools=[add])

        final = agent.invoke({"prompt": "add 4 and 6"})

        assert final.result == 10
        assert engine.call_count == 1

    def test_bare_call_and_literal_return(self) -> None:
        engine = FakeLLMEngine(responses=["log_message(message='hi')\nreturn 42"])
        agent = _make_agent(engine, tools=[log_message], context_enabled=True)

        final = agent.invoke({"prompt": "log then return"})

        assert final.result == 42
        record = agent.get_conversation()[-1]
        assert any(s.tool == "log_message" for s in record.statements)


class TestScriptActAgentRegenRepair:
    """Within-round regeneration: a malformed/invalid draft is rejected
    before ever being compiled, and the model gets corrective feedback in
    the SAME round -- no repair round, no needs_repair involvement."""

    def test_regen_recovers_from_undefined_reference(self) -> None:
        engine = FakeLLMEngine(responses=[
            "result = add(a=1, b=undefined_name)\nreturn result",
            "result = add(a=1, b=2)\nreturn result",
        ])
        agent = _make_agent(engine, tools=[add], regeneration_limit=1)

        final = agent.invoke({"prompt": "add"})

        assert final.result == 3
        assert engine.call_count == 2
        feedback = "\n".join(m["content"] for m in engine.calls[1] if m["role"] != "system")
        assert "Your plan could not be used" in feedback
        assert "undefined_name" in feedback

    def test_regeneration_limit_zero_raises_immediately(self) -> None:
        engine = FakeLLMEngine(responses=["result = add(a=1, b=undefined_name)\nreturn result"])
        agent = _make_agent(engine, tools=[add], regeneration_limit=0)

        with pytest.raises(ToolAgentError, match="exhausted after 1 attempt"):
            agent.invoke({"prompt": "add"})

    def test_regeneration_limit_exhausted_after_two_attempts(self) -> None:
        engine = FakeLLMEngine(responses=[
            "result = add(a=1, b=undefined_name)\nreturn result",
            "result = add(a=1, b=other_undefined)\nreturn result",
        ])
        agent = _make_agent(engine, tools=[add], regeneration_limit=1)

        with pytest.raises(ToolAgentError):
            agent.invoke({"prompt": "add"})

        assert engine.call_count == 2

    def test_conditional_statement_is_a_regen_repair_issue(self) -> None:
        # Pass 8: an `if` always raises BlackboardParseError -- fed back as
        # regen-repair feedback, never a silent continuation.
        engine = FakeLLMEngine(responses=[
            "if True:\n    y = 1",
            "y = add(a=1, b=2)\nreturn y",
        ])
        agent = _make_agent(engine, tools=[add], regeneration_limit=1)

        final = agent.invoke({"prompt": "add stuff"})

        assert final.result == 3
        assert engine.call_count == 2


class TestScriptActAgentRepairRounds:
    """Framework-driven repair rounds (Pass 8): a REAL resolution
    (prepare()) or execution (act()) failure -- never a model choice --
    grants a fresh continuation round, up to replanning_limit."""

    def test_resolution_failure_triggers_a_repair_round(self) -> None:
        engine = FakeLLMEngine(responses=[
            "x = 'hello'\ny = add(a=x + 1, b=2)\nreturn y",
            "return add(a=1, b=1)",
        ])
        agent = _make_agent(engine, tools=[add], replanning_limit=1, context_enabled=True)

        final = agent.invoke({"prompt": "do something"})

        assert final.result == 2
        assert engine.call_count == 2
        round2_content = "\n".join(m["content"] for m in engine.calls[1])
        assert "THIS BATCH FAILED" in round2_content
        assert "Continue planning the rest of this task" in round2_content
        record = agent.get_conversation()[-1]
        assert record.repair_rounds_used == 1
        assert not any(s.identifier == "y" for s in record.statements)
        assert any(s.identifier == "y" for s in record.failed_statements)

    def test_execution_failure_triggers_a_repair_round_preserving_the_other_success(self) -> None:
        engine = FakeLLMEngine(responses=[
            "a = add(a=1, b=1)\nb = fail_tool(x=1)\nreturn a",
            "return a",
        ])
        agent = _make_agent(engine, tools=[add, fail_tool], replanning_limit=1, context_enabled=True)

        final = agent.invoke({"prompt": "do something"})

        assert final.result == 2
        assert engine.call_count == 2
        round2_content = "\n".join(m["content"] for m in engine.calls[1])
        assert "simulated tool failure" in round2_content
        record = agent.get_conversation()[-1]
        assert record.repair_rounds_used == 1
        assert any(s.identifier == "a" for s in record.statements)
        assert any(s.tool == "fail_tool" for s in record.failed_statements)

    def test_fail_fast_raises_immediately_on_resolution_failure_without_repair(self) -> None:
        engine = FakeLLMEngine(responses=["x = 'hello'\ny = add(a=x + 1, b=2)\nreturn y"])
        agent = _make_agent(engine, tools=[add], replanning_limit=2, fail_fast=True)

        with pytest.raises(ToolAgentError, match="fail_fast=True"):
            agent.invoke({"prompt": "do something"})

        assert engine.call_count == 1

    def test_fail_fast_raises_immediately_on_execution_failure_without_repair(self) -> None:
        engine = FakeLLMEngine(responses=["return fail_tool(x=1)"])
        agent = _make_agent(engine, tools=[fail_tool], replanning_limit=2, fail_fast=True)

        with pytest.raises(ToolInvocationError, match="simulated tool failure"):
            agent.invoke({"prompt": "do something"})

        assert engine.call_count == 1

    def test_replanning_limit_zero_raises_immediately_on_resolution_failure(self) -> None:
        engine = FakeLLMEngine(responses=["x = 'hello'\ny = add(a=x + 1, b=2)\nreturn y"])
        agent = _make_agent(engine, tools=[add], replanning_limit=0)

        with pytest.raises(ToolAgentError, match="repair budget exhausted"):
            agent.invoke({"prompt": "do something"})

        assert engine.call_count == 1

    def test_replanning_limit_zero_raises_immediately_on_execution_failure(self) -> None:
        engine = FakeLLMEngine(responses=["return fail_tool(x=1)"])
        agent = _make_agent(engine, tools=[fail_tool], replanning_limit=0)

        with pytest.raises(ToolInvocationError, match="simulated tool failure"):
            agent.invoke({"prompt": "do something"})

        assert engine.call_count == 1

    def test_repair_round_not_exhausting_the_budget_proceeds_normally(self) -> None:
        engine = FakeLLMEngine(responses=[
            "return fail_tool(x=1)",
            "return add(a=1, b=1)",
        ])
        agent = _make_agent(engine, tools=[fail_tool, add], replanning_limit=2)

        final = agent.invoke({"prompt": "do something"})

        assert final.result == 2
        assert engine.call_count == 2


class TestScriptActAgentBudgets:
    def test_tool_calls_limit_enforced_via_regen_feedback(self) -> None:
        engine = FakeLLMEngine(responses=[
            "a = add(a=1, b=1)\nb = add(a=2, b=2)\nreturn a",
            "a = add(a=1, b=1)\nreturn a",
        ])
        agent = _make_agent(
            engine, tools=[add], tool_calls_limit=1, regeneration_limit=1,
        )

        final = agent.invoke({"prompt": "do something"})

        assert final.result == 2
        assert engine.call_count == 2
        feedback = "\n".join(m["content"] for m in engine.calls[1] if m["role"] != "system")
        assert "exceeding the configured limit" in feedback

    def test_tool_concurrency_limit_splits_a_single_plan_into_multiple_batches(self) -> None:
        engine = FakeLLMEngine(responses=[
            "a = add(a=1, b=1)\nb = add(a=2, b=2)\nreturn a",
        ])
        agent = _make_agent(engine, tools=[add], tool_concurrency_limit=1, context_enabled=True)

        final = agent.invoke({"prompt": "do something"})

        assert final.result == 2
        assert engine.call_count == 1
        record = agent.get_conversation()[-1]
        a_stmt = next(s for s in record.statements if s.identifier == "a")
        b_stmt = next(s for s in record.statements if s.identifier == "b")
        assert a_stmt.batch_index != b_stmt.batch_index


class TestScriptActAgentMutationSafety:
    def test_constant_mutation_persists_across_repair_rounds_within_one_invocation(self) -> None:
        engine = FakeLLMEngine(responses=[
            "x = K_MYLIST.append(99)\nb = fail_tool(x=1)\nreturn K_MYLIST",
            "return K_MYLIST",
        ])
        agent = _make_agent(engine, tools=[fail_tool], replanning_limit=1)
        agent.register_constant([1, 2, 3], alias="mylist")

        final = agent.invoke({"prompt": "mutate"})

        assert final.result == [1, 2, 3, 99]

    def test_constant_mutation_does_not_persist_to_a_fresh_invocation(self) -> None:
        engine = FakeLLMEngine(responses=[
            "x = K_MYLIST.append(99)\nb = fail_tool(x=1)\nreturn K_MYLIST",
            "return K_MYLIST",
            "return K_MYLIST",
        ])
        agent = _make_agent(engine, tools=[fail_tool], replanning_limit=1)
        agent.register_constant([1, 2, 3], alias="mylist")

        first = agent.invoke({"prompt": "mutate"})
        second = agent.invoke({"prompt": "check again"})

        assert first.result == [1, 2, 3, 99]
        assert second.result == [1, 2, 3]

    def test_atomic_immutable_constant_is_not_deep_copied(self) -> None:
        agent = _make_agent(FakeLLMEngine(responses=[]))
        original = "hello"
        agent.register_constant(original, alias="mystr")

        task = agent._initialize_task(turns=[], prompt="x", inputs={})

        assert task.constant_values["K_MYSTR"] is original

    def test_task_result_mutation_does_not_persist_across_invocations(self) -> None:
        engine = FakeLLMEngine(responses=[
            "return [1, 2, 3]",
            "x = task_result_0.append(99)\nreturn task_result_0",
            "return task_result_0",
        ])
        agent = _make_agent(engine, context_enabled=True)

        first = agent.invoke({"prompt": "make a list"})
        second = agent.invoke({"prompt": "mutate it"})
        third = agent.invoke({"prompt": "check again"})

        assert first.result == [1, 2, 3]
        assert second.result == [1, 2, 3, 99]
        assert third.result == [1, 2, 3]


class TestScriptActAgentDispatch:
    def test_registered_tool_with_ordinary_kwargs(self) -> None:
        engine = FakeLLMEngine(responses=["return add(a=2, b=3)"])
        agent = _make_agent(engine, tools=[add])

        final = agent.invoke({"prompt": "add"})

        assert final.result == 5

    def test_approved_builtin_dispatch(self) -> None:
        engine = FakeLLMEngine(responses=["return len([1, 2, 3])"])
        agent = _make_agent(engine)

        final = agent.invoke({"prompt": "count"})

        assert final.result == 3

    def test_attribute_call_on_a_constant(self) -> None:
        engine = FakeLLMEngine(responses=["return K_MYDICT.get('a')"])
        agent = _make_agent(engine)
        agent.register_constant({"a": 1}, alias="mydict")

        final = agent.invoke({"prompt": "get a"})

        assert final.result == 1

    def test_attribute_call_dispatches_when_the_real_method_has_a_keyword_argument_named_obj(self) -> None:
        # The real target method's own kwarg is literally named `obj`,
        # which would collide with attr_call_tool's own dispatcher
        # parameter of the same name if it were splatted directly.
        node = _Node()
        engine = FakeLLMEngine(responses=["return K_NODE.attach(obj=5)"])
        agent = _make_agent(engine)
        agent.register_constant(node, alias="node")

        final = agent.invoke({"prompt": "attach"})

        assert final.result == 5

    def test_builtin_call_with_keyword_arguments_dispatches_correctly(self) -> None:
        engine = FakeLLMEngine(responses=["return sorted([3, 1, 2], reverse=True)"])
        agent = _make_agent(engine)

        final = agent.invoke({"prompt": "sort"})

        assert final.result == [3, 2, 1]

    def test_dunder_attribute_call_raises_at_dispatch_level(self) -> None:
        with pytest.raises(ToolInvocationError, match="not available here"):
            attr_call_tool.invoke(
                {"obj": [1, 2], "method_name": "__class__", "args": (), "kwargs": {}}
            )

    def test_excluded_builtin_name_raises_at_dispatch_level(self) -> None:
        with pytest.raises(ToolInvocationError, match="not available here"):
            builtin_call_tool.invoke({"name": "eval", "args": (), "kwargs": {}})

    def test_dispatch_breakdown_counts_all_five_categories(self) -> None:
        engine = FakeLLMEngine(responses=[
            "c = add(a=1, b=2)\n"
            "d = len([1, 2, 3])\n"
            "e = K_MYDICT.get('a')\n"
            "f = c + d\n"
            "return f"
        ])
        agent = _make_agent(engine, tools=[add], context_enabled=True)
        agent.register_constant({"a": 1}, alias="mydict")

        final = agent.invoke({"prompt": "combine"})

        assert final.result == 6
        assert engine.call_count == 1
        record = agent.get_conversation()[-1]
        breakdown = record.dispatch_breakdown()
        assert breakdown.registered_tool_calls == 1
        assert breakdown.builtin_calls == 1
        assert breakdown.attribute_calls == 1
        assert breakdown.rhs_assignment_count == 1
        assert breakdown.binop_count == 1


class TestScriptActAgentPrepareActEdgeCases:
    def test_a_genuinely_empty_generation_completes_with_a_none_result(self) -> None:
        engine = FakeLLMEngine(responses=[""])
        agent = _make_agent(engine)

        final = agent.invoke({"prompt": "do nothing"})

        assert final.result is None

    def test_prepare_with_empty_pending_always_finalizes_regardless_of_needs_repair(self) -> None:
        # Per ScriptActAgent.prepare()'s own docstring: there is no "leave it
        # as-is, a repair round is still pending" branch -- think() always
        # clears needs_repair the moment it actually regenerates, so by the
        # time prepare() ever sees an empty `pending` again, needs_repair is
        # guaranteed already False in real operation. This direct unit-level
        # call (bypassing think() entirely) confirms prepare() finalizes
        # unconditionally on an empty `pending`, with no special-case for a
        # manually-forced needs_repair=True.
        agent = _make_agent(FakeLLMEngine(responses=[]))
        task = agent._initialize_task(turns=[], prompt="x", inputs={})
        task.pending = []
        task.needs_repair = True

        result = agent.prepare(task)

        assert result is task
        assert result.complete is True
        assert result.generated_response is None
        assert result.resolved_args == []

    def test_argument_binding_mismatch_against_a_real_tool_signature_triggers_a_repair_round(self) -> None:
        engine = FakeLLMEngine(responses=[
            "return add(a=1, c=2)",
            "return add(a=1, b=2)",
        ])
        agent = _make_agent(engine, tools=[add], replanning_limit=1)

        final = agent.invoke({"prompt": "add"})

        assert final.result == 3
        round2_content = "\n".join(m["content"] for m in engine.calls[1])
        assert "parameter contract" in round2_content

    def test_execution_failure_of_a_builtin_call_reports_a_readable_failure_label(self) -> None:
        engine = FakeLLMEngine(responses=[
            "x = sorted(None)\nreturn x",
            "return sorted([2, 1])",
        ])
        agent = _make_agent(engine, replanning_limit=1)

        final = agent.invoke({"prompt": "sort"})

        assert final.result == [1, 2]
        round2_content = "\n".join(m["content"] for m in engine.calls[1])
        assert "sorted(None)" in round2_content

    def test_execution_failure_of_an_attribute_call_reports_a_readable_failure_label(self) -> None:
        engine = FakeLLMEngine(responses=[
            "x = K_MYLIST.nonexistent_method()\nreturn x",
            "return K_MYLIST",
        ])
        agent = _make_agent(engine, replanning_limit=1)
        agent.register_constant([1, 2], alias="mylist")

        final = agent.invoke({"prompt": "call it"})

        assert final.result == [1, 2]
        round2_content = "\n".join(m["content"] for m in engine.calls[1])
        assert "K_MYLIST.nonexistent_method" in round2_content

    def test_a_batch_that_drains_without_a_return_finalizes_with_a_none_result(self) -> None:
        engine = FakeLLMEngine(responses=["log_message(message='hi')"])
        agent = _make_agent(engine, tools=[log_message])

        final = agent.invoke({"prompt": "log only"})

        assert final.result is None


class TestScriptActAgentAsyncLifecycle:
    def test_async_invoke_one_shot_lifecycle(self) -> None:
        engine = FakeLLMEngine(responses=["result = add(a=4, b=6)\nreturn result"])
        agent = _make_agent(engine, tools=[add])

        final = asyncio.run(agent.async_invoke({"prompt": "add 4 and 6"}))

        assert final.result == 10
        assert engine.call_count == 1

    def test_async_invoke_regen_recovers_from_undefined_reference(self) -> None:
        engine = FakeLLMEngine(responses=[
            "result = add(a=1, b=undefined_name)\nreturn result",
            "result = add(a=1, b=2)\nreturn result",
        ])
        agent = _make_agent(engine, tools=[add], regeneration_limit=1)

        final = asyncio.run(agent.async_invoke({"prompt": "add"}))

        assert final.result == 3
        assert engine.call_count == 2

    def test_async_invoke_resolution_failure_triggers_a_repair_round(self) -> None:
        engine = FakeLLMEngine(responses=[
            "x = 'hello'\ny = add(a=x + 1, b=2)\nreturn y",
            "return add(a=1, b=1)",
        ])
        agent = _make_agent(engine, tools=[add], replanning_limit=1)

        final = asyncio.run(agent.async_invoke({"prompt": "do something"}))

        assert final.result == 2
        assert engine.call_count == 2

    def test_async_invoke_replanning_limit_zero_raises_immediately(self) -> None:
        engine = FakeLLMEngine(responses=["x = 'hello'\ny = add(a=x + 1, b=2)\nreturn y"])
        agent = _make_agent(engine, tools=[add], replanning_limit=0)

        with pytest.raises(ToolAgentError, match="repair budget exhausted"):
            asyncio.run(agent.async_invoke({"prompt": "do something"}))

        assert engine.call_count == 1

    def test_async_invoke_regeneration_limit_exhausted_raises(self) -> None:
        engine = FakeLLMEngine(responses=["result = add(a=1, b=undefined_name)\nreturn result"])
        agent = _make_agent(engine, tools=[add], regeneration_limit=0)

        with pytest.raises(ToolAgentError, match="exhausted after 1 attempt"):
            asyncio.run(agent.async_invoke({"prompt": "add"}))
