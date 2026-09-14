from __future__ import annotations

import asyncio
from typing import Any

import pytest

from atomic_agentic.agents.script import ScriptAgent
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


def _make_agent(engine: FakeLLMEngine, *, tools: list[Any] | None = None, **kwargs: Any) -> ScriptAgent:
    return ScriptAgent(
        name="tests", namespace="tests", description="ScriptAgent under test.",
        llm_engine=engine, tools=tools, **kwargs,
    )


class TestScriptAgentConstruction:
    def test_defaults(self) -> None:
        agent = _make_agent(FakeLLMEngine(responses=[]))

        assert agent.regeneration_limit == 5
        assert agent.tool_calls_limit is None
        assert agent.planning_rounds_limit == 25

    def test_regeneration_limit_rejects_negative(self) -> None:
        with pytest.raises(ToolAgentError):
            _make_agent(FakeLLMEngine(responses=[]), regeneration_limit=-1)

    def test_regeneration_limit_rejects_none(self) -> None:
        with pytest.raises(ToolAgentError):
            _make_agent(FakeLLMEngine(responses=[]), regeneration_limit=None)  # type: ignore[arg-type]

    def test_both_budget_limits_none_is_legal(self) -> None:
        agent = _make_agent(
            FakeLLMEngine(responses=[]), tool_calls_limit=None, planning_rounds_limit=None,
        )

        assert agent.tool_calls_limit is None
        assert agent.planning_rounds_limit is None

    def test_tool_calls_limit_rejects_negative(self) -> None:
        with pytest.raises(ToolAgentError):
            _make_agent(FakeLLMEngine(responses=[]), tool_calls_limit=-1)

    def test_planning_rounds_limit_rejects_negative(self) -> None:
        with pytest.raises(ToolAgentError):
            _make_agent(FakeLLMEngine(responses=[]), planning_rounds_limit=-1)

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


class TestScriptAgentRegistrationEdgeCases:
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

    def test_register_tools_accepts_an_already_wrapped_invokable(self) -> None:
        agent = _make_agent(FakeLLMEngine(responses=[]))
        wrapped = Tool(function=add, name="add_tool", namespace="tests", description="Adds.")

        agent.register_tools([wrapped])

        assert agent.get_tool("add_tool") is wrapped

    def test_register_tools_wraps_toolify_failure_as_registration_error(self) -> None:
        agent = _make_agent(FakeLLMEngine(responses=[]))

        with pytest.raises(ToolRegistrationError):
            agent.register_tools([_NoNameCallable()])

    def test_register_tools_rejects_unsupported_item_type(self) -> None:
        agent = _make_agent(FakeLLMEngine(responses=[]))

        with pytest.raises(ToolRegistrationError):
            agent.register_tools([123])  # type: ignore[list-item]

    def test_register_tools_intra_batch_duplicate_raises(self) -> None:
        agent = _make_agent(FakeLLMEngine(responses=[]))

        with pytest.raises(ToolRegistrationError):
            agent.register_tools([add, add])

    def test_register_tools_collision_against_existing_toolbox_raises(self) -> None:
        agent = _make_agent(FakeLLMEngine(responses=[]), tools=[add])

        with pytest.raises(ToolRegistrationError):
            agent.register_tools([add])

    def test_register_tools_collision_against_existing_toolbox_skip(self) -> None:
        agent = _make_agent(FakeLLMEngine(responses=[]), tools=[add])
        original = agent.get_tool("add")

        result = agent.register_tools([add], name_collision_policy="skip")

        assert result is False
        assert agent.get_tool("add") is original

    def test_register_tools_collision_against_existing_toolbox_replace(self) -> None:
        agent = _make_agent(FakeLLMEngine(responses=[]), tools=[add])
        original = agent.get_tool("add")

        result = agent.register_tools([add], name_collision_policy="replace")

        assert result is True
        assert agent.get_tool("add") is not original


class TestScriptAgentToolAccessors:
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


class TestScriptAgentConstantAccessors:
    def test_register_constant_auto_names_without_an_alias(self) -> None:
        agent = _make_agent(FakeLLMEngine(responses=[]))

        agent.register_constant(1)
        agent.register_constant(2)

        assert agent.has_constant("K_0")
        assert agent.has_constant("K_1")

    def test_register_constant_rejects_empty_alias(self) -> None:
        agent = _make_agent(FakeLLMEngine(responses=[]))

        with pytest.raises(ToolAgentError):
            agent.register_constant(1, alias="   ")

    def test_register_constant_duplicate_raises_by_default(self) -> None:
        agent = _make_agent(FakeLLMEngine(responses=[]))
        agent.register_constant(1, alias="x")

        with pytest.raises(ToolAgentError):
            agent.register_constant(2, alias="x")

    def test_register_constant_duplicate_with_skip_returns_false(self) -> None:
        agent = _make_agent(FakeLLMEngine(responses=[]))
        agent.register_constant(1, alias="x")

        result = agent.register_constant(2, alias="x", name_collision_policy="skip")

        assert result is False
        assert agent.get_constant("x").value == 1

    def test_register_constant_duplicate_with_replace_overwrites(self) -> None:
        agent = _make_agent(FakeLLMEngine(responses=[]))
        agent.register_constant(1, alias="x")

        result = agent.register_constant(2, alias="x", name_collision_policy="replace")

        assert result is True
        assert agent.get_constant("x").value == 2

    def test_register_constant_duplicate_with_suffix_renames_instead_of_colliding(self) -> None:
        agent = _make_agent(FakeLLMEngine(responses=[]))
        agent.register_constant(1, alias="x")

        result = agent.register_constant(2, alias="x", name_collision_policy="suffix")

        assert result is True
        assert agent.get_constant("x").value == 1
        assert agent.get_constant("x_0").value == 2

    def test_register_constant_duplicate_with_suffix_advances_past_an_already_taken_candidate(self) -> None:
        # Forces the suffix retry loop to actually iterate (x -> x_0 already
        # taken -> x_1), not just succeed on its first candidate.
        agent = _make_agent(FakeLLMEngine(responses=[]))
        agent.register_constant(1, alias="x")
        agent.register_constant(2, alias="x", name_collision_policy="suffix")

        result = agent.register_constant(3, alias="x", name_collision_policy="suffix")

        assert result is True
        assert agent.get_constant("x_1").value == 3

    def test_register_constants_batch_mixes_auto_named_and_aliased_entries(self) -> None:
        agent = _make_agent(FakeLLMEngine(responses=[]))

        agent.register_constants([1, 2], aliases=["one", None], descriptions=["first", None])

        assert agent.get_constant("one").value == 1
        assert agent.get_constant("one").description == "first"
        assert agent.has_constant("K_0")

    def test_register_constants_batch_alias_length_mismatch_raises(self) -> None:
        agent = _make_agent(FakeLLMEngine(responses=[]))

        with pytest.raises(ValueError):
            agent.register_constants([1, 2], aliases=["only_one"])

    def test_register_constants_batch_description_length_mismatch_raises(self) -> None:
        agent = _make_agent(FakeLLMEngine(responses=[]))

        with pytest.raises(ValueError):
            agent.register_constants([1, 2], descriptions=["only_one"])

    def test_register_constants_batch_intra_batch_duplicate_raises(self) -> None:
        agent = _make_agent(FakeLLMEngine(responses=[]))

        with pytest.raises(ToolAgentError):
            agent.register_constants([1, 2], aliases=["x", "x"])

    def test_register_constants_batch_rejects_an_empty_alias(self) -> None:
        agent = _make_agent(FakeLLMEngine(responses=[]))

        with pytest.raises(ToolAgentError):
            agent.register_constants([1], aliases=["   "])

    def test_register_constants_batch_raises_against_an_existing_registry_key(self) -> None:
        agent = _make_agent(FakeLLMEngine(responses=[]))
        agent.register_constant(1, alias="x")

        with pytest.raises(ToolAgentError):
            agent.register_constants([2], aliases=["x"])

    def test_register_constants_batch_skip_policy(self) -> None:
        agent = _make_agent(FakeLLMEngine(responses=[]))
        agent.register_constant(1, alias="x")

        result = agent.register_constants([2], aliases=["x"], name_collision_policy="skip")

        assert result is False
        assert agent.get_constant("x").value == 1

    def test_register_constants_batch_replace_policy_overwrites_an_existing_key(self) -> None:
        agent = _make_agent(FakeLLMEngine(responses=[]))
        agent.register_constant(1, alias="x")

        result = agent.register_constants([2], aliases=["x"], name_collision_policy="replace")

        assert result is True
        assert agent.get_constant("x").value == 2

    def test_register_constants_batch_suffix_resolves_intra_batch_and_registry_collisions(self) -> None:
        agent = _make_agent(FakeLLMEngine(responses=[]))
        agent.register_constant(1, alias="x")

        agent.register_constants([2, 3], aliases=["x", "x"], name_collision_policy="suffix")

        assert agent.get_constant("x").value == 1
        assert agent.get_constant("x_0").value == 2
        assert agent.get_constant("x_1").value == 3

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

    def test_update_constant_description_raises_for_unknown_alias(self) -> None:
        agent = _make_agent(FakeLLMEngine(responses=[]))

        with pytest.raises(ToolAgentError):
            agent.update_constant_description("nonexistent", "new")

    def test_update_constant_description_raises_for_empty_description(self) -> None:
        agent = _make_agent(FakeLLMEngine(responses=[]))
        agent.register_constant(1, alias="x")

        with pytest.raises(ToolAgentError):
            agent.update_constant_description("x", "   ")

    def test_clear_constants(self) -> None:
        agent = _make_agent(FakeLLMEngine(responses=[]))
        agent.register_constant(1, alias="x")

        agent.clear_constants()

        assert agent.constants == {}


class TestScriptAgentPromptContextRendering:
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


class TestScriptAgentOneShotLifecycle:
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


class TestScriptAgentRegenRepair:
    def test_regen_repair_recovers_from_undefined_reference(self) -> None:
        engine = FakeLLMEngine(responses=[
            "result = add(a=1, b=undefined_name)\nreturn result",
            "result = add(a=1, b=2)\nreturn result",
        ])
        agent = _make_agent(engine, tools=[add], regeneration_limit=1, planning_rounds_limit=2)

        final = agent.invoke({"prompt": "add"})

        assert final.result == 3
        assert engine.call_count == 2
        feedback = "\n".join(m["content"] for m in engine.calls[1] if m["role"] != "system")
        assert "Your plan could not be used" in feedback
        assert "undefined_name" in feedback

    def test_regeneration_limit_zero_raises_immediately(self) -> None:
        engine = FakeLLMEngine(responses=["result = add(a=1, b=undefined_name)\nreturn result"])
        agent = _make_agent(engine, tools=[add], regeneration_limit=0, planning_rounds_limit=2)

        with pytest.raises(ToolAgentError, match="exhausted after 1 attempt"):
            agent.invoke({"prompt": "add"})

    def test_regeneration_limit_exhausted_after_two_attempts(self) -> None:
        engine = FakeLLMEngine(responses=[
            "result = add(a=1, b=undefined_name)\nreturn result",
            "result = add(a=1, b=other_undefined)\nreturn result",
        ])
        agent = _make_agent(engine, tools=[add], regeneration_limit=1, planning_rounds_limit=2)

        with pytest.raises(ToolAgentError):
            agent.invoke({"prompt": "add"})

        assert engine.call_count == 2


class TestScriptAgentCheckpointContinuation:
    def test_explicit_pause_triggers_continuation_with_correct_message_shape(self) -> None:
        engine = FakeLLMEngine(responses=[
            "x = add(a=1, b=2)\n# PAUSE",
            "y = add(a=x, b=1)\nreturn y",
        ])
        agent = _make_agent(engine, tools=[add], planning_rounds_limit=3)

        final = agent.invoke({"prompt": "add stuff"})

        assert final.result == 4
        round2 = engine.calls[1]
        assistant_msg = next(m["content"] for m in round2 if m["role"] == "assistant")
        user_msgs = "\n".join(m["content"] for m in round2 if m["role"] == "user")
        assert "x = add(a=1, b=2)" in assistant_msg
        assert "Cached values" in assistant_msg
        assert "Continue planning the rest of this task." in user_msgs

    def test_valid_if_cutoff_is_a_silent_continuation_with_no_note(self) -> None:
        engine = FakeLLMEngine(responses=[
            "x = add(a=1, b=2)\nif x > 0:\n    y = add(a=x, b=1)",
            "y = add(a=x, b=1)\nreturn y",
        ])
        agent = _make_agent(engine, tools=[add], planning_rounds_limit=3)

        final = agent.invoke({"prompt": "add stuff"})

        assert final.result == 4
        round2 = engine.calls[1]
        user_msgs = "\n".join(m["content"] for m in round2 if m["role"] == "user")
        assert "None" not in user_msgs
        assert "Continue planning the rest of this task." in user_msgs

    def test_malformed_if_cutoff_is_a_regen_repair_issue_not_a_continuation(self) -> None:
        engine = FakeLLMEngine(responses=[
            "if True:\n    y = 1",
            "y = add(a=1, b=2)\nreturn y",
        ])
        agent = _make_agent(engine, tools=[add], regeneration_limit=1, planning_rounds_limit=1)

        final = agent.invoke({"prompt": "add stuff"})

        # planning_rounds_limit=1 is what makes this a meaningful assertion:
        # if the malformed if-cutoff had been (incorrectly) treated as a
        # forced continuation into a genuine new planning round, think()'s
        # own ceiling check would raise here rather than let round 1's
        # retry loop repair it in place.
        assert final.result == 3
        assert engine.call_count == 2

    def test_resolution_failure_triggers_continuation_without_dispatching(self) -> None:
        engine = FakeLLMEngine(responses=[
            "x = 'hello'\ny = add(a=x + 1, b=2)\nreturn y",
            "return add(a=1, b=1)",
        ])
        agent = _make_agent(engine, tools=[add], planning_rounds_limit=3, context_enabled=True)

        final = agent.invoke({"prompt": "do something"})

        assert final.result == 2
        assert engine.call_count == 2
        round2_user = "\n".join(m["content"] for m in engine.calls[1] if m["role"] == "user")
        assert "could not be resolved" in round2_user
        record = agent.get_conversation()[-1]
        assert not any(s.identifier == "y" for s in record.statements)
        assert not any(s.identifier == "y" for s in record.failed_statements)
        assert record.tool_usage().registered_tool_calls == 1

    def test_execution_failure_triggers_continuation_preserving_the_other_success(self) -> None:
        engine = FakeLLMEngine(responses=[
            "a = add(a=1, b=1)\nb = fail_tool(x=1)\nreturn a",
            "return a",
        ])
        agent = _make_agent(
            engine, tools=[add, fail_tool], planning_rounds_limit=3, context_enabled=True,
        )

        final = agent.invoke({"prompt": "do something"})

        assert final.result == 2
        assert engine.call_count == 2
        round2_user = "\n".join(m["content"] for m in engine.calls[1] if m["role"] == "user")
        assert "simulated tool failure" in round2_user
        record = agent.get_conversation()[-1]
        assert any(s.identifier == "a" for s in record.statements)
        assert any(s.tool == "fail_tool" for s in record.failed_statements)
        assert record.tool_usage().registered_tool_calls == 2


class TestScriptAgentBudgets:
    def test_tool_calls_limit_enforced_silently_via_regen_repair(self) -> None:
        engine = FakeLLMEngine(responses=[
            "a = add(a=1, b=1)\nb = add(a=2, b=2)\nreturn a",
            "a = add(a=1, b=1)\nreturn a",
        ])
        agent = _make_agent(
            engine, tools=[add], tool_calls_limit=1, regeneration_limit=1, planning_rounds_limit=2,
        )

        final = agent.invoke({"prompt": "do something"})

        assert final.result == 2
        assert engine.call_count == 2
        system_msg = next(m["content"] for m in engine.calls[0] if m["role"] == "system")
        assert "tool_calls_limit" not in system_msg

    def test_final_round_model_authored_pause_is_regen_repair_not_continuation(self) -> None:
        engine = FakeLLMEngine(responses=[
            "x = add(a=1, b=1)\n# PAUSE",
            "return add(a=1, b=1)",
        ])
        agent = _make_agent(engine, tools=[add], regeneration_limit=1, planning_rounds_limit=1)

        final = agent.invoke({"prompt": "do something"})

        assert final.result == 2
        assert engine.call_count == 2

    def test_final_round_violation_exhausting_regeneration_limit_raises(self) -> None:
        engine = FakeLLMEngine(responses=["x = add(a=1, b=1)\n# PAUSE"])
        agent = _make_agent(engine, tools=[add], regeneration_limit=0, planning_rounds_limit=1)

        with pytest.raises(ToolAgentError, match="regeneration budget exhausted"):
            agent.invoke({"prompt": "do something"})

    def test_planning_rounds_limit_is_enforced_when_a_resolution_failure_forces_continuation(self) -> None:
        engine = FakeLLMEngine(responses=["x = 'hello'\ny = add(a=x + 1, b=2)\nreturn y"])
        agent = _make_agent(engine, tools=[add], planning_rounds_limit=1, tool_calls_limit=None)

        with pytest.raises(ToolAgentError, match="planning round budget exhausted"):
            agent.invoke({"prompt": "do something"})

        assert engine.call_count == 1

    def test_planning_rounds_limit_is_enforced_when_an_execution_failure_forces_continuation(self) -> None:
        engine = FakeLLMEngine(responses=["b = fail_tool(x=1)\nreturn b"])
        agent = _make_agent(engine, tools=[fail_tool], planning_rounds_limit=1, tool_calls_limit=None)

        with pytest.raises(ToolAgentError, match="planning round budget exhausted"):
            agent.invoke({"prompt": "do something"})

        assert engine.call_count == 1

    def test_forced_continuation_not_on_final_round_proceeds_normally(self) -> None:
        engine = FakeLLMEngine(responses=[
            "b = fail_tool(x=1)\nreturn b",
            "return add(a=1, b=1)",
        ])
        agent = _make_agent(
            engine, tools=[fail_tool, add], planning_rounds_limit=2, tool_calls_limit=None,
        )

        final = agent.invoke({"prompt": "do something"})

        assert final.result == 2
        assert engine.call_count == 2


class TestScriptAgentMutationSafety:
    def test_mutation_persists_across_rounds_within_one_invocation(self) -> None:
        engine = FakeLLMEngine(responses=[
            "x = K_MYLIST.append(99)\n# PAUSE",
            "return K_MYLIST",
        ])
        agent = _make_agent(engine, planning_rounds_limit=3)
        agent.register_constant([1, 2, 3], alias="mylist")

        final = agent.invoke({"prompt": "mutate"})

        assert final.result == [1, 2, 3, 99]

    def test_mutation_does_not_persist_to_a_fresh_invocation(self) -> None:
        engine = FakeLLMEngine(responses=[
            "x = K_MYLIST.append(99)\n# PAUSE",
            "return K_MYLIST",
            "return K_MYLIST",
        ])
        agent = _make_agent(engine, planning_rounds_limit=3)
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
        agent = _make_agent(engine, context_enabled=True, planning_rounds_limit=1)

        first = agent.invoke({"prompt": "make a list"})
        second = agent.invoke({"prompt": "mutate it"})
        third = agent.invoke({"prompt": "check again"})

        assert first.result == [1, 2, 3]
        assert second.result == [1, 2, 3, 99]
        assert third.result == [1, 2, 3]


class TestScriptAgentDispatch:
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
        # Registering `node` as a constant means the plan mutates a deep
        # copy (mutation-safety, tested separately in
        # TestScriptAgentMutationSafety), not this exact object -- the
        # success criterion here is that dispatch itself succeeds and
        # returns the right value, not that `node` itself was mutated.
        node = _Node()
        engine = FakeLLMEngine(responses=["return K_NODE.attach(obj=5)"])
        agent = _make_agent(engine)
        agent.register_constant(node, alias="node")

        final = agent.invoke({"prompt": "attach"})

        assert final.result == 5

    def test_builtin_call_with_keyword_arguments_dispatches_correctly(self) -> None:
        """
        No real Python builtin naturally has a keyword argument literally
        named `name` (the dispatcher's own internal parameter), so this
        confirms ordinary builtin keyword arguments dispatch correctly
        rather than reproducing an actual collision.
        """
        engine = FakeLLMEngine(responses=["return sorted([3, 1, 2], reverse=True)"])
        agent = _make_agent(engine)

        final = agent.invoke({"prompt": "sort"})

        assert final.result == [3, 2, 1]

    def test_dunder_attribute_call_raises_at_dispatch_level(self) -> None:
        """
        Defense-in-depth: parsing already rejects a dunder attribute call
        before a plan is ever dispatched; this tests attr_call_tool's own
        runtime gate directly. Tool.invoke() wraps the underlying
        ValueError in a ToolInvocationError (Tool.execute's own
        generic-exception wrapper), so that's the type surfacing here, not
        the bare ValueError.
        """
        with pytest.raises(ToolInvocationError, match="not available here"):
            attr_call_tool.invoke(
                {"obj": [1, 2], "method_name": "__class__", "args": (), "kwargs": {}}
            )

    def test_excluded_builtin_name_raises_at_dispatch_level(self) -> None:
        """
        Defense-in-depth mirror of the dunder-attribute-call test above, but
        for builtin_call_tool's own runtime gate: parsing/rewriting already
        keeps an excluded builtin name from ever reaching dispatch, but this
        tool's own check doesn't trust that upstream guarantee either.
        """
        with pytest.raises(ToolInvocationError, match="not available here"):
            builtin_call_tool.invoke({"name": "eval", "args": (), "kwargs": {}})


class TestScriptAgentPrepareActEdgeCases:
    def test_a_genuinely_empty_generation_completes_with_a_none_result(self) -> None:
        # No `# PAUSE`, no statements, no `return` -- parses to zero slots
        # and completes immediately with an inferred `None` result, exactly
        # like a plan that runs to the end without ever returning.
        engine = FakeLLMEngine(responses=[""])
        agent = _make_agent(engine)

        final = agent.invoke({"prompt": "do nothing"})

        assert final.result is None

    def test_prepare_is_a_no_op_when_pending_is_already_empty_and_continuation_is_pending(self) -> None:
        # Direct unit-level check of prepare()'s own top guard: entered with
        # pending already empty and continue_planning already True, it must
        # leave the task untouched rather than finalize it -- think() is
        # what actually regenerates the next round in that state, not
        # prepare() itself.
        agent = _make_agent(FakeLLMEngine(responses=[]))
        task = agent._initialize_task(turns=[], prompt="x", inputs={})
        task.pending = []
        task.continue_planning = True

        result = agent.prepare(task)

        assert result is task
        assert result.continue_planning is True
        assert result.complete is False
        assert result.resolved_args == []

    def test_argument_binding_mismatch_against_a_real_tool_signature_is_a_continuation_issue(self) -> None:
        # `c` is a legal identifier (passes parse-time reference validation)
        # but `add` has no such parameter -- the mismatch is only caught
        # once prepare() actually tries to bind it against add's real
        # signature.
        engine = FakeLLMEngine(responses=[
            "return add(a=1, c=2)",
            "return add(a=1, b=2)",
        ])
        agent = _make_agent(engine, tools=[add], planning_rounds_limit=2)

        final = agent.invoke({"prompt": "add"})

        assert final.result == 3
        round2_user = "\n".join(m["content"] for m in engine.calls[1] if m["role"] == "user")
        assert "parameter contract" in round2_user

    def test_execution_failure_of_a_builtin_call_reports_a_readable_failure_label(self) -> None:
        engine = FakeLLMEngine(responses=[
            "x = sorted(None)\nreturn x",
            "return sorted([2, 1])",
        ])
        agent = _make_agent(engine, planning_rounds_limit=2)

        final = agent.invoke({"prompt": "sort"})

        assert final.result == [1, 2]
        round2_user = "\n".join(m["content"] for m in engine.calls[1] if m["role"] == "user")
        assert "'sorted'" in round2_user

    def test_execution_failure_of_an_attribute_call_reports_a_readable_failure_label(self) -> None:
        engine = FakeLLMEngine(responses=[
            "x = K_MYLIST.nonexistent_method()\nreturn x",
            "return K_MYLIST",
        ])
        agent = _make_agent(engine, planning_rounds_limit=2)
        agent.register_constant([1, 2], alias="mylist")

        final = agent.invoke({"prompt": "call it"})

        assert final.result == [1, 2]
        round2_user = "\n".join(m["content"] for m in engine.calls[1] if m["role"] == "user")
        assert "K_MYLIST.nonexistent_method" in round2_user

    def test_a_batch_that_drains_without_a_return_finalizes_with_a_none_result(self) -> None:
        # One real dispatched call, no `# PAUSE`, no `return` -- the plan
        # naturally runs out of statements and must finalize right where
        # the batch drains, not leave the task hanging incomplete.
        engine = FakeLLMEngine(responses=["log_message(message='hi')"])
        agent = _make_agent(engine, tools=[log_message])

        final = agent.invoke({"prompt": "log only"})

        assert final.result is None


class TestScriptAgentAsyncLifecycle:
    def test_async_invoke_one_shot_lifecycle(self) -> None:
        engine = FakeLLMEngine(responses=["result = add(a=4, b=6)\nreturn result"])
        agent = _make_agent(engine, tools=[add])

        final = asyncio.run(agent.async_invoke({"prompt": "add 4 and 6"}))

        assert final.result == 10
        assert engine.call_count == 1

    def test_async_invoke_regen_repair_recovers_from_undefined_reference(self) -> None:
        engine = FakeLLMEngine(responses=[
            "result = add(a=1, b=undefined_name)\nreturn result",
            "result = add(a=1, b=2)\nreturn result",
        ])
        agent = _make_agent(engine, tools=[add], regeneration_limit=1, planning_rounds_limit=2)

        final = asyncio.run(agent.async_invoke({"prompt": "add"}))

        assert final.result == 3
        assert engine.call_count == 2

    def test_async_invoke_explicit_pause_triggers_continuation(self) -> None:
        engine = FakeLLMEngine(responses=[
            "x = add(a=1, b=2)\n# PAUSE",
            "y = add(a=x, b=1)\nreturn y",
        ])
        agent = _make_agent(engine, tools=[add], planning_rounds_limit=3)

        final = asyncio.run(agent.async_invoke({"prompt": "add stuff"}))

        assert final.result == 4
        assert engine.call_count == 2

    def test_async_invoke_planning_rounds_limit_is_enforced_on_forced_continuation(self) -> None:
        engine = FakeLLMEngine(responses=["x = 'hello'\ny = add(a=x + 1, b=2)\nreturn y"])
        agent = _make_agent(engine, tools=[add], planning_rounds_limit=1, tool_calls_limit=None)

        with pytest.raises(ToolAgentError, match="planning round budget exhausted"):
            asyncio.run(agent.async_invoke({"prompt": "do something"}))

        assert engine.call_count == 1

    def test_async_invoke_regeneration_limit_exhausted_raises(self) -> None:
        engine = FakeLLMEngine(responses=["result = add(a=1, b=undefined_name)\nreturn result"])
        agent = _make_agent(engine, tools=[add], regeneration_limit=0, planning_rounds_limit=2)

        with pytest.raises(ToolAgentError, match="exhausted after 1 attempt"):
            asyncio.run(agent.async_invoke({"prompt": "add"}))
