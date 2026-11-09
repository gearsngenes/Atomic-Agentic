from __future__ import annotations

import asyncio
from typing import Any

import pytest

from atomic_agentic.constants.agents import BUILTINS_PREAMBLE
from atomic_agentic.agents.scriptact import ScriptActAgent
from atomic_agentic.agents.tools import attr_call_tool, builtin_call_tool
from atomic_agentic.constants.agents import (
    ATTR_CALL_ALIAS,
    PY_BUILTIN_ALIAS,
    RETURN_ALIAS,
    RHS_ASSIGN_ALIAS,
    SUB_NAME_PREFIX,
)
from atomic_agentic.exceptions import (
    AgentInvocationError,
    ToolAgentError,
    ToolInvocationError,
    ToolRegistrationError,
)
from atomic_agentic.models.agents.tasks import ScriptActAgentTask
from atomic_agentic.tools import Tool
from ..fake_engines import FakeLLMEngine
from .conftest import make_script_agent


def add(a: int, b: int) -> int:
    """Returns a + b."""
    return a + b


def log_message(message: str) -> None:
    """Pretends to log a message; returns nothing meaningful."""
    return None


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
    """A bare agent: only the tools the test passes (the conftest maker
    pre-registers the shared math tools)."""
    return ScriptActAgent(
        name="tests", namespace="tests", description="ScriptActAgent under test.",
        llm_engine=engine, tools=tools, **kwargs,
    )


def _engine(*responses: str) -> FakeLLMEngine:
    return FakeLLMEngine(responses=list(responses))


def _new_task(agent: ScriptActAgent, prompt: str = "do the task") -> ScriptActAgentTask:
    task = agent.create_task({})
    task.user_prompt = prompt
    return task


def _start(agent: ScriptActAgent, prompt: str = "go") -> ScriptActAgentTask:
    """The invoke pipeline up to (not including) the think/prepare/act loop."""
    inputs = agent.filter_inputs({"prompt": prompt})
    task = agent.create_task(inputs)
    task = agent.summarize(task)
    return agent.pre_process(task)


def _round(agent: ScriptActAgent, task: ScriptActAgentTask) -> ScriptActAgentTask:
    task = agent.think(task)
    task = agent.prepare(task)
    return agent.act(task)


def _non_system(engine: FakeLLMEngine, call: int) -> str:
    return "\n".join(m["content"] for m in engine.calls[call] if m["role"] != "system")


class TestConstruction:
    def test_defaults(self) -> None:
        agent = _make_agent(_engine())

        assert agent.regeneration_limit == 5
        assert agent.tool_calls_limit is None
        assert agent.replanning_limit == 2
        assert agent.fail_fast is False
        assert agent.instructions is None

    def test_instructions_stored(self) -> None:
        agent = _make_agent(_engine(), instructions="Be terse.")
        assert agent.instructions == "Be terse."

    @pytest.mark.parametrize(
        "removed",
        [
            {"constants": [1]},
            {"constant_aliases": ["a"]},
            {"constant_descriptions": ["d"]},
            {"tool_instructions": "x"},
        ],
    )
    def test_removed_arguments_raise_type_error(self, removed: dict[str, Any]) -> None:
        with pytest.raises(TypeError):
            _make_agent(_engine(), **removed)

    def test_regeneration_limit_rejects_negative(self) -> None:
        with pytest.raises(ToolAgentError):
            _make_agent(_engine(), regeneration_limit=-1)

    def test_regeneration_limit_rejects_none(self) -> None:
        with pytest.raises(ToolAgentError):
            _make_agent(_engine(), regeneration_limit=None)  # type: ignore[arg-type]

    def test_both_budget_limits_default_legal(self) -> None:
        agent = _make_agent(_engine(), tool_calls_limit=None, replanning_limit=0)

        assert agent.tool_calls_limit is None
        assert agent.replanning_limit == 0

    def test_tool_calls_limit_rejects_negative(self) -> None:
        with pytest.raises(ToolAgentError):
            _make_agent(_engine(), tool_calls_limit=-1)

    def test_replanning_limit_rejects_negative(self) -> None:
        with pytest.raises(ToolAgentError):
            _make_agent(_engine(), replanning_limit=-1)

    def test_replanning_limit_rejects_none(self) -> None:
        with pytest.raises(ToolAgentError):
            _make_agent(_engine(), replanning_limit=None)  # type: ignore[arg-type]

    def test_fail_fast_rejects_non_bool(self) -> None:
        with pytest.raises(ToolAgentError):
            _make_agent(_engine(), fail_fast="yes")  # type: ignore[arg-type]

    def test_fail_fast_round_trips(self) -> None:
        assert _make_agent(_engine(), fail_fast=True).fail_fast is True

    def test_tool_concurrency_limit_defaults_to_none_and_round_trips(self) -> None:
        assert _make_agent(_engine()).tool_concurrency_limit is None
        assert _make_agent(_engine(), tool_concurrency_limit=2).tool_concurrency_limit == 2

    def test_tool_concurrency_limit_rejects_zero(self) -> None:
        with pytest.raises(ToolAgentError):
            _make_agent(_engine(), tool_concurrency_limit=0)

    def test_to_dict_includes_fail_fast_and_replanning_limit(self) -> None:
        d = _make_agent(_engine(), fail_fast=True, replanning_limit=3).to_dict()

        assert d["fail_fast"] is True
        assert d["replanning_limit"] == 3


class TestToolIdNaming:
    def test_registering_tool_colliding_with_a_real_builtin_raises(self) -> None:
        with pytest.raises(ToolRegistrationError):
            _make_agent(_engine()).register_tool(add, alias="len")

    @pytest.mark.parametrize(
        "sentinel", [RHS_ASSIGN_ALIAS, RETURN_ALIAS, PY_BUILTIN_ALIAS, ATTR_CALL_ALIAS],
    )
    def test_registering_tool_under_reserved_sentinel_raises(self, sentinel: str) -> None:
        with pytest.raises(ToolRegistrationError):
            _make_agent(_engine()).register_tool(add, alias=sentinel)

    @pytest.mark.parametrize("alias", ["for", "class", "lambda", "while"])
    def test_keyword_alias_is_rejected(self, alias: str) -> None:
        agent = _make_agent(_engine())

        with pytest.raises(ToolRegistrationError, match="keyword"):
            agent.register_tool(add, alias=alias)
        assert not agent.has_tool(alias)

    def test_keyword_alias_rejected_through_register_tools_and_constructor(self) -> None:
        agent = _make_agent(_engine())
        with pytest.raises(ToolRegistrationError, match="keyword"):
            agent.register_tools([add], aliases=["while"])

        wrapped = Tool(function=add, name="import", namespace="tests", description="Adds.")
        with pytest.raises(ToolRegistrationError, match="keyword"):
            _make_agent(_engine(), tools=[wrapped])


class TestRegistrationEdgeCases:
    def test_unknown_collision_policy_raises(self) -> None:
        with pytest.raises(ToolRegistrationError):
            _make_agent(_engine()).register_tool(add, name_collision_policy="bogus")  # type: ignore[arg-type]

    def test_non_identifier_alias_raises(self) -> None:
        with pytest.raises(ToolRegistrationError):
            _make_agent(_engine()).register_tool(add, alias="not valid!")

    def test_register_tool_accepts_an_already_wrapped_invokable_directly(self) -> None:
        agent = _make_agent(_engine())
        wrapped = Tool(function=add, name="add_tool", namespace="tests", description="Adds.")

        agent.register_tool(wrapped)

        assert agent.get_tool("add_tool") is wrapped

    def test_register_tool_wraps_toolify_failure_as_registration_error(self) -> None:
        with pytest.raises(ToolRegistrationError):
            _make_agent(_engine()).register_tool(_NoNameCallable())

    def test_register_tool_rejects_an_unsupported_component_type(self) -> None:
        with pytest.raises(ToolRegistrationError):
            _make_agent(_engine()).register_tool(123)  # type: ignore[arg-type]

    def test_register_tool_duplicate_raises_by_default(self) -> None:
        with pytest.raises(ToolRegistrationError):
            _make_agent(_engine(), tools=[add]).register_tool(add)

    def test_register_tool_duplicate_with_skip_policy_returns_false(self) -> None:
        agent = _make_agent(_engine(), tools=[add])
        assert agent.register_tool(add, name_collision_policy="skip") is False

    def test_register_tool_duplicate_with_replace_policy_overwrites(self) -> None:
        agent = _make_agent(_engine(), tools=[add])
        wrapped = Tool(function=add, name="add", namespace="tests", description="A different add.")

        assert agent.register_tool(wrapped, name_collision_policy="replace") is True
        assert agent.get_tool("add") is wrapped

    def test_register_tools_alias_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            _make_agent(_engine()).register_tools([add], aliases=["a", "b"])

    def test_register_tools_intra_batch_duplicate_raises(self) -> None:
        with pytest.raises(ToolRegistrationError):
            _make_agent(_engine()).register_tools([add, add])

    def test_register_tools_collision_against_existing_toolbox_skip(self) -> None:
        agent = _make_agent(_engine(), tools=[add])
        original = agent.get_tool("add")

        assert agent.register_tools([add], name_collision_policy="skip") is False
        assert agent.get_tool("add") is original

    def test_get_tool_raises_for_unknown_id(self) -> None:
        with pytest.raises(ToolAgentError):
            _make_agent(_engine()).get_tool("nonexistent")

    def test_list_has_remove_clear_tools_roundtrip(self) -> None:
        agent = _make_agent(_engine(), tools=[add])

        assert agent.has_tool("add")
        assert "add" in agent.list_tools()
        assert agent.remove_tool("add") is True
        assert agent.remove_tool("add") is False
        assert not agent.has_tool("add")

        agent.register_tool(add)
        agent.clear_tools()
        assert agent.list_tools() == {}


class TestPromptContextRendering:
    def test_actions_context_renders_alias_distinct_from_the_tool_name(self) -> None:
        agent = _make_agent(_engine())
        agent.register_tool(add, alias="plus")

        rendered = agent.actions_context()

        assert rendered.startswith("plus(")
        assert "def add" not in rendered

    def test_system_message_includes_excluded_builtins(self) -> None:
        agent = _make_agent(_engine())

        messages = agent.render_task(_new_task(agent))

        system_msg = next(m["content"] for m in messages if m["role"] == "system")
        assert "==== AVAILABLE TOOLS [START] ====" in system_msg
        assert system_msg.index(agent.actions_context()) < system_msg.index(BUILTINS_PREAMBLE)
        assert "eval" in BUILTINS_PREAMBLE and BUILTINS_PREAMBLE in system_msg

    def test_instructions_are_framed_in_the_system_message(self) -> None:
        agent = _make_agent(_engine(), instructions="Always say please.")

        messages = agent.render_task(_new_task(agent))

        system_msg = next(m["content"] for m in messages if m["role"] == "system")
        assert "USER-DEFINED INSTRUCTIONS" in system_msg
        assert "Always say please." in system_msg


class TestSeedTask:
    def test_keyword_named_variable_is_rejected(self) -> None:
        agent = make_script_agent(_engine())
        agent.get_conversation().remember("class", 1)

        with pytest.raises(AgentInvocationError, match="'class' is a Python keyword"):
            agent.create_task({})

    def test_tool_id_named_variable_is_rejected(self) -> None:
        agent = make_script_agent(_engine())
        agent.get_conversation().remember("add", 1)

        with pytest.raises(AgentInvocationError, match="'add' is a registered tool id"):
            agent.create_task({})

    def test_every_clash_is_named_in_one_error(self) -> None:
        agent = make_script_agent(_engine())
        conv = agent.get_conversation()
        conv.remember("while", 1)
        conv.remember("multiply", 2)

        with pytest.raises(AgentInvocationError) as excinfo:
            agent.create_task({})

        assert "'while'" in str(excinfo.value)
        assert "'multiply'" in str(excinfo.value)

    def test_builtin_named_variable_is_allowed(self) -> None:
        agent = make_script_agent(_engine())
        agent.get_conversation().remember("len", 3)

        task = agent.create_task({})

        assert task.context["len"].value == 3

    def test_clash_raises_before_any_model_call(self) -> None:
        engine = _engine("return 1")
        agent = make_script_agent(engine)
        agent.get_conversation().remember("add", 1)

        with pytest.raises(AgentInvocationError):
            agent.invoke({"prompt": "go"})

        assert engine.call_count == 0

    def test_task_result_entries_are_still_seeded(self) -> None:
        engine = _engine("return [1, 2]")
        agent = make_script_agent(engine, record_enabled=True)
        agent.invoke({"prompt": "first"})

        task = agent.create_task({})

        assert task.context["task_result_0"].value == [1, 2]
        assert task.context["task_result_0"].readonly is True


class TestRendering:
    def test_round_one_is_a_single_user_message_with_the_plan_directive(self) -> None:
        agent = make_script_agent(_engine())
        task = _new_task(agent, "sum these")

        messages = agent.render_task(task)

        assert [m["role"] for m in messages] == ["system", "user"]
        assert "sum these" in messages[-1]["content"]
        assert messages[-1]["content"].endswith("Write a plan to accomplish this task now.")
        assert "STEPS" not in messages[-1]["content"]
        assert agent._render_steps(task) == ""

    @pytest.mark.parametrize("limit,expected", [(1, "at most 1 tool call total."), (4, "at most 4 tool calls total.")])
    def test_round_one_states_the_total_tool_call_budget(self, limit: int, expected: str) -> None:
        agent = make_script_agent(_engine(), tool_calls_limit=limit)

        directive = agent._render_directive(_new_task(agent))

        assert expected in directive

    def test_round_one_has_no_budget_line_without_a_limit(self) -> None:
        agent = make_script_agent(_engine())
        assert "tool call" not in agent._render_directive(_new_task(agent))

    def test_repair_round_has_steps_block_and_remaining_budget(self) -> None:
        engine = _engine(
            "a = add(x=1, y=1)\nb = fail_tool()\nreturn a",
            "return a",
        )
        agent = make_script_agent(engine, tool_calls_limit=5, replanning_limit=1)

        agent.invoke({"prompt": "go"})

        repair = engine.calls[1]
        assert [m["role"] for m in repair] == ["system", "user", "assistant", "user"]
        steps, directive = repair[2]["content"], repair[3]["content"]
        assert "STEPS" in steps
        assert "# WORK COMPLETED SO FAR:" in steps
        assert "add(" in steps
        assert "# THIS BATCH FAILED:" in steps
        assert "intentional failure" in steps
        assert directive.startswith("Continue planning the rest of this task")
        assert directive.endswith("Tool calls remaining: 3 of 5.")
        assert "Write a plan" not in repair[1]["content"]

    def test_repair_steps_show_only_the_failed_slice(self) -> None:
        engine = _engine(
            "a = fail_tool()\nreturn a",
            "c = fail_tool()\nreturn c",
            "return 1",
        )
        agent = make_script_agent(engine, replanning_limit=2)

        agent.invoke({"prompt": "go"})

        third = engine.calls[2][2]["content"]
        assert "c = fail_tool()" in third
        assert "a = fail_tool()" not in third

    def test_repair_round_with_nothing_completed_says_so(self) -> None:
        engine = _engine("return fail_tool()", "return 1")
        agent = make_script_agent(engine, replanning_limit=1)

        agent.invoke({"prompt": "go"})

        assert "(nothing completed yet)" in engine.calls[1][2]["content"]

    def test_bound_values_reach_the_repair_round_through_the_context_block(self) -> None:
        engine = _engine("a = add(x=20, y=22)\nb = fail_tool()\nreturn a", "return a")
        agent = make_script_agent(engine, replanning_limit=1)

        final = agent.invoke({"prompt": "go"})

        assert final.result == 42
        assert "name: a" in engine.calls[1][1]["content"]
        assert "42" in engine.calls[1][1]["content"]


class TestProcessGenerationOutput:
    def test_valid_plan_compiles_and_advances_batch_counter(self) -> None:
        agent = make_script_agent(_engine())
        task = _new_task(agent)

        result = agent._process_generation_output("a = add(x=1, y=2)\nreturn a", task)

        assert isinstance(result, list)
        assert sum(len(batch) for batch in result) == 2
        assert task.batch_counter == len(result)

    def test_parse_error_is_returned_alone(self) -> None:
        agent = make_script_agent(_engine())

        result = agent._process_generation_output("a = add(x=1,", _new_task(agent))

        assert isinstance(result, str)
        assert not result.startswith("1. ")

    def test_undefined_name_is_flagged_but_a_context_name_is_known(self) -> None:
        agent = make_script_agent(_engine())
        agent.get_conversation().remember("known", 7)
        task = _new_task(agent)

        bad = agent._process_generation_output("a = add(x=missing, y=1)\nreturn a", task)
        good = agent._process_generation_output("a = add(x=known, y=1)\nreturn a", task)

        assert isinstance(bad, str) and "missing" in bad
        assert isinstance(good, list)

    def test_task_result_names_are_known(self) -> None:
        engine = _engine("return 5")
        agent = make_script_agent(engine, record_enabled=True)
        agent.invoke({"prompt": "first"})

        result = agent._process_generation_output("return task_result_0", _new_task(agent))

        assert isinstance(result, list)

    def test_readonly_target_is_rejected(self) -> None:
        agent = make_script_agent(_engine())
        agent.get_conversation().remember("policy", 1, readonly=True)

        result = agent._process_generation_output("policy = add(x=1, y=2)\nreturn policy", _new_task(agent))

        assert isinstance(result, str)
        assert "read-only context variable" in result

    def test_task_result_target_is_rejected_as_readonly(self) -> None:
        engine = _engine("return 5")
        agent = make_script_agent(engine, record_enabled=True)
        agent.invoke({"prompt": "first"})

        result = agent._process_generation_output("task_result_0 = 1\nreturn 1", _new_task(agent))

        assert isinstance(result, str)
        assert "task_result_0" in result

    def test_tool_id_target_is_rejected(self) -> None:
        agent = make_script_agent(_engine())

        result = agent._process_generation_output("add = 3\nreturn add", _new_task(agent))

        assert isinstance(result, str)
        assert "registered tool id" in result

    def test_writable_context_target_is_allowed(self) -> None:
        agent = make_script_agent(_engine())
        agent.get_conversation().remember("total", 0)

        result = agent._process_generation_output("total = add(x=1, y=2)\nreturn total", _new_task(agent))

        assert isinstance(result, list)

    def test_dry_bind_mismatch_is_reported(self) -> None:
        agent = make_script_agent(_engine())

        result = agent._process_generation_output("a = add(x=1, z=2)\nreturn a", _new_task(agent))

        assert isinstance(result, str)
        assert "do not fit its signature" in result
        assert "'add'" in result

    def test_every_issue_is_numbered_in_one_string(self) -> None:
        agent = make_script_agent(_engine())
        agent.get_conversation().remember("policy", 1, readonly=True)
        script = (
            "policy = add(x=1, y=2)\n"
            "b = add(x=1, z=2)\n"
            "c = add(x=missing, y=2)\n"
            "return c"
        )

        result = agent._process_generation_output(script, _new_task(agent))

        assert isinstance(result, str)
        lines = result.splitlines()
        assert [line.split(".")[0] for line in lines] == ["1", "2", "3"]
        assert "read-only context variable" in lines[0]
        assert "signature" in result and "missing" in result

    def test_remaining_budget_counts_work_already_spent(self) -> None:
        agent = make_script_agent(_engine(), tool_calls_limit=2)
        script = "a = add(x=1, y=1)\nb = add(x=2, y=2)\nreturn a"
        assert isinstance(agent._process_generation_output(script, _new_task(agent)), list)

        spent = _new_task(agent)
        spent.completed.extend(agent._process_generation_output("a = add(x=1, y=1)", spent)[0])
        over = agent._process_generation_output(script, spent)

        assert isinstance(over, str)
        assert "exceeding the configured limit" in over


class TestLifecycle:
    def test_registered_tool_call_and_return(self) -> None:
        engine = _engine("result = add(x=4, y=6)\nreturn result")
        agent = make_script_agent(engine)

        final = agent.invoke({"prompt": "add 4 and 6"})

        assert final.result == 10
        assert engine.call_count == 1

    def test_bare_call_and_literal_return(self) -> None:
        engine = _engine("log_message(message='hi')\nreturn 42")
        agent = make_script_agent(engine, record_enabled=True)
        agent.register_tool(log_message)

        final = agent.invoke({"prompt": "log then return"})

        assert final.result == 42
        record = agent.get_conversation().records[-1]
        assert any(s.tool == "log_message" for s in record.statements)

    def test_results_and_sub_temporaries_become_context_variables(self) -> None:
        agent = make_script_agent(_engine("y = add(x=add(x=1, y=2), y=3)\nreturn y"))
        task = _start(agent)

        while not task.complete:
            task = _round(agent, task)

        sub = f"{SUB_NAME_PREFIX}0"
        assert task.context["y"].value == 6
        assert task.context[sub].value == 3
        assert {"y", sub} <= task.changed_context
        assert task.generated_response == 6

    def test_writable_context_variable_is_updated_not_replaced(self) -> None:
        agent = make_script_agent(_engine("total = add(x=2, y=3)\nreturn total"))
        agent.get_conversation().remember("total", 0, "Running total", render_mode="r")
        task = _start(agent)

        while not task.complete:
            task = _round(agent, task)

        assert task.context["total"].value == 5
        assert task.context["total"].description == "Running total"
        assert task.context["total"].render_mode == "r"

    def test_later_statements_read_earlier_results_across_batches(self) -> None:
        engine = _engine("a = add(x=1, y=1)\nb = add(x=a, y=10)\nreturn b")
        agent = make_script_agent(engine)

        assert agent.invoke({"prompt": "chain"}).result == 12
        assert engine.call_count == 1

    def test_hidden_task_result_is_readable_from_a_script(self) -> None:
        engine = _engine("return [1, 2, 3]", "return len(task_result_0)")
        agent = make_script_agent(engine, record_enabled=True)

        agent.invoke({"prompt": "make a list"})
        second = agent.invoke({"prompt": "count it"})

        assert second.result == 3
        # the value is hidden in the prompt but resolvable in the script
        assert "[1, 2, 3]" not in _non_system(engine, 1).split("name: task_result_0")[-1].split("\n\n")[0]

    def test_context_variable_can_feed_a_tool_call(self) -> None:
        agent = make_script_agent(_engine("return add(x=base, y=1)"))
        agent.get_conversation().remember("base", 41)

        assert agent.invoke({"prompt": "go"}).result == 42

    def test_empty_generation_returns_none_from_act(self) -> None:
        agent = make_script_agent(_engine(""))
        task = _start(agent)

        task = _round(agent, task)

        assert task.complete is True
        assert task.generated_response is None

    def test_empty_generation_invoke_result_is_none(self) -> None:
        assert make_script_agent(_engine("")).invoke({"prompt": "nothing"}).result is None

    def test_drained_plan_without_return_returns_none_from_act(self) -> None:
        agent = make_script_agent(_engine("a = add(x=1, y=1)"))
        task = _start(agent)

        task = _round(agent, task)

        assert task.complete is True
        assert task.generated_response is None
        assert task.context["a"].value == 2

    def test_prepare_with_nothing_pending_only_resets_and_does_not_finalize(self) -> None:
        agent = make_script_agent(_engine())
        task = _new_task(agent)
        task.resolved_args = [{"stale": 1}]

        result = agent.prepare(task)

        assert result is task
        assert task.resolved_args == []
        assert task.complete is False

    def test_act_is_a_no_op_while_a_repair_is_pending(self) -> None:
        agent = make_script_agent(_engine())
        task = _new_task(agent)
        task.needs_repair = True

        result = agent.act(task)

        assert result is task
        assert task.complete is False

    def test_repair_is_granted_by_prepare_on_a_resolution_failure(self) -> None:
        agent = make_script_agent(_engine("x = 'hello'\ny = add(x=x + 1, y=2)\nreturn y"), replanning_limit=1)
        task = agent.act(agent.prepare(agent.think(_start(agent))))
        assert task.needs_repair is False and len(task.pending) == 2

        task = agent.prepare(task)

        assert task.needs_repair is True
        assert task.repair_rounds_used == 1
        assert task.pending == []
        assert task.resolved_args == []
        assert any(s.identifier == "y" for s in task.failed_statements)
        # act must not finalize the abandoned round
        assert agent.act(task).complete is False

    def test_repair_is_granted_by_act_on_an_execution_failure(self) -> None:
        agent = make_script_agent(_engine("a = add(x=1, y=1)\nb = fail_tool()\nreturn a"), replanning_limit=1)
        task = agent.prepare(agent.think(_start(agent)))

        task = agent.act(task)

        assert task.needs_repair is True
        assert task.repair_rounds_used == 1
        assert task.pending == []
        assert task.complete is False
        assert [s.identifier for s in task.completed] == ["a"]
        assert [s.tool for s in task.failed_statements] == ["fail_tool"]

    def test_successes_of_a_partly_failed_batch_stay_bound(self) -> None:
        agent = make_script_agent(_engine("a = add(x=1, y=1)\nb = fail_tool()\nreturn a"), replanning_limit=1)
        task = agent.act(agent.prepare(agent.think(_start(agent))))

        assert task.context["a"].value == 2
        assert "b" not in task.context
        assert "a" in task.changed_context

    def test_think_clears_needs_repair_after_regenerating(self) -> None:
        agent = make_script_agent(_engine("return fail_tool()", "return 1"), replanning_limit=1)
        task = agent.act(agent.prepare(agent.think(_start(agent))))
        assert task.needs_repair is True

        task = agent.think(task)

        assert task.needs_repair is False
        assert len(task.pending) == 1

    def test_resolution_failure_triggers_a_repair_round(self) -> None:
        engine = _engine(
            "x = 'hello'\ny = add(x=x + 1, y=2)\nreturn y",
            "return add(x=1, y=1)",
        )
        agent = make_script_agent(engine, replanning_limit=1, record_enabled=True)

        final = agent.invoke({"prompt": "do something"})

        assert final.result == 2
        assert engine.call_count == 2
        round2 = _non_system(engine, 1)
        assert "THIS BATCH FAILED" in round2
        record = agent.get_conversation().records[-1]
        assert record.repair_rounds_used == 1
        assert not any(s.identifier == "y" for s in record.statements)
        assert any(s.identifier == "y" for s in record.failed_statements)

    def test_execution_failure_repair_preserves_the_other_success(self) -> None:
        engine = _engine("a = add(x=1, y=1)\nb = fail_tool()\nreturn a", "return a")
        agent = make_script_agent(engine, replanning_limit=1, record_enabled=True)

        final = agent.invoke({"prompt": "do something"})

        assert final.result == 2
        assert "intentional failure" in _non_system(engine, 1)
        record = agent.get_conversation().records[-1]
        assert record.repair_rounds_used == 1
        assert any(s.identifier == "a" for s in record.statements)
        assert any(s.tool == "fail_tool" for s in record.failed_statements)

    def test_argument_binding_mismatch_at_prepare_triggers_a_repair_round(self) -> None:
        # check_call_bindings catches the shape mistake at plan time, so this
        # reaches prepare only via a value that resolves but does not bind:
        # a **kwargs unpack of unknown arity is skipped by the dry bind.
        engine = _engine("opts = {'z': 1}\nreturn add(**opts)", "return add(x=1, y=1)")
        agent = make_script_agent(engine, replanning_limit=1)

        final = agent.invoke({"prompt": "add"})

        assert final.result == 2
        assert "parameter contract" in _non_system(engine, 1)

    def test_builtin_execution_failure_reports_a_readable_label(self) -> None:
        engine = _engine("x = sorted(None)\nreturn x", "return sorted([2, 1])")
        agent = make_script_agent(engine, replanning_limit=1)

        final = agent.invoke({"prompt": "sort"})

        assert final.result == [1, 2]
        assert "sorted(None)" in _non_system(engine, 1)

    def test_attribute_call_failure_reports_a_readable_label(self) -> None:
        engine = _engine("x = items.nonexistent_method()\nreturn x", "return items")
        agent = make_script_agent(engine, replanning_limit=1)
        agent.get_conversation().remember("items", [1, 2])

        final = agent.invoke({"prompt": "call it"})

        assert final.result == [1, 2]
        assert "items.nonexistent_method" in _non_system(engine, 1)


class TestFailureBudgets:
    def test_fail_fast_raises_on_resolution_failure_without_repair(self) -> None:
        engine = _engine("x = 'hello'\ny = add(x=x + 1, y=2)\nreturn y")
        agent = make_script_agent(engine, replanning_limit=2, fail_fast=True)

        with pytest.raises(ToolAgentError, match="batch could not be resolved.*fail_fast=True"):
            agent.invoke({"prompt": "do something"})

        assert engine.call_count == 1

    def test_fail_fast_reraises_a_lone_tool_invocation_error(self) -> None:
        engine = _engine("return fail_tool()")
        agent = make_script_agent(engine, replanning_limit=2, fail_fast=True)

        with pytest.raises(ToolInvocationError, match="intentional failure"):
            agent.invoke({"prompt": "do something"})

        assert engine.call_count == 1

    def test_several_execution_failures_raise_a_tool_agent_error(self) -> None:
        engine = _engine("a = fail_tool()\nb = fail_tool()\nreturn a")
        agent = make_script_agent(engine, fail_fast=True)

        with pytest.raises(ToolAgentError, match="batch execution failed.*fail_fast=True") as excinfo:
            agent.invoke({"prompt": "do something"})

        assert not isinstance(excinfo.value, ToolInvocationError)

    def test_replanning_limit_zero_raises_on_resolution_failure(self) -> None:
        engine = _engine("x = 'hello'\ny = add(x=x + 1, y=2)\nreturn y")
        agent = make_script_agent(engine, replanning_limit=0)

        with pytest.raises(ToolAgentError, match="repair budget exhausted"):
            agent.invoke({"prompt": "do something"})

        assert engine.call_count == 1

    def test_replanning_limit_zero_raises_on_execution_failure(self) -> None:
        engine = _engine("return fail_tool()")
        agent = make_script_agent(engine, replanning_limit=0)

        with pytest.raises(ToolInvocationError, match="intentional failure"):
            agent.invoke({"prompt": "do something"})

        assert engine.call_count == 1

    def test_budget_exhausted_after_the_granted_repairs(self) -> None:
        engine = _engine("return fail_tool()", "a = fail_tool()\nb = fail_tool()\nreturn a")
        agent = make_script_agent(engine, replanning_limit=1)

        with pytest.raises(ToolAgentError, match="batch execution failed.*repair budget exhausted"):
            agent.invoke({"prompt": "do something"})

        assert engine.call_count == 2

    def test_repair_within_budget_proceeds_normally(self) -> None:
        engine = _engine("return fail_tool()", "return add(x=1, y=1)")
        agent = make_script_agent(engine, replanning_limit=2)

        assert agent.invoke({"prompt": "do something"}).result == 2
        assert engine.call_count == 2


class TestRegenerationAndBudgets:
    def test_regen_recovers_from_undefined_reference(self) -> None:
        engine = _engine(
            "result = add(x=1, y=undefined_name)\nreturn result",
            "result = add(x=1, y=2)\nreturn result",
        )
        agent = make_script_agent(engine, regeneration_limit=1)

        final = agent.invoke({"prompt": "add"})

        assert final.result == 3
        assert engine.call_count == 2
        feedback = _non_system(engine, 1)
        assert "Your plan could not be used" in feedback
        assert "undefined_name" in feedback

    def test_regeneration_limit_zero_raises_immediately(self) -> None:
        engine = _engine("result = add(x=1, y=undefined_name)\nreturn result")
        agent = make_script_agent(engine, regeneration_limit=0)

        with pytest.raises(ToolAgentError, match="exhausted after 1 attempt"):
            agent.invoke({"prompt": "add"})

    def test_regeneration_limit_exhausted_after_two_attempts(self) -> None:
        engine = _engine(
            "result = add(x=1, y=undefined_name)\nreturn result",
            "result = add(x=1, y=other_undefined)\nreturn result",
        )
        agent = make_script_agent(engine, regeneration_limit=1)

        with pytest.raises(ToolAgentError):
            agent.invoke({"prompt": "add"})

        assert engine.call_count == 2

    def test_conditional_statement_is_fed_back_as_regeneration_feedback(self) -> None:
        engine = _engine("if True:\n    y = 1", "y = add(x=1, y=2)\nreturn y")
        agent = make_script_agent(engine, regeneration_limit=1)

        assert agent.invoke({"prompt": "add stuff"}).result == 3
        assert engine.call_count == 2

    def test_dry_bind_mistake_is_regenerated_not_repaired(self) -> None:
        engine = _engine("return add(x=1, z=2)", "return add(x=1, y=2)")
        agent = make_script_agent(engine, regeneration_limit=1, replanning_limit=0)

        assert agent.invoke({"prompt": "add"}).result == 3
        assert "do not fit its signature" in _non_system(engine, 1)

    def test_tool_calls_limit_enforced_via_regen_feedback(self) -> None:
        engine = _engine(
            "a = add(x=1, y=1)\nb = add(x=2, y=2)\nreturn a",
            "a = add(x=1, y=1)\nreturn a",
        )
        agent = make_script_agent(engine, tool_calls_limit=1, regeneration_limit=1)

        assert agent.invoke({"prompt": "do something"}).result == 2
        assert "exceeding the configured limit" in _non_system(engine, 1)

    def test_tool_concurrency_limit_splits_a_plan_into_multiple_batches(self) -> None:
        engine = _engine("a = add(x=1, y=1)\nb = add(x=2, y=2)\nreturn a")
        agent = make_script_agent(engine, tool_concurrency_limit=1, record_enabled=True)

        assert agent.invoke({"prompt": "do something"}).result == 2
        record = agent.get_conversation().records[-1]
        a_stmt = next(s for s in record.statements if s.identifier == "a")
        b_stmt = next(s for s in record.statements if s.identifier == "b")
        assert a_stmt.batch_index != b_stmt.batch_index


class TestMutationByReference:
    def test_method_call_mutates_the_context_variable_in_place(self) -> None:
        engine = _engine("x = items.append(99)\nreturn items")
        agent = make_script_agent(engine)
        agent.get_conversation().remember("items", [1, 2, 3])

        final = agent.invoke({"prompt": "mutate"})

        assert final.result == [1, 2, 3, 99]
        # the conversation's variable is shared by reference (documented)
        assert agent.get_conversation().context["items"].value == [1, 2, 3, 99]

    def test_mutation_persists_across_repair_rounds_within_one_invocation(self) -> None:
        engine = _engine("x = items.append(99)\nb = fail_tool()\nreturn items", "return items")
        agent = make_script_agent(engine, replanning_limit=1)
        agent.get_conversation().remember("items", [1, 2, 3])

        assert agent.invoke({"prompt": "mutate"}).result == [1, 2, 3, 99]

    def test_task_result_mutation_does_not_persist_across_invocations(self) -> None:
        engine = _engine(
            "return [1, 2, 3]",
            "x = task_result_0.append(99)\nreturn task_result_0",
            "return task_result_0",
        )
        agent = make_script_agent(engine, record_enabled=True)

        first = agent.invoke({"prompt": "make a list"})
        second = agent.invoke({"prompt": "mutate it"})
        third = agent.invoke({"prompt": "check again"})

        assert first.result == [1, 2, 3]
        assert second.result == [1, 2, 3, 99]
        assert third.result == [1, 2, 3]


class TestDispatch:
    def test_registered_tool_with_ordinary_kwargs(self) -> None:
        assert make_script_agent(_engine("return add(x=2, y=3)")).invoke({"prompt": "add"}).result == 5

    def test_approved_builtin_dispatch(self) -> None:
        assert make_script_agent(_engine("return len([1, 2, 3])")).invoke({"prompt": "count"}).result == 3

    def test_attribute_call_on_a_context_variable(self) -> None:
        agent = make_script_agent(_engine("return mydict.get('a')"))
        agent.get_conversation().remember("mydict", {"a": 1})

        assert agent.invoke({"prompt": "get a"}).result == 1

    def test_attribute_call_dispatches_when_the_method_has_a_keyword_named_obj(self) -> None:
        # The real target method's own kwarg is literally named `obj`, which
        # would collide with attr_call_tool's own parameter if splatted.
        agent = make_script_agent(_engine("return node.attach(obj=5)"))
        agent.get_conversation().remember("node", _Node())

        assert agent.invoke({"prompt": "attach"}).result == 5

    def test_builtin_call_with_keyword_arguments_dispatches_correctly(self) -> None:
        engine = _engine("return sorted([3, 1, 2], reverse=True)")
        assert make_script_agent(engine).invoke({"prompt": "sort"}).result == [3, 2, 1]

    def test_dunder_attribute_call_raises_at_dispatch_level(self) -> None:
        with pytest.raises(ToolInvocationError, match="not available here"):
            attr_call_tool.invoke(
                {"obj": [1, 2], "method_name": "__class__", "args": (), "kwargs": {}}
            )

    def test_excluded_builtin_name_raises_at_dispatch_level(self) -> None:
        with pytest.raises(ToolInvocationError, match="not available here"):
            builtin_call_tool.invoke({"name": "eval", "args": (), "kwargs": {}})

    def test_dispatch_breakdown_counts_all_categories(self) -> None:
        engine = _engine(
            "c = add(x=1, y=2)\n"
            "d = len([1, 2, 3])\n"
            "e = mydict.get('a')\n"
            "f = c + d\n"
            "return f"
        )
        agent = make_script_agent(engine, record_enabled=True)
        agent.get_conversation().remember("mydict", {"a": 1})

        assert agent.invoke({"prompt": "combine"}).result == 6
        assert engine.call_count == 1
        breakdown = agent.get_conversation().records[-1].dispatch_breakdown()
        assert breakdown.registered_tool_calls == 1
        assert breakdown.builtin_calls == 1
        assert breakdown.attribute_calls == 1
        assert breakdown.rhs_assignment_count == 1
        assert breakdown.binop_count == 1


class TestPersistence:
    def test_response_to_context_creates_the_declared_variable(self) -> None:
        agent = make_script_agent(
            _engine("return add(x=2, y=3)"), record_enabled=True, response_to_context="answer",
        )

        agent.invoke({"prompt": "go"})

        assert agent.get_conversation().context["answer"].value == 5

    def test_undeclared_result_names_and_sub_temporaries_are_not_persisted(self) -> None:
        agent = make_script_agent(
            _engine("tmp = add(x=add(x=1, y=1), y=1)\nreturn tmp"),
            record_enabled=True, response_to_context="answer",
        )

        agent.invoke({"prompt": "go"})

        context = agent.get_conversation().context
        assert set(context) == {"answer"}
        assert not any(name.startswith(SUB_NAME_PREFIX) for name in context)

    def test_a_held_variable_updated_by_the_script_persists(self) -> None:
        agent = make_script_agent(
            _engine("total = add(x=2, y=3)\nreturn total"), record_enabled=True,
        )
        agent.get_conversation().remember("total", 0, "Running total")

        agent.invoke({"prompt": "go"})

        variable = agent.get_conversation().context["total"]
        assert variable.value == 5
        assert variable.description == "Running total"

    def test_a_held_variable_is_not_persisted_without_recording(self) -> None:
        agent = make_script_agent(_engine("total = add(x=2, y=3)\nreturn total"))
        agent.get_conversation().remember("total", 0)

        agent.invoke({"prompt": "go"})

        assert agent.get_conversation().context["total"].value == 0

    def test_a_readonly_variable_is_never_overwritten(self) -> None:
        agent = make_script_agent(
            _engine("return policy"), record_enabled=True, response_to_context="policy",
        )
        agent.get_conversation().remember("policy", "keep", readonly=True)

        agent.invoke({"prompt": "go"})

        assert agent.get_conversation().context["policy"].value == "keep"

    def test_task_result_entries_are_never_persisted(self) -> None:
        agent = make_script_agent(_engine("return 1", "return task_result_0"), record_enabled=True)

        agent.invoke({"prompt": "first"})
        agent.invoke({"prompt": "second"})

        assert not any(n.startswith("task_result") for n in agent.get_conversation().context)


def shout(result: Any) -> str:
    return f"{result}!"


def second_item(second: Any) -> str:
    return f"second={second}"


class TestPostInvoke:
    def test_post_invoke_receives_the_whole_response(self) -> None:
        agent = make_script_agent(
            _engine("return add(x=2, y=3)"), post_invoke=shout, response_to_post="result",
        )

        assert agent.invoke({"prompt": "go"}).result == "5!"

    def test_response_to_post_selector_slices_the_response(self) -> None:
        agent = make_script_agent(
            _engine("return [4, 5]"), post_invoke=second_item, response_to_post={1: "second"},
        )

        assert agent.invoke({"prompt": "go"}).result == "second=5"

    def test_response_to_post_and_context_can_route_together(self) -> None:
        agent = make_script_agent(
            _engine("return [4, 5]"), record_enabled=True, post_invoke=second_item,
            response_to_post={1: "second"}, response_to_context={0: "first"},
        )

        result = agent.invoke({"prompt": "go"})

        assert result.result == "second=5"
        assert agent.get_conversation().context["first"].value == 4


class TestAsyncMirrors:
    def test_one_shot_lifecycle(self) -> None:
        engine = _engine("result = add(x=4, y=6)\nreturn result")
        agent = make_script_agent(engine)

        final = asyncio.run(agent.async_invoke({"prompt": "add 4 and 6"}))

        assert final.result == 10
        assert engine.call_count == 1

    def test_regen_recovers_from_undefined_reference(self) -> None:
        engine = _engine(
            "result = add(x=1, y=undefined_name)\nreturn result",
            "result = add(x=1, y=2)\nreturn result",
        )
        agent = make_script_agent(engine, regeneration_limit=1)

        assert asyncio.run(agent.async_invoke({"prompt": "add"})).result == 3
        assert engine.call_count == 2

    def test_results_become_context_variables(self) -> None:
        agent = make_script_agent(_engine("y = add(x=add(x=1, y=2), y=3)\nreturn y"))

        async def run() -> ScriptActAgentTask:
            task = agent.pre_process(agent.summarize(agent.create_task(agent.filter_inputs({"prompt": "go"}))))
            while not task.complete:
                task = await agent.async_think(task)
                task = await agent.async_prepare(task)
                task = await agent.async_act(task)
            return task

        task = asyncio.run(run())

        assert task.context["y"].value == 6
        assert task.context[f"{SUB_NAME_PREFIX}0"].value == 3

    def test_empty_generation_returns_none(self) -> None:
        assert asyncio.run(make_script_agent(_engine("")).async_invoke({"prompt": "x"})).result is None

    def test_async_act_is_a_no_op_while_a_repair_is_pending(self) -> None:
        agent = make_script_agent(_engine())
        task = _new_task(agent)
        task.needs_repair = True

        assert asyncio.run(agent.async_act(task)).complete is False

    def test_resolution_failure_triggers_a_repair_round(self) -> None:
        engine = _engine("x = 'hello'\ny = add(x=x + 1, y=2)\nreturn y", "return add(x=1, y=1)")
        agent = make_script_agent(engine, replanning_limit=1)

        assert asyncio.run(agent.async_invoke({"prompt": "do something"})).result == 2
        assert engine.call_count == 2

    def test_execution_failure_triggers_a_repair_round(self) -> None:
        engine = _engine("a = add(x=1, y=1)\nb = fail_tool()\nreturn a", "return a")
        agent = make_script_agent(engine, replanning_limit=1)

        assert asyncio.run(agent.async_invoke({"prompt": "go"})).result == 2
        assert "intentional failure" in _non_system(engine, 1)

    def test_replanning_limit_zero_raises_immediately(self) -> None:
        engine = _engine("x = 'hello'\ny = add(x=x + 1, y=2)\nreturn y")
        agent = make_script_agent(engine, replanning_limit=0)

        with pytest.raises(ToolAgentError, match="repair budget exhausted"):
            asyncio.run(agent.async_invoke({"prompt": "do something"}))

        assert engine.call_count == 1

    def test_regeneration_limit_exhausted_raises(self) -> None:
        engine = _engine("result = add(x=1, y=undefined_name)\nreturn result")
        agent = make_script_agent(engine, regeneration_limit=0)

        with pytest.raises(ToolAgentError, match="exhausted after 1 attempt"):
            asyncio.run(agent.async_invoke({"prompt": "add"}))

    def test_post_invoke_routing(self) -> None:
        agent = make_script_agent(
            _engine("return [4, 5]"), post_invoke=second_item, response_to_post={1: "second"},
        )

        assert asyncio.run(agent.async_invoke({"prompt": "go"})).result == "second=5"

    def test_seed_clash_raises_before_any_model_call(self) -> None:
        engine = _engine("return 1")
        agent = make_script_agent(engine)
        agent.get_conversation().remember("for", 1)

        with pytest.raises(AgentInvocationError):
            asyncio.run(agent.async_invoke({"prompt": "go"}))

        assert engine.call_count == 0
