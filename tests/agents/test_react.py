"""ReActAgent on the statement core: one Python statement per round.

Scripted replies are plain Python text; ``FakeLLMEngine`` returns them as the
engine result. The tools come from ``conftest.register_math_tools``:
``add(x, y)``, ``multiply(x, y)``, ``join_text(prefix, value)`` and
``fail_tool()``.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from atomic_agentic.agents.react import ReActAgent
from atomic_agentic.constants.prompts import REACTOR_PROMPT
from atomic_agentic.exceptions import ToolAgentError, ToolInvocationError
from atomic_agentic.models.agents.blackboard_models import Statement
from atomic_agentic.models.results.agents import ToolAgentResult

from .conftest import FakeLLMEngine, make_react_agent


def fenced(code: str) -> str:
    return f"```python\n{code}\n```"


def run_round(agent: ReActAgent, task: Any) -> Any:
    return agent.act(agent.prepare(agent.think(task)))


def new_task(agent: ReActAgent, prompt: str = "run") -> Any:
    task = agent.create_task(agent.filter_inputs({"prompt": prompt}))
    task.user_prompt = prompt
    return task


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #
class TestReActConstruction:
    def test_fail_fast_defaults_false_and_forwards_true(self) -> None:
        assert make_react_agent([]).fail_fast is False
        assert make_react_agent([], fail_fast=True).fail_fast is True

    def test_fail_fast_must_be_bool(self) -> None:
        with pytest.raises(ToolAgentError):
            ReActAgent(
                name="tests", description=".", llm_engine=FakeLLMEngine([]),
                fail_fast="yes",  # type: ignore[arg-type]
            )

    def test_instructions_forwarded(self) -> None:
        agent = make_react_agent([], instructions="Be terse.")
        assert agent.instructions.template == "Be terse."

    @pytest.mark.parametrize(
        "removed", [{"constants": [1]}, {"tool_instructions": "x"}, {"tool_concurrency_limit": 1}]
    )
    def test_removed_constructor_arguments_raise_type_error(self, removed: dict[str, Any]) -> None:
        with pytest.raises(TypeError):
            ReActAgent(name="tests", description=".", llm_engine=FakeLLMEngine([]), **removed)

    def test_tool_calls_limit_defaults_to_25(self) -> None:
        agent = ReActAgent(name="tests", description=".", llm_engine=FakeLLMEngine([]))
        assert agent.tool_calls_limit == 25

    def test_tool_calls_limit_rejects_negative(self) -> None:
        with pytest.raises(ToolAgentError, match="tool_calls_limit"):
            ReActAgent(
                name="tests", description=".", llm_engine=FakeLLMEngine([]), tool_calls_limit=-1,
            )

    def test_limits_forwarded(self) -> None:
        agent = make_react_agent([], tool_calls_limit=3, regeneration_limit=2)
        assert agent.tool_calls_limit == 3
        assert agent.regeneration_limit == 2

    def test_prompt_and_python_grammar_hooks(self) -> None:
        assert ReActAgent.TOOL_INSTRUCTIONS is REACTOR_PROMPT
        agent = make_react_agent([])
        assert agent.has_tool("add") and not agent.has_tool("return")
        with pytest.raises(Exception):
            agent.register_tool(lambda: None, name="len")


# --------------------------------------------------------------------------- #
# Full invoke() round trip
# --------------------------------------------------------------------------- #
class TestReActInvokeRoundTrip:
    @staticmethod
    def _agent() -> ReActAgent:
        return make_react_agent(
            [
                "a = add(2, 3)",
                "b = multiply(a, 10)",
                "return b",
            ],
            tool_calls_limit=2,
        )

    def test_multi_round_invoke_runs_then_returns(self) -> None:
        assert self._agent().invoke({"prompt": "run"}).result == 50

    def test_async_invoke_matches(self) -> None:
        assert asyncio.run(self._agent().async_invoke({"prompt": "run"})).result == 50

    def test_fenced_reply_is_accepted(self) -> None:
        agent = make_react_agent([fenced("return 42")])
        assert agent.invoke({"prompt": "run"}).result == 42

    def test_return_none_is_a_legitimate_result(self) -> None:
        assert make_react_agent(["return"]).invoke({"prompt": "run"}).result is None

    def test_only_the_first_statement_is_used(self) -> None:
        agent = make_react_agent(
            ["a = add(1, 2)\nb = add(a, 3)", "return a"], tool_calls_limit=1,
        )
        assert agent.invoke({"prompt": "run"}).result == 3

    def test_inert_leading_note_is_ignored(self) -> None:
        agent = make_react_agent(['"thinking out loud"\nreturn 9'])
        assert agent.invoke({"prompt": "run"}).result == 9


# --------------------------------------------------------------------------- #
# think() / act() handoff
# --------------------------------------------------------------------------- #
class TestReActThinkActHandoff:
    def test_think_compiles_one_statement_without_running_it(self) -> None:
        agent = make_react_agent(["x = add(1, 2)"])
        task = agent.think(new_task(agent))

        assert isinstance(task.statement, Statement)
        assert task.statement.status == "not_run"
        assert task.statement.target == "x"
        assert task.completed == [] and task.failed_statements == []

    def test_prepare_is_a_noop(self) -> None:
        agent = make_react_agent(["x = add(1, 2)"])
        task = agent.think(new_task(agent))

        assert agent.prepare(task) is task
        assert task.statement is not None

    def test_act_commits_the_statement_and_clears_it(self) -> None:
        agent = make_react_agent(["x = add(1, 2)"])
        task = run_round(agent, new_task(agent))

        assert task.statement is None
        assert [s.source for s in task.completed] == ["x = add(1, 2)"]
        assert task.completed[0].status == "done"
        assert task.context["x"].value == 3
        assert task.complete is False

    def test_ordinals_run_across_completed_and_failed(self) -> None:
        agent = make_react_agent(["fail_tool()", "x = add(1, 2)"], tool_calls_limit=5)
        task = run_round(agent, new_task(agent))
        task.task_messages.clear()
        task = run_round(agent, task)

        assert task.failed_statements[0].ordinal == 0
        assert task.completed[0].ordinal == 1

    def test_return_sets_response_and_completes(self) -> None:
        agent = make_react_agent(["return add(2, 2)"], tool_calls_limit=1)
        task = run_round(agent, new_task(agent))

        assert task.complete is True
        assert task.generated_response == 4

    def test_async_round_matches(self) -> None:
        agent = make_react_agent(["x = add(1, 2)"])
        task = new_task(agent)

        async def go() -> Any:
            task_ = await agent.async_think(task)
            return await agent.async_act(await agent.async_prepare(task_))

        task = asyncio.run(go())
        assert task.context["x"].value == 3
        assert task.statement is None


# --------------------------------------------------------------------------- #
# Composed statements
# --------------------------------------------------------------------------- #
class TestReActComposedStatements:
    def test_composed_call_runs_innermost_first_and_binds_only_the_target(self) -> None:
        agent = make_react_agent(["x = add(multiply(2, 3), 4)"], tool_calls_limit=2)
        task = run_round(agent, new_task(agent))

        statement = task.completed[0]
        assert [s.status for s in statement.steps] == ["ran", "ran"]
        assert statement.steps[0].value == 6 and statement.value == 10
        assert task.context["x"].value == 10
        assert not any(name.startswith("_SUB_") for name in task.context)

    def test_failing_inner_call_records_what_ran_and_skips_the_rest(self) -> None:
        agent = make_react_agent(["x = add(multiply(2, 3), fail_tool())"], tool_calls_limit=5)
        task = run_round(agent, new_task(agent))

        failed = task.failed_statements[0]
        assert failed.status == "failed"
        assert [s.status for s in failed.steps] == ["ran", "failed", "not_run"]
        assert failed.steps[0].value == 6
        assert "x" not in task.context
        assert "sub-steps that already ran" in task.failure_note

    def test_expression_around_a_call_is_free(self) -> None:
        agent = make_react_agent(["x = add(1, 2) * 10", "return x"], tool_calls_limit=1)
        assert agent.invoke({"prompt": "run"}).result == 30

    def test_call_free_assignment_costs_one(self) -> None:
        agent = make_react_agent(["y = 1 + 1", "return y"], tool_calls_limit=1)
        result = agent.invoke({"prompt": "run"})
        assert result.result == 2


# --------------------------------------------------------------------------- #
# Auto-naming of bare calls
# --------------------------------------------------------------------------- #
class TestReActAutoNaming:
    def test_unnamed_results_get_contiguous_dunder_names(self) -> None:
        agent = make_react_agent(["add(1, 2)", "add(3, 4)", "return __r0__ + __r1__"], tool_calls_limit=2)
        assert agent.invoke({"prompt": "run"}).result == 10

    def test_bare_call_returning_none_is_not_named(self) -> None:
        agent = make_react_agent(["x = add(1, 1)"], tool_calls_limit=2)
        agent.register_tool(lambda: None, name="nothing", description="returns None")
        agent._llm_engine.responses = ["nothing()"]
        task = run_round(agent, new_task(agent))

        assert task.completed[0].value is None
        assert not any(name.startswith("__r") for name in task.context)

    def test_failed_statement_does_not_consume_a_number(self) -> None:
        agent = make_react_agent(
            ["fail_tool()", "add(1, 2)", "return __r0__"], tool_calls_limit=3,
        )
        assert agent.invoke({"prompt": "run"}).result == 3

    def test_named_assignment_keeps_its_name(self) -> None:
        agent = make_react_agent(["a = add(1, 2)"])
        task = run_round(agent, new_task(agent))
        assert "a" in task.context and "__r0__" not in task.context

    def test_auto_named_result_is_in_context_and_not_persisted(self) -> None:
        agent = make_react_agent(["add(2, 3)"], tool_calls_limit=1)
        task = run_round(agent, new_task(agent))

        assert task.context["__r0__"].value == 5
        assert "__r0__" in task.changed_context


# --------------------------------------------------------------------------- #
# Regeneration (rejected before anything runs)
# --------------------------------------------------------------------------- #
class TestReActGenerationRetry:
    @pytest.mark.parametrize(
        "bad",
        [
            "return missing",                       # undefined name
            "x = nonexistent_tool(1)",              # unknown tool
            "x = add(1, zzz=2)",                    # bind mismatch
            "x = a or b",                           # rejected form
            "for i in y: pass",                     # not a statement form
            "x = (",                                # syntax error
            "add = 3",                              # assignment to a tool name
        ],
    )
    def test_invalid_statement_regenerates_then_succeeds(self, bad: str) -> None:
        agent = make_react_agent([bad, "return 7"], regeneration_limit=1, record_enabled=True)

        assert agent.invoke({"prompt": "run"}).result == 7
        record = agent.get_conversation().records[-1]
        assert record.regenerations_used == 1
        assert record.failed_statements == ()
        assert len(record.llm_records) == 2

    def test_budget_exhausted_raises(self) -> None:
        agent = make_react_agent(["return missing"], regeneration_limit=0)
        with pytest.raises(ToolAgentError, match="regeneration budget exhausted"):
            agent.invoke({"prompt": "run"})

    def test_regeneration_feedback_reaches_the_next_prompt(self) -> None:
        agent = make_react_agent(["return missing", "return 1"], regeneration_limit=1)
        agent.invoke({"prompt": "run"})

        retry = agent._llm_engine.calls[1]
        assert retry[-2]["role"] == "assistant" and retry[-2]["content"] == "return missing"
        assert "undefined name 'missing'" in retry[-1]["content"]

    def test_over_budget_statement_is_rejected_with_the_need_and_the_remainder(self) -> None:
        agent = make_react_agent(
            ["x = add(multiply(1, 2), 3)", "return 1"], tool_calls_limit=1, regeneration_limit=1,
        )
        agent.invoke({"prompt": "run"})

        feedback = agent._llm_engine.calls[1][-1]["content"]
        assert "needs 2 action(s) but 1 remain" in feedback

    def test_no_actions_left_only_a_return_is_allowed(self) -> None:
        agent = make_react_agent(["x = add(1, 2)", "return 1"], tool_calls_limit=0, regeneration_limit=1)
        agent.invoke({"prompt": "run"})

        assert "no actions remain" in agent._llm_engine.calls[1][-1]["content"]

    def test_return_with_no_call_is_allowed_at_zero_budget(self) -> None:
        agent = make_react_agent(["return 1"], tool_calls_limit=0)
        assert agent.invoke({"prompt": "run"}).result == 1

    def test_return_with_a_call_costs_one(self) -> None:
        agent = make_react_agent(["return add(1, 2)"], tool_calls_limit=1)
        assert agent.invoke({"prompt": "run"}).result == 3
        over = make_react_agent(["return add(1, 2)", "return 0"], tool_calls_limit=0, regeneration_limit=1)
        assert over.invoke({"prompt": "run"}).result == 0

    def test_regeneration_budget_restarts_each_round(self) -> None:
        agent = make_react_agent(
            [
                "x = add(1, zzz=2)",
                "a = add(1, 2)",
                "return missing",
                "return a",
            ],
            tool_calls_limit=1,
            regeneration_limit=1,
            record_enabled=True,
        )
        result = agent.invoke({"prompt": "run"})

        assert result.result == 3
        assert result.regenerations_used == 2

    def test_async_retry_loop_regenerates_then_succeeds(self) -> None:
        agent = make_react_agent(
            ["return missing", "return 7"], regeneration_limit=1, record_enabled=True,
        )
        result = asyncio.run(agent.async_invoke({"prompt": "run"}))

        assert result.result == 7
        assert agent.get_conversation().records[-1].regenerations_used == 1

    def test_async_retry_loop_budget_exhausted_raises(self) -> None:
        agent = make_react_agent(["return missing"], regeneration_limit=0)
        with pytest.raises(ToolAgentError, match="regeneration budget exhausted"):
            asyncio.run(agent.async_invoke({"prompt": "run"}))


# --------------------------------------------------------------------------- #
# Execution failure
# --------------------------------------------------------------------------- #
class TestReActExecutionFailure:
    def test_failure_is_recorded_whole_and_the_run_continues(self) -> None:
        agent = make_react_agent(
            ["x = fail_tool()", "return 1"], tool_calls_limit=2, record_enabled=True,
        )
        result = agent.invoke({"prompt": "run"})

        assert result.result == 1
        assert result.failed_call_count == 1
        record = agent.get_conversation().records[-1]
        assert len(record.failed_statements) == 1
        failed = record.failed_statements[0]
        assert failed.status == "failed" and failed.steps[0].error is not None
        assert [s.source for s in record.statements] == ["return 1"]

    def test_fail_fast_records_the_statement_then_raises(self) -> None:
        agent = make_react_agent(["x = fail_tool()"], tool_calls_limit=2, fail_fast=True)
        task = new_task(agent)
        task = agent.prepare(agent.think(task))

        with pytest.raises(ToolInvocationError):
            agent.act(task)
        assert len(task.failed_statements) == 1
        assert task.failed_statements[0].status == "failed"
        assert task.statement is None

    def test_fail_fast_raises_through_invoke(self) -> None:
        agent = make_react_agent(["fail_tool()"], fail_fast=True)
        with pytest.raises(ToolInvocationError):
            agent.invoke({"prompt": "run"})

    def test_fail_fast_wraps_a_non_tool_invocation_error(self) -> None:
        agent = make_react_agent(["x = missing_value + 1"], fail_fast=True)
        task = new_task(agent)
        task.context["missing_value"] = _readonly_value(1)
        task = agent.prepare(agent.think(task))
        del task.context["missing_value"]

        with pytest.raises(ToolAgentError, match="statement failed at"):
            agent.act(task)
        assert len(task.failed_statements) == 1

    def test_async_failure_is_tolerated_when_not_fail_fast(self) -> None:
        agent = make_react_agent(["fail_tool()", "return 1"], tool_calls_limit=2, record_enabled=True)
        result = asyncio.run(agent.async_invoke({"prompt": "run"}))

        assert result.result == 1
        assert result.failed_call_count == 1

    def test_async_failure_raises_when_fail_fast(self) -> None:
        agent = make_react_agent(["fail_tool()"], fail_fast=True)
        with pytest.raises(ToolInvocationError):
            asyncio.run(agent.async_invoke({"prompt": "run"}))

    def test_failed_statement_does_not_update_a_writable_variable(self) -> None:
        agent = make_react_agent(["total = fail_tool()", "return total"], tool_calls_limit=2, record_enabled=True)
        agent.get_conversation().remember("total", 7)

        result = agent.invoke({"prompt": "run"})

        assert result.result == 7
        assert agent.get_conversation().context["total"].value == 7

    def test_failed_statement_still_spends_budget(self) -> None:
        agent = make_react_agent(["fail_tool()", "return 1"], tool_calls_limit=3)
        agent.invoke({"prompt": "run"})

        assert "Actions remaining: 2 of 3." in agent._llm_engine.calls[1][-1]["content"]

    def test_partial_failure_spends_every_step_that_ran_or_failed(self) -> None:
        agent = make_react_agent(
            ["x = add(multiply(2, 3), fail_tool())", "return 1"], tool_calls_limit=5,
        )
        agent.invoke({"prompt": "run"})

        assert "Actions remaining: 3 of 5." in agent._llm_engine.calls[1][-1]["content"]

    def test_keyboard_interrupt_in_a_tool_propagates(self) -> None:
        def interrupt() -> None:
            raise KeyboardInterrupt

        agent = make_react_agent(["interrupt()"], tool_calls_limit=2)
        agent.register_tool(interrupt)

        with pytest.raises(KeyboardInterrupt):
            agent.invoke({"prompt": "run"})


def _readonly_value(value: Any) -> Any:
    from atomic_agentic.models.agents.context import ContextVariable

    return ContextVariable(name="missing_value", value=value, readonly=True)


# --------------------------------------------------------------------------- #
# Record and result
# --------------------------------------------------------------------------- #
class TestReActRecordAndResult:
    def test_successful_run_records_statements_and_usage(self) -> None:
        agent = make_react_agent(
            ["a = add(2, 3)", "b = add(a, 1)", "return b"], tool_calls_limit=2, record_enabled=True,
        )
        result = agent.invoke({"prompt": "run"})

        assert result.result == 6
        record = agent.get_conversation().records[-1]
        assert [s.source for s in record.statements] == ["a = add(2, 3)", "b = add(a, 1)", "return b"]
        assert all(isinstance(s, Statement) and s.status == "done" for s in record.statements)
        assert record.failed_statements == ()

        by_tool = {u.tool_name: u.call_count for u in result.usage_report.by_tool}
        assert by_tool == {"add": 2}
        assert result.usage_report.total_dispatched == 2
        assert result.usage_report.total_failed == 0
        assert result.failed_call_count == 0

    def test_failed_run_usage_counts_the_failed_call(self) -> None:
        agent = make_react_agent(
            ["fail_tool()", "add(1, 2)", "return __r0__"], tool_calls_limit=3, record_enabled=True,
        )
        result = agent.invoke({"prompt": "run"})

        assert result.result == 3
        assert result.failed_call_count == 1
        assert result.usage_report.total_failed == 1
        assert result.usage_report.total_dispatched == 2

    def test_builtin_and_method_steps_report_their_real_names(self) -> None:
        agent = make_react_agent(
            ["n = len('abcd')", "u = 'ab'.upper()", "return n"], tool_calls_limit=2, record_enabled=True,
        )
        result = agent.invoke({"prompt": "run"})

        by_tool = {u.tool_name: u.call_count for u in result.usage_report.by_tool}
        assert by_tool == {"len": 1, "upper": 1}

    def test_record_renders_as_code_in_source_order(self) -> None:
        agent = make_react_agent(
            ["a = add(1, 2)", "fail_tool()", "return a"], tool_calls_limit=3, record_enabled=True,
        )
        agent.invoke({"prompt": "run"})

        code = agent.get_conversation().records[-1].render_as_code().split("\n")
        assert code[0] == "a = add(1, 2)"
        assert code[1].startswith("fail_tool()  # FAILED")
        assert code[-1] == "return a"

    def test_result_class_and_run_id(self) -> None:
        agent = make_react_agent(["return 42"], record_enabled=True)
        result = agent.invoke({"prompt": "go"})

        assert ReActAgent.result_cls is ToolAgentResult
        assert type(result) is ToolAgentResult
        assert result.run_id == agent.get_conversation().records[-1].run_id

    def test_zero_action_run_has_an_empty_usage_report(self) -> None:
        result = make_react_agent(["return 1"]).invoke({"prompt": "go"})
        assert result.usage_report.by_tool == ()


# --------------------------------------------------------------------------- #
# Context binding
# --------------------------------------------------------------------------- #
class TestReActContextBinding:
    def test_readonly_context_variable_resolves(self) -> None:
        agent = make_react_agent(["a = add(limit, 1)", "return a"], tool_calls_limit=1)
        agent.get_conversation().remember("limit", 10, readonly=True)

        assert agent.invoke({"prompt": "run"}).result == 11

    def test_writable_context_variable_resolves(self) -> None:
        agent = make_react_agent(["return notes"])
        agent.get_conversation().remember("notes", "draft")

        assert agent.invoke({"prompt": "run"}).result == "draft"

    def test_prior_turn_result_resolves_by_task_result_name(self) -> None:
        agent = make_react_agent(["return 42", "return task_result_0"], record_enabled=True)
        agent.invoke({"prompt": "first"})

        assert agent.invoke({"prompt": "second"}).result == 42

    def test_undeclared_new_name_is_not_persisted(self) -> None:
        agent = make_react_agent(["scratch = add(2, 3)", "return scratch"], tool_calls_limit=1, record_enabled=True)

        assert agent.invoke({"prompt": "run"}).result == 5
        assert agent.get_conversation().context == {}

    def test_writable_variable_updated_by_assignment_is_persisted(self) -> None:
        agent = make_react_agent(["total = add(2, 3)", "return total"], tool_calls_limit=1, record_enabled=True)
        agent.get_conversation().remember("total", 0, "Running total")

        agent.invoke({"prompt": "run"})

        variable = agent.get_conversation().context["total"]
        assert variable.value == 5
        assert variable.description == "Running total"

    def test_writable_variable_may_be_read_by_the_statement_that_updates_it(self) -> None:
        agent = make_react_agent(["total = add(total, 1)", "return total"], tool_calls_limit=1)
        agent.get_conversation().remember("total", 10)

        assert agent.invoke({"prompt": "run"}).result == 11

    def test_assignment_to_a_readonly_variable_is_regenerated(self) -> None:
        agent = make_react_agent(["limit = add(1, 1)", "return limit"], tool_calls_limit=1, regeneration_limit=1)
        agent.get_conversation().remember("limit", 10, readonly=True)

        agent.invoke({"prompt": "run"})
        assert "read-only" in agent._llm_engine.calls[1][-1]["content"]

    def test_context_name_that_is_a_tool_id_is_rejected_at_task_creation(self) -> None:
        agent = make_react_agent(["return 1"])
        agent.get_conversation().remember("add", 1)

        with pytest.raises(Exception, match="unusable in a script"):
            agent.invoke({"prompt": "run"})


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
class TestReActRendering:
    @staticmethod
    def _two_round_agent(**kwargs: Any) -> ReActAgent:
        agent = make_react_agent(["a = add(2, 3)", "return a"], **kwargs)
        agent.invoke({"prompt": "do the thing"})
        return agent

    def test_round_one_is_one_user_message_ending_with_the_directive(self) -> None:
        agent = self._two_round_agent(tool_calls_limit=3, record_enabled=True)
        messages = agent._llm_engine.calls[0]

        users = [m for m in messages if m["role"] == "user"]
        assert len(users) == 1
        assert "USER TASK PROMPT" in users[0]["content"]
        assert "STATEMENTS COMPLETED" not in users[0]["content"]
        assert users[0]["content"].rstrip().endswith("Actions remaining: 3 of 3.")

    def test_round_two_has_the_task_block_steps_and_directive(self) -> None:
        agent = self._two_round_agent(tool_calls_limit=3)
        messages = [m for m in agent._llm_engine.calls[1] if m["role"] != "system"]

        assert [m["role"] for m in messages] == ["user", "assistant", "user"]
        assert "# STATEMENTS COMPLETED SO FAR:\na = add(2, 3)" in messages[1]["content"]
        assert "Actions remaining: 2 of 3." in messages[2]["content"]

    def test_steps_list_the_sources_of_completed_statements_in_order(self) -> None:
        agent = make_react_agent(["a = add(1, 2)", "b = add(a, 1)", "return b"], tool_calls_limit=3)
        agent.invoke({"prompt": "run"})

        steps = [m for m in agent._llm_engine.calls[2] if m["role"] == "assistant"][0]["content"]
        assert "a = add(1, 2)\nb = add(a, 1)" in steps

    def test_bound_result_appears_in_the_round_two_context_block(self) -> None:
        agent = self._two_round_agent(tool_calls_limit=3)
        task_block = next(m["content"] for m in agent._llm_engine.calls[1] if m["role"] == "user")

        assert "RUNTIME CONTEXT" in task_block
        assert "name: a" in task_block
        assert "value:\n5" in task_block

    def test_hidden_context_limit_shows_value_hidden(self) -> None:
        agent = make_react_agent(["a = add(2, 3)", "return a"], tool_calls_limit=2)
        agent.get_conversation().context_preview_limit = 0
        agent.invoke({"prompt": "run"})

        task_block = next(m["content"] for m in agent._llm_engine.calls[1] if m["role"] == "user")
        assert "name: a" in task_block and "value hidden" in task_block

    def test_failure_shows_in_steps_and_directive_then_clears(self) -> None:
        agent = make_react_agent(
            ["fail_tool()", "add(1, 2)", "return __r0__"], tool_calls_limit=3,
        )
        agent.invoke({"prompt": "run"})

        second = [m for m in agent._llm_engine.calls[1] if m["role"] != "system"]
        assert "YOUR LAST STATEMENT FAILED" in second[1]["content"]
        assert "fail_tool()  # FAILED" in second[1]["content"]
        assert "intentional failure" in second[1]["content"]
        assert "Your last statement failed" in second[2]["content"]
        third = "\n".join(m["content"] for m in agent._llm_engine.calls[2] if m["role"] != "system")
        assert "YOUR LAST STATEMENT FAILED" not in third
        assert "Your last statement failed" not in third

    def test_multi_step_failure_note_names_the_substep_and_what_ran(self) -> None:
        agent = make_react_agent(["x = add(multiply(2, 3), fail_tool())", "return 1"], tool_calls_limit=5)
        agent.invoke({"prompt": "run"})

        steps = [m for m in agent._llm_engine.calls[1] if m["role"] == "assistant"][0]["content"]
        assert "# sub-step `fail_tool()` failed:" in steps
        assert "`multiply(2, 3)` -> 6" in steps

    def test_budget_line_is_absent_without_a_limit(self) -> None:
        agent = self._two_round_agent(tool_calls_limit=None)
        assert "Actions remaining" not in agent._llm_engine.calls[0][-1]["content"]

    def test_budget_exhausted_directive_names_the_return_only_rule(self) -> None:
        agent = self._two_round_agent(tool_calls_limit=1)
        assert "Only a return with no call in it is allowed now." in agent._llm_engine.calls[1][-1]["content"]

    def test_context_variable_renders_by_name(self) -> None:
        agent = make_react_agent(["return 1"])
        agent.get_conversation().remember("limit", 10, "A fixed limit.", readonly=True)
        agent.invoke({"prompt": "run"})

        user = agent._llm_engine.calls[0][-1]["content"]
        assert "name: limit" in user and "RUNTIME CONTEXT" in user

    def test_system_message_frames_tools_builtins_and_user_instructions(self) -> None:
        agent = make_react_agent(["return 1"], instructions="Be terse.")
        agent.invoke({"prompt": "run"})

        system = next(m["content"] for m in agent._llm_engine.calls[0] if m["role"] == "system")
        assert "TOOL-CALLING INSTRUCTIONS" in system
        assert "AVAILABLE TOOLS" in system
        assert "USER-DEFINED INSTRUCTIONS" in system and "Be terse." in system

    def test_instructions_slot_runs_once_across_a_multi_round_run(self) -> None:
        calls: list[str] = []

        def guidance() -> str:
            calls.append("called")
            return "Be terse."

        agent = make_react_agent(
            ["a = add(2, 3)", "b = add(1, 1)", "return 1"], instructions=guidance,
        )
        agent.invoke({"prompt": "run"})

        assert len(agent._llm_engine.calls) == 3 and calls == ["called"]


# --------------------------------------------------------------------------- #
# post_invoke / response_to_post / response_to_context routing
# --------------------------------------------------------------------------- #
def shout(result: Any) -> str:
    return f"{result}!"


def second_item(second: Any) -> str:
    return f"second={second}"


class TestReActPostInvokeRouting:
    def test_post_invoke_receives_the_whole_response(self) -> None:
        agent = make_react_agent(
            ["a = add(2, 3)", "return a"], tool_calls_limit=1, post_invoke=shout, response_to_post="result",
        )
        assert agent.invoke({"prompt": "go"}).result == "5!"

    def test_response_to_post_selector_slices_the_response(self) -> None:
        agent = make_react_agent(
            ["pair = [4, 5]", "return pair"], tool_calls_limit=1,
            post_invoke=second_item, response_to_post={1: "second"},
        )
        assert agent.invoke({"prompt": "go"}).result == "second=5"

    def test_response_to_post_and_context_can_route_together(self) -> None:
        agent = make_react_agent(
            ["pair = [4, 5]", "return pair"], tool_calls_limit=1, record_enabled=True,
            post_invoke=second_item, response_to_post={1: "second"}, response_to_context={0: "first"},
        )
        result = agent.invoke({"prompt": "go"})

        assert result.result == "second=5"
        assert agent.get_conversation().context["first"].value == 4

    def test_async_post_invoke_routing(self) -> None:
        agent = make_react_agent(
            ["pair = [4, 5]", "return pair"], tool_calls_limit=1,
            post_invoke=second_item, response_to_post={1: "second"},
        )
        assert asyncio.run(agent.async_invoke({"prompt": "go"})).result == "second=5"
