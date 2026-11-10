from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from atomic_agentic.agents.react import ReActAgent
from atomic_agentic.agents.tools import return_tool
from atomic_agentic.exceptions import ToolAgentError, ToolInvocationError, ToolRegistrationError
from atomic_agentic.models.agents.blackboard_models import ToolStatement

from .conftest import (
    FakeLLMEngine,
    arg,
    make_react_agent,
    react_step_json,
)


def rstep(**kwargs: Any) -> dict[str, Any]:
    """Build one scripted FakeLLMEngine response as a real dict.

    ``FakeLLMEngine`` never parses its scripted responses -- whatever object
    sits in its ``responses`` list becomes ``LLMResult.result`` verbatim (a
    real engine's own ``_extract_result`` does the ``json.loads`` when
    ``output_structure`` was requested; the fake intentionally does not
    reproduce that). ``conftest.react_step_json`` builds the correct
    wire-shaped payload as JSON *text*; this helper round-trips it back into
    a plain dict so it can be handed to ``FakeLLMEngine`` directly. Mirrors
    ``test_planact.py``'s own ``scripted()`` helper.
    """
    return json.loads(react_step_json(**kwargs))


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #
class TestReActConstruction:
    def test_fail_fast_defaults_false(self) -> None:
        agent = make_react_agent([])
        assert agent.fail_fast is False

    def test_fail_fast_forwarded_true(self) -> None:
        agent = make_react_agent([], fail_fast=True)
        assert agent.fail_fast is True

    def test_instructions_forwarded(self) -> None:
        agent = make_react_agent([], instructions="Be terse.")
        assert agent.instructions == "Be terse."

    @pytest.mark.parametrize("removed", [{"constants": [1]}, {"tool_instructions": "x"}])
    def test_removed_constructor_arguments_raise_type_error(self, removed: dict[str, Any]) -> None:
        with pytest.raises(TypeError):
            ReActAgent(
                name="tests",
                namespace="tests",
                description=".",
                llm_engine=FakeLLMEngine([]),
                **removed,
            )

    def test_fail_fast_must_be_bool(self) -> None:
        with pytest.raises(ToolAgentError):
            ReActAgent(
                name="tests",
                namespace="tests",
                description=".",
                llm_engine=FakeLLMEngine([]),
                fail_fast="yes",  # type: ignore[arg-type]
            )

    def test_tool_calls_limit_defaults_to_25(self) -> None:
        # ReActAgent's own divergent default from ToolAgent's None -- this
        # family has no second round-ceiling knob, so an unbounded default
        # would have zero structural backstop.
        agent = ReActAgent(
            name="tests",
            namespace="tests",
            description=".",
            llm_engine=FakeLLMEngine([]),
        )
        assert agent.tool_calls_limit == 25

    def test_tool_calls_limit_rejects_negative(self) -> None:
        with pytest.raises(ToolAgentError, match="tool_calls_limit"):
            ReActAgent(
                name="tests",
                namespace="tests",
                description=".",
                llm_engine=FakeLLMEngine([]),
                tool_calls_limit=-1,
            )

    def test_tool_calls_limit_and_regeneration_limit_forwarded(self) -> None:
        agent = make_react_agent([], tool_calls_limit=3, regeneration_limit=2)
        assert agent.tool_calls_limit == 3
        assert agent.regeneration_limit == 2

    def test_reserved_tools_registered_on_construction(self) -> None:
        agent = make_react_agent([])
        assert agent.has_tool("return")
        assert agent.has_tool("make_sequence")
        assert agent.has_tool("make_dict")
        assert agent.get_tool("return") is return_tool

    def test_reserved_tool_names_cannot_be_registered_through_public_api(self) -> None:
        def dummy() -> None:
            return None

        agent = make_react_agent([])
        with pytest.raises(ToolRegistrationError):
            agent.register_tool(dummy, alias="return")
        with pytest.raises(ToolRegistrationError):
            agent.register_tool(dummy, alias="make_sequence")


# --------------------------------------------------------------------------- #
# Full invoke() round trip
# --------------------------------------------------------------------------- #
class TestReActInvokeRoundTrip:
    def test_multi_round_invoke_dispatches_then_returns(self) -> None:
        agent = make_react_agent(
            [
                rstep(call="add", arguments=[arg("x", 2), arg("y", 3)], result_name="a"),
                rstep(call="multiply", arguments=[arg("x", "$a"), arg("y", 10)], result_name="b"),
                rstep(call="return", arguments=[arg("val", "$b")]),
            ],
            tool_calls_limit=2,
        )

        result = agent.invoke({"prompt": "run react"})

        assert result.result == 50

    def test_async_invoke_executes_rounds_and_returns_value(self) -> None:
        agent = make_react_agent(
            [
                rstep(call="add", arguments=[arg("x", 2), arg("y", 3)], result_name="a"),
                rstep(call="multiply", arguments=[arg("x", "$a"), arg("y", 10)], result_name="b"),
                rstep(call="return", arguments=[arg("val", "$b")]),
            ],
            tool_calls_limit=2,
        )

        result = asyncio.run(agent.async_invoke({"prompt": "run react"}))

        assert result.result == 50

    def test_single_round_return_with_literal_value(self) -> None:
        agent = make_react_agent([rstep(call="return", arguments=[arg("val", 42)])])

        result = agent.invoke({"prompt": "run"})

        assert result.result == 42

    def test_return_null_is_a_legitimate_result(self) -> None:
        agent = make_react_agent([rstep(call="return", arguments=[arg("val", None)])])

        result = agent.invoke({"prompt": "run"})

        assert result.result is None


# --------------------------------------------------------------------------- #
# think()/prepare() handoff -- prepare() is a documented no-op for this family
# --------------------------------------------------------------------------- #
class TestReActThinkPrepareHandoff:
    def test_think_resolves_exactly_one_call_per_round(self) -> None:
        agent = make_react_agent(
            [rstep(call="add", arguments=[arg("x", 1), arg("y", 2)])],
            tool_calls_limit=1,
        )
        task = agent.create_task(agent.filter_inputs({"prompt": "run"}))
        task.user_prompt = "run"
        assert task.generated_step is None
        assert task.resolved_args is None

        task = agent.think(task)

        assert task.generated_step is not None
        assert task.generated_step.tool == "add"
        assert task.resolved_args == {"x": 1, "y": 2}

    def test_prepare_is_a_documented_noop_passthrough(self) -> None:
        agent = make_react_agent(
            [rstep(call="return", arguments=[arg("val", 1)])],
        )
        task = agent.create_task(agent.filter_inputs({"prompt": "run"}))
        task.user_prompt = "run"
        task = agent.think(task)
        generated_before = task.generated_step
        resolved_before = task.resolved_args

        returned = agent.prepare(task)

        assert returned is task
        assert task.generated_step is generated_before
        assert task.resolved_args is resolved_before

    def test_act_consumes_generated_step_and_resets_it(self) -> None:
        agent = make_react_agent(
            [rstep(call="add", arguments=[arg("x", 1), arg("y", 2)])],
            tool_calls_limit=1,
        )
        task = agent.create_task(agent.filter_inputs({"prompt": "run"}))
        task.user_prompt = "run"
        task = agent.think(task)
        task = agent.prepare(task)

        task = agent.act(task)

        assert task.generated_step is None
        assert task.resolved_args is None
        assert len(task.completed) == 1


# --------------------------------------------------------------------------- #
# Auto-naming: an unnamed successful call gets __rN__, contiguous by
# successful calls only; a return call's identifier is always forced None.
# --------------------------------------------------------------------------- #
class TestReActAutoNaming:
    def test_unnamed_successful_calls_get_contiguous_dunder_names(self) -> None:
        agent = make_react_agent(
            [
                rstep(call="add", arguments=[arg("x", 2), arg("y", 3)]),
                rstep(call="multiply", arguments=[arg("x", "$__r0__"), arg("y", 10)]),
                rstep(call="return", arguments=[arg("val", "$__r1__")]),
            ],
            tool_calls_limit=2,
            record_enabled=True,
        )

        result = agent.invoke({"prompt": "run"})

        assert result.result == 50
        record = agent.get_conversation().records[-1]
        identifiers = [s.identifier for s in record.statements]
        assert identifiers == ["__r0__", "__r1__", None]

    def test_failed_attempt_does_not_consume_a_dunder_number(self) -> None:
        agent = make_react_agent(
            [
                rstep(call="fail_tool", arguments=[]),
                rstep(call="add", arguments=[arg("x", 1), arg("y", 2)]),
                rstep(call="return", arguments=[arg("val", "$__r0__")]),
            ],
            tool_calls_limit=2,
            fail_fast=False,
            record_enabled=True,
        )

        result = agent.invoke({"prompt": "run"})

        assert result.result == 3
        record = agent.get_conversation().records[-1]
        identifiers = [s.identifier for s in record.statements]
        assert identifiers == ["__r0__", None]

    def test_return_call_identifier_forced_to_none_even_when_model_names_it(self) -> None:
        agent = make_react_agent(
            [rstep(call="return", arguments=[arg("val", 42)], result_name="my_result")],
            record_enabled=True,
        )

        result = agent.invoke({"prompt": "run"})

        assert result.result == 42
        record = agent.get_conversation().records[-1]
        assert record.statements[-1].tool == "return"
        assert record.statements[-1].identifier is None

    def test_explicitly_named_successful_call_keeps_its_own_name(self) -> None:
        agent = make_react_agent(
            [
                rstep(call="add", arguments=[arg("x", 2), arg("y", 3)], result_name="sum_value"),
                rstep(call="return", arguments=[arg("val", "$sum_value")]),
            ],
            tool_calls_limit=1,
            record_enabled=True,
        )

        result = agent.invoke({"prompt": "run"})

        assert result.result == 5
        record = agent.get_conversation().records[-1]
        assert record.statements[0].identifier == "sum_value"


# --------------------------------------------------------------------------- #
# $name sigil resolution against task.context
# --------------------------------------------------------------------------- #
class TestReActSigilResolution:
    def test_whole_match_substitutes_real_typed_value(self) -> None:
        agent = make_react_agent(
            [
                rstep(call="add", arguments=[arg("x", 2), arg("y", 3)], result_name="a"),
                rstep(call="multiply", arguments=[arg("x", "$a"), arg("y", 2)], result_name="b"),
                rstep(call="return", arguments=[arg("val", "$b")]),
            ],
            tool_calls_limit=2,
        )

        result = agent.invoke({"prompt": "run"})

        assert result.result == 10

    def test_embedded_reference_interpolates_as_text(self) -> None:
        agent = make_react_agent(
            [
                rstep(call="add", arguments=[arg("x", 2), arg("y", 3)], result_name="a"),
                rstep(call="return", arguments=[arg("val", "sum is $a")]),
            ],
            tool_calls_limit=1,
        )

        result = agent.invoke({"prompt": "run"})

        assert result.result == "sum is 5"

    def test_unmatched_embedded_sigil_is_left_as_literal_text(self) -> None:
        agent = make_react_agent(
            [rstep(call="return", arguments=[arg("val", "nothing here: $ghost")])],
        )

        result = agent.invoke({"prompt": "run"})

        assert result.result == "nothing here: $ghost"

    def test_auto_named_result_resolvable_by_dunder_name_next_round(self) -> None:
        agent = make_react_agent(
            [
                rstep(call="add", arguments=[arg("x", 2), arg("y", 3)]),
                rstep(call="return", arguments=[arg("val", "$__r0__")]),
            ],
            tool_calls_limit=1,
        )

        result = agent.invoke({"prompt": "run"})

        assert result.result == 5


# --------------------------------------------------------------------------- #
# Generation retry loop: unresolvable $name / malformed call never lands in
# failed_statements -- caught and regenerated inside think()'s own loop.
# --------------------------------------------------------------------------- #
class TestReActGenerationRetry:
    def test_unresolvable_reference_triggers_regeneration_then_succeeds(self) -> None:
        invalid = rstep(call="return", arguments=[arg("val", "$missing")])
        valid = rstep(call="return", arguments=[arg("val", 7)])
        agent = make_react_agent(
            [invalid, valid],
            regeneration_limit=1,
            record_enabled=True,
        )

        result = agent.invoke({"prompt": "run"})

        assert result.result == 7
        record = agent.get_conversation().records[-1]
        assert record.regenerations_used == 1
        assert len(record.llm_records) == 2

    def test_unresolvable_reference_never_lands_in_failed_statements(self) -> None:
        invalid = rstep(call="return", arguments=[arg("val", "$missing")])
        valid = rstep(call="return", arguments=[arg("val", 7)])
        agent = make_react_agent(
            [invalid, valid],
            regeneration_limit=1,
            record_enabled=True,
        )

        agent.invoke({"prompt": "run"})

        record = agent.get_conversation().records[-1]
        assert record.failed_statements == ()

    def test_unresolvable_reference_budget_exhausted_raises(self) -> None:
        invalid = rstep(call="return", arguments=[arg("val", "$missing")])
        agent = make_react_agent([invalid], regeneration_limit=0)

        with pytest.raises(ToolAgentError, match="regeneration budget exhausted"):
            agent.invoke({"prompt": "run"})

    def test_unknown_tool_triggers_regeneration_then_succeeds(self) -> None:
        invalid = rstep(call="nonexistent_tool", arguments=[])
        valid = rstep(call="return", arguments=[arg("val", 42)])
        agent = make_react_agent([invalid, valid], regeneration_limit=1)

        result = agent.invoke({"prompt": "run"})

        assert result.result == 42

    def test_tool_calls_limit_exceeded_triggers_regeneration_then_succeeds(self) -> None:
        # At the budget boundary (tool_calls_used >= limit), the schema's
        # enum narrows to return-only -- but a scripted response can still
        # name a real tool outside that enum; translate_calls' own budget
        # check (remaining_budget <= 0) is the structural backstop tested
        # here directly via a plain over-budget attempt.
        over_budget = rstep(call="add", arguments=[arg("x", 1), arg("y", 2)])
        valid = rstep(call="return", arguments=[arg("val", 1)])
        agent = make_react_agent(
            [over_budget, valid],
            tool_calls_limit=0,
            regeneration_limit=1,
        )

        result = agent.invoke({"prompt": "run"})

        assert result.result == 1


# --------------------------------------------------------------------------- #
# Real dispatch failure: fail_fast=False tolerates, fail_fast=True raises.
# --------------------------------------------------------------------------- #
class TestReActDispatchFailure:
    def test_fail_fast_false_tolerates_failure_and_continues(self) -> None:
        agent = make_react_agent(
            [
                rstep(call="fail_tool", arguments=[]),
                rstep(call="add", arguments=[arg("x", 1), arg("y", 2)]),
                rstep(call="return", arguments=[arg("val", "$__r0__")]),
            ],
            tool_calls_limit=2,
            fail_fast=False,
            record_enabled=True,
        )

        result = agent.invoke({"prompt": "run"})

        assert result.result == 3
        record = agent.get_conversation().records[-1]
        assert len(record.failed_statements) == 1
        assert record.failed_statements[0].tool == "fail_tool"

    def test_fail_fast_true_raises_tool_invocation_error(self) -> None:
        agent = make_react_agent(
            [rstep(call="fail_tool", arguments=[])],
            fail_fast=True,
        )

        with pytest.raises(ToolInvocationError):
            agent.invoke({"prompt": "run"})

    def test_last_call_failed_flag_set_on_failure_and_cleared_on_success(self) -> None:
        agent = make_react_agent(
            [
                rstep(call="fail_tool", arguments=[]),
                rstep(call="add", arguments=[arg("x", 1), arg("y", 2)]),
            ],
            fail_fast=False,
        )
        task = agent.create_task(agent.filter_inputs({"prompt": "run"}))
        task.user_prompt = "run"
        task = agent.think(task)
        task = agent.act(task)

        assert task.last_call_failed is True
        assert len(task.failed_statements) == 1
        assert task.failed_statements[0].exception is not None
        assert task.completed == []

        task = agent.act(agent.prepare(agent.think(task)))

        assert task.last_call_failed is False
        assert len(task.completed) == 1
        assert len(task.failed_statements) == 1


# --------------------------------------------------------------------------- #
# Record / result shape
# --------------------------------------------------------------------------- #
class TestReActRecordAndResultShape:
    def test_successful_run_statements_and_usage_report(self) -> None:
        agent = make_react_agent(
            [
                rstep(call="add", arguments=[arg("x", 2), arg("y", 3)], result_name="a"),
                rstep(call="add", arguments=[arg("x", "$a"), arg("y", 1)], result_name="b"),
                rstep(call="return", arguments=[arg("val", "$b")]),
            ],
            tool_calls_limit=2,
            record_enabled=True,
        )

        result = agent.invoke({"prompt": "run"})

        assert result.result == 6

        record = agent.get_conversation().records[-1]
        assert len(record.statements) == 3  # a, b, return
        assert record.failed_statements == ()
        assert record.regenerations_used == 0

        by_tool = {u.tool_name: u.call_count for u in result.usage_report.by_tool}
        assert by_tool == {"add": 2}
        assert result.usage_report.total_dispatched == 2
        assert result.usage_report.total_failed == 0
        assert result.failed_call_count == 0
        assert result.regenerations_used == 0

    def test_failed_run_records_failed_statement_and_usage(self) -> None:
        agent = make_react_agent(
            [
                rstep(call="fail_tool", arguments=[]),
                rstep(call="add", arguments=[arg("x", 1), arg("y", 2)]),
                rstep(call="return", arguments=[arg("val", "$__r0__")]),
            ],
            tool_calls_limit=2,
            fail_fast=False,
            record_enabled=True,
        )

        result = agent.invoke({"prompt": "run"})

        assert result.result == 3
        assert result.failed_call_count == 1
        assert result.usage_report.total_failed == 1

    def test_llm_record_system_prompt_name_is_reason_then_act(self) -> None:
        agent = make_react_agent(
            [rstep(call="return", arguments=[arg("val", 1)])],
            record_enabled=True,
        )

        agent.invoke({"prompt": "run"})

        for rec in agent.get_conversation().records[-1].llm_records:
            assert rec.system_prompt_name == "reason_then_act"


class TestReActResultModelData:
    def test_zero_record_task_builds_result_with_engine_model_data(self) -> None:
        agent = make_react_agent([])
        task = agent.create_task(agent.filter_inputs({"prompt": "run"}))
        task.user_prompt = "run"
        task.ended_at = task.started_at

        result = agent.build_result_from_task(task, "done")

        assert result.llm_token_usage == ()
        assert result.llm_model_data == agent.llm_engine.model_data
        assert result.failed_call_count == 0
        assert result.regenerations_used == 0


# --------------------------------------------------------------------------- #
# Context binding: references, readonly / reserved names, persistence rules
# --------------------------------------------------------------------------- #
class TestReActContextBinding:
    def test_readonly_context_variable_resolves(self) -> None:
        agent = make_react_agent(
            [rstep(call="add", arguments=[arg("x", "$limit"), arg("y", 1)], result_name="a"),
             rstep(call="return", arguments=[arg("val", "$a")])],
            tool_calls_limit=1,
        )
        agent.get_conversation().remember("limit", 10, readonly=True)

        assert agent.invoke({"prompt": "run"}).result == 11

    def test_writable_context_variable_resolves(self) -> None:
        agent = make_react_agent([rstep(call="return", arguments=[arg("val", "$notes")])])
        agent.get_conversation().remember("notes", "draft")

        assert agent.invoke({"prompt": "run"}).result == "draft"

    def test_prior_turn_result_resolves_by_task_result_name(self) -> None:
        agent = make_react_agent(
            [rstep(call="return", arguments=[arg("val", 42)]),
             rstep(call="return", arguments=[arg("val", "$task_result_0")])],
            record_enabled=True,
        )
        agent.invoke({"prompt": "first"})

        assert agent.invoke({"prompt": "second"}).result == 42

    def test_undeclared_new_result_name_is_not_persisted(self) -> None:
        agent = make_react_agent(
            [rstep(call="add", arguments=[arg("x", 2), arg("y", 3)], result_name="scratch"),
             rstep(call="return", arguments=[arg("val", "$scratch")])],
            tool_calls_limit=1,
            record_enabled=True,
        )

        assert agent.invoke({"prompt": "run"}).result == 5
        assert agent.get_conversation().context == {}

    def test_writable_variable_updated_by_result_name_is_persisted(self) -> None:
        agent = make_react_agent(
            [rstep(call="add", arguments=[arg("x", 2), arg("y", 3)], result_name="total"),
             rstep(call="return", arguments=[arg("val", "$total")])],
            tool_calls_limit=1,
            record_enabled=True,
        )
        agent.get_conversation().remember("total", 0, "Running total")

        agent.invoke({"prompt": "run"})

        variable = agent.get_conversation().context["total"]
        assert variable.value == 5
        assert variable.description == "Running total"

    def test_writable_variable_may_be_read_by_the_call_that_updates_it(self) -> None:
        agent = make_react_agent(
            [rstep(call="add", arguments=[arg("x", "$total"), arg("y", 1)], result_name="total"),
             rstep(call="return", arguments=[arg("val", "$total")])],
            tool_calls_limit=1,
        )
        agent.get_conversation().remember("total", 10)

        assert agent.invoke({"prompt": "run"}).result == 11

    def test_auto_named_result_is_in_context_and_not_persisted(self) -> None:
        agent = make_react_agent(
            [rstep(call="add", arguments=[arg("x", 2), arg("y", 3)])],
            tool_calls_limit=1,
        )
        task = agent.create_task(agent.filter_inputs({"prompt": "run"}))
        task.user_prompt = "run"
        task = agent.act(agent.prepare(agent.think(task)))

        assert task.context["__r0__"].value == 5
        assert "__r0__" in task.changed_context

        persisted = make_react_agent(
            [rstep(call="add", arguments=[arg("x", 2), arg("y", 3)]),
             rstep(call="return", arguments=[arg("val", "$__r0__")])],
            tool_calls_limit=1,
            record_enabled=True,
        )
        persisted.invoke({"prompt": "run"})
        assert persisted.get_conversation().context == {}

    @pytest.mark.parametrize("bad_name", ["limit", "task_result_9", "__x__"])
    def test_rejected_result_name_is_regenerated(self, bad_name: str) -> None:
        bad = rstep(call="add", arguments=[arg("x", 1), arg("y", 2)], result_name=bad_name)
        good = rstep(call="add", arguments=[arg("x", 1), arg("y", 2)], result_name="a")
        agent = make_react_agent(
            [bad, good, rstep(call="return", arguments=[arg("val", "$a")])],
            tool_calls_limit=1,
            regeneration_limit=1,
            record_enabled=True,
        )
        agent.get_conversation().remember("limit", 10, readonly=True)

        assert agent.invoke({"prompt": "run"}).result == 3
        assert agent.get_conversation().records[-1].regenerations_used == 1

    def test_rejected_result_name_with_no_regeneration_budget_raises(self) -> None:
        bad = rstep(call="add", arguments=[arg("x", 1), arg("y", 2)], result_name="task_result_9")
        agent = make_react_agent([bad], tool_calls_limit=1, regeneration_limit=0)

        with pytest.raises(ToolAgentError, match="regeneration budget exhausted"):
            agent.invoke({"prompt": "run"})

    def test_repeated_keyword_name_is_regenerated(self) -> None:
        bad = rstep(call="add", arguments=[arg("x", 1), arg("x", 2), arg("y", 3)])
        good = rstep(call="return", arguments=[arg("val", 1)])
        agent = make_react_agent([bad, good], tool_calls_limit=1, regeneration_limit=1)

        assert agent.invoke({"prompt": "run"}).result == 1


# --------------------------------------------------------------------------- #
# Rendering: base frame (task block / STEPS / directive)
# --------------------------------------------------------------------------- #
class TestReActRendering:
    @staticmethod
    def _two_round_agent(**kwargs: Any) -> Any:
        agent = make_react_agent(
            [
                rstep(call="add", arguments=[arg("x", 2), arg("y", 3)], result_name="a"),
                rstep(call="return", arguments=[arg("val", "$a")]),
            ],
            **kwargs,
        )
        agent.invoke({"prompt": "do the thing"})
        return agent

    def test_round_one_is_one_user_message_ending_with_the_directive(self) -> None:
        agent = self._two_round_agent(tool_calls_limit=3, record_enabled=True)

        messages = agent._llm_engine.calls[0]

        users = [m for m in messages if m["role"] == "user"]
        assert len(users) == 1
        assert "USER TASK PROMPT" in users[0]["content"]
        assert "STEPS" not in users[0]["content"]
        assert "CURRENT TASK" not in users[0]["content"]
        assert users[0]["content"].rstrip().endswith("Tool calls remaining: 3 of 3.")
        record_messages = agent.get_conversation().records[-1].llm_records[0].messages
        assert [m for m in record_messages if m["role"] == "user"] == users

    def test_round_two_has_task_block_steps_and_directive(self) -> None:
        agent = self._two_round_agent(tool_calls_limit=3)

        messages = [m for m in agent._llm_engine.calls[1] if m["role"] != "system"]

        assert [m["role"] for m in messages] == ["user", "assistant", "user"]
        assert "STEPS" in messages[1]["content"]
        assert '"call": "add"' in messages[1]["content"]
        assert "Tool calls remaining: 2 of 3." in messages[2]["content"]

    def test_bound_result_appears_in_round_two_context_block(self) -> None:
        agent = self._two_round_agent(tool_calls_limit=3)

        task_block = next(
            m["content"] for m in agent._llm_engine.calls[1] if m["role"] == "user"
        )

        assert "RUNTIME CONTEXT" in task_block
        assert "name: $a" in task_block
        assert "value:\n5" in task_block

    def test_hidden_context_limit_shows_value_hidden(self) -> None:
        agent = make_react_agent(
            [
                rstep(call="add", arguments=[arg("x", 2), arg("y", 3)], result_name="a"),
                rstep(call="return", arguments=[arg("val", "$a")]),
            ],
            tool_calls_limit=2,
        )
        agent.get_conversation().context_preview_limit = 0

        agent.invoke({"prompt": "run"})

        task_block = next(m["content"] for m in agent._llm_engine.calls[1] if m["role"] == "user")
        assert "name: $a" in task_block
        assert "value hidden" in task_block

    def test_failure_shows_in_steps_and_directive_then_clears(self) -> None:
        agent = make_react_agent(
            [
                rstep(call="fail_tool", arguments=[]),
                rstep(call="add", arguments=[arg("x", 1), arg("y", 2)]),
                rstep(call="return", arguments=[arg("val", "$__r0__")]),
            ],
            tool_calls_limit=3,
            fail_fast=False,
        )

        agent.invoke({"prompt": "run"})

        second = [m for m in agent._llm_engine.calls[1] if m["role"] != "system"]
        assert "YOUR LAST CALL FAILED" in second[1]["content"]
        assert "intentional failure" in second[1]["content"]
        assert "Your last call failed" in second[2]["content"]
        third = "\n".join(
            m["content"] for m in agent._llm_engine.calls[2] if m["role"] != "system"
        )
        assert "YOUR LAST CALL FAILED" not in third
        assert "Your last call failed" not in third

    def test_budget_line_is_absent_without_a_limit(self) -> None:
        agent = self._two_round_agent(tool_calls_limit=None)

        assert "Tool calls remaining" not in agent._llm_engine.calls[0][-1]["content"]

    def test_budget_exhausted_directive_and_schema_narrow_to_return(self) -> None:
        agent = self._two_round_agent(tool_calls_limit=1)

        assert "Only the return tool is available now." in agent._llm_engine.calls[1][-1]["content"]
        assert agent._llm_engine.payloads[1]["output_structure"]["properties"]["call"]["enum"] == ["return"]
        first_enum = agent._llm_engine.payloads[0]["output_structure"]["properties"]["call"]["enum"]
        assert "return" in first_enum
        assert "add" in first_enum

    def test_context_variable_renders_with_dollar_prefix(self) -> None:
        agent = make_react_agent([rstep(call="return", arguments=[arg("val", 1)])])
        agent.get_conversation().remember("limit", 10, "A fixed limit.", readonly=True)

        agent.invoke({"prompt": "run"})

        user = agent._llm_engine.calls[0][-1]["content"]
        assert "name: $limit" in user
        assert "RUNTIME CONTEXT" in user

    def test_system_message_frames_tools_and_user_instructions(self) -> None:
        agent = make_react_agent(
            [rstep(call="return", arguments=[arg("val", 1)])], instructions="Be terse.",
        )

        agent.invoke({"prompt": "run"})

        system = next(m["content"] for m in agent._llm_engine.calls[0] if m["role"] == "system")
        assert "TOOL-CALLING INSTRUCTIONS" in system
        assert "AVAILABLE TOOLS" in system
        assert "USER-DEFINED INSTRUCTIONS" in system
        assert "Be terse." in system


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
            [rstep(call="add", arguments=[arg("x", 2), arg("y", 3)], result_name="a"),
             rstep(call="return", arguments=[arg("val", "$a")])],
            tool_calls_limit=1,
            post_invoke=shout,
            response_to_post="result",
        )

        assert agent.invoke({"prompt": "go"}).result == "5!"

    def test_response_to_post_selector_slices_the_response(self) -> None:
        agent = make_react_agent(
            [rstep(call="make_sequence", arguments=[arg(None, 4), arg(None, 5)], result_name="pair"),
             rstep(call="return", arguments=[arg("val", "$pair")])],
            tool_calls_limit=1,
            post_invoke=second_item,
            response_to_post={1: "second"},
        )

        assert agent.invoke({"prompt": "go"}).result == "second=5"

    def test_response_to_post_and_context_can_route_together(self) -> None:
        agent = make_react_agent(
            [rstep(call="make_sequence", arguments=[arg(None, 4), arg(None, 5)], result_name="pair"),
             rstep(call="return", arguments=[arg("val", "$pair")])],
            tool_calls_limit=1,
            record_enabled=True,
            post_invoke=second_item,
            response_to_post={1: "second"},
            response_to_context={0: "first"},
        )

        result = agent.invoke({"prompt": "go"})

        assert result.result == "second=5"
        assert agent.get_conversation().context["first"].value == 4

    def test_async_post_invoke_routing(self) -> None:
        agent = make_react_agent(
            [rstep(call="make_sequence", arguments=[arg(None, 4), arg(None, 5)], result_name="pair"),
             rstep(call="return", arguments=[arg("val", "$pair")])],
            tool_calls_limit=1,
            post_invoke=second_item,
            response_to_post={1: "second"},
        )

        result = asyncio.run(agent.async_invoke({"prompt": "go"}))

        assert result.result == "second=5"


# --------------------------------------------------------------------------- #
# Async paths, bind mismatches, and failure edge cases
# --------------------------------------------------------------------------- #
class TestReActAsyncAndFailureEdges:
    def test_async_retry_loop_regenerates_then_succeeds(self) -> None:
        agent = make_react_agent(
            [
                rstep(call="return", arguments=[arg("val", "$missing")]),
                rstep(call="return", arguments=[arg("val", 7)]),
            ],
            regeneration_limit=1,
            record_enabled=True,
        )

        result = asyncio.run(agent.async_invoke({"prompt": "run"}))

        assert result.result == 7
        assert agent.get_conversation().records[-1].regenerations_used == 1

    def test_async_retry_loop_budget_exhausted_raises(self) -> None:
        agent = make_react_agent(
            [rstep(call="return", arguments=[arg("val", "$missing")])],
            regeneration_limit=0,
        )

        with pytest.raises(ToolAgentError, match="regeneration budget exhausted"):
            asyncio.run(agent.async_invoke({"prompt": "run"}))

    def test_async_dispatch_failure_is_tolerated_when_not_fail_fast(self) -> None:
        agent = make_react_agent(
            [
                rstep(call="fail_tool", arguments=[]),
                rstep(call="return", arguments=[arg("val", 1)]),
            ],
            tool_calls_limit=2,
            fail_fast=False,
            record_enabled=True,
        )

        result = asyncio.run(agent.async_invoke({"prompt": "run"}))

        assert result.result == 1
        assert result.failed_call_count == 1

    def test_async_dispatch_failure_raises_when_fail_fast(self) -> None:
        agent = make_react_agent([rstep(call="fail_tool", arguments=[])], fail_fast=True)

        with pytest.raises(ToolInvocationError):
            asyncio.run(agent.async_invoke({"prompt": "run"}))

    def test_fail_fast_wraps_a_non_tool_invocation_error(self) -> None:
        agent = make_react_agent([], fail_fast=True)
        task = agent.create_task(agent.filter_inputs({"prompt": "run"}))
        call = ToolStatement(identifier=None, tool="add")

        with pytest.raises(ToolAgentError, match="tool call failed"):
            agent._apply_call_result(task, call, RuntimeError("boom"))

    def test_unknown_keyword_is_a_bind_mismatch_and_regenerates(self) -> None:
        bad = rstep(call="add", arguments=[arg("x", 1), arg("zzz", 2)])
        good = rstep(call="return", arguments=[arg("val", 1)])
        agent = make_react_agent(
            [bad, good], tool_calls_limit=1, regeneration_limit=1, record_enabled=True,
        )

        assert agent.invoke({"prompt": "run"}).result == 1
        record = agent.get_conversation().records[-1]
        assert record.regenerations_used == 1
        assert record.failed_statements == ()

    def test_failed_call_does_not_update_a_writable_variable(self) -> None:
        agent = make_react_agent(
            [
                rstep(call="fail_tool", arguments=[], result_name="total"),
                rstep(call="return", arguments=[arg("val", "$total")]),
            ],
            tool_calls_limit=2,
            fail_fast=False,
            record_enabled=True,
        )
        agent.get_conversation().remember("total", 7)

        result = agent.invoke({"prompt": "run"})

        assert result.result == 7
        assert agent.get_conversation().context["total"].value == 7

    def test_failed_call_still_spends_budget_in_the_directive(self) -> None:
        agent = make_react_agent(
            [
                rstep(call="fail_tool", arguments=[]),
                rstep(call="return", arguments=[arg("val", 1)]),
            ],
            tool_calls_limit=3,
            fail_fast=False,
        )

        agent.invoke({"prompt": "run"})

        directive = agent._llm_engine.calls[1][-1]["content"]
        assert "Your last call failed" in directive
        assert "Tool calls remaining: 2 of 3." in directive
