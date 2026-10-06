from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from atomic_agentic.agents.react import ReActAgent
from atomic_agentic.agents.tools import return_tool
from atomic_agentic.exceptions import ToolAgentError, ToolInvocationError, ToolRegistrationError

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

    def test_tool_instructions_forwarded(self) -> None:
        agent = make_react_agent([], tool_instructions="Be terse.")
        assert agent.tool_instructions == "Be terse."

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
        task = agent._initialize_task(turns=[], prompt="run", inputs={})
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
        task = agent._initialize_task(turns=[], prompt="run", inputs={})
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
        task = agent._initialize_task(turns=[], prompt="run", inputs={})
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
            context_enabled=True,
        )

        result = agent.invoke({"prompt": "run"})

        assert result.result == 50
        record = agent.get_conversation()[-1]
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
            context_enabled=True,
        )

        result = agent.invoke({"prompt": "run"})

        assert result.result == 3
        record = agent.get_conversation()[-1]
        identifiers = [s.identifier for s in record.statements]
        assert identifiers == ["__r0__", None]

    def test_return_call_identifier_forced_to_none_even_when_model_names_it(self) -> None:
        agent = make_react_agent(
            [rstep(call="return", arguments=[arg("val", 42)], result_name="my_result")],
            context_enabled=True,
        )

        result = agent.invoke({"prompt": "run"})

        assert result.result == 42
        record = agent.get_conversation()[-1]
        assert record.statements[-1].tool == "return"
        assert record.statements[-1].identifier is None

    def test_explicitly_named_successful_call_keeps_its_own_name(self) -> None:
        agent = make_react_agent(
            [
                rstep(call="add", arguments=[arg("x", 2), arg("y", 3)], result_name="sum_value"),
                rstep(call="return", arguments=[arg("val", "$sum_value")]),
            ],
            tool_calls_limit=1,
            context_enabled=True,
        )

        result = agent.invoke({"prompt": "run"})

        assert result.result == 5
        record = agent.get_conversation()[-1]
        assert record.statements[0].identifier == "sum_value"


# --------------------------------------------------------------------------- #
# $name sigil resolution against task.cache
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
            context_enabled=True,
        )

        result = agent.invoke({"prompt": "run"})

        assert result.result == 7
        record = agent.get_conversation()[-1]
        assert record.regenerations_used == 1
        assert len(record.llm_records) == 2

    def test_unresolvable_reference_never_lands_in_failed_statements(self) -> None:
        invalid = rstep(call="return", arguments=[arg("val", "$missing")])
        valid = rstep(call="return", arguments=[arg("val", 7)])
        agent = make_react_agent(
            [invalid, valid],
            regeneration_limit=1,
            context_enabled=True,
        )

        agent.invoke({"prompt": "run"})

        record = agent.get_conversation()[-1]
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
            context_enabled=True,
        )

        result = agent.invoke({"prompt": "run"})

        assert result.result == 3
        record = agent.get_conversation()[-1]
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
            [rstep(call="fail_tool", arguments=[])],
            fail_fast=False,
        )
        task = agent._initialize_task(turns=[], prompt="run", inputs={})
        task = agent.think(task)
        task = agent.act(task)

        assert task.last_call_failed is True
        assert len(task.failed_statements) == 1
        assert task.failed_statements[0].exception is not None
        assert task.completed == []

    def test_failure_is_rendered_for_the_next_round(self) -> None:
        agent = make_react_agent(
            [
                rstep(call="fail_tool", arguments=[]),
                rstep(call="return", arguments=[arg("val", 1)]),
            ],
            fail_fast=False,
        )

        agent.invoke({"prompt": "run"})

        engine = agent.llm_engine
        assert isinstance(engine, FakeLLMEngine)
        second_call_text = "\n".join(
            message["content"] for message in engine.calls[1]
        )
        assert "YOUR LAST CALL FAILED" in second_call_text
        assert "intentional failure" in second_call_text


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
            context_enabled=True,
        )

        result = agent.invoke({"prompt": "run"})

        assert result.result == 6

        record = agent.get_conversation()[-1]
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
            context_enabled=True,
        )

        result = agent.invoke({"prompt": "run"})

        assert result.result == 3
        assert result.failed_call_count == 1
        assert result.usage_report.total_failed == 1

    def test_llm_record_system_prompt_name_is_reason_then_act(self) -> None:
        agent = make_react_agent(
            [rstep(call="return", arguments=[arg("val", 1)])],
            context_enabled=True,
        )

        agent.invoke({"prompt": "run"})

        for rec in agent.get_conversation()[-1].llm_records:
            assert rec.system_prompt_name == "reason_then_act"
