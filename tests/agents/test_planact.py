from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from atomic_agentic.agents.planact import PlanActAgent
from atomic_agentic.exceptions import ToolAgentError, ToolInvocationError, ToolRegistrationError

from .conftest import (
    FakeLLMEngine,
    arg,
    make_planact_agent,
    planact_plan_json,
)


def scripted(*, plan: list[dict[str, Any]], return_value: Any = None, summary: str = "Running the plan.") -> dict[str, Any]:
    """Build one scripted FakeLLMEngine response as a real dict.

    ``FakeLLMEngine`` never parses its scripted responses -- whatever object
    sits in its ``responses`` list becomes ``LLMResult.result`` verbatim (a
    real engine's own ``_extract_result`` does the ``json.loads`` when
    ``output_structure`` was requested; the fake intentionally does not
    reproduce that). ``conftest.planact_plan_json`` builds the correct
    wire-shaped payload as JSON *text*; this helper round-trips it back into
    a plain dict so it can be handed to ``FakeLLMEngine`` directly.
    """
    return json.loads(planact_plan_json(plan=plan, return_value=return_value, summary=summary))


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #
class TestPlanActConstruction:
    def test_fail_fast_defaults_true(self) -> None:
        agent = make_planact_agent([])
        assert agent.fail_fast is True

    def test_fail_fast_forwarded_false(self) -> None:
        agent = make_planact_agent([], fail_fast=False)
        assert agent.fail_fast is False

    def test_fail_fast_must_be_bool(self) -> None:
        with pytest.raises(ToolAgentError):
            PlanActAgent(
                name="tests",
                namespace="tests",
                description=".",
                llm_engine=FakeLLMEngine([]),
                fail_fast="yes",  # type: ignore[arg-type]
            )

    def test_tool_calls_limit_and_regeneration_limit_forwarded(self) -> None:
        agent = make_planact_agent([], tool_calls_limit=3, regeneration_limit=2)
        assert agent.tool_calls_limit == 3
        assert agent.regeneration_limit == 2

    def test_reserved_tools_registered_on_construction(self) -> None:
        agent = make_planact_agent([])
        assert agent.has_tool("make_sequence")
        assert agent.has_tool("make_dict")

    def test_reserved_tool_names_cannot_be_registered_through_public_api(self) -> None:
        def dummy() -> None:
            return None

        agent = make_planact_agent([])
        with pytest.raises(ToolRegistrationError):
            agent.register_tool(dummy, alias="make_sequence")


# --------------------------------------------------------------------------- #
# Full invoke() round trip
# --------------------------------------------------------------------------- #
class TestPlanActInvokeRoundTrip:
    def test_multi_step_plan_with_dependencies_and_concurrency(self) -> None:
        plan = [
            {"call": "add", "arguments": [arg("x", 2), arg("y", 3)], "result_name": "a"},
            {"call": "multiply", "arguments": [arg("x", 4), arg("y", 5)], "result_name": "b"},
            {"call": "add", "arguments": [arg("x", "$a"), arg("y", "$b")], "result_name": "c"},
        ]
        agent = make_planact_agent(
            [scripted(plan=plan, return_value="$c")],
            context_enabled=True,
        )

        result = agent.invoke({"prompt": "run plan"})

        assert result.result == 25

        record = agent.get_conversation()[-1]
        by_identifier = {s.identifier: s.batch_index for s in record.statements if s.identifier is not None}
        # a and b are independent -- same concurrency batch.
        assert by_identifier["a"] == by_identifier["b"]
        # c depends on both -- a strictly later batch.
        assert by_identifier["c"] > by_identifier["a"]

    def test_single_call_plan_returns_value(self) -> None:
        plan = [{"call": "add", "arguments": [arg("x", 1), arg("y", 2)], "result_name": "a"}]
        agent = make_planact_agent([scripted(plan=plan, return_value="$a")])

        result = agent.invoke({"prompt": "run"})

        assert result.result == 3

    def test_async_invoke_executes_plan_and_returns_value(self) -> None:
        plan = [
            {"call": "add", "arguments": [arg("x", 2), arg("y", 3)], "result_name": "a"},
            {"call": "multiply", "arguments": [arg("x", "$a"), arg("y", 10)], "result_name": "b"},
        ]
        agent = make_planact_agent([scripted(plan=plan, return_value="$b")])

        result = asyncio.run(agent.async_invoke({"prompt": "run plan"}))

        assert result.result == 50


# --------------------------------------------------------------------------- #
# $name sigil resolution
# --------------------------------------------------------------------------- #
class TestPlanActSigilResolution:
    def test_whole_match_substitutes_real_typed_value(self) -> None:
        plan = [{"call": "add", "arguments": [arg("x", 2), arg("y", 3)], "result_name": "a"}]
        # "$a" used as a real int operand downstream -- proves the real type
        # (not a stringified form) was substituted.
        plan.append({"call": "multiply", "arguments": [arg("x", "$a"), arg("y", 2)], "result_name": "b"})
        agent = make_planact_agent([scripted(plan=plan, return_value="$b")])

        result = agent.invoke({"prompt": "run"})

        assert result.result == 10

    def test_embedded_reference_interpolates_as_text(self) -> None:
        plan = [{"call": "add", "arguments": [arg("x", 2), arg("y", 3)], "result_name": "a"}]
        agent = make_planact_agent([scripted(plan=plan, return_value="sum is $a")])

        result = agent.invoke({"prompt": "run"})

        assert result.result == "sum is 5"

    def test_unmatched_embedded_sigil_is_left_as_literal_text(self) -> None:
        agent = make_planact_agent([scripted(plan=[], return_value="nothing here: $ghost")])

        result = agent.invoke({"prompt": "run"})

        assert result.result == "nothing here: $ghost"

    def test_unmatched_whole_sigil_is_rejected_not_silently_literal(self) -> None:
        # Contrast with the embedded case: a *whole*-string unresolved
        # reference is a real validation issue (caught by validate/translate
        # via the regeneration loop), not treated as literal text.
        agent = make_planact_agent(
            [scripted(plan=[], return_value="$ghost")],
            regeneration_limit=0,
        )

        with pytest.raises(ToolAgentError, match="regeneration budget exhausted"):
            agent.invoke({"prompt": "run"})


# --------------------------------------------------------------------------- #
# Constant (K_*) and cross-turn (task_result_N) references
# --------------------------------------------------------------------------- #
class TestPlanActConstantAndCrossTurnReferences:
    def test_registered_constant_resolves_by_wire_name(self) -> None:
        agent = make_planact_agent([scripted(plan=[], return_value="$K_LIMIT")])
        agent.register_constant(10, alias="limit")

        result = agent.invoke({"prompt": "run"})

        assert result.result == 10

    def test_cross_turn_task_result_reference_resolves(self) -> None:
        first_plan = [{"call": "add", "arguments": [arg("x", 2), arg("y", 3)], "result_name": "a"}]
        agent = make_planact_agent(
            [
                scripted(plan=first_plan, return_value="$a"),
                scripted(plan=[], return_value="$task_result_0"),
            ],
            context_enabled=True,
        )

        first = agent.invoke({"prompt": "first"})
        second = agent.invoke({"prompt": "second"})

        assert first.result == 5
        assert second.result == 5


# --------------------------------------------------------------------------- #
# Generation retry loop (within-round regeneration on validation failure)
# --------------------------------------------------------------------------- #
class TestPlanActGenerationRetry:
    def test_undefined_reference_triggers_regeneration_then_succeeds(self) -> None:
        invalid = scripted(plan=[], return_value="$missing")
        valid = scripted(plan=[], return_value=7)
        agent = make_planact_agent(
            [invalid, valid],
            regeneration_limit=1,
            context_enabled=True,
        )

        result = agent.invoke({"prompt": "run"})

        assert result.result == 7
        record = agent.get_conversation()[-1]
        assert record.regenerations_used == 1
        assert len(record.llm_records) == 2

    def test_undefined_reference_budget_exhausted_raises(self) -> None:
        invalid = scripted(plan=[], return_value="$missing")
        agent = make_planact_agent([invalid], regeneration_limit=0)

        with pytest.raises(ToolAgentError, match="regeneration budget exhausted"):
            agent.invoke({"prompt": "run"})

    def test_tool_calls_limit_exceeded_triggers_regeneration_then_succeeds(self) -> None:
        too_many = scripted(
            plan=[
                {"call": "add", "arguments": [arg("x", 1), arg("y", 2)], "result_name": "a"},
                {"call": "multiply", "arguments": [arg("x", 3), arg("y", 4)], "result_name": "b"},
            ],
            return_value="$b",
        )
        ok = scripted(
            plan=[{"call": "add", "arguments": [arg("x", 1), arg("y", 2)], "result_name": "a"}],
            return_value="$a",
        )
        agent = make_planact_agent(
            [too_many, ok],
            tool_calls_limit=1,
            regeneration_limit=1,
            context_enabled=True,
        )

        result = agent.invoke({"prompt": "run"})

        assert result.result == 3
        assert agent.get_conversation()[-1].regenerations_used == 1

    def test_tool_calls_limit_exceeded_budget_exhausted_raises(self) -> None:
        too_many = scripted(
            plan=[
                {"call": "add", "arguments": [arg("x", 1), arg("y", 2)], "result_name": "a"},
                {"call": "multiply", "arguments": [arg("x", 3), arg("y", 4)], "result_name": "b"},
            ],
            return_value="$b",
        )
        agent = make_planact_agent([too_many], tool_calls_limit=1, regeneration_limit=0)

        with pytest.raises(ToolAgentError, match="exceeding the configured limit"):
            agent.invoke({"prompt": "run"})


# --------------------------------------------------------------------------- #
# Cascade failure propagation (fail_fast=False) vs. immediate raise (True)
# --------------------------------------------------------------------------- #
class TestPlanActCascadeFailures:
    def _cascade_plan(self) -> list[dict]:
        return [
            {"call": "fail_tool", "arguments": [], "result_name": "f"},
            {"call": "add", "arguments": [arg("x", "$f"), arg("y", 1)], "result_name": "dep"},
            {"call": "multiply", "arguments": [arg("x", 3), arg("y", 4)], "result_name": "indep"},
        ]

    def test_cascade_skips_dependents_independent_calls_still_run(self) -> None:
        agent = make_planact_agent(
            [scripted(plan=self._cascade_plan(), return_value="$indep")],
            fail_fast=False,
            context_enabled=True,
        )

        result = agent.invoke({"prompt": "cascade"})

        assert result.result == 12
        record = agent.get_conversation()[-1]
        assert [s.identifier for s in record.failed_statements] == ["f"]
        identifiers = [s.identifier for s in record.statements]
        assert "indep" in identifiers
        assert "dep" not in identifiers

    def test_return_depending_on_failed_call_raises(self) -> None:
        plan = [{"call": "fail_tool", "arguments": [], "result_name": "f"}]
        agent = make_planact_agent(
            [scripted(plan=plan, return_value="$f")],
            fail_fast=False,
        )

        with pytest.raises(ToolAgentError, match="return value"):
            agent.invoke({"prompt": "run"})

    def test_fail_fast_true_raises_immediately_no_cascade(self) -> None:
        agent = make_planact_agent(
            [scripted(plan=self._cascade_plan(), return_value="$indep")],
            fail_fast=True,
        )

        with pytest.raises(ToolInvocationError):
            agent.invoke({"prompt": "cascade"})


# --------------------------------------------------------------------------- #
# return-value handling: reference / literal / null
# --------------------------------------------------------------------------- #
class TestPlanActReturnValueHandling:
    def test_return_resolves_name_reference(self) -> None:
        plan = [{"call": "add", "arguments": [arg("x", 2), arg("y", 3)], "result_name": "a"}]
        agent = make_planact_agent([scripted(plan=plan, return_value="$a")])

        assert agent.invoke({"prompt": "run"}).result == 5

    def test_return_literal_value_with_empty_plan(self) -> None:
        agent = make_planact_agent([scripted(plan=[], return_value=42)])

        assert agent.invoke({"prompt": "run"}).result == 42

    def test_return_null(self) -> None:
        agent = make_planact_agent([scripted(plan=[], return_value=None)])

        assert agent.invoke({"prompt": "run"}).result is None


# --------------------------------------------------------------------------- #
# Record / result shape
# --------------------------------------------------------------------------- #
class TestPlanActRecordAndResultShape:
    def test_successful_run_statements_and_usage_report(self) -> None:
        plan = [
            {"call": "add", "arguments": [arg("x", 2), arg("y", 3)], "result_name": "a"},
            {"call": "add", "arguments": [arg("x", "$a"), arg("y", 1)], "result_name": "b"},
        ]
        agent = make_planact_agent(
            [scripted(plan=plan, return_value="$b")],
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

    def test_cascade_run_records_failed_statement_and_usage(self) -> None:
        plan = [
            {"call": "fail_tool", "arguments": [], "result_name": "f"},
            {"call": "add", "arguments": [arg("x", "$f"), arg("y", 1)], "result_name": "dep"},
            {"call": "multiply", "arguments": [arg("x", 3), arg("y", 4)], "result_name": "indep"},
        ]
        agent = make_planact_agent(
            [scripted(plan=plan, return_value="$indep")],
            fail_fast=False,
            context_enabled=True,
        )

        result = agent.invoke({"prompt": "run"})

        assert result.result == 12
        assert result.failed_call_count == 1
        assert result.usage_report.total_failed == 1
