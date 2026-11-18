from __future__ import annotations

import asyncio
import warnings
from typing import Any

import pytest

from .conftest import FakeLLMEngine

from atomic_agentic.agents.basic import BasicAgent
from atomic_agentic.agents.scratchpad import ScratchpadAgent
from atomic_agentic.constants.agents import (
    DEFAULT_INSTRUCTIONS,
    DEFAULT_THINKING_INSTRUCTIONS,
    STOP_EARLY_PROMPT_SECTION,
    STOP_EARLY_SCHEMA_TEMPLATE,
)
from atomic_agentic.exceptions import AgentError, AgentInvocationError
from atomic_agentic.core.prompt_template import PromptTemplate
from atomic_agentic.models.agents.records import ThinkingAgentRecord
from atomic_agentic.models.agents.tasks import ScratchpadTask
from atomic_agentic.models.parameters import ParamSpec
from atomic_agentic.models.results.agents import AgentResult, ThinkingAgentResult


THINKING_INSTRUCTIONS_START = "==== USER-DEFINED THINKING INSTRUCTIONS [START] ===="
THINKING_INSTRUCTIONS_END = "==== USER-DEFINED THINKING INSTRUCTIONS [END] ===="
THOUGHTS_START = "==== THOUGHTS [START] ===="
THOUGHTS_END = "==== THOUGHTS [END] ===="

ROLE = "You are a careful assistant."
THINK = "Think hard."


def make_agent(
    responses: list[Any] | None = None,
    *,
    thinking_rounds: int = 1,
    instructions: Any = ROLE,
    thinking_instructions: Any = THINK,
    record_enabled: bool = False,
    cls: type[ScratchpadAgent] = ScratchpadAgent,
    **kwargs: Any,
) -> ScratchpadAgent:
    return cls(
        name="tests",
        description="ScratchpadAgent under test.",
        llm_engine=FakeLLMEngine(responses if responses is not None else []),
        instructions=instructions,
        thinking_instructions=thinking_instructions,
        record_enabled=record_enabled,
        thinking_rounds=thinking_rounds,
        **kwargs,
    )


def step(thought: Any, more: bool) -> dict[str, Any]:
    """One scripted ``stop_early`` engine result."""
    return {"thought": thought, "more_thinking_required": more}


def calls(agent: ScratchpadAgent) -> list[list[dict[str, str]]]:
    return agent._llm_engine.calls


def structures(agent: ScratchpadAgent) -> list[Any]:
    """The ``output_structure`` each engine call was made with."""
    return [p["output_structure"] for p in agent._llm_engine.payloads]


def last_record(agent: ScratchpadAgent) -> ThinkingAgentRecord:
    return agent.get_conversation().records[-1]


class _Spy(ScratchpadAgent):
    """Keeps every task ``_thinking_done`` sees."""

    def _thinking_done(self, task: ScratchpadTask) -> bool:
        self.tasks.append(task)
        return super()._thinking_done(task)


class TestConstruction:
    def test_is_instantiable_with_defaults(self) -> None:
        agent = make_agent()

        assert agent.thinking_rounds == 1
        assert agent.stop_early is False
        assert agent.response_schema is None

    def test_thinking_rounds_is_required(self) -> None:
        with pytest.raises(TypeError, match="thinking_rounds"):
            ScratchpadAgent(
                name="tests", description="d",
                llm_engine=FakeLLMEngine([]),
            )

    def test_zero_rounds_is_allowed(self) -> None:
        assert make_agent(thinking_rounds=0).thinking_rounds == 0

    @pytest.mark.parametrize("bad", [-1, True, "2", 1.5, None])
    def test_invalid_thinking_rounds_raises(self, bad: Any) -> None:
        with pytest.raises(AgentError, match="thinking_rounds"):
            make_agent(thinking_rounds=bad)

    @pytest.mark.parametrize("bad", [1, 0, "yes", None])
    def test_non_bool_stop_early_raises(self, bad: Any) -> None:
        with pytest.raises(AgentError, match="stop_early"):
            make_agent(stop_early=bad)

    @pytest.mark.parametrize("bad", [-1, True, "2", 1.5, None])
    def test_setting_invalid_thinking_rounds_raises_and_keeps_the_value(self, bad: Any) -> None:
        agent = make_agent(thinking_rounds=2)

        with pytest.raises(AgentError, match="thinking_rounds"):
            agent.thinking_rounds = bad

        assert agent.thinking_rounds == 2

    def test_thinking_rounds_is_mutable(self) -> None:
        agent = make_agent(["t1", "t2", "t3", "R"], thinking_rounds=1)

        agent.thinking_rounds = 3
        result = agent.invoke({"prompt": "hi"})

        assert agent.thinking_rounds == 3
        assert result.thoughts_produced == 3
        assert agent._llm_engine.call_count == 4

    def test_lowering_thinking_rounds_to_zero_skips_thinking(self) -> None:
        agent = make_agent(["R"], thinking_rounds=3)

        agent.thinking_rounds = 0
        result = agent.invoke({"prompt": "hi"})

        assert result.thoughts_produced == 0
        assert agent._llm_engine.call_count == 1

    def test_mutated_rounds_cap_a_stop_early_agent(self) -> None:
        agent = make_agent(
            [step("t1", True), step("t2", True), "R"], thinking_rounds=5, stop_early=True,
        )

        agent.thinking_rounds = 2
        result = agent.invoke({"prompt": "hi"})

        assert result.thoughts_produced == 2

    def test_response_schema_rejects_non_mapping(self) -> None:
        with pytest.raises(AgentError, match="response_schema"):
            make_agent(response_schema="not a dict")  # type: ignore[arg-type]

    def test_thinking_instructions_rejects_bad_type(self) -> None:
        with pytest.raises(TypeError, match="thinking_instructions"):
            make_agent(thinking_instructions=123)  # type: ignore[arg-type]

    def test_stop_early_is_read_only(self) -> None:
        agent = make_agent()

        with pytest.raises(AttributeError):
            agent.stop_early = True

    def test_thinking_rounds_is_not_an_invoke_input(self) -> None:
        agent = make_agent()

        assert [p.name for p in agent.parameters] == ["prompt"]

    def test_thinking_and_main_instructions_are_stored_as_slots(self) -> None:
        agent = make_agent()

        assert agent.thinking_instructions.template == THINK
        assert agent.thinking_instructions.name == "tests_thinking_instructions"
        assert agent.instructions.template == ROLE
        assert agent.instructions.name == "tests_instructions"
        assert not hasattr(agent, "_system_prompts")
        assert not hasattr(agent, "role_prompt")
        assert not hasattr(ScratchpadAgent, "DEFAULT_THINKING_PROMPT")

    def test_response_schema_gives_the_default_conversation_json_mode(self) -> None:
        schema = {"type": "object", "properties": {"answer": {"type": "string"}}}

        assert make_agent(response_schema=schema).get_conversation().render_mode == "j"
        assert make_agent().get_conversation().render_mode == "s"

    def test_to_dict_includes_the_thinking_knobs(self) -> None:
        d = make_agent(thinking_rounds=3, stop_early=True).to_dict()

        assert d["thinking_instructions"]["template"] == THINK
        assert d["thinking_rounds"] == 3
        assert d["stop_early"] is True
        assert d["instructions"]["template"] == ROLE
        assert "role_prompt" not in d
        assert "thinking_schema" not in d

    def test_the_stop_early_output_structure_is_built_once(self) -> None:
        agent = make_agent(thinking_rounds=2, stop_early=True)

        assert agent._thinking_output == STOP_EARLY_SCHEMA_TEMPLATE
        assert agent._thinking_output is not STOP_EARLY_SCHEMA_TEMPLATE
        assert make_agent()._thinking_output is None


class TestFixedRounds:
    def test_zero_rounds_makes_one_basic_agent_call(self) -> None:
        basic = BasicAgent(
            name="tests", description="basic",
            llm_engine=FakeLLMEngine(["R"]), instructions=ROLE,
        )
        agent = make_agent(["R"], thinking_rounds=0)

        basic.invoke({"prompt": "hi"})
        result = agent.invoke({"prompt": "hi"})

        assert calls(agent) == calls(basic)
        assert result.result == "R"
        assert result.thoughts_produced == 0

    def test_n_rounds_make_n_thinking_calls_then_the_reply(self) -> None:
        agent = make_agent(["t1", "t2", "R"], thinking_rounds=2)

        result = agent.invoke({"prompt": "hi"})

        assert result.result == "R"
        assert agent._llm_engine.call_count == 3
        assert [c[0]["content"] for c in calls(agent)] == [THINK, THINK, ROLE]

    def test_each_thinking_round_and_the_reply_has_an_llm_record(self) -> None:
        agent = make_agent(["t1", "t2", "R"], thinking_rounds=2, record_enabled=True)

        agent.invoke({"prompt": "hi"})

        assert len(last_record(agent).llm_records) == 3

    def test_string_thoughts_are_stripped(self) -> None:
        agent = make_agent(["  a thought \n", "R"], record_enabled=True)

        agent.invoke({"prompt": "hi"})

        assert last_record(agent).thoughts == ("a thought",)

    def test_thinking_calls_are_free_text_without_a_schema(self) -> None:
        agent = make_agent(["t1", "R"])

        agent.invoke({"prompt": "hi"})

        assert structures(agent) == [None, None]

    def test_response_schema_reaches_only_the_reply_call(self) -> None:
        reply_schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
        agent = make_agent(["t1", {"answer": "y"}], response_schema=reply_schema)

        result = agent.invoke({"prompt": "hi"})

        assert structures(agent) == [None, reply_schema]
        assert result.result == {"answer": "y"}

    def test_stop_flag_is_untouched_without_stop_early(self) -> None:
        agent = make_agent(["t1", "t2", "R"], thinking_rounds=2, cls=_Spy)
        agent.tasks = []

        agent.invoke({"prompt": "hi"})

        assert all(t is agent.tasks[0] for t in agent.tasks)
        assert agent.tasks[0].more_thinking_required is True
        assert len(agent.tasks[0].thoughts) == 2

    def test_each_invoke_starts_fresh(self) -> None:
        agent = make_agent(["t1", "R1", "t2", "R2"], record_enabled=True)

        agent.invoke({"prompt": "p1"})
        agent.invoke({"prompt": "p2"})

        records = agent.get_conversation().records
        assert [r.thoughts for r in records] == [("t1",), ("t2",)]
        # Earlier thoughts never enter later history.
        assert not any("t1" in m["content"] for m in calls(agent)[2])


class TestStopEarly:
    def test_every_thinking_call_uses_the_wrapper_schema(self) -> None:
        agent = make_agent(
            [step("t1", True), step("t2", False), "R"],
            thinking_rounds=4, stop_early=True,
        )

        agent.invoke({"prompt": "hi"})

        assert structures(agent) == [STOP_EARLY_SCHEMA_TEMPLATE, STOP_EARLY_SCHEMA_TEMPLATE, None]

    def test_stop_after_round_one(self) -> None:
        agent = make_agent(
            [step("t1", False), "R"], thinking_rounds=3, stop_early=True,
            record_enabled=True,
        )

        result = agent.invoke({"prompt": "hi"})

        assert agent._llm_engine.call_count == 2
        assert last_record(agent).thoughts == ("t1",)
        assert result.thoughts_produced == 1
        assert result.result == "R"

    def test_stop_after_a_later_round(self) -> None:
        agent = make_agent(
            [step("t1", True), step("t2", True), step("t3", False), "R"],
            thinking_rounds=5, stop_early=True, record_enabled=True,
        )

        result = agent.invoke({"prompt": "hi"})

        assert agent._llm_engine.call_count == 4
        assert last_record(agent).thoughts == ("t1", "t2", "t3")
        assert result.thoughts_produced == 3

    def test_never_stopping_ends_at_the_cap(self) -> None:
        agent = make_agent(
            [step("t1", True), step("t2", True), "R"],
            thinking_rounds=2, stop_early=True, record_enabled=True,
        )

        result = agent.invoke({"prompt": "hi"})

        assert agent._llm_engine.call_count == 3
        assert last_record(agent).thoughts == ("t1", "t2")
        assert result.thoughts_produced == 2

    def test_a_cap_of_one_still_runs_one_round(self) -> None:
        agent = make_agent([step("t1", True), "R"], thinking_rounds=1, stop_early=True)

        result = agent.invoke({"prompt": "hi"})

        assert agent._llm_engine.call_count == 2
        assert result.thoughts_produced == 1

    def test_a_cap_of_zero_makes_no_thinking_call(self) -> None:
        agent = make_agent(["R"], thinking_rounds=0, stop_early=True)

        result = agent.invoke({"prompt": "hi"})

        assert agent._llm_engine.call_count == 1
        assert structures(agent) == [None]
        assert result.thoughts_produced == 0

    def test_thoughts_are_unwrapped_and_the_llm_record_keeps_the_wrapper(self) -> None:
        wrapped = step("f", False)
        agent = make_agent(
            [wrapped, "R"], thinking_rounds=2, stop_early=True, record_enabled=True,
        )

        agent.invoke({"prompt": "hi"})

        record = last_record(agent)
        assert record.thoughts == ("f",)
        assert record.llm_records[0].llm_result.result == wrapped

    @pytest.mark.parametrize("bad", [
        "plain text",
        {"more_thinking_required": False},
        {"thought": "t"},
        {"thought": 5, "more_thinking_required": False},
        {"thought": "t", "more_thinking_required": "false"},
    ])
    def test_a_malformed_reply_raises_invocation_error(self, bad: Any) -> None:
        agent = make_agent([bad], thinking_rounds=2, stop_early=True)

        with pytest.raises(AgentInvocationError, match="more_thinking_required"):
            agent.invoke({"prompt": "hi"})

    def test_a_malformed_reply_raises_on_the_async_path_too(self) -> None:
        agent = make_agent([{"thought": "t"}], thinking_rounds=2, stop_early=True)

        with pytest.raises(AgentInvocationError, match="more_thinking_required"):
            asyncio.run(agent.async_invoke({"prompt": "hi"}))

    def test_string_thoughts_are_stripped(self) -> None:
        agent = make_agent(
            [step("  padded \n", False), "R"], thinking_rounds=2, stop_early=True,
            record_enabled=True,
        )

        agent.invoke({"prompt": "hi"})

        assert last_record(agent).thoughts == ("padded",)

    def test_the_decision_lives_on_the_task(self) -> None:
        agent = make_agent(
            [step("t1", True), step("t2", False), "R"],
            thinking_rounds=4, stop_early=True, cls=_Spy,
        )
        agent.tasks = []

        agent.invoke({"prompt": "hi"})

        task = agent.tasks[0]
        assert isinstance(task, ScratchpadTask)
        assert task.more_thinking_required is False

    def test_a_second_invoke_does_not_inherit_the_decision(self) -> None:
        agent = make_agent(
            [step("t1", False), "R1", step("t2", False), "R2"],
            thinking_rounds=3, stop_early=True, record_enabled=True,
        )

        agent.invoke({"prompt": "p1"})
        agent.invoke({"prompt": "p2"})

        assert agent._llm_engine.call_count == 4
        assert [r.thoughts for r in agent.get_conversation().records] == [("t1",), ("t2",)]


class TestSystemMessage:
    def test_plain_instructions_without_stop_early(self) -> None:
        agent = make_agent(["t1", "R"])

        agent.invoke({"prompt": "hi"})

        assert calls(agent)[0][0] == {"role": "system", "content": THINK}

    def test_stop_early_frames_the_instructions_and_adds_the_format_section(self) -> None:
        agent = make_agent([step("t1", False), "R"], thinking_rounds=2, stop_early=True)

        agent.invoke({"prompt": "hi"})

        assert calls(agent)[0][0]["content"] == (
            f"{THINKING_INSTRUCTIONS_START}\n{THINK}\n{THINKING_INSTRUCTIONS_END}"
            f"\n\n{STOP_EARLY_PROMPT_SECTION}"
        )

    def test_the_reply_phase_never_gets_the_frame(self) -> None:
        agent = make_agent([step("t1", False), "R"], thinking_rounds=2, stop_early=True)

        agent.invoke({"prompt": "hi"})

        assert calls(agent)[-1][0]["content"] == ROLE

    def test_omitted_slots_use_the_default_constants(self) -> None:
        agent = ScratchpadAgent(
            name="tests", description="d",
            llm_engine=FakeLLMEngine(["t1", "R"]), thinking_rounds=1,
        )

        agent.invoke({"prompt": "hi"})

        assert calls(agent)[0][0]["content"] == DEFAULT_THINKING_INSTRUCTIONS
        assert calls(agent)[1][0]["content"] == DEFAULT_INSTRUCTIONS

    def test_none_slots_raise_type_error_and_blank_slots_value_error(self) -> None:
        with pytest.raises(TypeError, match="thinking_instructions"):
            make_agent(thinking_instructions=None)
        with pytest.raises(TypeError, match="instructions"):
            make_agent(instructions=None)
        with pytest.raises(ValueError):
            make_agent(thinking_instructions="   ")
        with pytest.raises(ValueError):
            make_agent(instructions="")

    def test_to_dict_shows_only_the_authors_instructions(self) -> None:
        agent = make_agent(stop_early=True, thinking_rounds=2)

        assert agent.to_dict()["thinking_instructions"]["template"] == THINK

    def test_thinker_does_not_see_the_main_instructions(self) -> None:
        agent = make_agent(["t1", "R"])

        agent.invoke({"prompt": "hi"})

        assert ROLE not in "".join(m["content"] for m in calls(agent)[0])


class TestPlaceholders:
    def test_thinking_only_placeholder_joins_the_signature_and_renders(self) -> None:
        agent = make_agent(
            ["t1", "R"],
            thinking_instructions=PromptTemplate(
                template="Focus on {angle}.", name="t", description="d"
            ),
        )

        agent.invoke({"prompt": "hi", "angle": "costs"})

        assert [p.name for p in agent.parameters] == ["prompt", "angle"]
        angle = next(p for p in agent.parameters if p.name == "angle")
        assert angle.kind == ParamSpec.POSITIONAL_OR_KEYWORD
        assert calls(agent)[0][0]["content"] == "Focus on costs."
        assert calls(agent)[1][0]["content"] == ROLE

    def test_placeholder_renders_inside_the_stop_early_frame(self) -> None:
        agent = make_agent(
            [step("t1", False), "R"],
            thinking_instructions=PromptTemplate(
                template="Focus on {angle}.", name="t", description="d"
            ),
            thinking_rounds=2, stop_early=True,
        )

        agent.invoke({"prompt": "hi", "angle": "costs"})

        content = calls(agent)[0][0]["content"]
        assert content.startswith(f"{THINKING_INSTRUCTIONS_START}\nFocus on costs.\n")

    def test_a_name_shared_by_both_prompts_is_one_input(self) -> None:
        agent = make_agent(
            ["t1", "R"],
            instructions=PromptTemplate(template="You are {persona}.", name="r", description="d"),
            thinking_instructions=PromptTemplate(
                template="Think as {persona}.", name="t", description="d"
            ),
        )

        agent.invoke({"prompt": "hi", "persona": "a chef"})

        assert [p.name for p in agent.parameters] == ["prompt", "persona"]
        assert calls(agent)[0][0]["content"] == "Think as a chef."
        assert calls(agent)[1][0]["content"] == "You are a chef."

    def test_compatible_overlap_warns_and_the_instructions_slot_wins(self) -> None:
        role = PromptTemplate(
            template="You are {persona}.", name="r", description="d",
            parameters=[ParamSpec(name="persona", index=0, kind=ParamSpec.POSITIONAL_OR_KEYWORD, type="Any", default="a helper")],
        )
        thinking = PromptTemplate(template="Think as {persona}.", name="t", description="d")

        with pytest.warns(UserWarning, match="not identical"):
            agent = make_agent(["t1", "R"], instructions=role, thinking_instructions=thinking)

        persona = next(p for p in agent.parameters if p.name == "persona")
        assert persona.default == "a helper"
        # The instructions slot's default reaches the thinking message too.
        agent.invoke({"prompt": "hi"})
        assert calls(agent)[0][0]["content"] == "Think as a helper."

    def test_the_overlap_warning_is_emitted_once_for_the_two_slots(self) -> None:
        role = PromptTemplate(
            template="You are {persona}.", name="r", description="d",
            parameters=[ParamSpec(name="persona", index=0, kind=ParamSpec.POSITIONAL_OR_KEYWORD, type="Any", default="a helper")],
        )
        thinking = PromptTemplate(template="Think as {persona}.", name="t", description="d")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            make_agent(instructions=role, thinking_instructions=thinking)

        overlap = [w for w in caught if "not identical" in str(w.message)]
        assert len(overlap) == 1
        # An invokable passed in keeps its own name, so the labels are "r" and "t".
        assert "/r/t" in str(overlap[0].message)

    def test_a_true_collision_raises(self) -> None:
        role = PromptTemplate(
            template="You are {persona}.", name="r", description="d",
            parameters=[ParamSpec(name="persona", index=0, kind=ParamSpec.POSITIONAL_OR_KEYWORD, type="str")],
        )
        thinking = PromptTemplate(
            template="Think as {persona}.", name="t", description="d",
            parameters=[ParamSpec(name="persona", index=0, kind=ParamSpec.POSITIONAL_OR_KEYWORD, type="int")],
        )

        with pytest.raises(AgentError, match="no compatible reconciliation"):
            make_agent(instructions=role, thinking_instructions=thinking)

    def test_a_placeholder_equal_to_a_pre_invoke_parameter(self) -> None:
        def about(*, topic: str) -> str:
            return f"About {topic}"

        thinking = PromptTemplate(
            template="Think about {topic}.", name="t", description="d",
            parameters=[ParamSpec(name="topic", index=0, kind=ParamSpec.POSITIONAL_OR_KEYWORD, type="str")],
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            agent = make_agent(["t1", "R"], thinking_instructions=thinking, pre_invoke=about)

        agent.invoke({"topic": "cats"})

        assert [p.name for p in agent.parameters] == ["topic"]
        assert calls(agent)[0][0]["content"] == "Think about cats."
        assert "About cats" in calls(agent)[0][1]["content"]

    def test_described_and_defaulted_placeholder_via_explicit_parameter(self) -> None:
        thinking = PromptTemplate(
            template="Take the {angle} angle.", name="t", description="d",
            parameters=[ParamSpec(name="angle", index=0, kind=ParamSpec.POSITIONAL_OR_KEYWORD, type="Any", default="costs", description="the angle to take")],
        )
        agent = make_agent(["t1", "R"], thinking_instructions=thinking)

        angle = next(p for p in agent.parameters if p.name == "angle")
        assert angle.description == "the angle to take"
        assert angle.default == "costs"
        agent.invoke({"prompt": "hi"})
        assert calls(agent)[0][0]["content"] == "Take the costs angle."


class TestSlotRendering:
    def test_sources_are_instructions_then_thinking_instructions(self) -> None:
        agent = make_agent()

        assert agent._extra_parameter_sources() == [
            agent.instructions, agent.thinking_instructions,
        ]

    @pytest.mark.parametrize("kind", ["callable", "agent"])
    def test_non_template_slots_render_both_phases(self, kind: str) -> None:
        if kind == "callable":
            def thinking(angle: str) -> str:
                return f"Focus on {angle}."

            def main(angle: str) -> str:
                return f"Answer about {angle}."
        else:
            thinking = BasicAgent(
                name="think_slot", description="d", llm_engine=FakeLLMEngine(["Think X."]),
            )
            main = BasicAgent(
                name="main_slot", description="d", llm_engine=FakeLLMEngine(["Main X."]),
            )
        agent = make_agent(["t1", "R"], instructions=main, thinking_instructions=thinking)

        agent.invoke({"prompt": "hi", **({"angle": "costs"} if kind == "callable" else {})})

        expected = (
            ("Focus on costs.", "Answer about costs.") if kind == "callable"
            else ("Think X.", "Main X.")
        )
        assert (calls(agent)[0][0]["content"], calls(agent)[1][0]["content"]) == expected

    def test_slots_run_once_per_invoke_across_all_rounds(self) -> None:
        counts = {"main": 0, "think": 0}

        def main() -> str:
            counts["main"] += 1
            return "main"

        def think() -> str:
            counts["think"] += 1
            return "think"

        agent = make_agent(
            ["t1", "t2", "t3", "R"], thinking_rounds=3,
            instructions=main, thinking_instructions=think,
        )

        agent.invoke({"prompt": "hi"})

        assert len(calls(agent)) == 4
        assert counts == {"main": 1, "think": 1}

    def test_zero_rounds_never_invokes_the_thinking_slot(self) -> None:
        seen: list[str] = []

        def think() -> str:
            seen.append("called")
            return "think"

        agent = make_agent(["R"], thinking_rounds=0, thinking_instructions=think)
        task = agent.create_task(agent.filter_inputs({"prompt": "hi"}))

        agent.pre_process(task)

        assert seen == []
        assert task.thinking_instructions == ""
        assert task.instructions == ROLE

    def test_thinking_slot_text_lands_on_the_task_field(self) -> None:
        agent = make_agent(["t1", "R"])
        task = agent.create_task(agent.filter_inputs({"prompt": "hi"}))

        agent.pre_process(task)

        assert task.thinking_instructions == THINK
        assert task.instructions == ROLE
        assert agent._render_system_message(task) == [{"role": "system", "content": THINK}]
        task.thinking_complete = True
        assert agent._render_system_message(task) == [{"role": "system", "content": ROLE}]

    def test_non_str_thinking_slot_result_raises_naming_the_slot_and_type(self) -> None:
        def think() -> int:
            return 5

        agent = make_agent(["t1", "R"], thinking_instructions=think)

        with pytest.raises(TypeError, match=r"tests_thinking_instructions.*int"):
            agent.invoke({"prompt": "hi"})
        with pytest.raises(TypeError, match=r"tests_thinking_instructions.*int"):
            asyncio.run(agent.async_invoke({"prompt": "hi"}))

    def test_async_run_renders_both_slots(self) -> None:
        agent = make_agent(["t1", "R"], thinking_instructions="Focus on {angle}.")

        asyncio.run(agent.async_invoke({"prompt": "hi", "angle": "costs"}))

        assert calls(agent)[0][0]["content"] == "Focus on costs."
        assert calls(agent)[1][0]["content"] == ROLE

    def test_each_slot_takes_only_its_own_names_and_the_runtime_dict(self) -> None:
        seen: dict[str, Any] = {}

        def main(persona: str, **runtime_context: Any) -> str:
            seen["main"] = (persona, dict(runtime_context))
            return "main"

        def think(angle: str, **runtime_context: Any) -> str:
            seen["think"] = (angle, dict(runtime_context))
            return "think"

        agent = make_agent(
            ["t1", "R"], instructions=main, thinking_instructions=think,
            use_runtime_context=True,
        )

        agent.invoke({
            "prompt": "hi", "persona": "chef", "angle": "costs",
            "runtime_context": {"k": 1},
        })

        assert seen["main"] == ("chef", {"k": 1})
        assert seen["think"] == ("costs", {"k": 1})

    def test_stop_early_frames_the_rendered_thinking_instructions(self) -> None:
        agent = make_agent(
            [step("t1", False), "R"],
            thinking_instructions="Focus on {angle}.",
            thinking_rounds=2, stop_early=True,
        )

        agent.invoke({"prompt": "hi", "angle": "costs"})

        assert calls(agent)[0][0]["content"] == (
            f"{THINKING_INSTRUCTIONS_START}\nFocus on costs.\n{THINKING_INSTRUCTIONS_END}"
            f"\n\n{STOP_EARLY_PROMPT_SECTION}"
        )
        assert calls(agent)[-1][0]["content"] == ROLE


class TestRendering:
    def _task(self, thoughts: list[Any]) -> ScratchpadTask:
        return ScratchpadTask(
            turns=[], inputs={}, user_prompt="",
            thoughts=thoughts,
        )

    def test_thoughts_are_joined_by_rules(self) -> None:
        text = make_agent()._render_thoughts(self._task(["one", "two"]))

        assert text == "one\n\n---\n\ntwo"

    def test_one_thought_has_no_separator(self) -> None:
        assert make_agent()._render_thoughts(self._task(["only"])) == "only"

    def test_no_round_labels_anywhere(self) -> None:
        agent = make_agent(["t1", "t2", "R"], thinking_rounds=2)

        agent.invoke({"prompt": "hi"})

        sent = " ".join(m["content"] for call in calls(agent) for m in call)
        assert "Round" not in sent

    def test_second_round_sees_the_first_thought_in_the_thoughts_frame(self) -> None:
        agent = make_agent(["t1", "t2", "R"], thinking_rounds=2)

        agent.invoke({"prompt": "hi"})

        second = calls(agent)[1]
        assert [m["role"] for m in second] == ["system", "user", "assistant", "user"]
        assert second[2]["content"] == f"{THOUGHTS_START}\nt1\n{THOUGHTS_END}"

    def test_reply_sees_every_thought_separated_by_rules(self) -> None:
        agent = make_agent(["t1", "t2", "R"], thinking_rounds=2)

        agent.invoke({"prompt": "hi"})

        reply = calls(agent)[2]
        assert [m["role"] for m in reply] == ["system", "user", "assistant", "user"]
        assert reply[2]["content"] == f"{THOUGHTS_START}\nt1\n\n---\n\nt2\n{THOUGHTS_END}"

    def test_stop_early_thoughts_render_unwrapped(self) -> None:
        agent = make_agent(
            [step("t1", True), "R"], thinking_rounds=1, stop_early=True,
        )

        agent.invoke({"prompt": "hi"})

        reply_thoughts = calls(agent)[1][2]["content"]
        assert reply_thoughts == f"{THOUGHTS_START}\nt1\n{THOUGHTS_END}"
        assert "more_thinking_required" not in reply_thoughts


class TestResultAndRecord:
    def test_result_is_a_thinking_agent_result(self) -> None:
        agent = make_agent(["t1", "t2", "R"], thinking_rounds=2)

        result = agent.invoke({"prompt": "hi"})

        assert type(result) is ThinkingAgentResult
        assert isinstance(result, AgentResult)
        assert result.thoughts_produced == 2

    def test_record_is_a_thinking_agent_record(self) -> None:
        agent = make_agent(["t1", "R"], record_enabled=True)

        agent.invoke({"prompt": "hi"})

        assert isinstance(last_record(agent), ThinkingAgentRecord)

    def test_thoughts_produced_survives_without_a_record(self) -> None:
        agent = make_agent(["t1", "R"], record_enabled=False)

        result = agent.invoke({"prompt": "hi"})

        assert result.thoughts_produced == 1
        assert agent.get_conversation().records == []

    def test_post_invoke_still_shapes_the_result_value(self) -> None:
        def shout(response: str) -> str:
            return response.upper()

        agent = make_agent(["t1", "reply"], post_invoke=shout, response_to_post="response")

        result = agent.invoke({"prompt": "hi"})

        assert result.result == "REPLY"
        assert result.thoughts_produced == 1


class TestAsyncParity:
    def test_fixed_rounds_match_the_sync_flow(self) -> None:
        sync_agent = make_agent(["t1", "t2", "R"], thinking_rounds=2, record_enabled=True)
        async_agent = make_agent(["t1", "t2", "R"], thinking_rounds=2, record_enabled=True)

        sync_result = sync_agent.invoke({"prompt": "hi"})
        async_result = asyncio.run(async_agent.async_invoke({"prompt": "hi"}))

        assert async_result.result == sync_result.result
        assert async_result.thoughts_produced == sync_result.thoughts_produced == 2
        assert calls(async_agent) == calls(sync_agent)
        assert last_record(async_agent).thoughts == last_record(sync_agent).thoughts

    def test_zero_rounds_match_the_sync_flow(self) -> None:
        sync_agent = make_agent(["R"], thinking_rounds=0)
        async_agent = make_agent(["R"], thinking_rounds=0)

        sync_agent.invoke({"prompt": "hi"})
        async_result = asyncio.run(async_agent.async_invoke({"prompt": "hi"}))

        assert calls(async_agent) == calls(sync_agent)
        assert async_result.thoughts_produced == 0

    def test_stop_early_matches_the_sync_flow(self) -> None:
        script = [step("t1", True), step("t2", False), "R"]
        sync_agent = make_agent(list(script), thinking_rounds=5, stop_early=True, record_enabled=True)
        async_agent = make_agent(list(script), thinking_rounds=5, stop_early=True, record_enabled=True)

        sync_result = sync_agent.invoke({"prompt": "hi"})
        async_result = asyncio.run(async_agent.async_invoke({"prompt": "hi"}))

        assert async_result.thoughts_produced == sync_result.thoughts_produced == 2
        assert calls(async_agent) == calls(sync_agent)
        assert structures(async_agent) == structures(sync_agent)
        assert last_record(async_agent).thoughts == ("t1", "t2")


class TestResultHook:
    def test_result_cls_and_no_per_family_builder(self) -> None:
        assert ScratchpadAgent.result_cls is ThinkingAgentResult
        assert not hasattr(ScratchpadAgent, "build_result_from_task")

    def test_run_id_matches_the_stored_record(self) -> None:
        agent = make_agent(["t1", "R"], record_enabled=True)

        result = agent.invoke({"prompt": "hi"})

        assert type(result) is ThinkingAgentResult
        assert result.run_id == last_record(agent).run_id
