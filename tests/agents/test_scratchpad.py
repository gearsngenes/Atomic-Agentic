from __future__ import annotations

import asyncio
import warnings
from typing import Any

import pytest

from .conftest import FakeLLMEngine

from atomic_agentic.agents.basic import BasicAgent
from atomic_agentic.agents.scratchpad import ScratchpadAgent
from atomic_agentic.constants.agents import (
    STOP_EARLY_PROMPT_SECTION,
    THINKING_INSTRUCTIONS_END,
    THINKING_INSTRUCTIONS_START,
    THOUGHTS_END,
    THOUGHTS_START,
)
from atomic_agentic.exceptions import AgentError
from atomic_agentic.models.agents.prompts import PromptConfig
from atomic_agentic.models.agents.records import ThinkingAgentRecord
from atomic_agentic.models.agents.tasks import ScratchpadTask
from atomic_agentic.models.results.agents import AgentResult, ThinkingAgentResult
from atomic_agentic.utils.agents import build_thinking_step_schema
from atomic_agentic.utils.rendering import stringify_value


_LEAD_SCHEMA = {
    "type": "object",
    "properties": {
        "focus": {"type": "string"},
        "leads": {"type": "array", "items": {"type": "object"}},
    },
}

ROLE = "You are a careful assistant."
THINK = "Think hard."


def make_agent(
    responses: list[Any] | None = None,
    *,
    thinking_rounds: int = 1,
    role_prompt: str | PromptConfig | None = ROLE,
    thinking_instructions: str | PromptConfig | None = THINK,
    record_enabled: bool = False,
    cls: type[ScratchpadAgent] = ScratchpadAgent,
    **kwargs: Any,
) -> ScratchpadAgent:
    return cls(
        name="tests",
        namespace="tests",
        description="ScratchpadAgent under test.",
        llm_engine=FakeLLMEngine(responses if responses is not None else []),
        role_prompt=role_prompt,
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
    """Keeps every task ``_thinking_done`` sees (it is evaluated at seed)."""

    def _thinking_done(self, task: ScratchpadTask) -> bool:
        self.tasks.append(task)
        return super()._thinking_done(task)


class TestConstruction:
    def test_is_instantiable_with_defaults(self) -> None:
        agent = make_agent()

        assert agent.thinking_rounds == 1
        assert agent.stop_early is False
        assert agent.thinking_schema is None
        assert agent.response_schema is None

    def test_thinking_rounds_is_required(self) -> None:
        with pytest.raises(TypeError, match="thinking_rounds"):
            ScratchpadAgent(
                name="tests", namespace="tests", description="d",
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

    def test_thinking_schema_stored_unwrapped(self) -> None:
        agent = make_agent(thinking_schema=_LEAD_SCHEMA, stop_early=True)

        assert agent.thinking_schema == _LEAD_SCHEMA

    def test_thinking_schema_rejects_non_mapping(self) -> None:
        with pytest.raises(AgentError, match="thinking_schema"):
            make_agent(thinking_schema="not a dict")  # type: ignore[arg-type]

    def test_response_schema_rejects_non_mapping(self) -> None:
        with pytest.raises(AgentError, match="response_schema"):
            make_agent(response_schema="not a dict")  # type: ignore[arg-type]

    def test_thinking_instructions_rejects_bad_type(self) -> None:
        with pytest.raises(TypeError, match="thinking_instructions"):
            make_agent(thinking_instructions=123)  # type: ignore[arg-type]

    @pytest.mark.parametrize("key", ["$defs", "definitions", "$ref"])
    def test_stop_early_rejects_root_level_reference_keys(self, key: str) -> None:
        schema = {"type": "object", key: {} if key != "$ref" else "#/$defs/x"}

        with pytest.raises(AgentError, match="stop_early"):
            make_agent(thinking_schema=schema, stop_early=True)

    @pytest.mark.parametrize("key", ["$defs", "definitions", "$ref"])
    def test_reference_keys_are_fine_without_stop_early(self, key: str) -> None:
        schema = {"type": "object", key: {} if key != "$ref" else "#/$defs/x"}

        assert make_agent(thinking_schema=schema).thinking_schema == schema

    def test_property_named_definitions_is_not_rejected(self) -> None:
        schema = {"type": "object", "properties": {"definitions": {"type": "string"}}}

        assert make_agent(thinking_schema=schema, stop_early=True).stop_early is True

    @pytest.mark.parametrize("name", ["thinking_schema", "thinking_rounds", "stop_early"])
    def test_properties_are_read_only(self, name: str) -> None:
        agent = make_agent()

        with pytest.raises(AttributeError):
            setattr(agent, name, 3)

    def test_removed_surfaces_are_gone(self) -> None:
        agent = make_agent()

        assert not hasattr(agent, "thinking_llm_engine")
        assert not hasattr(agent, "thinking_rounds_limit")
        assert ScratchpadAgent.get_reserved_parameters() == []
        with pytest.raises(TypeError):
            make_agent(thinking_llm_engine=FakeLLMEngine([]))
        with pytest.raises(TypeError):
            make_agent(thinking_rounds_limit=3)

    def test_thinking_rounds_is_not_an_invoke_input(self) -> None:
        agent = make_agent()

        assert [p.name for p in agent.parameters] == ["prompt"]

    def test_thinking_prompt_is_registered_under_its_name(self) -> None:
        agent = make_agent()

        assert agent._system_prompts["thinking"].template == THINK
        assert agent._system_prompts["role"].template == ROLE

    def test_response_schema_gives_the_default_conversation_json_mode(self) -> None:
        schema = {"type": "object", "properties": {"answer": {"type": "string"}}}

        assert make_agent(response_schema=schema).get_conversation().render_mode == "j"
        assert make_agent().get_conversation().render_mode == "s"

    def test_to_dict_includes_the_thinking_knobs(self) -> None:
        d = make_agent(
            thinking_schema=_LEAD_SCHEMA, thinking_rounds=3, stop_early=True
        ).to_dict()

        assert d["thinking_instructions"] == THINK
        assert d["thinking_schema"] == _LEAD_SCHEMA
        assert d["thinking_rounds"] == 3
        assert d["stop_early"] is True
        assert d["role_prompt"] == ROLE
        assert "thinking_llm" not in d
        assert "thinking_rounds_limit" not in d


class TestFixedRounds:
    def test_zero_rounds_makes_one_basic_agent_call(self) -> None:
        basic = BasicAgent(
            name="tests", namespace="tests", description="basic",
            llm_engine=FakeLLMEngine(["R"]), role_prompt=ROLE,
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

    def test_llm_records_are_tagged_with_the_phase_prompt_names(self) -> None:
        agent = make_agent(["t1", "t2", "R"], thinking_rounds=2, record_enabled=True)

        agent.invoke({"prompt": "hi"})

        names = [r.system_prompt_name for r in last_record(agent).llm_records]
        assert names == ["thinking", "thinking", "role"]

    def test_string_thoughts_are_stripped(self) -> None:
        agent = make_agent(["  a thought \n", "R"], record_enabled=True)

        agent.invoke({"prompt": "hi"})

        assert last_record(agent).thoughts == ("a thought",)

    def test_thinking_calls_are_free_text_without_a_schema(self) -> None:
        agent = make_agent(["t1", "R"])

        agent.invoke({"prompt": "hi"})

        assert structures(agent) == [None, None]

    def test_thinking_and_reply_schemas_are_independent(self) -> None:
        reply_schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
        agent = make_agent(
            [{"focus": "x"}, {"answer": "y"}],
            thinking_schema=_LEAD_SCHEMA,
            response_schema=reply_schema,
            record_enabled=True,
        )

        result = agent.invoke({"prompt": "hi"})

        assert structures(agent) == [_LEAD_SCHEMA, reply_schema]
        assert last_record(agent).thoughts == ({"focus": "x"},)
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

        wrapper = build_thinking_step_schema(None)
        assert structures(agent) == [wrapper, wrapper, None]
        assert wrapper["properties"]["thought"]["type"] == "string"

    def test_thinking_schema_is_nested_under_thought(self) -> None:
        agent = make_agent(
            [step({"focus": "f"}, False), "R"],
            thinking_schema=_LEAD_SCHEMA, thinking_rounds=3, stop_early=True,
        )

        agent.invoke({"prompt": "hi"})

        sent = structures(agent)[0]
        assert sent == build_thinking_step_schema(_LEAD_SCHEMA)
        assert sent["properties"]["thought"]["properties"] == _LEAD_SCHEMA["properties"]

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
        wrapped = step({"focus": "f"}, False)
        agent = make_agent(
            [wrapped, "R"], thinking_schema=_LEAD_SCHEMA, thinking_rounds=2,
            stop_early=True, record_enabled=True,
        )

        agent.invoke({"prompt": "hi"})

        record = last_record(agent)
        assert record.thoughts == ({"focus": "f"},)
        assert record.llm_records[0].llm_result.result == wrapped

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

    def test_default_instructions_apply_to_none_and_blank(self) -> None:
        for blank in (None, "   "):
            agent = make_agent(["t1", "R"], thinking_instructions=blank)

            agent.invoke({"prompt": "hi"})

            assert calls(agent)[0][0]["content"] == ScratchpadAgent.DEFAULT_THINKING_PROMPT

    def test_to_dict_shows_only_the_authors_instructions(self) -> None:
        agent = make_agent(stop_early=True, thinking_rounds=2)

        assert agent.to_dict()["thinking_instructions"] == THINK

    def test_thinker_does_not_see_the_role_prompt(self) -> None:
        agent = make_agent(["t1", "R"])

        agent.invoke({"prompt": "hi"})

        assert ROLE not in "".join(m["content"] for m in calls(agent)[0])


class TestPlaceholders:
    def test_thinking_only_placeholder_joins_the_signature_and_renders(self) -> None:
        agent = make_agent(
            ["t1", "R"],
            thinking_instructions=PromptConfig(template="Focus on {angle}.", description="d"),
        )

        agent.invoke({"prompt": "hi", "angle": "costs"})

        assert [p.name for p in agent.parameters] == ["prompt", "angle"]
        assert calls(agent)[0][0]["content"] == "Focus on costs."
        assert calls(agent)[1][0]["content"] == ROLE

    def test_placeholder_renders_inside_the_stop_early_frame(self) -> None:
        agent = make_agent(
            [step("t1", False), "R"],
            thinking_instructions=PromptConfig(template="Focus on {angle}.", description="d"),
            thinking_rounds=2, stop_early=True,
        )

        agent.invoke({"prompt": "hi", "angle": "costs"})

        content = calls(agent)[0][0]["content"]
        assert content.startswith(f"{THINKING_INSTRUCTIONS_START}\nFocus on costs.\n")

    def test_a_name_shared_by_both_prompts_is_one_input(self) -> None:
        agent = make_agent(
            ["t1", "R"],
            role_prompt=PromptConfig(template="You are {persona}.", description="d"),
            thinking_instructions=PromptConfig(template="Think as {persona}.", description="d"),
        )

        agent.invoke({"prompt": "hi", "persona": "a chef"})

        assert [p.name for p in agent.parameters] == ["prompt", "persona"]
        assert calls(agent)[0][0]["content"] == "Think as a chef."
        assert calls(agent)[1][0]["content"] == "You are a chef."

    def test_compatible_overlap_warns_and_the_role_prompt_wins(self) -> None:
        role = PromptConfig(
            template="You are {persona}.", description="d",
            field_specs={"persona": {"default": "a helper"}},
        )
        thinking = PromptConfig(template="Think as {persona}.", description="d")

        with pytest.warns(UserWarning, match="not identical"):
            agent = make_agent(["t1", "R"], role_prompt=role, thinking_instructions=thinking)

        persona = next(p for p in agent.parameters if p.name == "persona")
        assert persona.default == "a helper"
        # The role prompt's default reaches the thinking message too.
        agent.invoke({"prompt": "hi"})
        assert calls(agent)[0][0]["content"] == "Think as a helper."

    def test_the_overlap_warning_points_at_the_constructing_code(self) -> None:
        role = PromptConfig(
            template="You are {persona}.", description="d",
            field_specs={"persona": {"default": "a helper"}},
        )
        thinking = PromptConfig(template="Think as {persona}.", description="d")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            make_agent(role_prompt=role, thinking_instructions=thinking)

        overlap = [w for w in caught if "not identical" in str(w.message)]
        assert len(overlap) == 1
        assert overlap[0].filename == __file__

    def test_a_true_collision_raises(self) -> None:
        role = PromptConfig(
            template="You are {persona}.", description="d",
            field_specs={"persona": {"type": "str"}},
        )
        thinking = PromptConfig(
            template="Think as {persona}.", description="d",
            field_specs={"persona": {"type": "int"}},
        )

        with pytest.raises(AgentError, match="no compatible reconciliation"):
            make_agent(role_prompt=role, thinking_instructions=thinking)

    def test_a_placeholder_equal_to_a_pre_invoke_parameter(self) -> None:
        def about(*, topic: str) -> str:
            return f"About {topic}"

        thinking = PromptConfig(
            template="Think about {topic}.", description="d",
            field_specs={"topic": {"type": "str"}},
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            agent = make_agent(["t1", "R"], thinking_instructions=thinking, pre_invoke=about)

        agent.invoke({"topic": "cats"})

        assert [p.name for p in agent.parameters] == ["topic"]
        assert calls(agent)[0][0]["content"] == "Think about cats."
        assert "About cats" in calls(agent)[0][1]["content"]

    def test_described_and_defaulted_placeholder_via_field_specs(self) -> None:
        thinking = PromptConfig(
            template="Take the {angle} angle.", description="d",
            field_specs={"angle": {"description": "the angle to take", "default": "costs"}},
        )
        agent = make_agent(["t1", "R"], thinking_instructions=thinking)

        angle = next(p for p in agent.parameters if p.name == "angle")
        assert angle.description == "the angle to take"
        assert angle.default == "costs"
        agent.invoke({"prompt": "hi"})
        assert calls(agent)[0][0]["content"] == "Take the costs angle."


class TestRendering:
    def _task(self, thoughts: list[Any]) -> ScratchpadTask:
        return ScratchpadTask(
            turns=[], inputs={}, user_prompt="", system_prompt_name="thinking",
            thoughts=thoughts,
        )

    def test_strings_are_shown_as_is_and_values_as_json(self) -> None:
        agent = make_agent()
        value = {"focus": "ünï", "leads": [1, 2]}

        text = agent._render_thoughts(self._task(["plain", value]))

        assert text == f"plain\n\n---\n\n{stringify_value(value, mode='j')}"
        assert "ünï" in text

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

    def test_structured_thoughts_reach_the_model_as_json(self) -> None:
        agent = make_agent(
            [{"focus": "f"}, "R"], thinking_schema=_LEAD_SCHEMA, record_enabled=False,
        )

        agent.invoke({"prompt": "hi"})

        assert stringify_value({"focus": "f"}, mode="j") in calls(agent)[1][2]["content"]

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

        agent = make_agent(["t1", "reply"], post_invoke=shout, bind_response_to="response")

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

    def test_structured_thoughts_match_the_sync_flow(self) -> None:
        async_agent = make_agent(
            [step({"focus": "f"}, False), "R"], thinking_schema=_LEAD_SCHEMA,
            thinking_rounds=2, stop_early=True, record_enabled=True,
        )

        asyncio.run(async_agent.async_invoke({"prompt": "hi"}))

        assert last_record(async_agent).thoughts == ({"focus": "f"},)
