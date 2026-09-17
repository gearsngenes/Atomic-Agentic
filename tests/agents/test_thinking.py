from __future__ import annotations

from typing import Any
import asyncio
import json

import pytest

from .conftest import FakeLLMEngine

from atomic_agentic.agents.thinking import ThinkingAgent, _THINKING_FRAMING, _THINKING_CONTINUATION_NUDGE
from atomic_agentic.constants.agents import RUN_ID_PARAM, THINKING_ROUNDS_PARAM
from atomic_agentic.exceptions import AgentError, AgentInvocationError
from atomic_agentic.models.agents.prompts import PromptConfig
from atomic_agentic.models.results.agents import ThinkingAgentResult


_LEAD_SCHEMA = {
    "type": "object",
    "properties": {
        "focus": {"type": "string"},
        "leads": {"type": "array", "items": {"type": "object"}},
        "working_hypothesis": {"type": "string"},
        "ruled_out": {"type": "array", "items": {"type": "string"}},
    },
}


def make_agent(
    *,
    engine: FakeLLMEngine | None = None,
    thinking_llm_engine: FakeLLMEngine | None = None,
    role_prompt: str | PromptConfig | None = "You are a careful assistant.",
    thinking_instructions: str | PromptConfig | None = None,
    context_enabled: bool = True,
    response_schema: dict[str, Any] | None = None,
    thinking_schema: dict[str, Any] | None = None,
) -> ThinkingAgent:
    return ThinkingAgent(
        name="tests",
        namespace="tests",
        description="ThinkingAgent under test.",
        llm_engine=engine or FakeLLMEngine([]),
        role_prompt=role_prompt,
        thinking_instructions=thinking_instructions,
        context_enabled=context_enabled,
        thinking_llm_engine=thinking_llm_engine,
        response_schema=response_schema,
        thinking_schema=thinking_schema,
    )


def last_thoughts(agent: ThinkingAgent) -> tuple:
    """Convenience: thoughts of the active conversation's most recent
    record -- what every example now does after a single invoke()."""
    return agent.get_conversation(turns=1)[0].thoughts


class TestConstruction:
    def test_response_schema_defaults_to_none(self) -> None:
        assert make_agent().response_schema is None

    def test_response_schema_stored_as_given(self) -> None:
        schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
        assert make_agent(response_schema=schema).response_schema == schema

    def test_response_schema_rejects_non_mapping_non_none(self) -> None:
        with pytest.raises(AgentError, match="response_schema"):
            make_agent(response_schema="not a dict")  # type: ignore[arg-type]

    def test_thinking_schema_defaults_to_none(self) -> None:
        assert make_agent().thinking_schema is None

    def test_thinking_schema_stored_as_given(self) -> None:
        assert make_agent(thinking_schema=_LEAD_SCHEMA).thinking_schema == _LEAD_SCHEMA

    def test_thinking_schema_rejects_non_mapping_non_none(self) -> None:
        with pytest.raises(AgentError, match="thinking_schema"):
            make_agent(thinking_schema="not a dict")  # type: ignore[arg-type]

    def test_thinking_llm_engine_defaults_to_none(self) -> None:
        assert make_agent().thinking_llm_engine is None

    def test_thinking_llm_engine_rejects_non_engine_non_none_at_construction(self) -> None:
        with pytest.raises(AgentError, match="thinking_llm_engine"):
            make_agent(thinking_llm_engine="not an engine")  # type: ignore[arg-type]

    def test_thinking_llm_engine_setter_accepts_engine_or_none(self) -> None:
        agent = make_agent()
        other = FakeLLMEngine([])

        agent.thinking_llm_engine = other
        assert agent.thinking_llm_engine is other

        agent.thinking_llm_engine = None
        assert agent.thinking_llm_engine is None

    def test_thinking_llm_engine_setter_rejects_invalid_type(self) -> None:
        agent = make_agent()
        with pytest.raises(TypeError):
            agent.thinking_llm_engine = "bad"  # type: ignore[assignment]

    def test_role_prompt_and_thinking_instructions_incompatible_collision_raises(self) -> None:
        role = PromptConfig(
            template="Persona for {topic}.",
            description="d",
            field_specs={"topic": {"type": "int"}},
        )
        thinking = PromptConfig(
            template="Think about {topic}.",
            description="d",
            field_specs={"topic": {"type": "str"}},
        )
        with pytest.raises(AgentError, match="no compatible reconciliation"):
            make_agent(role_prompt=role, thinking_instructions=thinking)

    def test_role_prompt_and_thinking_instructions_compatible_overlap_warns_role_wins(self) -> None:
        role = PromptConfig(
            template="Persona for {topic}.",
            description="d",
            field_specs={"topic": {"type": "str", "default": "role-default"}},
        )
        thinking = PromptConfig(
            template="Think about {topic}.",
            description="d",
            field_specs={"topic": {"type": "str", "default": "thinking-default"}},
        )
        with pytest.warns(UserWarning, match="not identical"):
            agent = make_agent(role_prompt=role, thinking_instructions=thinking)
        topic_param = next(p for p in agent.parameters if p.name == "topic")
        assert topic_param.default == "role-default"

    def test_role_prompt_and_thinking_instructions_overlap_widens_type(self) -> None:
        role = PromptConfig(
            template="Persona for {count}.",
            description="d",
            field_specs={"count": {"type": "int"}},
        )
        thinking = PromptConfig(
            template="Think about {count}.",
            description="d",
            field_specs={"count": {"type": "Any"}},
        )
        agent = make_agent(role_prompt=role, thinking_instructions=thinking)
        count_param = next(p for p in agent.parameters if p.name == "count")
        assert count_param.type == ("int",)

    def test_role_prompt_and_thinking_instructions_own_params_both_land_in_schema(self) -> None:
        role = PromptConfig(
            template="Persona for {audience}.",
            description="d",
            field_specs={"audience": {"type": "str"}},
        )
        thinking = PromptConfig(
            template="Focus on {angle}.",
            description="d",
            field_specs={"angle": {"type": "str"}},
        )
        agent = make_agent(role_prompt=role, thinking_instructions=thinking)
        names = [p.name for p in agent.parameters]
        assert "audience" in names
        assert "angle" in names

    def test_get_reserved_parameters_orders_thinking_rounds_before_run_id(self) -> None:
        assert ThinkingAgent.get_reserved_parameters() == [THINKING_ROUNDS_PARAM, RUN_ID_PARAM]

        agent = make_agent()
        names = [p.name for p in agent.parameters]
        assert names.index("thinking_rounds") < names.index("run_id")


class TestThinkingRounds:
    def test_omitted_from_invoke_defaults_to_one(self) -> None:
        engine = FakeLLMEngine(["t1", "reply"])
        agent = make_agent(engine=engine)

        agent.invoke({"prompt": "hello"})

        assert len(engine.calls) == 2
        assert last_thoughts(agent) == ("t1",)

    def test_zero_skips_thinking_without_engine_call(self) -> None:
        engine = FakeLLMEngine(["final reply"])
        agent = make_agent(engine=engine)

        result = agent.invoke({"prompt": "hello", "thinking_rounds": 0})

        assert result.result == "final reply"
        assert len(engine.calls) == 1
        assert last_thoughts(agent) == ()

    def test_non_int_raises_agent_invocation_error(self) -> None:
        agent = make_agent(engine=FakeLLMEngine([]))

        with pytest.raises(AgentInvocationError, match="thinking_rounds"):
            agent.invoke({"prompt": "hello", "thinking_rounds": 1.5})

    def test_bool_raises_agent_invocation_error(self) -> None:
        agent = make_agent(engine=FakeLLMEngine([]))

        with pytest.raises(AgentInvocationError, match="thinking_rounds"):
            agent.invoke({"prompt": "hello", "thinking_rounds": True})

    def test_negative_raises_agent_invocation_error(self) -> None:
        agent = make_agent(engine=FakeLLMEngine([]))

        with pytest.raises(AgentInvocationError, match="thinking_rounds"):
            agent.invoke({"prompt": "hello", "thinking_rounds": -1})

    def test_exact_n_rounds_always_run_no_early_exit(self) -> None:
        engine = FakeLLMEngine(["t1", "t2", "t3", "reply"])
        agent = make_agent(engine=engine)

        agent.invoke({"prompt": "hello", "thinking_rounds": 3})

        assert len(engine.calls) == 4
        assert len(last_thoughts(agent)) == 3

    def test_async_exact_n_rounds_mirrors_sync(self) -> None:
        engine = FakeLLMEngine(["t1", "t2", "t3", "reply"])
        agent = make_agent(engine=engine)

        asyncio.run(agent.async_invoke({"prompt": "hello", "thinking_rounds": 3}))

        assert len(engine.calls) == 4
        assert len(last_thoughts(agent)) == 3


class TestThinkPhase:
    def test_appends_stripped_string_thought(self) -> None:
        engine = FakeLLMEngine([" padded thought ", "reply"])
        agent = make_agent(engine=engine)

        agent.invoke({"prompt": "hello", "thinking_rounds": 1})

        assert last_thoughts(agent) == ("padded thought",)

    def test_appends_falsy_string_unconditionally(self) -> None:
        engine = FakeLLMEngine(["", "reply"])
        agent = make_agent(engine=engine)

        agent.invoke({"prompt": "hello", "thinking_rounds": 1})

        assert last_thoughts(agent) == ("",)

    def test_stores_non_str_value_verbatim_when_schema_set(self) -> None:
        thought_value = {"focus": "x", "leads": [], "working_hypothesis": "h", "ruled_out": []}
        engine = FakeLLMEngine([thought_value, "reply"])
        agent = make_agent(engine=engine, thinking_schema={"type": "object"})

        agent.invoke({"prompt": "hello", "thinking_rounds": 1})

        assert last_thoughts(agent) == (thought_value,)

    def test_multiple_rounds_accumulate_in_order(self) -> None:
        engine = FakeLLMEngine(["first", "second", "third", "reply"])
        agent = make_agent(engine=engine)

        agent.invoke({"prompt": "hello", "thinking_rounds": 3})

        assert last_thoughts(agent) == ("first", "second", "third")

    def test_noops_once_in_role_phase(self) -> None:
        engine = FakeLLMEngine([])
        agent = make_agent(engine=engine)
        task = agent._initialize_task(turns=[], prompt="hello", inputs={"thinking_rounds": 3})
        task.system_prompt_name = "role"

        updated = agent.think(task)

        assert updated is task
        assert engine.calls == []

    def test_async_think_mirrors_sync_behavior(self) -> None:
        engine = FakeLLMEngine(["only", "reply"])
        agent = make_agent(engine=engine)

        asyncio.run(agent.async_invoke({"prompt": "hello", "thinking_rounds": 1}))

        assert last_thoughts(agent) == ("only",)


class TestActPhase:
    def test_act_no_ops_while_still_thinking(self) -> None:
        agent = make_agent(engine=FakeLLMEngine([]))
        task = agent._initialize_task(turns=[], prompt="hello", inputs={"thinking_rounds": 3})
        assert task.system_prompt_name != "role"

        updated = agent.act(task)

        assert updated is task
        assert updated.complete is False

    def test_async_act_no_ops_while_still_thinking(self) -> None:
        agent = make_agent(engine=FakeLLMEngine([]))
        task = agent._initialize_task(turns=[], prompt="hello", inputs={"thinking_rounds": 3})

        updated = asyncio.run(agent.async_act(task))

        assert updated is task
        assert updated.complete is False

    def test_act_delegates_to_basic_agent_body_once_in_reply_phase(self) -> None:
        engine = FakeLLMEngine(["reply text"])
        agent = make_agent(engine=engine)
        task = agent._initialize_task(turns=[], prompt="hello", inputs={"thinking_rounds": 0})
        task = agent.think(task)  # thinking_rounds=0 -> switches straight to "role"
        assert task.system_prompt_name == "role"

        updated = agent.act(task)

        assert updated.complete is True
        assert updated.generated_response == "reply text"


class TestSecondaryThinkingEngine:
    def test_defaults_to_main_engine_when_unset(self) -> None:
        engine = FakeLLMEngine(["t1", "t2", "reply"])
        agent = make_agent(engine=engine)

        agent.invoke({"prompt": "hello", "thinking_rounds": 2})

        assert len(engine.calls) == 3

    def test_used_for_thinking_calls_only_when_set(self) -> None:
        main_engine = FakeLLMEngine(["reply"])
        thinking_engine = FakeLLMEngine(["t1", "t2"])
        agent = make_agent(engine=main_engine, thinking_llm_engine=thinking_engine)

        agent.invoke({"prompt": "hello", "thinking_rounds": 2})

        assert len(thinking_engine.calls) == 2
        assert len(main_engine.calls) == 1

    def test_setter_swaps_engine_used_on_next_invocation(self) -> None:
        main_engine = FakeLLMEngine(["t1", "reply1", "reply2"])
        agent = make_agent(engine=main_engine)

        agent.invoke({"prompt": "first", "thinking_rounds": 1})
        assert len(main_engine.calls) == 2  # 1 thinking + 1 reply, both on main so far

        new_engine = FakeLLMEngine(["t2"])
        agent.thinking_llm_engine = new_engine

        agent.invoke({"prompt": "second", "thinking_rounds": 1})

        assert len(new_engine.calls) == 1
        assert len(main_engine.calls) == 3  # +1 reply call only; thinking moved to new_engine

    def test_property_getter_returns_raw_unresolved_value(self) -> None:
        assert make_agent().thinking_llm_engine is None


class TestThinkingSchema:
    def test_threaded_into_thinking_calls_only(self) -> None:
        engine = FakeLLMEngine([
            {"focus": "a", "leads": [], "working_hypothesis": "h1", "ruled_out": []},
            {"focus": "b", "leads": [], "working_hypothesis": "h2", "ruled_out": []},
            "final reply",
        ])
        agent = make_agent(engine=engine, thinking_schema=_LEAD_SCHEMA)

        agent.invoke({"prompt": "hello", "thinking_rounds": 2})

        assert engine.payloads[0]["output_structure"] == _LEAD_SCHEMA
        assert engine.payloads[1]["output_structure"] == _LEAD_SCHEMA
        assert engine.payloads[-1]["output_structure"] is None

    def test_independent_of_response_schema_both_set_simultaneously(self) -> None:
        response_schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
        engine = FakeLLMEngine([
            {"focus": "a", "leads": [], "working_hypothesis": "h1", "ruled_out": []},
            {"answer": "final"},
        ])
        agent = make_agent(engine=engine, thinking_schema=_LEAD_SCHEMA, response_schema=response_schema)

        agent.invoke({"prompt": "hello", "thinking_rounds": 1})

        assert engine.payloads[0]["output_structure"] == _LEAD_SCHEMA
        assert engine.payloads[-1]["output_structure"] == response_schema


class TestResponseSchema:
    def test_defaults_to_none(self) -> None:
        assert make_agent().response_schema is None

    def test_stored_as_given(self) -> None:
        schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
        assert make_agent(response_schema=schema).response_schema == schema

    def test_rejects_non_mapping_non_none(self) -> None:
        with pytest.raises(AgentError, match="response_schema"):
            make_agent(response_schema="not a dict")  # type: ignore[arg-type]

    def test_reply_phase_forwards_schema_thinking_phase_does_not(self) -> None:
        engine = FakeLLMEngine(["final reply"])
        agent = make_agent(engine=engine, response_schema={"type": "object"})

        agent.invoke({"prompt": "hello", "thinking_rounds": 0})

        assert engine.payloads[-1]["output_structure"] == {"type": "object"}


class TestRenderPipeline:
    def test_thinking_system_message_renders_default_prompt_when_no_instructions_given(self) -> None:
        agent = make_agent(engine=FakeLLMEngine([]), thinking_instructions=None)
        task = agent._initialize_task(turns=[], prompt="hello", inputs={"thinking_rounds": 3})

        rendered = agent._render_system_message(task)

        assert rendered[0]["content"] == ThinkingAgent.DEFAULT_THINKING_PROMPT

    def test_thinking_system_message_renders_given_instructions_verbatim(self) -> None:
        agent = make_agent(engine=FakeLLMEngine([]), thinking_instructions="Focus on edge cases.")
        task = agent._initialize_task(turns=[], prompt="hello", inputs={"thinking_rounds": 3})

        rendered = agent._render_system_message(task)

        assert rendered[0]["content"] == "Focus on edge cases."

    def test_role_phase_system_message_delegates_to_role_prompt(self) -> None:
        agent = make_agent(engine=FakeLLMEngine([]), role_prompt="You are a distinctly-worded persona.")
        task = agent._initialize_task(turns=[], prompt="hello", inputs={"thinking_rounds": 0})
        task = agent.think(task)  # switches to "role"

        rendered = agent._render_system_message(task)

        assert rendered[0]["content"] == "You are a distinctly-worded persona."

    def test_thinking_task_messages_banner_only_on_first_round(self) -> None:
        agent = make_agent(engine=FakeLLMEngine([]))
        task = agent._initialize_task(turns=[], prompt="hello", inputs={"thinking_rounds": 3})

        rendered = agent._render_task_messages(task)

        assert len(rendered) == 1
        assert "hello" in rendered[0]["content"]
        assert _THINKING_FRAMING in rendered[0]["content"]

    def test_thinking_task_messages_one_message_pair_per_completed_thought(self) -> None:
        agent = make_agent(engine=FakeLLMEngine([]))
        task = agent._initialize_task(turns=[], prompt="hello", inputs={"thinking_rounds": 3})
        task.thoughts.append("first thought")
        task.thoughts.append({"a": 1})

        rendered = agent._render_task_messages(task)

        assert rendered[0]["role"] == "user"
        assert "hello" in rendered[0]["content"]
        assert _THINKING_FRAMING in rendered[0]["content"]
        assert rendered[1] == {"role": "assistant", "content": "first thought"}
        assert rendered[2] == {"role": "user", "content": _THINKING_CONTINUATION_NUDGE}
        assert rendered[3] == {"role": "assistant", "content": json.dumps({"a": 1})}
        assert rendered[4] == {"role": "user", "content": _THINKING_CONTINUATION_NUDGE}
        assert len(rendered) == 5

    def test_thinking_phase_messages_never_contain_round_headers(self) -> None:
        agent = make_agent(engine=FakeLLMEngine([]))
        task = agent._initialize_task(turns=[], prompt="hello", inputs={"thinking_rounds": 3})
        task.thoughts.append("first thought")
        task.thoughts.append({"a": 1})

        rendered = agent._render_task_messages(task)

        assert all("## Round" not in m["content"] for m in rendered)

    def test_role_phase_task_messages_include_thoughts_snapshot_and_respond_instruction(self) -> None:
        agent = make_agent(engine=FakeLLMEngine([]))
        task = agent._initialize_task(turns=[], prompt="hello", inputs={"thinking_rounds": 3})
        task.thoughts.append("reasoned thing")
        task.system_prompt_name = "role"

        rendered = agent._render_task_messages(task)

        assert len(rendered) == 3
        assert "## Round 0" in rendered[1]["content"]
        assert "reasoned thing" in rendered[1]["content"]
        assert "respond to the current task" in rendered[2]["content"]

    def test_role_phase_task_messages_banner_only_when_no_thoughts(self) -> None:
        agent = make_agent(engine=FakeLLMEngine([]))
        task = agent._initialize_task(turns=[], prompt="hello", inputs={"thinking_rounds": 0})
        task.system_prompt_name = "role"

        rendered = agent._render_task_messages(task)

        assert len(rendered) == 1
        assert _THINKING_FRAMING not in rendered[0]["content"]

    def test_format_thoughts_renders_non_str_values_via_json_dumps(self) -> None:
        rendered = ThinkingAgent._format_thoughts(["plain", {"a": 1}])

        assert "## Round 0" in rendered
        assert "plain" in rendered
        assert "## Round 1" in rendered
        assert json.dumps({"a": 1}) in rendered

    def test_stringify_thought_matches_format_thoughts_rendering(self) -> None:
        # Guards the two call sites (per-thought thinking-phase rendering,
        # reply-phase combined block) against drifting on how a non-str
        # thought gets stringified.
        value = {"a": 1}
        assert ThinkingAgent._stringify_thought(value) == json.dumps(value)
        assert ThinkingAgent._stringify_thought(value) in ThinkingAgent._format_thoughts([value])


class TestRecordAndResultConstruction:
    def test_invoke_returns_thinking_agent_result_with_rounds_used(self) -> None:
        engine = FakeLLMEngine(["obs", "reply"])
        agent = make_agent(engine=engine)

        result = agent.invoke({"prompt": "hello", "thinking_rounds": 1})

        assert isinstance(result, ThinkingAgentResult)
        assert result.thinking_rounds_used == 1

    def test_record_thoughts_holds_full_content(self) -> None:
        engine = FakeLLMEngine(["obs", "reply"])
        agent = make_agent(engine=engine)

        result = agent.invoke({"prompt": "hello", "thinking_rounds": 1})
        record = agent.get_conversation(turns=1)[0]

        assert record.thoughts == ("obs",)
        assert record.final_result.run_id == result.run_id

    def test_second_invocation_records_its_own_thoughts_independently(self) -> None:
        engine = FakeLLMEngine(["first", "reply one", "second", "reply two"])
        agent = make_agent(engine=engine)

        agent.invoke({"prompt": "first", "thinking_rounds": 1})
        agent.invoke({"prompt": "second", "thinking_rounds": 1})

        first_record, second_record = agent.get_conversation()
        assert first_record.thoughts == ("first",)
        assert second_record.thoughts == ("second",)

    def test_to_dict_includes_thinking_schema_but_not_thoughts(self) -> None:
        engine = FakeLLMEngine(["obs", "reply"])
        agent = make_agent(engine=engine, thinking_schema={"type": "object"})
        agent.invoke({"prompt": "hello", "thinking_rounds": 1})

        data = agent.to_dict()

        assert data["thinking_schema"] == {"type": "object"}
        assert "thoughts" not in data


class TestToDictThinkingLlm:
    def test_omits_thinking_llm_when_unset(self) -> None:
        agent = make_agent(engine=FakeLLMEngine([]))

        assert "thinking_llm" not in agent.to_dict()

    def test_omits_thinking_llm_when_same_object_as_main_engine(self) -> None:
        engine = FakeLLMEngine([])
        agent = make_agent(engine=engine, thinking_llm_engine=engine)

        assert "thinking_llm" not in agent.to_dict()

    def test_includes_thinking_llm_when_distinct_engine_set(self) -> None:
        main_engine = FakeLLMEngine([])
        thinking_engine = FakeLLMEngine([])
        agent = make_agent(engine=main_engine, thinking_llm_engine=thinking_engine)

        assert agent.to_dict()["thinking_llm"] == thinking_engine.to_dict()
