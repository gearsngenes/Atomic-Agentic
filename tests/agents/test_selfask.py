from __future__ import annotations

from typing import Any
import asyncio

import pytest

from conftest import ScriptedLLMEngine

from atomic_agentic.agents.selfask import SelfAskAgent
from atomic_agentic.exceptions import AgentError, AgentInvocationError, ThinkingAgentError
from atomic_agentic.models.agents.prompts import PromptConfig
from atomic_agentic.models.agents.thought_models import AgentThought
from atomic_agentic.models.results.agents import ThinkingAgentResult


def thought_line(category: str, content: str) -> str:
    return f"[{category}] {content}"


def make_agent(
    *,
    engine: ScriptedLLMEngine | None = None,
    role_prompt: str | PromptConfig | None = "You are a careful assistant.",
    thinking_instructions: str | PromptConfig | None = None,
    max_thinking_rounds: int = 3,
    thoughts_per_round: int = 1,
    context_enabled: bool = True,
) -> SelfAskAgent:
    return SelfAskAgent(
        name="tests",
        namespace="tests",
        description="SelfAskAgent under test.",
        llm_engine=engine or ScriptedLLMEngine([]),
        role_prompt=role_prompt,
        thinking_instructions=thinking_instructions,
        max_thinking_rounds=max_thinking_rounds,
        thoughts_per_round=thoughts_per_round,
        context_enabled=context_enabled,
    )


class TestConstruction:
    def test_max_thinking_rounds_zero_is_legal(self) -> None:
        agent = make_agent(max_thinking_rounds=0)
        assert agent is not None

    def test_max_thinking_rounds_none_raises(self) -> None:
        with pytest.raises(AgentError, match="max_thinking_rounds"):
            make_agent(max_thinking_rounds=None)  # type: ignore[arg-type]

    def test_max_thinking_rounds_negative_raises(self) -> None:
        with pytest.raises(AgentError, match="max_thinking_rounds"):
            make_agent(max_thinking_rounds=-1)

    def test_max_thinking_rounds_non_int_raises(self) -> None:
        with pytest.raises(AgentError, match="max_thinking_rounds"):
            make_agent(max_thinking_rounds=1.5)  # type: ignore[arg-type]

    def test_thoughts_per_round_zero_raises(self) -> None:
        with pytest.raises(AgentError, match="thoughts_per_round"):
            make_agent(thoughts_per_round=0)

    def test_thoughts_per_round_negative_raises(self) -> None:
        with pytest.raises(AgentError, match="thoughts_per_round"):
            make_agent(thoughts_per_round=-1)

    def test_thoughts_per_round_non_int_raises(self) -> None:
        with pytest.raises(AgentError, match="thoughts_per_round"):
            make_agent(thoughts_per_round=1.5)  # type: ignore[arg-type]

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
        with pytest.raises(AgentError, match="collision"):
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


class TestThinkPhase:
    def test_max_thinking_rounds_zero_skips_thinking_without_engine_call(self) -> None:
        engine = ScriptedLLMEngine(["final reply"])
        agent = make_agent(engine=engine, max_thinking_rounds=0)

        result = agent.invoke({"prompt": "hello"})

        assert result.result == "final reply"
        assert len(engine.calls) == 1  # only the reply-phase call, no thinking round
        assert agent.get_thoughts(result.run_id) == []

    def test_stop_sentinel_ends_thinking_after_that_round(self) -> None:
        engine = ScriptedLLMEngine([
            thought_line("OBSERVATION", "noted something") + "\n|STOP_THINKING|",
            "final reply",
        ])
        agent = make_agent(engine=engine, max_thinking_rounds=5)

        result = agent.invoke({"prompt": "hello"})

        assert result.result == "final reply"
        thoughts = agent.get_thoughts(result.run_id)
        assert len(thoughts) == 1
        assert thoughts[0][0].category == "OBSERVATION"
        assert thoughts[0][0].content == "noted something"

    def test_stop_sentinel_reply_phase_receives_correct_instruction_not_stale_one(self) -> None:
        """Regression test for the phase-transition bug this session's
        critique caught and fixed: task.task_messages must be cleared when
        think() switches to the reply phase via |STOP_THINKING|, otherwise
        act() would reuse the stale self-ask-phase "produce next round of
        thoughts" instruction instead of building the real reply prompt."""
        engine = ScriptedLLMEngine([
            thought_line("OBSERVATION", "noted something") + "\n|STOP_THINKING|",
            "final reply",
        ])
        agent = make_agent(engine=engine, max_thinking_rounds=5)

        agent.invoke({"prompt": "hello"})

        reply_call_messages = engine.calls[-1]
        reply_user_messages = [m["content"] for m in reply_call_messages if m["role"] == "user"]
        joined = "\n".join(reply_user_messages)
        assert "respond to the current task" in joined
        assert "Produce the next round of thoughts" not in joined

    def test_round_budget_exhaustion_after_real_round_also_clears_stale_messages(self) -> None:
        engine = ScriptedLLMEngine([
            thought_line("OBSERVATION", "only round, no sentinel"),
            "final reply",
        ])
        agent = make_agent(engine=engine, max_thinking_rounds=1)

        agent.invoke({"prompt": "hello"})

        reply_call_messages = engine.calls[-1]
        reply_user_messages = [m["content"] for m in reply_call_messages if m["role"] == "user"]
        joined = "\n".join(reply_user_messages)
        assert "respond to the current task" in joined
        assert "Produce the next round of thoughts" not in joined

    def test_empty_raw_output_raises_thinking_agent_error(self) -> None:
        engine = ScriptedLLMEngine([""])
        agent = make_agent(engine=engine, max_thinking_rounds=3)

        with pytest.raises(ThinkingAgentError):
            agent.invoke({"prompt": "hello"})

    def test_bare_stop_sentinel_with_no_content_raises_thinking_agent_error(self) -> None:
        engine = ScriptedLLMEngine(["|STOP_THINKING|"])
        agent = make_agent(engine=engine, max_thinking_rounds=3)

        with pytest.raises(ThinkingAgentError):
            agent.invoke({"prompt": "hello"})

    def test_unmarked_text_degrades_to_single_other_thought(self) -> None:
        engine = ScriptedLLMEngine([
            "just some unmarked text" + "\n|STOP_THINKING|",
            "final reply",
        ])
        agent = make_agent(engine=engine, max_thinking_rounds=3)

        result = agent.invoke({"prompt": "hello"})

        thoughts = agent.get_thoughts(result.run_id)
        assert len(thoughts) == 1
        assert thoughts[0][0].category == "OTHER"
        assert thoughts[0][0].content == "just some unmarked text"

    def test_thoughts_per_round_truncates_excess_thoughts_silently(self) -> None:
        engine = ScriptedLLMEngine([
            "\n".join([
                thought_line("OBSERVATION", "first"),
                thought_line("QUESTION", "second"),
                thought_line("REASON", "third"),
            ]) + "\n|STOP_THINKING|",
            "final reply",
        ])
        agent = make_agent(engine=engine, max_thinking_rounds=3, thoughts_per_round=2)

        result = agent.invoke({"prompt": "hello"})

        thoughts = agent.get_thoughts(result.run_id)
        assert len(thoughts) == 1
        assert len(thoughts[0]) == 2
        assert [t.content for t in thoughts[0]] == ["first", "second"]

    def test_multiple_rounds_accumulate_before_stop(self) -> None:
        engine = ScriptedLLMEngine([
            thought_line("OBSERVATION", "round one"),
            thought_line("QUESTION", "round two") + "\n|STOP_THINKING|",
            "final reply",
        ])
        agent = make_agent(engine=engine, max_thinking_rounds=5)

        result = agent.invoke({"prompt": "hello"})

        thoughts = agent.get_thoughts(result.run_id)
        assert len(thoughts) == 2
        assert thoughts[0][0].content == "round one"
        assert thoughts[1][0].content == "round two"

    def test_async_think_mirrors_sync_behavior(self) -> None:
        engine = ScriptedLLMEngine([
            thought_line("OBSERVATION", "noted") + "\n|STOP_THINKING|",
            "final reply",
        ])
        agent = make_agent(engine=engine, max_thinking_rounds=5)

        result = asyncio.run(agent.async_invoke({"prompt": "hello"}))

        assert result.result == "final reply"
        thoughts = agent.get_thoughts(result.run_id)
        assert len(thoughts) == 1
        assert thoughts[0][0].content == "noted"


class TestActPhase:
    def test_act_no_ops_while_still_thinking(self) -> None:
        agent = make_agent(engine=ScriptedLLMEngine([]), max_thinking_rounds=3)
        task = agent._initialize_task(turns=[], prompt="hello", inputs={})
        assert task.system_prompt_name != "role"

        updated = agent.act(task)

        assert updated is task
        assert updated.complete is False

    def test_async_act_no_ops_while_still_thinking(self) -> None:
        agent = make_agent(engine=ScriptedLLMEngine([]), max_thinking_rounds=3)
        task = agent._initialize_task(turns=[], prompt="hello", inputs={})

        updated = asyncio.run(agent.async_act(task))

        assert updated is task
        assert updated.complete is False

    def test_act_delegates_to_basic_agent_body_once_in_reply_phase(self) -> None:
        engine = ScriptedLLMEngine(["reply text"])
        agent = make_agent(engine=engine, max_thinking_rounds=0)
        task = agent._initialize_task(turns=[], prompt="hello", inputs={})
        task = agent.think(task)  # max_thinking_rounds=0 -> switches straight to "role"
        assert task.system_prompt_name == "role"

        updated = agent.act(task)

        assert updated.complete is True
        assert updated.generated_response == "reply text"


class TestRenderPipeline:
    def test_self_ask_system_message_resolves_round_limit_and_thoughts_per_round(self) -> None:
        agent = make_agent(
            engine=ScriptedLLMEngine([]),
            max_thinking_rounds=4,
            thoughts_per_round=2,
        )
        task = agent._initialize_task(turns=[], prompt="hello", inputs={})

        rendered = agent._render_system_message(task)

        content = rendered[0]["content"]
        assert "4" in content
        assert "2" in content

    def test_self_ask_system_message_includes_thinking_instructions_when_non_empty(self) -> None:
        agent = make_agent(
            engine=ScriptedLLMEngine([]),
            thinking_instructions="Focus on edge cases.",
            max_thinking_rounds=3,
        )
        task = agent._initialize_task(turns=[], prompt="hello", inputs={})

        rendered = agent._render_system_message(task)

        assert "Focus on edge cases." in rendered[0]["content"]

    def test_self_ask_system_message_omits_instructions_section_when_absent(self) -> None:
        agent = make_agent(engine=ScriptedLLMEngine([]), thinking_instructions=None, max_thinking_rounds=3)
        task = agent._initialize_task(turns=[], prompt="hello", inputs={})

        rendered = agent._render_system_message(task)

        assert "ADDITIONAL INSTRUCTIONS" not in rendered[0]["content"]

    def test_role_phase_system_message_delegates_to_role_prompt(self) -> None:
        agent = make_agent(
            engine=ScriptedLLMEngine([]),
            role_prompt="You are a distinctly-worded persona.",
            max_thinking_rounds=0,
        )
        task = agent._initialize_task(turns=[], prompt="hello", inputs={})
        task = agent.think(task)  # switches to "role"

        rendered = agent._render_system_message(task)

        assert rendered[0]["content"] == "You are a distinctly-worded persona."

    def test_self_ask_task_messages_banner_only_on_first_round(self) -> None:
        agent = make_agent(engine=ScriptedLLMEngine([]), max_thinking_rounds=3)
        task = agent._initialize_task(turns=[], prompt="hello", inputs={})

        rendered = agent._render_task_messages(task)

        assert len(rendered) == 1
        assert "hello" in rendered[0]["content"]

    def test_self_ask_task_messages_includes_snapshot_on_later_rounds(self) -> None:
        agent = make_agent(engine=ScriptedLLMEngine([]), max_thinking_rounds=3)
        task = agent._initialize_task(turns=[], prompt="hello", inputs={})
        task.thoughts.append([AgentThought(category="OBSERVATION", content="prior thought")])

        rendered = agent._render_task_messages(task)

        assert len(rendered) == 3
        assert "prior thought" in rendered[1]["content"]
        assert "Produce the next round of thoughts" in rendered[2]["content"]

    def test_role_phase_task_messages_include_thoughts_snapshot_regardless_of_stop_reason(self) -> None:
        agent = make_agent(engine=ScriptedLLMEngine([]), max_thinking_rounds=3)
        task = agent._initialize_task(turns=[], prompt="hello", inputs={})
        task.thoughts.append([AgentThought(category="OBSERVATION", content="reasoned thing")])
        task.system_prompt_name = "role"

        rendered = agent._render_task_messages(task)

        assert len(rendered) == 3
        assert "reasoned thing" in rendered[1]["content"]
        assert "respond to the current task" in rendered[2]["content"]

    def test_role_phase_task_messages_banner_only_when_no_thoughts(self) -> None:
        agent = make_agent(engine=ScriptedLLMEngine([]), max_thinking_rounds=0)
        task = agent._initialize_task(turns=[], prompt="hello", inputs={})
        task.system_prompt_name = "role"

        rendered = agent._render_task_messages(task)

        assert len(rendered) == 1


class TestGetThoughts:
    def test_get_thoughts_none_resolves_to_latest_record(self) -> None:
        engine = ScriptedLLMEngine([
            thought_line("OBSERVATION", "obs one") + "\n|STOP_THINKING|",
            "reply one",
        ])
        agent = make_agent(engine=engine, max_thinking_rounds=3)
        agent.invoke({"prompt": "hello"})

        assert agent.get_thoughts(None) == [[AgentThought(category="OBSERVATION", content="obs one")]]

    def test_get_thoughts_unknown_run_id_raises(self) -> None:
        agent = make_agent(engine=ScriptedLLMEngine([]), max_thinking_rounds=3)

        with pytest.raises(AgentInvocationError):
            agent.get_thoughts("not-a-real-run-id")

    def test_get_thoughts_empty_history_returns_empty_list(self) -> None:
        agent = make_agent(engine=ScriptedLLMEngine([]), max_thinking_rounds=3)

        assert agent.get_thoughts(None) == []

    def test_get_thoughts_by_explicit_run_id_isolates_that_invocation(self) -> None:
        engine = ScriptedLLMEngine([
            thought_line("OBSERVATION", "first invoke thought") + "\n|STOP_THINKING|",
            "reply one",
            thought_line("QUESTION", "second invoke thought") + "\n|STOP_THINKING|",
            "reply two",
        ])
        agent = make_agent(engine=engine, max_thinking_rounds=3)
        first_result = agent.invoke({"prompt": "first"})
        second_result = agent.invoke({"prompt": "second"})

        first_thoughts = agent.get_thoughts(first_result.run_id)
        second_thoughts = agent.get_thoughts(second_result.run_id)

        assert first_thoughts == [[AgentThought(category="OBSERVATION", content="first invoke thought")]]
        assert second_thoughts == [[AgentThought(category="QUESTION", content="second invoke thought")]]


class TestClearMemory:
    def test_clear_memory_clears_records_and_thoughts(self) -> None:
        engine = ScriptedLLMEngine([
            thought_line("OBSERVATION", "obs") + "\n|STOP_THINKING|",
            "reply",
        ])
        agent = make_agent(engine=engine, max_thinking_rounds=3)
        agent.invoke({"prompt": "hello"})

        assert agent.records
        assert agent.get_thoughts(None)

        agent.clear_memory()

        assert agent.records == []
        assert agent.get_thoughts(None) == []


class TestRecordAndResultConstruction:
    def test_invoke_returns_thinking_agent_result_with_thoughts_span(self) -> None:
        engine = ScriptedLLMEngine([
            thought_line("OBSERVATION", "obs") + "\n|STOP_THINKING|",
            "reply",
        ])
        agent = make_agent(engine=engine, max_thinking_rounds=3)

        result = agent.invoke({"prompt": "hello"})

        assert isinstance(result, ThinkingAgentResult)
        assert result.thoughts_start == 0
        assert result.thoughts_end == 1

    def test_second_invocation_thoughts_span_starts_where_first_ended(self) -> None:
        engine = ScriptedLLMEngine([
            thought_line("OBSERVATION", "first") + "\n|STOP_THINKING|",
            "reply one",
            thought_line("QUESTION", "second") + "\n|STOP_THINKING|",
            "reply two",
        ])
        agent = make_agent(engine=engine, max_thinking_rounds=3)

        first_result = agent.invoke({"prompt": "first"})
        second_result = agent.invoke({"prompt": "second"})

        assert first_result.thoughts_start == 0
        assert first_result.thoughts_end == 1
        assert second_result.thoughts_start == 1
        assert second_result.thoughts_end == 2

    def test_to_dict_includes_persisted_thoughts(self) -> None:
        engine = ScriptedLLMEngine([
            thought_line("OBSERVATION", "obs") + "\n|STOP_THINKING|",
            "reply",
        ])
        agent = make_agent(engine=engine, max_thinking_rounds=3)
        agent.invoke({"prompt": "hello"})

        data = agent.to_dict()

        assert "thoughts" in data
        assert data["thoughts"] == [[{"category": "OBSERVATION", "content": "obs"}]]
