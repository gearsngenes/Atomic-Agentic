from __future__ import annotations

import asyncio
from typing import Any

import pytest

from conftest import ScriptedLLMEngine

from atomic_agentic.agents.thinking import ThinkingAgent
from atomic_agentic.exceptions import AgentError, AgentInvocationError
from atomic_agentic.llm import LLMEngine
from atomic_agentic.models.agents.prompts import PromptConfig
from atomic_agentic.models.agents.records import AgentRecord, ThinkingAgentRecord
from atomic_agentic.models.agents.tasks import ThinkingTask
from atomic_agentic.models.agents.thought_models import AgentThought
from atomic_agentic.models.parameters import ParamSpec
from atomic_agentic.models.results import ThinkingAgentResult


class _NonStringEngineResult:
    """Bare stand-in for an LLMResult exposing only `.result`."""

    def __init__(self, value: Any) -> None:
        self.result = value


class NonStringLLMEngine:
    """Minimal fake engine (not an LLMEngine subclass) whose invoke()/
    async_invoke() return a non-string `.result` directly.

    A real LLMEngine subclass can't produce this: LLMEngine.invoke() itself
    enforces that `_extract_text` returns a str, raising LLMEngineError
    before ever reaching Agent code. This fake bypasses that enforcement
    entirely, since _reply/_async_reply only call
    self._llm_engine.invoke()/.async_invoke() and read `.result` -- they
    don't care about the object's type.
    """

    def invoke(self, inputs: dict[str, Any]) -> _NonStringEngineResult:
        return _NonStringEngineResult(123)

    async def async_invoke(self, inputs: dict[str, Any]) -> _NonStringEngineResult:
        return _NonStringEngineResult(123)


class StubThinkingAgent(ThinkingAgent):
    """Minimal concrete ThinkingAgent for exercising the shared base.

    Each thinking round asks the engine for the next answer and appends one
    AgentThought; an engine response of "STOP" ends thinking without
    producing a thought (a self-declared stop signal, distinct from hitting
    max_thinking_rounds).
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._system_prompts["thinking"] = PromptConfig(
            template="Think about the task.", description="Stub thinking prompt"
        )

    def _initialize_task(
        self, *, turns: list[AgentRecord], prompt: str, inputs: dict
    ) -> ThinkingTask:
        return ThinkingTask(
            turns=turns, inputs=inputs, user_prompt=prompt, system_prompt_name="thinking"
        )

    def _think(self, task: ThinkingTask) -> ThinkingTask:
        messages = self.render_task(task)
        engine_result = self._llm_engine.invoke({"messages": messages})
        return self._apply_thinking_result(task, messages, engine_result)

    async def _async_think(self, task: ThinkingTask) -> ThinkingTask:
        messages = self.render_task(task)
        engine_result = await self._llm_engine.async_invoke({"messages": messages})
        return self._apply_thinking_result(task, messages, engine_result)

    def _apply_thinking_result(
        self, task: ThinkingTask, messages: list[dict[str, str]], engine_result: Any
    ) -> ThinkingTask:
        from atomic_agentic.models.agents.records import LLMRecord

        task.llm_records.append(
            LLMRecord(
                messages=list(messages),
                llm_result=engine_result,
                system_prompt_name=task.system_prompt_name,
            )
        )
        text = engine_result.result
        if text == "STOP":
            task.system_prompt_name = "role"
            return task
        task.thoughts.append(AgentThought(observation=None, question="What next?", answer=text))
        task.rounds_used += 1
        if task.rounds_used >= self._max_thinking_rounds:
            task.system_prompt_name = "role"
        return task

    def _render_thinking_messages(
        self,
        task: ThinkingTask,
        *,
        additional_messages: list[dict[str, str]] | None = None,
    ) -> list[dict[str, str]]:
        return [{"role": "user", "content": f"THINK: {task.user_prompt}"}] + list(
            additional_messages or []
        )


def make_stub_agent(
    *,
    engine: LLMEngine | None = None,
    max_thinking_rounds: int = 3,
    **kwargs: Any,
) -> StubThinkingAgent:
    return StubThinkingAgent(
        name="stub_thinker",
        namespace="tests",
        description="Stub ThinkingAgent under test.",
        llm_engine=engine or ScriptedLLMEngine(["insight", "STOP", "final response"]),
        max_thinking_rounds=max_thinking_rounds,
        **kwargs,
    )


class TestConstructionValidation:
    def test_max_thinking_rounds_is_required(self) -> None:
        with pytest.raises(TypeError):
            StubThinkingAgent(  # type: ignore[call-arg]
                name="a", namespace="tests", description="d", llm_engine=ScriptedLLMEngine(["x"])
            )

    @pytest.mark.parametrize("value", [0, -1, "3", 1.5, True])
    def test_max_thinking_rounds_rejects_non_positive_or_non_int(self, value: Any) -> None:
        with pytest.raises(TypeError, match="max_thinking_rounds"):
            make_stub_agent(max_thinking_rounds=value)  # type: ignore[arg-type]

    @pytest.mark.parametrize("value", [-1, "1", 1.5, True])
    def test_generation_retries_rejects_invalid(self, value: Any) -> None:
        with pytest.raises(TypeError, match="generation_retries"):
            make_stub_agent(generation_retries=value)  # type: ignore[arg-type]

    @pytest.mark.parametrize("value", [1, "True", 0])
    def test_render_thoughts_in_history_rejects_non_bool(self, value: Any) -> None:
        with pytest.raises(TypeError, match="render_thoughts_in_history"):
            make_stub_agent(render_thoughts_in_history=value)  # type: ignore[arg-type]


class TestRolePromptWrapping:
    def test_string_role_prompt_is_fenced(self) -> None:
        agent = make_stub_agent(role_prompt="You are a helpful poet.")
        template = agent._system_prompts["role"].template

        assert "=== RESPONSE INSTRUCTIONS START ===" in template
        assert "You are a helpful poet." in template
        assert "=== RESPONSE INSTRUCTIONS END ===" in template
        assert "thoughts" in template.lower()

    def test_default_role_prompt_used_when_none(self) -> None:
        agent = make_stub_agent(role_prompt=None)
        assert ThinkingAgent.DEFAULT_ROLE_PROMPT in agent._system_prompts["role"].template

    def test_prompt_config_role_prompt_placeholders_survive_the_wrap(self) -> None:
        agent = make_stub_agent(
            role_prompt=PromptConfig(template="You are a {persona}.", description="d")
        )
        param_names = {p.name for p in agent._system_prompts["role"].parameters}
        assert "persona" in param_names

    def test_role_prompt_placeholder_resolves_through_invoke(self) -> None:
        engine = ScriptedLLMEngine(["STOP", "A wizard's response."])
        agent = make_stub_agent(
            engine=engine,
            role_prompt=PromptConfig(template="You are a {persona}.", description="d"),
        )
        agent.invoke({"prompt": "hi", "persona": "wizard"})

        reply_call = engine.calls[-1]
        system_message = reply_call[0]["content"]
        assert "wizard" in system_message


class TestAdditionalInstructions:
    def test_stored_resolved_but_not_applied_to_role_prompt(self) -> None:
        agent = make_stub_agent(
            role_prompt="Be concise.",
            additional_instructions="Always answer in French.",
        )
        assert agent._additional_instructions_config is not None
        assert agent._additional_instructions_config.template == "Always answer in French."
        assert "French" not in agent._system_prompts["role"].template

    def test_none_additional_instructions_stored_as_none(self) -> None:
        agent = make_stub_agent()
        assert agent._additional_instructions_config is None


class TestExtraThinkingParams:
    def test_extra_thinking_params_reach_the_schema(self) -> None:
        agent = make_stub_agent(extra_thinking_params=["topic"])
        assert "topic" in {p.name for p in agent.parameters}

    def test_extra_thinking_params_collision_with_role_prompt_raises(self) -> None:
        # Role-prompt-discovered placeholders are always KEYWORD_ONLY/"Any",
        # so an incompatible collision requires a mismatched variadic kind
        # on the extra_thinking_params side sharing the same name.
        with pytest.raises(AgentError, match="topic"):
            make_stub_agent(
                role_prompt=PromptConfig(template="Discuss {topic}.", description="d"),
                extra_thinking_params=[
                    ParamSpec(name="topic", index=0, kind=ParamSpec.VAR_KEYWORD, type="Any"),
                ],
            )

    def test_extra_thinking_params_compatible_overlap_role_prompt_wins(self) -> None:
        agent = make_stub_agent(
            role_prompt=PromptConfig(template="Discuss {topic}.", description="d"),
            extra_thinking_params=["topic"],
        )
        topic_params = [p for p in agent.parameters if p.name == "topic"]
        assert len(topic_params) == 1


class TestProgressAndRenderTaskDispatch:
    def test_progress_dispatches_to_think_when_not_role(self) -> None:
        agent = make_stub_agent(engine=ScriptedLLMEngine(["STOP"]))
        task = agent._initialize_task(turns=[], prompt="write a haiku", inputs={"prompt": "write a haiku"})

        assert task.system_prompt_name == "thinking"
        agent._progress(task)
        assert task.system_prompt_name == "role"

    def test_progress_dispatches_to_reply_when_role(self) -> None:
        agent = make_stub_agent(engine=ScriptedLLMEngine(["final answer"]))
        task = agent._initialize_task(turns=[], prompt="write a haiku", inputs={"prompt": "write a haiku"})
        task.system_prompt_name = "role"

        agent._progress(task)

        assert task.complete is True
        assert task.generated_response == "final answer"

    def test_render_task_dispatches_to_thinking_messages(self) -> None:
        agent = make_stub_agent(engine=ScriptedLLMEngine(["STOP"]))
        task = agent._initialize_task(turns=[], prompt="write a haiku", inputs={"prompt": "write a haiku"})

        messages = agent.render_task(task)

        assert messages == [{"role": "user", "content": "THINK: write a haiku"}]

    def test_render_task_builds_shared_reply_shape(self) -> None:
        agent = make_stub_agent(engine=ScriptedLLMEngine(["final"]))
        task = agent._initialize_task(turns=[], prompt="write a haiku", inputs={"prompt": "write a haiku"})
        task.system_prompt_name = "role"

        messages = agent.render_task(task)
        contents = [m["content"] for m in messages]

        assert any("===== CURRENT TASK =====" in c and "write a haiku" in c for c in contents)
        assert any("THOUGHTS SO FAR" in c for c in contents)


class TestThoughtsSnapshot:
    def test_empty_case_fallback(self) -> None:
        agent = make_stub_agent()
        task = ThinkingTask(turns=[], inputs={}, user_prompt="x", system_prompt_name="role")

        assert agent._render_thoughts_snapshot(task) == "THOUGHTS SO FAR:\nNo thoughts yet."

    def test_non_empty_snapshot_includes_thought_fields(self) -> None:
        agent = make_stub_agent()
        task = ThinkingTask(turns=[], inputs={}, user_prompt="x", system_prompt_name="role")
        task.thoughts.append(AgentThought(observation="obs", question="q?", answer="a."))

        snapshot = agent._render_thoughts_snapshot(task)

        assert "THOUGHTS 0-0 SO FAR" in snapshot
        assert "obs" in snapshot and "q?" in snapshot and "a." in snapshot


class TestReplyNonStringResult:
    def test_reply_raises_on_non_string_result(self) -> None:
        agent = make_stub_agent(engine=NonStringLLMEngine())  # type: ignore[arg-type]
        task = agent._initialize_task(turns=[], prompt="hi", inputs={"prompt": "hi"})
        task.system_prompt_name = "role"

        with pytest.raises(AgentInvocationError, match="non-string result"):
            agent._reply(task)

    def test_async_reply_raises_on_non_string_result(self) -> None:
        agent = make_stub_agent(engine=NonStringLLMEngine())  # type: ignore[arg-type]
        task = agent._initialize_task(turns=[], prompt="hi", inputs={"prompt": "hi"})
        task.system_prompt_name = "role"

        async def run() -> None:
            await agent._async_reply(task)

        with pytest.raises(AgentInvocationError, match="non-string result"):
            asyncio.run(run())


class TestFullInvokeLifecycle:
    def test_invoke_thinks_then_replies(self) -> None:
        agent = make_stub_agent(engine=ScriptedLLMEngine(["insight one", "STOP", "final response"]))

        result = agent.invoke({"prompt": "write a haiku"})

        assert result.result == "final response"

    def test_max_thinking_rounds_forces_reply_without_self_declared_stop(self) -> None:
        agent = make_stub_agent(
            engine=ScriptedLLMEngine(["a", "b", "c", "final response"]),
            max_thinking_rounds=3,
        )

        result = agent.invoke({"prompt": "write a haiku"})

        assert result.result == "final response"

    def test_build_record_from_task_persists_thoughts_span(self) -> None:
        agent = make_stub_agent(engine=ScriptedLLMEngine(["insight one", "STOP", "final response"]))

        agent.invoke({"prompt": "write a haiku"})

        record = agent.records[-1]
        assert isinstance(record, ThinkingAgentRecord)
        assert record.thoughts_start == 0
        assert record.thoughts_end == 1
        assert agent._thoughts[record.thoughts_start:record.thoughts_end][0].answer == "insight one"

    def test_invoke_returns_thinking_agent_result_with_thoughts_span(self) -> None:
        agent = make_stub_agent(engine=ScriptedLLMEngine(["insight one", "STOP", "final response"]))

        result = agent.invoke({"prompt": "write a haiku"})

        assert isinstance(result, ThinkingAgentResult)
        assert result.thoughts_start == 0
        assert result.thoughts_end == 1
        assert agent._thoughts[result.thoughts_start:result.thoughts_end][0].answer == "insight one"

    def test_multi_round_result_thoughts_span_covers_every_round(self) -> None:
        agent = make_stub_agent(
            engine=ScriptedLLMEngine(["first", "second", "STOP", "final response"]),
            max_thinking_rounds=5,
        )

        result = agent.invoke({"prompt": "write a haiku"})

        answers = [t.answer for t in agent._thoughts[result.thoughts_start:result.thoughts_end]]
        assert answers == ["first", "second"]

    def test_async_invoke_thinks_then_replies(self) -> None:
        agent = make_stub_agent(engine=ScriptedLLMEngine(["insight one", "STOP", "final response"]))

        result = asyncio.run(agent.async_invoke({"prompt": "write a haiku"}))

        assert result.result == "final response"
        assert isinstance(result, ThinkingAgentResult)
        assert result.thoughts_end - result.thoughts_start == 1


class TestRenderTurnGating:
    def test_render_thoughts_in_history_false_matches_base_render_turn(self) -> None:
        agent = make_stub_agent(
            engine=ScriptedLLMEngine(["insight one", "STOP", "final response"]),
            render_thoughts_in_history=False,
        )
        agent.invoke({"prompt": "write a haiku"})
        record = agent.records[-1]

        assert agent.render_turn(record) == super(ThinkingAgent, agent).render_turn(record)

    def test_render_thoughts_in_history_true_splices_thought_trail(self) -> None:
        agent = make_stub_agent(
            engine=ScriptedLLMEngine(["insight one", "STOP", "final response"]),
            render_thoughts_in_history=True,
        )
        agent.invoke({"prompt": "write a haiku"})
        record = agent.records[-1]

        messages = agent.render_turn(record)

        assert len(messages) == 2
        assert "RESPONSE:" in messages[1]["content"]
        assert "insight one" in messages[1]["content"]

    def test_render_turn_rejects_wrong_record_type(self) -> None:
        agent = make_stub_agent()

        with pytest.raises(AgentInvocationError, match="ThinkingAgentRecord"):
            agent.render_turn(AgentRecord(user_prompt="x", generated_response="y"))
