from __future__ import annotations

import asyncio
from typing import Any

import pytest

from atomic_agentic.agents.basic import BasicAgent
from atomic_agentic.agents.thinking import ThinkingAgent
from atomic_agentic.constants.agents import (
    CONTEXT_START,
    REPLY_DIRECTIVE,
    TASK_PROMPT_END,
    TASK_PROMPT_START,
    THINKING_DIRECTIVE,
    THOUGHTS_END,
    THOUGHTS_START,
)
from atomic_agentic.models.agents.prompts import PromptConfig
from atomic_agentic.models.agents.records import LLMRecord, ThinkingAgentRecord
from atomic_agentic.models.agents.tasks import ThinkingTask
from atomic_agentic.models.results.agents import AgentResult, ThinkingAgentResult

from .conftest import FakeLLMEngine, StubThinker, make_stub_thinker

HOOKS = ("_thinking_done", "_generate_thoughts", "_agenerate_thoughts")
def wrap(prompt: str) -> str:
    return f"{TASK_PROMPT_START}\n{prompt}\n{TASK_PROMPT_END}"


def new_task(agent: StubThinker, prompt: str = "hi") -> ThinkingTask:
    """A task as ``invoke`` would hand it to the loop (after pre_process)."""
    task = agent.create_task(agent.filter_inputs({"prompt": prompt}))
    task = agent.summarize(task)
    return agent.pre_process(task)


def no_consecutive_users(messages: list[dict[str, str]]) -> bool:
    roles = [m["role"] for m in messages]
    return all(not (a == b == "user") for a, b in zip(roles, roles[1:]))


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────
def valid_kwargs() -> dict[str, Any]:
    return {
        "name": "tests", "namespace": "tests", "description": "abstractness",
        "llm_engine": FakeLLMEngine([]),
    }


class TestAbstractness:
    def test_base_cannot_be_instantiated(self) -> None:
        with pytest.raises(TypeError, match="abstract"):
            ThinkingAgent(**valid_kwargs())  # type: ignore[abstract]

    @pytest.mark.parametrize("missing", HOOKS)
    def test_subclass_missing_a_hook_cannot_be_instantiated(self, missing: str) -> None:
        impls = {
            "_thinking_done": lambda self, task: True,
            "_generate_thoughts": lambda self, task: None,
            "_agenerate_thoughts": lambda self, task: None,
        }
        partial = type("Partial", (ThinkingAgent,), {k: v for k, v in impls.items() if k != missing})

        with pytest.raises(TypeError, match=missing):
            partial(**valid_kwargs())

    def test_subclass_with_all_three_hooks_can_be_instantiated(self) -> None:
        full = type("Full", (ThinkingAgent,), {
            "_thinking_done": lambda self, task: True,
            "_generate_thoughts": lambda self, task: None,
            "_agenerate_thoughts": lambda self, task: None,
        })

        assert isinstance(full(**valid_kwargs()), ThinkingAgent)

    def test_stub_can_be_instantiated(self) -> None:
        assert isinstance(make_stub_thinker([]), ThinkingAgent)

    def test_abstract_methods_are_exactly_the_three_hooks(self) -> None:
        assert ThinkingAgent.__abstractmethods__ == frozenset(HOOKS)

    def test_task_class(self) -> None:
        assert ThinkingAgent._TASK_CLS is ThinkingTask

    def test_constructor_is_basic_agents(self) -> None:
        assert ThinkingAgent.__init__ is BasicAgent.__init__


class TestPhaseGating:
    def test_every_task_starts_in_the_thinking_phase(self) -> None:
        for done_after in (0, 1, 2):
            agent = make_stub_thinker([], done_after=done_after)

            task = agent.create_task(agent.filter_inputs({"prompt": "hi"}))

            assert task.system_prompt_name == "thinking"

    def test_think_flips_to_reply_when_done_without_a_call(self) -> None:
        agent = make_stub_thinker([], done_after=0)
        task = new_task(agent)

        agent.think(task)

        assert agent._llm_engine.call_count == 0
        assert task.thoughts == []
        assert task.system_prompt_name == "role"
        assert not task.complete

    def test_think_generates_while_not_done_and_does_not_flip_that_pass(self) -> None:
        agent = make_stub_thinker(["T0"], done_after=1)
        task = new_task(agent)

        agent.think(task)

        # the stub's hooks never assign the prompt name; the flip is the base's
        assert task.thoughts == ["T0"]
        assert task.system_prompt_name == "thinking"

        agent.think(task)

        assert task.system_prompt_name == "role"
        assert agent._llm_engine.call_count == 1

    def test_think_after_the_flip_never_generates_again(self) -> None:
        agent = make_stub_thinker(["T0"], done_after=1)
        task = new_task(agent)

        for _ in range(4):
            agent.think(task)

        assert agent._llm_engine.call_count == 1
        assert task.system_prompt_name == "role"

    def test_act_in_thinking_phase_is_a_noop(self) -> None:
        agent = make_stub_thinker([], done_after=1)
        task = new_task(agent)

        agent.act(task)

        assert agent._llm_engine.call_count == 0
        assert not task.complete
        assert task.system_prompt_name == "thinking"

    def test_async_gates_match_sync(self) -> None:
        reply_agent = make_stub_thinker([], done_after=0)
        think_agent = make_stub_thinker([], done_after=1)
        reply_task = new_task(reply_agent)
        think_task = new_task(think_agent)

        asyncio.run(reply_agent.async_think(reply_task))
        asyncio.run(think_agent.async_act(think_task))

        assert reply_task.system_prompt_name == "role"
        assert reply_agent._llm_engine.call_count == 0
        assert think_agent._llm_engine.call_count == 0
        assert not think_task.complete

    @pytest.mark.parametrize("done_after", [1, 2])
    def test_the_pass_after_the_last_thought_flips_and_replies(self, done_after: int) -> None:
        agent = make_stub_thinker([f"T{i}" for i in range(done_after)] + ["REPLY"], done_after=done_after)
        task = new_task(agent)

        for _ in range(done_after):
            task = agent.act(agent.prepare(agent.think(task)))
            assert not task.complete
            assert task.system_prompt_name == "thinking"

        task = agent.act(agent.prepare(agent.think(task)))

        assert task.complete
        assert task.generated_response == "REPLY"
        assert agent._llm_engine.call_count == done_after + 1
        assert [r.system_prompt_name for r in task.llm_records] == ["thinking"] * done_after + ["role"]

    def test_done_after_zero_makes_exactly_one_call(self) -> None:
        agent = make_stub_thinker(["REPLY"], done_after=0)

        result = agent.invoke({"prompt": "hi"})

        assert result.result == "REPLY"
        assert agent._llm_engine.call_count == 1


class TestRendering:
    def test_round_one_is_one_merged_user_message(self) -> None:
        agent = make_stub_thinker(["T", "R"], done_after=1)

        agent.invoke({"prompt": "hi"})

        first = agent._llm_engine.calls[0]
        assert [m["role"] for m in first] == ["system", "user"]
        assert first[1]["content"] == f"{wrap('hi')}\n\n{THINKING_DIRECTIVE}"

    def test_round_two_is_block_thoughts_directive(self) -> None:
        agent = make_stub_thinker(["T1", "T2", "R"], done_after=2)

        agent.invoke({"prompt": "hi"})

        second = agent._llm_engine.calls[1]
        assert [m["role"] for m in second] == ["system", "user", "assistant", "user"]
        assert second[1]["content"] == wrap("hi")
        assert second[2]["content"] == f"{THOUGHTS_START}\nT1\n{THOUGHTS_END}"
        assert second[3]["content"] == THINKING_DIRECTIVE

    def test_reply_with_thoughts_has_three_message_shape(self) -> None:
        agent = make_stub_thinker(["T1", "T2", "R"], done_after=2)

        agent.invoke({"prompt": "hi"})

        reply = agent._llm_engine.calls[2]
        assert [m["role"] for m in reply] == ["system", "user", "assistant", "user"]
        assert reply[2]["content"] == f"{THOUGHTS_START}\nT1\nT2\n{THOUGHTS_END}"
        assert reply[3]["content"] == REPLY_DIRECTIVE

    def test_reply_with_zero_thoughts_is_just_the_task_block(self) -> None:
        agent = make_stub_thinker(["R"], done_after=0)

        agent.invoke({"prompt": "hi"})

        call = agent._llm_engine.calls[0]
        assert [m["role"] for m in call] == ["system", "user"]
        assert call[1]["content"] == wrap("hi")
        assert REPLY_DIRECTIVE not in call[1]["content"]
        assert THINKING_DIRECTIVE not in call[1]["content"]

    @pytest.mark.parametrize("with_context", [False, True])
    def test_zero_thinking_call_equals_basic_agent_call(self, with_context: bool) -> None:
        basic = BasicAgent(
            name="tests", namespace="tests", description="basic",
            llm_engine=FakeLLMEngine(["R"]), role_prompt="You are a careful assistant.",
        )
        thinker = make_stub_thinker(["R"], done_after=0)
        if with_context:
            basic.get_conversation().remember("draft", "Hello there", "the latest draft")
            thinker.get_conversation().remember("draft", "Hello there", "the latest draft")

        basic.invoke({"prompt": "hi"})
        thinker.invoke({"prompt": "hi"})

        assert thinker._llm_engine.calls[0] == basic._llm_engine.calls[0]

    @pytest.mark.parametrize("done_after", [0, 1, 2, 3])
    def test_every_call_ends_on_a_user_message(self, done_after: int) -> None:
        agent = make_stub_thinker([f"T{i}" for i in range(done_after)] + ["R"], done_after=done_after)

        agent.invoke({"prompt": "hi"})

        assert all(call[-1]["role"] == "user" for call in agent._llm_engine.calls)

    def test_strategy_can_use_several_thinking_prompt_names(self) -> None:
        class StepThinker(StubThinker):
            def __init__(self, **kwargs) -> None:
                super().__init__(**kwargs)
                self._system_prompts["step0"] = PromptConfig(template="Step zero.", description="s0")
                self._system_prompts["step1"] = PromptConfig(template="Step one.", description="s1")

            _INITIAL_PROMPT_NAME = "step0"

            def _generate_thoughts(self, task: ThinkingTask) -> None:
                task.system_prompt_name = f"step{len(task.thoughts)}"
                messages = self.render_task(task)
                result = self._llm_engine.invoke({"messages": messages})
                task.llm_records.append(
                    LLMRecord(
                        messages=list(task.task_messages),
                        llm_result=result,
                        system_prompt_name=task.system_prompt_name,
                    )
                )
                task.thoughts.append(result.result)

        agent = StepThinker(
            name="tests", namespace="tests", description="steps",
            llm_engine=FakeLLMEngine(["A", "B", "R"]),
            role_prompt="Reply role.", record_enabled=True, done_after=2,
        )

        agent.invoke({"prompt": "hi"})

        calls = agent._llm_engine.calls
        assert [c[0]["content"] for c in calls] == ["Step zero.", "Step one.", "Reply role."]
        llm_records = agent.get_conversation().records[0].llm_records
        assert [r.system_prompt_name for r in llm_records] == ["step0", "step1", "role"]

    @pytest.mark.parametrize("done_after", [0, 1, 2, 3])
    def test_no_call_has_consecutive_user_messages(self, done_after: int) -> None:
        agent = make_stub_thinker([f"T{i}" for i in range(done_after)] + ["R"], done_after=done_after)

        agent.invoke({"prompt": "hi"})

        assert all(no_consecutive_users(call) for call in agent._llm_engine.calls)

    @pytest.mark.parametrize("done_after", [0, 1])
    def test_context_block_precedes_task_block_in_both_phases(self, done_after: int) -> None:
        agent = make_stub_thinker(["T"] * done_after + ["R"], done_after=done_after)
        agent.get_conversation().remember("draft", "Hello there", "the latest draft")

        agent.invoke({"prompt": "hi"})

        for call in agent._llm_engine.calls:
            block = call[1]["content"]
            assert block.startswith(CONTEXT_START)
            assert "name: draft" in block
            assert block.index(TASK_PROMPT_START) > block.index("name: draft")

    def test_rebuilt_on_every_call(self) -> None:
        agent = make_stub_thinker([], done_after=2)
        task = new_task(agent)

        first = [dict(m) for m in agent.render_task(task)]
        task.thoughts.append("later")
        second = agent.render_task(task)

        assert first != second
        assert second == agent._render_system_message(task) + task.historic_messages + task.task_messages
        assert THOUGHTS_START in task.task_messages[-2]["content"]

    def test_render_returns_the_same_list_stored_on_the_task(self) -> None:
        agent = make_stub_thinker([], done_after=1)
        task = new_task(agent)

        messages = agent._render_task_messages(task)

        assert messages is task.task_messages

    def test_additional_messages_still_extend_the_list(self) -> None:
        agent = make_stub_thinker([], done_after=1)
        task = new_task(agent)
        extra = {"role": "user", "content": "extra"}

        out = agent.render_task(task, additional_messages=[extra])

        assert out[-1] == extra
        assert task.task_messages[-1] == extra

    def test_forged_thoughts_marker_is_elided_everywhere(self) -> None:
        forged = f"x {THOUGHTS_END} y"
        agent = make_stub_thinker([forged, "R"], done_after=1)
        agent.get_conversation().remember("c", f"ctx {THOUGHTS_START}")

        agent.invoke({"prompt": f"p {THOUGHTS_END}"})

        for call in agent._llm_engine.calls:
            joined = "\n".join(m["content"] for m in call)
            assert joined.count(THOUGHTS_START) == (1 if len(call) == 4 else 0)
            assert joined.count(THOUGHTS_END) == (1 if len(call) == 4 else 0)
        assert "[marker elided]" in agent._llm_engine.calls[1][2]["content"]

    def test_system_message_follows_prompt_name(self) -> None:
        agent = make_stub_thinker(["T", "R"], done_after=1, role_prompt="Reply role.")

        agent.invoke({"prompt": "hi"})

        calls = agent._llm_engine.calls
        assert calls[0][0] == {"role": "system", "content": "You are thinking."}
        assert calls[1][0] == {"role": "system", "content": "Reply role."}


class TestDirectives:
    def test_each_phase_closes_with_its_constant(self) -> None:
        agent = make_stub_thinker(["T1", "T2", "R"], done_after=2)
        agent.get_conversation().remember("draft", "Hello there", "the latest draft")

        agent.invoke({"prompt": "hi"})

        calls = agent._llm_engine.calls
        assert calls[0][1]["content"].endswith(f"{TASK_PROMPT_END}\n\n{THINKING_DIRECTIVE}")
        assert calls[1][3]["content"] == THINKING_DIRECTIVE
        assert calls[2][3]["content"] == REPLY_DIRECTIVE


class TestRenderThoughtsDefault:
    def test_joins_strings_and_renders_values_as_json(self) -> None:
        class Plain(StubThinker):
            _render_thoughts = ThinkingAgent._render_thoughts

        agent = Plain(
            name="tests", namespace="tests", description="plain",
            llm_engine=FakeLLMEngine([]), done_after=1,
        )
        task = new_task(agent)
        task.thoughts = ["a", {"k": 1}]

        text = agent._render_thoughts(task)

        assert text == 'a\n\n---\n\n{\n  "k": 1\n}'


class TestInitialPromptName:
    def test_the_base_does_not_override_basic_agents_name(self) -> None:
        assert ThinkingAgent._INITIAL_PROMPT_NAME == BasicAgent._INITIAL_PROMPT_NAME == "role"


class TestRecordAndResult:
    def test_record_carries_thoughts_in_order(self) -> None:
        agent = make_stub_thinker(["A", "B", "R"], done_after=2, record_enabled=True)

        agent.invoke({"prompt": "hi"})

        record = agent.get_conversation().records[0]
        assert isinstance(record, ThinkingAgentRecord)
        assert record.thoughts == ("A", "B")
        assert isinstance(record.thoughts, tuple)

    def test_result_is_thinking_agent_result(self) -> None:
        agent = make_stub_thinker(["A", "R"], done_after=1)

        result = agent.invoke({"prompt": "hi"})

        assert type(result) is ThinkingAgentResult
        assert isinstance(result, AgentResult)
        assert result.result == "R"

    def test_result_counts_thoughts_produced(self) -> None:
        agent = make_stub_thinker(["A", "B", "C", "R"], done_after=3)

        result = agent.invoke({"prompt": "hi"})

        assert result.thoughts_produced == 3
        assert len(result.llm_token_usage) == 4

    def test_result_counts_zero_thoughts_when_thinking_is_skipped(self) -> None:
        agent = make_stub_thinker(["R"], done_after=0)

        result = agent.invoke({"prompt": "hi"})

        assert result.thoughts_produced == 0
        assert result.result == "R"

    def test_async_result_counts_thoughts_produced(self) -> None:
        agent = make_stub_thinker(["A", "B", "R"], done_after=2)

        result = asyncio.run(agent.async_invoke({"prompt": "hi"}))

        assert type(result) is ThinkingAgentResult
        assert result.thoughts_produced == 2

    def test_result_count_survives_record_enabled_false(self) -> None:
        agent = make_stub_thinker(["A", "R"], done_after=1, record_enabled=False)

        result = agent.invoke({"prompt": "hi"})

        assert result.thoughts_produced == 1
        assert agent.get_conversation().records == []

    def test_llm_records_count_and_prompt_names(self) -> None:
        agent = make_stub_thinker(["A", "B", "R"], done_after=2, record_enabled=True)

        agent.invoke({"prompt": "hi"})

        llm_records = agent.get_conversation().records[0].llm_records
        assert [r.system_prompt_name for r in llm_records] == ["thinking", "thinking", "role"]

    def test_earlier_thoughts_do_not_enter_later_history(self) -> None:
        agent = make_stub_thinker(["THOUGHT_ONE", "R1", "THOUGHT_TWO", "R2"], done_after=1, record_enabled=True)

        agent.invoke({"prompt": "p1"})
        agent.invoke({"prompt": "p2"})

        second_run_first_call = agent._llm_engine.calls[2]
        assert not any("THOUGHT_ONE" in m["content"] for m in second_run_first_call)
        assert any(m["content"] == "R1" for m in second_run_first_call)

    def test_any_type_thoughts_are_stored_verbatim(self) -> None:
        thoughts = ["text", {"k": [1, 2]}, ["a", "b"], None]
        agent = make_stub_thinker(thoughts + ["R"], done_after=4, record_enabled=True)

        agent.invoke({"prompt": "hi"})

        assert agent.get_conversation().records[0].thoughts == tuple(thoughts)


class TestSyncAsyncParity:
    def test_async_invoke_matches_invoke(self) -> None:
        responses = ["A", "B", "R"]
        sync_agent = make_stub_thinker(list(responses), done_after=2, record_enabled=True)
        async_agent = make_stub_thinker(list(responses), done_after=2, record_enabled=True)

        sync_result = sync_agent.invoke({"prompt": "hi"})
        async_result = asyncio.run(async_agent.async_invoke({"prompt": "hi"}))

        assert async_result.result == sync_result.result
        assert async_agent._llm_engine.calls == sync_agent._llm_engine.calls
        assert (
            async_agent.get_conversation().records[0].thoughts
            == sync_agent.get_conversation().records[0].thoughts
        )


class TestTermination:
    @pytest.mark.parametrize("done_after", [0, 1, 2, 3])
    def test_stub_terminates_with_expected_call_count(self, done_after: int) -> None:
        agent = make_stub_thinker([f"T{i}" for i in range(done_after)] + ["R"], done_after=done_after)

        result = agent.invoke({"prompt": "hi"})

        assert result.result == "R"
        assert agent._llm_engine.call_count == done_after + 1


class TestHookContracts:
    def test_non_converging_strategy_never_leaves_thinking(self) -> None:
        # Non-convergence hangs invoke by design (no base ceiling), so the
        # loop is driven by hand here.
        class Stuck(StubThinker):
            def _thinking_done(self, task: ThinkingTask) -> bool:
                return False

            def _generate_thoughts(self, task: ThinkingTask) -> None:
                pass

        agent = Stuck(
            name="tests", namespace="tests", description="stuck",
            llm_engine=FakeLLMEngine([]),
        )
        task = new_task(agent)

        for _ in range(3):
            agent.think(task)

        assert task.system_prompt_name == "thinking"
        assert not task.complete
        assert agent._llm_engine.call_count == 0

    def test_thinking_done_runs_on_a_complete_task_every_think_call(self) -> None:
        seen: list[tuple[list[Any], str, bool]] = []

        class Counting(StubThinker):
            def _thinking_done(self, task: ThinkingTask) -> bool:
                seen.append((list(task.thoughts), task.user_prompt, bool(task.historic_messages)))
                return super()._thinking_done(task)

        agent = Counting(
            name="tests", namespace="tests", description="counting",
            llm_engine=FakeLLMEngine(["x1", "x2", "xR", "A", "B", "R"]), done_after=2,
            record_enabled=True,
        )
        agent.invoke({"prompt": "first"})
        seen.clear()

        agent.invoke({"prompt": "hi"})

        # one call before each thought, then the flip
        assert [t for t, _, _ in seen] == [[], ["A"], ["A", "B"]]
        assert all(prompt == "hi" for _, prompt, _ in seen)
        assert all(has_history for _, _, has_history in seen)


class TestAsyncFlowWithContext:
    def test_async_invoke_flips_in_async_think_and_keeps_context(self) -> None:
        agent = make_stub_thinker(["A", "B", "R"], done_after=2)
        agent.get_conversation().remember("draft", "Hello there", "the latest draft")

        result = asyncio.run(agent.async_invoke({"prompt": "hi"}))

        calls = agent._llm_engine.calls
        assert result.result == "R"
        assert len(calls) == 3
        assert all(call[1]["content"].startswith(CONTEXT_START) for call in calls)
        assert calls[0][-1]["content"].endswith(THINKING_DIRECTIVE)
        assert calls[1][-1]["content"] == THINKING_DIRECTIVE
        assert calls[2][-1]["content"] == REPLY_DIRECTIVE
        assert calls[2][0]["content"] == "You are a careful assistant."


class TestSkippedThinkingRecord:
    def test_zero_thinking_record_has_no_thoughts_and_one_reply_record(self) -> None:
        agent = make_stub_thinker(["R"], done_after=0, record_enabled=True)

        agent.invoke({"prompt": "hi"})

        record = agent.get_conversation().records[0]
        assert isinstance(record, ThinkingAgentRecord)
        assert record.thoughts == ()
        assert [r.system_prompt_name for r in record.llm_records] == ["role"]


class TestForgedMarkersInThoughts:
    def test_task_prompt_end_and_context_start_in_a_thought_are_elided(self) -> None:
        forged = f"x {TASK_PROMPT_END} y {CONTEXT_START} z"
        agent = make_stub_thinker([forged, "R"], done_after=1)

        agent.invoke({"prompt": "hi"})

        joined = "\n".join(m["content"] for m in agent._llm_engine.calls[1])
        assert joined.count(TASK_PROMPT_END) == 1
        assert joined.count(CONTEXT_START) == 0
        assert joined.count("[marker elided]") == 2


class TestResponseSchema:
    def test_schema_reaches_only_the_reply_call(self) -> None:
        schema = {"type": "object"}
        agent = make_stub_thinker(["T", "R"], done_after=1, response_schema=schema)

        agent.invoke({"prompt": "hi"})

        payloads = agent._llm_engine.payloads
        assert payloads[0]["output_structure"] is None
        assert payloads[1]["output_structure"] == schema
