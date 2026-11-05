from __future__ import annotations

import asyncio
import pytest

from atomic_agentic.agents.basic import BasicAgent
from atomic_agentic.agents.thinking import ThinkingAgent
from atomic_agentic.constants.agents import (
    CONTEXT_START,
    TASK_PROMPT_END,
    TASK_PROMPT_START,
    THOUGHTS_END,
    THOUGHTS_START,
)
from atomic_agentic.models.agents.records import ThinkingAgentRecord
from atomic_agentic.models.agents.tasks import ThinkingTask
from atomic_agentic.models.results.agents import AgentResult, ThinkingAgentResult

from .conftest import FakeLLMEngine, StubThinker, make_stub_thinker

HOOKS = ("_thinking_done", "_generate_thoughts", "_agenerate_thoughts", "_render_thoughts")
THINK_DIRECTIVE = "Continue thinking as preparation for the final reply, not the reply itself."
REPLY_DIRECTIVE = (
    "Given the above RUNTIME CONTEXT (if present) and THOUGHTS (if present), "
    "complete the USER TASK PROMPT."
)


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
class TestAbstractness:
    def test_base_cannot_be_instantiated(self) -> None:
        with pytest.raises(TypeError):
            ThinkingAgent()  # type: ignore[abstract]

    @pytest.mark.parametrize("missing", HOOKS)
    def test_subclass_missing_a_hook_cannot_be_instantiated(self, missing: str) -> None:
        impls = {
            "_thinking_done": lambda self, task: True,
            "_generate_thoughts": lambda self, task: None,
            "_agenerate_thoughts": lambda self, task: None,
            "_render_thoughts": lambda self, task: "",
        }
        partial = type("Partial", (ThinkingAgent,), {k: v for k, v in impls.items() if k != missing})

        with pytest.raises(TypeError):
            partial()  # type: ignore[abstract]

    def test_stub_can_be_instantiated(self) -> None:
        assert isinstance(make_stub_thinker([]), ThinkingAgent)

    def test_abstract_methods_are_exactly_the_four_hooks(self) -> None:
        assert ThinkingAgent.__abstractmethods__ == frozenset(HOOKS)

    def test_task_class(self) -> None:
        assert ThinkingAgent._TASK_CLS is ThinkingTask

    def test_constructor_is_basic_agents(self) -> None:
        assert ThinkingAgent.__init__ is BasicAgent.__init__


class TestSeed:
    def test_done_after_zero_starts_in_reply_phase(self) -> None:
        agent = make_stub_thinker([], done_after=0)

        assert agent.create_task(agent.filter_inputs({"prompt": "hi"})).system_prompt_name == "role"

    @pytest.mark.parametrize("done_after", [1, 2])
    def test_otherwise_starts_in_thinking_phase(self, done_after: int) -> None:
        agent = make_stub_thinker([], done_after=done_after)

        assert agent.create_task(agent.filter_inputs({"prompt": "hi"})).system_prompt_name == "thinking"

    def test_thinking_done_is_safe_on_empty_task(self) -> None:
        agent = make_stub_thinker([], done_after=1)
        task = agent.create_task(agent.filter_inputs({"prompt": "hi"}))

        assert task.thoughts == []
        assert agent._thinking_done(task) is False

    def test_subclass_seed_before_super_is_seen_by_the_flip_check(self) -> None:
        class Seeded(StubThinker):
            def _seed_task(self, task: ThinkingTask) -> None:
                task.thoughts.append("seeded")
                super()._seed_task(task)

        agent = Seeded(
            name="tests", namespace="tests", description="seeded",
            llm_engine=FakeLLMEngine([]),
            done_after=1,
        )
        task = agent.create_task(agent.filter_inputs({"prompt": "hi"}))

        assert task.thoughts == ["seeded"]
        assert task.system_prompt_name == "role"


class TestPhaseGating:
    def test_think_in_reply_phase_is_a_noop(self) -> None:
        agent = make_stub_thinker([], done_after=0)
        task = new_task(agent)

        agent.think(task)

        assert agent._llm_engine.call_count == 0
        assert task.thoughts == []
        assert task.system_prompt_name == "role"
        assert not task.complete

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

        assert reply_agent._llm_engine.call_count == 0
        assert think_agent._llm_engine.call_count == 0
        assert not think_task.complete

    @pytest.mark.parametrize("done_after", [1, 2])
    def test_pass_that_makes_last_thought_also_replies(self, done_after: int) -> None:
        agent = make_stub_thinker([f"T{i}" for i in range(done_after)] + ["REPLY"], done_after=done_after)
        task = new_task(agent)

        for _ in range(done_after - 1):
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

    def test_flip_is_set_by_the_base(self) -> None:
        agent = make_stub_thinker(["T0"], done_after=1)
        task = new_task(agent)

        agent.think(task)

        # the stub's hooks never assign the prompt name
        assert task.thoughts == ["T0"]
        assert task.system_prompt_name == "role"


class TestRendering:
    def test_round_one_is_one_merged_user_message(self) -> None:
        agent = make_stub_thinker(["T", "R"], done_after=1)

        agent.invoke({"prompt": "hi"})

        first = agent._llm_engine.calls[0]
        assert [m["role"] for m in first] == ["system", "user"]
        assert first[1]["content"] == f"{wrap('hi')}\n\n{THINK_DIRECTIVE}"

    def test_round_two_is_block_thoughts_directive(self) -> None:
        agent = make_stub_thinker(["T1", "T2", "R"], done_after=2)

        agent.invoke({"prompt": "hi"})

        second = agent._llm_engine.calls[1]
        assert [m["role"] for m in second] == ["system", "user", "assistant", "user"]
        assert second[1]["content"] == wrap("hi")
        assert second[2]["content"] == f"{THOUGHTS_START}\nT1\n{THOUGHTS_END}"
        assert second[3]["content"] == THINK_DIRECTIVE

    def test_reply_with_thoughts_has_three_message_shape(self) -> None:
        agent = make_stub_thinker(["T1", "T2", "R"], done_after=2)

        agent.invoke({"prompt": "hi"})

        reply = agent._llm_engine.calls[2]
        assert [m["role"] for m in reply] == ["system", "user", "assistant", "user"]
        assert reply[2]["content"] == f"{THOUGHTS_START}\nT1\nT2\n{THOUGHTS_END}"
        assert reply[3]["content"] == REPLY_DIRECTIVE

    def test_reply_with_zero_thoughts_is_one_merged_message(self) -> None:
        agent = make_stub_thinker(["R"], done_after=0)

        agent.invoke({"prompt": "hi"})

        call = agent._llm_engine.calls[0]
        assert [m["role"] for m in call] == ["system", "user"]
        assert call[1]["content"] == f"{wrap('hi')}\n\n{REPLY_DIRECTIVE}"
        assert THOUGHTS_START not in call[1]["content"]

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
        assert second[1:] == task.task_messages
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

    def test_render_directive_override_is_honoured(self) -> None:
        class Custom(StubThinker):
            def _render_directive(self, task: ThinkingTask) -> str:
                return f"step {len(task.thoughts)}"

        agent = Custom(
            name="tests", namespace="tests", description="custom",
            llm_engine=FakeLLMEngine(["T1", "T2", "R"]), done_after=2,
        )

        agent.invoke({"prompt": "hi"})

        directives = [call[-1]["content"] for call in agent._llm_engine.calls]
        assert directives[0].endswith("step 0")
        assert directives[1:] == ["step 1", "step 2"]

    def test_system_message_follows_prompt_name(self) -> None:
        agent = make_stub_thinker(["T", "R"], done_after=1, role_prompt="Reply role.")

        agent.invoke({"prompt": "hi"})

        calls = agent._llm_engine.calls
        assert calls[0][0] == {"role": "system", "content": "You are thinking."}
        assert calls[1][0] == {"role": "system", "content": "Reply role."}


class TestRecordAndResult:
    def test_record_carries_thoughts_in_order(self) -> None:
        agent = make_stub_thinker(["A", "B", "R"], done_after=2, record_enabled=True)

        agent.invoke({"prompt": "hi"})

        record = agent.get_conversation().records[0]
        assert isinstance(record, ThinkingAgentRecord)
        assert record.thoughts == ("A", "B")
        assert isinstance(record.thoughts, tuple)

    def test_result_is_plain_agent_result(self) -> None:
        agent = make_stub_thinker(["A", "R"], done_after=1)

        result = agent.invoke({"prompt": "hi"})

        assert type(result) is AgentResult
        assert not isinstance(result, ThinkingAgentResult)
        assert result.result == "R"

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
