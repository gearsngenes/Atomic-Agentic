from __future__ import annotations

import asyncio
from typing import Any

import pytest

from atomic_agentic.agents.basic import BasicAgent
from atomic_agentic.core.prompt_template import PromptTemplate
from atomic_agentic.tools import Tool
from ..fake_engines import FakeLLMEngine, echo_latest_user

TASK_PROMPT_START = "==== USER TASK PROMPT [START] ===="
TASK_PROMPT_END = "==== USER TASK PROMPT [END] ===="


def wrap(prompt: str) -> str:
    """The user message BasicAgent sends for ``prompt`` with empty context."""
    return f"{TASK_PROMPT_START}\n{prompt}\n{TASK_PROMPT_END}"


def make_agent(
    name: str,
    reply: str | None = None,
    **kwargs: Any,
) -> BasicAgent:
    """A BasicAgent over a fake engine: a fixed ``reply``, else an echo."""
    engine = (
        FakeLLMEngine(response_fn=lambda messages: reply)
        if reply is not None
        else FakeLLMEngine(response_fn=echo_latest_user())
    )
    return BasicAgent(
        name=name,
        description=f"{name} under test.",
        llm_engine=engine,
        record_enabled=False,
        **kwargs,
    )


def shout(result: str) -> str:
    return result.upper()


def build_chain() -> tuple[BasicAgent, BasicAgent, BasicAgent, BasicAgent]:
    """An outer agent whose instructions and pre stage are agents and whose
    post stage is a Tool: (outer, instructions_agent, pre_agent, post_tool)."""
    instructions_agent = make_agent("inst_agent", "You are a pirate.")
    pre_agent = make_agent("pre_agent", "arr, tell me about maps")
    post_tool = Tool(shout, name="shout", description="Uppercase the reply.")
    outer = make_agent(
        "outer",
        instructions=instructions_agent,
        pre_invoke=pre_agent,
        post_invoke=post_tool,
        response_to_post="result",
    )
    return outer, instructions_agent, pre_agent, post_tool


class TestInvokablesInEveryPromptPosition:
    def test_agent_in_instructions_pre_and_post_in_one_invoke(self) -> None:
        outer, instructions_agent, pre_agent, _ = build_chain()
        counts = {"inst": 0, "pre": 0}
        for key, inner in (("inst", instructions_agent), ("pre", pre_agent)):
            real = inner.invoke

            def counted(*args: Any, _real: Any = real, _key: str = key, **kwargs: Any) -> Any:
                counts[_key] += 1
                return _real(*args, **kwargs)

            inner.invoke = counted  # type: ignore[method-assign]

        result = outer.invoke({"prompt": "hello"})

        engine_calls = outer._llm_engine.calls
        assert len(engine_calls) == 1
        assert engine_calls[0][0] == {"role": "system", "content": "You are a pirate."}
        assert engine_calls[0][-1]["content"] == wrap("arr, tell me about maps")
        assert result.result == f"ECHO: {wrap('arr, tell me about maps')}".upper()
        assert counts == {"inst": 1, "pre": 1}

    def test_invokable_parameters_join_the_signature_in_order(self) -> None:
        template = PromptTemplate(
            template="You are a {persona}.", name="persona_t", description="d",
        )
        agent = make_agent("outer", instructions=template, pre_invoke="{topic}: {prompt}")

        assert [p.name for p in agent.parameters] == ["topic", "prompt", "persona"]

        agent.invoke({"topic": "maps", "prompt": "where", "persona": "pirate"})

        calls = agent._llm_engine.calls
        assert calls[0][0] == {"role": "system", "content": "You are a pirate."}
        assert calls[0][-1]["content"] == wrap("maps: where")

    def test_async_chain_never_calls_sync_invoke(self, monkeypatch: pytest.MonkeyPatch) -> None:
        outer, instructions_agent, pre_agent, _ = build_chain()

        def boom(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("sync invoke must not be called on an async path")

        monkeypatch.setattr(instructions_agent, "invoke", boom)
        monkeypatch.setattr(pre_agent, "invoke", boom)

        result = asyncio.run(outer.async_invoke({"prompt": "hello"}))

        calls = outer._llm_engine.calls
        assert calls[0][0] == {"role": "system", "content": "You are a pirate."}
        assert result.result == f"ECHO: {wrap('arr, tell me about maps')}".upper()
