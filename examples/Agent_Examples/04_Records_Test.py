from __future__ import annotations

import os
import pprint
from typing import Any
from dotenv import load_dotenv

from atomic_agentic.agents import Agent
from atomic_agentic.engines.LLMEngines import OpenAIEngine

load_dotenv()  # Load environment variables from .env file

def summarize_result(*, result: str) -> dict[str, Any]:
    """
    Non-default post-invoke hook.

    The Agent stores both:
    - raw_response: direct LLM text
    - final_response: this structured dict
    """
    return {
        "text": result.strip(),
        "char_count": len(result),
        "post_processed": True,
    }


def main() -> None:
    engine = OpenAIEngine("gpt-4o-mini")

    agent = Agent(
        name="basic_turn_demo_agent",
        namespace="examples",
        description="Basic Agent demo showing raw/final turn rendering.",
        llm_engine=engine,
        role_prompt="You are concise. Answer in one short sentence.",
        context_enabled=True,
        records_window=None,
        post_invoke=summarize_result,
        response_preview_limit=300,
        assistant_response_source="raw",
    )

    result = agent.invoke(
        {
            "prompt": "Give me a one-sentence definition of agent memory."
        }
    )

    print("\n=== Final invoke() result after post_invoke ===")
    pprint.pp(result.result)

    print("\n=== Canonical turn history: agent.records ===")
    for i, turn in enumerate(agent.records):
        print(f"\nTurn {i}")
        print("prompt:")
        pprint.pp(turn.user_prompt)
        print("raw_response:")
        pprint.pp(turn.generated_response)
        print("final_response:")
        pprint.pp(turn.final_result)

    print("\n=== Rendered turn using raw assistant response ===")
    agent.assistant_response_source = "raw"
    pprint.pp(agent.render_turn(agent.records[0]))

    print("\n=== Rendered turn using final assistant response ===")
    agent.assistant_response_source = "final"
    pprint.pp(agent.render_turn(agent.records[0]))

    print("\n=== Messages that would be sent on the next invoke ===")
    next_messages = agent.build_messages(
        system_prompt = agent.role_prompt,
        turns = agent.records,
        prompt = "Now explain why turn-native memory is useful."
    )
    pprint.pp(next_messages)


if __name__ == "__main__":
    main()
