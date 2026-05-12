# 05_Passthrough_Test.py

from __future__ import annotations

import os
import pprint
from typing import Any
from dotenv import load_dotenv

from atomic_agentic.agents import Agent
from atomic_agentic.engines.LLMEngines import OpenAIEngine


load_dotenv()


def build_prompt(
    *,
    prompt: str,
    audience: str = "general reader",
    output_label: str = "answer",
    include_metadata: bool = True,
) -> str:
    """
    pre_invoke hook.

    Only `prompt` and `audience` are used to build the LLM-facing prompt.
    `output_label` and `include_metadata` are declared here so they are part of
    the Agent input schema and can be passed through to post_invoke.
    """
    return (
        f"Answer the following for a {audience}.\n\n"
        f"Question:\n{prompt}\n\n"
        "Keep the answer concise."
    )


def package_response(
    *,
    raw_answer: str,
    output_label: str = "answer",
    include_metadata: bool = True,
) -> dict[str, Any]:
    """
    post_invoke hook.

    Receives:
    - raw_answer from the Agent's raw LLM result via post_result_key
    - output_label from passthrough_inputs
    - include_metadata from passthrough_inputs
    """
    payload: dict[str, Any] = {
        output_label: raw_answer.strip(),
    }

    if include_metadata:
        payload["metadata"] = {
            "label": output_label,
            "char_count": len(raw_answer),
            "post_processed": True,
        }

    return payload


def main() -> None:
    llm = OpenAIEngine(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-5-mini")

    agent = Agent(
        name="PassthroughDemoAgent",
        description="Demonstrates post-invoke passthrough inputs.",
        llm_engine=llm,
        role_prompt="You are a concise assistant.",
        context_enabled=False,
        pre_invoke=build_prompt,
        post_invoke=package_response,
        post_result_key="raw_answer",
        passthrough_inputs=["output_label", "include_metadata"],
    )

    result = agent.invoke(
        {
            "prompt": "What is agent memory?",
            "audience": "new Python developer",
            "output_label": "definition",
            "include_metadata": True,
        }
    )

    print("\n=== Post-Processed Result ===\n")
    pprint.pp(result)


if __name__ == "__main__":
    main()