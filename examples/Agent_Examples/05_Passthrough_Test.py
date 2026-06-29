# 05_Auto_Graft_Test.py
"""
Demonstrates post-invoke parameter auto-grafting (v2.0.0a14+).

Before a14, post_invoke parameters beyond the result key had to be listed
explicitly in ``passthrough_inputs``.  That API is gone.

Now, every non-result, non-variadic post_invoke parameter is automatically
wired into the agent's schema as a KEYWORD_ONLY input.  Callers pass them
alongside the normal user inputs; the agent routes them to post_invoke without
any extra configuration.

Schema produced by this agent:
  prompt            (POSITIONAL_OR_KEYWORD, required)   ← identity pre_invoke
  audience          (POSITIONAL_OR_KEYWORD, default "general reader")  ← pre_invoke
  output_label      (KEYWORD_ONLY, default "answer")    ← auto-grafted from post_invoke
  include_metadata  (KEYWORD_ONLY, default True)        ← auto-grafted from post_invoke
  continue_from     (KEYWORD_ONLY, default None)        ← framework-reserved
"""

from __future__ import annotations

import os
import pprint
from typing import Any
from dotenv import load_dotenv

from atomic_agentic.agents import BasicAgent
from atomic_agentic.engines.LLMEngines import OpenAIEngine


load_dotenv()


def build_prompt(
    *,
    prompt: str,
    audience: str = "general reader",
) -> str:
    """pre_invoke: tailor the LLM-facing prompt for the intended audience."""
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
    """post_invoke: structure the raw LLM result.

    ``raw_answer`` receives the agent's raw generated text (the result key).
    ``output_label`` and ``include_metadata`` are auto-grafted into the agent
    schema — callers supply them as regular inputs, no extra configuration needed.
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
    llm = OpenAIEngine(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")

    agent = BasicAgent(
        name="AutoGraftDemoAgent",
        namespace="examples",
        description="Demonstrates post-invoke parameter auto-grafting.",
        llm_engine=llm,
        role_prompt="You are a concise assistant.",
        context_enabled=False,
        pre_invoke=build_prompt,
        post_invoke=package_response,
        post_result_key="raw_answer",
        # No passthrough_inputs needed — output_label and include_metadata
        # are auto-grafted from package_response into the agent schema.
    )

    # Show the composed schema so the auto-grafting is visible.
    print("\n=== Agent schema (auto-grafted parameters shown) ===\n")
    for param in agent.parameters:
        default = "(required)" if param.default.__class__.__name__ == "_NO_VAL" else repr(param.default)
        print(f"  {param.name:<20} {param.kind:<25} default={default}")

    # Invoke with a custom output_label — passed as a regular input.
    result = agent.invoke(
        {
            "prompt": "What is agent memory?",
            "audience": "new Python developer",
            "output_label": "definition",
            "include_metadata": True,
        }
    )

    print("\n=== Post-Processed Result ===\n")
    pprint.pp(result.result)

    # Invoke again relying on the defaults (no output_label or include_metadata supplied).
    result_default = agent.invoke(
        {
            "prompt": "What is a tool in an agentic system?",
        }
    )

    print("\n=== Result with default output_label and include_metadata ===\n")
    pprint.pp(result_default.result)


if __name__ == "__main__":
    main()
