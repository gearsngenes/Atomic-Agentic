# 07_Prompt_Parameters.py
"""
Demonstrates role_prompt placeholders as agent schema parameters.

When a BasicAgent's role_prompt contains {placeholder} variables, those
names are auto-discovered and become KEYWORD_ONLY parameters in the agent's
schema.  Callers supply them alongside the task prompt at invocation time;
the framework renders the role prompt before each LLM call.

This example runs the exact same task prompt through two different persona +
rules combinations to show how the rendered role prompt shapes the response.
"""
from __future__ import annotations

import os
from dotenv import load_dotenv

from atomic_agentic.agents import BasicAgent
from atomic_agentic.engines.LLMEngines import OpenAIEngine

load_dotenv()


ROLE_PROMPT = (
    "You are {persona}.\n\n"
    "Rules you must follow:\n"
    "{rules}"
)

TASK = "What is a variable in programming?"

PERSONAS = [
    {
        "persona": "a sarcastic senior developer",
        "rules": (
            "Be dismissive of the question. Use heavy technical jargon. "
            "End your answer with a complaint about the state of CS education."
        ),
    },
    {
        "persona": "a patient teacher explaining things to a 10-year-old",
        "rules": (
            "Use one simple real-world analogy. Avoid all jargon. "
            "Be warm and encouraging. Keep the answer to two or three sentences."
        ),
    },
    {
        "persona": "a whimsical Dr. Seuss character",
        "rules": (
            "Respond entirely in rhyming couplets in the style of Dr. Seuss. "
            "Invent playful made-up words where they fit the rhythm. "
            "Keep it to four to six lines."
        ),
    },
]


def main() -> None:
    engine = OpenAIEngine(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")

    agent = BasicAgent(
        name="prompt_params_demo",
        namespace="examples",
        description="Shows role_prompt placeholders as KEYWORD_ONLY schema parameters.",
        llm_engine=engine,
        role_prompt=ROLE_PROMPT,
        context_enabled=False,
    )

    # The placeholders {persona} and {rules} appear as KEYWORD_ONLY params.
    print("\n=== Agent schema ===\n")
    for param in agent.parameters:
        default = "(required)" if param.default.__class__.__name__ == "_NO_VAL" else repr(param.default)
        print(f"  {param.name:<20} {param.kind:<25} default={default}")

    print(f"\n=== Task prompt (identical for every invocation) ===\n  {TASK!r}\n")

    for i, ctx in enumerate(PERSONAS, 1):
        print(f"{'─' * 64}")
        print(f"  Invocation {i}")
        print(f"  persona : {ctx['persona']}")
        print(f"  rules   : {ctx['rules']}")
        print(f"{'─' * 64}\n")

        result = agent.invoke({"prompt": TASK, **ctx})

        print(f"  {result.result}\n")


if __name__ == "__main__":
    main()
