"""
planact_pydanticai_demo.py

Demonstrates interoperability between Atomic-Agentic's PlanActAgent and PydanticAI.

Pattern:
- Atomic PlanActAgent does one-shot planning + tool execution.
- Tools are simple Python functions registered to the PlanActAgent.
- Each tool wraps a PydanticAI agent call:
      agent.run_sync(input=..., deps=DepsModel(**kwargs))
- Tools include rich docstrings so PlanAct's planner sees what they do and what args they accept.

Matches the invocation style shown in 01_plugins_test.py:
  agent.invoke({"prompt": task_prompt})
"""

from __future__ import annotations

from dotenv import load_dotenv
import logging

from pydantic import BaseModel, Field
from pydantic_ai import Agent as PydanticAgent

from atomic_agentic.agents import PlanActAgent
from atomic_agentic.tools.Plugins import CONSOLE_TOOLS
from atomic_agentic.engines.LLMEngines import OpenAIEngine

load_dotenv()

# ──────────────────────────  Pydantic deps models  ──────────────────────────

class ClassifyDeps(BaseModel):
    """Runtime context provided to the classification PydanticAI agent."""
    user_id: int = Field(..., description="ID of the requesting user")
    domain: str = Field("general", description="Optional domain hint (e.g. 'finance', 'support', 'legal')")


class SummarizeDeps(BaseModel):
    """Runtime context provided to the summarization PydanticAI agent."""
    user_id: int = Field(..., description="ID of the requesting user")
    max_words: int = Field(40, description="Maximum words in the summary")
    style: str = Field("plain", description="Summary style hint (e.g. 'plain', 'bullet', 'executive')")


# ──────────────────────────  PydanticAI agents  ──────────────────────────

# These are pure PydanticAI agents; they are NOT Atomic-Agentic engines.
# They will call their configured provider/model independently.
CLASSIFIER = PydanticAgent(
    "openai:gpt-4o-mini",
    deps_type=ClassifyDeps,
    instructions=(
        "You are a semantic classifier. Return ONLY a short category label.\n"
        "Examples: finance, support, engineering, scheduling, legal, general.\n"
        "Do not include punctuation or extra text."
    ),
)

SUMMARIZER = PydanticAgent(
    "openai:gpt-4o-mini",
    deps_type=SummarizeDeps,
    instructions=(
        "You are a concise summarizer. Follow the requested max_words and style.\n"
        "Return ONLY the summary text."
    ),
)


# ──────────────────────────  Wrapper tools  ──────────────────────────

def classify_text(*, text: str, user_id: int, domain: str = "general") -> str:
    """
    Specialty: semantic classification into a short category label.
      - Returns a single category label as a string.
    """
    deps = ClassifyDeps(user_id=user_id, domain=domain)
    result = CLASSIFIER.run_sync(text, deps=deps)
    return result.output.strip()


def summarize_text(*, text: str, user_id: int, max_words: int = 40, style: str = "plain") -> str:
    """
    Specialty: concise summarization with a controllable length + style.
      - Returns the summary as a string.
    """
    deps = SummarizeDeps(user_id=user_id, max_words=max_words, style=style)
    result = SUMMARIZER.run_sync(text, deps=deps)
    return result.output.strip()


# ──────────────────────────  Main demo  ──────────────────────────

def main() -> None:
    logging.basicConfig(level=logging.INFO)

    print("\n───────────────────────────────\n")
    print("Atomic-Agentic PlanActAgent + PydanticAI tools demo")

    # Atomic planning engine (used ONLY for the PlanAct planner call)
    llm_engine = OpenAIEngine(model="gpt-4o-mini")

    agent = PlanActAgent(
        name="Hybrid_PlanAct",
        description="One-shot planner/executor that can call PydanticAI-backed wrapper tools.",
        llm_engine=llm_engine,
        context_enabled=False,
        run_concurrent=False,
    )

    # Register console tools so the agent can print labeled results (like your plugins example).
    agent.batch_register(CONSOLE_TOOLS)

    # Register our two PydanticAI wrappers as tools.
    # ToolAgent.register() will toolify() these callables, using __name__ and __doc__.
    agent.register(classify_text)
    agent.register(summarize_text)

    task_prompt = """
You are a one-shot planner/executor.

TASK:
1) Use the classify_text tool to classify the TEXT below.
2) Use the summarize_text tool to summarize the TEXT below in <= 25 words, style="executive".
3) Print BOTH results, each preceded by a label:
   - "CATEGORY:"
   - "SUMMARY:"
4) Return None as the final result.

IMPORTANT:
- Use user_id=101 for BOTH tool calls.
- Use domain="finance" for classify_text.

TEXT:
"The Q2 revenue declined due to increased cloud costs, but customer retention improved significantly and churn dropped."
"""

    print("\n⇢ Executing hybrid demo …")
    agent.invoke({"prompt": task_prompt})

    print("\n───────────────────────────────\n")
    print("Done.")


if __name__ == "__main__":
    main()
