# 07_Prompt_Parameters.py
"""
Controlled 3×3 grid: three role prompts × three languages = nine responses.

Two independent axes:

  SYSTEM PROMPT AXIS — three role configs swapped via ``agent.role_prompt``:
    Academic · Casual · Pirate
    Each is a plain PromptConfig (no placeholders), so the role prompt is
    static per invocation but mutable between them.

  CONTEXT AXIS — three languages injected through ``context["language"]``:
    English · German · Spanish
    ``language`` is declared as an ``extra_context_property`` on the agent.
    The pre_invoke returns a PromptConfig whose ``{language}`` placeholder
    is auto-filled from ``context`` by the framework at step ⑥, before the
    message is sent to the LLM.

The post_invoke uses a ``{}``-style positional format string to wrap the
raw LLM response in a labelled block.
"""
from __future__ import annotations

import os
from dotenv import load_dotenv

from atomic_agentic.agents import BasicAgent
from atomic_agentic.engines.LLMEngines import OpenAIEngine
from atomic_agentic.constants.core import NO_VAL
from atomic_agentic.models.parameters import ParamSpec
from atomic_agentic.models.agents.prompts import PromptConfig

load_dotenv()


# ── System-prompt axis ────────────────────────────────────────────────────────

ROLE_CONFIGS: list[tuple[str, PromptConfig]] = [
    (
        "Academic",
        PromptConfig(
            template=(
                "You are a formal professor of earth sciences. "
                "Be precise, technical, and educational."
            ),
            description="Academic professor persona",
        ),
    ),
    (
        "Casual",
        PromptConfig(
            template=(
                "You are a stoic robot, mechanical and programatic. "
                "Your responses always resemble pseudo-code."
            ),
            description="Casual friend persona",
        ),
    ),
    (
        "Pirate",
        PromptConfig(
            template=(
                "You are a salty old pirate captain. "
                "Explain everything with nautical flair and pirate speech."
            ),
            description="Pirate persona",
        ),
    ),
]

# ── Context axis ──────────────────────────────────────────────────────────────

LANGUAGES = ["English", "German", "Spanish"]

TASK = "What is the ocean?"

# ── Pre / post invoke ─────────────────────────────────────────────────────────

def build_task_prompt(prompt: str) -> PromptConfig:
    """Wrap the task in a PromptConfig whose {language} placeholder is filled
    from ``context`` by the framework at step ⑥.

    ``prompt`` is embedded directly via f-string so only ``{language}``
    remains as a live placeholder for context rendering.
    """
    return PromptConfig(
        template=(
            f"Question: {prompt}\n\n"
            "Answer entirely in {language}."
        ),
        description="Task prompt with context-rendered language directive",
    )


_RESULT_WRAPPER = "[Response]\n\n{}\n\n[End]"


def wrap_response(raw: str) -> str:
    """Wrap the LLM response using a positional {} format field."""
    return _RESULT_WRAPPER.format(raw)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    engine = OpenAIEngine(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")

    # Start with the first role config; the loop swaps it via the setter.
    agent = BasicAgent(
        name="language_persona_grid",
        namespace="examples",
        description=(
            "Runs a 3×3 grid of role × language combinations against a single task. "
            "Demonstrates extra_context_properties, PromptConfig pre-invoke, "
            "and mutable role_prompt."
        ),
        llm_engine=engine,
        role_prompt=ROLE_CONFIGS[0][1],
        extra_context_properties=[
            ParamSpec(
                name="language",
                index=0,
                kind=ParamSpec.KEYWORD_ONLY,
                type="str",
                default=NO_VAL,
                description="Language the agent must respond in.",
            ),
        ],
        pre_invoke=build_task_prompt,
        post_invoke=wrap_response,
        context_enabled=False,
    )

    print("\n=== Agent schema ===\n")
    for param in agent.parameters:
        default_str = "(required)" if param.default is NO_VAL else repr(param.default)
        print(f"  {param.name:<20} {param.kind:<25} default={default_str}")
        if param.description:
            preview = param.description[:100] + ("…" if len(param.description) > 100 else "")
            print(f"  {'':20} {preview}")

    print(f"\n=== Task ===\n  {TASK!r}\n")

    for role_label, role_config in ROLE_CONFIGS:
        agent.role_prompt = role_config          # swap persona via B2 setter

        print(f"\n{'═' * 64}")
        print(f"  ROLE: {role_label}")
        print(f"{'═' * 64}")

        for language in LANGUAGES:
            print(f"\n  ── {language} ──\n")

            result = agent(
                prompt=TASK,
                context={
                    "language": language
                }
            )

            print(result.result)


if __name__ == "__main__":
    main()
