# 07_Prompt_Parameters.py
"""
Isolates role_prompt as the single variable under test: three BasicAgent
instances share one PromptConfig template shape — a fixed behavioral base
clause plus one required {domain} placeholder — but each has a distinct
literal style clause baked into its role_prompt at construction time. All
three are invoked once with byte-identical inputs (same task prompt, same
domain value); any difference in output is attributable only to role_prompt.

role_prompt is fixed at construction with no mutator (matches every other
AtomicInvokable's fixed-topology/immutability invariant) — so "swap the
persona" now means "construct a different agent," not "mutate this one."

There is no run_id="new" (removed in v2.0.0a14): three independent fresh
conversations are three separate agent instances, each invoked exactly once
— not three calls against one shared-history agent.
"""
from __future__ import annotations

import os
import pprint
from dotenv import load_dotenv

from atomic_agentic.agents import BasicAgent
from atomic_agentic.constants.core import NO_VAL
from atomic_agentic.llm import OpenAIEngine
from atomic_agentic.models.agents.prompts import PromptConfig

load_dotenv()


# ── Shared base clause ───────────────────────────────────────────────────
# Plain str constant — {domain} is literal text here, not evaluated until
# PromptConfig.render() formats the composed template later.
BASE_CLAUSE = (
    "You are a knowledgeable tutor. A student has asked you to explain a "
    "topic in the field of {domain}. Regardless of your explanation style, "
    "always keep your answer to two short paragraphs."
)

# ── Style axis — the one thing that differs between the three agents ────
STYLE_AUGMENTATIONS = [
    (
        "ELI5",
        "eli5",
        "Explain using only simple, everyday analogies a five-year-old "
        "could follow. Never use technical jargon.",
    ),
    (
        "Rigorous",
        "rigorous",
        "Explain with precise technical terminology and formal "
        "definitions, assuming graduate-level background knowledge.",
    ),
    (
        "Sports Analogy",
        "sports_analogy",
        "Explain every part of your answer through a sports or athletics "
        "analogy, however much of a stretch that requires. Keep it playful.",
    ),
]

DOMAIN = "how vaccines train the immune system"
TASK = "Please explain this to me."


def build_persona_prompt(augmentation: str) -> PromptConfig:
    """Compose one persona's role_prompt: shared base clause + its style clause.

    {domain} is the only live placeholder in the resulting template — the
    f-string substitutes BASE_CLAUSE/augmentation by name, not their
    contents, so {domain} passes through untouched for PromptConfig's own
    discovery/rendering.
    """
    template = f"{BASE_CLAUSE}\n\n{augmentation}"
    return PromptConfig(
        template=template,
        description="Tutor persona prompt with a required domain slot.",
        field_specs={
            "domain": {
                "type": "str",
                "description": "The subject area the student wants explained.",
            },
        },
    )


def main() -> None:
    engine = OpenAIEngine(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")

    agents: list[tuple[str, BasicAgent]] = []
    for label, suffix, augmentation in STYLE_AUGMENTATIONS:
        role_config = build_persona_prompt(augmentation)
        agent = BasicAgent(
            name=f"tutor_{suffix}",
            namespace="examples",
            description=f"{label}-style tutor persona demo agent.",
            llm_engine=engine,
            role_prompt=role_config,
            context_enabled=False,
        )
        agents.append((label, agent))

    # All three agents share the same parameter shape (base clause + style
    # differ only in role_prompt text, not in declared schema) — print one
    # to show the auto-grafted, required `domain` parameter.
    print("\n=== Agent schema (shared across all three personas) ===\n")
    for param in agents[0][1].parameters:
        default_str = "(required)" if param.default is NO_VAL else repr(param.default)
        print(f"  {param.name:<20} {param.kind:<25} default={default_str}")
        if param.description:
            print(f"  {'':20} {param.description}")

    print(f"\n=== Domain (identical for every call) ===\n  {DOMAIN!r}")
    print(f"\n=== Task prompt (identical for every call) ===\n  {TASK!r}\n")

    for label, agent in agents:
        print(f"\n{'=' * 64}")
        print(f"  PERSONA: {label}")
        print(f"{'=' * 64}\n")

        result = agent(prompt=TASK, domain=DOMAIN)
        pprint.pp(result.result)


if __name__ == "__main__":
    main()
