"""
Example: Custom pre-invoke Tool with its own schema (Lesson Plan Composer)

This script shows how to define a custom pre-invoke Tool that expects a specific
input schema and converts that mapping into a single prompt string for the Agent.

It includes:
  1) A STRICT Tool (unknown top-level keys -> error).
  2) A PERMISSIVE Tool (accepts extra keys via **kwargs pattern).

Usage:
  - Run as-is to use the STRICT tool.
  - Set use_permissive=True in main() to try the permissive variant.
"""
from atomic_agentic.Agents import Agent
from atomic_agentic.Tools import Tool
from atomic_agentic.LLMEngines import OpenAIEngine  # swap for another engine if desired
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()

# --------------------------- STRICT schema Tool --------------------------- #
def lesson_prompt_strict(
    *,
    grade_level: str,
    subject: str,
    topic: str,
    duration_min: int = 45,
    objectives: List[str],
    constraints: Optional[List[str]] = None,
    tone: str = "practical",
) -> str:
    """
    STRICT schema:
      Required keys: grade_level, subject, topic, objectives(list[str])
      Optional keys: duration_min(int, default 45), constraints(list[str], default []), tone(str, default 'practical')
    Unknown top-level keys will raise (matches Tool's strict binding rules).
    Returns a single prompt string to feed the Agent/LLM.
    """
    constraints = constraints or []

    def _fmt_block(title: str, items: List[str]) -> str:
        if not items:
            return f"{title}: (none)"
        return f"{title}:\n" + "\n".join(f"- {x}" for x in items)

    lines = [
        f"Compose a {tone} lesson plan.",
        f"Grade level: {grade_level}",
        f"Subject: {subject}",
        f"Topic: {topic}",
        f"Duration: {duration_min} minutes",
        _fmt_block("Learning objectives", objectives),
        _fmt_block("Constraints", constraints),
        "Deliverables:",
        "- A brief lesson overview",
        "- A step-by-step outline",
        "- A quick formative assessment item",
        "- One extension activity",
        "Style: concise, actionable, teacher-friendly.",
    ]
    return "\n".join(lines)


strict_tool = Tool(
    func=lesson_prompt_strict,
    name="lesson_prompt_strict",
    description="Strict schema: {grade_level, subject, topic, duration_min?, objectives[], constraints[]?, tone?} → prompt",
)


# ------------------------- PERMISSIVE schema Tool ------------------------ #
def lesson_prompt_permissive(**kwargs) -> str:
    """
    PERMISSIVE variant:
      - Accepts any keys; uses known ones with defaults and ignores unknowns.
      - Useful while iterating on your schema.
    Known (optional) keys: grade_level, subject, topic, duration_min, objectives(list[str]), constraints(list[str]), tone(str)
    """
    grade_level = kwargs.get("grade_level", "unspecified")
    subject = kwargs.get("subject", "general studies")
    topic = kwargs.get("topic", "general topic")
    duration_min = int(kwargs.get("duration_min", 45))
    objectives = list(kwargs.get("objectives", []))
    constraints = list(kwargs.get("constraints", []))
    tone = kwargs.get("tone", "practical")

    def _fmt_block(title: str, items: List[str]) -> str:
        if not items:
            return f"{title}: (none)"
        return f"{title}:\n" + "\n".join(f"- {x}" for x in items)

    lines = [
        f"Compose a {tone} lesson plan.",
        f"Grade level: {grade_level}",
        f"Subject: {subject}",
        f"Topic: {topic}",
        f"Duration: {duration_min} minutes",
        _fmt_block("Learning objectives", objectives),
        _fmt_block("Constraints", constraints),
        "Deliverables:",
        "- A brief lesson overview",
        "- A step-by-step outline",
        "- A quick formative assessment item",
        "- One extension activity",
        "Style: concise, actionable, teacher-friendly.",
    ]
    return "\n".join(lines)


permissive_tool = Tool(
    func=lesson_prompt_permissive,
    name="lesson_prompt_permissive",
    description="Permissive: accepts extra keys; uses defaults; builds a lesson-planning prompt.",
)


def main(use_permissive: bool = False) -> None:
    # 1) Choose an engine (OpenAI here; replace if needed)
    engine = OpenAIEngine(model="gpt-4o-mini")

    # 2) Build an Agent and install the pre-invoke Tool
    agent = Agent(
        name="LessonPlanner",
        description="Builds lesson plans from structured inputs.",
        llm_engine=engine,
        role_prompt="You are an expert teacher who writes concise, practical lesson plans.",
        context_enabled=True,
        history_window=8,
        pre_invoke=(permissive_tool if use_permissive else strict_tool),
    )

    # 3) Prepare inputs (MAPPING ONLY!). Match the Tool’s schema.
    inputs = {
        "grade_level": "Middle School (6–8)",
        "subject": "Science",
        "topic": "Magnetic fields and flux",
        "duration_min": 50,
        "objectives": [
            "Define magnetic field and magnetic flux in simple terms",
            "Demonstrate field direction with a compass",
            "Relate flux changes to everyday examples",
        ],
        "constraints": [
            "Classroom has only 5 compasses",
            "No live electricity experiments",
        ],
        "tone": "hands-on",
        # STRICT tool note:
        #   Any unknown top-level key here would raise ToolInvocationError.
        # PERMISSIVE tool note:
        #   Unknown keys would be ignored.
        # "extra_note": "This would break strict mode.",
    }

    # 4) Invoke (the Tool converts the mapping into a prompt string)
    lesson_plan = agent.invoke(inputs)

    # 5) Print the non-deterministic LLM output
    print("\n=== Generated Lesson Plan ===\n")
    print(lesson_plan)


if __name__ == "__main__":
    # Toggle to True to see the permissive behavior
    main(use_permissive=False)
