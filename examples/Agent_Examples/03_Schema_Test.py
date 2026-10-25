"""
Example: Custom pre-invoke Tool with its own schema (Lesson Plan Composer)

This script shows how to define a custom pre-invoke Tool that expects a specific
input schema and converts that mapping into a single prompt string for the Agent.

The Tool declares explicit keyword-only params (no **kwargs sink): missing
required keys raise, and unrecognized keys are silently dropped. A pre_invoke
(or post_invoke) that declares **kwargs is rejected when the Agent is built --
every name it consumes must be known up front. For a lenient schema, give the
optional parameters defaults instead.
"""
from atomic_agentic.agents import BasicAgent
from atomic_agentic.tools import Tool
from atomic_agentic.llm import OpenAIEngine  # swap for another engine if desired
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()

# ------------------------------ Schema Tool ------------------------------ #
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
    Required schema:
      Required keys: grade_level, subject, topic, objectives(list[str])
      Optional keys: duration_min(int, default 45), constraints(list[str], default []), tone(str, default 'practical')
    Missing required keys raise; unknown top-level keys are silently dropped
    (there is no **kwargs sink here to collect them).
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
    function=lesson_prompt_strict,
    name="lesson_prompt_strict",
    description="Required schema: {grade_level, subject, topic, duration_min?, objectives[], constraints[]?, tone?} → prompt",
)


def main() -> None:
    # 1) Choose an engine (OpenAI here; replace if needed)
    engine = OpenAIEngine(model="gpt-4o-mini")

    # 2) Build an Agent and install the pre-invoke Tool
    agent = BasicAgent(
        name="LessonPlanner",
        namespace="examples",
        description="Builds lesson plans from structured inputs.",
        llm_engine=engine,
        role_prompt="You are an expert teacher who writes concise, practical lesson plans.",
        record_enabled=True,
        records_window=8,
        pre_invoke=strict_tool,
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
        # An unknown top-level key here would be silently dropped (no
        # **kwargs sink to collect it).
        # "extra_note": "This key would be dropped.",
    }

    # 4) Invoke (the Tool converts the mapping into a prompt string)
    lesson_plan = agent.invoke(inputs)

    # 5) Print the non-deterministic LLM output
    print("\n=== Generated Lesson Plan ===\n")
    print(lesson_plan.result)


if __name__ == "__main__":
    main()
