# 09_Shared_Conversation_Context.py
"""
A writer and a reviewer share ONE Conversation and keep separate context.

  * ``writer`` (plain text) binds its response to the ``draft`` variable.
  * ``reviewer`` (structured response) binds two fields of its response to two
    variables, ``verdict`` and ``critique``.

Both agents are ``BasicAgent`` instances, so ``import_conversation`` can
register the writer's Conversation object under the reviewer's key (a share,
not a copy). Consequences shown below:

  * The reviewer "reads" the draft through the shared HISTORY: the writer's
    turns are replayed into the reviewer's prompt, and vice versa for the
    critique. (Context variables are not rendered into prompts yet -- a later
    pass -- so that is how the agents inform each other today.)
  * Each agent's response lands in its own context variable via
    ``response_to_context``, so after every call the caller can read the latest
    draft, critique and verdict separately from the conversation.
  * Each record carries the ``agent_id`` of the agent that wrote it.

A bound name only writes to an EXISTING, writable variable (nothing is
created), so the variables are ``remember``-ed up front.
"""
from __future__ import annotations

import os

from dotenv import load_dotenv

from atomic_agentic.agents import BasicAgent
from atomic_agentic.llm import OpenAIEngine

load_dotenv()

MAX_ROUNDS = 3

REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["approve", "revise"]},
        "critique": {"type": "string"},
    },
    "required": ["verdict", "critique"],
    "additionalProperties": False,
}


def main() -> None:
    engine = OpenAIEngine(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-5-mini")

    writer = BasicAgent(
        name="writer",
        namespace="examples",
        description="Writes and revises a short story draft.",
        llm_engine=engine,
        role_prompt=(
            "You are a fiction writer. Reply with ONLY the story text, at most "
            "100 words. When the conversation contains a reviewer critique, "
            "revise your latest draft to address it."
        ),
        response_to_context="draft",
    )
    reviewer = BasicAgent(
        name="reviewer",
        namespace="examples",
        description="Critiques the latest story draft in the conversation.",
        llm_engine=engine,
        role_prompt=(
            "You are a strict editor. Review the writer's LATEST draft in this "
            "conversation. Approve only if it has a clear arc and a strong "
            "ending; otherwise ask for a revision with one specific critique."
        ),
        response_schema=REVIEW_SCHEMA,
        response_to_context=["verdict", "critique"],
    )

    # -- Shared context, defined once on the writer's conversation ---------
    conv = writer.get_conversation()
    conv.remember("draft", "", "Latest story draft (writer)")
    conv.remember("verdict", "", "Latest review verdict (reviewer)")
    conv.remember("critique", "", "Latest critique (reviewer)")

    # -- Share the SAME conversation object with the reviewer --------------
    reviewer.import_conversation(conv, name="shared")
    reviewer.switch_conversation("shared")
    assert reviewer.get_conversation() is conv

    prompt = "Write a short story about a lighthouse keeper who finds a message in a bottle."
    for round_no in range(1, MAX_ROUNDS + 1):
        writer.invoke({"prompt": prompt})
        print(f"\n--- Round {round_no} ---")
        print(f"draft    : {conv.context['draft'].value}")

        review = reviewer.invoke({"prompt": "Review the latest draft."})
        print(f"verdict  : {conv.context['verdict'].value}")
        print(f"critique : {conv.context['critique'].value}")
        print(f"(reviewer returned: {review.result})")

        if conv.context["verdict"].value == "approve":
            break
        prompt = "Revise your latest draft to address the reviewer's critique."

    print("\n=== Shared history ===")
    for i, record in enumerate(conv.records):
        who = record.agent_id.split(";")[0]
        print(f"[{i}] {who}: {str(record.generated_response)[:90]!r}")


if __name__ == "__main__":
    main()
