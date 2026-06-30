# 06_Branching_Conversations.py
"""
Demonstrates conversation branching via the ``run_id`` parameter.

The scenario uses grounded, verifiable facts so the agent's context isolation
is unambiguous — it cannot invent the right answer if the fact was never in
its chain, so missing context produces a clear, observable "I don't know."

Conversation tree
-----------------

    T0 (root)  ──────────────────────────┐
        │                                │
    T_A1  reveal dog name                │
        │                                │
    T_A2  "what do you know             T_B1  "what's my dog's
          about my dog?"                        name?" (fork from T0)
          → agent recalls Biscuit               → agent: "you haven't
                                                  told me"

Key concepts shown
------------------
- ``run_id=<run_id>``   fork from any prior turn, not just the tail
- ``Agent.get_conversation()`` reconstruct any branch chain from flat history

Breaking change (v2.0.0a14)
----------------------------
``run_id="new"`` has been removed.  It previously opened a fresh
conversation root without clearing flat history.  The new design always
appends records unconditionally; to get a fresh conversation per-invocation
use ``context_enabled=False``, or instantiate a separate agent.

Post-invoke parameters are no longer declared via ``passthrough_inputs``.
Any non-result, non-variadic post_invoke parameter (e.g. ``style`` below)
is auto-grafted into the agent schema as KEYWORD_ONLY — callers pass it
as a regular input.
"""
from __future__ import annotations

import os
import textwrap
from dotenv import load_dotenv

from atomic_agentic.agents import BasicAgent
from atomic_agentic.engines.LLMEngines import OpenAIEngine

load_dotenv()

_W = 66  # display column width


# ---------------------------------------------------------------------------
# Lifecycle hooks
# ---------------------------------------------------------------------------

def build_message(message: str, style: str = "friendly") -> str:
    """Pre-invoke: inject a style directive before the user's message."""
    directives = {
        "friendly": "Respond warmly and conversationally.",
        "formal":   "Respond in a professional, formal tone.",
        "concise":  "Respond in one sentence only. Be direct.",
    }
    directive = directives.get(style, "Respond naturally.")
    return f"[{directive}]\n\n{message}"


def package_reply(result: str, style: str = "friendly") -> dict:
    """Post-invoke: wrap the raw LLM response with lightweight metadata.

    ``style`` is auto-grafted from this function's signature into the agent
    schema — no ``passthrough_inputs`` configuration needed.
    """
    return {
        "reply":      result.strip(),
        "style":      style,
        "char_count": len(result.strip()),
    }


# ---------------------------------------------------------------------------
# Display helper
# ---------------------------------------------------------------------------

def print_chain(chain: list, label: str) -> None:
    """Pretty-print a conversation chain returned by get_conversation()."""
    print(f"\n{'━' * _W}")
    print(f"  {label}")
    print(f"  {len(chain)} turn(s) in chain")
    print(f"{'━' * _W}")

    for i, record in enumerate(chain):
        run_short  = record.final_result.run_id[:8]
        prev_short = (
            f"{record.prev.final_result.run_id[:8]}..."
            if record.prev else "None  ← chain root"
        )
        payload = record.final_result.result

        print(f"\n  [Turn {i + 1}]  run: {run_short}...   parent: {prev_short}")
        print(f"  style: {payload['style']}  ·  {payload['char_count']} chars")

        flat_msg = " ".join(record.user_prompt.split())
        msg_lines = textwrap.wrap(flat_msg, width=_W - 13)
        if msg_lines:
            print(f"\n  you     : {msg_lines[0]}")
            for line in msg_lines[1:]:
                print(f"            {line}")

        reply_lines = textwrap.wrap(payload["reply"], width=_W - 13)
        if reply_lines:
            print(f"\n  agent   : {reply_lines[0]}")
            for line in reply_lines[1:]:
                print(f"            {line}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    engine = OpenAIEngine(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")

    agent = BasicAgent(
        name="branching_demo_agent",
        namespace="examples",
        description="Conversational agent for branching context demonstration.",
        llm_engine=engine,
        role_prompt=(
            "You are a helpful assistant in a conversation with the user. "
            "You only know what has been said in this conversation. "
            "When asked about something that has not been mentioned, "
            "say clearly that you don't have that information — "
            "never guess or invent details."
        ),
        context_enabled=True,
        records_window=None,
        pre_invoke=build_message,
        post_invoke=package_reply,
        post_result_key="result",
        # 'style' is auto-grafted from package_reply into the agent schema.
        # No passthrough_inputs declaration needed.
    )

    # ------------------------------------------------------------------ #
    # T0 — root: establish the user
    # ------------------------------------------------------------------ #
    print(f"\n{'═' * _W}")
    print("  T0  ·  Root  —  common starting point for all branches")
    print(f"{'═' * _W}")

    r0 = agent(
        message="Hi, I'm Sam. Nice to meet you.",
        style="friendly",
    )
    run_0 = r0.run_id
    print(f"  run  : {run_0[:8]}...   parent: None")
    print(f"  reply: {r0.result['reply']}")

    # ------------------------------------------------------------------ #
    # Branch A — T_A1: reveal a fact the agent must remember
    # ------------------------------------------------------------------ #
    print(f"\n{'═' * _W}")
    print("  T_A1  ·  Branch A  —  revealing a fact")
    print(f"{'═' * _W}")

    r_a1 = agent(
        message="I have a golden retriever named Biscuit. She loves fetch.",
        style="friendly",
    )
    run_a1 = r_a1.run_id
    print(f"  run  : {run_a1[:8]}...   parent: {run_0[:8]}...")
    print(f"  reply: {r_a1.result['reply']}")

    # ------------------------------------------------------------------ #
    # Branch A — T_A2: test recall of the fact
    # ------------------------------------------------------------------ #
    print(f"\n{'═' * _W}")
    print("  T_A2  ·  Branch A  —  recalling the fact  (agent should know)")
    print(f"{'═' * _W}")

    r_a2 = agent(
        message="What do you know about my dog?",
        style="friendly",
    )
    run_a2 = r_a2.run_id
    print(f"  run  : {run_a2[:8]}...   parent: {run_a1[:8]}...")
    print(f"  reply: {r_a2.result['reply']}")

    # ------------------------------------------------------------------ #
    # Branch B — T_B1: fork from T0, ask the same recall question
    #
    # This branch's context is T0 only. T_A1 and T_A2 never happened here.
    # The agent cannot know Biscuit's name.
    # ------------------------------------------------------------------ #
    print(f"\n{'═' * _W}")
    print("  T_B1  ·  Branch B  —  forked from T0  (run_id=run_0)")
    print("  ↳ T_A1 and T_A2 are invisible — agent has no dog info")
    print(f"{'═' * _W}")

    r_b1 = agent(
        message="What's my dog's name?",
        style="friendly",
        run_id=run_0,
    )
    run_b1 = r_b1.run_id
    print(f"  run  : {run_b1[:8]}...   parent: {run_0[:8]}...")
    print(f"  reply: {r_b1.result['reply']}")

    # ------------------------------------------------------------------ #
    # Reconstruct and display each branch chain
    # ------------------------------------------------------------------ #
    print(f"\n\n{'═' * _W}")
    print(f"  BRANCH CHAINS  ({len(agent.records)} records in flat history)")
    print(f"{'═' * _W}")

    print_chain(
        agent.get_conversation(run_id=run_a2),
        "Branch A  ·  root → fact revealed → fact recalled",
    )
    print_chain(
        agent.get_conversation(run_id=run_b1),
        "Branch B  ·  root → same question, no prior fact  (fork at T0)",
    )

    # ------------------------------------------------------------------ #
    # Flat history — all records stored, regardless of branch
    # ------------------------------------------------------------------ #
    print(f"\n{'═' * _W}")
    print(f"  Flat records  —  {len(agent.records)} records total")
    print(f"{'═' * _W}")
    labels = {
        run_0:  "T0     root",
        run_a1: "T_A1   branch A — fact revealed",
        run_a2: "T_A2   branch A — fact recalled",
        run_b1: "T_B1   branch B — parallel fork from T0",
    }
    for i, record in enumerate(agent.records):
        rid    = record.final_result.run_id
        parent = (
            f"{record.prev.final_result.run_id[:8]}..."
            if record.prev else "None"
        )
        print(f"  [{i}]  {rid[:8]}...  parent: {parent:<18}  {labels.get(rid, '')}")
    print(f"{'═' * _W}\n")


if __name__ == "__main__":
    main()
