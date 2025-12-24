"""
StateIOFlow + LangGraph demo: Minimal Ticket Triage (with Human-in-the-Loop)

What this shows:
- A *single shared state schema* across all nodes.
- Each LangGraph node is a StateIOFlow wrapping either an Agent or a Tool.
- Conditional routing:
    - If we need more info -> ask a question -> get user input -> merge -> draft answer
    - Else -> draft answer directly
- No per-node logging; we only print the final state.

Requirements:
- pip install langgraph
- pip install python-dotenv
- OPENAI_API_KEY in environment or .env
"""

from __future__ import annotations

import json
import re
from typing import Any, TypedDict

from dotenv import load_dotenv

try:
    from langgraph.graph import StateGraph, END
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "This example requires 'langgraph'. Install with: pip install langgraph"
    ) from exc

from atomic_agentic.LLMEngines import OpenAIEngine
from atomic_agentic.Primitives import Agent, Tool
from atomic_agentic.Workflows import StateIOFlow


# ---------------------------------------------------------------------
# 1) Minimal State Schema
# ---------------------------------------------------------------------
class TicketState(TypedDict, total=False):
    ticket: str
    category: str               # "billing" | "bug" | "feature" | "other"
    needs_info: bool
    question: str
    user_reply: str
    answer: str


# ---------------------------------------------------------------------
# 2) Small JSON extraction helper for strict-ish LLM formatting
# ---------------------------------------------------------------------
_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json_object(text: str) -> dict[str, Any]:
    """
    Best-effort JSON object extractor for LLM outputs.

    - Strips common fenced blocks
    - Finds first {...} block (DOTALL)
    - json.loads it
    """
    s = (text or "").strip()

    # strip common fenced blocks like ```json ... ```
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s).strip()

    m = _JSON_OBJ_RE.search(s)
    if not m:
        raise ValueError(f"No JSON object found in LLM output: {text!r}")
    return json.loads(m.group(0))


# ---------------------------------------------------------------------
# 3) Agent builders
#
# IMPORTANT: These Agents will be invoked via StateIOFlow, which passes
# the *entire state mapping* as inputs. Therefore, all pre_invoke tools
# accept **_ to tolerate extra keys.
# ---------------------------------------------------------------------
def make_categorize_agent(engine: OpenAIEngine) -> Agent:
    def pre(ticket: str, **_: Any) -> str:
        return (
            "You are a ticket triage assistant.\n"
            "Classify the ticket into one category:\n"
            '  "billing" | "bug" | "feature" | "other"\n\n'
            "Return ONLY a JSON object like:\n"
            '{"category": "billing"}\n\n'
            f"TICKET:\n{ticket}"
        )

    def post(result: Any) -> dict[str, Any]:
        obj = _extract_json_object(str(result))
        cat = str(obj.get("category", "other")).strip().lower()
        if cat not in {"billing", "bug", "feature", "other"}:
            cat = "other"
        return {"category": cat}

    pre_tool = Tool(pre, name="categorize_pre", namespace="ticket_triage")
    post_tool = Tool(post, name="categorize_post", namespace="ticket_triage")

    return Agent(
        name="categorize_agent",
        description="Categorize a support ticket",
        llm_engine=engine,
        role_prompt="You follow output format exactly.",
        context_enabled=False,
        pre_invoke=pre_tool,
        post_invoke=post_tool,
        history_window=0,
    )


def make_check_missing_info_agent(engine: OpenAIEngine) -> Agent:
    def pre(ticket: str, category: str, **_: Any) -> str:
        return (
            "You are a ticket triage assistant.\n"
            "Decide if we have enough info to respond immediately.\n\n"
            "Rules of thumb (simple):\n"
            "- billing: often needs invoice/transaction/account detail\n"
            "- bug: often needs repro steps or an error message\n"
            "- feature: usually enough to acknowledge + ask clarifying only if unclear\n"
            "- other: usually ask one clarifying question\n\n"
            "Return ONLY JSON:\n"
            '{"needs_info": true}  or  {"needs_info": false}\n\n'
            f"CATEGORY: {category}\n"
            f"TICKET:\n{ticket}"
        )

    def post(result: Any) -> dict[str, Any]:
        obj = _extract_json_object(str(result))
        needs = obj.get("needs_info", True)

        if isinstance(needs, bool):
            needs_bool = needs
        elif isinstance(needs, (int, float)):
            needs_bool = bool(needs)
        else:
            needs_bool = str(needs).strip().lower() in {"true", "1", "yes", "y"}

        return {"needs_info": needs_bool}

    pre_tool = Tool(pre, name="missing_pre", namespace="ticket_triage")
    post_tool = Tool(post, name="missing_post", namespace="ticket_triage")

    return Agent(
        name="check_missing_info_agent",
        description="Determine whether ticket needs follow-up info",
        llm_engine=engine,
        role_prompt="Be conservative: ask for missing info if required.",
        context_enabled=False,
        pre_invoke=pre_tool,
        post_invoke=post_tool,
        history_window=0,
    )


def make_ask_question_agent(engine: OpenAIEngine) -> Agent:
    def pre(ticket: str, category: str, **_: Any) -> str:
        return (
            "You are a ticket triage assistant.\n"
            "We need ONE follow-up question to proceed.\n"
            "Ask exactly one crisp question.\n\n"
            "Return ONLY JSON like:\n"
            '{"question": "..." }\n\n'
            f"CATEGORY: {category}\n"
            f"TICKET:\n{ticket}"
        )

    def post(result: Any) -> dict[str, Any]:
        obj = _extract_json_object(str(result))
        q = str(obj.get("question", "")).strip()
        if q and not q.endswith("?"):
            q += "?"
        return {"question": q}

    pre_tool = Tool(pre, name="ask_pre", namespace="ticket_triage")
    post_tool = Tool(post, name="ask_post", namespace="ticket_triage")

    return Agent(
        name="ask_question_agent",
        description="Generate one follow-up question",
        llm_engine=engine,
        role_prompt="Ask one question only. No extra text.",
        context_enabled=False,
        pre_invoke=pre_tool,
        post_invoke=post_tool,
        history_window=0,
    )


def make_draft_answer_agent(engine: OpenAIEngine) -> Agent:
    def pre(ticket: str, category: str, **_: Any) -> str:
        return (
            "You are a helpful support agent.\n"
            "Draft a short response based on the ticket.\n"
            "If you can't fully resolve, provide the next best steps.\n"
            "Do NOT request passwords or secret credentials.\n\n"
            "Return ONLY JSON like:\n"
            '{"answer": "..."}\n\n'
            f"CATEGORY: {category}\n"
            f"TICKET (may include follow-up answer already):\n{ticket}"
        )

    def post(result: Any) -> dict[str, Any]:
        obj = _extract_json_object(str(result))
        ans = str(obj.get("answer", "")).strip()
        return {"answer": ans}

    pre_tool = Tool(pre, name="answer_pre", namespace="ticket_triage")
    post_tool = Tool(post, name="answer_post", namespace="ticket_triage")

    return Agent(
        name="draft_answer_agent",
        description="Draft the final answer",
        llm_engine=engine,
        role_prompt="Be concise, helpful, and policy-safe.",
        context_enabled=False,
        pre_invoke=pre_tool,
        post_invoke=post_tool,
        history_window=0,
    )


# ---------------------------------------------------------------------
# 4) Deterministic tool nodes (user input + merge)
#
# IMPORTANT: These tools are invoked via StateIOFlow which passes the entire
# state mapping. Therefore, accept **_ to tolerate extra keys.
# ---------------------------------------------------------------------
def make_get_user_input_tool() -> Tool:
    def get_user_input(question: str, **_: Any) -> dict[str, Any]:
        print("\n=== NEED USER INPUT ===")
        print(f"Q: {question}")
        reply = input("Your reply: ").strip()
        return {"user_reply": reply}

    return Tool(get_user_input, name="get_user_input", namespace="ticket_triage")


def make_merge_user_reply_tool() -> Tool:
    def merge_user_reply(ticket: str, question: str, user_reply: str, **_: Any) -> dict[str, Any]:
        enriched = (
            ticket.rstrip()
            + "\n\nFOLLOW-UP QUESTION:\n"
            + question.strip()
            + "\nFOLLOW-UP ANSWER:\n"
            + user_reply.strip()
        )
        return {
            "ticket": enriched,
            "needs_info": False,
            "question": "",
        }

    return Tool(merge_user_reply, name="merge_user_reply", namespace="ticket_triage")


# ---------------------------------------------------------------------
# 5) LangGraph wiring (each node is a StateIOFlow)
# ---------------------------------------------------------------------
def main() -> None:
    load_dotenv()

    engine = OpenAIEngine(model="gpt-4o-mini")  # expects OPENAI_API_KEY via env/.env

    # Components
    categorize_agent = make_categorize_agent(engine)
    missing_agent = make_check_missing_info_agent(engine)
    ask_agent = make_ask_question_agent(engine)
    answer_agent = make_draft_answer_agent(engine)

    get_user_input_tool = make_get_user_input_tool()
    merge_reply_tool = make_merge_user_reply_tool()

    # Wrap every component as StateIOFlow (LangGraph-compatible)
    n_categorize = StateIOFlow(categorize_agent, state_schema=TicketState)
    n_missing = StateIOFlow(missing_agent, state_schema=TicketState)
    n_ask = StateIOFlow(ask_agent, state_schema=TicketState)
    n_get_user = StateIOFlow(get_user_input_tool, state_schema=TicketState)
    n_merge = StateIOFlow(merge_reply_tool, state_schema=TicketState)
    n_answer = StateIOFlow(answer_agent, state_schema=TicketState)

    graph = StateGraph(TicketState)

    # Directly pass flow.invoke (no wrapper)
    graph.add_node("categorize", n_categorize.invoke)
    graph.add_node("check_missing", n_missing.invoke)
    graph.add_node("ask_question", n_ask.invoke)
    graph.add_node("get_user_input", n_get_user.invoke)
    graph.add_node("merge_user_reply", n_merge.invoke)
    graph.add_node("draft_answer", n_answer.invoke)

    graph.set_entry_point("categorize")
    graph.add_edge("categorize", "check_missing")

    def route_after_missing(state: TicketState) -> str:
        return "ask_question" if state.get("needs_info") is True else "draft_answer"

    graph.add_conditional_edges(
        "check_missing",
        route_after_missing,
        {"ask_question": "ask_question", "draft_answer": "draft_answer"},
    )

    graph.add_edge("ask_question", "get_user_input")
    graph.add_edge("get_user_input", "merge_user_reply")
    graph.add_edge("merge_user_reply", "draft_answer")
    graph.add_edge("draft_answer", END)

    app = graph.compile()

    # Seed state (minimal)
    seed: TicketState = {
        "ticket": "I was charged twice this month. Can you refund one?",
    }

    final_state = app.invoke(seed)

    print("\n================ FINAL STATE ================")
    for k in ["ticket", "category", "needs_info", "question", "user_reply", "answer"]:
        if k in final_state:
            print(f"{k}: {final_state[k]}")


if __name__ == "__main__":
    main()
