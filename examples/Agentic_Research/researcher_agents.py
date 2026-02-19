from typing import Any, Mapping
from dotenv import load_dotenv

from atomic_agentic.agents import Agent
from atomic_agentic.engines.LLMEngines import OpenAIEngine

load_dotenv()

# LLM
llm = OpenAIEngine(model="gpt-4o-mini")

# ---------------------------------------------------------------------
# Writer Agent: Produces APA report drafts using research & feedback
# ---------------------------------------------------------------------
def writer_pre(*, query: str = "", sources: list[str] | None = None, revision_notes: str = "") -> str:
    sources = sources or []
    # If revision notes are present, this is a revision step:
    if revision_notes.strip():
        return (
            "Use the following REVISION NOTES to revise your latest research report draft:\n"
            f"{revision_notes.strip()}\n\n"
        )
    # First draft: include query + sources
    src_block = "\n\n".join(sources) if sources else "(no sources returned)"
    return (
        "Write an APA-formatted research report given the following query and sources.\n\n"
        f"RESEARCH QUESTION:\n{query.strip()}\n\n"
        "SOURCES (formatted passages + URLs):\n"
        f"{src_block}\n\n"
    )

def writer_post(*, result: Any) -> Mapping[str, str]:
    return {"draft": str(result).strip()}

writer = Agent(
    name="report_writer",
    description="Drafts and revises an APA-style research report.",
    llm_engine=llm,
    role_prompt=(
        "You are an expert research writer.\n"
        "Write APA-style reports grounded in the provided sources.\n"
        "Never fabricate citations."
    ),
    pre_invoke=writer_pre,
    post_invoke=writer_post,
    filter_extraneous_inputs=True,
    context_enabled=True,
)


# ---------------------------------------------------------------------
# Critic agent: updates {"revision_notes": "..."} and may return "<<APPROVED>>"
# ---------------------------------------------------------------------
def critic_pre(*, draft: str) -> str:
    return (
        "Review and provide feedback for the following research draft:\n\n"
        f"{draft.strip()}\n"
    )

def critic_post(*, result: str) -> Mapping[str, str]:
    return {"revision_notes": str(result).strip()}

critic = Agent(
    name="report_critic",
    description="Reviews the report draft and returns revision notes or <<APPROVED>>.",
    llm_engine=llm,
    role_prompt=(
        "You are a strict but fair APA research report reviewer.\n\n"
        "Evaluate the draft for:\n"
        "- APA structure (title/abstract/headings/references)\n"
        "- Citation integrity (no invented sources)\n"
        "- Logical flow and clarity\n"
        "- Faithful grounding in provided sources (no hallucinated facts)\n\n"
        f"If the draft is ready, reply ONLY with: <<APPROVED>>\n\n"
        "Otherwise, provide revision notes (bullet points) on what to add/remove/change."
    ),
    pre_invoke=critic_pre,
    post_invoke=critic_post,
    filter_extraneous_inputs=True,
    context_enabled=False,
)

if __name__ == "__main__":
    # Example Agent usage
    query = "What are the main benefits and risks of CRISPR gene editing in medicine?"
    sources = [
        "Source 1: CRISPR has shown promise in treating genetic diseases (Smith et al., 2020). [URL]",
        "Source 2: Ethical concerns around CRISPR include potential off-target effects and germline editing (Doe, 2021). [URL]",
    ]
    draft = writer.invoke({"query": query, "sources": sources})
    print("Draft:\n", draft["draft"])