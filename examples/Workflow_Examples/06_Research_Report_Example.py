import sys, logging
from pathlib import Path
from typing import Any
from langchain_tavily import TavilySearch, TavilyExtract

# Setting the root
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from modules.Agents import Agent
from modules.LLMEngines import OpenAIEngine
from modules.ToolAgents import PlannerAgent
from modules.Workflows import *
from modules.Tools import Tool

def research(query: str, max_sources: int = 3) -> tuple:
    """
    Search the web for up to `max_sources` results with Tavily,
    then extract and concatenate their page content into one string.
    """
    # --- 1) Search ---
    search_tool = TavilySearch(
        max_results=max_sources,
        search_depth="advanced",
        include_answer=False,
        include_images=False,
        include_raw_content=False,   # we'll extract in step 2
    )
    search_res = search_tool.invoke({"query": query})
    # TavilySearch returns a dict with a "results" list of dicts (url/title/content/score).
    urls: List[str] = [r["url"] for r in (search_res.get("results") or [])][:max_sources]

    if not urls:
        return ""

    # --- 2) Extract (scrape) ---
    extract_tool = TavilyExtract(
        extract_depth="advanced",     # "basic" | "advanced"
        include_images=False,
        include_favicon=False,
        # Optional: format="text"  # defaults to "markdown"
    )
    extract_res = extract_tool.invoke({"urls": urls})
    # IMPORTANT: TavilyExtract returns a dict with "results" (list of dicts).
    results = extract_res.get("results", [])

    parts: List[str] = []
    for res in results:
        url = res.get("url", "")
        title = res.get("title") or url
        raw = (res.get("raw_content") or "").strip()  # <-- use raw_content
        if raw:
            parts.append(f"# {title}\nSource: {url}\n\n{raw}\n")

    return query, "\n\n---\n\n".join(parts)

research_tool = Tool("Research_Tool", research, description="A tool that takes in a research query and a maximum number of sources to retrieve")

def build_report_request_prompt(query, research_results):
  return (
    "Synthesize a comprehensive research report that accurately and thoroughly "
    f"addresses the user's query: {query}\n"
    "The report should not just simply answer the question itself, nor just list facts "
    "from the research content. It should provide clear insights and observations that "
    "you extract from the research data, as well. For references, use the research provided "
    f"below: \n {research_results}"
  )
report_prompt_tool = Tool("report_prompt_tool", build_report_request_prompt, description="constructs a formatted query to give to our report-builder")

researcher_prompt = """
You are the Research Report Writer.

Your job is to take the research results (raw text and sources) and turn them into a clear, well-structured report.

Follow these simple rules:
1. **Report format**:
   - Title
   - Introduction (what the query is about)
   - Findings (organized into 3–5 subsections, each with supporting evidence)
   - Discussion (synthesize evidence, note agreements/disagreements, limits)
   - Conclusion (key takeaways, recommendations if relevant)
   - References (list of all sources used)

2. **Use of results**:
   - Paraphrase and summarize the source material.
   - Compare and combine findings rather than copying text.
   - Point out if sources conflict or if data is limited.

3. **Citing**:
   - Inline citations in the form: (Source: [title or URL]).
   - At the end, include a full References list with titles and URLs.

Keep language professional, concise, and evidence-driven.
"""

editor_prompt = """
You are the Research Report Critic.

Your job is to review a draft report and give constructive feedback before finalization.

Judge the report on these criteria:
1. Structure: Does it follow the required format (Intro, Findings, Discussion, Conclusion, References)?
2. Evidence: Are claims backed with sources? Are conflicts or gaps noted?
3. Citations: Are sources cited consistently and fully listed?
4. Clarity: Is the writing clear, concise, and professional?
5. Synthesis: Does the report connect ideas instead of just listing facts?
6. Conclusions: Do they logically follow from the findings?

For each section, briefly note strengths and improvements. Then give an overall recommendation.
"""

LLM = OpenAIEngine("gpt-4o")
report_maker = MakerChecker(
    "report-builder",
    "Takes in research output and synthesizes well-formatted research reports",
    researcher_prompt,
    editor_prompt,
    LLM,
    LLM,
    max_revisions=3,
)

researcher = ChainOfThought(
    "Researcher",
    "Researches a user query, passes the query & research to a writer & editor, and returns the final output",
    steps = [research_tool, report_prompt_tool, report_maker]
)
logging.getLogger().setLevel(logging.INFO)
report_topic = input("Give a report topic query for us to investigate: ")
max_sources = input("How many sources should we be allowed to use (give a number using 0-9)? ")
drafts, final = researcher.invoke(report_topic, int(max_sources))
print("FINAL DRAFT\n",final)