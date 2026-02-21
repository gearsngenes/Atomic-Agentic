"""
Module for reusable research tools used in the Agentic Research examples
"""
from tavily import TavilyClient
from typing import Mapping
from dotenv import load_dotenv

from atomic_agentic.tools import Tool

# Load TAVILY_API_KEY from environment or .env file
load_dotenv()

# Instantiate a Tavily client (will use TAVILY_API_KEY from env by default)
client = TavilyClient(api_key=None)

# ------------------------------------------------------
# Research Tool using Tavily SDK
# ------------------------------------------------------
def tavily_research(*, query: str, max_results: int = 5) -> Mapping[str, list[str]]:
    """
    Expected Arguments:
    - query: str

    This function uses the Tavily SDK to perform a search based on
    the input query and then extracts relevant information from 
    the search results. It returns a mapping containing a list of
    sources, where each source includes the title, URL, and an 
    excerpt of the content.
    """
    search = client.search(
        query=query,
        search_depth="advanced",
        max_results=max_results,
        include_answer=False,
        include_raw_content=False,
    )

    results = list(search.get("results", []) or [])
    urls = [str(r.get("url", "")).strip() for r in results if str(r.get("url", "")).strip()]

    extracted_items: list[str] = []

    if urls:
        extract = client.extract(urls=urls)
        extract_results = list(extract.get("results", []) or [])
        by_url = {str(item.get("url", "")).strip(): item for item in extract_results}

        for i, r in enumerate(results, start=1):
            url = str(r.get("url", "")).strip()
            title = str(r.get("title", "")).strip()
            snippet = str(r.get("content", "")).strip()

            ext = by_url.get(url, {})
            extracted_text = (
                str(ext.get("raw_content", "")).strip()
                or str(ext.get("content", "")).strip()
                or ""
            )

            body = (extracted_text or snippet).strip()
            if not url:
                continue

            extracted_items.append(
                f"SOURCE {i}\nTITLE: {title or '(no title)'}\nURL: {url}\nEXCERPT:\n{body if body else '(no excerpt returned)'}\n"
            )
    else:
        for i, r in enumerate(results, start=1):
            url = str(r.get("url", "")).strip()
            title = str(r.get("title", "")).strip()
            snippet = str(r.get("content", "")).strip()
            if not url:
                continue
            extracted_items.append(
                f"SOURCE {i}\nTITLE: {title or '(no title)'}\nURL: {url}\nEXCERPT:\n{snippet}\n"
            )

    return {"query": query, "sources": extracted_items}

research_tool = Tool(
    function=tavily_research,
    name="tavily_research",
    description="Run Tavily search+extract for a query and return sources[].",
    filter_extraneous_inputs=True,
)

# ------------------------------------------------------
# Iteration Increment Tool
# ------------------------------------------------------
def increment_iteration(*, iteration: int = 0) -> Mapping[str, int]:
    return {"iteration": int(iteration) + 1}

iterator_tool = Tool(
    function=increment_iteration,
    name="increment_iteration",
    description="Increment the iteration count by 1.",
    filter_extraneous_inputs=True,
)

# ------------------------------------------------------
# Judge tool for MakerChecker approval
# ------------------------------------------------------
def judge_approved(*, revision_notes: str) -> bool:
    # query/sources intentionally unused; kept for schema compatibility with maker inputs
    return "<<APPROVED>>" in (revision_notes or "")

judge = Tool(
    judge_approved,
    name="approval_judge",
    description=f"Return True iff critic feedback contains <<APPROVED>>.",
    filter_extraneous_inputs=True,
)


if __name__ == "__main__":
    # Example Tool usage
    query = "What are the main benefits and risks of CRISPR gene editing in medicine?"
    result = research_tool.invoke({"query": query})
    import json
    print("Result:", json.dumps(result, indent=2))