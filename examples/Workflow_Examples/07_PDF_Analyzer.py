import os
import re
import io
import json
from typing import Dict, Any, List, Optional
import os, sys, time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
# ---- PyMuPDF import: prefer modern name; fall back to legacy 'fitz' ----
try:
    import pymupdf  # modern import name (PyMuPDF >= 1.24)  # noqa: F401
except Exception:  # pragma: no cover
    import fitz as pymupdf  # type: ignore
    print("[WARN] Using legacy 'fitz' alias. Prefer `pip install -U pymupdf` and `import pymupdf`.")

# ---- Atomic-Agentic imports (adjust if your local path differs) ----
from modules.Tools import Tool
from modules.Agents import Agent
from modules.LLMEngines import OpenAIEngine
from modules.Workflows import ToolFlow, AgentFlow, MapFlow, ChainFlow

# -------- Config: no CLI needed; try ./test.pdf then assets path --------
SAMPLE_PDF = "./examples/Workflow_Examples/test.pdf"
DEFAULT_MODEL = "gpt-4o-mini"


# -----------------------
# PDF extractor functions
# -----------------------
def pdf_to_text(*, pdf_path: str, max_chars: int = 24000) -> Dict[str, str]:
    """Extract running text from a PDF (born-digital preferred)."""
    if not os.path.isfile(pdf_path):
        return {"report_text": ""}
    doc = pymupdf.open(pdf_path)
    chunks: List[str] = []
    for page in doc:
        try:
            # sort=True tends to produce a more natural reading order
            chunks.append(page.get_text("text", sort=True))
        except Exception:
            continue
    doc.close()
    text = "\n".join(chunks).strip()
    if max_chars and len(text) > max_chars:
        text = text[:max_chars] + "\n[Truncated]"
    return {"report_text": text}


def pdf_tables_to_tsv(*, pdf_path: str, max_tables: int = 5) -> Dict[str, str]:
    """Detect tables and serialize first N to TSV using built-in find_tables()."""
    if not os.path.isfile(pdf_path):
        return {"tables_tsv": ""}
    doc = pymupdf.open(pdf_path)
    out = io.StringIO()
    count = 0
    for page in doc:
        try:
            tsets = page.find_tables()
        except Exception:
            continue
        if not tsets or not getattr(tsets, "tables", None):
            continue
        for t in tsets.tables:
            if count >= max_tables:
                break
            mat = t.extract()  # list[list[str]]
            if mat:
                for row in mat:
                    out.write("\t".join((str(c) if c is not None else "").strip() for c in row) + "\n")
                out.write("\n")
                count += 1
        if count >= max_tables:
            break
    doc.close()
    return {"tables_tsv": out.getvalue().strip()}


_CAPTION_RX = re.compile(r"^(?:Figure|Fig\.)\s*\d*[:\-\.]?\s*(.+)$", re.IGNORECASE)


def pdf_figure_captions(*, pdf_path: str, max_chars: int = 4000) -> Dict[str, str]:
    """Heuristically collect lines that look like figure captions."""
    if not os.path.isfile(pdf_path):
        return {"figure_captions": ""}
    doc = pymupdf.open(pdf_path)
    caps: List[str] = []
    for page in doc:
        try:
            for line in page.get_text("text", sort=True).splitlines():
                s = line.strip()
                if _CAPTION_RX.match(s):
                    caps.append(s)
        except Exception:
            continue
    doc.close()
    text = "\n".join(caps)
    if max_chars and len(text) > max_chars:
        text = text[:max_chars] + "\n[Truncated]"
    return {"figure_captions": text}


# -----------------------
# Agent pre-invoke builder (MUST return str)
# -----------------------
def build_summary_prompt(
    *,
    report_text: str = "",
    tables_tsv: str = "",
    figure_captions: str = "",
    summary_style: Optional[str] = "executive-brief",
) -> str:
    """Assemble one prompt string for the Agent (pre-invoke must return str)."""
    # Defensive coercions: Tools / Workflows may pass None for omitted optionals.
    summary_style = summary_style or "executive-brief"
    report_text = report_text or ""
    tables_tsv = tables_tsv or ""
    figure_captions = figure_captions or ""

    blocks: List[str] = []
    blocks.append(
        "# TASK\n"
        "Summarize the document’s key findings and risks.\n"
        "Output:\n"
        "- <=120-word paragraph\n"
        "- 3–5 bullet highlights\n"
        "- Mention trends implied by any tables\n"
    )
    blocks.append(f"# STYLE\n{summary_style}")
    if report_text:
        blocks.append("# REPORT_TEXT\n" + report_text)
    if tables_tsv:
        blocks.append("# TABLES_TSV\n" + tables_tsv)
    if figure_captions:
        blocks.append("# FIGURE_CAPTIONS\n" + figure_captions)
    final_prompt = "\n\n".join(blocks).strip()
    print("~~~FORMATED_PROMPT~~~",final_prompt,"\n~~~~~~~~~~~~~~~~~~~~")
    return final_prompt


# -----------------------
# Wiring: Tools → Flows → Agent
# -----------------------
def make_workflow(model: str = DEFAULT_MODEL) -> ChainFlow:
    # 1) Tools
    t_text = Tool(pdf_to_text, name="pdf_to_text", description="Extract running text from PDF")
    t_tables = Tool(pdf_tables_to_tsv, name="pdf_tables_to_tsv", description="Detect tables and export as TSV")
    t_caps = Tool(pdf_figure_captions, name="pdf_figure_captions", description="Collect figure-like caption lines")
    pre_prompt = Tool(
        build_summary_prompt,
        name="build_summary_prompt",
        description="Merge extracted fields into a single prompt string",
    )

    # 2) Flows: parallel prehooks → flattened mapping
    f_text = ToolFlow(t_text, name="extract_text")
    f_tables = ToolFlow(t_tables, name="extract_tables")
    f_caps = ToolFlow(t_caps, name="extract_captions")
    pre = MapFlow(
        name="PDF_PreHooks",
        description="Parallel PDF extractors → flattened mapping for agent pre-invoke",
        branches=[f_text, f_tables, f_caps],
        flatten=True,
        output_schema=["__wf_result__"],
        bundle_all=True,
    )

    # 3) Agent with pre-invoke Tool
    engine = OpenAIEngine(model=model)
    agent = Agent(
        name="ReportSummarizer",
        description="Turns extracted PDF text/tables/captions into an executive summary.",
        llm_engine=engine,
        role_prompt=(
            "You are a precise executive analyst. Be concise, factual, and avoid speculation. "
            "Prefer bullet points with concrete numbers / trends when available. "
            "Be sure to highlight specific trends/figures from specific tables and noteworthy figures"
        ),
        pre_invoke=pre_prompt,  # MUST return str
        context_enabled=False,
        history_window=16,
    )
    f_agent = AgentFlow(agent, name="Summarizer")

    # 4) Chain: prehooks → agent
    flow = ChainFlow(
        name="PDF_ExecSummary_Flow",
        description="Extract (text/tables/captions) → Summarize",
        steps=[pre, f_agent],
        output_schema=["summary"],
        bundle_all=True,
    )
    return flow


def run() -> Dict[str, Any]:
    """Run the flow with the hardcoded SAMPLE_PDF."""
    print("Reading:", SAMPLE_PDF)
    flow = make_workflow(model=DEFAULT_MODEL)

    # MapFlow input shape: {branch_name: {...}}
    inputs = {
        "extract_text": {"pdf_path": SAMPLE_PDF, "max_chars": 3000},
        "extract_tables": {"pdf_path": SAMPLE_PDF, "max_tables": 5},
        "extract_captions": {"pdf_path": SAMPLE_PDF, "max_chars": 4000},
    }
    return flow.invoke(inputs)


if __name__ == "__main__":
    out = run()
    print(out["summary"])
