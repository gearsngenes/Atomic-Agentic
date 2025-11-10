import sys
import logging
from pathlib import Path

# Project root
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from modules.Agents import Agent
from modules.ToolAgents import PlannerAgent
from modules.Plugins import MATH_TOOLS
from modules.LLMEngines import OpenAIEngine
from modules.Workflows import Selector, AgentFlow, ToolFlow
from modules.Tools import Tool

logging.getLogger().setLevel(logging.INFO)#logging.DEBUG)

LLM = OpenAIEngine(model="gpt-4o-mini")

# --- Branches (all must accept the same input schema: ["text"]) ---

# Paleontology answerer (text -> text)
def format_prompt(text:str) -> str:
    return (f"Only respond to the parts of the input request related to paleontolgy, and refuse to answer, "
            f"otherwise. The prompt:\n{text}")
            
prompt_tool = Tool(
    func = format_prompt,
    name = "format_prompt",
    description="formats the prompt with minor clarifications"
)
paleo_agent = Agent(
    name="paleo",  # Selection string must match this name
    description="Answers paleontology questions in depth.",
    llm_engine=LLM,
    role_prompt="You are a paleontology expert. Answer the user's question clearly and accurately.",
    context_enabled=True,
    pre_invoke=prompt_tool
)

# Math solver (text -> text)
math_agent = PlannerAgent(
    name="mathematician",  # Selection string must match this name
    description="Solves math problems and explains the steps.",
    llm_engine=LLM,
    pre_invoke=prompt_tool
)
math_agent.batch_register(MATH_TOOLS)

# Address formatter (text -> text)
def format_addr_from_text(text: str) -> str:
    """
    Expect a comma-separated address-like string, e.g.:
      '123, Fort Hamilton, New York, NY, 10364'
    Return a formatted single-line address.
    """
    parts = [p.strip() for p in text.split(",")]
    if len(parts) < 5:
        return text
    bld, street, city, state, zipc = parts[:5]
    return f"Bld: {bld}, St: {street}, City: {city}, State: {state} Zip: {zipc}"

addr_tool = Tool(
    name = "AddressFormatterText",  # Selection string must match this name
    func = format_addr_from_text,
    description = "Formats a comma-separated address-like string into a single-line address.",
)

# --- Judge (Tool). Must return the *branch name* as a string. ---

def route_selector(text: str) -> str:
    """
    Return the branch name to run based on the text.
    Must return one of: 'Agent1', 'Agent2', 'AddressFormatterText'
    """
    lowered = (text or "").lower()

    # contains digits and commas -> address
    if any(ch.isdigit() for ch in lowered) and "," in lowered:
        return "AddressFormatterText"
    # math-y keywords/symbols -> math
    math_tokens = ["+", "-", "*", "/", "power", "sqrt", "^", "sin", "cos", "tan", "log"]
    if any(tok in lowered for tok in math_tokens) or any(ch.isdigit() for ch in lowered):
        return "mathematician"
    # default -> paleo
    return "paleo"

judge_tool = Tool(
    name = "RouteSelector",
    func = route_selector,
    description = "Routes input text to the correct branch by returning its name.",
)

# --- Build selector ---
# Input schema is inferred from the judge (['text'] via the tool signature).
# Branches must have set-equivalent input schemas to the judge.
# Selector's output_schema/bundle_all are independent from branch outputs.
workflow = Selector(
    name="ConditionalWorkflowExample",
    description="Routes a text query to the appropriate branch based on judge decision.",
    branches=[paleo_agent, math_agent, addr_tool],
    judge=judge_tool,
    output_schema=["selection_result"],
    bundle_all=False,  # let base packaging project the 'text' value directly
)

# --- Demo inputs (dict-only invocation) ---
examples = [
    {"text": "Difference between tyrannosaurs and dromaeosaurs?"},
    {"text": "what is (twenty minus three) to the power of 3.14159?"},
    {"text": "123, Fort Hamilton, New York, NY, 10364"},
]

if __name__ == "__main__":
    for i, task in enumerate(examples, start=1):
        print(f"\n=== Example {i} ===")
        result = workflow.invoke(task)  # base .invoke handles validation + packaging
        # Do NOT rely on selector-specific checkpoints; print packaged result only.
        print("Input:", task["text"])
        print("Output:", result.get("selection_result"))
