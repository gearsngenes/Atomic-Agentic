import sys
import logging
from pathlib import Path

# Project root
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from modules.Agents import Agent
from modules.ToolAgents import PlannerAgent
from modules.Plugins import MathPlugin
from modules.LLMEngines import OpenAIEngine
from modules.Workflows import Selector, AgentFlow, ToolFlow
from modules.Tools import Tool

logging.getLogger().setLevel(logging.INFO)

LLM = OpenAIEngine(model="gpt-4o")

# --- Branches (all must accept the same input schema: ["text"]) ---

# Paleontology answerer (text -> text)
paleo_agent = Agent(
    name="Agent1",  # Selection string must match this name
    description="Answers paleontology questions in depth.",
    llm_engine=LLM,
    role_prompt=(
        "You are a paleontology expert. Answer the user's question clearly and accurately."
    ),
    context_enabled=True,
)
paleo_branch = AgentFlow(
    paleo_agent,
    input_schema=["text"],
    output_schema=["text"],  # selector's output schema
)

# Math solver (text -> text)
math_agent = PlannerAgent(
    name="Agent2",  # Selection string must match this name
    description="Solves math problems and explains the steps.",
    llm_engine=LLM
)
math_agent.register(MathPlugin)
math_branch = AgentFlow(
    math_agent,
    input_schema=["text"],
    output_schema=["text"],
)

# Address formatter (text -> text). We create a text-based adapter so its input schema matches ["text"].
def format_addr_from_text(text: str) -> str:
    """
    Expect a comma-separated address-like string, e.g.:
      '123, Fort Hamilton, New York, NY, 10364'
    Return a formatted single-line address.
    """
    parts = [p.strip() for p in text.split(",")]
    if len(parts) < 5:
        # Keep it lenient: if under-specified, just echo back
        return text
    bld, street, city, state, zipc = parts[:5]
    return f"Address: {bld} {street}, {city}, {state} {zipc}"

addr_tool = Tool(
    "AddressFormatterText",               # Selection string must match this name
    format_addr_from_text,
    description="Formats a comma-separated address-like string into a single-line address.",
)
addr_branch = ToolFlow(
    addr_tool,
    output_schema=["text"],
)

# --- Judge (must be a Workflow). Single output key of length 1 that resolves to a string selection. ---

def route_selector(text: str) -> str:
    """
    Return the branch name to run based on the text.
    Must return one of: 'Agent1', 'Agent2', 'AddressFormatterText'
    """
    lowered = (text or "").lower()

    # naive routing:
    # - contains digits and commas -> address
    if any(ch.isdigit() for ch in lowered) and "," in lowered:
        return "AddressFormatterText"
    # - math-y keywords or symbols -> math
    if any(tok in lowered for tok in ["+", "-", "*", "/", "power", "sqrt", "^", "sin", "cos", "tan", "log"]) or any(
        ch in lowered for ch in "0123456789"
    ):
        return "Agent2"
    # default -> paleo
    return "Agent1"

judge_tool = Tool(
    "RouteSelector",
    route_selector,
    description="Routes input text to the correct branch by returning its name.",
)

# Wrap judge as a ToolFlow. Judge must output a single key; use ['selection'].
judge_flow = ToolFlow(judge_tool, output_schema=["selection"])

# --- Build selector (input schema inferred from judge; branches must match judge.input_schema & selector.output_schema) ---

workflow = Selector(
    name="ConditionalWorkflowExample",
    description="Routes a text query to the appropriate branch based on judge decision.",
    branches=[paleo_branch, math_branch, addr_branch],
    judge=judge_flow,
    output_schema=["text"],  # selector's packaged output schema
)

# --- Demo inputs (dict-only invocation) ---

input1 = {"text": "Difference between tyrannosaurs and dromaeosaurs?"}
input2 = {"text": "what is twenty minus three all to the power of 3.14159?"}
input3 = {"text": "123, Fort Hamilton, New York, NY, 10364"}

# Choose one:
task = input1

result = workflow.invoke(task)
print("\n--- RESULT ---")
print(workflow.checkpoints[-1]["selection"],",",result["text"])
