# examples/Workflow_Examples/01_AgentFlow.py
import sys, logging
from pathlib import Path

# Repo root on path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from modules.Agents import Agent
from modules.LLMEngines import OpenAIEngine
from modules.Tools import Tool
from modules.Workflows import AgentFlow

logging.getLogger().setLevel(logging.INFO)  # or DEBUG

print("=== AgentFlow examples (schema-driven dict inputs) ===")

# -------------------------------------------------------------------
# Base Agent setup
# Contract: Agent.invoke accepts a MAPPING; default pre-invoke requires {'prompt': str}.
# -------------------------------------------------------------------
my_agent = Agent(
    name="MyAgent",
    description="Concise assistant.",
    role_prompt="You are a concise assistant. Reply in one short paragraph.",
    llm_engine=OpenAIEngine(model="gpt-4o-mini"),
    context_enabled=True,
)

# AgentFlow requires explicit output_schema (single-key is common for text answers)
flow = AgentFlow(agent=my_agent, output_schema=["answer"])

# 1) Basic call (strict default: {'prompt': str})
res1 = flow.invoke({"prompt": "Hello! In one line, what is entropy?"})
print("1) ", res1)

# 2) Another call
res2 = flow.invoke({"prompt": "Give a one-sentence definition of Bayes' theorem."})
print("2) ", res2)

# -------------------------------------------------------------------
# Custom pre-invoke Tool: accept {'question': str} -> returns a prompt string.
# Demonstrates schema flexibility WITHOUT relying on hidden key remapping.
# -------------------------------------------------------------------
def question_to_prompt(*, question: str) -> str:
    return f"Answer in one sentence: {question}"

q_tool = Tool(func=question_to_prompt, name="question_to_prompt", description="{'question': str} -> prompt")
my_agent.pre_invoke = q_tool

flow_q = AgentFlow(agent=my_agent, output_schema=["answer"])

# 3) Invoke with custom key schema supported by pre-invoke Tool
res3 = flow_q.invoke({"question": "State the central limit theorem in one sentence."})
print("3) ", res3)