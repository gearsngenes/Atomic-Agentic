# examples/Workflow_Examples/01_AgentFlow.py
import logging
from atomic_agentic.Agents import Agent
from atomic_agentic.LLMEngines import OpenAIEngine
from atomic_agentic.Tools import Tool
from atomic_agentic.Workflows import AgentFlow
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)  # or DEBUG

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