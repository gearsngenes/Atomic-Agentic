# examples/Workflow_Examples/02_AgentFlow.py
from __future__ import annotations

import logging

from dotenv import load_dotenv

from atomic_agentic.engines.LLMEngines import OpenAIEngine
from atomic_agentic.agents.tool_agents import Agent
from atomic_agentic.tools import Tool
from atomic_agentic.workflows.workflows import AgentFlow

load_dotenv()
logging.basicConfig(level=logging.INFO)  # or DEBUG

print("=== AgentFlow examples (schema-driven dict inputs) ===")

# -------------------------------------------------------------------
# Base Agent setup
# Contract: Agent.invoke accepts a mapping; default pre_invoke requires {'prompt': str}.
# -------------------------------------------------------------------
my_agent = Agent(
    name="MyAgent",
    description="Concise assistant.",
    llm_engine=OpenAIEngine(model="gpt-4o-mini"),
    role_prompt="You are a concise assistant. Reply in one short paragraph.",
    context_enabled=True,
)

# AgentFlow packages the agent's raw output under the schema.
flow = AgentFlow(agent=my_agent, output_schema=["answer"])

# 1) Basic call (strict default: {'prompt': str})
res1 = flow.invoke({"prompt": "Hello! In one line, what is entropy?"})
print("1)", res1)

# 2) Another call
res2 = flow.invoke({"prompt": "Give a one-sentence definition of Bayes' theorem."})
print("2)", res2)

# -------------------------------------------------------------------
# Custom pre-invoke Tool: accept {'question': str} -> returns a prompt string.
# Demonstrates schema flexibility (Agent.arguments_map mirrors pre_invoke).
# -------------------------------------------------------------------
def question_to_prompt(*, question: str) -> str:
    return f"Answer in one sentence: {question}"


q_tool = Tool(
    function=question_to_prompt,
    name="question_to_prompt",
    namespace=my_agent.name,
    description="Convert {'question': str} into a prompt string.",
)

my_agent.pre_invoke = q_tool

# New AgentFlow instance (or reuse the old one; both will work).
flow_q = AgentFlow(agent=my_agent, output_schema=["answer"])

# 3) Invoke with custom key schema supported by pre-invoke Tool
res3 = flow_q.invoke({"question": "State the central limit theorem in one sentence."})
print("3)", res3)
