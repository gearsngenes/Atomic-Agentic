import sys, json, logging
from pathlib import Path

# Setting the repo root on path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from modules.Agents import Agent
from modules.LLMEngines import OpenAIEngine
from modules.Workflows import AgentFlow

logging.getLogger().setLevel(logging.INFO)#logging.DEBUG)

print("=== AgentFlow examples (NEW uniform `invoke(inputs: dict)` contract) ===")

# -------------------------------------------------------------------
# Base Agent setup
# -------------------------------------------------------------------
my_agent = Agent(
    name="MyAgent",
    description="Example agent that echoes the input prompt.",
    role_prompt="You are a concise assistant. Reply in one short paragraph.",
    llm_engine=OpenAIEngine(model="gpt-4o-mini"),
    context_enabled=True,
)

# AgentFlow requires an explicit output schema; input schema is FIXED to ["prompt"]
workflow = AgentFlow(agent=my_agent, output_schema=["answer"])

# 1) Basic call
res1 = workflow.invoke({"prompt": "Hello! In one line, what is entropy?"})
print("1) ", res1)

# 2) Another call
res2 = workflow.invoke({"prompt": "Give a one-sentence definition of Bayes' theorem."})
print("2) ", res2)


# ===================================================================
# OPTIONAL DEMO: Custom key variant
# -------------------------------------------------------------------
# Sometimes you may want the input key to be called something else
# (e.g., "question" or "query") while keeping the same Agent behavior.
# AgentFlow in the library defaults to input_schema ["prompt"], but we
# can format the input using a specific key.

# when multiple keys are provided or needed
workflow_q = AgentFlow(agent=my_agent,
                       input_schema=["user", "session_id", "question", "temperature"],
                       output_schema=["answer"],)

# 3) Invoke with many keys; only "question" is consumed
res3 = workflow_q.invoke({
    "user": "alice",
    "session_id": "abc123",
    "question": "State the central limit theorem in one sentence.",
    "temperature": 0.0,
})
print("3) ", res3)


# 4) Another custom key set
workflow_query = AgentFlow(agent=my_agent,
                           input_schema=["query", "foo", "meta"],
                           output_schema=["answer"])
# Missing key values won't cause issues, provided it isn't empty, or if
# the agent is needing them to perform the task
res4 = workflow_query.invoke({
    "query": "What does O(n log n) usually refer to?"
})
print("4) ", res4)
