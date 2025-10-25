import sys
from pathlib import Path

# Setting the repo root on path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from modules.Agents import Agent
from modules.LLMEngines import OpenAIEngine
from modules.Workflows import AgentFlow


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

# 3) Show latest checkpoint shape (inputs/result)
print("=== Latest checkpoint (AgentFlow w/ 'prompt') ===")
print(workflow.checkpoints[-1])


# ===================================================================
# OPTIONAL DEMO: Custom key variant
# -------------------------------------------------------------------
# Sometimes you may want the input key to be called something else
# (e.g., "question" or "query") while keeping the same Agent behavior.
# AgentFlow in the library defaults to input_schema ["prompt"], but we
# can format the input using a specific key.

# Use a different key name: "question"
workflow_q = AgentFlow(agent=my_agent, input_schema=["question"], output_schema=["answer"],)

# 4) Invoke with many keys; only "question" is consumed
res4 = workflow_q.invoke({
    "user": "alice",
    "session_id": "abc123",
    "question": "State the central limit theorem in one sentence.",
    "temperature": 0.0,
})
print("4) ", res4)
print("=== Latest checkpoint (AgentFlow w/ input key 'question') ===")
print(workflow_q.checkpoints[-1])


# 6) Another custom key: "query"
workflow_query = AgentFlow(agent=my_agent,  input_schema=["query"], output_schema=["answer"])
res5 = workflow_query.invoke({
    "query": "What does O(n log n) usually refer to?",
    "foo": "bar",   # ignored
    "meta": {"source": "unit-test"}  # ignored
})
print("5) ", res5)

print("=== Latest checkpoint (AgentFlow w/ input key 'query') ===")
print(workflow_query.checkpoints[-1])
