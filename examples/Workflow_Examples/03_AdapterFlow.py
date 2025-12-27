# examples/Workflow_Examples/03_AdapterFlow.py
from __future__ import annotations

import logging

from dotenv import load_dotenv

import atomic_agentic.workflows.Workflows as wfmod
from atomic_agentic.agents.toolagents import Agent
from atomic_agentic.engines.LLMEngines import OpenAIEngine
from atomic_agentic.Tools import Tool
from atomic_agentic.workflows.Workflows import (
    AdapterFlow,
    AgentFlow,
    BundlingPolicy,
    MappingPolicy,
    ToolFlow,
)

load_dotenv()
logging.basicConfig(level=logging.INFO)
print("=== AdapterFlow examples (generic wrapper boundary) ===")


# -------------------------------------------------------------------
# Example tool: simple primitive return
# -------------------------------------------------------------------
def str_len(*, s: str) -> int:
    return len(s)


len_tool = Tool(
    function=str_len,
    name="str_len",
    namespace="examples",
    description="Compute the length of s.",
)


# -------------------------------------------------------------------
# A) AdapterFlow wrapping a Workflow (ToolFlow) and preventing duplicate nesting
# -------------------------------------------------------------------
# ToolFlow returns {"length": <int>} (schema len == 1 => bundle)
wf_len = ToolFlow(
    tool=len_tool,
    output_schema=["length"],
    bundling_policy=BundlingPolicy.BUNDLE,
)

# AdapterFlow executes the component workflow -> gets an already-packaged mapping.
# Then it re-applies its own packaging boundary. The key behavior here:
# If the raw mapping already looks like {"length": 7} and adapter is bundling under "length",
# it avoids producing {"length": {"length": 7}}.
adapter_len = AdapterFlow(
    component=wf_len,
    output_schema=["length"],
    bundling_policy=BundlingPolicy.BUNDLE,
)

print("\n-- A) wrapping a Workflow (ToolFlow): single-key 'don't nest' behavior --")
print("component (ToolFlow) invoke:", wf_len.invoke({"s": "chatgpt"}))
print("adapter   (AdapterFlow) invoke:", adapter_len.invoke({"s": "chatgpt"}))

ckpt_a = adapter_len.latest_checkpoint
depth_a = (ckpt_a.metadata or {}).get(wfmod.ADAPTER_DEPTH) if ckpt_a else None
print("adapter depth:", depth_a)


# -------------------------------------------------------------------
# B) AdapterFlow is a mutable configuration boundary (mirrors config into component)
# -------------------------------------------------------------------
print("\n-- B) mirroring config down into the wrapped component --")
print("before: adapter.output_schema:", adapter_len.output_schema)
print("before: component.output_schema:", adapter_len.component.output_schema)

adapter_len.output_schema = ["out"]  # should also update the wrapped component schema
print("after:  adapter.output_schema:", adapter_len.output_schema)
print("after:  component.output_schema:", adapter_len.component.output_schema)

print("invoke with new schema:", adapter_len.invoke({"s": "chatgpt"}))


# -------------------------------------------------------------------
# C) AdapterFlow repackaging: mapping -> bundled mapping
# -------------------------------------------------------------------
def split_profile(*, raw: str) -> list[str]:
    # e.g., "Ada,37,CA" -> ["Ada", "37", "CA"]
    return raw.split(",")


profile_tool = Tool(
    function=split_profile,
    name="split_profile",
    namespace="examples",
    description="Comma-split raw into [name, age, state].",
)

# First boundary: ToolFlow UNBUNDLEs into mapping with 3 keys
wf_profile = ToolFlow(
    tool=profile_tool,
    output_schema=["name", "age", "state"],
    bundling_policy=BundlingPolicy.UNBUNDLE,
    mapping_policy=MappingPolicy.STRICT,
)

# Second boundary: AdapterFlow bundles that mapping into a single "profile" key
adapter_profile = AdapterFlow(
    component=wf_profile,
    output_schema=["profile"],
    bundling_policy=BundlingPolicy.BUNDLE,
)

print("\n-- C) re-packaging boundary (mapping -> bundled mapping) --")
print("component (ToolFlow) invoke:", wf_profile.invoke({"raw": "Ada,37,CA"}))
print("adapter   (AdapterFlow) invoke:", adapter_profile.invoke({"raw": "Ada,37,CA"}))


# -------------------------------------------------------------------
# D) Nested AdapterFlow layers (depth increments)
# -------------------------------------------------------------------
adapter_outer = AdapterFlow(
    component=adapter_profile,
    output_schema=["wrapped"],
    bundling_policy=BundlingPolicy.BUNDLE,
)

print("\n-- D) nested adapters increment adapter depth metadata --")
res_inner = adapter_profile.invoke({"raw": "Ada,37,CA"})
ckpt_inner = adapter_profile.latest_checkpoint
depth_inner = (ckpt_inner.metadata or {}).get(wfmod.ADAPTER_DEPTH) if ckpt_inner else None

res_outer = adapter_outer.invoke({"raw": "Ada,37,CA"})
ckpt_outer = adapter_outer.latest_checkpoint
depth_outer = (ckpt_outer.metadata or {}).get(wfmod.ADAPTER_DEPTH) if ckpt_outer else None

print("inner adapter result:", res_inner)
print("inner adapter depth:", depth_inner)
print("outer adapter result:", res_outer)
print("outer adapter depth:", depth_outer)


# -------------------------------------------------------------------
# E) AdapterFlow wrapping a Tool directly (Tool -> normalized to ToolFlow)
# -------------------------------------------------------------------
adapter_tool_direct = AdapterFlow(
    component=len_tool,                 # <-- Tool, not ToolFlow
    output_schema=["length"],
    bundling_policy=BundlingPolicy.BUNDLE,
)

print("\n-- E) wrapping a Tool directly (Tool -> ToolFlow normalization) --")
print("adapter (Tool direct) invoke:", adapter_tool_direct.invoke({"s": "mississippi"}))
print("normalized component type:", type(adapter_tool_direct.component).__name__)


# -------------------------------------------------------------------
# F) AdapterFlow wrapping an Agent directly (Agent -> normalized to AgentFlow)
# -------------------------------------------------------------------
my_agent = Agent(
    name="MyAgent",
    description="Concise assistant.",
    llm_engine=OpenAIEngine(model="gpt-4o-mini"),
    role_prompt="You are a concise assistant. Reply in one short paragraph.",
)

# Wrap the agent directly: AdapterFlow should normalize Agent -> AgentFlow internally.
adapter_agent_direct = AdapterFlow(
    component=my_agent,                 # <-- Agent, not AgentFlow
    output_schema=["answer"],
    bundling_policy=BundlingPolicy.BUNDLE,
)

print("\n-- F) wrapping an Agent directly (Agent -> AgentFlow normalization) --")
print("adapter (Agent direct) invoke:", adapter_agent_direct.invoke({"prompt": "What is a black hole?"}))
print("normalized component type:", type(adapter_agent_direct.component).__name__)

# Optional: show that the normalized component is indeed an AgentFlow
if isinstance(adapter_agent_direct.component, AgentFlow):
    print("normalized AgentFlow.agent.name:", adapter_agent_direct.component.agent.name)
