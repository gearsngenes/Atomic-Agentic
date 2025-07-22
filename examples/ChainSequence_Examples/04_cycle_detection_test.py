import sys
from pathlib import Path
import os
# Set root to repo root
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

"""
Example: Deliberate ChainSequenceAgent cycle creation and error handling
"""
from modules.Agents import Agent, ChainSequenceAgent
from modules.LLMEngines import LLMEngine

# Dummy llm engine for demonstration
class DummyEngine(LLMEngine):
    def invoke(self, messages):
        return "response"

# Create seed agents
llm_engine = DummyEngine()
agent_a = Agent("A", llm_engine)
agent_b = Agent("B", llm_engine)
agent_c = Agent("C", llm_engine)

# Wrap in PolymerAgents
poly_a = ChainSequenceAgent(agent_a)
poly_b = ChainSequenceAgent(agent_b)
poly_c = ChainSequenceAgent(agent_c)

# Link them linearly
poly_a.talks_to(poly_b)
poly_b.talks_to(poly_c)

# Attempt to create a cycle: link poly_c back to poly_a
try:
    poly_c.talks_to(poly_a)
except ValueError as e:
    print(f"Error: {e}")
else:
    print("No cycle detected (unexpected)")
