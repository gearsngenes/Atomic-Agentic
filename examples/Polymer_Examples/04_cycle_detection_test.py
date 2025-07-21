import sys
from pathlib import Path
import os
# Set root to repo root
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

"""
Example: Deliberate PolymerAgent cycle creation and error handling
"""
from modules.Agents import Agent, PolymerAgent
from modules.LLMNuclei import LLMNucleus

# Dummy nucleus for demonstration
class DummyNucleus(LLMNucleus):
    def invoke(self, messages):
        return "response"

# Create seed agents
nucleus = DummyNucleus()
agent_a = Agent("A", nucleus)
agent_b = Agent("B", nucleus)
agent_c = Agent("C", nucleus)

# Wrap in PolymerAgents
poly_a = PolymerAgent(agent_a)
poly_b = PolymerAgent(agent_b)
poly_c = PolymerAgent(agent_c)

# Link them linearly
poly_a.talks_to(poly_b)
poly_b.talks_to(poly_c)

# Attempt to create a cycle: link poly_c back to poly_a
try:
    poly_c.talks_to(poly_a)
except ValueError as e:
    print(f"Cycle Error: {e}")
else:
    print("No cycle detected (unexpected)")
