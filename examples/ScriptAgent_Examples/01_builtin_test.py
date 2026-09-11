"""01_prebuilt_tools_test.py

Basic decomposition test -- mirrors PlanAct_Examples/01_prebuilt_tools_test.py
(formerly "plugins test"; renamed since the prebuilt tool lists were never
called "plugins"), rebuilt on ScriptAgent instead of PlanActAgent.

Registers the same three prebuilt tool lists and a PI constant, then hands
the agent a three-part math task it must decompose into a straight-line
script on its own. After the run, prints the reconstructed source via
ScriptAgentRecord.render_as_code() -- showing exactly what the agent
executed, not just its final answer.
"""
import logging
import math

from atomic_agentic.agents import ScriptAgent
from atomic_agentic.tools.prebuilt import EXPONENT_TOOLS, BASIC_MATH_TOOLS, CONSOLE_TOOLS

from shared_engine import llm_engine

logging.basicConfig(level=logging.INFO)

print("Testing prebuilt tools with ScriptAgent's one-shot decomposition")

# ──────────────────────────  SET-UP  ───────────────────────────
agent = ScriptAgent(
    name="Test_ScriptAgent",
    namespace="examples",
    description="Testing the prebuilt tool lists with one-shot planning + execution.",
    llm_engine=llm_engine,
    context_enabled=True,
    planning_rounds_limit=2,
)

# Register tool lists
# agent.register_tools(EXPONENT_TOOLS)
# agent.register_tools(BASIC_MATH_TOOLS)
# agent.register_tools(CONSOLE_TOOLS)

# Register the pi constant (value first, alias second -- ScriptAgent's
# register_constant signature is the reverse of v1 ToolAgent's).
agent.register_constant(math.pi, alias="PI", description="Hardcodes the math constant to 3.14...")

# ──────────────────────────  TASK  ─────────────────────────────
task_prompt = """
Answer ONLY these questions and call 'print' on their results as '#) <question>: <answer>' format:
1) Compute the area of a circle with a radius of 5.
2) Compute the length of the hypotenuse of a triangle with legs a=3, b=4
3) Compute the volume of a cylinder with radius of 2 and height of 10.
"""

print("\n⇢ Executing math demo …")
result = agent.invoke({"prompt": task_prompt})

print("\n=== FINAL AGENT RESULT ===")
print(result.result)

record = agent.get_conversation()[-1]
print("\n=== EXECUTED SCRIPT ===")
print(record.render_as_code())
