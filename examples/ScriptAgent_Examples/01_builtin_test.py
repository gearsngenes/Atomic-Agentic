"""01_builtin_test.py

Basic decomposition test -- originally mirrored PlanAct_Examples/
01_prebuilt_tools_test.py (registering EXPONENT_TOOLS/BASIC_MATH_TOOLS/
CONSOLE_TOOLS), renamed and repurposed once Python builtin calls landed
(Pass 5.5): no tool lists are registered at all now -- only a PI constant.
The whole three-part math task below is solvable via bare arithmetic
(power/sqrt via `**`) plus builtins (`print`) alone, testing that the
builtin-support grammar path genuinely replaces what those tool lists used
to provide for basic math, not just supplements them.

After the run, prints the reconstructed source via
ScriptAgentRecord.render_as_code() -- showing exactly what the agent
executed, not just its final answer.
"""
import logging
import math

from atomic_agentic.agents import ScriptAgent

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

# No tool lists registered -- the task below is solvable via bare
# arithmetic and Python builtins alone (see module docstring).

# Register the pi constant (value first, alias second -- ScriptAgent's
# register_constant signature is the reverse of v1 ToolAgent's).
agent.register_constant(math.pi, alias="PI", description="Hardcodes the math constant to 3.14...")

# ──────────────────────────  TASK  ─────────────────────────────
task_prompt = """
Answer ONLY these questions and call 'print' on their results as
'<question #>) <question>: <answer>' format:
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
