import sys,json
from pathlib import Path
# Setting the root
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from modules.ToolAgents import PlannerAgent
from modules.LLMEngines import *

llm_engine = OpenAIEngine(model = "gpt-4o-mini")
planner = PlannerAgent(name="Test-Planner", description="Testing the prebuilt plugins", llm_engine=llm_engine)

def tool_1(seed):
    print("Tool 1 executed")
    return f"1) Result from tool 1. Seed was: {seed}"
def tool_2(t1_result):
    print("Tool 2 executed")
    return t1_result + "\n2) Result from tool 2"
def tool_3(t2_result):
    print("Tool 3 executed")
    return t2_result + "\n3) Result from tool 3"

planner.register(tool_1, description="This tool processes seed input and passes it to the next tool.")
planner.register(tool_2, description="This tool processes the result from tool 1.")
planner.register(tool_3, description="This tool finalizes the result based on tool 2's output.")

seed = 32

task_prompt = f"""
Call the tools in sequence: tool_1 with a seed value of {seed}, then tool_2 with tool_1's result, then tool_3 with tool_2's result.
After executing all tools, return the final result from tool_3.
"""

print("Generating plan...")
plan = planner.strategize(task_prompt)
print(f"Plan: {json.dumps(plan["steps"], indent=2)}")
print("\nExecuting plan...")
result = planner.execute(plan)
print(f"\nFinal Result: {result}")