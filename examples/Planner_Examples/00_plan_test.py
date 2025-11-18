from dotenv import load_dotenv
import os
from atomic_agentic.ToolAgents import PlannerAgent
from atomic_agentic.LLMEngines import OpenAIEngine

load_dotenv()

# LLM engine
llm_engine = OpenAIEngine(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))

# Planner agent (context optional—showing disabled here)
planner = PlannerAgent(
    name="Test-Planner",
    description="Testing simple, sequential planning over local python tools.",
    llm_engine=llm_engine,
    context_enabled=False,
)

# Simple local tools
def tool_1(seed: int):
    print("Tool 1 executed")
    return f"1) Result from tool_1. Seed was: {seed}"

def tool_2(t1_result: str):
    print("Tool 2 executed")
    return t1_result + "\n2) Result from tool_2"

def tool_3(t2_result: str):
    print("Tool 3 executed")
    return t2_result + "\n3) Result from tool_3"

# Register tools (callables require name & description)
planner.register(tool_1, name="tool_1", description="Processes seed input and passes it to the next tool.")
planner.register(tool_2, name="tool_2", description="Processes the result from tool_1.")
planner.register(tool_3, name="tool_3", description="Finalizes the result based on tool_2 output.")

seed = 32

task_prompt = (
    f"Call the tools in sequence: tool_1 with a seed value of {seed}, "
    f"then tool_2 with tool_1's result, then tool_3 with tool_2's result. "
    f"After executing all tools, return the final result from tool_3."
)

print("Invoking planner…")
final = planner.invoke({"prompt": task_prompt})
print("\nFinal Result:\n", final)
