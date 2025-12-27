"""04_planner_delegator.py

Three planning agents demonstrating delegation:

- batch_haiku_planner: writes multiple haiku using a HaikuWriter Agent
- batch_math_planner: solves multiple math problems using math tools
- super_planner: delegates tasks to the above planners

Updated to use PlanActAgent (ReWOO-style: plan once, then execute).
"""
import logging
import time

from dotenv import load_dotenv

from atomic_agentic.agents.toolagents import Agent, PlanActAgent
from atomic_agentic.tools.Plugins import MATH_TOOLS
from atomic_agentic.engines.LLMEngines import OpenAIEngine

load_dotenv()
logging.basicConfig(level=logging.INFO)

llm_engine = OpenAIEngine(model="gpt-4o-mini")

# ----- Haiku Writer Agent -----
haiku_agent = Agent(
    name="HaikuWriter",
    description="Writes Haiku when given a topic.",
    llm_engine=llm_engine,
    role_prompt=(
        "You are a master of writing haiku. Given a topic, write a "
        "3-line haiku about it, following a 5-7-5 syllable structure. "
        "Be creative; do not just restate the topic."
    ),
)

# ----- Batch Haiku PlanAct Agent -----
batch_haiku_planner = PlanActAgent(
    name="BatchHaikuPlanner",
    description="Orchestrates calls to the Haiku Writer Agent and prints outputs",
    llm_engine=llm_engine,
    run_concurrent=True,
)
haiku_tool_id = batch_haiku_planner.register(haiku_agent)[0]


def print_haiku(haiku_topic: str, haiku: str) -> None:
    print(f"---\n**{haiku_topic}**\n{haiku}\n---")


print_haiku_tool_id = batch_haiku_planner.register(
    print_haiku,
    name="print_haiku",
    description="Print a haiku with its topic as the title.",
)[0]

# ----- Batch Math PlanAct Agent -----
batch_math_planner = PlanActAgent(
    name="BatchMathPlanner",
    description="Handles tasks involving math problems and printing solutions",
    llm_engine=llm_engine,
    run_concurrent=False,
)
batch_math_planner.batch_register(MATH_TOOLS)


def print_math_solution(problem: str, solution: str) -> None:
    print(f"Question: {problem}\nAnswer: {solution}")


print_math_tool_id = batch_math_planner.register(
    print_math_solution,
    name="print_math_solution",
    description="Print the math problem and its computed solution.",
)[0]

# ----- Super Planner (delegates to both planners) -----
super_planner = PlanActAgent(
    name="SuperPlanner",
    description="Planner that decomposes and delegates tasks to sub-planners",
    llm_engine=llm_engine,
    run_concurrent=True,
)
haiku_planner_tool_id = super_planner.register(batch_haiku_planner)[0]
math_planner_tool_id = super_planner.register(batch_math_planner)[0]

# ----- Run Example Batch Tasks -----
haiku_prompts = [
    "A frog jumps in pond",
    "Autumn leaves falling",
    "Snow on mountain peak",
    "A roaring fire",
    "balsam flowers",
]

math_problems = [
    "12 * 8 + 5",
    "9 plus 7",
    "(3 + 4) * 2",
    "2^5-3",
    "maximum of [3, 7, 2, 9, 4]",
]

haiku_task = (
    f"For each topic in the provided list seen here:\n{haiku_prompts}\n\n"
    "Do the following for each topic:\n"
    f"- Use the haiku writer tool ({haiku_tool_id}) to write a haiku for the given topic.\n"
    f"- Then use the printer tool ({print_haiku_tool_id}) to print the formatted result.\n"
)

math_task = (
    f"Print each math problem and its solution. The problems are here:\n{math_problems}\n\n"
    f"Use any of the available math tools to compute the answer, then call {print_math_tool_id} to print it."
)

super_task = (
    "You have two sub-tasks.\n\n"
    f"TASK 1 (haiku): Call the following planner tool ONCE: {haiku_planner_tool_id}\n"
    "with args {'prompt': <task_string>} where task_string is:\n"
    f"<task_string>\n{haiku_task}\n</task_string>\n\n"
    f"TASK 2 (math): Call the following planner tool ONCE: {math_planner_tool_id}\n"
    "with args {'prompt': <task_string>} where task_string is:\n"
    f"<task_string>\n{math_task}\n</task_string>\n\n"
    "Send each task_string EXACTLY as written to the appropriate planner tool."
)

start = time.time()
super_result = super_planner.invoke({"prompt": super_task})
end = time.time()
print(f"Super Planner result: {super_result}\nCompleted task in {end - start:.2f} seconds")
