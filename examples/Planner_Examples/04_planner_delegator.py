"""
Three Planner Agents
- batch_haiku_planner: writes multiple haiku using a haiku agent
- batch_math_planner: solves multiple math problems using math tools
- super_planner: delegates tasks to the above planners
"""
from dotenv import load_dotenv
import time, logging
from atomic_agentic.Agents import Agent
from atomic_agentic.ToolAgents import PlannerAgent
from atomic_agentic.Plugins import MATH_TOOLS
from atomic_agentic.LLMEngines import OpenAIEngine

load_dotenv()

logging.getLogger().setLevel(level=logging.INFO)

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
    )
)

# ----- Batch Haiku Planner -----
batch_haiku_planner = PlannerAgent(
    name="BatchHaikuPlanner",
    description="Orchestrates calls to the Haiku Writer Agent and prints outputs",
    llm_engine=llm_engine,
    run_concurrent=True
)
batch_haiku_planner.register(haiku_agent)

def print_haiku(haiku_topic: str, haiku: str) -> None:
    print(f"---\n**{haiku_topic}**\n{haiku}\n---")

batch_haiku_planner.register(
    print_haiku,
    name="print_haiku",
    description="Print a haiku with its topic as the title."
)

# ----- Batch Math Planner -----
batch_math_planner = PlannerAgent(
    name="BatchMathPlanner",
    description="Handles tasks involving math problems and printing solutions",
    llm_engine=llm_engine,
    run_concurrent=False,
)
batch_math_planner.batch_register(MATH_TOOLS)

def print_math_solution(problem: str, solution: str) -> None:
    print(f"Question: {problem}\nAnswer: {solution}")

batch_math_planner.register(
    print_math_solution,
    name="print_math_solution",
    description="Print the math problem and its computed solution."
)

# ----- Super Planner (delegates to both planners) -----
super_planner = PlannerAgent(
    name="SuperPlanner",
    description="Planner that decomposes and delegates tasks to sub-planners",
    llm_engine=llm_engine,
    run_concurrent=True
)
super_planner.register(batch_haiku_planner)
super_planner.register(batch_math_planner)

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
    f"For each topic in the provided list seen here:\n{haiku_prompts}\n"
    "Do the following:\n"
    "Write a haiku for the given topic. Then print the formatted result."
)

math_task = f"Print each math problem and its solution. The problems are here:\n{math_problems}"

super_task = (
    "You have two sub-tasks.\n"
    "TASK 1: Send the following string to BatchHaikuPlanner:\n"
    f"<task_string>\n{haiku_task}</task_string>\n"
    "TASK 2: For the BatchMathPlanner:\n"
    f"<task_string>{math_task}</task_string>\n"
    "Send each task_string EXACTLY as written to the appropriate planner."
)

start = time.time()
super_result = super_planner.invoke({"prompt": super_task})
end = time.time()
print(f"Super Planner result: {super_result}\nCompleted task in {end-start:.2f} seconds")
