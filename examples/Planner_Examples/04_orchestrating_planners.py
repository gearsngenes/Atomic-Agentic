"""
Three Planner Agents
- batch_haiku_planner: writes multiple haiku using a haiku agent
- batch_math_planner: solves multiple math problems using math plugin
- super_planner: delegates tasks to the above planners
"""
import sys
from pathlib import Path
# Setting the root
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# ───────────────────────────  imports  ───────────────────────────
import time, logging

logging.basicConfig(level=logging.INFO)

# --- Atomic Agentic Imports ---
from modules.Agents import PlannerAgent, Agent
from modules.Plugins import MathPlugin
from modules.LLMNuclei import *

# define a global nucleus to give to each of our agents
nucleus = OpenAINucleus(model = "gpt-4o-mini")

# -------------------------------------
# ----- Build Batch Haiku Planner -----
# -------------------------------------

# Define Haiku Writing Agent
haiku_agent = Agent(
    name        = "HaikuWriter",
    nucleus     = nucleus,
    role_prompt = (
        "You are a master of writing haiku. Given a topic, write a "
        "3-line haiku about it, following a 5-syllable, 7-syllable, "
        "5-syllable rhythmic patter. Do not simply re-state the topic "
        "in the haiku, be creative! Example, given the topic "
        "'The morning sun', a valid haiku could be:\n\n"
        "A gol-den orb shines\n"
        "With red streaks a-cross the sky,\n"
        "Crisp, bright, mor-ning light\n\n"
        "Be sure to follow the 5-7-5 syllable structure, and "
        "hyphenate every multi-syllable word used."
    )
)

# Define Batch Haiku Planner
batch_haiku_planner = PlannerAgent(
    name    ="BatchHaikuPlanner",
    nucleus = nucleus,
    is_async= True,
)

# Register Haiku Writing Agent
batch_haiku_planner.register_agent(haiku_agent, description="Writes a haiku for a given prompt.")

# Define and Register Print Haiku Tool
def print_haiku(haiku_topic, haiku):
    print(f"---\n**{haiku_topic}**\n{haiku}\n---")
batch_haiku_planner.register_tool(
    "print_haiku",
    print_haiku,
    "Prints a haiku formatted with topic as the title."
)    


# ------------------------------------
# ----- Build Batch Math Planner -----
# ------------------------------------

# Define Batch Math Planner
batch_math_planner = PlannerAgent(
    name    = "BatchMathPlanner",
    nucleus = nucleus,
    is_async= True,
)

# Register Math Plugin and print method
batch_math_planner.register_plugin(MathPlugin())

def print_math_solution(problem, solution):
    print(f"Question: {problem}\nAnswer: {solution}")

batch_math_planner.register_tool(
    "print_solution",
    print_math_solution,
    "Prints the formatted math problem its solution."
)


# --------------------------------------
# ----- Build Batch Haiku Planner ------
# --------------------------------------

# Define Super Planner
super_planner = PlannerAgent(
    name    = "SuperPlanner",
    nucleus = nucleus,
    is_async= False,
)

# Register the two batch planners to the super planner
super_planner.register_agent(batch_haiku_planner, description="Handles batch haiku writing tasks.")
super_planner.register_agent(batch_math_planner, description="Handles batch math solving tasks.")


# -----------------------------------
# ----- Run Example Batch Tasks -----
# -----------------------------------

# Define an example batch of haiku prompts
haiku_prompts = [
    "A frog jumps in pond",
    "Autumn leaves falling",
    "Snow on mountain peak",
    "A roaring fire",
    "balsam flowers"
]

# Define an example batch of math problems
math_problems = [
    "12 * 8 + 5",
    "9 plus 7",
    "(3 + 4) * 2",
    "2^5-3",
    "maximum of [3, 7, 2, 9, 4]"
]

# --- 1. Batch Haiku Planner task ---
haiku_task = (
    f"For each topic in the provided list seen here:\n{haiku_prompts}\n"
    "Do the following:\n"
    "Write a haiku for the given topic. Then print the formatted result."
)

# --- 2. Batch Math Planner task ---
math_task = f"For each math problem in the provided list seen here:\n{math_problems}\n"
"Solve and PRINT(using print_solution) each math problem and its solution."


# --- 3. Super Planner Demo ---
super_task = (
    "You have two sub-tasks.\n"
    "TASK 1: Send the following string to BatchHaikuPlanner:\n"
    f"<task_string>\n{haiku_task}</task_string>\n"
    "TASK 2: For the BatchMathPlanner:\n"
    f"<task_string>{math_task}</task_string>\n"
    "Send each task_string EXACTLY as written to the appropriate planner."
)
start = time.time()
super_result = super_planner.invoke(super_task)
end = time.time()
print(f"Super Planner completed task in {end-start:.2f} seconds")