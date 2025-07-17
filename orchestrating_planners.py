"""
Three Planner Agents
- batch_haiku_planner: writes multiple haiku using a haiku agent
- batch_math_planner: solves multiple math problems using math plugin
- super_planner: delegates tasks to the above planners
"""
import time
import logging

logging.basicConfig(level=logging.INFO)

# --- Atomic Agentic Imports ---
from atomic_agents.Agents import PlannerAgent, Agent
from atomic_agents.Plugins import ConsolePlugin, MathPlugin

# -------------------------------------
# ----- Build Batch Haiku Planner -----
# -------------------------------------

# Define Haiku Writing Agent
haiku_agent = Agent(
    name="HaikuWriter",
    role_prompt=(
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
    name="BatchHaikuPlanner", 
    is_async=True,
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
    name="BatchMathPlanner",
    is_async=True,
)

# Register Math and Console Plugins
batch_math_planner.register_plugin(MathPlugin())
batch_math_planner.register_plugin(ConsolePlugin())


# --------------------------------------
# ----- Build Batch Haiku Planner ------
# --------------------------------------

# Define Super Planner
super_planner = PlannerAgent(
    name="SuperPlanner",
    is_async=False,
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
    "Snow on mountain peak"
]

# Define an example batch of math problems
math_problems = [
    "12 * 8 + 5",
    "sqrt(144) + 7",
    "(3 + 4) * 2"
]

# --- 1. Batch Haiku Planner Demo ---
haiku_task = (
    f"For each topic in the provided list seen here:\n{haiku_prompts}\n"
    "Do the following:\n"
    "Write a haiku for the given topic. Then print the formatted result."
)
haiku_result = batch_haiku_planner.invoke(haiku_task)


# --- 2. Batch Math Planner Demo ---
math_task = f"Solve each of the following math problems and print the questions and their answers to the console:\n{math_problems}\n"
"When printing each answer, format it exactly as:\n"
"Question: <math problem>\nAnswer: <solution>\n"
math_result = batch_math_planner.invoke(math_task)


# --- 3. Super Planner Demo ---
super_task = (
    "You have two sub-tasks.\n"
    "TASK 1: Send the following string to BatchHaikuPlanner:\n"
    f"<task_string>\n{haiku_task}</task_string>\n"
    "TASK 2: For the BatchMathPlanner:\n"
    f"<task_string>{math_task}</task_string>\n"
    "Send each task_string to the appropriate planner EXACTLY as written."
)
start = time.time()
super_result = super_planner.invoke(super_task)
end = time.time()
print(f"Super Planner completed task in {end-start:.2f} seconds")