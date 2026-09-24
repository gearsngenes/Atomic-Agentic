"""04_planner_delegator.py

Three planning agents demonstrating delegation:

- batch_haiku_planner: writes multiple haiku using a HaikuWriter Agent
- batch_math_planner: solves multiple math problems using math tools
- super_planner: delegates tasks to the above planners

Updated to use PlanActAgent (ReWOO-style: plan once, then execute).
"""
import logging

from dotenv import load_dotenv

from atomic_agentic.agents import BasicAgent, PlanActAgent
from atomic_agentic.tools.prebuilt import BASIC_MATH_TOOLS, EXPONENT_TOOLS
from atomic_agentic.llm import OpenAIEngine

load_dotenv()
logging.basicConfig(level=logging.INFO)

llm_engine = OpenAIEngine(model="gpt-4o-mini")

# ----- Haiku Writer Agent -----
haiku_agent = BasicAgent(
    name="HaikuWriter",
    namespace="examples",
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
    namespace="examples",
    description="Orchestrates calls to the Haiku Writer Agent and prints outputs",
    llm_engine=llm_engine,
)
batch_haiku_planner.register_tool(haiku_agent)
haiku_tool_id = haiku_agent.name  # effective id defaults to the bare agent name


def print_haiku(haiku_topic: str, haiku: str) -> None:
    print(f"---\n**{haiku_topic}**\n{haiku}\n---")


batch_haiku_planner.register_tool(
    print_haiku,
    description="Print a haiku with its topic as the title.",
)
print_haiku_tool_id = "print_haiku"  # effective id defaults to the bare function name

# ----- Batch Math PlanAct Agent -----
batch_math_planner = PlanActAgent(
    name="BatchMathPlanner",
    namespace="examples",
    description="Handles tasks involving math problems and printing solutions",
    llm_engine=llm_engine,
)
batch_math_planner.register_tools(BASIC_MATH_TOOLS)
batch_math_planner.register_tools(EXPONENT_TOOLS)

def print_math_solution(problem: str, solution: str) -> None:
    print(f"Question: {problem}\nAnswer: {solution}")


batch_math_planner.register_tool(
    print_math_solution,
    description="Print the math problem and its computed solution.",
)
print_math_tool_id = "print_math_solution"  # effective id defaults to the bare function name

# ----- Super Planner (delegates to both planners) -----
super_planner = PlanActAgent(
    name="SuperPlanner",
    namespace="examples",
    description="Planner that decomposes and delegates tasks to sub-planners",
    llm_engine=llm_engine,
)
super_planner.register_tool(batch_haiku_planner)
super_planner.register_tool(batch_math_planner)
haiku_planner_tool_id = batch_haiku_planner.name
math_planner_tool_id = batch_math_planner.name

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

super_task = f"""
    Send the following tasks to appropriate planners:
    
    To {haiku_planner_tool_id}, give the following task verbatim:
    {haiku_task}
    
    To {math_planner_tool_id}, give the following task verbatim:
    {math_task}
    
    You don't need to print or return anything, let the planners handle that.
    """

super_result = super_planner.invoke({"prompt": super_task})
print(f"Super Planner result: {super_result.result}\nCompleted task in {super_result.elapsed_s:.2f} seconds")
