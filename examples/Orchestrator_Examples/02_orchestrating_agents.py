import os, sys, logging
from pathlib import Path
from typing import Any
# Setting the root
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from modules.LLMEngines import OpenAIEngine
from modules.OrchestratorAgents import OrchestratorAgent
from modules.Agents import Agent

logging.basicConfig(level=logging.INFO)

# Set up the LLM engine
llm = OpenAIEngine(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))

# Define the helper agents
builder = Agent(
    name="CodeBuilderAgent",
    description="Generates code based on user requests",
    llm_engine=llm,
    role_prompt="""
You are a senior software engineer who writes clean, working Python code for requested tasks.
You return ONLY the code, without explanations or output.
""".strip()
)

optimizer = Agent(
    name="CodeOptimizerAgent",
    description="Reviews and refines code to optimize it for simplicity and effectiveness",
    llm_engine=llm,
    role_prompt="""
You are an expert Python performance analyst. When given a code snippet, you rewrite it for efficiency,
readability, and maintainability. You return only the revised code, without comments or extra explanation.
""".strip()
)

# Set up the orchestrator
orchestrator = OrchestratorAgent("AgenticOrchestrator",
                                 description="orchestrates calls between the code builder and code refiner",
                                 llm_engine= llm,
                                 allow_agentic=True)

# Register both agents
orchestrator.register(builder)
orchestrator.register(optimizer)

# Run a dynamic task
task =  (
    "Write a Python-based class that can help construct a design-pattern/oop oriented design "
    "for agentic AI that also is platform agnostic (i.e. bedrock vs openai vs llama-cpp-python, etc.). "
    "Return the final draft once it's done. Prioritize OOP-design, simplified interfaces that are easy "
    "to expand on in subclasses and consistent to inherit. It should require minimal amounts of if-else "
    "statements for configuration. Be sure to use helper classes, as well."
)

result = orchestrator.invoke(task)

print("\n=== Final Result ===\n")
print(result)
