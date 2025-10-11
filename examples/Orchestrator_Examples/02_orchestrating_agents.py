import os, sys, logging
from pathlib import Path
from typing import Any
# Setting the root
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from modules.LLMEngines import OpenAIEngine
from modules.ToolAgents import OrchestratorAgent
from modules.Agents import Agent

logging.getLogger().setLevel(level=logging.INFO)

# Set up the LLM engine
llm = OpenAIEngine(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))

# Define the helper agents
builder = Agent(
    name="CodeBuilderAgent",
    description="Handles tasks related to generating code based on user requests and revisions",
    llm_engine=llm,
    role_prompt="""
    You are a senior software engineer who writes Python code for requested tasks.
    You return ONLY the code, without explanations or output.
    """.strip(),
    context_enabled=True
)

optimizer = Agent(
    name="CodeReviewer",
    description="Handles tasks related to reviewing code from the code-builder and provides revision notes",
    llm_engine=llm,
    role_prompt="""
    You are an expert Python performance analyst. When given a code snippet, thoroughly, and brutally evaluate its quality 
    in terms of accuracy, readability, and design, checking whether it follows SOLID principles. Once you've evaluated
    the code, return ONLY your revision SUGGESTIONS for the code builder to work on, otherwise return only "Approved".
    """.strip(),
    context_enabled=True
)

# Set up the orchestrator
orchestrator = OrchestratorAgent("AgenticOrchestrator",
                                 description="orchestrates calls between the code builder and code refiner",
                                 llm_engine= llm)

# Register both agents
orchestrator.register(builder)
orchestrator.register(optimizer)

# Run a dynamic task
task =  (
    "Write a Python-based class that can help construct a design-pattern/oop oriented design "
    "for agentic AI that also is platform agnostic (i.e. bedrock vs openai vs llama-cpp-python, etc.). "
    "Be sure that once the code is generated, its reviewed first, and then re-built with any improvements "
    "suggested by the code reviewer. Then, once the code-reviewer approves, return the latest code draft."
)

result = orchestrator.invoke(task)

print("\n=== Final Result ===\n")
print(result)
