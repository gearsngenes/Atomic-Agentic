import sys, logging, json
from pathlib import Path
from typing import Any
# Setting the root
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from modules.Agents import Agent
from modules.LLMEngines import OpenAIEngine
from modules.ToolAgents import PlannerAgent
from modules.Workflows import BatchFlow
from modules.Tools import Tool
from modules.Plugins import MathPlugin
import logging

# logging.getLogger().setLevel(logging.INFO)

LLM = OpenAIEngine(model="gpt-4o")

agent1 = Agent(
	name = "Paleontologist",
	description = "An expert in paleontology",
	role_prompt = "You are a paleontology expert. Answer the user's questions about dinosaurs with detailed and accurate information.",
	llm_engine = LLM,
	context_enabled=True
)

agent2 = PlannerAgent(
	name = "Mathematician",
	description = "A mathematician who excels in calculations",
	llm_engine = LLM,
	context_enabled=True,
	is_async=True
)
agent2.register(MathPlugin)

agent3 = Agent(
	name = "Jokester",
	description = "Tells jokes and acts as a general assistant",
	role_prompt = "You are a jokester who tells short, friendly jokes.",
	llm_engine = LLM,
)

def compare_lengths(str1, str2):
	if len(str1) == len(str2):
		return f"Lengths of '{str1}' and '{str2}' are equal"
	longer = len(str1) > len(str2)
	if longer:
		return f"'{str1}' is longer than '{str2}'"
	return f"'{str2}' is longer than '{str1}'"

tool1 = Tool("Compare_Lengths",
			 compare_lengths,
			 description="Compares the lengths of two strings and returns an analysis of which is longer")

workflow = BatchFlow(
	name = "BatchFlowExample",
	description = "A batch workflow that runs agents/tools in parallel",
	branches=[agent1, agent2, agent3, tool1]
)

starred_list = [
	"Tell me about Brachiosaurus",
	"Compute sqrt(1024)",
    "Tell a limerick"
]
starred_kwargs = {
	"Compare_Lengths": ("short","longerstring")
}

def print_results(title, results):
	print('\n' + title)
	for source, result in results.items():
		print(f"{source}: {result}")
		print('---')


if __name__ == '__main__':
	# 1) positional-only
	res1 = workflow.invoke(
        "Tell me about the Stegosaurus",    # Paleontologist
        "Compute 5**4",                     # Mathematician
        "Give me a short joke",             # Jokester
        ("alpha", "beta")                   # compare_lengths expects two strings
    )
	print_results('Example 1: positional-only', res1)

	# 2) keyword-only
	res2 = workflow.invoke(
        Paleontologist = 'Tell me about Tyrannosaurus Rex',
        Mathematician = 'Compute 2**16 divided by three',
        Jokester = 'Tell a one-liner',
        Compare_Lengths = ('dogs','dolphins')
    )
	print_results('Example 2: keyword-only', res2)

	# 3) mix: positionals with kwargs overriding some branches
	res3 = workflow.invoke(
        "Tell me about Allosaurus",
        "Compute 7*7",
        Jokester = "Say something witty about ducks",
        Compare_Lengths = ("one","two")
    )
	print_results('Example 3: mixed positional + keyword (kwargs override)', res3)

	# 4) starred list and dict
	res4 = workflow.invoke(*starred_list, **starred_kwargs)
	print_results('Example 4: starred list and dict', res4)
