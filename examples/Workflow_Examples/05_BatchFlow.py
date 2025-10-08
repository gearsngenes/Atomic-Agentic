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

'''
Define our agents & tools. These can be anything.
'''
LLM = OpenAIEngine(model="gpt-4o")

# a paleontology expert
agent1 = Agent(
	name = "Paleontologist",
	description = "An expert in paleontology",
	role_prompt = "You are a paleontology expert. Answer the user's questions about dinosaurs with detailed and accurate information.",
	llm_engine = LLM,
	context_enabled=True
)

# a mathematician
agent2 = PlannerAgent(
	name = "Mathematician",
	description = "A mathematician who excels in calculations",
	llm_engine = LLM,
	context_enabled=True,
	is_async=True
)
agent2.register(MathPlugin)

# a joke-telling agent
agent3 = Agent(
	name = "Jokester",
	description = "Tells jokes",
	role_prompt = "You are a jokester who tells short, friendly jokes.",
	llm_engine = LLM,
)

# a string length-comparing tool
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

'''
Define our batch-workflow, handling agents 1-3 and our compare-lengths tool

Note: with four workflows, we must have a result schema of either length one
or of length four. The example is set to length 4, meaning the results will be
structured like:

        {
            wf1_key: wf1_result,
            wf2_key: wf2_result,
            ...
        }

But if we set to a schema with only one output parameter, it will look like:

        {
            key1: {
                wf1_key: wf1_result,
                wf2_key: wf2_result,
                ...
            }
        }
'''
workflow = BatchFlow(
	name = "BatchFlowExample",
	description = "A batch workflow that runs agents/tools in parallel",
	branches=[agent1, agent2, agent3, tool1],
    result_schema=["paleo", "math", "joke", "comp"]
)

# result-format printing
def print_results(title, results):
	print('\n' + title)
	for source, result in results.items():
		print(f"{source}: {result}")
		print('---')



'''
Run our workflow with some example inputs. The example below shows
that the expected key-word parameter names are the same names as of
the names of the branch workflows within the batch flow.

This can be done as seen below, but they can also be assigned as
positional parameters, instead, or mix-and-match appropriately.

Note: be sure to structure your sub-arguments for each workflow
branch correctly, like grouping them as tuples if they have more
than one argument, or as dictionaries if they are meant to be
key-word arguments, etc.
'''
if __name__ == '__main__':
	res = workflow.invoke(
        Paleontologist = 'Tell me about Tyrannosaurus Rex',
        Mathematician = 'Compute 2**16 divided by three',
        Jokester = 'Tell a one-liner',
        Compare_Lengths = ('dogs','dolphins')
    )
	print_results('Example usage:', res)