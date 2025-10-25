import sys, logging,json
from pathlib import Path
from typing import Any
# Setting the root
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from modules.Agents import Agent
from modules.LLMEngines import OpenAIEngine
from modules.Workflows import ChainFlow
from modules.Tools import Tool


def json_parse(string: str):
    return json.loads(string)
tool1 = Tool("parser", json_parse)

def separate(output: dict):
    return list(output.keys()), sum(list(output.values()))
tool2 = Tool("separator", separate)

def format_out(a, b):
    return f"The value of 'A' is {a}, and the value of 'B' is {b}"
tool3 = Tool("formatter", format_out)

workflow = ChainFlow(
    name = "ChainFlow_Example",
    description ="A chain of thought workflow with three agents.",
    steps = [tool1,tool2,tool3],
    output_schema = ["final_string"]
)

print("CHECKPOINTS BEFORE INVOCATION: ")
print(f"{workflow.name} Checkpoint history:\n{workflow.checkpoints}")
for step in workflow.steps:
    print(f"Branch {step.name} Checkpoint history:\n{step.checkpoints}")


# invoke with dict input matching the first step's input_schema (key name is the
# parameter name of `json_parse`, which is 'string')
result1 = workflow.invoke({"string": "{\"a\":-1.74,\"b\":7,\"c\":4}"})
result2 = workflow.invoke({"string": "{\"a\":30,\"b\":1,\"c\":-3}"})
result3 = workflow.invoke({"string": "{\"a\":5.2,\"b\":3,\"c\":8}"})
print(f"\n\nINVOCATION RESULTS: {json.dumps([result1, result2, result3], indent=1)}")

print("\n\nCHECKPOINTS AFTER INVOCATION: ")
print(f"{workflow.name} Checkpoint history:\n{json.dumps(workflow.checkpoints, indent=1)}")
for step in workflow.steps:
    print(f"\nBranch {step.name} Checkpoint history:\n{json.dumps(step.checkpoints, indent=1)}")

print("\n\nLATEST RESULTS:")
print(f"{workflow.name} Last saved result:\n{json.dumps(workflow.latest_result, indent=1)}")
for step in workflow.steps:
    print(f"\nBranch {step.name} Last saved result:\n{json.dumps(step.latest_result, indent=1)}")
