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
    result_schema = ["final_string"]
)

print("CHECKPOINTS BEFORE INVOCATION: ")
print(f"{workflow.name} Checkpoint history:\n{workflow.checkpoints}")
for branch in workflow._steps:
    print(f"Branch {branch.name} Checkpoint history:\n{branch.checkpoints}")


result1 = workflow.invoke('{"a":-1.74,"b":7,"c":4}')
result2 = workflow.invoke('{"a":30,"b":1,"c":-3}')
result3 = workflow.invoke('{"a":5.2,"b":3,"c":8}')
print(f"\n\nINVOCATION RESULTS: {json.dumps([result1, result2, result3], indent=1)}")

print("\n\nCHECKPOINTS AFTER INVOCATION: ")
print(f"{workflow.name} Checkpoint history:\n{json.dumps(workflow.checkpoints, indent=1)}")
for branch in workflow.steps:
    print(f"\nBranch {branch.name} Checkpoint history:\n{json.dumps(branch.checkpoints, indent=1)}")

print("\n\nLATEST RESULTS:")
print(f"{workflow.name} Last saved result:\n{json.dumps(workflow.latest_result, indent=1)}")
for branch in workflow.steps:
    print(f"\nBranch {branch.name} Last saved result:\n{json.dumps(branch.latest_result, indent=1)}")
