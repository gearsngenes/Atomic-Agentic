import sys, logging, json
from pathlib import Path
from typing import Any
# Setting the root
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from modules.Workflows import ToolFlow
from modules.Tools import Tool

def json_parse(string: str):
    return json.loads(string)
tool1 = Tool("parser", json_parse)
wf1 = ToolFlow(tool1)
print(wf1.invoke('{"a":5, "b": [1,3,5, -1]}'))

def separate(output: dict):
    return {"keys":list(output.keys()), "sum": sum(list(output.values()))}
tool2 = Tool("separator", separate)
wf2 = ToolFlow(tool2)
print(wf2.invoke({"john":1, "abby":3, "watson":-0.5}))

def printer(a, b):
    print(f"The value of 'A' is {a}, and the value of 'B' is {b}")
tool3 = Tool("Printer", printer)
wf3 = ToolFlow(tool3)
wf3.invoke(-3.1, {"a":1, "b":0, "c":-1})