import sys, logging
from pathlib import Path
from typing import Any
# Setting the root
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from modules.Workflows import ToolWorkflow
from modules.Tools import Tool

logging.getLogger().setLevel(logging.INFO)

def test_1(a,b,c): print(f"A: {a}, B: {b}, C:{c}")
tool1 = Tool("Hello", test_1)
wf1 = ToolWorkflow(tool1)

def test_2(nums, name="Bob"): print(f"Sum: {sum(nums)}, Name: {name}")
tool2 = Tool("Hello", test_2)
wf2 = ToolWorkflow(tool2)

def test_3(people:dict):
    for person, age in people.items():
        print(f"Person: {person}, Age: {age}")
tool3 = Tool("Hello", test_3)
wf3 = ToolWorkflow(tool3)

wf1.invoke(1,2,3)
wf2.invoke([1,2,3,4])
wf3.invoke({"Al":24, "Joe":14, "Casey":64})
wf1.invoke({"a":2,"b":4,"c":6})