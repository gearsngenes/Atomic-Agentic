import sys, logging, json
from pathlib import Path
from typing import Any
# Setting the root
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from modules.Workflows import ToolFlow
from modules.Tools import Tool


# ===========================================================
def double_json_keys(num, string, map):
    return num * 2, json.loads(string), map.keys()
tool1 = Tool("double_and_json", double_json_keys)
wf = ToolFlow(tool1)
print("===Scalar inputs only===")
print(wf.invoke(5, '{"a":5, "b": [1,3,5, -1]}', {"a":1, "!":-4, "007": 4}))

print("\n\n===Keyword inputs only===")
print(wf.invoke(num = 13, string = '{"a":"cats & dogs", "b": [1,3,5, -1]}', map = {"3":1, "hello":-4, "#$": 4}))


# ===========================================================
def multiply(a,b):
    return a*b
tool2 = Tool("multiply", multiply)

print("\n\n===Lists/tuples as positional inputs===")
wf = ToolFlow(tool2)
try:
    print("(.invoke([5,10])): ",wf.invoke([5,10]))
except Exception as e:
    print("ERROR: failed to pass in raw [5,10] as input, since it is "
          f"viewed as a single parameter, resulting in the given error:\n```{e}```")
print("\n**Correct way to pass list/tuple in as an *args**:\n(.invoke(*[5,10])): ", wf.invoke(*[5, 10]))


# ===========================================================
def join(a, b, c):
    return "Joined result: " + ", ".join([str(a),str(b),str(c)])
tool3 = Tool("join", join)

print("\n\n===Dictionary as keywords-only inputs===")
wf = ToolFlow(tool3)
try:
    print(wf.invoke({"a":1, "b":3, "c":-0.5}))
except Exception as e:
    print(f"ERROR: failed to pass in raw {{'a':1, 'b':3, 'c':-0.5}} as input, since it is "
          f"viewed as a single parameter, resulting in the given error:\n```{e}```")
    print("\n**Correct invoke(**{...})**:\n",wf.invoke(**{"a":1, "b":3, "c":-0.5}))

# ===========================================================
def format_menu(menu: dict):
    return "MENU:\n"+"\n".join([f"- {k}: {v}" for k,v in menu.items()])
tool4 = Tool("format_menu", format_menu)
print("\n\n===Passing in argument dictionary===")
wf = ToolFlow(tool4)
print(wf.invoke({"biscuit":"A dry cookie", "salmon":"baked with pepper & salt seasoning"}))
print("\n**Correct Keyword way: .invoke(menu = {...})**:\n", wf.invoke(menu = {"biscuit":"A dry cookie", "salmon":"baked with pepper & salt seasoning"}))

# ===========================================================
print("\n===Mix-and-match positional & keyword example===")
def formatter(a:Any, b:dict):
    return f"The value of 'A' is {a}, and the sum of the values of 'B' is {sum(list(b.values()))}"
tool5 = Tool("Printer", formatter)
wf = ToolFlow(tool5)
print(wf.invoke(3.1, b={"x":1, "y":0, "z":-1}))

# ===========================================================
print("\n===Default-Value Examples===")
def string_any(a, b = 2, c = "hello", f = "john"):
    return f"a: {a}, b: {b}, c: {c}, f: {f}"
tool6 = Tool("string-any", string_any)
wf = ToolFlow(tool6)
print(wf.invoke("al"))