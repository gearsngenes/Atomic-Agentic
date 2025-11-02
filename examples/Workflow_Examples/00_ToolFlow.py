import sys, logging, json
from pathlib import Path
# Setting the repo root on path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from typing import Any
from modules.Workflows import ToolFlow
from modules.Tools import Tool


logging.getLogger().setLevel(logging.INFO)#logging.DEBUG)

print("=== ToolFlow examples (NEW uniform `invoke(inputs: dict)` contract) ===")

# ===========================================================
# Example 1: multiple inputs → tuple result mapped by output_schema
def double_json_keys(num, string, map_):
    return num * 2, json.loads(string), list(map_.keys())

tool1 = Tool("double_and_json", double_json_keys)
wf1 = ToolFlow(tool=tool1, output_schema=["doubled", "json", "keys"], bundle_all=False)

print("\n[1] Mixed inputs (all via single dict):")
out1 = wf1.invoke({
    "num": 5,
    "string": '{"a":5, "b":[1,3,5,-1]}',
    "map_": {"a":1, "!":-4, "007":4},
})
print(out1)


# ===========================================================
# Example 2: simple numeric op → scalar under first output field
def multiply(a, b):
    return a * b

tool2 = Tool("multiply", multiply)
wf2 = ToolFlow(tool=tool2, output_schema=["product"])

print("\n[2] Multiply (dict inputs only):")
out2 = wf2.invoke({"a": 13, "b": 7})
print(out2)


# ===========================================================
# Example 3: three-arg join → single string
def join(a, b, c):
    return "Joined result: " + ", ".join([str(a), str(b), str(c)])

tool3 = Tool("join", join)
wf3 = ToolFlow(tool=tool3, output_schema=["text"])

print("\n[3] Join three fields (dict inputs only):")
out3 = wf3.invoke({"a": 1, "b": 3, "c": -0.5})
print(out3)


# ===========================================================
# Example 4: tool that expects a single dict param
def format_menu(menu: dict):
    return "MENU:\n" + "\n".join([f"- {k}: {v}" for k, v in menu.items()])

tool4 = Tool("format_menu", format_menu)
wf4 = ToolFlow(tool=tool4, output_schema=["text"])

print("\n[4] Single-dict param (still passed via `inputs`):")
out4 = wf4.invoke({
    "menu": {
        "biscuit": "A dry cookie",
        "salmon": "baked with pepper & salt seasoning",
    }
})
print(out4)


# ===========================================================
# Example 5: mix types — Any + dict
def formatter(a: Any, b: dict):
    return f"The value of 'A' is {a}, and the sum of the values of 'B' is {sum(list(b.values()))}"

tool5 = Tool("printer", formatter)
wf5 = ToolFlow(tool=tool5, output_schema=["text"])

print("\n[5] Mixed types via dict:")
out5 = wf5.invoke({"a": 3.1, "b": {"x": 1, "y": 0, "z": -1}})
print(out5)


# ===========================================================
# Example 6: defaulted parameters — only need to provide non-defaults
def string_any(a, b=2, c="hello", f="john"):
    return f"Single string -- a: {a}, b: {b}, c: {c}, f: {f}"

tool6 = Tool("string_any", string_any)
wf6 = ToolFlow(tool=tool6, output_schema=["text"])

print("\n[6] Defaults respected (only required key provided):")
out6 = wf6.invoke({"a": "al"})
print(out6)


# ===========================================================
# Example 7: output schema used to fan out list return into named fields
def string_to_list(string: str):
    # Return 3 fields; output_schema will positionally map them by order
    return string.split(",")[:3]

tool7 = Tool("string_to_list", string_to_list)
wf7 = ToolFlow(tool=tool7, output_schema=["name", "age", "state"], bundle_all=False)

print("\n[7] Output schema mapping:")
out7 = wf7.invoke({"string": "John,37,New Jersey"})
print(out7)
print("This enables downstream workflows/agents/tools to accept named inputs "
      "matching ['name','age','state'] without manual unpacking.")