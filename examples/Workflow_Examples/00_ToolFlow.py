# examples/Workflow_Examples/00_ToolFlow.py
from typing import Any, Dict, List
import logging
from atomic_agentic.Tools import Tool
from atomic_agentic.Workflows import ToolFlow

logging.basicConfig(level=logging.INFO)
print("=== ToolFlow examples (schema-driven dict inputs) ===")

# -------------------------------------------------------------------
# Example tool: map -> map
# -------------------------------------------------------------------
def double_json_keys(text: str, n: int = 2) -> Dict[str, Any]:
    return text * n, n

my_tool = Tool(
    func=double_json_keys,
    name="double_and_json",
    description="Duplicate 'text' n times and expose both outputs."
)

tf = ToolFlow(tool=my_tool, output_schema=["n_x", "x"], bundle_all=False)

print("-- tool --")
print("name:", my_tool.name)
print("arguments_map:", my_tool.arguments_map)  # derived signature
print("-- workflow --")
print("arguments_map (proxied from tool):", tf.arguments_map)
print("output_schema:", tf.output_schema)

inputs = {"text": "ab", "n": 3}
result = tf.invoke(inputs)
print("invoke result:", result)

# -------------------------------------------------------------------
# Basic scalar returns -> single-key packaging
# -------------------------------------------------------------------
def str_len(s: str) -> int:
    return len(s)

len_tool = Tool(func=str_len, name="str_len", description="Compute the length of s")
wf_len = ToolFlow(tool=len_tool, output_schema=["length"])  # single-key schema

print("\n-- scalar return -> single-key packaging --")
print("arguments_map:", wf_len.arguments_map)
print("invoke:", wf_len.invoke({"s": "chatgpt"}))

# -------------------------------------------------------------------
# List return -> bundling under a single key
# -------------------------------------------------------------------
def string_to_list(s: str) -> List[str]:
    return s.split()

list_tool = Tool(func=string_to_list, name="string_to_list", description="Split string on whitespace")
wf_list_bundle = ToolFlow(tool=list_tool, output_schema=["tokens"], bundle_all=True)  # bundle list under 'tokens'

print("\n-- list return -> bundle under 'tokens' --")
print("invoke:", wf_list_bundle.invoke({"s": "quick brown fox"}))

# -------------------------------------------------------------------
# List return -> zip to schema (positional)
# -------------------------------------------------------------------
def name_age_state(raw: str) -> List[str]:
    return raw.split(",")  # e.g., "Ada,37,CA"

pos_tool = Tool(func=name_age_state, name="name_age_state", description="Comma-split name,age,state")
wf_list_zip = ToolFlow(tool=pos_tool, output_schema=["name", "age", "state"], bundle_all=False)

print("\n-- list return -> zip to schema --")
print("invoke:", wf_list_zip.invoke({"raw": "Ada,37,CA"}))

# -------------------------------------------------------------------
# Set return -> bundle under a single key
# -------------------------------------------------------------------
def unique_chars(s: str) -> set[str]:
    return set(s)

set_tool = Tool(func=unique_chars, name="unique_chars", description="Set of unique characters")
wf_set_bundle = ToolFlow(tool=set_tool, output_schema=["chars"])  # must be single key for sets

print("\n-- set return -> bundle under 'chars' --")
print("invoke:", wf_set_bundle.invoke({"s": "mississippi"}))
