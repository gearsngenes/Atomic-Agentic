# examples/Workflow_Examples/01_BasicFlow.py
from __future__ import annotations

import logging
from typing import Any

from atomic_agentic.tools import Tool
from atomic_agentic.workflows import BundlingPolicy, MappingPolicy
from atomic_agentic.workflows import BasicFlow

logging.basicConfig(level=logging.INFO)
print("=== BasicFlow examples (schema-driven dict inputs) ===")


# -------------------------------------------------------------------
# Example tool: mapping return -> mapping packaging
# -------------------------------------------------------------------
def duplicate_text(*, text: str, n: int = 2) -> dict[str, Any]:
    # Return a mapping on purpose to exercise mapping-policy packaging.
    return {"x": text * n, "n": n}


my_tool = Tool(
    function=duplicate_text,
    name="duplicate_text",
    namespace="examples",
    description="Duplicate 'text' n times and return {'x': <str>, 'n': <int>}.",
)

tf = BasicFlow(
    component=my_tool,
    output_schema=["x", "n"],
    # BundlingPolicy is ignored when schema length != 1; UNBUNDLE behavior applies.
    bundling_policy=BundlingPolicy.BUNDLE,
    mapping_policy=MappingPolicy.STRICT,
)

print("\n-- tool --")
print("name:", my_tool.name)
print("full_name:", my_tool.full_name)
print("signature:", my_tool.signature)
print("parameters:", my_tool.parameters)

print("\n-- workflow --")
print("parameters (proxied from tool):", tf.parameters)
print("output_schema:", tf.output_schema)

inputs = {"text": "ab", "n": 3}
result = tf.invoke(inputs)
print("invoke result:", result)


# -------------------------------------------------------------------
# Scalar return -> single-key packaging (bundling applies because schema len == 1)
# -------------------------------------------------------------------
def str_len(*, s: str) -> int:
    return len(s)


len_tool = Tool(
    function=str_len,
    name="str_len",
    namespace="examples",
    description="Compute the length of s.",
)

wf_len = BasicFlow(
    component=len_tool,
    output_schema=["length"],  # single-key schema
    bundling_policy=BundlingPolicy.BUNDLE,  # bundles raw scalar under 'length'
)

print("\n-- scalar return -> single-key bundling --")
print("invoke:", wf_len.invoke({"s": "chatgpt"}))


# -------------------------------------------------------------------
# List return -> bundled under a single key (schema len == 1)
# -------------------------------------------------------------------
def string_to_list(*, s: str) -> list[str]:
    return s.split()


list_tool = Tool(
    function=string_to_list,
    name="string_to_list",
    namespace="examples",
    description="Split string on whitespace.",
)

wf_list_bundle = BasicFlow(
    component=list_tool,
    output_schema=["tokens"],
    bundling_policy=BundlingPolicy.BUNDLE,  # bundles list under 'tokens'
)

print("\n-- list return -> bundle under 'tokens' --")
print("invoke:", wf_list_bundle.invoke({"s": "quick brown fox"}))


# -------------------------------------------------------------------
# List return -> unbundle (positional zip to schema)
# -------------------------------------------------------------------
def name_age_state(*, raw: str) -> list[str]:
    # e.g., "Ada,37,CA" -> ["Ada", "37", "CA"]
    return raw.split(",")


pos_tool = Tool(
    function=name_age_state,
    name="name_age_state",
    namespace="examples",
    description="Comma-split raw into [name, age, state].",
)

wf_list_zip = BasicFlow(
    component=pos_tool,
    output_schema=["name", "age", "state"],
    bundling_policy=BundlingPolicy.UNBUNDLE,  # activates sequence->schema packaging
    mapping_policy=MappingPolicy.STRICT,
)

print("\n-- list return -> unbundle + zip to schema --")
print("invoke:", wf_list_zip.invoke({"raw": "Ada,37,CA"}))


# -------------------------------------------------------------------
# Sequence overflow -> IGNORE_EXTRA truncation (policy-dependent)
# -------------------------------------------------------------------
def four_numbers(*, x: int) -> list[int]:
    return [x, x + 1, x + 2, x + 3]


overflow_tool = Tool(
    function=four_numbers,
    name="four_numbers",
    namespace="examples",
    description="Return four ints for overflow packaging demo.",
)

wf_overflow_truncate = BasicFlow(
    component=overflow_tool,
    output_schema=["a", "b", "c"],  # only 3 schema keys
    bundling_policy=BundlingPolicy.UNBUNDLE,
    mapping_policy=MappingPolicy.IGNORE_EXTRA,  # truncates extra sequence items
)

print("\n-- sequence overflow -> IGNORE_EXTRA truncation --")
print("invoke:", wf_overflow_truncate.invoke({"x": 10}))


# -------------------------------------------------------------------
# Set return -> single-key bundling (schema len == 1)
# Note: sets are not sequences for the Workflow packer; bundling is the right mode.
# -------------------------------------------------------------------
def unique_chars(*, s: str) -> set[str]:
    return set(s)


set_tool = Tool(
    function=unique_chars,
    name="unique_chars",
    namespace="examples",
    description="Set of unique characters.",
)

wf_set_bundle = BasicFlow(component=set_tool, output_schema=["chars"])
print("\n-- set return -> bundle under 'chars' --")
print("invoke:", wf_set_bundle.invoke({"s": "mississippi"}))
