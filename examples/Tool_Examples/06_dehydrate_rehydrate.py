from atomic_agentic.Factory import load_tool
from atomic_agentic.Toolify import toolify
from atomic_agentic.Tools import Tool, ToolInvocationError
from atomic_agentic.Agents import Agent
from atomic_agentic.LLMEngines import OpenAIEngine
import json
func = len
tool_from_callable = toolify(
    func,
    name="length_calculator",
    description="Calculate the length of the given input.",
    source="local",
)[0]

print("Length of [3,1,4,1,5,9]:", tool_from_callable.invoke({"obj": [3, 1, 4, 1, 5, 9]}))  # Should print 6

dict_representation = tool_from_callable.to_dict()
print("Serialized Tool from callable:\n", json.dumps(dict_representation, indent=2))

reconstructed_tool = load_tool(dict_representation)
print("Reconstructed Tool length of 'hello world':", reconstructed_tool.invoke({"obj": "hello world"}))  # Should print 11


import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from examples.Tool_Examples.sample_mcp_server import multiply

custom_tool = toolify(
    multiply,
    name="multiplier",
    description="Multiply two numbers or a number and string. Args: a:int|float|str (required), b:int|float (required). Returns: int.",
    source = "local"
)[0]

print("====   Custom Tool Tests   ====")

print("Result of 5 * 4: ", custom_tool.invoke({"a": 5, "b": 4}))  # Should print 20
print("Result of 'ha' * 3: ", custom_tool.invoke({"a": "ha", "b": 3}))  # Should print 'hahaha'

dict_repr_custom = custom_tool.to_dict()
print("Serialized Custom Tool:\n", json.dumps(dict_repr_custom, indent=2))

reconstructed_custom_tool = load_tool(dict_repr_custom)
print("Reconstructed Custom Tool result of 7 * 6: ", reconstructed_custom_tool.invoke({"a": 7, "b": 6}))  # Should print 42