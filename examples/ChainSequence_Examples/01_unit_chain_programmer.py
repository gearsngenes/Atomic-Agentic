import sys
from pathlib import Path
from typing import Any
# Setting the root
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from modules.Agents import PrePostAgent, ChainSequenceAgent
from modules.LLMEngines import *

# define a global llm engine to give to each of our agents
llm_engine = OpenAIEngine(model = "gpt-4o-mini")

# Example 1: Single PolymerAgent (monomer) as a program builder/executor
# Seed agent: instance of Agent with code-writing role prompt
CODE_WRITER_PROMPT = """
You are a Python code generator. Using any user-prompts given, construct
a valid python script that best accomplishes the user's task. Respond ONLY
with valid Python code, no extra text, tags (i.e. ```python...), or 
explanations. Your output should be copy-pastable and ready to execute
directly in the built-in exec() function.

Be sure to return a non-none value if requested by the user's task. If
a return value is expected, be sure that in the script that the last
variable it is saved in is called 'output'.
"""

# Define our seed agent
_internal_coder = PrePostAgent(
    name         = "Coder",
    llm_engine      = llm_engine,
    role_prompt  = CODE_WRITER_PROMPT)

# define an execution tool to run after our code-writer seed creates code string
def _exec(code: str) -> Any:
    print("[POST-PROCESS] RUNNING CODE:\n======\n", code,"\n======\n[/POST-PROCESS]")
    try:
        exec(code, globals())
        # If 'output' is set in globals, return it
        if 'output' in globals():
            return globals()['output']
        else:
            return f"{code}\n---\nAbove code returned 'None'"
    except Exception as e:
        return f"Erroneous Code:\n{code}\n\nError: {e}"

# Register the execution method with the coder
_internal_coder.add_poststep(_exec)

# Define our 1-unit polymer agent
single_link_programmer = ChainSequenceAgent("Programmer")
single_link_programmer.add(_internal_coder)
# Set up example
math_function = "3x^3 + 2x + 1"
x_equals = 1

prompt = f"""
Return the derivative of {math_function} at x = {x_equals}. Use nested
functions to first calculate the general derivative of any function. Then 
evaluate the calculated derivative at x = {x_equals} using those methods. Finally, 
return the final result as a string formatted as:
'Derivative of f(x = {x_equals}) = {{result here}}'.
"""
# invoke the monomer_programmer and print the result string
result = single_link_programmer.invoke(prompt)
print(f"[SINGLE LINK OUTPUT] CODE RESULT: {result}")