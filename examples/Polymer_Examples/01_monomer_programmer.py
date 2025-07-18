import sys
from pathlib import Path
from typing import Any
# Setting the root
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from modules.Agents import Agent, PolymerAgent
from modules.LLMNuclei import *

# define a global nucleus to give to each of our agents
nucleus = OpenAINucleus(model = "gpt-4o-mini")

# Example 1: Single PolymerAgent (monomer) as a program builder/executor
# Seed agent: instance of Agent with code-writing role prompt
CODE_WRITER_PROMPT = """
You are a Python code generator. Using any user-prompts given, construct
a valid python script that best accomplishes the user's task. Respond ONLY
with valid Python code, no extra text, tags (i.e. ```python...), or 
explanations. Your output should be copy-pastable and ready to execute
directly in the built-in exec() function.

Regardless of the task, at the end of every script you generate, assign the
final result to a variable called 'output', and ensure the last line of the
script is 'output'.
"""

CodeWriter = Agent(name         = "CodeWriter",
                   nucleus      = nucleus,
                   role_prompt  = CODE_WRITER_PROMPT)

# Create PolymerAgent (monomer)
monomer_programmer = PolymerAgent(CodeWriter)

# define an execution tool to run after our code-writer seed creates code string
def _exec(code: str) -> Any:
    print("Running the following code:\n", code)
    try:
        exec(code, globals())
        # If 'output' is set in globals, return it
        if 'output' in globals():
            return globals()['output']
        else:
            return f"{code}\n---\nAbove code returned 'None'"
    except Exception as e:
        return f"Erroneous Code:\n{code}\n\nError: {e}"
monomer_programmer.register_tool(_exec)


# Set up example
math_function = "3x^3 + 2x + 1"
x_equals = 1

prompt = f"""
Return the derivative of {math_function} at x = {x_equals}. Use nested
functions to first calculate the general derivative of any function. Then 
evaluate the calculated derivative at x = 5 using those methods. Finally, 
return the final result as a string formatted as:
'Derivative of f(x = {{point here}}) = {{result here}}'.

All of this should be wrapped inside one 'main()' function.
"""
# invoke the monomer_programmer and
result = monomer_programmer.invoke(prompt)
# print the result
print("Monomer output:\n", result)