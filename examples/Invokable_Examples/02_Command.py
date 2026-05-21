from atomic_agentic import Command
from atomic_agentic.tools import Tool


def multiply(*, a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


multiply_tool = Tool(
    function=multiply,
    name="multiply",
    namespace="example",
    description="Multiply two integers.",
    filter_extraneous_inputs=False,
)

multiply_6_by_7 = Command(
    executor=multiply_tool,
    fixed_inputs={"a": 6, "b": 7},
    name="multiply_6_by_7",
    description="Command that multiplies 6 by 7.",
)

print(multiply_6_by_7.signature)
# Command.multiply_6_by_7() -> int

print(multiply_6_by_7.parameters)
# []

result = multiply_6_by_7.invoke({})
print(result)
# 42

result = multiply_6_by_7()
print(result)
# 42

try:
    multiply_6_by_7.invoke({"a": 100})
except TypeError as exc:
    print(f"Rejected runtime inputs: {exc}")