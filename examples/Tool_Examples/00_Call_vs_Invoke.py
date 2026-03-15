from atomic_agentic.tools import Tool

def to_uppercase(text: str, id: int) -> str:
    """Convert input text to uppercase."""
    return f"[{id}] {text.upper()}"

uppercase_tool = Tool(to_uppercase)
print(uppercase_tool)
print("Positional call:", uppercase_tool("hello positional world", id=1))  # Output: "[1] HELLO POSITIONAL WORLD"
print("Keyword call:", uppercase_tool(text="hello keyword world", id=2))  # Output: "[2] HELLO KEYWORD WORLD"
print("Mixed call:", uppercase_tool("hello mixed world", id=3))  # Output: "[3] HELLO MIXED WORLD"
print("Call Tool with Invoke:", uppercase_tool.invoke({"text": "hello invoked world", "id": 4}))  # Output: "[4] HELLO INVOKE WORLD"