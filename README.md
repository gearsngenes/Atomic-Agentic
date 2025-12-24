# Atomic-Agentic

Atomic-Agentic is a small, opinionated Python library for building **agentic systems** out of three composable primitives:

- **Tool**: a *dict-first* callable with introspectable schema + metadata.
- **Agent**: a schema-driven LLM wrapper (**pre_invoke → LLMEngine → post_invoke**).
- **Workflow**: a deterministic **packaging boundary** for orchestration and IO normalization.

The library is designed so that tools, agents, and workflows can be wrapped, composed, and adapted in a consistent way—without losing inspectability.

---

## Installation

From a cloned repo:

```bash
python -m build
```
From there, a `./dist/` folder will be created, from which a .tgz and .whl file will be generated. Use the latest created of either of these files and run:
```bash
pip install ./dist/atomic-agentic-<rest of filename + extension here>
```

---

## Quickstart: a plan-and-execute tool agent

`PlanActAgent` makes **one** LLM call to generate a JSON plan (list of tool calls), executes the tools, and returns the final result.

```python
from atomic_agentic.Agents import PlanActAgent
from atomic_agentic.LLMEngines import OpenAIEngine
from atomic_agentic.Plugins import MATH_TOOLS

# 1) Create an engine (model is required)
engine = OpenAIEngine(model="gpt-4.1-mini")  # expects OPENAI_API_KEY env var

# 2) Create a planning agent
agent = PlanActAgent(
    name="planner",
    description="Plans and solves simple tasks using tools.",
    llm_engine=engine,
    tool_calls_limit=6,  # non-return calls only
)

# 3) Register tools (callables or Tool instances both work)
agent.batch_register(MATH_TOOLS)

# 4) Invoke (Agent inputs are ALWAYS a mapping)
result = agent.invoke({"prompt": "Compute (6*7) + 5. Return only the number."})
print(result)
```

---

## Core Concepts

### Tool IDs and the dict-first contract

Every tool has a stable identifier:

```
<Type>.<namespace>.<name>
```

Example: `Tool.default.add`

All Tools in Atomic-Agentic are **dict-first**:

```python
tool.invoke({"a": 2, "b": 3})
```

No positional calling through `invoke()`. The Tool converts a mapping into `(*args, **kwargs)` internally (based on signature inspection).

This “dict-first” rule is what makes tools:
- introspectable (schema/arguments are inspectable),
- safe for LLM tool use (keys are explicit),
- composable across agent and workflow layers.

---

## Tools

### `Tool`

A `Tool` wraps a Python callable and exposes:

- `arguments_map`: an ordered schema derived from the callable signature
- `return_type`: derived from the return annotation (or best-effort)
- `signature`: a human-readable string
- `full_name`: `<Type>.<namespace>.<name>`
- `invoke(inputs: Mapping[str, Any]) -> Any`: the only public execution entrypoint

```python
from atomic_agentic.Tools import Tool

def add(a: int, b: int) -> int:
    return a + b

t = Tool(
    function=add,
    name="add",
    namespace="default",
    description="Add two integers.",
)

print(t.full_name)      # Tool.default.add
print(t.signature)      # Tool.default.add(a: int, b: int) -> int
print(t.invoke({"a": 2, "b": 3}))
```

**Unknown keys:** by default, inputs must match the callable’s signature. Unknown keys are only accepted if the wrapped callable declares `**kwargs`.

---

### MCP tools: `MCPProxyTool`

`MCPProxyTool` adapts a single tool from an MCP server into a normal `Tool`.

```python
from atomic_agentic.Tools import MCPProxyTool

tool = MCPProxyTool(
    server_url="http://localhost:8000/mcp",
    tool_name="weather_lookup",
    namespace="mcp",
    headers={},  # pass a dict even if empty
)

print(tool.full_name)
print(tool.invoke({"city": "New York"}))
```

---

### Wrapping an Agent as a Tool: `AgentTool`

`AgentTool` turns an `Agent` into a `Tool`:

- tool name is always `"invoke"`
- namespace is `agent.name`
- arguments schema mirrors the agent’s **pre_invoke** schema
- return type mirrors the agent’s **post_invoke** return type

```python
from atomic_agentic.Tools import AgentTool
from atomic_agentic.Primitives import Agent
from atomic_agentic.LLMEngines import OpenAIEngine

engine = OpenAIEngine(model="gpt-4.1-mini")
agent = Agent(name="qa", description="Answers questions.", llm_engine=engine)

agent_tool = AgentTool(agent)
print(agent_tool.full_name)  # AgentTool.qa.invoke
print(agent_tool.invoke({"prompt": "What is the capital of France?"}))
```

---

### `toolify(component, **kwargs)`

`toolify()` is a single entrypoint that converts one of:

- `callable`
- `Tool`
- `Agent`
- `str` URL (MCP server URL, or A2A endpoint fallback)

…into a list of `Tool` instances.

```python
from atomic_agentic.Toolify import toolify
from atomic_agentic.Primitives import Agent
from atomic_agentic.LLMEngines import OpenAIEngine

def hello(name: str) -> str:
    return f"Hello, {name}!"

tools = toolify(hello, namespace="default", description="Greets a person.")
print([t.full_name for t in tools])

engine = OpenAIEngine(model="gpt-4.1-mini")
agent = Agent(name="qa", description="Answers questions.", llm_engine=engine)

tools = toolify(agent)
print([t.full_name for t in tools])  # AgentTool.qa.invoke
```

**String URLs (MCP or A2A):**

- If `component` is a URL string, `toolify()` first tries MCP discovery.
- If MCP discovery fails, it attempts to construct an **A2A** client tool (`A2AgentTool`).

Important: when toolifying a remote endpoint string, `toolify()` requires you to pass the `headers=` keyword
(even if the value is `None`).

```python
tools = toolify(
    "http://localhost:4242",  # A2A or MCP URL
    headers={},               # must be provided; may be None
)
```

For MCP servers that expose many tools, `toolify()` can return multiple tools. Use `include=[...]` / `exclude=[...]` to filter.

---

## Agents

### Base `Agent` (schema-driven LLM agent)

`Agent` is a stateful LLM wrapper:

1) **pre_invoke** Tool turns `inputs: Mapping[str, Any]` into a **prompt string**  
2) `LLMEngine.invoke(messages)` produces raw text  
3) **post_invoke** Tool converts raw output into the final return value  
4) if `context_enabled=True`, the conversation turn is stored

Default behavior:
- `pre_invoke` is a strict identity Tool requiring `{"prompt": str}`
- `post_invoke` is an identity Tool for the raw LLM output

```python
from atomic_agentic.Primitives import Agent
from atomic_agentic.LLMEngines import OpenAIEngine

engine = OpenAIEngine(model="gpt-4.1-mini")
agent = Agent(name="helper", description="Helpful assistant.", llm_engine=engine)

print(agent.invoke({"prompt": "Write a haiku about snow."}))
```

---

### Tool-using agents: `ToolAgent`, `PlanActAgent`, `ReActAgent`

Atomic-Agentic provides a base `ToolAgent` (inherits `Agent`) that adds:

- a **toolbox** (`register`, `batch_register`)
- a **blackboard** of executed steps (for tool-call traceability)
- canonical placeholder syntax: `<<__step__N>>` to reference prior results
- a **tool call limit** (non-return calls only)

Two built-in strategies:

#### `PlanActAgent` (plan-first; one LLM call)
- LLM produces a JSON array of steps: `[{ "tool": "...", "args": {...} }, ...]`
- Steps execute sequentially or in dependency “waves” (`run_concurrent=True`)
- Enforces the final canonical return tool: `Tool.ToolAgents.return`

#### `ReActAgent` (step-by-step orchestrator)
- Repeatedly chooses **one** next tool call as a JSON object
- Stops when it calls `Tool.ToolAgents.return`

Example:

```python
from atomic_agentic.Agents import PlanActAgent, ReActAgent
from atomic_agentic.LLMEngines import OpenAIEngine
from atomic_agentic.Plugins import MATH_TOOLS

engine = OpenAIEngine(model="gpt-4.1-mini")

planner = PlanActAgent(
    name="planner",
    description="Plans tool usage in one shot.",
    llm_engine=engine,
    tool_calls_limit=6,
)
planner.batch_register(MATH_TOOLS)
print(planner.invoke({"prompt": "Compute (12*3) - 8 and return the number."}))

reactor = ReActAgent(
    name="reactor",
    description="Chooses one tool call at a time.",
    llm_engine=engine,
    tool_calls_limit=6,
)
reactor.batch_register(MATH_TOOLS)
print(reactor.invoke({"prompt": "Compute (12*3) - 8 and return the number."}))
```

---

## A2A Interop

### Hosting a local Agent via A2A: `A2AgentHost`

`A2AgentHost` wraps an `Agent` and serves it over the A2A protocol. It supports two function calls:

- `invoke(payload=<mapping>)`
- `agent_metadata() -> {arguments_map, return_type}`

```python
from atomic_agentic.Agents import A2AgentHost
from atomic_agentic.Primitives import Agent
from atomic_agentic.LLMEngines import OpenAIEngine

engine = OpenAIEngine(model="gpt-4.1-mini")
agent = Agent(name="trivia", description="Trivia expert.", llm_engine=engine)

host = A2AgentHost(seed_agent=agent, host="127.0.0.1", port=4242)
host.run(debug=True)
```

### Calling an A2A agent as a Tool: `A2AgentTool`

`A2AgentTool` is a client-side proxy Tool that forwards dict inputs to a remote A2A agent.
You’ll usually get it via `toolify("http://host:port", headers=...)` when MCP discovery fails.

---

## Workflows

Workflows are deterministic **packaging boundaries**:

- Inputs are always `Mapping[str, Any]`
- Subclasses implement `_invoke(inputs) -> (metadata: Mapping[str, Any], raw: Any)`
- The base `Workflow.invoke()` packages `raw` into an ordered `output_schema`

### Output schemas and defaults

If you don’t provide an output schema, workflows default to a single key:

```text
DEFAULT_WF_KEY = "__RESULT__"
```

So “scalar” outputs become `{ "__RESULT__": raw }` under the default policy.

### Packaging policies

- **BundlingPolicy**
  - `BUNDLE`: if `output_schema` length is 1, bundle raw into that key
  - `UNBUNDLE`: try to interpret raw as mapping/sequence/scalar and map it into schema
- **MappingPolicy** (when raw is mapping-shaped)
  - `STRICT`, `IGNORE_EXTRA`, `MATCH_FIRST_STRICT`, `MATCH_FIRST_LENIENT`
- **AbsentValPolicy** (final completeness handling)
  - `RAISE`, `DROP`, `FILL`

---

### Built-in workflow adapters

- `ToolFlow`: wrap a single Tool as a Workflow
- `AgentFlow`: wrap a single Agent as a Workflow
- `AdapterFlow`: generic wrapper for Tool/Agent/Workflow with re-packaging
- `StateIOFlow`: graph-node wrapper for stateful orchestration frameworks (e.g., LangGraph)

---

### Example: `StateIOFlow` inside a mini LangGraph graph

`StateIOFlow` is designed for “node functions” that:
- accept a **state dict**,
- read only the keys they declare,
- return a **partial state update** dict (subset of the state).

That makes it easy to plug Atomic-Agentic components into orchestration frameworks like LangGraph
without hand-writing input filtering / output shaping for every node.

> Optional dependency:
> ```bash
> pip install -U langgraph
> ```

```python
from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from atomic_agentic.Primitives import Agent
from atomic_agentic.LLMEngines import OpenAIEngine
from atomic_agentic.Workflows import AgentFlow, StateIOFlow

# 1) Define the LangGraph state schema
class QAState(TypedDict, total=False):
    question: str
    answer: str

# 2) Build an Agent that consumes `{"question": ...}` instead of `{"prompt": ...}`
def question_to_prompt(question: str) -> str:
    return f"Answer clearly and concisely:\n\nQuestion: {question}"

engine = OpenAIEngine(model="gpt-4.1-mini")
qa_agent = Agent(
    name="qa",
    description="Answers a question.",
    llm_engine=engine,
    pre_invoke=question_to_prompt,  # maps {"question": ...} -> prompt string
)

# 3) Wrap the agent so its output is a state update mapping {"answer": ...}
qa_flow = AgentFlow(qa_agent, output_schema=["answer"])  # bundles the LLM string into {"answer": ...}

# 4) Wrap again as a stateful node adapter:
#    - validates the component's declared inputs are a subset of QAState keys
#    - filters the incoming state down to only the needed keys (here: {"question"})
#    - returns a partial update compatible with LangGraph
qa_node = StateIOFlow(qa_flow, state_schema=QAState)

def answer_node(state: QAState) -> QAState:
    return qa_node.invoke(state)  # returns {"answer": "..."} (a Partial<QAState>)

# 5) Build and run a tiny graph
graph = StateGraph(QAState)
graph.add_node("answer", answer_node)
graph.add_edge(START, "answer")
graph.add_edge("answer", END)

app = graph.compile()
final_state = app.invoke({"question": "What is the capital of France?"})

print(final_state["answer"])
```

---

## Embeddings

Embedding engines live in `EmbedEngines.py` and implement:

```python
vectorize(text: str) -> list[float]
```

Example:

```python
from atomic_agentic.EmbedEngines import OpenAIEmbedEngine

engine = OpenAIEmbedEngine(model="text-embedding-3-small")
vec = engine.vectorize("hello world")
print(len(vec), vec[:5])
```

---

## Plugins

`Plugins.py` provides ready-made tool bundles, e.g.:

- `MATH_TOOLS`
- `CONSOLE_TOOLS`
- `PARSER_TOOLS`

These are intended to be registered into a `ToolAgent` quickly.

---

## Serialization notes

Most primitives implement `to_dict()` for **diagnostic snapshots** (safe to log / inspect).
Rehydration helpers in `Factory.py` are currently in flux and may lag behind recent class changes.
For now, treat `to_dict()` as introspection rather than a stable persistence format.

---
