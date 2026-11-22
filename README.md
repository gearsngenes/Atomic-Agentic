# Atomic-Agentic

## Introduction

**Atomic-Agentic** is an agentic AI framework that addresses two
structural problems in real-world AI systems:

-   prompts overloaded with deterministic runtime logic
-   heterogeneous interfaces between tools, engines, agents, and
    composites that require significant plumbing to scale

Atomic-Agentic acts as an **adapter-first execution substrate** that
moves runtime responsibilities out of natural language and into
mandatory, testable code paths and places all executable components
behind a single, dictionary-based interface. The result is cleaner
composition, less integration friction, and systems that are easier to
reason about and extend.

Through Atomic-Agentic, users can build **agentic systems** out of four
composable primitives:

-   **Tool** -- A *dict-first adapter layer* for functions, methods, and
    remote endpoints (such as MCP tools or A2A agents).
-   **LLM Engine** -- A provider adapter that wraps LLM APIs behind a
    common contract for generating the next message from a sequence of
    messages.
-   **Agent** -- An autonomous reasoning component that uses LLM Engines
    and Tools to complete tasks and interact with its environment.
-   **Composite** -- An **orchestration** layer that composes tools,
    engines, agents and other composites into graphs, pipelines and
    routers.

Together, these primitives form a composable system where **LLMs handle
reasoning while deterministic execution logic lives in code.**

------------------------------------------------------------------------

## Installation

The core install carries only the protocol libraries (`mcp`, `a2a-sdk`,
`httpx`). LLM provider SDKs are optional extras, so install only the ones you
use:

``` bash
pip install "atomic-agentic[openai]"      # OpenAIEngine
pip install "atomic-agentic[anthropic]"   # AnthropicEngine
pip install "atomic-agentic[gemini]"      # GeminiEngine
pip install "atomic-agentic[mistral]"     # MistralEngine
pip install "atomic-agentic[litellm]"     # LiteLLMEngine
pip install "atomic-agentic[llama]"       # LlamaCppEngine (builds llama-cpp-python from source)
pip install "atomic-agentic[all_llm]"     # every provider above
```

Constructing an engine whose SDK is missing raises an error that names the
extra to install. The only other extra is `dev` (test tooling).

`LiteLLMEngine` can also talk to a LiteLLM proxy over HTTP (`base_url`). The
proxy server is a separate program: install it with `litellm[proxy]`, ideally
in its own environment.

### Install directly from GitHub

``` bash
pip install "atomic-agentic[openai] @ git+https://github.com/gearsngenes/Atomic-Agentic.git"
```

### Install from source

Clone the repository and install from the generated build artifacts:

``` bash
git clone https://github.com/gearsngenes/Atomic-Agentic
cd Atomic-Agentic

pip install --upgrade build
python -m build
pip install ./dist/atomic_agentic-*.whl
```

Once installed, explore the `examples/` directory for complete
demonstrations of **Tools, Engines, Agents, and Composites**.

------------------------------------------------------------------------

![](images/AtomicInvokable.gif)

------------------------------------------------------------------------

## Quickstart A: Tools

`Tool` wraps a Python callable and exposes a **dict-first** interface.
Tools adapt functions, methods, MCP endpoints, A2A services, and other
callables into the Atomic-Agentic invocation contract.

``` python
from atomic_agentic.tools import Tool

accounts = [
    {"name": "Alice Johnson", "birthdate": "1985-03-15", "account_balance": 15750.50, "annual_interest_rate": 0.025},
    {"name": "Bob Smith", "birthdate": "1990-07-22", "account_balance": 42300.75, "annual_interest_rate": 0.030},
    {"name": "Caleb Donavan", "birthdate": "1974-09-08", "account_balance": 36130.01, "annual_interest_rate": 0.027},
]

def get_account_details(index: int) -> dict:
    """Retrieve bank account details by index."""
    return accounts[index]

tool = Tool(
    function=get_account_details,
    name="get_account_details",
    description="Retrieve bank account details by index."
)

print(tool.full_name)
print(tool.signature)
print(tool.parameters)

result = tool.invoke({"index": 0})
print(result.result)
```

------------------------------------------------------------------------

## Quickstart B: LLM Engines

`LLMEngine` classes wrap model providers behind a standardized
message-based interface. They receive a sequence of messages and
generate the next response message.

``` python
from atomic_agentic.llm import OpenAIEngine

engine = OpenAIEngine(model="gpt-4o-mini")

response = engine.invoke({
    "messages": [
        {"role": "user", "content": "Explain Newton's third law."}
    ]
})

print(response.result)
```

LLM Engines intentionally remain **thin adapters**.\
They standardize provider APIs but do not perform reasoning or
orchestration. Those responsibilities belong to agents.

------------------------------------------------------------------------

## Quickstart C: Basic Agents

`Agent` is an autonomous unit that uses an LLM Engine to complete tasks.
Agents may also use Tools to interact with their environment.

``` python
from atomic_agentic.agents import BasicAgent
from atomic_agentic.llm import OpenAIEngine

accounts = [
    {"name": "Alice Johnson", "birthdate": "1985-03-15", "account_balance": 15750.50, "annual_interest_rate": 0.025},
    {"name": "Bob Smith", "birthdate": "1990-07-22", "account_balance": 42300.75, "annual_interest_rate": 0.060},
    {"name": "Carol Davis", "birthdate": "1988-11-08", "account_balance": 28900.25, "annual_interest_rate": 0.028},
]

def format_finance_request(account_index: int, sector: str) -> str:
    account = accounts[account_index]
    return f"""
Customer: {account['name']} (DOB: {account['birthdate']})
Capital: ${account['account_balance']:,.2f}
Rate: {account['annual_interest_rate']*100:.1f}%
---
Desired Sector to invest: {sector.capitalize()}
"""

engine = OpenAIEngine(model="gpt-4o-mini")

advisor = BasicAgent(
    name="finance_advisor",
    description="Investment advisor.",
    llm_engine=engine,
    instructions="""You are an expert financial advisor at a bank. 
When you receive customer banking data and a desired sector,
provide a bulleted list of relevant investment tickers.""",
    pre_invoke=format_finance_request,
)

result = advisor.invoke({"account_index": 0, "sector": "technology"})
print(result.result)
```

------------------------------------------------------------------------


## Quickstart D: Tool-Calling Agents

Atomic-Agentic also supports autonomous **tool-calling agent classes**.

`PlanActAgent` decomposes prompts into a sequence of steps and executes those steps using tools.

```python
from atomic_agentic.agents import PlanActAgent
from atomic_agentic.llm import OpenAIEngine

def add(a: float, b: float) -> float:
    """Return the sum of two numbers."""
    return a + b

def multiply(a: float, b: float) -> float:
    """Return the product of two numbers."""
    return a * b

engine = OpenAIEngine(model="gpt-4.1-mini")

agent = PlanActAgent(
    name="planner",
    description="Plans and solves tasks using tools.",
    llm_engine=engine,
)

agent.register_tools([add, multiply])

result = agent.invoke({"prompt": "Compute (6*7) + 5. Return only the number."})
print(result.result)
```

**Note:** `PlanActAgent` plans an entire batch of tool calls upfront, in
one JSON-structured generation; `ReActAgent` decides and dispatches one
tool call at a time instead, reacting to each result before choosing the
next -- see `examples/ReAct_Examples/` for worked examples of the latter.

------------------------------------------------------------------------

## Quickstart E: ScriptActAgent

`ScriptActAgent` is the successor to the JSON-based tool-calling agents
above. Instead of planning in JSON, it plans by writing native,
restricted-grammar Python statements — calling registered tools, a curated
set of approved Python builtins, and attribute/method access on
already-resolved values. Statements with no data dependency on each other
are batched and dispatched concurrently, inferred automatically from the
plan's own call-dependency graph, with no model-facing concurrency signal
needed.

```python
from atomic_agentic.agents import ScriptActAgent
from atomic_agentic.llm import OpenAIEngine

engine = OpenAIEngine(model="gpt-4.1-mini")

agent = ScriptActAgent(
    name="scripter",
    description="Plans and solves tasks by writing Python-style call statements.",
    llm_engine=engine,
)

result = agent.invoke({"prompt": "Compute (6*7) + 5. Return only the number."})
print(result.result)
```

(No tools registered above -- `ScriptActAgent` solves bare arithmetic like
this via its own restricted-grammar builtins; see `examples/ScriptAct_Examples/`
for tool registration in context.)

The grammar accepts positional args, keyword args, `*args`, and `**kwargs`
unpacking, but deliberately rejects `for`/`while` loops, comprehensions,
and lambdas outright — it is not a general code-execution sandbox. See
`examples/ScriptAct_Examples/` for more, including concurrent dispatch,
repair-on-failure replanning, and cross-invocation result addressing.

------------------------------------------------------------------------

## Shaping a Step's Output

A pipeline step passes a mapping to the next step, so a step's function returns a dict whose keys are the next step's parameter names.

```python
from atomic_agentic.tools import Tool
from atomic_agentic.composites import Pipeline

def add(x: int, y: int) -> dict[str, int]:
    return {"value": x + y}

def double(value: int) -> dict[str, int]:
    return {"result": value * 2}

pipeline = Pipeline(
    name="add_then_double",
    description="Add two numbers, then double the sum.",
    steps=[Tool(add), Tool(double)],
)
print(pipeline.invoke({"x": 2, "y": 3}).result)  # {'result': 10}
```

An agent does the same through `post_invoke` (with `response_to_post="result"`): the function receives the agent's reply and returns the dict.

------------------------------------------------------------------------

## Agents Inside Agents

`instructions`, `thinking_instructions`, `pre_invoke` and `post_invoke` each accept a string (a prompt template), a callable, or any `AtomicInvokable`, agents included. An agent used this way is invoked once per call of the outer agent, and its parameters join the outer agent's signature. There is no cycle guard, so avoid an agent that (directly or not) uses itself.

```python
from atomic_agentic.agents import BasicAgent
from atomic_agentic.llm import OpenAIEngine

engine = OpenAIEngine(model="gpt-4o-mini")

# The persona writer is an agent with its own pre_invoke template. Its two
# placeholders, `persona` and `tone`, become parameters of whoever uses it.
persona_writer = BasicAgent(
    name="persona_writer",
    description="Writes the system prompt for a persona.",
    llm_engine=engine,
    instructions=(
        "You write system prompts. Given a persona and a tone, reply with two or three "
        "sentences that tell an assistant who it is, how it speaks and what it avoids. "
        "Reply with the system prompt only."
    ),
    pre_invoke="Persona: {persona}\nTone: {tone}",
    record_enabled=False,  # a helper agent: keep no per-call records
)

# The assistant's instructions are that agent, so its system message is written
# fresh on every call. Its own pre_invoke template shapes the user task.
assistant = BasicAgent(
    name="assistant",
    description="Answers in a persona another agent wrote.",
    llm_engine=engine,
    instructions=persona_writer,
    pre_invoke="Topic: {topic}\nQuestion: {question}",
)

print([p.name for p in assistant.parameters])  # ['topic', 'question', 'persona', 'tone']

result = assistant.invoke({
    "topic": "navigation",
    "question": "How do I find north without a compass?",
    "persona": "a retired pirate captain",
    "tone": "gruff but encouraging",
})
print(result.result)
```

`topic` and `question` come from the assistant's own `pre_invoke` template and make up the user task. `persona` and `tone` come from the `persona_writer` and shape only the system message. The assistant's signature lists all four. A helper agent keeps a record of every call by default, so set `record_enabled=False` on one that runs inside another agent.

------------------------------------------------------------------------

## Composites

Composites orchestrate Atomic-Agentic primitives deterministically. Every step, node, selector or route is an invokable, a callable (wrapped as a `Tool`) or a string (a prompt template), so tools, agents and other composites mix freely.

| Class | What it is | Returns |
|---|---|---|
| `Graph` | Mutable nodes, edges and a `start`; nodes run in supersteps over a shared state and may loop back | `GraphResult` (`trace` of node names per superstep) |
| `Pipeline` | Frozen ordered steps; each step feeds the next | plain `AtomicResult` |
| `Router` | Frozen selector plus routes; only the chosen branch runs | `RouterResult` (`route`, `branch`) |

`Pipeline` and `Router` are built over a private `Graph`. A `Graph` loop uses `add_sequence(..., repeat_until=...)` with plain tools:

```python
from atomic_agentic.tools import Tool
from atomic_agentic.composites import Graph

def step(count: int) -> int:
    return count + 3

graph = Graph(name="count_up", description="Add 3 until the count reaches 9.", returns="count")
graph.add_sequence({"step": (Tool(step), "count")}, repeat_until={"count": 9})
graph.set_start("step")

result = graph.invoke({"count": 0})
print(result.result)  # 9
print(result.trace)   # (('step',), ('step',), ('step',))
```

A `Router` runs its selector, looks the response up in its routes and runs that branch:

```python
from atomic_agentic.tools import Tool
from atomic_agentic.composites import Router

def kind(n: int) -> str:
    return "even" if n % 2 == 0 else "odd"

router = Router(
    name="parity",
    description="Describe a number by its parity.",
    selector=Tool(kind),
    routes={
        "even": lambda n: f"{n} is even",
        "odd": lambda n: f"{n} is odd",
    },
)
out = router.invoke({"n": 7})
print(out.result, out.route, out.branch)  # 7 is odd odd parity_route_1
```

**Results:** `GraphResult.trace` lists node names (routers do not appear), `RouterResult.route` is the selector's value and `RouterResult.branch` the chosen branch's name, and `Pipeline` returns the plain `AtomicResult`. A non-final `Pipeline` step must return a mapping unless it is given a binding; see Shaping a Step's Output. A `Graph` that still has nodes ready when it reaches `max_traversal_depth` raises `RuntimeError`.

**For practical usage and advanced patterns, see the examples in:**
`examples/Composite_Examples/`

------------------------------------------------------------------------

## What's New in v2

v2 is a deliberate breaking-change line with two headline improvements:

**Reorganized package structure.** `src/atomic_agentic/` is now
organized into concern-based layers — `exceptions/`, `constants/`,
`models/`, and `utils/` sit below the domain packages (`agents/`,
`tools/`, `composites/`, `llm/`, `mcp/`, `a2a/`), with `invokables/`
holding only the shared invocation contract. The dependency topology
is explicit and there are no cross-layer back-edges.

**AtomicResult-family return contract.** Every `AtomicInvokable` —
Tool, Agent, ToolAgent, Composite, Engine
— now returns a typed `AtomicResult`-family envelope from
`invoke()`/`async_invoke()`. The `.result` field is always the
caller-facing payload; envelope fields (`started_at`, `ended_at`, derived `elapsed_time`,
`invoker_id` -- the invokable's `full_name` -- and subclass-specific fields
like token usage, step traces, and tool-call accounting) carry timing
and provenance. `run_id` appears only on agent results. Composites return `GraphResult` (with a node-name
`trace`) or `RouterResult` (with `route` and `branch`); `Pipeline` returns
the plain `AtomicResult`.

For a full breakdown of breaking changes and a v1→v2 migration guide:
- [`docs/MIGRATION.md`](docs/MIGRATION.md) — v1→v2 migration guide *(coming soon)*
- [`docs/CHANGELOG.md`](docs/CHANGELOG.md) — full release history

------------------------------------------------------------------------

## Repository Structure

```
Atomic-Agentic/
|-- examples/
|   |-- Agent_Examples/
|   |-- Agentic_Research/
|   |-- Composite_Examples/
|   |-- LLM_Examples/
|   |-- PlanAct_Examples/
|   |-- ReAct_Examples/
|   |-- ScriptAct_Examples/
|   |-- Tool_Examples/
|
|-- docs/
|   |-- CHANGELOG.md
|
|-- images/
|
|-- src/
|   |-- atomic_agentic/
|       |-- a2a/
|       |-- adapters/
|       |-- agents/
|       |-- composites/
|       |-- constants/
|       |-- exceptions/
|       |-- invokables/
|       |-- llm/
|       |-- mcp/
|       |-- models/
|       |-- tools/
|       |-- utils/
|       |-- __init__.py
|       |-- _version.py
|       |-- py.typed
|
|-- tests/
|-- README.md
|-- pyproject.toml
```

### Package topology

The subpackages of `src/atomic_agentic/` form a strict layered
dependency hierarchy — no back-edges:

```
{exceptions, constants} → models → utils → invokables → {agents, tools, composites, llm, mcp, a2a}
```

| Layer | Packages | What lives here |
|---|---|---|
| Base | `exceptions/`, `constants/` | Exception classes and pure literals/sentinels; zero dependencies |
| Data | `models/` | Dataclasses: `ParamSpec`, runtime records, result envelopes |
| Functions | `utils/` | Pure functions: async bridging, parameter ops, MCP and agent helpers |
| Contract | `invokables/` | Shared invocation contract: `AtomicInvokable` |
| Domain | `agents/`, `tools/`, `composites/`, `llm/`, `mcp/`, `a2a/` | Behavior implementations |

------------------------------------------------------------------------

## Running the Tests

Install the test tooling (`pytest`, `pytest-cov` and the A2A HTTP server)
from a source checkout, then run from the repository root:

``` bash
pip install -e ".[dev]"

pytest -m "not llm and not network" -q    # offline suite (what CI runs)
pytest tests/composites -q                # one folder
pytest tests/core/test_invokable.py -q    # one file
pytest -m llm                             # live tests; need API keys in .env
```

Coverage (`pytest-cov`; branch coverage is on, and `pyproject.toml` sets
the source and exclusions):

``` bash
pytest -m "not llm and not network" --cov=atomic_agentic --cov-report=term-missing
pytest -m "not llm and not network" --cov=atomic_agentic --cov-report=html   # open htmlcov/index.html
```

------------------------------------------------------------------------

## Changelog

See [`docs/CHANGELOG.md`](https://github.com/gearsngenes/Atomic-Agentic/blob/main/docs/CHANGELOG.md)
for the release history of notable changes.
