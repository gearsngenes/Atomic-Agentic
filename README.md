# Atomic-Agentic

## Introduction

**Atomic-Agentic** is an agentic AI framework that addresses two
structural problems in real-world AI systems:

-   prompts overloaded with deterministic runtime logic
-   heterogeneous interfaces between tools, engines, agents, and
    workflows that require significant plumbing to scale

Atomic-Agentic acts as an **adapter-first execution substrate** that
moves runtime responsibilities out of natural language and into
mandatory, testable code paths and places all executable components
behind a single, dictionary-based interface. The result is cleaner
composition, less integration friction, and systems that are easier to
reason about and extend.

Through Atomic-Agentic, users can build **agentic systems** out of five
composable primitives:

-   **Tool** -- A *dict-first adapter layer* for functions, methods, and
    remote endpoints (such as MCP tools or A2A agents).
-   **LLM Engine** -- A provider adapter that wraps LLM APIs behind a
    common contract for generating the next message from a sequence of
    messages.
-   **Agent** -- An autonomous reasoning component that uses LLM Engines
    and Tools to complete tasks and interact with its environment.
-   **Workflow** -- An **orchestration and tracing** layer that
    coordinates tools, engines, agents, and other workflows into structured
    pipelines.

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
extra to install. Other extras: `dev` (test tooling) and `python-a2a`
(deprecated; the `python-a2a` client and host are removed in the final 2.0.0
release, use `A2AClientHub` and `A2AtomicExecutor` instead).

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
demonstrations of **Tools, Engines, Agents, and Workflows**.

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
    role_prompt="""You are an expert financial advisor at a bank. 
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

A workflow step passes a mapping to the next step, so a step's function returns a dict whose keys are the next step's parameter names.

```python
from atomic_agentic.tools import Tool
from atomic_agentic.workflows import SequentialFlow

def add(x: int, y: int) -> dict[str, int]:
    return {"value": x + y}

def double(value: int) -> dict[str, int]:
    return {"result": value * 2}

flow = SequentialFlow(
    name="add_then_double",
    description="Add two numbers, then double the sum.",
    steps=[Tool(add), Tool(double)],
)
print(flow.invoke({"x": 2, "y": 3}).result)  # {'result': 10}
```

An agent does the same through `post_invoke` (with `response_to_post="result"`): the function receives the agent's reply and returns the dict.

------------------------------------------------------------------------

## Workflows

Workflows orchestrate Atomic-Agentic primitives into deterministic pipelines. They provide patterns for composition, branching, iteration, and parallelism, enabling you to build complex agentic systems from modular components.

**Workflow classes include:**
- `SequentialFlow` — chains steps in sequence
- `ParallelFlow` — runs branches concurrently
- `RoutingFlow` — routes input to a selected branch
- `IterativeFlow` — loops until a checker condition is met
- `GraphFlow` — cyclic, dynamically-routed node graph; nodes can loop back
  to earlier nodes based on runtime routing decisions, superstep-batched
  concurrent execution

**Note:**
Workflows return typed result envelopes (`*FlowResult`) with `.result`, a
`trace` of the actual child results produced that run (opt out via
`include_trace=False`), and full run metadata. A non-final `SequentialFlow`
step must return a mapping; see Shaping a Step's Output.

**For practical workflow usage and advanced patterns, see the examples in:**
`examples/Workflow_Examples/`

------------------------------------------------------------------------

## What's New in v2

v2 is a deliberate breaking-change line with two headline improvements:

**Reorganized package structure.** `src/atomic_agentic/` is now
organized into concern-based layers — `exceptions/`, `constants/`,
`models/`, and `utils/` sit below the domain packages (`agents/`,
`tools/`, `workflows/`, `llm/`, `mcp/`, `a2a/`), with `core/`
holding only the shared invocation contract. The dependency topology
is explicit and there are no cross-layer back-edges.

**AtomicResult-family return contract.** Every `AtomicInvokable` —
Tool, Agent, ToolAgent, Workflow, Engine
— now returns a typed `AtomicResult`-family envelope from
`invoke()`/`async_invoke()`. The `.result` field is always the
caller-facing payload; envelope fields (`started_at`, `ended_at`, derived `elapsed_time`,
`invoker_id` -- the invokable's `full_name` -- and subclass-specific fields
like token usage, step traces, and tool-call accounting) carry timing
and provenance. `run_id` appears only on agent results. Workflows return typed `*FlowResult` envelopes
(`SequentialFlowResult`, `IterativeFlowResult`, `GraphFlowResult`, etc.)
with a per-run `trace` of the actual child results produced, opt-out via
`include_trace=False`.

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
|   |-- LLM_Examples/
|   |-- PlanAct_Examples/
|   |-- ReAct_Examples/
|   |-- ScriptAct_Examples/
|   |-- Tool_Examples/
|   |-- Workflow_Examples/
|
|-- docs/
|   |-- CHANGELOG.md
|
|-- images/
|
|-- src/
|   |-- atomic_agentic/
|       |-- a2a/
|       |-- agents/
|       |-- constants/
|       |-- core/
|       |-- exceptions/
|       |-- llm/
|       |-- mcp/
|       |-- models/
|       |-- tools/
|       |-- utils/
|       |-- workflows/
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
{exceptions, constants} → models → utils → core → {agents, tools, workflows, llm, mcp, a2a}
```

| Layer | Packages | What lives here |
|---|---|---|
| Base | `exceptions/`, `constants/` | Exception classes and pure literals/sentinels; zero dependencies |
| Data | `models/` | Dataclasses: `ParamSpec`, runtime records, result envelopes, workflow checkpoints |
| Functions | `utils/` | Pure functions: async bridging, parameter ops, MCP and agent helpers |
| Contract | `core/` | Shared invocation contract: `AtomicInvokable` |
| Domain | `agents/`, `tools/`, `workflows/`, `llm/`, `mcp/`, `a2a/` | Behavior implementations |

------------------------------------------------------------------------

## Running the Tests

Install the test tooling (`pytest`, `pytest-cov` and the A2A HTTP server)
from a source checkout, then run from the repository root:

``` bash
pip install -e ".[dev]"

pytest -m "not llm and not network" -q    # offline suite (what CI runs)
pytest tests/workflows -q                 # one folder
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
