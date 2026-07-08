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
-   **StructuredInvokable** -- An explicit output-projection adapter
    that wraps any other primitive and maps its raw output into a
    validated, dictionary-based output schema.
-   **Workflow** -- An **orchestration and checkpointing** layer that
    coordinates tools, engines, agents, and other workflows into structured
    pipelines.

Together, these primitives form a composable system where **LLMs handle
reasoning while deterministic execution logic lives in code.**

------------------------------------------------------------------------

## Installation

### Install directly from GitHub

``` bash
pip install git+https://github.com/gearsngenes/Atomic-Agentic.git
```

### Install from source

Clone the repository and install from the generated build artifacts:

``` bash
git clone https://github.com/gearsngenes/Atomic-Agentic
cd Atomic-Agentic

pip install --upgrade build
python -m build
pip install ./dist/atomic-agentic-*.whl
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
    namespace="banking",
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
from atomic_agentic.agents import Agent
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

advisor = Agent(
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
from atomic_agentic.tools.prebuilt import MATH_TOOLS

engine = OpenAIEngine(model="gpt-4.1-mini")

agent = PlanActAgent(
    name="planner",
    description="Plans and solves tasks using tools.",
    llm_engine=engine,
)

agent.batch_register(MATH_TOOLS)

result = agent.invoke({"prompt": "Compute (6*7) + 5. Return only the number."})
print(result.result)
```

------------------------------------------------------------------------

## Structured Output: StructuredInvokable

Atomic-Agentic uses `StructuredInvokable` to transform and validate outputs from tools, agents, or other AtomicInvokable objects. This ensures outputs conform to a specified schema and handles missing or extra fields robustly.

**Minimal Example:**

```python
from atomic_agentic import StructuredInvokable
from atomic_agentic.tools import Tool

def raw_tool(x, y):
    return x + y, x * y

schema = ["sum", "product"]
structured = StructuredInvokable(component=Tool(raw_tool), output_schema=schema)
result = structured.invoke({"x": 2, "y": 3})
print(result.result)  # {'sum': 5, 'product': 6}
```

See the `StructuredInvokable` docstring for advanced options (absent value handling, mapping extras, etc).

------------------------------------------------------------------------

## Workflows

Workflows orchestrate Atomic-Agentic primitives into deterministic pipelines. They provide patterns for composition, branching, iteration, and parallelism, enabling you to build complex agentic systems from modular components.

**Workflow classes include:**
- `BasicFlow` â€“ wraps a single component
- `SequentialFlow` â€“ chains steps in sequence
- `ParallelFlow` â€“ runs branches concurrently
- `RoutingFlow` â€“ routes input to a selected branch
- `IterativeFlow` â€“ loops until a judge condition is met

**Note:**
Workflows return typed result envelopes (`*FlowResult`) with `.result` and full run metadata. Use `StructuredInvokable` explicitly when you need to project a step's output dict to a fixed schema â€” workflow steps themselves do not apply schema projection.

**For practical workflow usage and advanced patterns, see the examples in:**
`examples/Workflow_Examples/`

------------------------------------------------------------------------

## What's New in v2

v2 is a deliberate breaking-change line with two headline improvements:

**Reorganized package structure.** `src/atomic_agentic/` is now
organized into concern-based layers â€” `exceptions/`, `constants/`,
`models/`, and `utils/` sit below the domain packages (`agents/`,
`tools/`, `workflows/`, `engines/`, `mcp/`, `a2a/`), with `core/`
holding only the shared invocation contract. The dependency topology
is explicit and there are no cross-layer back-edges.

**AtomicResult-family return contract.** Every `AtomicInvokable` â€”
Tool, Agent, ToolAgent, Workflow, Engine, Command, StructuredInvokable
â€” now returns a typed `AtomicResult`-family envelope from
`invoke()`/`async_invoke()`. The `.result` field is always the
caller-facing payload; envelope fields (`run_id`, `started_at`,
`ended_at`, `elapsed_s`, `invoker_id`, and subclass-specific fields
like token usage, step traces, and tool-call accounting) carry timing
and provenance. Workflows return typed `*FlowResult` envelopes
(`SequentialFlowResult`, `IterativeFlowResult`, etc.) with per-run
history accessible via checkpoint helpers.

For a full breakdown of breaking changes and a v1â†’v2 migration guide:
- [`docs/MIGRATION.md`](docs/MIGRATION.md) â€” v1â†’v2 migration guide *(coming soon)*
- [`docs/CHANGELOG.md`](docs/CHANGELOG.md) â€” full release history

------------------------------------------------------------------------

## Repository Structure

    Atomic-Agentic/
    â”œâ”€â”€ examples/
    â”‚   â”œâ”€â”€ Agent_Examples/
    â”‚   â”œâ”€â”€ Agentic_Research/
    â”‚   â”œâ”€â”€ Invokable_Examples/
    â”‚   â”œâ”€â”€ LLM_Examples/
    â”‚   â”œâ”€â”€ PlanAct_Examples/
    â”‚   â”œâ”€â”€ ReAct_Examples/
    â”‚   â”œâ”€â”€ Tool_Examples/
    â”‚   â””â”€â”€ Workflow_Examples/
    â”‚
    â”œâ”€â”€ docs/
    â”‚   â””â”€â”€ CHANGELOG.md
    â”‚
    â”œâ”€â”€ images/
    â”‚
    â”œâ”€â”€ src/
    â”‚   â””â”€â”€ atomic_agentic/
    â”‚       â”œâ”€â”€ a2a/
    â”‚       â”œâ”€â”€ agents/
    â”‚       â”œâ”€â”€ constants/
    â”‚       â”œâ”€â”€ core/
    â”‚       â”œâ”€â”€ engines/
    â”‚       â”œâ”€â”€ exceptions/
    â”‚       â”œâ”€â”€ mcp/
    â”‚       â”œâ”€â”€ models/
    â”‚       â”œâ”€â”€ tools/
    â”‚       â”œâ”€â”€ utils/
    â”‚       â”œâ”€â”€ workflows/
    â”‚       â”œâ”€â”€ __init__.py
    â”‚       â”œâ”€â”€ _version.py
    â”‚       â””â”€â”€ py.typed
    â”‚
    â”œâ”€â”€ tests/
    â”œâ”€â”€ README.md
    â”œâ”€â”€ pyproject.toml
    â””â”€â”€ requirements.txt

### Package topology

The subpackages of `src/atomic_agentic/` form a strict layered
dependency hierarchy â€” no back-edges:

```
{exceptions, constants} â†’ models â†’ utils â†’ core â†’ {agents, tools, workflows, engines, mcp, a2a}
```

| Layer | Packages | What lives here |
|---|---|---|
| Base | `exceptions/`, `constants/` | Exception classes and pure literals/sentinels; zero dependencies |
| Data | `models/` | Dataclasses: `ParamSpec`, runtime records, result envelopes, workflow checkpoints |
| Functions | `utils/` | Pure functions: async bridging, parameter ops, MCP and agent helpers |
| Contract | `core/` | Shared invocation contract: `AtomicInvokable`, `Command`, `StructuredInvokable` |
| Domain | `agents/`, `tools/`, `workflows/`, `engines/`, `mcp/`, `a2a/` | Behavior implementations |

------------------------------------------------------------------------

## Changelog

See [`docs/CHANGELOG.md`](https://github.com/gearsngenes/Atomic-Agentic/blob/main/docs/CHANGELOG.md)
for the release history of notable changes.
