# Atomic-Agentic

## Introduction
**Atomic-Agentic** is an agentic AI framework that addresses two structural problems in real-world AI systems:
* prompts overloaded with deterministic runtime logic
* heterogeneous interfaces between tools, agents, and workflows that require significant plumbing to scale

Atomic-Agentic acts as an **adapter-fist, execution substrate** that moves runtime responsibilities out of natural language and into mandatory, testable code paths and places all executable components behind a single, dictionary-based interface. The result is cleaner composition, less integration friction, and systems that are easier to reason about and extend.

Through Atomic-Agentic, users can build **agentic systems** out of three composable primitives:

- **Tool**: A *dict-first* dapter layer for functions and remote endpoints (like MCP or A2A).
- **Agent**: An autonomous, LLM-driven component that can orchestrate other tools & agents.
- **Workflow**: A deterministic **packaging boundary** for tool, agent, and workflow orchestration and IO normalization.
---

## Installation

To get started with Atomic-Agentic, first clone or download this repository to your prefered location. Then open a terminal and navigate to the repo folder.

From there, activate your preferred python environment. If you don't have the `build` library installed yet, run:

```bash
pip install --upgrade build
# or 
python -m pip install build
```

Then run:

```bash
python -m build
```
This will create a `./dist/` folder with a generated .tgz and .whl file. Use the latest created of either of these files and run:
```bash
pip install ./dist/atomic-agentic-<rest of filename + extension here>
```

Once you've installed Atomic-Agentic, you can explore the [./examples/](./examples/) directory for documented examples and explorations of **Tools, Agents,** and **Workflows**, but for a few quick start examples, continue reading.

---
## Quickstart A: Tools

`Tool` wraps a Python callable and exposes a **dict-first** interface. All Tools execute their internal callable via `invoke(inputs: dict)` instead of positional calls, making each argument explicit, but also removes any positional arguments concern, as well.

```python
from atomic_agentic.tools import Tool
accounts = [
        {"name": "Alice Johnson", "birthdate": "1985-03-15", "account_balance": 15750.50, "annual_interest_rate": 0.025},
        {"name": "Bob Smith", "birthdate": "1990-07-22", "account_balance": 42300.75, "annual_interest_rate": 0.030},
        {"name": "Caleb Donavan", "birthdate": "1974-09-08", "account_balance": 36130.01, "annual_interest_rate": 0.027},
    ]
# 1) Define a plain Python function
def get_account_details(index: int) -> dict:
    """Retrieve bank account details by index."""
    return accounts[index]

# 2) Wrap it as a Tool
tool = Tool(
    function=get_account_details,
    name="get_account_details",
    namespace="banking",
    description="Retrieve bank account details by index."
)

# 3) Inspect the Tool
print(tool.full_name)      # Tool.banking.get_account_details
print(tool.signature)      # Tool.banking.get_account_details(index: int) -> dict
print(tool.parameters)     # [{"name": "index", "index":0, "type":"int", "kind":"POSITIONAL_OR_KEYWORD"}]

# 4) Invoke with dict input (the dict-first contract)
result = tool.invoke({"index": 0})
print(result)  # {"name": "Alice Johnson", ...}
```

---
## Quickstart B: Basic Agents

`Agent` is an autonomous unit that uses an LLM to complete tasks given to it. By default, when calling `invoke`, you only need to give an input string formatted like `{'prompt':<input str here>}`, but you can provide it a custom **pre_invoke** function that preprocesses inputs into a prompt string. This minimizes code-integration friction and broadens your potential input shapes. We use a custom pre-invoke in the example below.

```python
from atomic_agentic.agents import Agent
from atomic_agentic.engines.LLMEngines import OpenAIEngine

# Hardcoded bank accounts
accounts = [
    {"name": "Alice Johnson", "birthdate": "1985-03-15", "account_balance": 15750.50, "annual_interest_rate": 0.025},
    {"name": "Bob Smith", "birthdate": "1990-07-22", "account_balance": 42300.75, "annual_interest_rate": 0.060},
    {"name": "Carol Davis", "birthdate": "1988-11-08", "account_balance": 28900.25, "annual_interest_rate": 0.028},
]

# 1) Define a pre_invoke function with a custom schema (account_index + sector)
def format_finance_request(account_index: int, sector: str) -> str:
    """Format account data + sector into a prompt."""
    account = accounts[account_index]
    return f"""
Customer: {account['name']} (DOB: {account['birthdate']})
Capital: ${account['account_balance']:,.2f}
Rate: {account['annual_interest_rate']*100:.1f}%
---
Desired Sector to invest: {sector.capitalize()}"""

# 2) Create an Agent with custom pre_invoke and role_prompt
llm = OpenAIEngine(model="gpt-4o-mini")
advisor = Agent(
    name="finance_advisor",
    description="Investment advisor.",
    llm_engine=llm,
    role_prompt="""You are an expert financial advisor at a
    bank. You await requests that provide custormer banking data 
    and the sector they wish to invest in. When you receive a 
    request, provide a bulleted list of tickers that fall in 
    that sector and are within their budge to invest in. Also 
    consider how aggressive their annual interest rate is.""",
    pre_invoke=format_finance_request,
)

# 3) Invoke with dict input matching the pre_invoke schema
result = advisor.invoke({"account_index": 0, "sector": "technology"})
print(result)
```

---
## Quickstart C: Tool-Calling Agents

In addition to the traditional string-in string-out Agent class, Atomic-Agentic also supports autonomous **tool-calling** agent classes. `PlanActAgent` is a plan-first agent class that decomposes prompts & tasks into a list of steps and executes those steps as a sequence of tool calls, returning the final result of those calls. See the below example, which registers a list of math tools and can now perform mathematics problems and return actual results instead of just LLM text responses. 

```python
from atomic_agentic.agents import PlanActAgent
from atomic_agentic.engines.LLMEngines import OpenAIEngine
from atomic_agentic.tools.Plugins import MATH_TOOLS

# 1) Create an engine (model is required)
engine = OpenAIEngine(model="gpt-4.1-mini")  # expects OPENAI_API_KEY env var

# 2) Create a planning agent
agent = PlanActAgent(
    name="planner",
    description="Plans and solves simple tasks using tools.",
    llm_engine=engine,
)

# 3) Register tools (callables or Tool instances both work)
agent.batch_register(MATH_TOOLS)

# 4) Invoke (Agent inputs are ALWAYS a mapping)
result = agent.invoke({"prompt": "Compute (6*7) + 5. Return only the number."})
print(result, type(result))
```

---

## Repository Structure

```
Atomic-Agentic/
├── examples/
│   ├── Agent_Examples/
│   ├── PlanAct_Examples/
│   ├── ReAct_Examples/
│   ├── Tool_Examples/
│   ├── RAG_Examples/
│   └── Workflow_Examples/
├── src/
│   └── atomic_agentic/
│       ├── a2a/            # A2A hosting support
│       ├── agents/         # Agents & Tool-Calling Agents
│       ├── core/           # AtomicInvokable class, Exceptions, Prompts, Sentinels
│       ├── engines/        # LLM provider classes
│       ├── tools/          # Tools, and interop support for A2A & MCP
│       └── workflows/      # Single step & composite classes
├── images/
├── README.md
├── pyproject.toml
└── requirements.txt # deprecated file now
```
