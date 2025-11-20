# Atomic-Agentic

## What is Atomic-Agentic?

Atomic-Agentic is a Python framework for building reliable, provider-agnostic **tool- and agent-based** systems. It offers a small set of orthogonal primitives—**Tools**, **LLM Engines**, **Agents**, and **Workflows**—that compose cleanly. Local callables, in-process Agents, and remote tools (via **MCP**) are unified behind one Tool abstraction; remote Agents interoperate over **A2A**. There are no DSLs or black-box orchestrators here—just clear, inspectable contracts in plain Python.

## What it provides

* **Portable actions (Tools):** Stateless wrappers around callables (and adapters for Agents/MCP tools) that expose an introspectable signature and a JSON-safe `arguments_map`. Tools accept **dictionary-only** inputs and surface required/optional parameters for validation, UIs, and remote calls.
* **Swappable model adapters (LLM Engines):** Provider implementations behind a stable `invoke(messages, file_paths=…) -> str` contract. Orchestration code remains vendor-agnostic.
* **Stateful reasoning (Agents):** Components that turn validated input into prompts, call an **LLM Engine**, and manage history/attachments. Agents can also use Tools for environment interaction and multi-step execution.
* **Deterministic coordination (Workflows):** Wrappers that execute tools/agents and **package** raw returns into declared output schemas for predictable downstream chaining and easy testing.
* **Interoperability:**

  * **MCP tools** become first-class Tools via schema extraction and strict argument mapping.
  * **A2A** enables cross-process Agent calls using the same mapping-only payload contract.
* **Introspection & UX hooks:** Action catalogs (ids, signatures, required keys) and `to_dict()` outputs support documentation, audits, and UI generation.
* **Observability:** Checkpoints capture inputs, raw outputs, packaged outputs, and timing to aid debugging and post-mortems.

## Mental model

Think of the stack in three layers, connected by one contract:

* **Base**: **Tools** wrap callables; **LLM Engines** wrap providers.
* **Middle**: **Agents** combine Tools + Engines to plan and execute work, holding persona and history.
* **Top**: **Workflows** coordinate Tools and Agents in recognizable patterns, enforcing deterministic hand-offs.

Adapters “flatten” the ecosystem so local functions, Agents, and remote tools all present the same Tool interface.

## Tools — portable, introspectable actions

Tools turn Python functions (and via adapters, MCP tools or even Agents) into consistent, discoverable actions. Each Tool publishes: a canonical signature, a JSON-safe `arguments_map` (names, required keys, defaults), and a dict-only invoke surface. This makes Tools easy to validate, list, document, test, and reuse across processes.

## LLM Engines — provider adapters

Engines are stateless adapters around model SDKs. They accept structured chat messages (and optional file paths) and return a string. Attachment policies and vendor specifics live here, not in your orchestration code—so switching providers doesn’t force architectural changes.

## Agents — stateful planners & executors

Agents own persona, history windowing, and attachments. They transform validated inputs (often via a pre-invoke Tool) into prompts and call an **LLM Engine**. Tool-using Agents (e.g., Planner/Orchestrator) consume the same Tool specs to decide and execute steps. Any Agent can be **adapted into a Tool** for reuse inside other Agents or Workflows.

## Workflows — deterministic coordination patterns

Workflows wrap Tools or Agents as steps and manage inputs/outputs end-to-end. They offer recognizable patterns such as **Chain** (sequential), **Selector** (route to one branch), **Scatter/Map** (fan-out and recombine), and **Maker–Checker** (produce → judge). Crucially, Workflows **package** raw step returns to declared shapes so downstream steps don’t see ad-hoc “mystery dicts.”

## Interoperability — adapters, remote tools, remote agents

`toolify(…)` converts local callables, Agents, and **MCP** tools into first-class Tools. MCP discovery pulls remote tool schemas and maps them to strict argument plans; calls look local to the planner. **A2A** applies the same mapping-only contract to remote Agents. Everything remains listable and inspectable (names, signatures, required keys).

## Observability & operations

At each boundary, Workflows and Agents record checkpoints: inputs, raw outputs, packaged outputs, and timings. Deterministic packaging reduces flaky integrations; explicit errors for missing/unknown fields surface early. The result is behavior you can debug, test, and trust in CI/CD.

## Quickstart (30 seconds)
In your python virtual environment (venv, virtualenv, or conda), open the Atomic-Agentic repository folder.

This project is packaged for pip install. To build and install the package locally, run:

```powershell
pip install --upgrade build
python -m build
pip install dist\<PACKAGE_FILENAME>.whl
```

- Replace `<PACKAGE_FILENAME>` with the generated wheel filename (for example, `atomic_agentic-0.1.0-py3-none-any.whl`) or install a source distribution:

```powershell
pip install dist\*.tar.gz
```

If you want to install development/runtime dependencies from the repository (optional), you can still run:

```powershell
pip install -r requirements.txt
```

Then initialize API keys for any LLM providers you plan to use (OpenAI, Mistral, Gemini, etc.) as environment variables.

After installing the package you can run example code. Example usage (importing from the installed package):

```python
from atomic_agentic.LLMEngines import OpenAIEngine
from atomic_agentic.ToolAgents import PlannerAgent
from atomic_agentic.ToolAdapters import toolify

# 1) Define a local callable and turn it into a Tool
def greet(name: str) -> str:
  print("Generating a greeting...")
  return f"Hello, {name}! Did you know that your name has {len(name)} letters in it?"

# toolify() accepts callables; name/description are required for callables
greet_tool = toolify(greet, name="greet", description="Say hello to a person")[0]

# 2) Create a planner-style Agent with a swappable LLM Engine
engine = OpenAIEngine(model="gpt-4o-mini")   # uses OPENAI_API_KEY env var if not passed explicitly
planner = PlannerAgent(
  name="Planner",
  description="Plans tool calls and returns the final result.",
  llm_engine=engine,
)

# 3) Register the Tool and inspect the action catalog
planner.register(greet_tool)
print(planner.actions_context())  # shows ids, signatures, and required keys

# 4) Run a tiny task (pre-invoke expects a mapping with a 'prompt' string by default)
result = planner.invoke({"prompt": "Greet Ada Lovelace and then return the message."})
print("Result:", result)
```

Tip: For more patterns, see the example scripts for Chain, Selector, Scatter/Map, and Maker–Checker.

## Who is it for (and not for)

Atomic-Agentic is for engineers who want explicit contracts, composable parts, and provider choice. It is not a DSL and not a “do-everything” platform; there is no hidden state or magic. If you value clarity over wizardry, you’ll feel at home.

## Schema-first contract (shared I/O across the stack)

All invocations use **dictionary-only** inputs. Required keys are enforced; unknown keys are rejected; defaults are JSON-safe. Tools, Agents (via pre-invoke), and Workflows publish arguments and outputs in a **consistent, printable form**. Workflows then **package** outputs to declared schemas so downstream steps receive predictable shapes. These practices make systems auditable, composable, and safe to evolve.
