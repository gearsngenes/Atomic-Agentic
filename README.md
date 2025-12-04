# Atomic-Agentic

Atomic-Agentic is a Python framework for building reliable, testable “agentic” systems from small, composable pieces:

- **Tools** (portable actions),
- **LLM Engines** (provider adapters),
- **Agents** (stateful planners/executors),
- **Workflows** (deterministic orchestration patterns).

Everything speaks the **same language**: a single, schema-first **dictionary I/O contract**. No DSLs, no black-box orchestrators—just clear, inspectable Python.

---

## Quickstart (30 seconds)

### 1. Install from a local checkout

In your Python virtual environment (`venv`, `virtualenv`, or `conda`), from the **repository root**:

```bash
pip install --upgrade build
python -m build

# On Windows (PowerShell / cmd)
pip install dist\<PACKAGE_FILENAME>.whl

# On macOS / Linux
pip install dist/*.whl
```

> Replace `<PACKAGE_FILENAME>` with the generated wheel filename (e.g. `atomic_agentic-0.1.0-py3-none-any.whl`).
> Alternatively, you can install the source distribution:
>
> ```bash
> pip install dist/*.tar.gz
> ```

If you also want development/runtime dependencies from the repo:

```bash
pip install -r requirements.txt
```

### 2. Configure API keys

Set environment variables for whichever providers you plan to use:

* `OPENAI_API_KEY`
* `MISTRAL_API_KEY`
* `GOOGLE_API_KEY`

(See `LLMEngines.py` for details on each adapter.)

### 3. Run a tiny example

```python
from atomic_agentic.LLMEngines import OpenAIEngine
from atomic_agentic.ToolAgents import PlannerAgent
from atomic_agentic.Toolify import toolify

# 1) Define a local callable and turn it into a Tool
def greet(name: str) -> str:
    print("Generating a greeting...")
    return f"Hello, {name}! Did you know your name has {len(name)} letters?"

# toolify() accepts callables; name/description are required for callables
greet_tool = toolify(
    greet,
    name="greet",
    description="Say hello to a person."
)[0]

# 2) Create a planner-style Agent with a swappable LLM Engine
engine = OpenAIEngine(model="gpt-4o-mini")  # uses OPENAI_API_KEY if api_key not passed
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

Tip: For more complex orchestration patterns, see the `Workflows` module and its `ChainFlow`, `ScatterFlow`, `Selector`, and `MakerChecker` classes.

---

## What it provides

* **Portable actions (Tools)**
  Stateless wrappers around callables (and adapters around MCP tools / remote agents) that:

  * Enforce **dict-only** input (`invoke(inputs: Mapping[str, Any]) -> Any`),
  * Introspect signatures into `arguments_map` (required/optional params),
  * Expose JSON-friendly metadata via `to_dict()` for validation, UIs, and remote calls.

* **Swappable model adapters (LLM Engines)**
  Provider implementations behind a stable:

  * `invoke(messages: list[dict]) -> str`

  contract, plus attachment management via `attach(path)` / `detach(path)`. Orchestration code remains vendor-agnostic, and Agents include a small compatibility shim for engines that accept additional attachment-related parameters.

* **Stateful reasoning (Agents)**
  Components that:

  * Turn validated **input mappings** into prompt strings via a **pre-invoke Tool**,
  * Own persona (system role-prompt), history windowing, and attachments,
  * Call an `LLMEngine` for non-deterministic model outputs,
  * Can themselves be wrapped as Tools or used inside Workflows.

* **Deterministic coordination (Workflows)**
  Wrappers that:

  * Accept a mapping input,
  * Execute Tools / Agents / other Workflows as steps,
  * Package outputs according to an explicit `output_schema` (plus optional `bundle_all`),
  * Support patterns like `ChainFlow`, `ScatterFlow`, `Selector`, and `MakerChecker`,
  * Record checkpoints for observability.

* **Interoperability**

  * **MCP tools** become first-class `Tool` instances via schema extraction and strict argument mapping.
  * **A2A** (python-a2a) allows cross-process Agent calls using the same mapping-only contract.

* **Introspection & UX hooks**

  * Tool catalogs (`Tool.to_dict()` and `PlannerAgent.actions_context()`),
  * Workflow specs (`Workflow.to_dict()`),
  * Useful for documentation, audits, and building UIs.

* **Observability**

  * Workflows append checkpoints (`inputs`, `raw`, `result`, timestamps),
  * Errors are surfaced through domain-specific exception types (`ToolError`, `AgentError`, `WorkflowError`, etc).

---

## Mental model

Think of the stack in three layers, all talking via the **same** mapping-based contract:

* **Base layer**

  * **Tools** wrap callables or MCP tools.
  * **LLM Engines** wrap provider SDKs (OpenAI, Mistral, Gemini).

* **Middle layer**

  * **Agents** own persona + history + attachments, call LLM Engines,
  * `ToolAgent` / `PlannerAgent` manage a toolbox of Tools and orchestrate tool calls.

* **Top layer**

  * **Workflows** orchestrate Tools/Agents deterministically,
  * Provide hard boundaries with `output_schema`, packaging, and checkpointing.

The single, shared contract is:

> “Everything is invoked with a **mapping**, and any higher-level type system sits on top of that.”

---

## Tools — portable, introspectable actions

A `Tool` is a thin wrapper around a callable (or MCP tool / adapter) with a strict dict-only interface:

```python
from atomic_agentic.Tools import Tool

def add(a: float, b: float) -> float:
    return a + b

add_tool = Tool(
    func=add,
    name="add",
    description="Return a + b.",
    source="Math",
)

result = add_tool.invoke({"a": 2, "b": 3})  # 5.0
```

### Key behavior

* **Invocation**

  * `invoke(inputs: Mapping[str, Any]) -> Any`
  * Top-level keys are matched to function parameters.
  * Extra keys are optionally routed to `**kwargs` (if present).
  * Validation errors raise `ToolInvocationError`.

* **Signature & schema**

  * `arguments_map` is an `OrderedDict` describing parameters, required/optional, defaults, etc.
  * `signature` is a human-readable string (for prompts and UIs).
  * `to_dict()` produces a JSON-friendly metadata structure suitable for registries and remote calls.

* **Construction paths**

  * Wrap a callable directly via `Tool(...)`, or
  * Use `toolify(...)` to normalize callables, Agents, MCP tools, and URLs into `Tool` instances.

---

## LLM Engines — provider adapters

Engines are **stateless** adapters around model SDKs (e.g., OpenAI, Mistral, Gemini). They hide provider details behind a simple contract:

```python
from atomic_agentic.LLMEngines import OpenAIEngine

engine = OpenAIEngine(model="gpt-4o-mini")
reply = engine.invoke([
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Say hello."},
])
```

### Core contract

* `invoke(messages: list[dict]) -> str`

  * Messages are provider-agnostic dicts (role/content).
  * Engines return the assistant’s message text as a `str`.
* Attachments:

  * `attach(path: str)` and `detach(path: str)` manage a persistent mapping of file paths.
  * Engines decide how to upload/inline/attach files.
* Agents include a compatibility shim so engines that accept extra attachment-related parameters still work.

### Included engines (see `LLMEngines.py`)

* `OpenAIEngine` — uses the OpenAI Responses API; API key via `OPENAI_API_KEY`.
* `MistralEngine` — Mistral client; API key via `MISTRAL_API_KEY`.
* `GeminiEngine` — Google Gemini; API key via `GOOGLE_API_KEY`.

Each engine focuses on:

* Correct attachment handling,
* Reasonable defaults (temperature, inline cutoffs),
* Clear error surfaces (type / value errors for unsupported file types, etc).

---

## Agents — stateful planners & executors

Agents connect input schemas, prompt building, model calls, and history.

### Base `Agent`

From `Agents.py`:

* **Core behavior**

  * `invoke(inputs: Mapping[str, Any]) -> str`

    1. Validate `inputs` is a mapping.
    2. Run the **pre-invoke Tool**: `pre_invoke.invoke(inputs) -> str` (prompt).

       * By default, this is a strict identity Tool that requires `{"prompt": str}`.
    3. Build messages: `[system?] + [last N turns] + [user(prompt)]`.
    4. Call `llm_engine.invoke(...)` (with attachment shim).
    5. Expect a `str` result; append to history if `context_enabled`.

* **Constructor (simplified)**

  * `Agent(name, description, llm_engine, role_prompt=None, context_enabled=True, pre_invoke=None, history_window=50)`

* **Key properties**

  * `name`, `description`
  * `role_prompt` (system persona)
  * `llm_engine` (must be an `LLMEngine`)
  * `context_enabled: bool`
  * `history_window: int` — send-window in turns (user+assistant pairs)
  * `history: list[dict]` — read-only view of all stored turns
  * `attachments: dict[path, metadata]` — read-only view
  * `pre_invoke: Tool` — mapping → prompt converter

### Tool-using Agents

`ToolAgents.py` introduces agents that manage a toolbox of `Tool` instances.

#### `ToolAgent`

* Inherits from `Agent`.
* Manages a `_toolbox: OrderedDict[str, Tool]` keyed by `Tool.full_name`.
* Core APIs:

  * `register(...)` — normalize and register a single Tool/Agent/URL/callable via `toolify`.
  * `batch_register(iterable)` — bulk registration for Tool/Agent/URL lists.
  * `actions_context()` — returns a **human-readable** summary (ids, signatures, required keys) for prompting.

Subclasses override `_invoke(self, prompt: str) -> Any` to define planning/execution semantics.

#### `PlannerAgent`

A single-shot planner:

* Uses `Prompts.PLANNER_PROMPT` with `{TOOLS}` replaced by `actions_context()`.
* Calls the LLM once to generate a JSON array of steps.
* Ensures the final step is a canonical `_return` tool.
* Executes steps synchronously or, when `run_concurrent=True`, with async fan-out (event-loop-safe fallback).
* Designed for “tool use” style tasks where the model decides which Tools to call in which order.

---

## Workflows — deterministic coordination patterns

`Workflows.py` provides **stateful orchestration boundaries** with explicit output schemas and checkpointing.

### Base `Workflow`

* Public API:

  * `invoke(inputs: Mapping[str, Any]) -> dict`

    * Delegates to `_process_inputs(inputs)` in subclasses,
    * Packages the raw result into a dict shaped by `output_schema`.
* Boundary controls:

  * `output_schema: list[str]` — required, default `["__wf_result__"]` if not supplied.
  * `bundle_all: bool` — optional single-key envelope for non-conforming raw results.
  * `arguments_map` — proxies expected inputs of wrapped components (subclasses implement).
  * `input_schema: list[str]` — derived from `arguments_map` (keys in order).
* Checkpointing:

  * Each `invoke` appends a checkpoint with:

    * `time` (ISO timestamp),
    * `inputs`,
    * `raw` (unpackaged result),
    * `result` (packaged dict).
  * Accessible via a `checkpoints` property (see `Workflows.py`).

### Included patterns

* **`ChainFlow`**
  Sequential composition:

  * Takes a list of steps (Tool/Agent/Workflow wrappers).
  * Forwards each step’s mapping output as the next input.
  * `arguments_map` proxies the first step’s `arguments_map`.
  * Useful as a deterministic, testable alternative to “chain-of-thought” style planning.

* **`ScatterFlow`**
  Scatter/gather pattern:

  * Fans out inputs across multiple branches (Tools/Agents/Workflows).
  * Collects results into a normalized mapping shaped by `output_schema`.
  * Good for parallelizable tasks (e.g., multi-model or multi-datasource queries).

* **`Selector`**
  Conditional router:

  * Uses a judge (Workflow/Tool/Agent) to decide which branch to run.
  * Judge outputs a standardized decision (`JUDGE_RESULT`).
  * Returns the chosen branch’s packaged result.
  * Cleanly separates decision logic from branch implementations.

* **`MakerChecker`**
  Maker–Checker composite:

  * `Maker` produces a draft result.
  * `Checker` reviews/edits/approves or rejects the draft.
  * Optional `Judge` can arbitrate or choose between maker/edits.
  * Composite exposes input schema derived from the Maker’s `arguments_map`.

Together, these patterns help you:

* Keep orchestration deterministic and testable,
* Centralize packaging and error handling,
* Observe behavior via checkpoints.

---

## Interoperability — MCP & A2A

### MCP (Model Context Protocol)

Using `__utils__` and `Toolify`, the library can:

* Discover MCP tools exposed by an MCP server,
* Normalize them into regular `Tool` instances,
* Enforce strict argument mapping (no hidden params),
* Make MCP tools indistinguishable from local Tools to Agents/Workflows.

This means you can mix:

* Local Python callables,
* MCP tools,
* Remote Agents,

in the same Planner/Workflow without changing orchestration code.

### A2A (Agent-to-Agent)

`A2Agents.py` provides schema-driven A2A adapters:

* **`A2AProxyAgent`** — client-side proxy:

  * Sends a single `payload` mapping to a remote A2A Agent via python-a2a,
  * Returns the raw message,
  * Convenience method `invoke_response(inputs) -> dict | None` extracts the response payload.

* **`A2AServerAgent`** — server-side adapter:

  * Wraps a local seed `Agent`,
  * Exposes it as a python-a2a server,
  * Uses the same mapping-only contract as local Agents (`inputs: Mapping`).

This keeps cross-process calls aligned with the rest of the library: **no special types**, just dicts.

---

## Who is it for (and not for)

Atomic-Agentic is for engineers who:

* Prefer **explicit contracts** over “auto-magic”,
* Want to reason about input/output shapes as part of design,
* Care about **interoperability** (MCP, A2A, multiple model vendors),
* Need deterministic orchestration and debuggable behavior.

It is **not** for use cases where you want:

* A one-click, end-user-oriented “agent platform”,
* Heavy-weight UIs or dashboards baked in,
* A proprietary DSL to hide Python.

If you value clarity over wizardry, you’ll feel at home.

---

## Schema-first contract (shared I/O across the stack)

All core components share the same principles:

* **Dict-only inputs**

  * `Tool.invoke(inputs: Mapping[str, Any])`
  * `Agent.invoke(inputs: Mapping[str, Any])`
  * `Workflow.invoke(inputs: Mapping[str, Any])`

* **Explicit schemas**

  * Tools expose `arguments_map` describing required/optional keys.
  * Workflows expose `output_schema` and derived `input_schema`.
  * Agents rely on pre-invoke Tools to adapt richer schemas into a simple `prompt` string.

* **JSON-safe defaults**

  * `to_dict()` methods produce JSON-friendly metadata.
  * Inputs/outputs are kept close to primitives (`str`, `float`, `bool`, lists/dicts).

This makes systems:

* Easier to validate and test,
* Safer to evolve (breaking changes are visible in the schemas),
* Ready for documentation, UI generation, and remote execution.

---

## Next steps

* Explore:

  * `Tools.py` for Tool behavior and introspection.
  * `LLMEngines.py` for provider-specific adapters.
  * `Agents.py` and `ToolAgents.py` for Agents and Planners.
  * `Workflows.py` for orchestration patterns.
  * `Plugins.py` for prebuilt tool lists (e.g., `MATH_TOOLS`, `CONSOLE_TOOLS`, `PARSER_TOOLS`).

* Start small:

  * Wrap a couple of local functions as Tools,
  * Add a `PlannerAgent` with an `LLMEngine`,
  * Compose them with a `ChainFlow` or `MakerChecker` Workflow.

Atomic-Agentic gives you small, sharp building blocks; the rest is just Python.
