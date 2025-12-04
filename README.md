# Atomic-Agentic

Atomic-Agentic is a Python framework for building reliable, testable “agentic” systems from small, composable pieces:

- **Tools** (portable actions),
- **LLM Engines** (provider adapters),
- **Agents** (stateful planners / orchestrators),
- **Workflows** (deterministic orchestration patterns).

Everything speaks the **same language**: a single, schema-first **dictionary I/O contract**.  
No DSLs, no black-box orchestrators—just clear, inspectable Python.

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
````

> Replace `<PACKAGE_FILENAME>` with the generated wheel filename
> (e.g. `atomic_agentic-0.1.0-py3-none-any.whl`).
>
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

(See `LLMEngines.py` and `EmbedEngines.py` for provider-specific details.)

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

Tip: For more complex orchestration patterns, see the `Workflows` module and its
`ChainFlow`, `MapFlow`, `ScatterFlow`, `Selector`, and `MakerChecker` classes.

---

## What it provides

* **Portable actions (Tools)**
  Stateless wrappers around callables (and adapters around Agents, MCP tools, and remote agents) that:

  * Enforce **dict-only** input: `invoke(inputs: Mapping[str, Any]) -> Any`,
  * Introspect signatures into `arguments_map` (required/optional params),
  * Expose JSON-friendly metadata via `to_dict()` for validation, UIs, and remote calls.

* **Swappable model adapters (LLM Engines)**
  Provider implementations behind a stable:

  * `invoke(messages: list[dict]) -> str`

  contract, plus attachment management via `attach(path)` / `detach(path)`. Orchestration code remains vendor-agnostic, and Agents include a small compatibility shim for engines that accept additional attachment-related parameters.

* **Embedding engines (for vector search/RAG)**
  A parallel stack of `EmbedEngine` implementations to turn text into vectors, with the same emphasis on simple contracts and provider-agnostic usage.

* **Stateful reasoning and orchestration (Agents)**
  Components that:

  * Turn validated **input mappings** into prompt strings via a **pre-invoke Tool**,
  * Own persona (system role-prompt), history windowing, and attachments,
  * Call an `LLMEngine` for non-deterministic model outputs,
  * Optionally orchestrate Tools iteratively (`OrchestratorAgent`) or as a single-shot planner (`PlannerAgent`),
  * Can themselves be wrapped as Tools or used inside Workflows.

* **Deterministic coordination (Workflows)**
  Wrappers that:

  * Accept a mapping input,
  * Execute Tools / Agents / other Workflows as steps,
  * Package outputs according to an explicit `output_schema` (plus optional `bundle_all`),
  * Support patterns like `ChainFlow`, `MapFlow`, `ScatterFlow`, `Selector`, and `MakerChecker`,
  * Record checkpoints for observability.

* **Interoperability**

  * **MCP**: Remote MCP tools become first-class `MCPProxyTool` instances via schema extraction and strict argument mapping.
  * **A2A**: python-a2a adapters allow cross-process Agent calls using the same mapping-only contract.

* **Introspection & UX hooks**

  * Tool catalogs (`Tool.to_dict()` and `PlannerAgent.actions_context()`),
  * Workflow specs (`Workflow.to_dict()`),
  * Useful for documentation, audits, and building UIs.

* **Observability**

  * Workflows append checkpoints (`inputs`, `raw`, `result`, timestamps),
  * Errors are surfaced through domain-specific exception types.

---

## Mental model

Think of the stack in three layers, all talking via the **same** mapping-based contract:

* **Base layer**

  * **Tools** wrap callables, Agents, and MCP tools.
  * **LLM Engines** wrap provider SDKs (OpenAI, Mistral, Gemini, LlamaCpp).
  * **Embed Engines** wrap provider embedding APIs.

* **Middle layer**

  * **Agents** own persona + history + attachments, call LLM Engines,
  * `ToolAgent` / `PlannerAgent` / `OrchestratorAgent` manage a toolbox of Tools and orchestrate tool calls.

* **Top layer**

  * **Workflows** orchestrate Tools/Agents deterministically,
  * Provide hard boundaries with `output_schema`, packaging, and checkpointing.

The single, shared contract is:

> “Everything is invoked with a **mapping**, and any higher-level type system sits on top of that.”

---

## Tools — portable, introspectable actions

A `Tool` is a thin wrapper around a callable (or MCP tool / adapter / Agent) with a strict dict-only interface:

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

### Core behavior

* **Invocation**

  * `invoke(inputs: Mapping[str, Any]) -> Any`
  * Top-level keys are matched to function parameters.
  * Extra keys are optionally routed to `**kwargs` (if present).
  * Validation errors raise `ToolInvocationError`.

* **Signature & schema**

  * `arguments_map` is an `OrderedDict` describing parameters, required/optional, defaults, etc.
  * `signature` is a human-readable string (for prompts and UIs).
  * `to_dict()` produces a JSON-friendly metadata structure suitable for registries and remote calls.

### Tool adapters

Atomic-Agentic provides several adapter types built on top of `Tool`:

#### AgentTool — Agents as Tools

Defined in `Agents.py`:

* **`AgentTool`** exposes an `Agent` as a `Tool`:

  * Metadata:

    * `type = "agent"`
    * `source = agent.name`
    * `name = "invoke"`
    * `description = agent.description`
  * **Schema exposure:**

    * Mirrors the Agent’s **pre-invoke Tool** call-plan.
    * `arguments_map`, required/optional sets, and var-args flags are copied from the Agent’s `pre_invoke` Tool.
  * `invoke(inputs)` simply calls `agent.invoke(inputs)` and re-raises errors as `ToolInvocationError`.

This makes Agents indistinguishable from Tools to higher layers (planner agents, workflows, registries).

#### MCPProxyTool — remote MCP tools as local Tools

Defined in `Tools.py`:

* **`MCPProxyTool`** proxies a single MCP server tool as a normal dict-first Tool:

  * On construction:

    * Opens a short-lived MCP session,
    * Calls `initialize` / `list_tools`,
    * Extracts the tool’s `inputSchema` and description,
    * Builds a keyword-only signature in server property order for `arguments_map`.
  * On `invoke(inputs)`:

    * Opens a fresh MCP session,
    * Calls `initialize` / `call_tool`,
    * Returns **structured content** when possible or a text fallback,
    * Closes cleanly (no background loop).

MCPProxyTool is what the library uses under the hood to call MCP tools remotely while still enforcing the same dict-only, schema-first behavior as local Tools.

#### toolify — normalizing arbitrary inputs into Tools

`Toolify.toolify(...)` is the main helper for getting everything into Tool form:

* If given a `Tool` → returns `[Tool]`.
* If given an `Agent` → returns `[AgentTool(agent)]`.
* If given an MCP URL string → discovers MCP tools and returns a list of `MCPProxyTool` instances (one per MCP tool).
* If given a callable → validates name/description and wraps it in a `Tool`.

`ToolAgent.register` and `ToolAgent.batch_register` use `toolify` internally, so callers can pass callables, Agents, MCP endpoints, or Tools interchangeably.

---

## LLM Engines — provider adapters

Engines are **stateless** adapters around model SDKs (e.g., OpenAI, Mistral, Gemini, LlamaCpp). They hide provider details behind a simple contract:

```python
from atomic_agentic.LLMEngines import OpenAIEngine

engine = OpenAIEngine(model="gpt-4o-mini")
reply = engine.invoke([
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user",   "content": "Say hello."},
])
```

### Core contract

* `invoke(messages: list[dict]) -> str`

  * Messages are provider-agnostic dicts (`role` / `content`).
  * Engines return the assistant’s message text as a `str`.

* Attachments:

  * `attach(path: str)` and `detach(path: str)` manage a persistent mapping of file paths.
  * Engines decide how to upload/inline/attach files.
  * Agents include a compatibility shim so engines that accept extra attachment-related parameters still work.

### Included engines (see `LLMEngines.py`)

* `OpenAIEngine`
  Uses OpenAI’s Responses API; API key via `OPENAI_API_KEY`.

* `MistralEngine`
  Mistral client; API key via `MISTRAL_API_KEY`.

* `GeminiEngine`
  Google Gemini client; API key via `GOOGLE_API_KEY`.

* `LlamaCppEngine`
  Local inference using `llama_cpp`. Useful when you want a local/offline LLM with the same engine interface.

---

## Embedding Engines — vectorization layer

`EmbedEngines.py` contains embedding engines that follow a simple contract:

* Base `EmbedEngine`:

  * `embed(texts: Sequence[str]) -> list[list[float]]`
  * Handles normalization, dimension consistency, and type coercion to `list[float]`.

### Included embedding engines

* `OpenAIEmbedEngine` — OpenAI embeddings.
* `GeminiEmbedEngine` — Google embedding APIs via `google.genai`.
* `MistralEmbedEngine` — Mistral embeddings.
* `LlamaCppEmbedEngine` — local inference using `llama_cpp`.

These can be dropped into your own vector stores / RAG stacks without coupling orchestration logic to a specific provider.

---

## Agents — stateful planners & orchestrators

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
  * `history: list[dict]` — read-only view of stored turns
  * `attachments: dict[path, metadata]` — read-only view
  * `pre_invoke: Tool` — mapping → prompt converter

### ToolAgent — Agents with a toolbox

`ToolAgents.py` introduces tool-using Agents.

#### `ToolAgent`

* Inherits from `Agent`.
* Manages a `_toolbox: OrderedDict[str, Tool]` keyed by `Tool.full_name`.
* Core APIs:

  * `register(...)` — normalize and register a single Tool/Agent/URL/callable via `toolify`.
  * `batch_register(iterable)` — bulk registration for Tool/Agent/URL lists.
  * `actions_context()` — returns a **human-readable** summary (ids, signatures, required keys) for prompting.

Subclasses override `_invoke(self, prompt: str) -> Any` to define planning/execution semantics.

#### `PlannerAgent` — single-shot planner

A single-shot planner that uses the LLM once to produce a full plan:

* Uses `Prompts.PLANNER_PROMPT` with `{TOOLS}` replaced by `actions_context()`.
* LLM returns a JSON array of steps.
* Planner:

  * Validates the plan,
  * Ensures the final step is a canonical `_return` tool (exposed as `RETURN_KEY`),
  * Executes each step in order (optionally concurrently when `run_concurrent=True`),
  * Returns the final `_return` payload.

Suitable for “tool use” style tasks where a one-shot plan is acceptable.

#### `OrchestratorAgent` — iterative orchestrator

`OrchestratorAgent` is an **iterative, schema-driven tool orchestrator**:

* Uses `Prompts.ORCHESTRATOR_PROMPT`, again with `{TOOLS}` filled from `actions_context()`.

* Each iteration, the LLM returns exactly one JSON object:

  ```jsonc
  {
    "step_call": { "function": "<type>.<source>.<name>", "args": { ... } },
    "explanation": "why this step",
    "status": "CONTINUE" | "COMPLETE"
  }
  ```

  No prose, no markdown fences.

* OrchestratorAgent:

  * Looks up the Tool by its canonical `full_name`,
  * Calls `Tool.invoke(args)` and records the result,
  * Maintains a structured `_previous_steps` list (function, args, explanation, result, and flags),
  * Injects a compact “Previous Steps” summary into each iteration’s prompt,
  * Iterates until:

    * `status == "COMPLETE"`, or
    * `max_steps` steps executed, or
    * `max_failures` failed iterations.

* History policy:

  * The full, multi-step reasoning is visible **within** each call’s prompt (but not exploded into separate stored turns).
  * At the end, if `context_enabled=True`, the original prompt and the final answer are appended to `history`.

This pattern is useful when you want the model to **plan, execute, and adapt** step-by-step, while still keeping control over schemas, tools, and limits.

---

## Workflows — deterministic coordination patterns

`Workflows.py` provides **stateful orchestration boundaries** with explicit output schemas and checkpointing.

### Base `Workflow`

* Public API:

  * `invoke(inputs: Mapping[str, Any]) -> dict`

    * Delegates to `_process_inputs(inputs)` in subclasses,
    * Packages the raw result into a dict shaped by `output_schema`.

* Boundary controls:

  * `output_schema: list[str]` — required; defaults to `[WF_RESULT]` (where `WF_RESULT = "__wf_result__"`) if not supplied.
  * `bundle_all: bool` — optional single-key envelope (must align with a single-key `output_schema`).
  * `arguments_map` — proxies expected inputs of wrapped components (subclasses implement).
  * `input_schema: list[str]` — derived from `arguments_map`.

* Checkpointing:

  * Each `invoke` appends a checkpoint with:

    * `time` (ISO timestamp),
    * `inputs`,
    * `raw` (unpackaged result),
    * `result` (packaged dict).
  * Accessible via a `checkpoints` property.

Workflows intentionally **don’t** declare their own input schema; they surface the schema of what they wrap.

### ToolFlow and AgentFlow — wrapping existing components

To make composition uniform, Atomic-Agentic wraps Tools and Agents in Workflow adapters:

* **`AgentFlow`**

  * Wraps a single `Agent`.
  * Inputs are forwarded as `agent.invoke(inputs)`.
  * `arguments_map` is a read-only proxy to `agent.arguments_map`.
  * Output packaging handled by `Workflow`.

* **`ToolFlow`**

  * Wraps a single `Tool` or callable.
  * Callables are first turned into a `Tool`.
  * Inputs are forwarded as `tool.invoke(inputs)`.
  * `arguments_map` proxies `tool.arguments_map`.

A helper, **`_to_workflow(obj)`**, normalizes:

* `Workflow` → returned as-is,
* `Agent` → wrapped as `AgentFlow`,
* `Tool` / callable → wrapped as `ToolFlow`.

All composite flows below use `_to_workflow`, so you can pass Tools, Agents, callables, or Workflows interchangeably.

### Included patterns

#### ChainFlow — sequential composition

`ChainFlow` sequentially composes a list of steps (any combination of `ToolFlow`, `AgentFlow`, or other `Workflow` instances):

* Forwards each step’s mapping output as the next step’s input.
* `arguments_map` proxies the **first** step’s `arguments_map`.
* `output_schema` / `bundle_all` define the final result shape; the **last step** is aligned to these overlay values.

Useful as a deterministic, testable alternative to “chain-of-thought” style planning.

#### MapFlow — per-branch payload fan-out

`MapFlow` is a **tailored fan-out** pattern where each branch gets its own payload:

* Inputs:

  * Expects a mapping of `branch_name -> payload_mapping`.
  * Unknown branch names raise `ValidationError`.
  * If a branch has no payload:

    * `flatten=False` → branch appears in output as `None`,
    * `flatten=True` → branch is skipped entirely.

* Schema surfacing:

  * `arguments_map` is proxied from an internal “schema supplier” Tool whose keyword-only parameters mirror the current branch names, annotated as `dict[str, Any]`.
  * When branches change, the supplier is rebuilt; `input_schema` remains derived from `arguments_map`.

* Outputs:

  * `flatten=False` (default):

    * Ordered mapping `branch_name -> branch_result | None` in branch order.

  * `flatten=True`:

    * Left-to-right merge of branch outputs:

      * Plain dicts are merged key-by-key,
      * Special envelopes `{WF_RESULT: ...}` / `{JUDGE_RESULT: ...}` and non-dicts are stored under the branch name.

Good when each branch operates on a **different** part of the input.

#### ScatterFlow — broadcast fan-out

`ScatterFlow` is a broadcast fan-out where **every branch** sees the **same** validated input mapping:

* Contract:

  * Every branch must have an `input_schema` **set-equal** to the others.
  * At construction and on `add_branch`, new branches are checked for schema equality; mismatches raise `ValidationError`.
  * `arguments_map` proxies the first branch’s `arguments_map`.

* Execution:

  * The same `inputs` mapping is dispatched to every branch concurrently via asyncio.
  * Results can be:

    * Collected per-branch, or
    * Flattened via a merge policy similar to `MapFlow` (branch results merged left-to-right, with special handling for envelopes and collisions).

Good for multi-model or multi-datasource queries where everyone needs the same input.

#### Selector — judge-based router

`Selector` uses a judge component to decide which branch to execute:

* Judge:

  * Is itself a `Workflow`/Tool/Agent (normalized via `_to_workflow`).
  * Returns a standardized decision under the `JUDGE_RESULT` key.

* Flow:

  1. Inputs go to the judge,
  2. Judge picks a branch name,
  3. Inputs are forwarded to the chosen branch,
  4. The branch’s packaged result becomes the Selector’s result.

Branches are stored in an ordered mapping; name collisions are handled by a `name_collision_policy` (`"fail_fast" | "skip" | "replace"`) shared with `ScatterFlow`.

#### MakerChecker — maker–checker composite

`MakerChecker` composes maker/checker (and optionally judge) workflows:

* Maker:

  * Produces a draft result from the initial inputs.

* Checker:

  * Reviews, edits, or approves the draft.

* Optional Judge:

  * Can arbitrate or choose between drafts/edits in more complex setups.

`MakerChecker` exposes an input schema derived from the Maker’s `arguments_map` and packages the final result via its own `output_schema`.

---

## Interoperability — MCP & A2A

### MCP (Model Context Protocol)

Atomic-Agentic treats MCP tools as first-class primitives via `MCPProxyTool` and `toolify`:

* It can:

  * Connect to an MCP server,
  * Discover tools and their `inputSchema`,
  * Materialize them as `MCPProxyTool` instances.

* Each `MCPProxyTool`:

  * Presents a local `Tool`-like interface (`invoke(mapping)`),
  * Enforces a keyword-only schema based on the remote `inputSchema`,
  * Returns structured content whenever possible.

You can mix:

* Local Python callables,
* MCP tools (via `MCPProxyTool`),
* AgentTools,

in the same Planner/Orchestrator/Workflow without changing orchestration code.

### A2A (Agent-to-Agent)

`A2Agents.py` provides schema-driven A2A adapters:

* **`A2AProxyAgent`** — client-side proxy:

  * `invoke(inputs)` sends a single `payload` mapping to a remote A2A Agent via python-a2a and returns the raw message.
  * Convenience `invoke_response(inputs) -> dict | None` extracts the response payload when present.

* **`A2AServerAgent`** — server-side adapter:

  * Wraps a local seed `Agent`,
  * Exposes it as a python-a2a server,
  * Uses the same mapping-only contract as local Agents.

Cross-process calls stay aligned with the rest of the library: **no special types**, just dicts.

---

## Factory helpers — reconstructing from dicts

`Factory.py` provides helpers to reconstruct objects from their `to_dict()` snapshots:

* `load_llm(data: Mapping[str, Any], **kwargs) -> LLMEngine`
* `load_tool(data: Mapping[str, Any], **kwargs) -> Tool`
* `load_agent(data: Mapping[str, Any], **kwargs) -> Agent`

These use the `"provider"` / `"type"` metadata in the snapshots and support:

* Core Engines (`OpenAIEngine`, `MistralEngine`, `GeminiEngine`, `LlamaCppEngine`),
* Tools (`Tool`, `AgentTool`, `MCPProxyTool`),
* Agents (`Agent`, `PlannerAgent`, `OrchestratorAgent`, `A2AProxyAgent`, `A2AServerAgent`).

This is useful for:

* Config-driven systems (YAML/JSON configs → objects),
* Persistence and replay,
* Simple remote registries.

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

  * `Tools.py` for Tool behavior, `AgentTool`, `MCPProxyTool`, and introspection.
  * `LLMEngines.py` for provider-specific LLM adapters.
  * `EmbedEngines.py` for embedding engines.
  * `Agents.py` and `ToolAgents.py` for Agents, Planners, and Orchestrators.
  * `Workflows.py` for orchestration patterns (`ChainFlow`, `MapFlow`, `ScatterFlow`, `Selector`, `MakerChecker`, `AgentFlow`, `ToolFlow`).
  * `Plugins.py` for prebuilt tool lists (e.g., `MATH_TOOLS`, `CONSOLE_TOOLS`, `PARSER_TOOLS`).

* Start small:

  * Wrap a couple of local functions as Tools,
  * Add a `PlannerAgent` or `OrchestratorAgent` with an `LLMEngine`,
  * Compose them with a `ChainFlow`, `MapFlow`, `ScatterFlow`, or `MakerChecker` Workflow.

Atomic-Agentic gives you small, sharp building blocks; the rest is just Python.
