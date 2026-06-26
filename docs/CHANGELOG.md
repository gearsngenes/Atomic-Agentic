# Changelog

All notable changes to Atomic-Agentic are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Atomic-Agentic's v2 line is currently pre-1.0 alpha (`2.0.0aN`).

## [2.0.0a12] - 2026-06-26

### Fixed

- `ToolAgent` sync `invoke` crash when called from within a running event loop
  — replaced `asyncio.run(...)` with `run_coro_sync(...)`.
- `ToolAgent` `TypeError` when a placeholder resolves to an unhashable value
  in a set-typed argument — set comprehension replaced with a list comprehension.
- `LLMRecord` construction: `messages` was passed as a list literal where
  `LLMRecord.__post_init__` requires a tuple; corrected in `Agent._ainvoke` and
  `Agent._invoke`.
- `OpenAIEngine._extract_token_usage`: unguarded `output_tokens_details` and
  `input_tokens_details` attribute access — now guarded before accessing
  sub-fields.
- `AgentRecord.__post_init__` now enforces that `prev`, when set, points to a
  completed record (`final_result is not None`).
- `ToolAgentRecord` was missing a `to_dict()` override; `blackboard_start` and
  `blackboard_end` fields were silently dropped from serialization.

### Added

- Native async paths for `ToolAgent` hooks: `_ainitialize_run_state` and
  `_aprepare_next_batch` on the `ToolAgent` base (default: `asyncio.to_thread`
  wrap); `PlanActAgent._agenerate_plan` / `_ainitialize_run_state` and
  `ReActAgent._agenerate_next_step` / `_aprepare_next_batch` use `async_invoke`
  directly, avoiding thread-pool dispatch on every LLM call.
- `ReActStepMeta` dataclass — consolidates `observable: int` and
  `description: str` per step slot in `ReActRunState`, replacing former parallel
  lists.

### Changed

- `AzureOpenAIEngine` and `BedrockEngine` stub classes (which raised a
  confusing `TypeError` on instantiation) removed.
- Dead `token_usage is not None` filter in `Agent.make_result` removed —
  `LLMResult.token_usage` is always a `TokenUsage` instance.
- Redundant `tool_calls_limit` runtime guards in `ReActAgent._initialize_run_state`,
  `_generate_next_step`, and `_agenerate_next_step` removed — the invariant is
  enforced once at `__init__`.
- `prepared_steps` reset idiom unified to `= []` throughout `tool_agents.py`.

### Removed

- `LLMEngine.invoke_messages` — pre-v2 text-only wrapper that bypassed
  `LLMResult` and emitted `DeprecationWarning`.

### Documentation

- Docstrings added to `ToolAgent.register()`, `ToolAgent.batch_register()`,
  `ReActAgent._initialize_run_state`, `ToolAgent.blackboard`,
  `ToolAgent.peek_at_cache`, and all six toolbox query/mutator methods
  (`list_tools`, `has_tool`, `get_tool`, `remove_tool`, `clear_tools`,
  `clear_memory`).
- `LLMEngine._on_detach` abstract-method docstring corrected — removed
  `# Intentionally a no-op by default` comment that contradicted
  `@abstractmethod + raise NotImplementedError`.
- `StructuredInvokable.map_single_fields` property docstring corrected —
  semantics were inverted (True/False meanings were swapped).
- Stale, misleading, and duplicate docstring content cleaned up across
  `engines/LLMEngines.py`, `agents/base.py`, `agents/tools.py`,
  `exceptions/core.py`, and `core/Invokable.py`.

## [2.0.0a11] - 2026-06-25

### Changed

- `LLMRecord.user_prompt: str` replaced by `messages: tuple[dict[str, str], ...]`
  — the delta of messages appended on top of the rendered conversation history
  immediately before each LLM call; system message and prior rendered turns are
  excluded. `to_dict()` serializes this field as `"messages"` (a list of dicts).
- `LLMRecord` validation hardened: rejects strings/bytes passed as `messages`,
  empty sequences, non-dict elements, empty dicts, and non-string keys or values;
  list inputs are normalized to tuple.
- `Agent._invoke` / `_ainvoke`: `LLMRecord` is now constructed with
  `messages=[messages[-1]]` (the current user message dict) instead of
  `user_prompt=prompt`.
- `PlanActAgent._generate_plan`: same one-element delta.
- `ReActAgent._generate_next_step`: accepts a new keyword-only
  `delta: list[dict[str, str]]` parameter; uses it for `LLMRecord` construction.
- `ReActAgent._prepare_next_batch`: computes a three-element delta
  `[state.messages[-1], working_messages[-2], working_messages[-1]]` — original
  task, running-plan snapshot, step-request stub — before each
  `_generate_next_step` call.

## [2.0.0a10] - 2026-06-25

### Added

- `LLMEngine.timeout_seconds` — read-only property exposing the per-call timeout
  baked into provider SDK clients at construction; previously accessible only as
  the private `_timeout_seconds`.

### Changed

- `OpenAIEngine.inline_cutoff_chars` and `MistralEngine.inline_cutoff_chars` are
  now frozen (read-only property backed by `_inline_cutoff_chars`). These values
  are consumed at `attach()` time — the truncated text is baked into stored
  attachment metadata — so changing them mid-session previously created silent
  state inconsistency.
- `LlamaCppEngine`: source/download parameters (`model_path`, `repo_id`,
  `filename`, `revision`, `cache_dir`, `local_dir`, `local_files_only`,
  `force_download`) and model-load parameters (`n_ctx`, `n_threads`,
  `n_threads_batch`, `n_gpu_layers`, `chat_format`, `verbose`) are now frozen
  as `_private` + read-only properties. All were consumed once at construction
  by `Llama(...)` and could not retroactively affect the loaded model. External
  property names are unchanged.
- `OpenAIEngine.llm`, `GeminiEngine.client`, `MistralEngine.client`: renamed to
  `_llm`/`_client` (fully private, no public property). These SDK client objects
  are construction-time artifacts; direct access is no longer part of the public
  API.

### Removed

- `Agent.attach`, `Agent.detach`, `Agent.clear_attachments`, `Agent.attachments`
  — attachment state is engine-coupled; access via `agent.llm_engine` directly.
- `ToolAgent.preview_limit` compatibility alias (and equivalent aliases on
  `PlanActAgent`, `ReActAgent`).
- `refresh()` and other lifecycle methods from `MCPProxyTool` and
  `PyA2AtomicTool` — tools are static once constructed.
- Mutable `headers` setters from `MCPClientHub` and `PyA2AtomicClient` — header
  ownership belongs to the proxy tools, not the client helpers.

**Frozen (no setter, construction-time only):**
- `AtomicInvokable`: `name`, `namespace`, `parameters`, `return_type`.
- `Agent` family: `response_preview_limit`, `assistant_response_source`,
  `peek_at_cache`, `blackboard_preview_limit`.

## [2.0.0a9] - 2026-06-22

### Added

- `namespace: str` as a constructor parameter and read-only property on
  `AtomicInvokable`; propagated to all derived classes (`Command`,
  `StructuredInvokable`, `Agent`, `ToolAgent`, `PlanActAgent`, `ReActAgent`,
  all `Workflow` kinds, `LLMEngine` and its adapters).
- `full_name` property unified to `Type.namespace.name` format across all
  invokable types — replaces per-class overrides.
- `to_dict()` extended with `"namespace"` on all classes.

### Changed

- breaking: `namespace` is now a **required** positional-or-keyword parameter
  (no default) on all identity-owning classes: `AtomicInvokable` base,
  `Agent`, `ToolAgent`, `PlanActAgent`, `ReActAgent`, `Workflow` base,
  `SequentialFlow`, `ParallelFlow`, `RoutingFlow`, and `IterativeFlow`.
  `namespace` appears immediately after `name` in all signatures.
- LLM engine adapters (`LLMEngine`, `OpenAIEngine`, `GeminiEngine`,
  `MistralEngine`, `LlamaCppEngine`) receive `namespace: str = "llm"` — a
  meaningful categorical default; no change required at call sites.
- Wrapper classes (`Tool`, `Command`, `StructuredInvokable`, `BasicFlow`)
  retain `namespace: Optional[str] = None`; namespace resolution from the
  wrapped component or `"default"` fallback is preserved.
- All `examples/` files updated: `namespace=` added to every identity-owning
  constructor; convention `"examples"` for agent/workflow examples,
  `"research"` for `Agentic_Research/` files.

### Removed

- `Tool`'s independent `_namespace` storage, `namespace` property setter, and
  `full_name` override — `Tool` now inherits the read-only `namespace`
  property from `AtomicInvokable` base.

## [2.0.0a8] - 2026-06-20

### Added

- `AgentRecord.prev: AgentRecord | None` — each record carries a pointer to
  the last record used as context when it was created, forming a conversation
  chain within the flat `_records` list.
- `AgentRecord.to_dict()` emits `prev_run_id`: the `run_id` of the predecessor
  record, or `None` for chain roots.
- `Agent.get_conversation(run_id=None, turns=None) -> list[AgentRecord]` —
  public method that walks the `prev` chain from a target record, returning
  results oldest-first. `run_id=None` starts from the most-recent record;
  `turns=None` returns the full chain; `turns=0` raises `ValueError`;
  unresolvable `run_id` raises `AgentInvocationError`.
- `continue_from: str | None = None` — reserved `KEYWORD_ONLY` parameter
  grafted onto every `Agent`'s input schema. Three modes:
  - `None` (default) — standard tail-of-history behavior, respecting
    `records_window`.
  - `"new"` — parallel fresh root: no prior context is sent; the committed
    record's `prev` is `None`. The record is still appended to `_records`.
  - `<run_id>` — forks from the identified record: `get_conversation` resolves
    the branch chain from that turn.
- `AgentRecord.llm_records: tuple[LLMRecord, ...]` — per-invocation LLM call
  records now live on the record, populated in the completion `replace` step.

### Changed

- `AgentResult`: `llm_records` replaced by `llm_token_usage:
  tuple[TokenUsage, ...]` (one entry per LLM call, ordered) — the result
  carries derived token-usage accounting; full records live on `AgentRecord`.
- `Agent.make_result`: derives `llm_token_usage` from metadata `llm_records`
  rather than receiving records directly.
- `Agent.records_window` (formerly `history_window`) — constructor parameter,
  public property, and private variable (`_records_window`) renamed throughout
  `base.py`, `tool_agents.py`, examples, and tests.
- `Agent.to_dict()`: `"history"` key removed; `"turn_history"` key renamed to
  `"records"`; rendered-history build loop removed.
- `LLMRecord` re-homed to `models/agents/records.py` (moved back from
  `models/results/agents.py`); re-exported from `models/agents`.

### Removed

- `Agent.history` deprecated property (and the `import warnings` it required).
- `LLMRecord` from `models/results` exports — import from `models/agents`
  instead.

## [2.0.0a7] - 2026-06-19

### Added

- `MCPToolResult` (`ToolResult` subclass): remote-identity fields
  `transport_mode`, `endpoint` / `command`, and `remote_name`; emitted by
  `MCPProxyTool.make_result`.
- `PyA2AtomicToolResult` (`ToolResult` subclass): fields `url`, `remote_name`,
  `invokable_type`; emitted by `PyA2AtomicTool.make_result`.
- `ToolUsageRecord` dataclass: `tool_name: str`, `call_count: int`; tracks
  non-return tool calls for a single `ToolAgent` invocation, ordered by
  first-call order.
- `ToolAgentResult` (`AgentResult` subclass): new result type for `ToolAgent`
  invocations; carries `tool_usage: tuple[ToolUsageRecord, ...]` alongside the
  full `AgentResult` envelope.
- `LLMRecord` re-homed to `models/results/agents.py` and re-exported from
  `models/results`; eliminates a latent circular import between
  `models/agents/` and `models/results/`.

### Changed

- `AgentResult` enriched: `llm_records: tuple[LLMRecord, ...]` replaces the
  former `llm_token_usage` field — per-call LLM records now travel with the
  result rather than the record.
- `AgentRecord` simplified: `final_response` renamed to `final_result:
  AgentResult | None`; `run_id` and `llm_records` fields removed; draft
  records carry `None`, completed records carry the `AgentResult`.
- `Agent._invoke` / `_ainvoke` return a 2-tuple `(draft_record, metadata_dict)`
  instead of a bare `AgentRecord`; metadata carries `llm_records` and
  `llm_model_data` for `make_result`.
- `ToolAgent._invoke` / `_ainvoke` compute `tool_usage` during the execution
  loop and include it in the returned metadata dict.
- `Agent.make_result` accepts `llm_records` and `llm_model_data` via metadata
  kwargs; UUID minted via `AtomicResult.__post_init__`; emits `AgentResult`.
- `ToolAgent.make_result` override additionally accepts `tool_usage` and emits
  `ToolAgentResult`; no derivation from the record — direct construction only.
- `invoke` / `async_invoke` complete the draft via
  `dataclasses.replace(draft, final_result=agent_result)` before appending to
  turn history.
- A2A typed error serialization: remote exception class names are now preserved
  across the wire instead of surfacing as bare `RuntimeError`.

## [2.0.0a6] - 2026-06-18

### Added

- `exceptions/` top-level package with `exceptions/core.py` as the canonical
  definition site for all 13 exception classes; `exceptions/__init__.py` is
  the single import surface (`from atomic_agentic.exceptions import X`).
- `constants/` top-level package — pure literals and sentinels subdivided by
  originating domain (`core.py`, `agents.py`, `a2a.py`).
- `models/` top-level package — dataclasses subdivided by kind:
  `models/parameters.py` (`ParamSpec`), `models/results/` (the full
  `AtomicResult`-family, relocated wholesale), `models/agents/` (agent
  records/runstates/blackboard, split from the former `data_classes.py`),
  `models/workflows/` (`WorkflowCheckpoint`).
- `utils/` top-level package — pure functions subdivided by domain:
  `utils/core.py` (async bridging, header normalization), `utils/parameters.py`
  (parameter extraction and validation), `utils/mcp.py` (MCP conversion
  helpers), `utils/agents.py` (dependency extraction).
- `agents/tools.py` — identity pre/post tools and return-tool helpers
  extracted from `agents/base.py` and `agents/tool_agents.py`.
- `workflows/tools.py` — fallback judge factory extracted from
  `workflows/iterative.py`.
- package topology table and "What's New in v2" section added to `README.md`.

### Changed

- `core/` narrowed to the shared invocation contract only (`AtomicInvokable`,
  `Command`, `StructuredInvokable`); all other former `core/` residents moved
  to their respective concern packages above.
- `tools/prebuilt.py` — renamed from `tools/Plugins.py`; reflects its actual
  role as a prebuilt tool-instance collection.
- `mcp/utils.py` removed; its helpers relocated to `utils/mcp.py`.
- test suite restructured to mirror the new layout (`tests/constants/`,
  `tests/models/`, `tests/models/agents/`, `tests/utils/`, `tests/exceptions/`).
- breaking (type-annotation only): `ToolAgent` de-Genericed — `Generic[RS]`
  and the `RS` TypeVar removed; no runtime behavior change.

### Removed

- `core/Exceptions.py` shim (callers now import from `atomic_agentic.exceptions`).
- `mcp/utils.py` (merged into `utils/mcp.py`).

## [2.0.0a5] - 2026-06-14

### Added

- `AtomicResult`-family return contract: every invokable (`Tool`, `Command`,
  `StructuredInvokable`, `Agent`, `ToolAgent`, every `LLMEngine`, and every
  `Workflow` kind) now returns a typed result object from `invoke` /
  `async_invoke` / `__call__` / `async_call` — `ToolResult`, `CommandResult`,
  `StructuredResult`, `AgentResult`, `LLMResult`, and the `*FlowResult` family
  (`BasicFlowResult`, `SequentialFlowResult`, `ParallelFlowResult`,
  `RoutingFlowResult`, `IterativeFlowResult`). `.result` is the caller-facing
  payload; `run_id`, `started_at` / `ended_at`, `elapsed_s`, and `invoker_id`
  carry timing and provenance.
- `LLMResult` carries per-call token-usage and model-identity data for every
  supported engine (OpenAI, Gemini, Mistral, LlamaCpp).
- `AgentResult` reports the LLM activity (token usage, model data) behind a
  single Agent invocation. `Agent.turn_history` now stores typed
  `AgentRecord` memory entries (`ToolAgentRecord` for `ToolAgent`s), linked to
  their `AgentResult` by `run_id`.

### Changed

- breaking: `invoke` / `async_invoke` / `__call__` / `async_call` return an
  `AtomicResult`-family object instead of a raw dict — unwrap `.result` to
  get the value that was previously returned directly.
- breaking: workflow result classes are named `*FlowResult`
  (`SequentialFlowResult`, `RoutingFlowResult`, etc.), not `*WorkflowResult`.
- breaking: blackboard entries used by `ToolAgent`s now store `AtomicResult`
  objects directly instead of raw dict slots.
- `StructuredInvokable` now packages its output as a `StructuredResult`
  envelope, replacing the interim plain-dict output introduced in
  `2.0.0a4`.
- the local A2A host's `_invoke_registered_invokable` now returns the
  invoked component's unwrapped `.result` rather than its result envelope.

### Removed

- `StructuredResultDict`, deprecated in `2.0.0a4`.

## [2.0.0a4] - 2026-06-01

### Changed

- `BasicFlow` is now the general `Workflow | AtomicInvokable` adapter;
  `SequentialFlow`, `ParallelFlow`, `RoutingFlow`, and `IterativeFlow`
  normalize any non-workflow step through it.
- `StructuredInvokable` is an explicit output-shaping/projection adapter, no
  longer a required or privileged workflow child type.

### Deprecated

- `StructuredResultDict`, retained temporarily as a compatibility surface
  (removed in `2.0.0a5`).

## [2.0.0a3] - 2026-05-29

### Removed

- `AdapterTool` (`tools/adapter.py`) — superseded by `Tool` / `toolify(...)`.
- the `EmbedEngines` embedding-engine subsystem.

## [2.0.0a2] - 2026-05-29

### Added

- `Tool` wraps plain callables and other `AtomicInvokable` instances
  directly.

### Changed

- `toolify(...)` is the single normalization entry point for callables and
  `AtomicInvokable` instances.
- breaking: re-wrapping an already-wrapped callable/invokable with `Tool` /
  `toolify(...)` produces a new `Tool` rather than mutating the existing one.

## [2.0.0a1] - 2026-05-25

### Changed

- breaking: removed deprecated parameter aliases across the public API.
- `ParamSpec` is now an immutable (frozen) dataclass.

### Removed

- dead pre-v2 compatibility shim modules.

## [1.4.1] - 2026-05-23

### Fixed

- the `function` property on PyA2A and MCP proxy tools is now read-only,
  reducing risk of accidental mutation.

## [1.4.0] - 2026-05-23

### Added

- `Tool` can wrap a plain callable or an `AtomicInvokable` instance directly;
  `Tool.wraps_invokable` indicates which.

### Changed

- invokable-backed `Tool`s reuse the wrapped invokable's declared parameters
  and return type instead of introspecting the callable.
- async execution for invokable-backed `Tool`s dispatches through
  `async_call()`.

### Deprecated

- `AdapterTool`, superseded by `Tool`'s direct invokable wrapping (removed in
  `2.0.0a3`).

## [1.3.1] - 2026-05-22

### Fixed

- assorted documentation, method, and error-message cleanups throughout the
  codebase.

## [1.3.0] - 2026-05-22

### Added

- forward-looking deprecation warnings ahead of the planned `2.0.0` cleanup.

### Changed

- `ParamSpec` construction validates names, indices, kinds, and defaults more
  strictly.
- mapping-style `ParamSpec` access (`__getitem__`-style dict access) now
  emits a `FutureWarning` (removed in `2.0.0a1`).
- `StructuredInvokable` constants normalized to uppercase.
- `AdapterTool` construction hardened.

## [1.2.0] - 2026-05-21

### Added

- `Command`, an `AtomicInvokable` adapter that pairs an executor with a fixed,
  validated set of inputs.

### Changed

- `StructuredInvokable` and `StructuredResultDict` relocated into
  `core/Invokable.py` alongside the other core adapters.

## [1.1.2] - 2026-05-18

### Changed

- frequently used values consolidated into per-module `constants.py` files.

## [1.1.1] - 2026-05-15

### Changed

- `Agent.build_messages()` made more stateless via improved message wiring.

## [1.1.0] - 2026-05-13

### Added

- configurable post-processing routing via `post_result_key`.
- `passthrough_inputs`, letting selected inputs bypass `pre_invoke`
  processing.
- construction-time configuration of pre/post-invoke lifecycle behavior.

### Changed

- composed Agent input schemas to support post-invoke passthroughs.
- improved PlanAct/ReAct prompt instructions.

## [1.0.9] - 2026-05-11

### Changed

- stricter `AtomicInvokable` input-filtering contract, with clearer
  input-representation methods.

## [1.0.8] - 2026-05-07

### Added

- `ConstantSpec`, a dataclass for symbolic placeholders in `ToolAgent` plans.

### Changed

- ReAct run cycle refactored with duration tracking and a "next best tool"
  producer pattern.

## [1.0.7] - 2026-05-03

### Added

- dedicated generation helpers for `ToolAgent` planning.

## [1.0.6] - 2026-05-02

### Changed

- `BlackboardSlot` became the central `ToolAgent` planning dataclass.

## [1.0.5] - 2026-04-30

### Changed

- Agent memory restructured into discrete per-invocation "turns".

## [1.0.4] - 2026-04-28

### Fixed

- sync tools failing to load correctly into threads during async invocation.

## [1.0.3] - 2026-04-26

### Changed

- `pyproject.toml` dependencies pinned to explicit versions.

## [1.0.2] - 2026-04-26

### Deprecated

- `EmbedEngines`, the embedding-engine subsystem (removed in `2.0.0a3`).

## [1.0.1] - 2026-04-26

### Added

- a `tests/` suite (unit and integration), mirroring the `src/atomic_agentic/`
  layout, at roughly 89% coverage.

## [1.0.0] - 2026-04-12

### Added

- `Workflow` split into five classes: `BasicFlow`, `SequentialFlow`,
  `ParallelFlow`, `RoutingFlow`, and `IterativeFlow`.
- `StructuredInvokable`, a stateless wrapper for output packaging.
- native async support (`async_invoke` / `async_call`) for invokables.

### Changed

- `ToolAgent`s reworked around a unified prepare-and-execute pattern, with
  PlanAct and ReAct agents distinguished by initialization and
  step-preparation.
