# Changelog

All notable changes to Atomic-Agentic are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Atomic-Agentic's v2 line is currently pre-1.0 alpha (`2.0.0aN`).

## [2.0.0a19] - 2026-07-04

### Added

- `context_properties: list[str] | list[ParamSpec] | None` constructor parameter
  on all agent types (`BasicAgent`, `ToolAgent`, `PlanActAgent`, `ReActAgent`).
  Declares which entries a caller must supply (required) or may optionally supply
  (with defaults) in the `context` dict input. Missing required properties raise
  `AgentInvocationError` before `pre_invoke` runs; optional properties have their
  defaults injected into the context dict automatically.
- `Agent.set_context_properties(properties)` — public API for post-construction
  context schema updates. Normalizes, re-indexes, stores, and refreshes the
  `context` parameter description in one call. `ToolAgent` subclasses inherit
  directly; `BasicAgent` overrides to raise (use `set_extra_context_properties()`
  or `update_prompt("role", ...)` instead).
- `BasicAgent.extra_context_properties` constructor parameter and
  `set_extra_context_properties(properties)` mutation method for declaring context
  schema entries beyond what the role prompt auto-discovers.
- `BasicAgent.role_prompt` setter — accepts `str | PromptConfig | None`; delegates
  to `update_prompt("role", ...)`.
- `AgentRecord.context: dict` — the context dict at invocation time is now stored
  on every record, enabling exact replay of prompt rendering from history.
- `normalize_context_properties` and `normalize_role_prompt` exported from
  `atomic_agentic.utils.agents`.
- `PlanActAgent.update_prompt` guard: raises `ToolAgentError` when key is
  `"plan_first"` (the built-in planning prompt). Other keys pass through to base.
- `ReActAgent.update_prompt` guard: raises `ToolAgentError` when key is
  `"reason_then_act"` (the built-in orchestrator prompt). Other keys pass through
  to base.
- `AtomicInvokable` instances passed as `pre_invoke` or `post_invoke` are stored
  as-is; raw callables still go through `toolify()`.
- `PromptConfig.to_dict()` serialization method.

### Changed

- **Context parameter model redesigned.** `context_keys: list[str]` replaced by
  `context_properties: list[str] | list[ParamSpec] | None`. Callers no longer
  pass each context entry as an individual keyword argument — a single
  `context: dict` parameter carries all context entries.
- `AgentRecord.user_prompt` type changed from `str` to `PromptConfig`. The
  pre-render config is stored on the record; `render_turn()` re-renders it against
  the stored context dict for exact replay.
- `BasicAgent.update_prompt` behavior changed: `"role"` key is now **allowed**
  (triggers context schema rebuild); any other key **raises** `AgentError`.
  Previously `"role"` raised and other keys were accepted.
- `ToolAgent`, `PlanActAgent`, `ReActAgent` system prompt ownership fully migrated
  to subclass hooks. `tool_instructions` and `prompt_key` constructor parameters
  removed from `ToolAgent`; `PlanActAgent` owns `"plan_first"` and `ReActAgent`
  owns `"reason_then_act"` in their respective `_system_prompts` registries.
- `PromptConfig.render()` passes non-identifier brace expressions (`{0}`, `{}`,
  `{'a':b}`) through as literal text rather than raising.
- `_initialize_run_state` hook signature on `ToolAgent`, `PlanActAgent`, and
  `ReActAgent` loses the `context: dict` parameter — it was silently ignored in
  all concrete implementations.
- `post_result_key` now rejects `"context"` and `"run_id"` (reserved agent
  parameter names).

### Fixed

- `BasicAgent` context property rebuild now correctly re-indexes the combined
  role + extra param list. Previously two independent zero-based index ranges were
  concatenated without normalization, producing colliding indices.
- `invoke` and `async_invoke` context dict is always a fresh copy, preventing
  aliasing against the `ParamSpec` default `{}` sentinel.
- Optional context property default injection now present on the sync `invoke`
  path (was missing; `async_invoke` already had it).

## [2.0.0a18] - 2026-07-04

### Changed

- `agents/tool_agents.py` split into three dedicated modules:
  `agents/toolagent.py` (base `ToolAgent` only), `agents/planact.py`
  (`PlanActAgent`), and `agents/react.py` (`ReActAgent`). No behavior changes.
- `agents/__init__.py` updated to re-export `ToolAgent`, `PlanActAgent`, and
  `ReActAgent` from their new module locations. Public import surface unchanged.
- Test suite reorganized to mirror the module split: `tests/agents/conftest.py`
  (shared fixtures), `test_tool_agent.py`, `test_planact.py`, `test_react.py`.
  Monolithic `test_tool_agents.py` deleted.
- Module docstrings added to `planact.py` and `react.py`; `__init__` docstrings
  added to both subclasses; `toolagent.py` module docstring updated with file
  paths for concrete subclasses; minor documentation normalization across all
  three files (section labels, separator style, missing parameter entries).

### Fixed

- `planact.py` and `react.py`: `NO_VAL` now imported from `..constants.core`
  (was incorrectly referenced via `..core`, which does not re-export it).
- `react.py`: `import pprint` (module) replaces `from pprint import pprint`
  (function), fixing an `AttributeError` on `pprint.pformat(...)`.

## [2.0.0a17] - 2026-07-04

### Added

- `generation_retries: int = 0` constructor parameter on `ToolAgent`, `PlanActAgent`,
  and `ReActAgent`. `N` means N additional attempts beyond the first; default `0`
  preserves all existing behavior. Stored as `self._generation_retries`; exposed via
  `ToolAgent.generation_retries` read-only property. Added to `to_dict()`.
- `PlanActAgent` generation retry loop inside `_generate_plan` / `_agenerate_plan`:
  two error categories — JSON-decode failure (raw LLM output injected as context) and
  spec-validation failure (clean re-serialized plan injected). Each attempt produces its
  own `LLMRecord`; all are accumulated in `state.llm_records`.
- `ReActAgent` generation retry loop inside `_generate_next_step` / `_agenerate_next_step`:
  same two-category feedback scheme. Budget is **shared across all step generations in
  one invoke** via `ReActRunState.retries_used`. Budget exhaustion mid-run raises
  `ToolAgentError` regardless of `fail_fast`.
- `ReActRunState.retries_used: int = 0` field — cumulative retry count across all
  step-generation turns in a single run.
- `ToolAgentRunState.valid_cache_indices: frozenset[int]` and `failed_cache_indices:
  frozenset[int]` — conversation-scoped index sets computed once at run-init time in
  `_invoke` / `_ainvoke` by walking `ToolAgentRecord.blackboard_start/end` spans in
  the `turns` chain. Neither set is ever recomputed during a run.
- `ToolAgent._compute_cache_index_sets(turns)` helper — produces both frozensets from
  the conversation's `ToolAgentRecord` history; returns empty frozensets when
  `context_enabled=False`.
- Three-category cache-ref validation in both `PlanActAgent._validate_planned_slots`
  and `ReActAgent._process_next_step_output`:
  1. **Out-of-range** — index does not exist in the cache at all.
  2. **Failed-in-conversation** — index is in `failed_cache_indices`; error message
     includes the tool name and stored error string from the slot.
  3. **Out-of-conversation** — index is in range but was not produced in this
     conversation (not in either frozenset).
  All three categories produce retry-injectable feedback strings.
- `ReActAgent._build_react_messages` (B1): `FAILED` running-blackboard slots are now
  included in the step snapshot rendered to the LLM, with `status="FAILED"` and
  `error` fields and no `result_ref`. Previously these slots were silently omitted
  under `fail_fast=False`, making failures invisible to the LLM for subsequent steps.
- Pre-gather tool-existence check in `_execute_prepared_batch` (sync path) — validates
  all tools in the prepared batch before any execution begins, matching the existing
  async path behavior.
- `examples/ReAct_Examples/03_context_enabled_reactor.py` — new example demonstrating
  a context-enabled `ReActAgent` across multiple invocations.

### Changed

- All LLM-facing validation failures in the planning pipeline (`_validate_tool_step_dict`,
  `_normalize_planned_slots`, `_validate_planned_slots`, `_process_plan_output`,
  `_process_next_step_output`) now return feedback strings instead of raising
  `ToolAgentError`. This makes every validation category retryable without special
  exception handling in the retry loop.
- `cache_blackboard` on `RunState` is now a snapshot copy
  (`[slot.copy() for slot in self.blackboard]`), not a live reference to
  `self._blackboard`. Mutations during a run no longer affect the persisted blackboard.
- Removed `slot.step = i` mutations in `PlanActAgent._setup_plan_init` and
  `ReActAgent._initialize_run_state` that were silently mutating persisted blackboard
  slots as a side effect of starting a run.
- `max_duration` is now computed once per ReAct step turn in `_prepare_next_batch` /
  `_aprepare_next_batch` and passed explicitly to both `_build_react_messages` and
  `_generate_next_step` / `_agenerate_next_step`. Neither method independently
  recomputes it.
- `_generate_next_step` and `_agenerate_next_step` now accept and thread
  `valid_cache_indices` and `failed_cache_indices` into `_process_next_step_output`.
- Unknown-tool check in `_process_next_step_output` now uses `self.has_tool()` for
  parity with `PlanActAgent._validate_planned_slots`.
- `_generate_plan` / `_agenerate_plan` and `_build_planact_run_state` thread frozensets
  through to `PlanActRunState`.
- `_extract_from_json_string`: parse failures now raise `json.JSONDecodeError` (was a
  plain `ToolAgentError`), allowing retry loops to distinguish JSON errors from
  engine-contract violations cleanly.
- Running-plan snapshot header in `_build_react_messages` updated to acknowledge that
  steps may be `EXECUTED` or `FAILED`.

### Fixed

- `_agenerate_plan` JSON-exhaustion error message now matches the sync path:
  `"Last error: {feedback}"` (was `"Last error is a JSONDecodeError: {exc}"`).
- `_tool_step_dict_to_slot` docstring corrected — it is a converter, not a field-set
  validator; upstream `_validate_tool_step_dict` owns field validation.
- `_compute_cache_index_sets` accessed `self.blackboard` (the property, which returns
  a full copy of `_blackboard`) twice per slot — once for the length guard, once to
  read the slot. Both calls replaced with direct `self._blackboard` access. An
  `ToolAgentError` is now raised for any slot with unexpected status (PLANNED/PREPARED/
  EMPTY) in a persisted blackboard span — these are framework invariant violations that
  were previously silently skipped.
- `_setup_plan_init` (PlanActAgent) and `ReActAgent._initialize_run_state` both
  constructed `cache_blackboard` as `[slot.copy() for slot in self.blackboard]`.
  Because `self.blackboard` (the property) already returns copies, this produced
  copies-of-copies. Replaced with `[slot.copy() for slot in self._blackboard]`.
- `_apply_react_step_result` no longer independently recomputes `max_duration` — it
  now receives it as a keyword-only parameter from `_prepare_next_batch` /
  `_aprepare_next_batch`, which remain the single authoritative computation site.
- `_validate_tool_step_dict` error message for an invalid `await` value type now names
  the LLM-facing field `'await'` instead of the internal field name `'await_step'`.
- `render_turn` all-executed header format standardized to `CACHED STEPS [N, ..., M]
  PRODUCED:` to match the mixed-path (executed+failed) format (was `CACHED STEPS #N-M
  PRODUCED:`).
- `_async_execute_prepared_batch` error message wording unified with the sync path:
  `"tool call failed at step {idx}"` (was `"at index {idx}"`).
- `ScriptedToolAgent._initialize_run_state` (test harness): `cache_blackboard` now
  built with `[slot.copy() for slot in self._blackboard]` instead of `list(self._blackboard)`,
  preventing test code from holding shared references into agent-internal state.

### Removed

- Dead guard D1: `isinstance(idx, int)` before slot lookup in `_execute_prepared_batch`
  and `_async_execute_prepared_batch` — `prepared_steps: list[int]` guarantees int
  indices by contract.
- Dead guard D2: `isinstance(slot.step, int)` prefix on the step-mismatch check — the
  `isinstance` prefix inverted the guard (non-int skipped check instead of raising).
- Dead guard D3: `isinstance(state, ToolAgentRunState)` in `_invoke` / `_ainvoke` —
  `_initialize_run_state` is abstract; the return type is always a valid subclass.
- Dead guard D4: `isinstance(state, ReActRunState)` in `_prepare_next_batch` /
  `_aprepare_next_batch` — internal contract; inconsistent with PlanAct (no such guard).
- Dead guard D5: `isinstance(generated_slot, BlackboardSlot)` in
  `_apply_react_step_result` — `_generate_next_step` always returns a validated slot
  or raises.

## [2.0.0a16] - 2026-07-03

### Added

- `fail_fast: bool = True` constructor parameter on `ToolAgent`, `PlanActAgent`,
  and `ReActAgent`. Default preserves all existing behavior. With `fail_fast=False`,
  individual tool-call failures are recorded as `FAILED` blackboard slots rather
  than aborting the run immediately.
- `ToolAgentResult.exception_records: tuple[tuple[int, Exception], ...]` — reports
  the global blackboard index and stored exception for every `FAILED` slot produced
  during a `fail_fast=False` run; empty tuple when `fail_fast=True`.
- Cascade `FAILED` propagation (`PlanActAgent._prepare_next_batch` and
  `ReActAgent._apply_react_step_result`): when `fail_fast=False`, any slot whose
  `args` reference a `FAILED` step via `<<__sN__>>` is cascade-marked `FAILED` and
  skipped before preparation. Return-step cascade-fail is always fatal regardless
  of `fail_fast`.
- FAILED cache-ref detection in `_validate_planned_slots` (`PlanActAgent`) and
  `_process_next_step_output` (`ReActAgent`): a plan referencing a `<<__cN__>>`
  slot that is `FAILED` in the persisted cache now raises `ToolAgentError` at
  plan-processing time, before any execution.
- `ToolAgent.fail_fast` read-only property.
- `"fail_fast"` key added to `ToolAgent.to_dict()`.
- `batch_register`: `remote_names` entries not found on the client now raise
  `ToolRegistrationError`; previously they were silently ignored.

### Changed

- `update_blackboard` is now called unconditionally at the end of `_invoke` and
  `_ainvoke`, regardless of `context_enabled`. `context_enabled` continues to
  control only whether prior-run slots are loaded as `cache_blackboard`.
  `ToolAgentRecord.blackboard_start` / `blackboard_end` are now always integers.
- `update_blackboard` persists all non-empty slots — `EXECUTED` and `FAILED` —
  preserving global index continuity across runs.
- `render_turn`: when the blackboard span contains `FAILED` slots, the assistant
  content is split into a `CACHED STEPS` section (executed slots, full data) and
  a `FAILED STEPS` section (step index, tool name, truncated error string — no
  args). All-executed spans retain the existing single-section format.
- `_resolve_placeholders`: cache-ref error message now branches on status —
  "permanently FAILED" for `FAILED` slots, "not executed" otherwise, removing the
  misleading "not yet" phrasing for a terminal state.
- Sync `_invoke` loop violation message corrected: `"prepared_indices"` →
  `"prepared_steps"` to match the async path and the actual field name.
- `_validate_react_prepare_state` docstring updated to reflect the relaxed
  invariant: the previous step must be processed (executed, or `FAILED` when
  `fail_fast=False`).

### Fixed

- `render_turn` with `peek_at_cache=True` no longer crashes when a `FAILED` slot
  is present in the blackboard span (`FAILED` slots carry `result = NO_VAL`;
  result access is now gated inside `is_executed()`).

### Removed

- Dead code DA: unreachable post-`super()` guard in `ReActAgent.__init__`.
- Dead code DB: unreachable step-mismatch guard in `_apply_react_step_result`.
- Dead code DC: unreachable return-tool identity check at end of
  `_normalize_planned_slots`.

## [2.0.0a15] - 2026-07-01

### Added

- `ParamSpec.description: str | None = None` — new field on the frozen
  dataclass. Validated on construction (type check, whitespace-strip,
  empty-string → None); serialized conditionally in `to_dict()` (omitted
  when None); round-tripped in `from_dict()`; preserved through all
  `to_paramspec_list` copy paths.
- `_unwrap_annotated` helper in `utils/parameters.py` — extracts the base
  type and first bare string from `Annotated[T, ...]` metadata, enabling
  `Annotated[T, "description string"]` as the AA-native inline description
  convention for `extract_io`-instrumented callables.
- `extract_io` now calls `get_type_hints(fn, include_extras=True)` and
  unwraps `Annotated` types per-parameter and for the return annotation; a
  `NameError` fallback is preserved. TypedDict path in `to_paramspec_list`
  gains the same treatment.
- `_format_annotation` NoneType fix: `type(None)` now renders as `"None"`.
- `_build_mcp_tool_metadata` in `utils/mcp.py` — extracts per-property
  descriptions from a FastMCP tool's JSON schema; normalizes non-string and
  whitespace-only values to None.
- `AtomicInvokable.description` getter augmented: collects parameters with
  non-None descriptions and appends a `"- {name}: {description}"` bullet
  list (blank-line separated). Invokables with no described parameters
  return `_description` unchanged.
- `StructuredInvokable.description` getter now calls `super().description`
  before appending the output-schema summary, so parameter bullets and
  schema notes stack correctly.
- `StructuredInvokable.PASSTHROUGH[0]` (`"__passthrough_mapping__"` `ParamSpec`)
  carries a description explaining its passthrough-without-field-checks
  semantics.
- `PLANNER_PROMPT` strengthened: two explicit rules now prohibit
  `<<__cN__>>` when no "CACHE STEPS" section is visible in the
  conversation, preventing LLM-generated plans that reference out-of-range
  cache indices on fresh runs.

### Changed

- `AtomicInvokable.to_dict()` emits `self._description` (raw stored value)
  rather than the getter output, preventing augmented text from being baked
  into serialized records or wrapper descriptions.
- Four double-wrap copy sites updated to use `._description` instead of
  `.description`: `Tool.__init__`, `toolify()`, `BasicFlow.__init__`, and
  `StructuredInvokable.__init__`.
- `_compose_agent_parameters` now preserves `description` when rebuilding
  `ParamSpec` objects during the four-tier parameter graft.
- `_warn_reserved_name_collisions` `semantically_equal` check extended to
  include `param.description == RUN_ID_PARAM.description`.
- MCP example servers updated to use `Annotated[T, Field(description="...")]`
  (pydantic `FieldInfo`) for parameter descriptions; `04_MCPProxyTool.py`
  updated to print parameter descriptions.

### Removed

- `Agent.description` getter and setter override — the base
  `AtomicInvokable.description` getter now handles all description
  augmentation; the prior hardcoded `run_id`-usage note appended in the
  override is removed.
- `Agent.to_dict()` raw-description guard — now handled uniformly by the
  base `to_dict()`.

## [2.0.0a14] - 2026-06-29

### Added

- `BasicAgent` — concrete single-turn LLM agent exposed from
  `atomic_agentic.agents`. Accepts `role_prompt: str | PromptConfig | None`;
  renders the role prompt (with optional placeholder context) before each LLM
  call. Replaces direct instantiation of `Agent`, which is now abstract.
- `PromptConfig` — frozen dataclass in `models/agents/prompts.py` pairing a
  format-string template with metadata and per-placeholder defaults; parameters
  are auto-discovered from `{placeholder}` slots; `render(inputs)` fills them.
  Exported from `atomic_agentic.models.agents`.
- `LLMRecord.system_prompt_name: str | None` — records which `system_prompts`
  key produced the system prompt for a given LLM call (`None` for static /
  legacy callers).
- `Agent.context_keys` — constructor parameter
  `(list[str] | list[ParamSpec] | None)` declaring which inputs are consumed by
  role-prompt context rendering rather than forwarded to `pre_invoke`.
- `Agent.system_prompts` read-only property and `Agent.update_prompt(key,
  config)` for per-key prompt management; `BasicAgent.update_prompt` raises
  `AgentError` for the reserved `"role"` key.
- `RUN_ID_PARAM` — canonical `ParamSpec` constant in `constants/agents.py`
  defining the framework-reserved `run_id` keyword argument grafted onto every
  agent's composed schema.
- `AtomicInvokable.filter_inputs` now injects defaults for absent non-variadic
  parameters with non-`NO_VAL` defaults, removing the need for call-site
  duplication in `Agent._split_inputs`.
- All three blackboard rendering surfaces now include `run_id` for every
  executed slot: `ToolAgent.blackboard_serialized()`, `ToolAgent.render_turn()`,
  and `ReActAgent._build_react_messages()`. Unexecuted (PLANNED/EMPTY) slots
  carry no `run_id` key.
- `PLANNER_PROMPT` and `ORCHESTRATOR_PROMPT` updated to document the `run_id`
  field in step records and clarify it is a plain quoted string literal, not a
  `<<__sN__>>` placeholder.

### Changed

- breaking: `Agent` is now an abstract base class (`ABC`). Direct instantiation
  raises `TypeError`; use `BasicAgent` instead.
- breaking: `passthrough_inputs` constructor parameter removed from `Agent`.
  Post-invoke parameters beyond the result key are now auto-grafted into the
  agent schema from the `post_invoke` callable's signature — no explicit
  declaration needed.
- breaking: framework-reserved parameter renamed `continue_from` → `run_id`
  across all agents, examples, and tests.
- breaking: `run_id="new"` mode removed. To start a context-free invocation use
  `context_enabled=False`; for a parallel conversation root use a separate
  agent instance.
- `Agent._invoke` / `_ainvoke` signature changed to
  `(turns, prompt, context: dict)`; `context` carries inputs consumed by
  role-prompt rendering.
- `Agent.description` property overridden to append a `run_id`-usage note when
  the agent is registered as a sub-tool, ensuring orchestrating LLMs see
  correct argument semantics in the registered schema.

### Removed

- Direct `Agent` instantiation — class is now abstract.
- `Agent.passthrough_inputs` constructor parameter and the internal
  `_normalize_passthrough_inputs` / `_validate_passthrough_parameter_shapes`
  helpers.
- Redundant default-fill loop from `Agent._split_inputs` — superseded by
  `filter_inputs` default injection.

### Fixed

- `examples/Workflow_Examples/05_IterativeFlow.py`: `gpt-5-mini` typo
  corrected to `gpt-4o-mini`.

## [2.0.0a13] - 2026-06-27

### Added

- `ToolAgent` now accepts any `AtomicInvokable` (Tool, Agent, Workflow, remote
  proxy) directly in `register()` — stored under its own `full_name` with no
  wrapping or mutation. Registering an `AtomicInvokable` with `name` or
  `description` overrides raises `ToolRegistrationError`.
- `batch_register()` redesigned: accepts a `tools` list of
  `AtomicInvokable | Callable`, a remote `client`, or both in one call; new
  `remote_names` whitelist param for selective client registration; intra-batch
  duplicate `full_name` detection always raises regardless of
  `name_collision_mode`.

### Changed

- `ToolAgent._toolbox` widened from `dict[str, Tool]` to
  `dict[str, AtomicInvokable]`; `list_tools()` and `get_tool()` return types
  updated accordingly.
- `register()` simplified: `namespace`, `filter_extraneous_inputs`, and
  `remote_name` params removed; `MCPClientHub` / `PyA2AtomicClient` input
  routes removed (use `batch_register(client=...)` instead); callable route
  uses `self.name` as the tool namespace.
- `batch_register()` old `(sources: List[...], batch_namespace)` signature
  replaced by `(tools, client, *, remote_names, name_collision_mode,
  batch_filter_inputs)`.

### Removed

- `batch_toolify` removed from `tool_agents.py` internal imports.

### Documentation

- `StructuredInvokable` class docstring expanded to a narrative covering the
  absent-value mode contract (RAISE / DROP / FILL); `__init__` docstring
  expanded with full 13-param Parameters section.
- `ToolAgent.__init__` docstring added; documents all 16 constructor
  parameters.

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
