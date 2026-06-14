# Changelog

All notable changes to Atomic-Agentic are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Atomic-Agentic's v2 line is currently pre-1.0 alpha (`2.0.0aN`).

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
