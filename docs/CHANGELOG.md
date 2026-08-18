# Changelog

All notable changes to Atomic-Agentic are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Atomic-Agentic's v2 line is currently pre-1.0 alpha (`2.0.0aN`).

## [2.0.0a28] - 2026-08-17

New, additive `a2a-sdk`-backed A2A integration (`A2AClientHub`,
`A2AtomicExecutor`, `A2AProxyTool`) alongside the existing
`python_a2a`-backed host/client/tool. `python_a2a`'s host and client are
behaviorally untouched; `PyA2AtomicTool`'s constructor is not — see
"Changed" below. `MCPClientHub` gains a real persistent-connection
option, fixing per-call subprocess/session-reopen overhead. Every
remote-proxy tool (new and existing) now lands on one consistent,
client/hub-only construction convention.

### Added

- `A2AClientHub` (`a2a/A2AClientHub.py`) — connectivity to any
  spec-compliant A2A server over JSON-RPC/REST/gRPC. Required
  `persistent: bool` selects a dedicated background-loop-owned
  persistent connection or a per-call open/use/close cycle (mirrors
  `MCPClientHub`). Detects any remote's published Atomic skills and can
  call them directly (`get_atomic_skills()`, `call_atomic_skill(...)`).
- `A2AtomicExecutor` (`a2a/A2AtomicExecutor.py`) — hosts a registry of
  local `AtomicInvokable`s over `a2a-sdk`, publishing each one's full
  parameter schema so `A2AClientHub` can discover and call it with a
  typed signature, not just free-form messages.
- `A2AProxyTool` (`tools/a2a_sdk.py`) — proxies an `A2AClientHub` as a
  normal `Tool`. Skill mode (bound to one discovered Atomic skill, typed
  signature) or generic mode (`parts`/`metadata`, for talking to
  arbitrary/foreign A2A agents). `toolify(hub, remote_name=...)` and
  `batch_toolify([hub])` both support it; `ToolAgent.batch_register(
  client=...)` now accepts an `A2AClientHub` too.
- `MCPClientHub` gains a required `persistent: bool` connection option
  — holding the transport/session open across calls instead of
  reopening it on every single tool call. Measured payoff: ~14.7s →
  0.03s for 6 repeated `stdio` calls in internal testing.
- New example pair `a2a_sdk_atomic_host_server.py`/
  `a2a_sdk_foreign_host_server.py`, plus `06_A2AProxyTool.py` and new
  sections in `07_toolify_example.py` (renamed from
  `06_toolify_example.py`) demonstrating the new routes end to end.

### Changed

- breaking: `MCPProxyTool`/`PyA2AtomicTool` constructors no longer
  accept raw transport parameters (`transport_mode`/`endpoint`/
  `command`/`args`/`headers`/`persistent` for MCP; `url`/`headers` for
  A2A) — construct the client/hub explicitly and pass it as
  `client_hub=`/`client=` instead. Matches `A2AProxyTool`'s convention
  from the start; no capability loss, just one construction path
  instead of two. (`MCPProxyTool` briefly gained a `persistent`
  passthrough alongside `MCPClientHub`'s new option above, then lost it
  again here — it never shipped as part of this release's actual public
  surface.)
- `A2AProxyTool`'s default namespace, when not explicitly provided, is
  now derived from the remote agent's own card name (sanitized into a
  valid identifier) where possible, falling back to `"a2a"` only when
  that isn't usable — was previously always `"a2a"`.

### Fixed

- README: `BasicAgent`/`PlanActAgent` quickstart examples were missing
  the required `namespace` argument and would raise `TypeError` on
  construction; the "install from source" instructions referenced a
  wheel filename pattern that never matched a real build.

## [2.0.0a27] - 2026-08-13

Parameter-reconciliation release: `ParamSpec.type` becomes a structural
`tuple[str, ...]`, and every family that composes multiple `AtomicInvokable`
declarations into one schema — `Agent`, `SelfAskAgent`, `ParallelFlow`,
`RoutingFlow`, `GraphFlow` — is rebuilt around one shared N-way
reconciliation primitive instead of five independent pairwise folds.
`IterativeFlow`/`GraphFlow`'s per-item topology records move out of
`models/` into `NamedTuple`s nested inside their owning workflow class. A
closing cleanup sweep promotes several reconciliation primitives to public
API and relocates the parameter-kind vocabulary into `constants/`; a
handful of example scripts also get fixed along the way.

### Added

- `ParameterReport` (`models/parameters.py`) + `build_parameter_reports`/
  `apply_parameter_reports` (`utils/parameters.py`) — the new canonical
  N-way parameter reconciliation layer. Groups declarations by name across
  any number of priority-ordered sources; computes a compatible type
  witness set, joint kind compatibility, and a winning source per name;
  `apply_parameter_reports` then does raise-or-construct-or-warn in one
  traversal, emitting at most one grouped `UserWarning` per construction
  call (not one per name) for compatible-but-not-identical overlaps.
- `n_way_type_witness`/`n_way_kind_compatible` (`utils/parameters.py`) —
  the pure N-way compatibility primitives `build_parameter_reports` is
  built on.
- `constants/parameters.py` — new module: canonical parameter-kind
  vocabulary (`POSITIONAL_ONLY`, `POSITIONAL_OR_KEYWORD`, `VAR_POSITIONAL`,
  `KEYWORD_ONLY`, `VAR_KEYWORD`), `VALID_KINDS`, `KIND_PRIORITY`.
  `ParamSpec`'s own class-level kind constants become read-only shims onto
  these values — public surface (`ParamSpec.POSITIONAL_ONLY`, etc.)
  unchanged for every existing caller.
- `core/core_api.py`: `parameter_overlap(sources: Sequence[Callable], *,
  unanimous_only: bool = False)` and `parameter_collisions(sources:
  Sequence[Callable])` — public, N-way-generalized replacements for the old
  pairwise versions, relocated alongside `extract_io` (same
  `AtomicInvokable`-awareness/circular-import justification). Curated into
  `core/__init__.py` and the root package's public exports alongside
  `semantically_compatible`/`semantically_identical` (unchanged,
  `utils/parameters.py`).
- `GraphFlow.Node` — new `NamedTuple` nested inside `GraphFlow`
  (`invokable`, `incoming`, `outgoing`, `routers`, `priority: int`),
  replacing the `models/workflows/graph.py`-owned `GraphFlowNode` dataclass.
- `IterativeFlow.Checker` — new `NamedTuple` nested inside `IterativeFlow`
  (`judge`, `approval_value`), used internally only; external
  constructor/property surface stays plain `tuple[AtomicInvokable, Any] |
  None` per body-step position (position *is* the index — no separate
  `index` field, unlike the deleted `CheckerSpec`).
  `IterativeFlow.update_checker(index, approval_value)` added alongside
  `add_checker`/`remove_checker`.
- `GraphFlow.parameters` — no longer validation-only: now exactly `start`'s
  own parameter names, each name shared with another node widened to the
  whole-graph compatible type witness set (`kind`/`default`/`description`
  always stay `start`'s own).
- `RoutingFlow`: branches may now introduce parameters the router doesn't
  declare, unconditionally. Every such name gets `default=None` with
  `"None"` unioned into its type, unless every declaring branch shares the
  exact same real (non-`NO_VAL`) default, in which case that shared value
  wins instead.

### Changed

- breaking: `ParamSpec.type` is now `tuple[str, ...]` (was `str`) — sorted,
  deduplicated at construction. Every call site across `src/`,
  `AtomicInvokable.signature` (now `" | ".join(spec.type)`), `to_dict()`/
  `from_dict()`, and every type-comparison utility updated accordingly.
- breaking: `Agent.__init__`'s `pre_invoke`/`post_invoke`/`extra_parameters`
  reconciliation rebuilt around `build_parameter_reports`/
  `apply_parameter_reports`, replacing the old asymmetric 2-tier scheme —
  `extra_parameters` can now raise on a genuine conflict, not just
  warn-and-drop.
- breaking: `SelfAskAgent.__init__`'s hand-rolled `role_prompt`/
  `thinking_instructions` reconciliation reworked onto the same primitives
  (matching `Agent.__init__`'s own idiom), fixing a per-name-vs-grouped
  warning inconsistency and a missing type-widening gap in the process.
- breaking: `ParallelFlow`'s branch-parameter reconciliation rebuilt from an
  order-dependent left-fold into true N-way peer reconciliation.
- breaking: `RoutingFlow.schema_mode` (`STRICT`/`PARTIAL`/`OPEN`) removed
  entirely — two design iterations within this release: first collapsed to
  a binary `strict_schema: bool`, then dropped altogether once a
  branch-specific required parameter forcing itself onto unrelated routing
  calls was judged unintuitive (only one branch ever runs per invocation).
  Router-shared reconciliation (the old STRICT case) is unaffected:
  per-branch witness union against the router, still raises on genuine
  conflict.
- breaking: `CheckerSpec`/`GraphFlowNode` (`models/workflows/`) removed —
  replaced by `IterativeFlow.Checker`/`GraphFlow.Node`, the first nested
  classes in this codebase, so neither needs a `TYPE_CHECKING`-only
  `AtomicInvokable` import workaround.
- `GraphFlow.Node.priority` is a plain `int` field (`set_priority` rebuilds
  the entry via `_replace()`). Originally shipped this release as a
  single-item mutable `list[int]` "box," mutated in place; found before
  release to leak mutability straight through the documented-read-only
  `nodes` property (`MappingProxyType` only guards the outer dict, not a
  mutable list nested inside each entry), letting a caller bypass
  `set_priority`'s own type check entirely and corrupt state that then
  failed later, confusingly, deep inside the collision-tiebreak path.
  Corrected before release: plain `int`, whole-entry replacement — trades
  node-identity-preservation across a `set_priority` call for closing the
  leak.

### Fixed

- `examples/Tool_Examples/01_Tool.py`/`03_Tool_wrapping_test.py`/
  `Agent_Examples/05_Passthrough_Test.py`: the same recurring sentinel-check
  defect previously fixed in `04_MCPProxyTool.py` (a21) and
  `07_Prompt_Parameters.py` (a22) — `param.default.__class__.__name__ ==
  "NO_VAL"`/`"_NO_VAL"`, always `False` — reintroduced in a further-widened
  set of files. Required-parameter display never showed "(no default)" for
  any parameter, required or not. Restored to `param.default is NO_VAL`.
- `examples/Workflow_Examples/04_IterativeFlow.py`/`05_GraphFlow.py`:
  missing `output_dir.mkdir(exist_ok=True)` before writing generated
  output, unlike the sibling `PlanAct_Examples`/`ReAct_Examples` scripts
  that write to the same (`.gitignore`d) directory — a fresh checkout would
  have hit `FileNotFoundError` running either script first.
- `examples/Agent_Examples/05_Passthrough_Test.py`: stale `#
  05_Auto_Graft_Test.py` header comment corrected to match the file's
  actual name.
- Stray `"extract_io"` leftover `__all__` entry in `utils/parameters.py`,
  dead since `extract_io`'s a25 relocation to `core/core_api.py`.
- Stale docstring/comment references to symbols deleted this release
  (`models/parameters.py`'s `ParameterReport` docstring referenced the
  deleted `ParamNameReport`; `workflows/routing.py`'s `_group_by_equality`
  docstring referenced the deleted `n_way_parameter_report`).

### Removed

- breaking: `n_way_parameter_report`/`ParamNameReport`
  (`utils/parameters.py`/`models/parameters.py`) — this release's own
  earlier primitives, fully superseded by
  `build_parameter_reports`/`ParameterReport`; zero remaining `src/`
  callers confirmed before deletion.
- breaking: `variadic_compatible` (`utils/parameters.py`) — superseded,
  zero remaining callers; the resulting trade-off is documented directly in
  `workflows/parallel.py`.
- `agents/base.py`'s `_PARAM_SOURCE_LABELS` module constant — inlined at
  its one call site.

## [2.0.0a26] - 2026-08-07

Workflow revision: removes `Workflow`'s checkpoint/run-history mechanism in
favor of a per-run trace on the result envelope, and ships `GraphFlow` —
the first cyclic, dynamically-routed workflow kind.

### Added

- `GraphFlow` — a new `Workflow` kind for cyclic, dynamically-routed node
  graphs. Construction: `nodes: dict[str, AtomicInvokable]` (alias → node),
  `edges: list[tuple[str, str | None | AtomicInvokable]]` (a fixed edge, an
  explicit dead-end marker, or a router attachment — multiple routers per
  node fall out of the same flat list), a single `start: str`. Execution is
  superstep-batched: everything ready runs concurrently against a frozen
  state snapshot each round, and nothing merges until the whole round
  completes. A node can loop back to an earlier node based on a runtime
  routing decision — the capability none of the other five workflow kinds
  support. State flows through one shared, accumulating pool of named
  values (every node must return a dict); same-round write collisions are
  resolved via per-key `raise_on_collision`/`tiebreak` policy plus
  per-node, mutable `priority`. `max_edge_traversals` bounds runaway
  cycles; `stop_early` opts into stopping as soon as everything requested
  is already available. Returns a `GraphFlowResult` with `edge_traversals`
  (which nodes fired, round by round — always populated, independent of
  `include_trace`) and `termination_reason`
  (`"queue_empty"`/`"cap_hit"`/`"early_exit"`). See
  `examples/Workflow_Examples/05_GraphFlow.py`.
- `Workflow.include_trace: bool = True` — every workflow kind now exposes a
  mutable toggle for whether its result populates `trace` (see Changed).
- `IterativeFlow.checkers` — a list of `CheckerSpec` (judge + approval
  value tied to a specific body-step position), mutable post-construction
  via `add_checker`/`remove_checker`, replacing the old single fixed judge.
  `result_setting_indices` (plural) replaces `return_index`, letting more
  than one body step contribute to the running "current answer."
- `RoutingFlow.schema_mode` (`STRICT`/`PARTIAL`/`OPEN`, default `STRICT`) —
  controls how far a branch's own parameters may diverge from the
  router's, replacing the old `parameters` constructor override.
- `ParallelFlow.result_mode`/`selected_branches`/`result_keys` — replaces
  `output_type`/`output_indices`/`output_range`; automatic N-way
  cross-branch parameter reconciliation replaces the old `parameters`
  constructor override.

### Changed

- breaking: `Workflow`'s checkpoint/run-history mechanism is removed
  entirely — `checkpoints`, `get_checkpoint()`, `clear_memory()` no longer
  exist on any workflow. In its place, every `*FlowResult` carries a
  `trace: tuple[AtomicResult, ...] | None` field (the actual child results
  produced by that run, in true execution order), populated whenever
  `include_trace=True` (the default).
- breaking: `IterativeFlow`'s loop body now executes step-by-step directly
  (no nested `SequentialFlow`), letting a checker's early exit fire
  mid-body instead of only after the whole body completes each iteration.
- `IterativeFlowResult`/`RoutingFlowResult`/`ParallelFlowResult` gain
  fields reflecting the above (`exited_early`, `triggering_step`,
  `selected_branch`, `result_mode`, `selected_indices`, etc.) — see each
  class's own docstring for the full field list.

### Removed

- breaking: `BasicFlow` and `BasicFlowResult` — deleted outright.
  `BasicFlow`'s only reason to exist beyond calling its wrapped component
  directly was giving that component a `get_checkpoint()` method; once
  checkpoints are gone, nothing needs a `Workflow`-typed wrapper around a
  plain `AtomicInvokable` child anymore. `Sequential`/`Parallel`/`Routing`/
  `Iterative` now accept any `AtomicInvokable` step/branch/judge directly,
  no wrapping.
- breaking: `ParallelFlow.get_branch_results()`/`get_branch_result()`,
  `SequentialFlow.get_step_results()`/`get_step_result()`,
  `RoutingFlow.get_router_decision()`, `IterativeFlow.get_iteration_results()`
  — all removed outright (relied on the removed checkpoint mechanism). Use
  the new `trace` field on the result envelope instead.

## [2.0.0a25] - 2026-08-04

Cleanup release: four independent, low-risk tracks bundled into one pass
(not a single signature feature the way a24 was).

### Added

- `AtomicInvokable.__call__`/`async_call` gain `return_atomic_result_object:
  bool = False` — a keyword-only opt-in for the full `AtomicResult` envelope;
  guarded at construction time against colliding with a declared parameter
  name.
- `tests/fake_engines.py` — canonical `FakeLLMEngine` (dual response mode:
  dynamic `response_fn` or static `responses` list, with a friendly
  exhaustion error) plus an `echo_latest_user(prefix=...)` helper,
  consolidating 5 independently-duplicated fake-LLM-engine fixtures
  scattered across `tests/`. Includes a real `_call_provider_async`
  override — none of the prior fixtures supported scripted async responses.
  Test-only; not exported from `src/`.

### Changed

- breaking: `AtomicInvokable.__call__`/`async_call` now return the
  unwrapped `.result` payload by default instead of the full `AtomicResult`
  envelope — pass `return_atomic_result_object=True` to get the envelope
  back.
- `AtomicInvokable.__name__`/`__doc__` are now kept in sync with
  `.name`/`.description` (refreshed on every `description` mutation, not
  just at construction), letting `Tool` treat any `AtomicInvokable` as an
  already-well-behaved plain callable with no special-case branch — except
  one narrow, deliberate exception: `Tool.async_execute` dispatches through
  `async_call` for an `AtomicInvokable` target, so the real async path is
  awaited instead of silently thread-offloading the sync `__call__`.
- `extract_io` relocated from `utils/parameters.py` to new
  `core/core_api.py` (resolves a real circular-import constraint — `utils`
  sits below `core` in the layered dependency topology) and gained a direct
  `AtomicInvokable` branch that reuses declared `parameters`/`return_type`
  instead of falling back to signature introspection. Remains a public root
  export.
- `toolify()` / `ToolAgent.register()` / `batch_register()`: wrapping an
  existing `AtomicInvokable` (`Tool` subclasses included) with identity
  overrides is now delegation-by-reference — a new `Tool` that calls the
  original's real `.invoke()`/`__call__` under the hood, never mutating the
  original — instead of raising `ToolRegistrationError`.
- breaking: `filter_extraneous_inputs: bool` removed as a constructor
  parameter across every `AtomicInvokable` subclass (`Tool`, `MCPProxyTool`,
  `PyA2AtomicTool`, every `Workflow` kind, every `Agent` kind, every
  `LLMEngine` adapter) — `invoke()`/`async_invoke()` now always silently
  drop unrecognized input keys; there is no longer a way to opt into a
  hard-reject `TypeError` on extras via that path. (`__call__`/`async_call`'s
  own keyword-argument binding was already unconditionally strict,
  independent of this flag, and required no change.)
- `Command`'s runtime-input handling softened: caller-supplied inputs to
  `invoke()`/`async_invoke()` are now silently ignored (never merged into
  `fixed_inputs`) instead of raising `TypeError`.
- `toolify()`/`batch_toolify()` and `ToolAgent.register()`/`batch_register()`
  drop their own `filter_extraneous_inputs`/`batch_filter_inputs` override
  parameters.
- A2A wire protocol: `PyA2AtomicHost` no longer emits
  `filter_extraneous_inputs` in remote invokable metadata;
  `PyA2AtomicClient.get_invokable_metadata()` no longer requires it as a
  response key. MCP unaffected (never referenced it).

### Fixed

- `MCPProxyTool.execute()` / `PyA2AtomicTool.execute()` now wrap their
  transport call in the same `try/except -> ToolInvocationError` handling
  their `async_execute()` overrides already had — a sync/async
  exception-parity gap identified during the a24 hygiene audit, dormant for
  every internal `ToolAgent` call path (always dispatches via
  `async_invoke`) but real for any caller invoking one of these proxy tools
  directly.
- `Tool._extra_description()` — restored the override (lost during the
  `AtomicInvokable`-as-plain-callable rework) that chains into a wrapped
  `AtomicInvokable`'s own `_extra_description()`. Wrapping a
  `Workflow`/`Command`/`StructuredInvokable` in a `Tool` — directly, via
  `toolify()`, or via `register(name=..., description=...)` — was silently
  dropping that descriptive text (e.g. "Output schema: [...]") from both
  `.description` and A2A metadata. Found by a post-implementation
  code-integrity audit, not by the test suite: the regression test that
  would have caught it was deleted rather than adapted during the rework.
- Stale docstrings/comments describing the removed `filter_extraneous_inputs`
  dual-mode behavior (`core/Invokable.py`'s `filter_inputs()` docstring,
  `tools/Toolify.py`'s existing-Tool route comment).
- `examples/Agent_Examples/03_Schema_Test.py`'s "STRICT vs PERMISSIVE"
  narrative — this file never referenced `filter_extraneous_inputs` directly
  so it was invisible to the grep-based casualty sweep, but its premise
  (unknown keys raise on the "strict" tool) broke once filtering became
  unconditional; rewritten to describe the real distinction (required
  keyword-only params with no `**kwargs` sink vs. a permissive `**kwargs`
  catch-all).
- `examples/Invokable_Examples/02_Command.py`: a pre-existing bug where
  `multiply_6_by_7()`'s `__call__` result — now unwrapped by default per
  this release's own `__call__` change — was still accessed via `.result`.

### Removed

- `AtomicInvokable.filter_extraneous_inputs` property/setter.

## [2.0.0a24] - 2026-08-02

### Added

- `SelfAskAgent` rebuilt from scratch as a direct `BasicAgent` subclass:
  adaptive, free-flowing self-questioning (`[CATEGORY] content` lines, no
  JSON) that produces one or more categorized thoughts per round
  (`OBSERVATION`/`QUESTION`/`CLARIFICATION`/`ASSUMPTION`/`REASON`/
  `INSTRUCTION`/`OTHER`) until it signals it's done or hits
  `max_thinking_rounds`. `thoughts_per_round` caps how many thoughts survive
  each round — set it to `1` to force a task through multiple rounds
  structurally rather than by instruction alone. `thinking_instructions`
  accepts its own parameterized `PromptConfig`, reconciled against
  `role_prompt`'s own parameters at construction time. A public `thoughts`
  property exposes the full per-round thought history.
- Three new example scripts under `examples/Thinking_Examples/` (easy,
  medium, and a structurally-sequential hard puzzle demonstrating the
  multi-round guarantee).

### Changed

- Every agent (`BasicAgent`, `ToolAgent`, `PlanActAgent`, `ReActAgent`,
  `SelfAskAgent`) now runs a unified three-hook lifecycle — `think`/
  `prepare`/`act` (+ async mirrors) — in place of the single `_progress`/
  `_async_progress` hook introduced in a23. `Agent.render_task()` is now a
  concrete, shared method on the base class; subclasses implement only a
  small `_render_task_messages` hook instead of rebuilding the whole render
  pipeline.
- `PlanActAgent`/`ReActAgent`'s generation-retry loops (render → call LLM →
  decode → validate → retry-with-feedback) now share one implementation.
  Retry feedback messages no longer duplicate the raw model output or a
  failed step's text a second time in the same turn.
- `tools/prebuilt.py`'s `safe_eval` now uses `ast.literal_eval` instead of a
  stripped-down `eval`, so it only ever accepts literal Python values
  (strings, numbers, containers, booleans, `None`) instead of arbitrary
  expression syntax.

### Removed

- The `ThinkingAgent`/`PlanAskAgent` classes and the JSON-based batch
  self-questioning approach they implemented — fully superseded by the
  single `SelfAskAgent` above.

## [2.0.0a23] - 2026-07-25

### Added

- Unified **task-oriented agent lifecycle**: a new `AgentTask` hierarchy
  (`ToolAgentTask`/`PlanActTask`/`ReActTask`) plus `Agent._initialize_task`/
  `_progress` hooks replace the old per-subclass `_invoke`/`_ainvoke` shell
  and the `ToolAgentRunState`/`PlanActRunState`/`ReActRunState` family it
  relied on. Every agent type — `BasicAgent`, `PlanActAgent`, `ReActAgent`
  — now runs through one shared loop in `Agent.invoke()`/`async_invoke()`:
  `while not task.complete: task = progress(task)`.
- `Agent.render_task(task, ...) -> list[dict[str, str]]` — new per-subclass
  hook that builds the exact message list sent to the LLM for a given
  phase, replacing ad hoc inline message construction. `AgentTask` gains
  `system_prompt_name`, `historic_messages`, and `task_messages` fields.
- `ToolAgent._compile_batches_from_deps`/`_check_cascade_failure` — shared
  batch-preparation and cascade-failure helpers, generalized from logic
  that previously lived only in `PlanActAgent`.

### Changed

- `LLMRecord.messages` now stores a full, self-contained copy of every
  message built for a given generation attempt (growing across retries),
  rather than only the newest delta on top of prior calls.
- `ReActAgent`'s per-round prompt content tightened for reliability on
  cheaper models: the running-plan snapshot shown each round is now a
  directive-free data block, and the final per-round instruction drops
  rules already covered by the system prompt in favor of an explicit
  reminder not to redo work already available via cache or existing
  placeholders.
- `PlanActAgent`'s task prompt is now visually separated (wrapped in a
  `===== CURRENT TASK =====` banner) from the planning instruction that
  follows it, instead of the two being glued into one unstructured block.

### Fixed

- `ToolAgent.clear_tools()` no longer removes the mandatory return tool.
- A missing required field on the base agent task-construction path, found
  during this release's test repair.

### Removed

- The old `ToolAgentRunState`/`PlanActRunState`/`ReActRunState` family and
  the `_invoke`/`_ainvoke`/`make_result` lifecycle it backed, fully
  superseded by the task-oriented lifecycle above.
- `ReActKAgent`, a K-fixed-batch step-planning agent that was built out in
  full and live-tested, then removed after testing showed it less reliable
  than the existing `PlanActAgent`/`ReActAgent` strategies — reasoning that
  informs later steps in a fixed batch isn't reliably persisted or
  inferable across separate LLM calls the way it is within a single
  planning call or a single-step reactive call.

## [2.0.0a22] - 2026-07-17

### Added

- `AtomicInvokable._extra_description()` — new overridable hook (default
  `""`) composed into the `description` getter after any parameter-description
  bullets. Implemented across every local wrapper-shaped class (`Tool`,
  `Command`, `BasicFlow`, `SequentialFlow`, `IterativeFlow`, `ParallelFlow`,
  `RoutingFlow`) so each surfaces its wrapped component's extra description
  content per its own "source of truth" for `return_type`.
- `PyA2AtomicHost`/`PyA2AtomicTool` — new `extra_description` wire field so a
  remote wrapper's real extra description content survives the A2A round trip.
- `PromptConfig.field_specs: dict[str, dict[str, Any]] | None` — sparse
  per-field `{description, type, default}` spec, replacing the old flat
  `defaults` map. A field is required iff its entry omits `default`.
- `Agent.get_reserved_parameters()` — new overridable classmethod for
  subclasses to declare additional reserved parameter names beyond the base
  `run_id`.
- New `utils/parameters.py` primitives: `semantically_compatible`/
  `semantically_identical`, `parameter_overlap`/`parameter_collisions`, a
  variadic cross-source compatibility check, and `insert_by_category` (a
  stable-sort-based parameter-graft helper superseding the old
  keyword-only-before-`**kwargs` insertion).
- `_normalize_prompt_template` / `_try_parse_clean_field` (`utils/parameters.py`)
  — construction-time template scanner: malformed/non-identifier brace shapes
  are brace-escaped into inert literal text instead of raising or silently
  mis-parsing.

### Changed

- `Agent.__init__`'s `context_properties` constructor param renamed to
  `extra_parameters` and normalized via `to_paramspec_list` (declared kinds
  preserved, no forced `KEYWORD_ONLY` coercion). Reserved-name reconciliation
  against `pre_invoke`/`post_invoke`/`extra_parameters` now runs through the
  new `utils/parameters.py` primitives.
- `Agent._invoke`/`_ainvoke` (and every concrete subclass's override):
  `context: dict` renamed to `inputs: dict`, now receiving the full filtered
  inputs dict rather than a context-only slice.
- `AgentRecord.user_prompt`: `PromptConfig` reverted to plain `str` (already
  fully rendered at invocation time — no more redundant re-render in
  `render_turn`). `AgentRecord.context` renamed to `inputs`.
- `ToolAgentRunState`/`PlanActRunState`/`ReActRunState` gain a required
  `inputs: dict[str, Any]` field, threaded through each `_initialize_run_state`/
  `_ainitialize_run_state` override.
- `PromptConfig.render()` collapsed to a single resolve-or-collect-missing pass
  (raises one `ValueError` naming every missing required field, not
  fail-on-first).
- `PromptConfig.to_dict()` emits `"parameters"` (full `ParamSpec.to_dict()` per
  field) instead of the old `"defaults"` key.
- `StructuredInvokable`'s `description` setter override removed — fully
  inherits the base setter's unified `ValueError`-for-both-cases contract
  (previously split `TypeError`/`ValueError`, uniquely among invokable types).
- `PlanActAgent`/`ReActAgent.__init__`: `context_enabled` moved from a
  keyword-only parameter to positional-or-keyword, matching
  `Agent.__init__`/`ToolAgent.__init__`'s parameter ordering.

### Fixed

- `PlanActAgent._generate_plan`/`_agenerate_plan`: retry attempts now capture
  both the model's failed output and the injected feedback message in the
  `LLMRecord` delta (previously captured only the last message, silently
  dropping the feedback message from the record on every retry).
- `ToolAgent.make_result`: removed a dead defensive `is not None` filter over
  `llm_records`' token usage — `LLMResult.token_usage` is a required,
  non-`None`-validated field, so the guarded state could never occur.
- `PyA2AtomicHost`'s metadata payload previously sent the *rendered*
  description (parameter bullets already baked in); now sends the raw
  description plus a separate `extra_description` field, preventing duplicated
  bullet content after a remote round trip.
- `examples/Agent_Examples/07_Prompt_Parameters.py`'s required-parameter
  display used an incorrect sentinel check (`param.default.__class__.__name__
  == "_NO_VAL"`, always `False`) and always showed a bogus default value for
  the demo's required parameter — same defect class previously fixed in
  `examples/Tool_Examples/04_MCPProxyTool.py` (a21), reintroduced in this
  release's rewrite of the file; now uses the correct `param.default is
  NO_VAL` identity check.
- Six example files (`Agent_Examples/03_Schema_Test.py`,
  `05_Passthrough_Test.py`, `06_Branching_Conversations.py`,
  `PlanAct_Examples/01_plugins_test.py`, `03_agentic_story_builder.py`,
  `Tool_Examples/06_toolify_example.py`) had pre-existing UTF-8 mojibake
  corruption (multi-generation mis-decode/re-encode of em-dashes,
  multiplication signs, box-drawing characters, smart quotes) repaired at the
  byte level — no logic or behavior changes.

### Removed

- `Agent.update_prompt` (base) and its `BasicAgent`/`PlanActAgent`/
  `ReActAgent` overrides — zero real callers anywhere in `src/`/`examples/`.
- `BasicAgent`'s `extra_context_properties` constructor param, `role_prompt`
  setter, `set_context_properties`, `set_extra_context_properties` — role-prompt
  placeholders are now `BasicAgent`'s sole `extra_parameters` source;
  `role_prompt` is read-only post-construction, matching every other
  `AtomicInvokable`'s fixed-topology invariant.
- `ToolAgent.__init__`'s `context_properties` param — no legitimate extra-
  parameter source exists for `ToolAgent` today, so it was dropped rather than
  renamed.
- `Agent._validate_pre_post_overlap_shapes`, `_warn_reserved_name_collisions`,
  `_compose_agent_parameters` (+ `_insert_before_varkw`),
  `_update_context_param`, `_set_context_properties`, public
  `set_context_properties`; module-level `CONTEXT_PARAM`,
  `normalize_context_properties`, `build_context_description`. No `context:
  dict` schema parameter exists anywhere in base `Agent` anymore.
- `PromptConfig`'s `strict` construction-time toggle — traced dead on the only
  call path that mattered (`Agent`'s own required-context-property check
  already raises before `render()` is reached).

## [2.0.0a21] - 2026-07-12

### Added

- `MCPClientHub` — `read_timeout_seconds`, `client_kwargs`, `session_kwargs`
  constructor params: `client_kwargs` forwards into whichever transport
  constructor is active (`StdioServerParameters`/`sse_client`/
  `streamable_http_client`); `session_kwargs` + converted
  `read_timeout_seconds` forward into `ClientSession`.
- `MCPClientHub.async_list_tools`/`async_call_tool` — public API (renamed
  from private `_alist_tools`/`_acall_tool`); `call_tool`/`list_tools` are
  now thin sync wrappers over these.
- `MCPError`, `MCPConnectionError`, `MCPToolError` typed exceptions.
- `PyA2AtomicClient` — `timeout: float = 600`, `google_a2a_compatible: bool
  = False` constructor params (matching `A2AClient`'s full signature), with
  read-only `timeout`/`google_a2a_compatible` properties.
- `PyA2AtomicClient.async_create(...)` — non-blocking construction
  (including the agent-card fetch) via `asyncio.to_thread`.
- `PyA2AtomicError`, `PyA2AtomicConnectionError` typed exceptions.
- `PyA2AtomicHost` — dict-form construction (`invokables: Mapping[str,
  AtomicInvokable]`) and `register(invokable, remote_name=None)`: the
  registry key is now a registry-local `remote_name` alias, independent of
  an invokable's intrinsic `.name`. Reserved wire function names
  (`list_invokables`, `get_invokable_metadata`) are now rejected as
  `remote_name`s across all registration paths.
- `dataclass_record_to_dict` — public util in `utils/core.py` (relocated).

### Changed

- `MCPClientHub.refresh(headers=None, client_kwargs=None,
  session_kwargs=None)` — raises `ValueError` if all three are `None`;
  each provided bucket wholesale-replaces the corresponding stored value.
- `PyA2AtomicClient.refresh(headers=None, timeout=None,
  google_a2a_compatible=None)` — raises `ValueError` if all three are
  `None`; mutates the existing `A2AClient` in place rather than
  reconstructing a new one.
- `PyA2AtomicHost`'s `get_invokable_metadata`/`list_invokables` responses
  report the registry key (remote name) in their `"name"` field, not
  necessarily the invokable's intrinsic `.name`.

### Fixed

- MCP and A2A transport/protocol/tool failures now raise typed exceptions
  instead of a bare `RuntimeError`, letting callers distinguish connection
  failures from malformed responses from remote-side errors.
- `PyA2AtomicClient.__init__` no longer silently swallows a construction-time
  agent-card fetch failure (removed unreachable dead-code try/except around
  a method that can never raise).
- `PyA2AtomicHost` no longer silently drops invokables that share an
  intrinsic `.name` across different namespaces — same-name invokables can
  now be registered under distinct `remote_name`s.
- `MCPClientHub._awith_session` no longer double-wraps exceptions that were
  already typed by the inner operation.
- `MCPClientHub.async_call_tool`'s input validation now runs consistently
  on both the sync and async call paths (previously async-only calls could
  bypass it).
- **`MCPClientHub.refresh()` and `PyA2AtomicClient.refresh()` no longer
  partially apply a failed refresh.** Both previously mutated stored
  state before the one step that could fail (a validation check /  the
  agent-card re-fetch), so a raised exception left the new — possibly
  invalid — config committed anyway. Both now validate/roll back before
  any state change is visible to callers.
- `PyA2AtomicClient.url` now reflects the live transport endpoint instead
  of a value cached at construction — `python_a2a`'s `A2AClient` rewrites
  its own endpoint after a successful call, which the cached copy never
  picked up.
- `examples/Tool_Examples/04_MCPProxyTool.py`'s required-parameter display
  used an incorrect sentinel check and always showed a bogus default value
  for required params.

### Removed

- `MCPProxyTool.headers` setter (getter-only; mutate via
  `client_hub.headers = ...` or `client_hub.refresh(...)`).
- `PyA2AtomicTool.headers` setter (getter-only; mutate via `client.headers
  = ...` or `client.refresh(...)`).

## [2.0.0a20] - 2026-07-11

### Added

- `AnthropicEngine` — new `llm/` adapter using the Anthropic Messages API.
  Single injected/auto-built `Anthropic`/`AsyncAnthropic` client; inline-only
  attachment pipeline (base64 image/PDF/document blocks, no Files API yet);
  `thinking_config`, `stop_sequences`, `top_p`, `top_k` request-level knobs;
  `block_separator` constructor param controlling how multiple response text
  blocks are rejoined (defaults to `""`, matching Anthropic's own
  streaming-SDK text-reconstruction convention).
- `LiteLLMEngine` — new `llm/` adapter over litellm's `completion`/
  `acompletion`, the first genuinely provider-agnostic gateway in this
  family. No injectable client (litellm exposes bare module functions);
  `provider`/`model` combine at call time into litellm's `f"{provider}/
  {model}"` convention; `drop_params` constructor knob controls whether
  litellm silently drops or raises on provider-unsupported generation
  params; inline attachment pipeline mirrors `AnthropicEngine`'s, with a
  documented gap for inline PDF attachments against `provider="mistral"`.
- `LLMEngine` base gained native async infrastructure: `_call_provider_async`,
  `_call_model_async`, `_call_with_retries_async`, and `async_invoke` (mirrors
  the sync path; remote engines override with native async SDK clients where
  available).
- `TokenUsage.response_tokens: int` — new required base field: the
  visible-reply token count, distinct from hidden reasoning/thinking tokens
  folded into `generated_tokens`. Populated by every engine (trivial equality
  for providers with no reasoning concept, exact subtraction/native passthrough
  otherwise).
- `AnthropicTokenUsage`, `LiteLLMTokenUsage` result records.
- Per-engine `_should_retry` overrides recognizing each provider SDK's real
  transient-error shape (connection/timeout exceptions, retryable HTTP status
  codes) — the shared base default never matched any real provider exception
  hierarchy and was effectively dead code.
- `inline_cutoff_chars` constructor param added to `AnthropicEngine` and
  `LiteLLMEngine`, matching the existing OpenAI/Mistral text-attachment
  truncation behavior.
- `constants/llm.py` and `utils/llm.py` (renamed from `constants/engines.py` /
  `utils/engines.py`): shared attachment extension/MIME-prefix policy and the
  `validate_attachment_path` helper used by every engine.

### Changed

- `engines/LLMEngines.py` split into a dedicated `llm/` package with one
  module per provider (`base.py`, `openai_engine.py`, `gemini_engine.py`,
  `mistral_engine.py`, `llama_engine.py`, plus the two new adapters above).
  Clean break, no compatibility shim.
- `OpenAIEngine`, `GeminiEngine`, `MistralEngine` repaired and upgraded to the
  new async infrastructure: single injectable sync-or-async client with
  isinstance-based call routing, `temperature: float | None` (omits the
  param when `None` instead of a fragile model-name string match), broken/
  bare exception handling replaced with real SDK exception types.
- `LlamaCppEngine` constructor overhauled: identity params moved to the top
  of the signature, generation defaults made explicit and `None`-omitted,
  Hugging Face download options made keyword-only, remaining Llama
  constructor surface absorbed into `**llama_kwargs`.
- Attachment placement standardized across all five remote engines: content
  is now appended to the **last** user turn (previously a 3-vs-2 split
  between prepending to the first user turn and appending to the last).
- `LiteLLMEngine`'s `timeout_seconds` default changed from 30s to 600s,
  matching every other remote engine.
- `LLMEngine.invoke`/`async_invoke` now include the original exception's
  message text in the wrapped `LLMEngineError`, not just a generic
  "...invoke failed" string.

### Fixed

- **`MistralEngine._extract_token_usage` crashed on every real call** —
  accessed `usage.prompt_tokens_details` as a plain attribute, which the real
  `mistralai.UsageInfo` model never declares. Masked in tests by a
  `SimpleNamespace` fixture that always had the attribute present.
- **`AnthropicEngine._extract_token_usage` crashed whenever extended
  thinking actually returned thinking-token data** — called `.get(...)` on
  `output_tokens_details`, which the real SDK types as a Pydantic model with
  no `.get()` method, not a dict.
- Inconsistent attachment-error wrapping: `AnthropicEngine`/`LiteLLMEngine`
  now wrap unexpected `_prepare_attachment` exceptions into `LLMEngineError`,
  matching the existing `OpenAIEngine`/`MistralEngine` behavior.
- `OpenAIEngine.to_dict()` no longer hardcodes a redundant `"type"` key
  (the base `to_dict()` already sets it generically).

### Removed

- `engines/LLMEngines.py` (fully replaced by the `llm/` package split).
- 13+ dead read-only accessor properties on `LlamaCppEngine`.
- Redundant per-subclass `response_tokens` declarations on
  `OpenAITokenUsage`/`MistralTokenUsage`/`LlamaCppTokenUsage`, and
  `GeminiTokenUsage.candidates_token_count` (now the base `response_tokens`
  field, populated natively for Gemini).

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
