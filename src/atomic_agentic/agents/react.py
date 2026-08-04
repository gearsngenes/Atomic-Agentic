"""
ReActAgent: Iterative LLM Actor with Per-Step Reactive Planning

This module provides ``ReActAgent``, a concrete ``ToolAgent`` subclass that
implements a **dynamic iteration** strategy: each invoke step is driven by a
fresh LLM call that observes the current running-plan snapshot and emits exactly
one next tool call.

Iteration Model
---------------
At each step the agent builds a temporary message thread from the static base
messages plus a compact running-plan snapshot showing completed steps (with
placeholder references, optional raw-result previews while their observability
window is open, and one-sentence descriptions). The LLM responds with a single
JSON step object specifying the next tool to call, its arguments, an
observability duration, and a human-readable description.

Observability Window
--------------------
Each step may declare a ``duration`` field (integer >= 0). While ``duration > 0``
the step's raw result appears in subsequent snapshots; once exhausted the result
is visible only as a ``<<__sN__>>`` placeholder. This limits context growth while
keeping recency-relevant results accessible.

Execution
---------
Each round, ``think()`` generates and validates exactly one step via the
LLM; ``prepare()`` cascade-checks, resolves placeholders, and stamps it
into the running blackboard; ``act()`` (base ``ToolAgent``, final) executes
it. Repeats until the return tool is emitted.

Contrast
--------
For one-shot full-plan generation with concurrent batches see
``agents/planact.py`` (``PlanActAgent``). For the shared iteration loop,
blackboard management, and tool registry see ``agents/toolagent.py``
(``ToolAgent``).
"""

from __future__ import annotations
from typing import Any, Callable, Literal, Mapping, Optional
import pprint

from .toolagent import ToolAgent
from .prompts import ORCHESTRATOR_PROMPT
from ..constants.agents import (
    STEP_FIELD,
    DESCRIPTION_FIELD,
    TOOL_FIELD,
    ARGS_FIELD,
    DURATION_FIELD,
    RETURN_TOOL_FULL_NAME,
    REACT_FIELDS,
    REQUIRED_REACT_FIELDS,
    BASE_STEP_FIELDS,
)
from ..core import AtomicInvokable
from ..constants.core import NO_VAL
from ..llm.base import LLMEngine
from ..exceptions import ToolAgentError
from ..models.agents.tasks import ReActTask, ReActStepMeta
from ..models.agents import BlackboardSlot
from ..models.agents.records import AgentRecord
from ..utils.agents import extract_dependencies

# --------------------------------------------------------------------------- #
# ReAct Agent
# --------------------------------------------------------------------------- #
class ReActAgent(ToolAgent):
    """
    Iterative agent with reactive step-by-step planning (ReAct-style architecture).

    **Design**: ReActAgent implements dynamic iteration: one step is emitted per LLM
    turn from a compact running-plan snapshot. Result references are always visible
    by placeholder; raw result previews are shown only while their observability
    duration remains active. Generated step descriptions preserve semantic intent
    across turns without exposing raw results.

    1. **Initialization** (``_initialize_task``)
       - Pre-allocates a fixed-size running blackboard: ``tool_calls_limit + 1`` slots
         to accommodate non-return tool calls plus one return call.
       - Requires ``tool_calls_limit`` to be a concrete integer.
       - Initializes ReAct-specific per-step metadata:
         observability counters and generated step descriptions.

    2. **Decision** (``think()`` — one step per round)
       - Builds a fresh temporary message list from static base messages.
       - Appends a compact running-plan snapshot containing executed steps,
         descriptions, unresolved args, result_ref placeholders, and any currently
         observable_result values.
       - Asks the LLM for the next single step; validates it end-to-end.
       - Stashes the validated ``(slot, duration, description)`` onto
         ``task.generated_step`` for ``prepare()`` to apply next round.

    3. **Preparation** (``prepare()``)
       - Cascade-checks dependencies; resolves placeholders into concrete tool args.
       - Stores the prepared slot in ``running_blackboard[step_index]``.
       - Stores duration and description in the ReAct task's ``step_meta``.
       - Sets ``prepared_steps = [step_index]``.

    4. **Execution** (``act()``, base ``ToolAgent``, final)
       - Executes the single prepared step. Result is stored; the loop
         returns to ``think()`` for the next step.

    5. **Termination**
       - When the return tool is emitted and executed, ``task.complete`` is set.
       - Running blackboard is persisted by ``_build_record_from_task`` if
         ``context_enabled=True``.

    Advantages
    ~~~~~~~~~~
    - Fully adaptive: each step can react to prior tool results.
    - Context-hygienic: running-plan messages are rebuilt fresh each turn.
    - Interpretable: descriptions, placeholders, and optional observations provide
      a compact execution trace.

    Limitations
    ~~~~~~~~~~~
    - Higher latency: one LLM call per step.
    - No concurrency: only one step executes per iteration.
    - Step quality depends on the model's ability to select the next best action.

    Parameters (construction)
    ~~~~~~~~~~~~~~~~~~~~~~~~
    ``tool_calls_limit`` (int, REQUIRED): Must be a concrete integer >= 0.
    Determines pre-allocated blackboard size. Cannot be ``None``.
    """

    def __init__(
        self,
        name: str,
        namespace: str,
        description: str,
        llm_engine: LLMEngine,
        context_enabled: bool = False,
        *,
        tool_calls_limit: int = 25,
        fail_fast: bool = True,
        generation_retries: int = 0,
        peek_at_cache: bool = False,
        response_preview_limit: Optional[int] = None,
        blackboard_preview_limit: Optional[int] = None,
        pre_invoke: AtomicInvokable | Callable[..., Any] | None = None,
        post_invoke: AtomicInvokable | Callable[..., Any] | None = None,
        post_result_key: Optional[str] = None,
        records_window: int | None = None,
        assistant_response_source: Literal["raw", "final"] = "raw",
    ) -> None:
        """
        Initialize a ReActAgent.

        ``tool_calls_limit`` defaults to ``25`` and must be a concrete integer
        >= 0 — ``None`` is not accepted because the running blackboard is
        pre-allocated to ``tool_calls_limit + 1`` slots at initialization.
        ``"reason_then_act"`` is the key under which the built-in orchestrator
        prompt is registered in ``self._system_prompts``. No extra_parameters
        keyword is passed to ``super().__init__(...)`` at all (``ToolAgent.__init__``
        accepts none).
        """
        super().__init__(
            name=name,
            namespace=namespace,
            description=description,
            llm_engine=llm_engine,
            context_enabled=context_enabled,
            tool_calls_limit=tool_calls_limit,
            fail_fast=fail_fast,
            generation_retries=generation_retries,
            peek_at_cache=peek_at_cache,
            response_preview_limit=response_preview_limit,
            blackboard_preview_limit=blackboard_preview_limit,
            pre_invoke=pre_invoke,
            post_invoke=post_invoke,
            post_result_key=post_result_key,
            records_window=records_window,
            assistant_response_source=assistant_response_source,
        )
        self._system_prompts["reason_then_act"] = ORCHESTRATOR_PROMPT

    # ------------------------------------------------------------------ #
    # Property Overrides
    # ------------------------------------------------------------------ #
    @property
    def tool_calls_limit(self) -> int:
        """Max allowed non-return tool calls per invoke() run. Must be an int >= 0."""
        return self._tool_calls_limit

    @tool_calls_limit.setter
    def tool_calls_limit(self, value: int) -> None:
        """Reject None and non-integer values; ReActAgent pre-allocates by this count."""
        if type(value) is not int or value < 0:
            raise ToolAgentError("ReActAgent requires tool_calls_limit to be an int >= 0.")
        self._tool_calls_limit = value

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #
    def _validate_react_prepare_state(self, task: ReActTask) -> None:
        """
        Validate cursor bounds, prior-step processing, and step_meta length.
        """
        prefix_len = task.next_step_index
        if type(prefix_len) is not int or prefix_len < 0:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: next_step_index must be an "
                f"int >= 0; got {prefix_len!r}."
            )
        if prefix_len >= len(task.running_blackboard):
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: next_step_index exceeds run "
                f"blackboard capacity ({prefix_len} >= {len(task.running_blackboard)})."
            )
        if prefix_len > 0:
            prev = task.running_blackboard[prefix_len - 1]
            # With fail_fast=False a previous step may be FAILED rather than EXECUTED;
            # both count as "processed" and allow generation of the next step.
            prev_processed = prev.is_executed() or (not self._fail_fast and prev.is_failed())
            if not prev_processed:
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: step {prefix_len - 1} was "
                    f"not executed before the next prepare call "
                    f"(status={prev.status!r})."
                )
        if len(task.step_meta) != len(task.running_blackboard):
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: step_meta length must match "
                f"running_blackboard length "
                f"({len(task.step_meta)} != {len(task.running_blackboard)})."
            )

    def _process_next_step_output(
        self,
        *,
        parsed: Any,
        expected_step: int,
        cache_blackboard: list[BlackboardSlot],
        max_duration: int,
        valid_cache_indices: frozenset[int],
        failed_cache_indices: frozenset[int],
    ) -> tuple[BlackboardSlot, int, str] | str:
        """
        Validate one parsed ReAct step and return the slot, duration, and description.

        Receives the already-extracted ``parsed`` value. LLMRecord construction
        lives in ``ToolAgent._run_generation_retry_loop`` (the shared caller,
        via ``_generate_next_step``'s ``validate`` binding). All LLM-facing
        validation failures return a plain feedback string (no class/name
        prefix); engine-contract violations raise ``ToolAgentError``.

        Budget enforcement (step 9 below) is a real boundary check, not a
        dead guard: unlike ``PlanActAgent`` (whose budget is validated once
        against the *entire* plan before anything executes), ``ReActAgent``
        generates one step at a time, so this is the only point that can
        catch "the model didn't terminate by the last available slot"
        before that step is ever prepared/executed. Nothing else in the
        lifecycle enforces it — dropping it (as an earlier pass briefly
        did) lets an over-budget non-return tool call actually run.

        Steps
        -----
        1. Reject non-Mapping parsed values.
        2. Build ``raw_payload``; check for unsupported keys.
        3. Require ``DURATION_FIELD`` and ``DESCRIPTION_FIELD`` present.
        4. Delegate to ``_validate_tool_step_dict``; propagate string feedback.
        5. Extract and range-validate ``duration``.
        6. Extract, type-check, strip, and non-empty-check ``description``.
        7. Convert to ``BlackboardSlot`` via ``_tool_step_dict_to_slot``.
        8. Verify tool is registered; unknown tool → feedback string.
        9. Budget enforcement: if this is the last available slot
           (``expected_step == self._tool_calls_limit``), the tool must be
           the return tool.
        10. Require return-tool duration == 0.
        11. Validate cache refs — three categories: out-of-range, failed-in-conv
            (with tool+error detail), out-of-conv.
        12. Validate step dependencies are prior-only.
        13. Return ``(slot, duration, description)``.
        """
        # 1. Structural check — LLM must return a JSON object, not an array or scalar.
        if not isinstance(parsed, Mapping):
            return f"next step output must be a JSON object; got {type(parsed).__name__!r}."

        raw_payload = dict(parsed)

        # 2. Field-set checks.
        extra = set(raw_payload) - REACT_FIELDS
        if extra:
            return f"next step contains unsupported keys: {sorted(extra)!r}."

        # 3. Presence checks for ReAct-specific required fields.
        if DURATION_FIELD not in raw_payload:
            return f"next step missing required key {DURATION_FIELD!r}."

        if DESCRIPTION_FIELD not in raw_payload:
            return f"next step missing required key {DESCRIPTION_FIELD!r}."

        # 4. Core step-dict validation (tool name, args shape, await_step, etc.).
        step_payload = self._validate_tool_step_dict(
            raw_payload,
            expected_step=expected_step,
            allowed_fields=REACT_FIELDS,
            required_fields=REQUIRED_REACT_FIELDS,
            context="next step",
        )
        if isinstance(step_payload, str):
            return step_payload

        # 5. Duration extraction and range check.
        duration = step_payload.pop(DURATION_FIELD)
        if type(duration) is not int or duration < 0 or duration > max_duration:
            return (
                f"next step {DURATION_FIELD!r} must be an int in "
                f"[0, {max_duration}] for expected_step={expected_step}; got {duration!r}."
            )

        # 6. Description extraction, type check, and normalisation.
        description = step_payload.pop(DESCRIPTION_FIELD)
        if type(description) is not str:
            return f"next step {DESCRIPTION_FIELD!r} must be a string; got {type(description).__name__!r}."

        description = description.strip()
        if not description:
            return f"next step {DESCRIPTION_FIELD!r} cannot be empty."

        # 7. Convert to BlackboardSlot — engine contract; let ToolAgentError propagate.
        slot = self._tool_step_dict_to_slot(
            step_payload,
            step=expected_step,
            allowed_fields=BASE_STEP_FIELDS,
            context="next step",
        )

        # 8. Verify tool is registered.
        if not self.has_tool(slot.tool):
            return f"next step references unknown tool {slot.tool!r}."

        # 9. Budget enforcement: the final available slot must be the return step.
        if expected_step == self._tool_calls_limit and slot.tool != RETURN_TOOL_FULL_NAME:
            return (
                f"step {expected_step} is the last available slot under "
                f"tool_calls_limit={self._tool_calls_limit}; it must be the return tool."
            )

        # 10. Return-tool must use duration 0.
        if slot.tool == RETURN_TOOL_FULL_NAME and duration != 0:
            return f"return tool must use {DURATION_FIELD!r} 0; got {duration!r}."

        # 11. Cache reference validation — three categories.
        cache_len = len(cache_blackboard)
        cache_refs = extract_dependencies(slot.args, placeholder_pattern=self.CACHE_REF_PATTERN)

        # Category 1: index outside the cache entirely.
        out_of_range = [idx for idx in cache_refs if idx < 0 or idx >= cache_len]
        if out_of_range:
            return (
                f"next step references cache indices that do not exist: "
                f"{sorted(set(out_of_range))!r} (cache has {cache_len} entries)."
            )

        # Category 2: in-conversation slot that failed — include tool+error.
        failed_in_conv = [idx for idx in cache_refs if idx in failed_cache_indices]
        if failed_in_conv:
            details = "; ".join(
                f"entry {idx} ({cache_blackboard[idx].tool}): {cache_blackboard[idx].error}"
                for idx in sorted(set(failed_in_conv))
            )
            return (
                f"next step references cache entries that failed in this conversation "
                f"and cannot be used: {details}."
            )

        # Category 3: in range but not from this conversation.
        out_of_conv = [
            idx for idx in cache_refs
            if 0 <= idx < cache_len
            and idx not in valid_cache_indices
            and idx not in failed_cache_indices
        ]
        if out_of_conv:
            return (
                f"next step references cache indices not part of this conversation: "
                f"{sorted(set(out_of_conv))!r}."
            )

        # 12. Step dependency validation.
        bad_step_deps = [
            dep for dep in slot.step_dependencies
            if dep < 0 or dep >= expected_step
        ]
        if bad_step_deps:
            return (
                f"next step has illegal deps "
                f"{sorted(set(bad_step_deps))!r}; deps must be < {expected_step}."
            )

        return slot, duration, description

    def prepare(self, task: ReActTask) -> ReActTask:
        """
        Apply this round's already-generated step: cascade-check, resolve
        placeholders, stamp into the running blackboard, advance the
        cursor.

        Unpacks ``task.generated_step`` (set by ``think()``/
        ``async_think()``) and resets it to ``NO_VAL``. Duration range,
        return-duration-zero, description shape, and ``PLANNED`` status are
        already guaranteed by ``_process_next_step_output``'s own success
        path (not re-checked — dead code, same cleanup ``PlanActAgent``
        applied to ``_finalize_planact_task``). The running-blackboard slot
        mismatch/already-filled checks are also dropped:
        ``running_blackboard`` is fixed-size and index-aligned from
        allocation (``step=i`` at position ``i``, never reordered), and
        ``next_step_index`` is written only here, always ``+1``, so slot
        ``prefix_len`` is targeted exactly once across the whole run.

        **Cascade path** (``fail_fast=False``): if any ``step_dependencies``
        entry is FAILED in the running blackboard, the return tool raises
        immediately; non-return slots are marked FAILED and this method
        returns early with ``prepared_steps`` left empty — ``act()`` will
        no-op this round.

        ``task.task_messages`` is cleared unconditionally before
        returning — on both the cascade-fail early return and the normal
        path — since an LLM call happened this round either way and the
        next round rebuilds fresh.
        """
        prefix_len = task.next_step_index
        generated_slot, observe_duration, description = task.generated_step
        task.generated_step = NO_VAL

        # A successful generation turn consumed any raw results that were visible.
        for meta in task.step_meta:
            if meta.observable > 0:
                meta.observable -= 1

        # Fill the preallocated running-blackboard slot.
        slot = task.running_blackboard[prefix_len]
        slot.tool = generated_slot.tool
        slot.args = generated_slot.args
        slot.result = NO_VAL
        slot.error = NO_VAL
        slot.step_dependencies = generated_slot.step_dependencies
        slot.await_step = generated_slot.await_step

        # Cascade-fail: when fail_fast=False, propagate failures through arg dependencies.
        if not self._fail_fast:
            board = task.running_blackboard
            if self._check_cascade_failure(slot, board):
                task.step_meta[prefix_len].description = description
                task.next_step_index = prefix_len + 1
                task.task_messages.clear()
                return task  # prepared_steps stays []; act() will no-op this round

        # Resolve placeholders after stamping the planned slot into the running task.
        slot.resolved_args = self._resolve_placeholders(slot.args, task=task)
        slot.status = BlackboardSlot.PREPARED

        # Write per-slot metadata.
        task.step_meta[prefix_len].observable = observe_duration
        task.step_meta[prefix_len].description = description

        task.prepared_steps = [prefix_len]
        task.next_step_index = prefix_len + 1

        task.task_messages.clear()
        return task

    async def async_prepare(self, task: ReActTask) -> ReActTask:
        """Async mirror of ``prepare``. Direct passthrough — ``prepare``
        has no I/O of its own to justify a thread offload."""
        return self.prepare(task)

    # ------------------------------------------------------------------ #
    # Task-lifecycle hooks
    # ------------------------------------------------------------------ #
    def _render_task_messages(self, task: ReActTask) -> list[dict[str, str]]:
        """
        Build this round's 3-message thread: banner / running-plan
        snapshot / next-call instruction.

        Build-once contract: returns ``task.task_messages`` as-is if
        already non-empty (cleared by ``prepare()`` at the end of the
        prior round, so this branch is fresh at the start of every new
        round):

        1. user — ``self._render_task_banner(task)`` directly.
        2. assistant — a thin, directive-free snapshot: a one-line header
           plus the running-plan data. Reads as state, not instruction,
           matching ``ToolAgent.render_turn``'s ``CACHED STEPS`` convention.
        3. user — the per-round instruction. Deliberately omits any rule
           already owned by ``ORCHESTRATOR_PROMPT``'s ``RUNTIME STATE``
           section (descriptions, result_ref usage, observable_result
           semantics, don't-copy-into-args); carries only what's new each
           round: the produce-next-call/return directive, an unconditional
           anti-duplication reminder, a FAILED-reference warning shown only
           when this round's snapshot contains a FAILED entry, the
           output-format directive, and the duration bound.

        EXECUTED steps are rendered with ``result_ref``, ``run_id``, and
        optionally ``observable_result``. FAILED steps are rendered with
        ``status="FAILED"`` and ``error``; they carry no ``result_ref`` (so
        the LLM cannot attempt to reference a non-existent result).
        """
        if task.task_messages:
            return task.task_messages

        prefix_len = task.next_step_index
        max_duration = max(0, self._tool_calls_limit - prefix_len)

        running_records: list[dict[str, Any]] = []
        for idx in range(prefix_len):
            slot = task.running_blackboard[idx]

            if slot.is_executed():
                record: dict[str, Any] = {
                    STEP_FIELD: slot.step,
                    DESCRIPTION_FIELD: task.step_meta[idx].description,
                    TOOL_FIELD: slot.tool,
                    ARGS_FIELD: slot.args,
                    "result_ref": f"<<__s{idx}__>>",
                    "run_id": slot.result.run_id,
                }
                if task.step_meta[idx].observable > 0:
                    record["observable_result"] = self._preview_blackboard_result(slot.result.result)
                running_records.append(record)

            elif slot.is_failed():
                running_records.append({
                    STEP_FIELD: slot.step,
                    DESCRIPTION_FIELD: task.step_meta[idx].description,
                    TOOL_FIELD: slot.tool,
                    ARGS_FIELD: slot.args,
                    "status": "FAILED",
                    "error": str(slot.error),
                })
            # Empty/PLANNED slots are not yet part of the running plan; skip.

        if running_records:
            snapshot_text = (
                f"STEPS 0-{prefix_len - 1} SO FAR:\n\n"
                + pprint.pformat(running_records, indent=2, width=160, sort_dicts=False)
            )
        else:
            snapshot_text = "STEPS SO FAR:\nNo steps executed yet."

        task.task_messages = [
            self._render_task_banner(task),
            {"role": "assistant", "content": snapshot_text},
            {
                "role": "user",
                "content": (
                    "Produce the NEXT BEST single tool call for the current task. "
                    "Pick the return tool if the running plan above has completed all needed work. "
                    "Do NOT repeat a tool call or redo work already available above or in cache — "
                    "reuse its result_ref, cache, or constant placeholder instead of recomputing or re-deriving the value."
                    + (
                        " Some steps above are marked FAILED — they have no result_ref; do not reference one."
                        if any(r.get("status") == "FAILED" for r in running_records)
                        else ""
                    )
                    + " Output exactly one JSON object with keys {step, tool, args, duration, description}."
                    + f" duration must be an int from 0 to {max_duration}."
                ),
            },
        ]

        return task.task_messages

    def _initialize_task(
        self,
        *,
        turns: list[AgentRecord],
        prompt: str,
        inputs: dict,
    ) -> ReActTask:
        """
        Initialize a ReActTask for a single ReAct invocation.

        Pre-allocates the fixed-size running blackboard and stamps the
        orchestrator system-prompt name. No LLM call here — step planning
        is deferred to ``think()``. No async override needed: unlike
        ``PlanActAgent``, there is no blocking I/O in this hook to bridge
        natively.
        """
        valid_cache_indices, failed_cache_indices = self._compute_cache_index_sets(turns)

        running_blackboard = [BlackboardSlot(step=i) for i in range(self._tool_calls_limit + 1)]

        return ReActTask(
            turns=turns,
            inputs=inputs,
            user_prompt=prompt,
            system_prompt_name="reason_then_act",
            running_blackboard=running_blackboard,
            executed_steps=set(),
            prepared_steps=[],
            tool_calls_used=0,
            valid_cache_indices=valid_cache_indices,
            failed_cache_indices=failed_cache_indices,
            next_step_index=0,
            step_meta=[ReActStepMeta() for _ in running_blackboard],
        )

    def _generate_next_step(self, *, task: ReActTask) -> tuple[BlackboardSlot, int, str]:
        """
        Generate and validate one ReAct tool step, via the shared
        ``ToolAgent._run_generation_retry_loop``.

        Observable counters are NOT decremented here — only when a step
        commits in ``prepare()``.

        Returns
        -------
        tuple[BlackboardSlot, int, str]
            Validated slot, duration, and description.
        """
        expected_step = task.next_step_index
        max_duration = max(0, self._tool_calls_limit - expected_step)
        return self._run_generation_retry_loop(
            task=task,
            validate=lambda parsed: self._process_next_step_output(
                parsed=parsed,
                expected_step=expected_step,
                cache_blackboard=self._blackboard,
                max_duration=max_duration,
                valid_cache_indices=task.valid_cache_indices,
                failed_cache_indices=task.failed_cache_indices,
            ),
            json_error_template=(
                "Your output could not be parsed as valid JSON.\n\n"
                "Decoder error: {exc}\n\n"
                "Produce a correctly formatted JSON object."
            ),
            spec_error_template=(
                "The step you produced contains an error.\n\n"
                "Error: {feedback}\n\n"
                "Reflect on this and produce a corrected step."
            ),
        )

    async def _agenerate_next_step(self, *, task: ReActTask) -> tuple[BlackboardSlot, int, str]:
        """Async mirror of ``_generate_next_step``, via
        ``ToolAgent._arun_generation_retry_loop``."""
        expected_step = task.next_step_index
        max_duration = max(0, self._tool_calls_limit - expected_step)
        return await self._arun_generation_retry_loop(
            task=task,
            validate=lambda parsed: self._process_next_step_output(
                parsed=parsed,
                expected_step=expected_step,
                cache_blackboard=self._blackboard,
                max_duration=max_duration,
                valid_cache_indices=task.valid_cache_indices,
                failed_cache_indices=task.failed_cache_indices,
            ),
            json_error_template=(
                "Your output could not be parsed as valid JSON.\n\n"
                "Decoder error: {exc}\n\n"
                "Produce a correctly formatted JSON object."
            ),
            spec_error_template=(
                "The step you produced contains an error.\n\n"
                "Error: {feedback}\n\n"
                "Reflect on this and produce a corrected step."
            ),
        )

    def think(self, task: ReActTask) -> ReActTask:
        """
        Generate this round's next step via the LLM, without applying it.

        ``_validate_react_prepare_state`` runs first — a precondition on
        the cursor/step_meta bookkeeping before deciding what to request,
        not part of this pass's dead-guard drops. No
        ``if task.prepared_steps: raise`` re-entry guard here — dead by the
        same construction as base ``ToolAgent.act()``'s dropped guard (1c):
        ``act()`` always leaves ``task.prepared_steps`` empty by the time
        the next round's ``think()`` runs.

        The generated ``(slot, duration, description)`` is stashed on
        ``task.generated_step`` for ``prepare()`` to consume next; applying
        it (resolve placeholders, cascade-check, stamp into
        ``running_blackboard``) is deliberately not done here.
        """
        self._validate_react_prepare_state(task)
        task.generated_step = self._generate_next_step(task=task)
        return task

    async def async_think(self, task: ReActTask) -> ReActTask:
        """Async mirror of ``think``: uses ``_agenerate_next_step`` so the
        per-step LLM call goes through ``async_invoke`` rather than a
        worker thread."""
        self._validate_react_prepare_state(task)
        task.generated_step = await self._agenerate_next_step(task=task)
        return task
