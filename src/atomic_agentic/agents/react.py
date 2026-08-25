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
window is open, and one-sentence reasons). The LLM responds with a single
JSON step object specifying the next tool to call, its arguments, an
observability duration, and a human-readable reason.

Observability Window
--------------------
Each step may declare a ``duration`` field (integer >= 0). While ``duration > 0``
the step's raw result appears in subsequent snapshots; once exhausted the result
is visible only as a ``|STEP.N|`` placeholder. This limits context growth while
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
import json
import pprint

from .toolagent import ToolAgent
from .prompts import ORCHESTRATOR_PROMPT
from ..constants.agents import (
    STEP_FIELD,
    REASON_FIELD,
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
from ..utils.agents import extract_dependencies, format_generation_issues

# --------------------------------------------------------------------------- #
# ReAct Agent
# --------------------------------------------------------------------------- #
class ReActAgent(ToolAgent):
    """
    Iterative agent with reactive step-by-step planning (ReAct-style architecture).

    **Design**: ReActAgent implements dynamic iteration: one step is emitted per LLM
    turn from a compact running-plan snapshot. Result references are always visible
    by placeholder; raw result previews are shown only while their observability
    duration remains active. Generated step reasons preserve semantic intent
    across turns without exposing raw results.

    1. **Initialization** (``_initialize_task``)
       - Pre-allocates a fixed-size running blackboard: ``tool_calls_limit + 1`` slots
         to accommodate non-return tool calls plus one return call.
       - Requires ``tool_calls_limit`` to be a concrete integer.
       - Initializes ReAct-specific per-step metadata:
         observability counters (generated step reasons now live on
         ``BlackboardSlot.reason`` directly, not per-step metadata).

    2. **Decision** (``think()`` — one step per round)
       - Builds a fresh temporary message list from static base messages.
       - Appends a compact running-plan snapshot containing executed steps,
         reasons, unresolved args, result_ref placeholders, and any currently
         observable_result values.
       - Asks the LLM for the next single step; validates it end-to-end.
       - Stashes the validated ``(slot, duration)`` onto
         ``task.generated_step`` for ``prepare()`` to apply next round.

    3. **Preparation** (``prepare()``)
       - Cascade-checks dependencies; resolves placeholders into concrete tool args.
       - Stores the prepared slot in ``running_blackboard[step_index]``.
       - Stores the observability duration in the ReAct task's ``step_meta``.
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
    - Interpretable: reasons, placeholders, and optional observations provide
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

    def _validate_generation_output(self, raw_output: str, *, task: ReActTask) -> tuple[BlackboardSlot, int] | str:
        """
        Concrete implementation of ``ToolAgent._validate_generation_output``
        for ReAct. Owns JSON decoding and the entire per-step validation
        pipeline; supersedes the old ``_process_next_step_output``.

        Two tiers, both fully accumulating (no short-circuit within a
        tier): tier 1 (structural/type) checks everything needed to build a
        candidate slot; if anything fails, returns immediately without
        attempting tier 2. Tier 2 (semantic/cross-referential) checks
        everything else. ReAct validates exactly one step per call, so this
        two-tier accumulation is this family's complete realization of
        "report every problem in one retry turn" — there is no separate
        multi-step loop the way PlanAct has.

        Budget enforcement (tier 2, step 12b below) is a real boundary
        check, not a dead guard: unlike ``PlanActAgent`` (whose budget is
        validated once against the *entire* plan before anything
        executes), ``ReActAgent`` generates one step at a time, so this is
        the only point that can catch "the model didn't terminate by the
        last available slot" before that step is ever prepared/executed.

        Steps
        -----
        1. Attempt ``self._extract_from_json_string(raw_output)``; on
           ``json.JSONDecodeError``, return the single-item decode-error
           feedback.
        2. Compute ``expected_step``/``max_duration`` from ``task``/``self``.
        3. Reject non-Mapping parsed values (single-item feedback).
        4. Build ``raw_payload``; tier-1 issue list starts empty.
        5. Unsupported-key check.
        6. Delegate to ``_validate_tool_step_dict``; extend tier-1 issues on
           a list return. Its own required-fields check (against
           ``REQUIRED_REACT_FIELDS``, which includes ``DURATION_FIELD``/
           ``REASON_FIELD``) is the sole source of a missing-duration/
           missing-reason issue — no separate standalone presence check
           here, to avoid reporting the same absence twice.
        7. If ``step_dict`` was structurally valid and ``DURATION_FIELD``
           was present: extract and range-validate ``duration``.
        8. If ``step_dict`` was structurally valid and ``REASON_FIELD`` was
           present: extract, type-check, strip, and non-empty-check
           ``reason``.
        9. If tier-1 issues: return the joined feedback.
        10. Convert to ``BlackboardSlot`` via ``_tool_step_dict_to_slot``;
            set ``slot.reason = reason``.
        11. Tier 2 — fresh issue list, every check independent: tool
            registered; budget (last available slot must be the return
            tool); return-tool duration == 0; cache refs (three
            categories); step dependencies prior-only.
        12. If tier-2 issues: return the joined feedback.
        13. Return ``(slot, duration)``.
        """
        # 1. JSON decode.
        try:
            parsed = self._extract_from_json_string(raw_output)
        except json.JSONDecodeError as exc:
            return format_generation_issues(
                [f"Your output could not be parsed as valid JSON. Decoder error: {exc}"]
            )

        # 2. Round-local context.
        expected_step = task.next_step_index
        max_duration = max(0, self._tool_calls_limit - expected_step)

        # 3. Structural check — LLM must return a JSON object, not an array or scalar.
        if not isinstance(parsed, Mapping):
            return format_generation_issues(
                [f"next step output must be a JSON object; got {type(parsed).__name__!r}."]
            )

        raw_payload = dict(parsed)
        issues: list[str] = []

        # 5. Field-set check.
        extra = set(raw_payload) - REACT_FIELDS
        if extra:
            issues.append(f"next step contains unsupported keys: {sorted(extra)!r}.")

        # 6. Core step-dict validation (tool name, args shape, await, etc.).
        # Its own required-fields check already covers a missing duration/
        # reason -- no separate standalone presence check here, to avoid
        # reporting the same absence twice.
        step_payload = self._validate_tool_step_dict(
            raw_payload,
            expected_step=expected_step,
            allowed_fields=REACT_FIELDS,
            required_fields=REQUIRED_REACT_FIELDS,
            context="next step",
        )
        structurally_valid = not isinstance(step_payload, list)
        if not structurally_valid:
            issues.extend(step_payload)

        duration: int | None = None
        reason: str | None = None

        # 7. Duration extraction and range check -- only when the shape checks
        # above didn't already fail, and the key is actually present.
        if structurally_valid and DURATION_FIELD in raw_payload:
            duration = step_payload.pop(DURATION_FIELD)
            if type(duration) is not int or duration < 0 or duration > max_duration:
                issues.append(
                    f"next step {DURATION_FIELD!r} must be an int in "
                    f"[0, {max_duration}] for expected_step={expected_step}; got {duration!r}."
                )

        # 8. Reason extraction, type check, and normalisation -- same guard.
        if structurally_valid and REASON_FIELD in raw_payload:
            reason = step_payload.pop(REASON_FIELD)
            if type(reason) is not str:
                issues.append(f"next step {REASON_FIELD!r} must be a string; got {type(reason).__name__!r}.")
            else:
                reason = reason.strip()
                if not reason:
                    issues.append(f"next step {REASON_FIELD!r} cannot be empty.")

        # 9. Tier 1 stops here -- tier 2 assumes a structurally valid,
        # fully-extracted (step_dict, duration, reason).
        if issues:
            return format_generation_issues(issues)

        # 10. Convert to BlackboardSlot -- engine contract; let ToolAgentError propagate.
        slot = self._tool_step_dict_to_slot(
            step_payload,
            step=expected_step,
            allowed_fields=BASE_STEP_FIELDS,
            context="next step",
        )
        slot.reason = reason

        # 11. Tier 2 -- semantic/cross-referential checks, all independent.
        issues = []
        cache_blackboard = self._blackboard
        valid_cache_indices = task.valid_cache_indices
        failed_cache_indices = task.failed_cache_indices

        if not self.has_tool(slot.tool):
            issues.append(f"next step references unknown tool {slot.tool!r}.")

        if expected_step == self._tool_calls_limit and slot.tool != RETURN_TOOL_FULL_NAME:
            issues.append(
                f"step {expected_step} is the last available slot under "
                f"tool_calls_limit={self._tool_calls_limit}; it must be the return tool."
            )

        if slot.tool == RETURN_TOOL_FULL_NAME and duration != 0:
            issues.append(f"return tool must use {DURATION_FIELD!r} 0; got {duration!r}.")

        cache_len = len(cache_blackboard)
        cache_refs = extract_dependencies(slot.args, placeholder_pattern=self.CACHE_REF_PATTERN)

        out_of_range = [idx for idx in cache_refs if idx < 0 or idx >= cache_len]
        if out_of_range:
            issues.append(
                f"next step references cache indices that do not exist: "
                f"{sorted(set(out_of_range))!r} (cache has {cache_len} entries)."
            )

        failed_in_conv = [idx for idx in cache_refs if idx in failed_cache_indices]
        if failed_in_conv:
            details = "; ".join(
                f"entry {idx} ({cache_blackboard[idx].tool}): {cache_blackboard[idx].error}"
                for idx in sorted(set(failed_in_conv))
            )
            issues.append(
                f"next step references cache entries that failed in this conversation "
                f"and cannot be used: {details}."
            )

        out_of_conv = [
            idx for idx in cache_refs
            if 0 <= idx < cache_len
            and idx not in valid_cache_indices
            and idx not in failed_cache_indices
        ]
        if out_of_conv:
            issues.append(
                f"next step references cache indices not part of this conversation: "
                f"{sorted(set(out_of_conv))!r}."
            )

        bad_step_deps = [
            dep for dep in slot.step_dependencies
            if dep < 0 or dep >= expected_step
        ]
        if bad_step_deps:
            issues.append(
                f"next step has illegal deps "
                f"{sorted(set(bad_step_deps))!r}; deps must be < {expected_step}."
            )

        # 12. Tier 2 result.
        if issues:
            return format_generation_issues(issues)

        # 13. Success.
        return slot, duration

    def prepare(self, task: ReActTask) -> ReActTask:
        """
        Apply this round's already-generated step: cascade-check, resolve
        placeholders, stamp into the running blackboard, advance the
        cursor.

        Unpacks ``task.generated_step`` (set by ``think()``/
        ``async_think()``) and resets it to ``NO_VAL``. Duration range,
        return-duration-zero, reason shape, and ``PLANNED`` status are
        already guaranteed by ``_validate_generation_output``'s own success
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
        generated_slot, observe_duration = task.generated_step
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
        slot.reason = generated_slot.reason

        # Cascade-fail: when fail_fast=False, propagate failures through arg dependencies.
        if not self._fail_fast:
            board = task.running_blackboard
            if self._check_cascade_failure(slot, board):
                task.next_step_index = prefix_len + 1
                task.task_messages.clear()
                return task  # prepared_steps stays []; act() will no-op this round

        # Resolve placeholders after stamping the planned slot into the running task.
        slot.resolved_args = self._resolve_placeholders(slot.args, task=task)
        slot.status = BlackboardSlot.PREPARED

        # Write per-slot metadata.
        task.step_meta[prefix_len].observable = observe_duration

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
           section (reasons, result_ref usage, observable_result
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
                    REASON_FIELD: slot.reason,
                    TOOL_FIELD: slot.tool,
                    ARGS_FIELD: slot.args,
                    "result_ref": f"|STEP.{idx}|",
                    "run_id": slot.result.run_id,
                }
                if task.step_meta[idx].observable > 0:
                    record["observable_result"] = self._preview_blackboard_result(slot.result.result)
                running_records.append(record)

            elif slot.is_failed():
                running_records.append({
                    STEP_FIELD: slot.step,
                    REASON_FIELD: slot.reason,
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
                    + " Output exactly one JSON object with keys {step, tool, args, duration, reason}."
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

    def _generate_next_step(self, *, task: ReActTask) -> tuple[BlackboardSlot, int]:
        """
        Generate and validate one ReAct tool step, via the shared
        ``ToolAgent._run_generation_retry_loop``.

        Observable counters are NOT decremented here — only when a step
        commits in ``prepare()``.

        Returns
        -------
        tuple[BlackboardSlot, int]
            Validated slot and duration. The step's reason is already set
            on the slot itself.
        """
        return self._run_generation_retry_loop(task=task)

    async def _agenerate_next_step(self, *, task: ReActTask) -> tuple[BlackboardSlot, int]:
        """Async mirror of ``_generate_next_step``, via
        ``ToolAgent._arun_generation_retry_loop``."""
        return await self._arun_generation_retry_loop(task=task)

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

        The generated ``(slot, duration)`` is stashed on
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
