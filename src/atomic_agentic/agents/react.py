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
The base ``ToolAgent`` template-method loop calls ``_prepare_next_batch`` once
per iteration to generate, validate, and stamp a single step; the loop executes
it and repeats until the return tool is emitted.

Contrast
--------
For one-shot full-plan generation with concurrent batches see
``agents/planact.py`` (``PlanActAgent``). For the shared iteration loop,
blackboard management, and tool registry see ``agents/toolagent.py``
(``ToolAgent``).
"""

from __future__ import annotations
import json
from typing import Any, Callable, Mapping, Optional
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
from ..engines.LLMEngines import LLMEngine
from ..exceptions import ToolAgentError
from ..models.agents import ReActRunState, ReActStepMeta
from ..models.agents import BlackboardSlot
from ..models.agents.records import LLMRecord
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

    1. **Initialization** (``_initialize_run_state``)
       - Pre-allocates a fixed-size running blackboard: ``tool_calls_limit + 1`` slots
         to accommodate non-return tool calls plus one return call.
       - Requires ``tool_calls_limit`` to be a concrete integer.
       - Snapshots cached blackboard entries only when ``context_enabled=True``.
       - Initializes ReAct-specific per-step metadata:
         observability counters and generated step descriptions.

    2. **Preparation** (``_prepare_next_batch`` – single step per turn)
       - Builds a fresh temporary message list from static base messages.
       - Appends a compact running-plan snapshot containing executed steps,
         descriptions, unresolved args, result_ref placeholders, and any currently
         observable_result values.
       - Asks the LLM for the next single step.
       - Step generation returns a planned slot, observability duration, and
         one-sentence description.
       - Validates step index matches the expected cursor position.
       - Validates placeholder dependencies are prior-only.
       - Resolves placeholders into concrete tool args.
       - Stores the prepared slot in ``running_blackboard[step_index]``.
       - Stores duration and description in ReAct run state.
       - Sets ``prepared_steps = [step_index]``.

    3. **Execution**
       - Base loop executes the single prepared step.
       - Result is stored; loop returns to preparation for the next step.

    4. **Termination**
       - When the return tool is emitted and executed, loop exits.
       - Running blackboard is persisted if ``context_enabled=True``.

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
        filter_extraneous_inputs: Optional[bool] = None,
        *,
        context_enabled: bool = False,
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
    ) -> None:
        """
        Initialize a ReActAgent.

        ``tool_calls_limit`` defaults to ``25`` and must be a concrete integer
        >= 0 — ``None`` is not accepted because the running blackboard is
        pre-allocated to ``tool_calls_limit + 1`` slots at initialization.
        ``tool_instructions`` and ``prompt_key`` are not exposed: they are
        hard-wired to the built-in orchestrator prompt and the
        ``"reason_then_act"`` key respectively.
        """
        super().__init__(
            name=name,
            namespace=namespace,
            description=description,
            llm_engine=llm_engine,
            tool_instructions=ORCHESTRATOR_PROMPT,
            filter_extraneous_inputs=filter_extraneous_inputs,
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
            prompt_key="reason_then_act",
            records_window=records_window,
        )

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
    def _validate_react_prepare_state(self, state: ReActRunState) -> None:
        """
        Validate cursor bounds, prior-step processing, and step_meta length.
        """
        prefix_len = state.next_step_index
        if type(prefix_len) is not int or prefix_len < 0:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: next_step_index must be an "
                f"int >= 0; got {prefix_len!r}."
            )
        if prefix_len >= len(state.running_blackboard):
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: next_step_index exceeds run "
                f"blackboard capacity ({prefix_len} >= {len(state.running_blackboard)})."
            )
        if prefix_len > 0:
            prev = state.running_blackboard[prefix_len - 1]
            # With fail_fast=False a previous step may be FAILED rather than EXECUTED;
            # both count as "processed" and allow generation of the next step.
            prev_processed = prev.is_executed() or (not self._fail_fast and prev.is_failed())
            if not prev_processed:
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: step {prefix_len - 1} was "
                    f"not executed before the next prepare call "
                    f"(status={prev.status!r})."
                )
        if len(state.step_meta) != len(state.running_blackboard):
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: step_meta length must match "
                f"running_blackboard length "
                f"({len(state.step_meta)} != {len(state.running_blackboard)})."
            )

    def _build_react_messages(
        self,
        state: ReActRunState,
        prefix_len: int,
        *,
        max_duration: int,
    ) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
        """
        Build the working message list and delta for one ReAct step generation.

        ``max_duration`` is pre-computed by the caller and injected here so the
        prompt and the validator in ``_generate_next_step`` share a single value.

        EXECUTED steps are rendered with ``result_ref``, ``run_id``, and optionally
        ``observable_result``. FAILED steps are rendered with ``status="FAILED"`` and
        ``error``; they carry no ``result_ref`` (so the LLM cannot attempt to reference
        a non-existent result).

        Returns ``(working_messages, delta)``.
        """
        working_messages: list[dict[str, str]] = [dict(m) for m in state.messages]

        running_records: list[dict[str, Any]] = []
        for idx in range(prefix_len):
            slot = state.running_blackboard[idx]

            if slot.is_executed():
                record: dict[str, Any] = {
                    STEP_FIELD: slot.step,
                    DESCRIPTION_FIELD: state.step_meta[idx].description,
                    TOOL_FIELD: slot.tool,
                    ARGS_FIELD: slot.args,
                    "result_ref": f"<<__s{idx}__>>",
                    "run_id": slot.result.run_id,
                }
                if state.step_meta[idx].observable > 0:
                    record["observable_result"] = self._preview_blackboard_result(slot.result.result)
                running_records.append(record)

            elif slot.is_failed():
                running_records.append({
                    STEP_FIELD: slot.step,
                    DESCRIPTION_FIELD: state.step_meta[idx].description,
                    TOOL_FIELD: slot.tool,
                    ARGS_FIELD: slot.args,
                    "status": "FAILED",
                    "error": str(slot.error),
                })
            # Empty/PLANNED slots are not yet part of the running plan; skip.

        if running_records:
            running_text = (
                f"RUNNING PLAN STEPS 0-{prefix_len - 1} SO FAR:\n"
                "Steps may be EXECUTED (result available via result_ref) or FAILED "
                "(error shown; do not reference result_ref for failed steps).\n"
                "Use descriptions to understand what each step was intended to do.\n"
                "Use result_ref placeholders when a new arg needs a prior executed step's value.\n"
                "observable_result fields are for OBSERVATION ONLY: use them only to choose "
                "the next tool or branch.\n"
                "Do not copy observable_result values into new args.\n\n"
                + pprint.pformat(running_records, indent=2, width=160, sort_dicts=False)
            )
        else:
            running_text = (
                "RUNNING PLAN STEPS SO FAR:\n"
                "No steps executed yet.\n"
                "When steps execute, their results will be available by result_ref "
                "placeholders like <<__s0__>>."
            )

        working_messages.append({"role": "assistant", "content": running_text})
        working_messages.append(
            {
                "role": "user",
                "content": (
                    "Produce the NEXT BEST single tool call for the current task. "
                    "Pick the return tool if the running plan has completed all needed work. "
                    "Output exactly one JSON object with keys {step, tool, args, duration, description}. "
                    "Preserve symbolic dataflow with quoted placeholders; do not copy observable_result values into args. "
                    f"For this output step, duration must be an int from 0 to {max_duration}."
                ),
            }
        )

        delta = [state.messages[-1], working_messages[-2], working_messages[-1]]
        return working_messages, delta

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

        Receives the already-extracted ``parsed`` value. LLMRecord construction has
        moved to ``_generate_next_step``. All LLM-facing validation failures return
        a plain feedback string (no class/name prefix); engine-contract violations
        raise ``ToolAgentError``.

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
        9. Require return-tool duration == 0.
        10. Validate cache refs — three categories: out-of-range, failed-in-conv
            (with tool+error detail), out-of-conv.
        11. Validate step dependencies are prior-only.
        12. Return ``(slot, duration, description)``.
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

        # 9. Return-tool must use duration 0.
        if slot.tool == RETURN_TOOL_FULL_NAME and duration != 0:
            return f"return tool must use {DURATION_FIELD!r} 0; got {duration!r}."

        # 10. Cache reference validation — three categories.
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

        # 11. Step dependency validation.
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

    def _apply_react_step_result(
        self,
        state: ReActRunState,
        prefix_len: int,
        generated_slot: BlackboardSlot,
        observe_duration: int,
        description: str,
        llm_records: list[LLMRecord],
        *,
        max_duration: int,
    ) -> ReActRunState:
        """
        Apply one validated ReAct step generation result to the run state.

        Validates the returned tuple fields, decrements observable counters,
        fills the preallocated running-blackboard slot, then either cascade-fails
        the slot (``fail_fast=False`` only) or resolves placeholders and marks it
        prepared. Advances the cursor and writes ``step_meta``.

        ``max_duration`` is pre-computed by ``_prepare_next_batch`` /
        ``_aprepare_next_batch`` and passed in; this method does not recompute it.

        **Cascade path** (``fail_fast=False``): if any ``step_dependencies`` entry
        is FAILED in the running blackboard, the return tool raises immediately;
        non-return slots are marked FAILED and the method returns early with
        ``prepared_steps`` left empty — the ``_invoke`` loop will skip execution
        and continue to the next generation turn.
        """
        state.llm_records.extend(llm_records)

        if type(observe_duration) is not int or observe_duration < 0 or observe_duration > max_duration:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: observe_duration must be an int in "
                f"[0, {max_duration}]; got {observe_duration!r}."
            )

        if generated_slot.tool == RETURN_TOOL_FULL_NAME and observe_duration != 0:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: return tool must use duration 0; "
                f"got {observe_duration!r}."
            )

        if type(description) is not str:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: description must be a string; "
                f"got {type(description).__name__!r}."
            )

        description = description.strip()

        if not generated_slot.is_planned():
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: generated slot must be planned; "
                f"got status={generated_slot.status!r}."
            )

        # A successful generation turn consumed any raw results that were visible.
        for meta in state.step_meta:
            if meta.observable > 0:
                meta.observable -= 1

        # Fill the preallocated running-blackboard slot.
        slot = state.running_blackboard[prefix_len]
        if slot.step != prefix_len:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: running slot step mismatch at index {prefix_len}: "
                f"slot.step={slot.step}."
            )

        if not slot.is_empty():
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: attempted to prepare into non-empty slot {prefix_len}."
            )

        slot.tool = generated_slot.tool
        slot.args = generated_slot.args
        slot.result = NO_VAL
        slot.error = NO_VAL
        slot.step_dependencies = generated_slot.step_dependencies
        slot.await_step = generated_slot.await_step

        # Cascade-fail: when fail_fast=False, propagate failures through arg dependencies.
        # Use extract_dependencies(slot.args) for the same reason as PlanActAgent: only
        # steps actually referenced in args need to resolve successfully.
        if not self._fail_fast:
            board = state.running_blackboard
            failed_arg_deps = sorted(
                d
                for d in extract_dependencies(slot.args, placeholder_pattern=self.STEP_REF_PATTERN)
                if board[d].is_failed()
            )
            if failed_arg_deps:
                dep_str = ", ".join(str(d) for d in failed_arg_deps)
                if slot.tool == RETURN_TOOL_FULL_NAME:
                    raise ToolAgentError(
                        f"{type(self).__name__}.{self.name}: return step cannot execute; "
                        f"dependency step(s) {dep_str} failed."
                    )
                slot.error = ToolAgentError(
                    f"{type(self).__name__}.{self.name}: step {prefix_len} skipped — "
                    f"dependency step(s) {dep_str} failed."
                )
                slot.status = BlackboardSlot.FAILED
                state.step_meta[prefix_len].description = description
                state.next_step_index = prefix_len + 1
                return state  # prepared_steps stays []; _invoke loop will skip execute and continue

        # Resolve placeholders after stamping the planned slot into the running state.
        slot.resolved_args = self._resolve_placeholders(slot.args, state=state)
        slot.status = BlackboardSlot.PREPARED

        # Write per-slot metadata.
        state.step_meta[prefix_len].observable = observe_duration
        state.step_meta[prefix_len].description = description

        state.prepared_steps = [prefix_len]
        state.next_step_index = prefix_len + 1

        return state

    # ------------------------------------------------------------------ #
    # Tool-Agent Hooks
    # ------------------------------------------------------------------ #
    def _initialize_run_state(
        self,
        *,
        messages: list[dict[str, str]],
        valid_cache_indices: frozenset[int],
        failed_cache_indices: frozenset[int],
    ) -> ReActRunState:
        """
        Initialize run state for a single ReAct invocation.

        Satisfies the ``ToolAgent._initialize_run_state`` abstract hook. Unlike
        ``PlanActAgent``, ReAct performs no LLM call at initialization — the
        running blackboard is pre-allocated and step planning is deferred to
        ``_prepare_next_batch`` / ``_aprepare_next_batch``.

        Steps
        -----
        1. Validate ``messages`` is non-empty.
        2. Snapshot the persisted blackboard as a copy (empty list when
           ``context_enabled=False``); store ``valid_cache_indices`` and
           ``failed_cache_indices`` on state.
        3. Copy ``messages`` into a mutable working list.
        4. Pre-allocate a fixed-size ``running_blackboard`` of
           ``tool_calls_limit + 1`` slots — one slot per allowed non-return
           call plus one slot for the mandatory return call.
        5. Initialize ``step_meta`` as a list of ``ReActStepMeta()`` instances
           of the same length as ``running_blackboard``.
        6. Construct and return ``ReActRunState`` with empty ``llm_records``
           (records are appended by each ``_prepare_next_batch`` call).

        Returns
        -------
        ReActRunState

        Raises
        ------
        ToolAgentError
            If ``messages`` is empty.
        """
        if not messages:
            raise ToolAgentError(f"{type(self).__name__}.{self.name}: messages must be non-empty.")

        cache_blackboard = (
            [slot.copy() for slot in self._blackboard] if self.context_enabled else []
        )

        working_messages = [dict(m) for m in messages]

        # Preallocate fixed-size run blackboard: non-return calls + 1 return call.
        running_blackboard = [BlackboardSlot(step=i) for i in range(self._tool_calls_limit + 1)]

        return ReActRunState(
            messages=working_messages,
            cache_blackboard=cache_blackboard,
            running_blackboard=running_blackboard,
            executed_steps=set(),
            prepared_steps=[],
            tool_calls_used=0,
            llm_records=[],
            valid_cache_indices=valid_cache_indices,
            failed_cache_indices=failed_cache_indices,
            is_done=False,
            return_value=NO_VAL,
            next_step_index=0,
            step_meta=[ReActStepMeta() for _ in running_blackboard],
        )

    def _generate_next_step(
        self,
        *,
        messages: list[dict[str, str]],
        cache_blackboard: list[BlackboardSlot],
        expected_step: int,
        delta: list[dict[str, str]],
        retries_used: int,
        max_duration: int,
        valid_cache_indices: frozenset[int],
        failed_cache_indices: frozenset[int],
    ) -> tuple[BlackboardSlot, int, str, list[LLMRecord], int]:
        """
        Generate and validate one ReAct tool step, with a bounded retry loop.

        Mirrors ``_generate_plan`` in structure. LLMRecord construction lives here;
        ``_process_next_step_output`` receives the already-extracted ``parsed`` value.
        The retry loop catches ``json.JSONDecodeError`` (JSON path) and string returns
        from ``_process_next_step_output`` (spec-validation path), injecting structured
        feedback into the working message thread on each failed attempt.

        ``retries_used`` carries the shared per-run budget consumed by prior step
        generations. The returned int is the updated value; the caller writes it back
        to ``state.retries_used``.

        LLMRecord messages convention:
        - First attempt: ``delta`` (task user + snapshot assistant + step-request user).
        - Retry attempt N: ``working_messages[-2:]`` (failed-output assistant + feedback user).

        Observable counters are NOT decremented here — only when a step commits in
        ``_apply_react_step_result``.

        Steps
        -----
        1. Contract guards: messages empty, cache_blackboard not a list,
           expected_step not a non-negative int → raise ToolAgentError.
        2. Copy ``messages`` to ``working_messages``; ``max_duration`` is pre-computed
           by the caller and passed in.
        3. Loop:
           a. LLM call.
           b. Construct LLMRecord (``delta`` on first attempt, ``working_messages[-2:]``
              on retries); append to ``llm_records``.
           c. JSON extraction — on ``json.JSONDecodeError``: check budget, inject
              feedback, increment ``retries_used``, continue.
           d. Spec validation via ``_process_next_step_output`` — on string return:
              check budget, inject re-serialised step + feedback, increment, continue.
           e. Success: return ``(slot, duration, description, llm_records, retries_used)``.

        Parameters
        ----------
        messages : list[dict[str, str]]
            Base LLM-facing messages. Copied locally; original unchanged.
        cache_blackboard : list[BlackboardSlot]
            Snapshot of persisted prior results for cache-ref validation.
        expected_step : int
            Authoritative plan-local step index.
        delta : list[dict[str, str]]
            Pre-computed new messages for the first attempt's LLMRecord.
        retries_used : int
            Shared budget counter on entry (retries consumed by prior steps).

        Returns
        -------
        tuple[BlackboardSlot, int, str, list[LLMRecord], int]
            Validated slot, duration, description, all LLMRecords for this call
            (one per attempt), and the updated ``retries_used``.
        """
        if not messages:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: messages must be non-empty."
            )

        if not isinstance(cache_blackboard, list):
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: cache_blackboard must be a list."
            )

        if type(expected_step) is not int or expected_step < 0:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: expected_step must be an int >= 0; "
                f"got {expected_step!r}."
            )

        working_messages: list[dict[str, str]] = list(messages)
        llm_records: list[LLMRecord] = []

        while True:
            engine_result = self._llm_engine.invoke({"messages": [dict(m) for m in working_messages]})
            raw_output: str = engine_result.result

            # First attempt uses the pre-computed delta; retries use the two injected feedback messages.
            record_messages = delta if not llm_records else list(working_messages[-2:])
            llm_records.append(LLMRecord(
                messages=record_messages,
                llm_result=engine_result,
                system_prompt_name=self._tool_prompt_key,
            ))

            # JSON extraction
            try:
                parsed = self._extract_from_json_string(raw_output)
            except json.JSONDecodeError as exc:
                feedback = (
                    f"Your output could not be parsed as valid JSON.\n\n"
                    f"Decoder error: {exc}\n\n"
                    f"The response you produced was:\n\n{raw_output}\n\n"
                    "Produce a correctly formatted JSON object."
                )
                if retries_used >= self._generation_retries:
                    raise ToolAgentError(
                        f"{type(self).__name__}.{self.name}: generation retry budget exhausted "
                        f"after {retries_used + 1} attempt(s). Last error is a JSONDecodeError: {exc}"
                    )
                working_messages.append({"role": "assistant", "content": raw_output})
                working_messages.append({"role": "user", "content": feedback})
                retries_used += 1
                continue

            # Spec validation
            result = self._process_next_step_output(
                parsed=parsed,
                expected_step=expected_step,
                cache_blackboard=cache_blackboard,
                max_duration=max_duration,
                valid_cache_indices=valid_cache_indices,
                failed_cache_indices=failed_cache_indices,
            )
            if isinstance(result, str):
                feedback = result
                if retries_used >= self._generation_retries:
                    raise ToolAgentError(
                        f"{type(self).__name__}.{self.name}: generation retry budget exhausted "
                        f"after {retries_used + 1} attempt(s). Last error: {feedback}"
                    )
                step_repr = json.dumps(parsed, indent=2)
                working_messages.append({"role": "assistant", "content": step_repr})
                working_messages.append({"role": "user", "content": (
                    f"The step you produced contains an error:\n\n"
                    f"{step_repr}\n\n"
                    f"Error: {feedback}\n\n"
                    "Reflect on this and produce a corrected step."
                )})
                retries_used += 1
                continue

            slot, duration, description = result
            return slot, duration, description, llm_records, retries_used

    async def _agenerate_next_step(
        self,
        *,
        messages: list[dict[str, str]],
        cache_blackboard: list[BlackboardSlot],
        expected_step: int,
        delta: list[dict[str, str]],
        retries_used: int,
        max_duration: int,
        valid_cache_indices: frozenset[int],
        failed_cache_indices: frozenset[int],
    ) -> tuple[BlackboardSlot, int, str, list[LLMRecord], int]:
        """
        Async mirror of ``_generate_next_step``: uses ``async_invoke`` for each LLM call.

        Full async loop — not delegated to ``asyncio.to_thread``. Retry logic,
        feedback injection, LLMRecord accumulation, and return type are identical
        to the sync version. The only difference is the LLM call:
        ``await self._llm_engine.async_invoke(...)``.
        """
        if not messages:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: messages must be non-empty."
            )

        if not isinstance(cache_blackboard, list):
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: cache_blackboard must be a list."
            )

        if type(expected_step) is not int or expected_step < 0:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: expected_step must be an int >= 0; "
                f"got {expected_step!r}."
            )

        working_messages: list[dict[str, str]] = list(messages)
        llm_records: list[LLMRecord] = []

        while True:
            engine_result = await self._llm_engine.async_invoke(
                {"messages": [dict(m) for m in working_messages]}
            )
            raw_output: str = engine_result.result

            record_messages = delta if not llm_records else list(working_messages[-2:])
            llm_records.append(LLMRecord(
                messages=record_messages,
                llm_result=engine_result,
                system_prompt_name=self._tool_prompt_key,
            ))

            try:
                parsed = self._extract_from_json_string(raw_output)
            except json.JSONDecodeError as exc:
                feedback = (
                    f"Your output could not be parsed as valid JSON.\n\n"
                    f"Decoder error: {exc}\n\n"
                    f"The response you produced was:\n\n{raw_output}\n\n"
                    "Produce a correctly formatted JSON object."
                )
                if retries_used >= self._generation_retries:
                    raise ToolAgentError(
                        f"{type(self).__name__}.{self.name}: generation retry budget exhausted "
                        f"after {retries_used + 1} attempt(s). Last error is a JSONDecodeError: {exc}"
                    )
                working_messages.append({"role": "assistant", "content": raw_output})
                working_messages.append({"role": "user", "content": feedback})
                retries_used += 1
                continue

            result = self._process_next_step_output(
                parsed=parsed,
                expected_step=expected_step,
                cache_blackboard=cache_blackboard,
                max_duration=max_duration,
                valid_cache_indices=valid_cache_indices,
                failed_cache_indices=failed_cache_indices,
            )
            if isinstance(result, str):
                feedback = result
                if retries_used >= self._generation_retries:
                    raise ToolAgentError(
                        f"{type(self).__name__}.{self.name}: generation retry budget exhausted "
                        f"after {retries_used + 1} attempt(s). Last error: {feedback}"
                    )
                step_repr = json.dumps(parsed, indent=2)
                working_messages.append({"role": "assistant", "content": step_repr})
                working_messages.append({"role": "user", "content": (
                    f"The step you produced contains an error:\n\n"
                    f"{step_repr}\n\n"
                    f"Error: {feedback}\n\n"
                    "Reflect on this and produce a corrected step."
                )})
                retries_used += 1
                continue

            slot, duration, description = result
            return slot, duration, description, llm_records, retries_used

    def _prepare_next_batch(self, state: ReActRunState) -> ReActRunState:
        """
        Prepare the next single-step batch via one LLM call.

        ReAct generates exactly one step per turn. A temporary LLM message list is
        assembled from the static base messages plus a running-plan snapshot; the LLM
        returns a single planned step that is validated, placeholder-resolved, and
        stamped into the preallocated running blackboard.

        The temporary running-plan messages do not persist between turns; state.messages
        remains the static base message list for this invoke.

        Execution
        ~~~~~~~~~
        1. **Guard**: ``prepared_steps`` must be empty.
        2. **Snapshot messages**: compute ``max_duration`` and call
           ``_build_react_messages`` to assemble the temporary message thread and
           ``delta`` for the LLMRecord.
        3. **Step generation**: call ``_generate_next_step``, which handles the LLM
           call, JSON extraction, spec validation, and generation retries. Returns one
           planned ``BlackboardSlot``, an observability duration, a step description,
           LLMRecords for this call, and the updated ``retries_used`` budget.
        4. **Commit**: delegate to ``_apply_react_step_result`` — decrement
           observability counters on prior steps, stamp the slot into the running
           blackboard, resolve placeholders, mark prepared, update ``step_meta``.
        5. **Cursor**: ``next_step_index`` advances by 1; ``prepared_steps`` is
           ``[prefix_len]``.

        Parameters
        ----------
        state : ReActRunState
            Current run state. ``prepared_steps`` must be empty; ``next_step_index``
            identifies the preallocated slot to fill.

        Returns
        -------
        ReActRunState
            Updated state with one prepared step and an incremented
            ``next_step_index``.

        Raises
        ------
        ToolAgentError
            If ``prepared_steps`` is non-empty, the cursor is out of bounds, or
            the generation retry budget is exhausted.
        """
        if state.prepared_steps:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: cannot prepare next batch while prepared_steps is non-empty."
            )

        prefix_len = state.next_step_index
        self._validate_react_prepare_state(state)
        max_duration = max(0, self._tool_calls_limit - prefix_len)
        working_messages, delta = self._build_react_messages(state, prefix_len, max_duration=max_duration)

        generated_slot, observe_duration, description, llm_records, new_retries_used = (
            self._generate_next_step(
                messages=working_messages,
                cache_blackboard=state.cache_blackboard,
                expected_step=prefix_len,
                delta=delta,
                retries_used=state.retries_used,
                max_duration=max_duration,
                valid_cache_indices=state.valid_cache_indices,
                failed_cache_indices=state.failed_cache_indices,
            )
        )
        state.retries_used = new_retries_used
        return self._apply_react_step_result(
            state, prefix_len, generated_slot, observe_duration, description, llm_records,
            max_duration=max_duration,
        )

    async def _aprepare_next_batch(self, state: ReActRunState) -> ReActRunState:
        """
        Async override: uses ``_agenerate_next_step`` so the per-step LLM call
        goes through ``async_invoke`` rather than a worker thread.
        """
        if state.prepared_steps:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: cannot prepare next batch while prepared_steps is non-empty."
            )

        prefix_len = state.next_step_index
        self._validate_react_prepare_state(state)
        max_duration = max(0, self._tool_calls_limit - prefix_len)
        working_messages, delta = self._build_react_messages(state, prefix_len, max_duration=max_duration)

        generated_slot, observe_duration, description, llm_records, new_retries_used = (
            await self._agenerate_next_step(
                messages=working_messages,
                cache_blackboard=state.cache_blackboard,
                expected_step=prefix_len,
                delta=delta,
                retries_used=state.retries_used,
                max_duration=max_duration,
                valid_cache_indices=state.valid_cache_indices,
                failed_cache_indices=state.failed_cache_indices,
            )
        )
        state.retries_used = new_retries_used
        return self._apply_react_step_result(
            state, prefix_len, generated_slot, observe_duration, description, llm_records,
            max_duration=max_duration,
        )
