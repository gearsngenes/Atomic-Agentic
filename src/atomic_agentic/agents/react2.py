"""
ReActAgent2: Iterative LLM Actor on the Agent2 think/prepare/execute Lifecycle

Reworks ``ReActAgent`` (``agents/react.py``) onto the ``ToolAgent2`` base:
each round is driven by a fresh LLM call that observes the current
running-plan snapshot and emits exactly one next tool call. Decoupled from
thinking entirely — ``prepare``/``execute`` run every round regardless of
``task.keep_thinking`` (``_permits_unbounded_thinking = True``), so while
thinking is active, each round performs both a thinking generation and a
step generation. A per-round previous-vs-current thoughts split
(``task.newest_thoughts``, stamped by an overridden ``think``/``async_think``)
keeps the step-generation prompt honest about which thoughts were produced
for the step currently being decided versus earlier steps — and correctly
empties out once thinking has concluded, rather than staying pinned to a
stale round.

See ``.claude/brainstorms/toolagent2-lifecycle.md`` and
``.claude/specs/planact2-react2.md`` for the full design record, including
why several of ``_apply_react_step_result``'s v1 validation checks were
dropped as provably dead code (already guaranteed by
``_process_next_step_output``'s own success path).

Observability Window
--------------------
Each step may declare a ``duration`` field (integer >= 0). While
``duration > 0`` the step's raw result appears in subsequent snapshots;
once exhausted the result is visible only as a ``<<__sN__>>`` placeholder.

Contrast
--------
For one-shot full-plan generation with concurrent batches see
``agents/planact2.py`` (``PlanActAgent2``). For the shared thinking/toolbox/
blackboard machinery see ``agents/toolagent2.py`` (``ToolAgent2``).
"""

from __future__ import annotations

import json
import pprint
from typing import Any, Callable, Literal, Mapping, Optional

from .toolagent2 import ToolAgent2
from .prompts import ORCHESTRATOR_PROMPT, REACT_THINKING_PROMPT
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
from ..models.agents.prompts import PromptConfig
from ..models.agents.tasks import ReActTask, ReActStepMeta
from ..models.agents import BlackboardSlot
from ..models.agents.records import AgentRecord, LLMRecord
from ..utils.agents import extract_dependencies

# --------------------------------------------------------------------------- #
# ReAct Agent2
# --------------------------------------------------------------------------- #
class ReActAgent2(ToolAgent2):
    """
    Iterative agent on the ``ToolAgent2`` lifecycle with reactive
    step-by-step planning (ReAct-style architecture).

    **Design**: one step is emitted per LLM turn from a compact running-plan
    snapshot. Result references are always visible by placeholder; raw
    result previews are shown only while their observability duration
    remains active. Generated step descriptions preserve semantic intent
    across turns without exposing raw results.

    1. **Thinking** (optional, shared ``Agent2`` machinery) — decoupled from
       preparation/execution entirely (``_permits_unbounded_thinking =
       True``). While active, every round performs both a thinking
       generation and a step generation, not one after the other.
    2. **Preparation** (``_prepare_next_batch`` — single step per turn)
       - Builds a fresh temporary message list from static base messages.
       - Appends a compact running-plan snapshot (executed steps,
         descriptions, unresolved args, result_ref placeholders, any
         currently observable_result values) plus the previous/current
         thoughts split.
       - Asks the LLM for the next single step; validates, resolves
         placeholders, stores the prepared slot, sets ``prepared_steps``.
    3. **Execution** (``execute()``, shared/final on ``ToolAgent2``) —
       executes the single prepared step; loop returns to preparation for
       the next step.
    4. **Termination** — when the return tool is emitted and executed,
       ``task.complete`` is set.

    Advantages
    ~~~~~~~~~~
    - Fully adaptive: each step can react to prior tool results.
    - Context-hygienic: running-plan messages are rebuilt fresh each turn.
    - Interpretable: descriptions, placeholders, and optional observations
      provide a compact execution trace.

    Limitations
    ~~~~~~~~~~~
    - Higher latency: one LLM call per step (two while thinking is active).
    - No concurrency: only one step executes per iteration.
    - Step quality depends on the model's ability to select the next best
      action.

    Parameters (construction)
    ~~~~~~~~~~~~~~~~~~~~~~~~
    ``tool_calls_limit`` (int, REQUIRED): must be a concrete integer >= 0.
    Determines pre-allocated blackboard size. Cannot be ``None``.
    """

    _THINK_PROMPT: PromptConfig = REACT_THINKING_PROMPT
    _permits_unbounded_thinking: bool = True

    def __init__(
        self,
        name: str,
        namespace: str,
        description: str,
        llm_engine: LLMEngine,
        filter_extraneous_inputs: Optional[bool] = None,
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
        assistant_response_source: Literal["raw", "final"] = "raw",
        thinking_rounds: int | None = 0,
        thoughts_per_round: int = 1,
        thoughts_window: int | None = 0,
    ) -> None:
        """
        Initialize a ReActAgent2.

        No ``*`` keyword-only separator, same departure from v1 as
        ``PlanActAgent2``. ``tool_calls_limit`` defaults to ``25`` and must
        be a concrete integer >= 0 — ``None`` is not accepted because the
        running blackboard is pre-allocated to ``tool_calls_limit + 1``
        slots at initialization. ``"reason_then_act"`` is the key under
        which the built-in orchestrator prompt is registered.
        ``thinking_rounds`` stays ``int | None = 0`` — thinking defaults to
        off even though ``_permits_unbounded_thinking=True`` legalizes
        ``None`` (unbounded) as a value.
        """
        super().__init__(
            name=name,
            namespace=namespace,
            description=description,
            llm_engine=llm_engine,
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
            records_window=records_window,
            assistant_response_source=assistant_response_source,
            thinking_rounds=thinking_rounds,
            thoughts_per_round=thoughts_per_round,
            thoughts_window=thoughts_window,
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
        """Reject None and non-integer values; ReActAgent2 pre-allocates by this count."""
        if type(value) is not int or value < 0:
            raise ToolAgentError("ReActAgent2 requires tool_calls_limit to be an int >= 0.")
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
        task: ReActTask,
        prefix_len: int,
        generated_slot: BlackboardSlot,
        observe_duration: int,
        description: str,
    ) -> ReActTask:
        """
        Apply one validated ReAct step generation result to the task.

        Duration range, return-duration-zero, description shape, and
        ``generated_slot``'s PLANNED status are already guaranteed by
        ``_process_next_step_output``'s own success path — not re-checked
        here (v1 re-checked all four; dropped as dead code, same cleanup as
        ``PlanActAgent2``'s dropped ``_finalize_planact_task`` validation).

        Decrements observable counters, fills the preallocated
        running-blackboard slot, then either cascade-fails the slot
        (``fail_fast=False`` only) or resolves placeholders and marks it
        prepared. Advances the cursor and writes ``step_meta``.

        **Cascade path** (``fail_fast=False``): if any ``step_dependencies``
        entry is FAILED in the running blackboard, the return tool raises
        immediately; non-return slots are marked FAILED and the method
        returns early with ``prepared_steps`` left empty — the loop will
        skip execution and continue to the next generation turn.

        ``task.task_messages`` and ``task.newest_thoughts`` are cleared
        unconditionally before returning — on both the cascade-fail early
        return and the normal path — since an LLM call happened this round
        either way and the next round rebuilds fresh.
        """
        # A successful generation turn consumed any raw results that were visible.
        for meta in task.step_meta:
            if meta.observable > 0:
                meta.observable -= 1

        # Fill the preallocated running-blackboard slot.
        slot = task.running_blackboard[prefix_len]
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
        if not self._fail_fast:
            board = task.running_blackboard
            if self._check_cascade_failure(slot, board):
                task.step_meta[prefix_len].description = description
                task.next_step_index = prefix_len + 1
                task.task_messages.clear()
                task.newest_thoughts = []
                return task  # prepared_steps stays []; execute() has nothing to run this round

        # Resolve placeholders after stamping the planned slot into the running task.
        slot.resolved_args = self._resolve_placeholders(slot.args, task=task)
        slot.status = BlackboardSlot.PREPARED

        # Write per-slot metadata.
        task.step_meta[prefix_len].observable = observe_duration
        task.step_meta[prefix_len].description = description

        task.prepared_steps = [prefix_len]
        task.next_step_index = prefix_len + 1

        task.task_messages.clear()
        task.newest_thoughts = []
        return task

    # ------------------------------------------------------------------ #
    # Thinking
    # ------------------------------------------------------------------ #
    def think(self, task: ReActTask) -> ReActTask:
        """Thin wrapper around ``Agent2.think``: stamps ``task.newest_thoughts``
        when a new thinking round was actually produced this call,
        independent of what ``task.keep_thinking`` becomes as a side
        effect of that same call — this is what lets the exact round
        thinking concludes still render as "current" for that round's
        step, rather than being swallowed into "previous"."""
        before = len(task.thoughts)
        task = super().think(task)
        if len(task.thoughts) > before:
            task.newest_thoughts = task.thoughts[-1]
        return task

    async def async_think(self, task: ReActTask) -> ReActTask:
        """Async mirror of ``think`` — same before/after length comparison
        around ``await super().async_think(task)``."""
        before = len(task.thoughts)
        task = await super().async_think(task)
        if len(task.thoughts) > before:
            task.newest_thoughts = task.thoughts[-1]
        return task

    # ------------------------------------------------------------------ #
    # Render pipeline
    # ------------------------------------------------------------------ #
    def _render_system_message(self, task: ReActTask) -> list[dict[str, str]]:
        """Dispatch ``"reason_then_act"`` locally; delegate everything else
        (i.e. ``"think"``) to ``Agent2``'s own implementation.

        The empty base context is sufficient — ``ToolAgent2._render_system_prompt``
        already injects ``TOOLS``/``CONSTANTS``/``TOOL_CALLS_LIMIT``
        automatically, no manual ``render_context`` construction needed here
        (a real simplification over v1's own ``render_task``).
        """
        if task.system_prompt_name != "reason_then_act":
            return super()._render_system_message(task)

        rendered = self._render_system_prompt(task, {})
        return [{"role": "system", "content": rendered}]

    def _render_task_messages(self, task: ReActTask) -> list[dict[str, str]]:
        """Build this phase's task messages, "think" and "reason_then_act" alike.

        Delegates the "think" branch to ``Agent2``'s shared
        ``_render_thinking_task_messages`` entirely. Owns only the
        "reason_then_act" branch: mirrors v1's own running-plan-snapshot
        convention (banner / self-labeled snapshot as an assistant turn /
        fixed instruction), now with a previous-vs-current thoughts split
        folded into the same assistant turn, driven by
        ``task.newest_thoughts``. Both segments -- and the whole thoughts
        block -- are omitted when empty, matching the "invisible when
        empty" convention used elsewhere in this design (always true when
        ``thinking_rounds=0``).

        EXECUTED steps are rendered with ``result_ref``, ``run_id``, and
        optionally ``observable_result``. FAILED steps are rendered with
        ``status="FAILED"`` and ``error``; they carry no ``result_ref`` (so
        the LLM cannot attempt to reference a non-existent result).
        """
        if task.task_messages:
            return task.task_messages

        if task.system_prompt_name == "think":
            task.task_messages = self._render_thinking_task_messages(task)
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

        # Previous-vs-current thoughts split.
        current_thoughts = task.newest_thoughts
        previous_thoughts = task.thoughts[:-1] if current_thoughts else task.thoughts

        thoughts_parts: list[str] = []
        if previous_thoughts:
            thoughts_parts.append(f"PREVIOUS THOUGHTS:\n{self._format_thoughts(previous_thoughts)}")
        if current_thoughts:
            thoughts_parts.append(f"CURRENT THOUGHTS:\n{self._format_thoughts([current_thoughts])}")
        thoughts_text = "\n\n".join(thoughts_parts)

        assistant_content = f"{thoughts_text}\n\n{snapshot_text}" if thoughts_text else snapshot_text

        task.task_messages = [
            {
                "role": "user",
                "content": f"===== CURRENT TASK =====\n{task.user_prompt}\n===== END TASK =====",
            },
            {"role": "assistant", "content": assistant_content},
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

    # ------------------------------------------------------------------ #
    # Step generation
    # ------------------------------------------------------------------ #
    def _generate_next_step(self, *, task: ReActTask) -> tuple[BlackboardSlot, int, str]:
        """
        Generate and validate one ReAct tool step, with a bounded retry loop.

        Mirrors ``_generate_plan`` in structure. ``_process_next_step_output``
        receives the already-extracted ``parsed`` value. The retry loop
        catches ``json.JSONDecodeError`` (JSON path) and string returns from
        ``_process_next_step_output`` (spec-validation path), injecting
        structured feedback as ``additional_messages`` for the next
        ``render_task`` call on each failed attempt.

        Observable counters are NOT decremented here — only when a step
        commits in ``_apply_react_step_result``.

        Steps
        -----
        1. ``expected_step = task.next_step_index``; ``max_duration`` derived
           from it.
        2. ``additional_messages`` starts empty.
        3. Loop:
           a. ``messages = self.render_task(task, additional_messages=additional_messages)``.
           b. LLM call.
           c. Append an ``LLMRecord`` (``messages=list(task.task_messages)``)
              directly onto ``task.llm_records``.
           d. JSON extraction — on ``json.JSONDecodeError``: check
              ``task.retries_used`` against ``self._generation_retries``,
              raise if exhausted; else set ``additional_messages`` to the
              feedback pair and increment ``task.retries_used``.
           e. Spec validation via ``_process_next_step_output`` (passing
              ``cache_blackboard=self._blackboard``, the live persisted
              blackboard) — on string return: same budget check,
              re-serialised step + feedback as ``additional_messages``,
              increment.
           f. Success: return ``(slot, duration, description)``.

        Returns
        -------
        tuple[BlackboardSlot, int, str]
            Validated slot, duration, and description.
        """
        expected_step = task.next_step_index
        max_duration = max(0, self._tool_calls_limit - expected_step)
        additional_messages: list[dict[str, str]] = []

        while True:
            messages = self.render_task(task, additional_messages=additional_messages)
            engine_result = self._llm_engine.invoke({"messages": messages})
            raw_output: str = engine_result.result

            task.llm_records.append(LLMRecord(
                messages=list(task.task_messages),
                llm_result=engine_result,
                system_prompt_name=task.system_prompt_name,
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
                if task.retries_used >= self._generation_retries:
                    raise ToolAgentError(
                        f"{type(self).__name__}.{self.name}: generation retry budget exhausted "
                        f"after {task.retries_used + 1} attempt(s). Last error is a JSONDecodeError: {exc}"
                    )
                additional_messages = [
                    {"role": "assistant", "content": raw_output},
                    {"role": "user", "content": feedback},
                ]
                task.retries_used += 1
                continue

            # Spec validation
            result = self._process_next_step_output(
                parsed=parsed,
                expected_step=expected_step,
                cache_blackboard=self._blackboard,
                max_duration=max_duration,
                valid_cache_indices=task.valid_cache_indices,
                failed_cache_indices=task.failed_cache_indices,
            )
            if isinstance(result, str):
                feedback = result
                if task.retries_used >= self._generation_retries:
                    raise ToolAgentError(
                        f"{type(self).__name__}.{self.name}: generation retry budget exhausted "
                        f"after {task.retries_used + 1} attempt(s). Last error: {feedback}"
                    )
                step_repr = json.dumps(parsed, indent=2)
                additional_messages = [
                    {"role": "assistant", "content": step_repr},
                    {"role": "user", "content": (
                        f"The step you produced contains an error:\n\n"
                        f"{step_repr}\n\n"
                        f"Error: {feedback}\n\n"
                        "Reflect on this and produce a corrected step."
                    )},
                ]
                task.retries_used += 1
                continue

            slot, duration, description = result
            return slot, duration, description

    async def _agenerate_next_step(self, *, task: ReActTask) -> tuple[BlackboardSlot, int, str]:
        """
        Async mirror of ``_generate_next_step``: uses ``async_invoke`` for each LLM call.

        Full async loop — not delegated to ``asyncio.to_thread``. Retry logic,
        feedback injection, and return type are identical to the sync version.
        """
        expected_step = task.next_step_index
        max_duration = max(0, self._tool_calls_limit - expected_step)
        additional_messages: list[dict[str, str]] = []

        while True:
            messages = self.render_task(task, additional_messages=additional_messages)
            engine_result = await self._llm_engine.async_invoke({"messages": messages})
            raw_output: str = engine_result.result

            task.llm_records.append(LLMRecord(
                messages=list(task.task_messages),
                llm_result=engine_result,
                system_prompt_name=task.system_prompt_name,
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
                if task.retries_used >= self._generation_retries:
                    raise ToolAgentError(
                        f"{type(self).__name__}.{self.name}: generation retry budget exhausted "
                        f"after {task.retries_used + 1} attempt(s). Last error is a JSONDecodeError: {exc}"
                    )
                additional_messages = [
                    {"role": "assistant", "content": raw_output},
                    {"role": "user", "content": feedback},
                ]
                task.retries_used += 1
                continue

            result = self._process_next_step_output(
                parsed=parsed,
                expected_step=expected_step,
                cache_blackboard=self._blackboard,
                max_duration=max_duration,
                valid_cache_indices=task.valid_cache_indices,
                failed_cache_indices=task.failed_cache_indices,
            )
            if isinstance(result, str):
                feedback = result
                if task.retries_used >= self._generation_retries:
                    raise ToolAgentError(
                        f"{type(self).__name__}.{self.name}: generation retry budget exhausted "
                        f"after {task.retries_used + 1} attempt(s). Last error: {feedback}"
                    )
                step_repr = json.dumps(parsed, indent=2)
                additional_messages = [
                    {"role": "assistant", "content": step_repr},
                    {"role": "user", "content": (
                        f"The step you produced contains an error:\n\n"
                        f"{step_repr}\n\n"
                        f"Error: {feedback}\n\n"
                        "Reflect on this and produce a corrected step."
                    )},
                ]
                task.retries_used += 1
                continue

            slot, duration, description = result
            return slot, duration, description

    # ------------------------------------------------------------------ #
    # Task-lifecycle hooks
    # ------------------------------------------------------------------ #
    def _initialize_task(
        self,
        *,
        turns: list[AgentRecord],
        prompt: str,
        inputs: dict,
    ) -> ReActTask:
        """
        Initialize a ReActTask for a single ReAct invocation.

        Pre-allocates the fixed-size running blackboard. No LLM call here —
        step planning is deferred to ``_prepare_next_batch``.

        1. Compute ``valid_cache_indices``/``failed_cache_indices`` via
           ``_compute_cache_index_sets(turns)``.
        2. ``running_blackboard = [BlackboardSlot(step=i) for i in
           range(self._tool_calls_limit + 1)]`` — unchanged fixed-size
           preallocation.
        3. Return a ``ReActTask`` seeded with ``system_prompt_name="think"``
           (not ``"reason_then_act"`` directly — ``_prepare_next_batch``
           stamps that once thinking allows it to run) and
           ``keep_thinking = self._thinking_rounds != 0``.
        """
        valid_cache_indices, failed_cache_indices = self._compute_cache_index_sets(turns)

        running_blackboard = [BlackboardSlot(step=i) for i in range(self._tool_calls_limit + 1)]

        task = ReActTask(
            turns=turns,
            inputs=inputs,
            user_prompt=prompt,
            system_prompt_name="think",
            running_blackboard=running_blackboard,
            executed_steps=set(),
            prepared_steps=[],
            tool_calls_used=0,
            valid_cache_indices=valid_cache_indices,
            failed_cache_indices=failed_cache_indices,
            next_step_index=0,
            step_meta=[ReActStepMeta() for _ in running_blackboard],
        )
        task.keep_thinking = self._thinking_rounds != 0
        return task

    def _prepare_next_batch(self, task: ReActTask) -> ReActTask:
        """
        Prepare the next single-step batch via one LLM call.

        ReAct generates exactly one step per turn; ``_generate_next_step``
        handles rendering (via ``render_task``), the LLM call, JSON
        extraction, spec validation, and generation retries.

        Execution
        ~~~~~~~~~
        1. Stamp ``task.system_prompt_name = "reason_then_act"``
           unconditionally (``_initialize_task`` seeds ``"think"`` now, not
           this directly; idempotent on later calls).
        2. **Validate state**: cursor bounds, prior-step processing.
        3. **Step generation**: call ``_generate_next_step``, which returns
           one planned ``BlackboardSlot``, an observability duration, and a
           step description — ``task.llm_records``/``task.retries_used`` are
           mutated directly inside it.
        4. **Commit**: delegate to ``_apply_react_step_result`` — decrement
           observability counters on prior steps, stamp the slot into the
           running blackboard, resolve placeholders, mark prepared, update
           ``step_meta``, clear ``task.task_messages``/``task.newest_thoughts``.

        No ``if task.prepared_steps: raise`` guard — structurally
        unreachable given ``execute()`` always clears ``prepared_steps``
        before the next ``prepare()`` call.

        Raises
        ------
        ToolAgentError
            If the cursor is out of bounds or the generation retry budget
            is exhausted.
        """
        task.system_prompt_name = "reason_then_act"

        prefix_len = task.next_step_index
        self._validate_react_prepare_state(task)
        generated_slot, observe_duration, description = self._generate_next_step(task=task)
        return self._apply_react_step_result(task, prefix_len, generated_slot, observe_duration, description)

    async def _aprepare_next_batch(self, task: ReActTask) -> ReActTask:
        """
        Async override: uses ``_agenerate_next_step`` so the per-step LLM call
        goes through ``async_invoke`` rather than a worker thread.
        """
        task.system_prompt_name = "reason_then_act"

        prefix_len = task.next_step_index
        self._validate_react_prepare_state(task)
        generated_slot, observe_duration, description = await self._agenerate_next_step(task=task)
        return self._apply_react_step_result(task, prefix_len, generated_slot, observe_duration, description)
