"""
ReActKAgent: Hybrid PlanAct/ReAct Agent with Per-Round Batched Generation

This module provides ``ReActKAgent``, a concrete ``ToolAgent`` subclass that
hybridizes ``PlanActAgent`` and ``ReActAgent``: each LLM round reacts to the
running-plan snapshot (ReAct-style) but returns up to ``steps_per_round``
steps at once, structured and dependency-compiled the same way
``PlanActAgent`` structures a full plan (PlanAct-style) — a partial,
per-round subplan rather than a start-to-finish one.

Not a ``ReActAgent`` subclass. Shares no code with ``ReActAgent`` beyond
what base ``ToolAgent`` already provides.

Iteration Model
---------------
At each round, if a queue of already-compiled batches from a prior round is
non-empty, the next batch is popped and prepared with zero further LLM
calls. Once that queue empties, one LLM call generates the next subplan (up
to ``steps_per_round`` steps, array-shaped like a PlanAct plan but each item
also carrying ``duration``/``description`` like a ReAct step), which is
validated, committed into the preallocated running blackboard, compiled into
concurrency batches, and enqueued for draining.

Observability Window
---------------------
Identical semantics to ``ReActAgent``'s ``duration`` field, except the
decrement happens once per generation round (once per committed subplan)
rather than once per step — a round producing several concurrency batches
only costs everyone else's observability window a single tick.

Return Semantics
-----------------
A return step is optional in any given subplan — most rounds won't include
one. If one is generated, it is relocated to the end of that subplan and
forced to depend on every step committed in the run so far, across all
rounds, not just its own subplan.

Contrast
--------
For one-shot full-plan generation see ``agents/planact.py``
(``PlanActAgent``). For single-step-per-turn iteration see
``agents/react.py`` (``ReActAgent``) — untouched by this class. For the
shared iteration loop, blackboard management, and tool registry see
``agents/toolagent.py`` (``ToolAgent``).
"""

from __future__ import annotations
import json
import pprint
from typing import Any, Callable, Mapping, Optional

from .toolagent import ToolAgent
from .prompts import SUBPLAN_PROMPT
from ..constants.agents import (
    STEP_FIELD,
    TOOL_FIELD,
    ARGS_FIELD,
    DURATION_FIELD,
    DESCRIPTION_FIELD,
    RETURN_TOOL_FULL_NAME,
    BASE_STEP_FIELDS,
    SUBPLAN_FIELDS,
    REQUIRED_SUBPLAN_FIELDS,
)
from ..core import AtomicInvokable
from ..constants.core import NO_VAL
from ..llm.base import LLMEngine
from ..exceptions import ToolAgentError
from ..models.agents import ReActKRunState, ReActStepMeta, BlackboardSlot
from ..models.agents.records import AgentRecord, LLMRecord
from ..utils.agents import extract_dependencies


# --------------------------------------------------------------------------- #
# ReActK Agent
# --------------------------------------------------------------------------- #
class ReActKAgent(ToolAgent):
    """
    Hybrid PlanAct/ReAct agent: batched, reactive iteration.

    **Design**: each round generates up to ``steps_per_round`` steps at once
    from a fresh running-plan snapshot (ReAct-style input), but the returned
    steps are structured, dependency-compiled, and concurrency-batched the
    same way a full ``PlanActAgent`` plan is (PlanAct-style output) — just a
    partial subplan instead of a complete one.

    1. **Initialization** (``_initialize_run_state``)
       - Pre-allocates a fixed-size running blackboard:
         ``tool_calls_limit + 1`` slots, same sizing rule as ``ReActAgent``.
       - Requires ``tool_calls_limit`` to be a concrete integer.
       - No LLM call here — generation is deferred to ``_prepare_next_batch``,
         same as ``ReActAgent``.

    2. **Preparation** (``_prepare_next_batch`` — pop-or-generate)
       - If a queue of already-compiled batches from a prior round is
         non-empty: pop the front batch and prepare it (cascade-check,
         resolve placeholders, mark prepared) — zero LLM calls.
       - Otherwise: generate the next subplan via one LLM call, validate it,
         commit it into the blackboard, compile it into concurrency
         batches, and enqueue them for the next call to drain.

    3. **Execution**
       - Base loop executes whatever batch was just prepared.

    4. **Termination**
       - Once a generated subplan includes a return step, it is isolated
         into its own final batch by the shared batch compiler; once that
         batch drains, the base loop detects completion.

    Parameters (construction)
    ~~~~~~~~~~~~~~~~~~~~~~~~
    ``tool_calls_limit`` (int, REQUIRED): same requirement as ``ReActAgent``
    — must be a concrete integer >= 0; determines pre-allocated blackboard
    size. ``steps_per_round`` (int, default 1): max steps (including a
    return, if present) generated per LLM round; mutable post-construction.
    """

    STEPS_PER_ROUND_FIELD = "STEPS_PER_ROUND"

    def __init__(
        self,
        name: str,
        namespace: str,
        description: str,
        llm_engine: LLMEngine,
        filter_extraneous_inputs: Optional[bool] = None,
        context_enabled: bool = False,
        *,
        steps_per_round: int = 1,
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
        Initialize a ReActKAgent.

        ``tool_calls_limit`` defaults to ``25`` and must be a concrete
        integer >= 0 — ``None`` is not accepted because the running
        blackboard is pre-allocated to ``tool_calls_limit + 1`` slots at
        initialization, same requirement as ``ReActAgent``.
        ``"react_subplan"`` is the key under which the built-in subplan
        prompt is registered in ``self._system_prompts``. No
        extra_parameters keyword is passed to ``super().__init__(...)`` at
        all (``ToolAgent.__init__`` accepts none).
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
        )
        self._steps_per_round: int = 1
        self.steps_per_round = steps_per_round
        self._system_prompts["react_subplan"] = SUBPLAN_PROMPT

    # ------------------------------------------------------------------ #
    # Property Overrides
    # ------------------------------------------------------------------ #
    @property
    def tool_calls_limit(self) -> int:
        """Max allowed non-return tool calls per invoke() run. Must be an int >= 0."""
        return self._tool_calls_limit

    @tool_calls_limit.setter
    def tool_calls_limit(self, value: int) -> None:
        """Reject None and non-integer values; ReActKAgent pre-allocates by this count."""
        if type(value) is not int or value < 0:
            raise ToolAgentError("ReActKAgent requires tool_calls_limit to be an int >= 0.")
        self._tool_calls_limit = value

    @property
    def steps_per_round(self) -> int:
        """Max steps (including a return, if present) generated per LLM round. Mutable."""
        return self._steps_per_round

    @steps_per_round.setter
    def steps_per_round(self, value: int) -> None:
        if type(value) is not int or value < 1:
            raise ToolAgentError("ReActKAgent requires steps_per_round to be an int >= 1.")
        self._steps_per_round = value

    # ------------------------------------------------------------------ #
    # Tool-Agent Hooks — Initialization
    # ------------------------------------------------------------------ #
    def _initialize_run_state(
        self,
        *,
        turns: list[AgentRecord],
        prompt: str,
        inputs: dict,
        valid_cache_indices: frozenset[int],
        failed_cache_indices: frozenset[int],
    ) -> ReActKRunState:
        """
        Initialize run state for a single ReActK invocation.

        No LLM call here — generation is deferred to ``_prepare_next_batch``,
        same as ``ReActAgent``, so no ``_ainitialize_run_state`` override is
        needed; the base default (``asyncio.to_thread`` wrap) suffices.

        1. Render ``self._system_prompts["react_subplan"]`` with TOOLS/
           TOOL_CALLS_LIMIT/CONSTANTS (same three sources every ToolAgent
           subclass uses) plus STEPS_PER_ROUND from ``self.steps_per_round``.
        2. Build messages via ``build_messages``; raise if empty.
        3. Snapshot cache_blackboard if context is enabled.
        4. Pre-allocate a fixed-size running blackboard of
           ``tool_calls_limit + 1`` slots.
        5. Construct and return ``ReActKRunState`` with an empty
           ``pending_batches`` queue.
        """
        render_context = {
            ToolAgent.TOOLS_FIELD: self.actions_context(),
            ToolAgent.LIMIT_FIELD: str(self._tool_calls_limit),
            ToolAgent.CONSTANTS_FIELD: self.constants_context(),
            self.STEPS_PER_ROUND_FIELD: str(self.steps_per_round),
        }
        system = self._system_prompts["react_subplan"].render(render_context)
        messages = self.build_messages(system, turns, prompt)
        if not messages:
            raise ToolAgentError(f"{type(self).__name__}.{self.name}: messages must be non-empty.")

        cache_blackboard = (
            [slot.copy() for slot in self._blackboard] if self.context_enabled else []
        )

        working_messages = [dict(m) for m in messages]

        # Preallocate fixed-size run blackboard: non-return calls + 1 return call.
        running_blackboard = [BlackboardSlot(step=i) for i in range(self._tool_calls_limit + 1)]

        return ReActKRunState(
            inputs=inputs,
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
            pending_batches=[],
            step_meta=[ReActStepMeta() for _ in running_blackboard],
            retries_used=0,
        )

    # ------------------------------------------------------------------ #
    # Message building
    # ------------------------------------------------------------------ #
    def _build_reactk_messages(
        self,
        state: ReActKRunState,
        cursor: int,
        *,
        max_duration: int,
        round_cap: int,
    ) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
        """
        Build the working message list and delta for one subplan-generation round.

        Adapted from ``ReActAgent._build_react_messages``: identical
        running-plan-snapshot rendering (executed/failed steps 0..cursor-1,
        with description/result_ref/run_id/observable_result), but the
        closing ask requests a subplan of up to ``round_cap`` steps instead
        of exactly one. Returns ``(working_messages, delta)``.
        """
        working_messages: list[dict[str, str]] = [dict(m) for m in state.messages]

        running_records: list[dict[str, Any]] = []
        for idx in range(cursor):
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
                f"RUNNING PLAN STEPS 0-{cursor - 1} SO FAR:\n"
                "Steps may be EXECUTED (result available via result_ref) or FAILED "
                "(error shown; do not reference result_ref for failed steps).\n"
                "Use descriptions to understand what each step was intended to do.\n"
                "Use result_ref placeholders when a new arg needs a prior executed step's value.\n"
                "observable_result fields are for OBSERVATION ONLY: use them only to choose "
                "the next tool(s) or branch.\n"
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
                    f"Produce the NEXT subplan of UP TO {round_cap} step(s) for the current task, "
                    "as a JSON array (even if it contains only one element). "
                    "Include a return step only once the running plan has fully completed the "
                    "task, and only as the LAST element of the array. "
                    "Preserve symbolic dataflow with quoted placeholders; do not copy "
                    "observable_result values into args; no step may need to see a sibling in "
                    "this same array's actual result to choose its own tool. "
                    f"For every step in this array, duration must be an int from 0 to {max_duration}."
                ),
            }
        )

        delta = [state.messages[-1], working_messages[-2], working_messages[-1]]
        return working_messages, delta

    # ------------------------------------------------------------------ #
    # Subplan processing
    # ------------------------------------------------------------------ #
    def _process_subplan_output(
        self,
        *,
        parsed: Any,
        cursor: int,
        max_duration: int,
    ) -> list[tuple[BlackboardSlot, int, str]] | str:
        """
        Validate one parsed subplan array's per-item shape and convert each
        item into a ``(slot, duration, description)`` triple.

        Only per-item shape/type validation happens here — tool existence,
        dependency bounds, cache refs, and budget checks are deferred to
        ``_validate_subplan_slots``, which runs after
        ``_normalize_subplan_slots`` (return relocation can change final
        ordering). Uses absolute step indices (``cursor + i``), unlike
        ``PlanActAgent._process_plan_output``'s plan-local ``i`` — a
        subplan lands mid-run in the preallocated blackboard, not at a
        fresh index 0.
        """
        if not isinstance(parsed, list) or not parsed:
            return "The subplan must be a non-empty JSON array."

        results: list[tuple[BlackboardSlot, int, str]] = []

        for i, item in enumerate(parsed):
            expected_step = cursor + i

            if not isinstance(item, Mapping):
                return (
                    f"Subplan step at absolute index {expected_step} must be a JSON object "
                    f"(dict), not {type(item).__name__!r}."
                )

            step_dict = self._validate_tool_step_dict(
                item,
                expected_step=expected_step,
                allowed_fields=SUBPLAN_FIELDS,
                required_fields=REQUIRED_SUBPLAN_FIELDS,
                context="subplan step",
            )
            if isinstance(step_dict, str):
                return step_dict

            duration = step_dict.pop(DURATION_FIELD)
            if type(duration) is not int or duration < 0 or duration > max_duration:
                return (
                    f"subplan step {DURATION_FIELD!r} must be an int in "
                    f"[0, {max_duration}] for expected_step={expected_step}; got {duration!r}."
                )

            description = step_dict.pop(DESCRIPTION_FIELD)
            if type(description) is not str:
                return f"subplan step {DESCRIPTION_FIELD!r} must be a string; got {type(description).__name__!r}."

            description = description.strip()
            if not description:
                return f"subplan step {DESCRIPTION_FIELD!r} cannot be empty."

            slot = self._tool_step_dict_to_slot(
                step_dict,
                step=expected_step,
                allowed_fields=BASE_STEP_FIELDS,
                context="subplan step",
            )

            results.append((slot, duration, description))

        return results

    def _normalize_subplan_slots(
        self,
        items: list[tuple[BlackboardSlot, int, str]],
        *,
        cursor: int,
    ) -> list[tuple[BlackboardSlot, int, str]] | str:
        """
        Normalize a generated subplan into final absolute running-blackboard order.

        Adapted from ``PlanActAgent._normalize_planned_slots``. Unlike a
        full plan, a return step is OPTIONAL here — most rounds won't have
        one, so absence is never synthesized. If present, it is relocated
        to the end of *this subplan* (not the whole run) and forced to
        depend on every step committed in the run so far, across all
        rounds, using absolute indices (``cursor``-offset instead of
        starting at 0).

        Known shared limitation: like ``_normalize_planned_slots``,
        argument placeholders are not rewritten if relocation shifts a
        non-return item's absolute position — accepted parity with
        PlanAct's existing behavior, not introduced here (see spec
        Validation notes).
        """
        items = list(items)

        return_positions = [
            i for i, (slot, _duration, _description) in enumerate(items)
            if slot.tool == RETURN_TOOL_FULL_NAME
        ]

        if len(return_positions) > 1:
            return (
                f"subplan contains multiple return steps at positions {return_positions!r}. "
                "Include at most one return step per subplan."
            )

        if len(return_positions) == 1:
            return_item = items.pop(return_positions[0])
            items.append(return_item)

        for local_i, (slot, _duration, _description) in enumerate(items):
            slot.step = cursor + local_i

        if return_positions:
            return_slot = items[-1][0]
            return_slot.step_dependencies = tuple(range(return_slot.step))
            return_slot.await_step = NO_VAL

        return items

    def _validate_subplan_slots(
        self,
        *,
        items: list[tuple[BlackboardSlot, int, str]],
        cursor: int,
        round_cap: int,
        cache_blackboard: list[BlackboardSlot],
        valid_cache_indices: frozenset[int],
        failed_cache_indices: frozenset[int],
    ) -> str | None:
        """
        Validate a normalized subplan's invariants not already guaranteed by
        ``_normalize_subplan_slots``.

        Adapted from ``PlanActAgent._validate_planned_slots``, plus the two
        new per-round budget checks: ``len(items) <= steps_per_round`` and
        ``non_return_count <= round_cap`` (``round_cap`` already folds in
        the remaining ``tool_calls_limit`` budget).
        """
        if not items:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: internal error: normalized subplan is empty."
            )

        if len(items) > self.steps_per_round:
            return (
                f"subplan contains {len(items)} step(s), exceeding steps_per_round of "
                f"{self.steps_per_round}."
            )

        return_name = RETURN_TOOL_FULL_NAME
        non_return_count = sum(1 for slot, _duration, _description in items if slot.tool != return_name)
        if non_return_count > round_cap:
            return (
                f"subplan contains {non_return_count} non-return step(s), exceeding the "
                f"remaining tool-call budget of {round_cap} for this round."
            )

        cache_len = len(cache_blackboard)

        for slot, duration, _description in items:
            absolute = slot.step

            if not self.has_tool(slot.tool):
                return (
                    f"subplan step {absolute} uses an unknown tool: {slot.tool!r}. "
                    "Use only tools from the available list."
                )

            if slot.tool == return_name and duration != 0:
                return f"return tool must use {DURATION_FIELD!r} 0; got {duration!r}."

            if slot.tool == return_name:
                step_refs = extract_dependencies(slot.args, placeholder_pattern=self.STEP_REF_PATTERN)
                bad_step_refs = [ref for ref in step_refs if ref < 0 or ref >= absolute]
                if bad_step_refs:
                    return (
                        f"return step has invalid step references {sorted(set(bad_step_refs))!r} "
                        f"in args; step refs must be earlier steps (< {absolute})."
                    )
            else:
                bad_step_deps = [dep for dep in slot.step_dependencies if dep < 0 or dep >= absolute]
                if bad_step_deps:
                    return (
                        f"subplan step {absolute} has invalid step dependencies "
                        f"{sorted(set(bad_step_deps))!r}; all deps must be earlier steps (< {absolute})."
                    )

            if slot.await_step is not NO_VAL and slot.await_step >= absolute:
                return (
                    f"subplan step {absolute} has an invalid await_step {slot.await_step!r}; "
                    f"await_step must reference an earlier step (< {absolute})."
                )

            cache_refs = extract_dependencies(slot.args, placeholder_pattern=self.CACHE_REF_PATTERN)

            out_of_range = [idx for idx in cache_refs if idx < 0 or idx >= cache_len]
            if out_of_range:
                return (
                    f"subplan step {absolute} references cache indices that do not exist: "
                    f"{sorted(set(out_of_range))!r} (cache has {cache_len} entries)."
                )

            failed_in_conv = [idx for idx in cache_refs if idx in failed_cache_indices]
            if failed_in_conv:
                details = "; ".join(
                    f"entry {idx} ({cache_blackboard[idx].tool}): {cache_blackboard[idx].error}"
                    for idx in sorted(set(failed_in_conv))
                )
                return (
                    f"subplan step {absolute} references cache entries that failed in this "
                    f"conversation and cannot be used: {details}."
                )

            out_of_conv = [
                idx for idx in cache_refs
                if 0 <= idx < cache_len
                and idx not in valid_cache_indices
                and idx not in failed_cache_indices
            ]
            if out_of_conv:
                return (
                    f"subplan step {absolute} references cache indices not part of this "
                    f"conversation: {sorted(set(out_of_conv))!r}."
                )

        return None

    # ------------------------------------------------------------------ #
    # Generation (retry loop, fully inline — not shared with PlanAct/ReAct)
    # ------------------------------------------------------------------ #
    def _generate_next_subplan(
        self,
        *,
        messages: list[dict[str, str]],
        cursor: int,
        delta: list[dict[str, str]],
        retries_used: int,
        max_duration: int,
        round_cap: int,
        cache_blackboard: list[BlackboardSlot],
        valid_cache_indices: frozenset[int],
        failed_cache_indices: frozenset[int],
    ) -> tuple[list[tuple[BlackboardSlot, int, str]], list[LLMRecord], int]:
        """
        Generate, parse, and validate one subplan, with bounded retry.

        Mirrors ``PlanActAgent._generate_plan``'s retry-loop structure
        (array output) — not ``ReActAgent._generate_next_step``'s (single
        object). Each attempt: LLM call -> LLMRecord -> JSON extraction ->
        ``_process_subplan_output`` -> ``_normalize_subplan_slots`` ->
        ``_validate_subplan_slots``, injecting structured feedback and
        retrying against the shared ``retries_used`` budget on any failure.
        """
        if not messages:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: messages must be non-empty."
            )

        working_messages: list[dict[str, str]] = list(messages)
        llm_records: list[LLMRecord] = []

        while True:
            engine_result = self._llm_engine.invoke({"messages": [dict(m) for m in working_messages]})
            raw_output: str = engine_result.result

            record_messages = delta if not llm_records else list(working_messages[-2:])
            llm_records.append(LLMRecord(
                messages=record_messages,
                llm_result=engine_result,
                system_prompt_name="react_subplan",
            ))

            # JSON extraction
            try:
                parsed = self._extract_from_json_string(raw_output)
            except json.JSONDecodeError as exc:
                feedback = (
                    f"Your output could not be parsed as valid JSON.\n\n"
                    f"Decoder error: {exc}\n\n"
                    f"The response you produced was:\n\n{raw_output}\n\n"
                    "Produce a correctly formatted JSON array."
                )
                if retries_used >= self._generation_retries:
                    raise ToolAgentError(
                        f"{type(self).__name__}.{self.name}: generation retry budget exhausted "
                        f"after {retries_used + 1} attempt(s). Last error is JSONDecodeError: {exc}"
                    )
                working_messages.append({"role": "assistant", "content": raw_output})
                working_messages.append({"role": "user", "content": feedback})
                retries_used += 1
                continue

            # Per-item shape validation
            result = self._process_subplan_output(
                parsed=parsed,
                cursor=cursor,
                max_duration=max_duration,
            )
            if isinstance(result, str):
                if retries_used >= self._generation_retries:
                    raise ToolAgentError(
                        f"{type(self).__name__}.{self.name}: generation retry budget exhausted "
                        f"after {retries_used + 1} attempt(s). Last error: {result}"
                    )
                subplan_repr = json.dumps(parsed, indent=2)
                working_messages.append({"role": "assistant", "content": subplan_repr})
                working_messages.append({"role": "user", "content": result})
                retries_used += 1
                continue

            # Return relocation + forced dependency
            normalized = self._normalize_subplan_slots(result, cursor=cursor)
            if isinstance(normalized, str):
                if retries_used >= self._generation_retries:
                    raise ToolAgentError(
                        f"{type(self).__name__}.{self.name}: generation retry budget exhausted "
                        f"after {retries_used + 1} attempt(s). Last error: {normalized}"
                    )
                subplan_repr = json.dumps(parsed, indent=2)
                working_messages.append({"role": "assistant", "content": subplan_repr})
                working_messages.append({"role": "user", "content": normalized})
                retries_used += 1
                continue

            # Cross-item invariants + budget checks
            feedback = self._validate_subplan_slots(
                items=normalized,
                cursor=cursor,
                round_cap=round_cap,
                cache_blackboard=cache_blackboard,
                valid_cache_indices=valid_cache_indices,
                failed_cache_indices=failed_cache_indices,
            )
            if feedback is not None:
                if retries_used >= self._generation_retries:
                    raise ToolAgentError(
                        f"{type(self).__name__}.{self.name}: generation retry budget exhausted "
                        f"after {retries_used + 1} attempt(s). Last error: {feedback}"
                    )
                subplan_repr = json.dumps(parsed, indent=2)
                working_messages.append({"role": "assistant", "content": subplan_repr})
                working_messages.append({"role": "user", "content": feedback})
                retries_used += 1
                continue

            return normalized, llm_records, retries_used

    async def _agenerate_next_subplan(
        self,
        *,
        messages: list[dict[str, str]],
        cursor: int,
        delta: list[dict[str, str]],
        retries_used: int,
        max_duration: int,
        round_cap: int,
        cache_blackboard: list[BlackboardSlot],
        valid_cache_indices: frozenset[int],
        failed_cache_indices: frozenset[int],
    ) -> tuple[list[tuple[BlackboardSlot, int, str]], list[LLMRecord], int]:
        """Async mirror of ``_generate_next_subplan``: uses ``async_invoke`` for
        each LLM call. Retry logic, feedback injection, and return type identical."""
        if not messages:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: messages must be non-empty."
            )

        working_messages: list[dict[str, str]] = list(messages)
        llm_records: list[LLMRecord] = []

        while True:
            engine_result = await self._llm_engine.async_invoke({"messages": [dict(m) for m in working_messages]})
            raw_output: str = engine_result.result

            record_messages = delta if not llm_records else list(working_messages[-2:])
            llm_records.append(LLMRecord(
                messages=record_messages,
                llm_result=engine_result,
                system_prompt_name="react_subplan",
            ))

            try:
                parsed = self._extract_from_json_string(raw_output)
            except json.JSONDecodeError as exc:
                feedback = (
                    f"Your output could not be parsed as valid JSON.\n\n"
                    f"Decoder error: {exc}\n\n"
                    f"The response you produced was:\n\n{raw_output}\n\n"
                    "Produce a correctly formatted JSON array."
                )
                if retries_used >= self._generation_retries:
                    raise ToolAgentError(
                        f"{type(self).__name__}.{self.name}: generation retry budget exhausted "
                        f"after {retries_used + 1} attempt(s). Last error is JSONDecodeError: {exc}"
                    )
                working_messages.append({"role": "assistant", "content": raw_output})
                working_messages.append({"role": "user", "content": feedback})
                retries_used += 1
                continue

            result = self._process_subplan_output(
                parsed=parsed,
                cursor=cursor,
                max_duration=max_duration,
            )
            if isinstance(result, str):
                if retries_used >= self._generation_retries:
                    raise ToolAgentError(
                        f"{type(self).__name__}.{self.name}: generation retry budget exhausted "
                        f"after {retries_used + 1} attempt(s). Last error: {result}"
                    )
                subplan_repr = json.dumps(parsed, indent=2)
                working_messages.append({"role": "assistant", "content": subplan_repr})
                working_messages.append({"role": "user", "content": result})
                retries_used += 1
                continue

            normalized = self._normalize_subplan_slots(result, cursor=cursor)
            if isinstance(normalized, str):
                if retries_used >= self._generation_retries:
                    raise ToolAgentError(
                        f"{type(self).__name__}.{self.name}: generation retry budget exhausted "
                        f"after {retries_used + 1} attempt(s). Last error: {normalized}"
                    )
                subplan_repr = json.dumps(parsed, indent=2)
                working_messages.append({"role": "assistant", "content": subplan_repr})
                working_messages.append({"role": "user", "content": normalized})
                retries_used += 1
                continue

            feedback = self._validate_subplan_slots(
                items=normalized,
                cursor=cursor,
                round_cap=round_cap,
                cache_blackboard=cache_blackboard,
                valid_cache_indices=valid_cache_indices,
                failed_cache_indices=failed_cache_indices,
            )
            if feedback is not None:
                if retries_used >= self._generation_retries:
                    raise ToolAgentError(
                        f"{type(self).__name__}.{self.name}: generation retry budget exhausted "
                        f"after {retries_used + 1} attempt(s). Last error: {feedback}"
                    )
                subplan_repr = json.dumps(parsed, indent=2)
                working_messages.append({"role": "assistant", "content": subplan_repr})
                working_messages.append({"role": "user", "content": feedback})
                retries_used += 1
                continue

            return normalized, llm_records, retries_used

    # ------------------------------------------------------------------ #
    # Commit
    # ------------------------------------------------------------------ #
    def _apply_subplan_result(
        self,
        state: ReActKRunState,
        cursor: int,
        items: list[tuple[BlackboardSlot, int, str]],
        llm_records: list[LLMRecord],
    ) -> ReActKRunState:
        """
        Commit one validated subplan into the running blackboard and enqueue
        its compiled batches.

        1. Extend ``state.llm_records``.
        2. Decrement every ``step_meta[i].observable > 0`` by 1, ONCE for
           the whole round (same loop body as
           ``ReActAgent._apply_react_step_result``, invoked once per
           subplan here instead of once per step).
        3. Stamp each item into its preallocated absolute slot (status
           ``PLANNED`` — no placeholder resolution yet; that happens when
           this slot's batch is later popped in ``_prepare_next_batch``).
        4. Compile new batches against the full absolute-indexed prefix
           (``running_blackboard[: cursor + len(items)]``) so dependency
           lookups spanning earlier rounds resolve correctly, then filter
           every batch down to indices ``>= cursor`` before enqueueing —
           keeps only this round's newly introduced steps.
        5. Advance the cursor by ``len(items)``.
        """
        state.llm_records.extend(llm_records)

        for meta in state.step_meta:
            if meta.observable > 0:
                meta.observable -= 1

        for slot, duration, description in items:
            target = state.running_blackboard[slot.step]

            if target.step != slot.step:
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: running slot step mismatch at index "
                    f"{slot.step}: target.step={target.step}."
                )
            if not target.is_empty():
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: attempted to commit into non-empty "
                    f"slot {slot.step}."
                )

            target.tool = slot.tool
            target.args = slot.args
            target.step_dependencies = slot.step_dependencies
            target.await_step = slot.await_step
            target.status = BlackboardSlot.PLANNED

            state.step_meta[slot.step].observable = duration
            state.step_meta[slot.step].description = description

        return_idx = (
            items[-1][0].step
            if items and items[-1][0].tool == RETURN_TOOL_FULL_NAME
            else None
        )

        batches = self._compile_batches_from_deps(
            planned_slots=state.running_blackboard[: cursor + len(items)],
            return_idx=return_idx,
        )

        new_batches: list[list[int]] = []
        for batch in batches:
            filtered = [i for i in batch if i >= cursor]
            if filtered:
                new_batches.append(filtered)

        state.pending_batches.extend(new_batches)
        state.next_step_index = cursor + len(items)

        return state

    # ------------------------------------------------------------------ #
    # Tool-Agent Hooks — Preparation
    # ------------------------------------------------------------------ #
    def _drain_pending_batch(self, state: ReActKRunState) -> ReActKRunState:
        """
        Pop and prepare the front of ``state.pending_batches`` — zero LLM
        calls. Identical per-slot loop body to
        ``PlanActAgent._prepare_next_batch``: bounds/status checks, cascade
        check under ``fail_fast=False``, placeholder resolution, and
        marking ``PREPARED``. Shared by both the sync and async
        ``_prepare_next_batch`` overrides since this branch has no I/O.
        """
        batch = state.pending_batches.pop(0)
        board = state.running_blackboard
        board_len = len(board)

        prepared_in_batch: list[int] = []
        for i in batch:
            if i < 0 or i >= board_len:
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: batch index {i} out of range "
                    f"(plan length={board_len})."
                )

            slot = board[i]

            if slot.step != i:
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: running slot step mismatch at index "
                    f"{i}: slot.step={slot.step}."
                )
            if slot.is_executed():
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: batch references already executed step {i}."
                )
            if slot.is_prepared():
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: batch references already prepared step {i}."
                )
            if slot.is_failed():
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: batch references failed step {i}."
                )
            if not slot.is_planned():
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: batch references non-planned step {i} "
                    f"with status={slot.status!r}."
                )
            if slot.tool is NO_VAL or not isinstance(slot.tool, str) or not slot.tool.strip():
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: step {i} has invalid tool name in "
                    "running_blackboard."
                )

            # Cascade-fail: when fail_fast=False, propagate failures through arg dependencies.
            if not self._fail_fast and self._check_cascade_failure(slot, board):
                continue

            slot.resolved_args = self._resolve_placeholders(slot.args, state=state)

            if slot.result is not NO_VAL:
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: internal error: step {i} result was "
                    "set during preparation."
                )

            slot.error = NO_VAL
            slot.status = BlackboardSlot.PREPARED
            prepared_in_batch.append(i)

        state.prepared_steps = sorted(prepared_in_batch)
        return state

    def _prepare_next_batch(self, state: ReActKRunState) -> ReActKRunState:
        """
        Pop-or-generate: drain a pending batch if one exists, else generate
        the next subplan.

        1. Guard: ``prepared_steps`` must be empty.
        2. If ``pending_batches`` non-empty: delegate to
           ``_drain_pending_batch`` — zero LLM calls.
        3. Else: compute ``max_duration``/``round_cap``, build messages via
           ``_build_reactk_messages``, generate via
           ``_generate_next_subplan``, commit via
           ``_apply_subplan_result``. Returns with ``prepared_steps`` still
           empty — the base loop's ``continue`` re-invokes this method next
           iteration, which now takes the drain branch.
        """
        if state.prepared_steps:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: cannot prepare next batch while "
                "prepared_steps is non-empty."
            )

        if state.pending_batches:
            return self._drain_pending_batch(state)

        cursor = state.next_step_index
        max_duration = max(0, self._tool_calls_limit - cursor)
        round_cap = min(self.steps_per_round, self._tool_calls_limit - state.tool_calls_used)

        working_messages, delta = self._build_reactk_messages(
            state, cursor, max_duration=max_duration, round_cap=round_cap,
        )

        items, llm_records, new_retries_used = self._generate_next_subplan(
            messages=working_messages,
            cursor=cursor,
            delta=delta,
            retries_used=state.retries_used,
            max_duration=max_duration,
            round_cap=round_cap,
            cache_blackboard=state.cache_blackboard,
            valid_cache_indices=state.valid_cache_indices,
            failed_cache_indices=state.failed_cache_indices,
        )
        state.retries_used = new_retries_used

        return self._apply_subplan_result(state, cursor, items, llm_records)

    async def _aprepare_next_batch(self, state: ReActKRunState) -> ReActKRunState:
        """
        Async override — needed unlike ``PlanActAgent`` (never generates
        here) or ``ReActAgent`` (always generates here): this method's
        generation branch has a genuine LLM call, its drain branch does
        not. Mirrors sync ``_prepare_next_batch`` exactly except the
        generation branch awaits ``_agenerate_next_subplan``.
        """
        if state.prepared_steps:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: cannot prepare next batch while "
                "prepared_steps is non-empty."
            )

        if state.pending_batches:
            return self._drain_pending_batch(state)

        cursor = state.next_step_index
        max_duration = max(0, self._tool_calls_limit - cursor)
        round_cap = min(self.steps_per_round, self._tool_calls_limit - state.tool_calls_used)

        working_messages, delta = self._build_reactk_messages(
            state, cursor, max_duration=max_duration, round_cap=round_cap,
        )

        items, llm_records, new_retries_used = await self._agenerate_next_subplan(
            messages=working_messages,
            cursor=cursor,
            delta=delta,
            retries_used=state.retries_used,
            max_duration=max_duration,
            round_cap=round_cap,
            cache_blackboard=state.cache_blackboard,
            valid_cache_indices=state.valid_cache_indices,
            failed_cache_indices=state.failed_cache_indices,
        )
        state.retries_used = new_retries_used

        return self._apply_subplan_result(state, cursor, items, llm_records)
