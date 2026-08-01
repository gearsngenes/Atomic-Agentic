"""
PlanActAgent2: One-Shot LLM Planner on the Agent2 think/prepare/execute Lifecycle

Reworks ``PlanActAgent`` (``agents/planact.py``) onto the ``ToolAgent2``
base: thinking (if enabled) fully concludes before a single LLM call
produces the whole plan as a JSON array of tool-call steps, which is then
compiled into topologically-sorted concurrent batches and executed without
further LLM interaction. Strictly sequential — ``prepare``/``execute`` stay
gated by ``task.keep_thinking`` (``_permits_unbounded_thinking`` not
overridden, stays ``Agent2``'s ``False``), matching v1 PlanActAgent's own
posture that a one-shot batch planner has no coherent "partial plan"
concept.

Plan generation itself moves from an eager ``_initialize_task`` call
(v1) to the first ``_prepare_next_batch``/``_aprepare_next_batch`` call —
gated on ``task.batches`` being empty — so it correctly waits for thinking
to conclude first. See ``.claude/brainstorms/toolagent2-lifecycle.md`` and
``.claude/specs/planact2-react2.md`` for the full design record, including
why ``_finalize_planact_task`` doesn't survive as a separate method (its own
validation was provably dead code given what ``_normalize_planned_slots``
already guarantees).

Contrast
--------
For adaptive, step-by-step iteration see ``agents/react2.py``
(``ReActAgent2``). For the shared thinking/toolbox/blackboard machinery see
``agents/toolagent2.py`` (``ToolAgent2``).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Literal, Mapping, Optional

from .toolagent2 import ToolAgent2
from .prompts import PLANNER_PROMPT, PLANACT_THINKING_PROMPT
from ..constants.agents import (
    RETURN_TOOL_FULL_NAME,
    RETURN_VALUE_FIELD,
    PLAN_FIELDS,
    REQUIRED_PLAN_FIELDS,
)
from ..core import AtomicInvokable
from ..constants.core import NO_VAL
from ..llm.base import LLMEngine
from ..exceptions import ToolAgentError
from ..models.agents.prompts import PromptConfig
from ..models.agents.tasks import PlanActTask
from ..models.agents import BlackboardSlot
from ..models.agents.records import AgentRecord, LLMRecord
from ..utils.agents import extract_dependencies

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# PlanAct Agent2
# --------------------------------------------------------------------------- #
class PlanActAgent2(ToolAgent2):
    """
    One-shot planner agent on the ``ToolAgent2`` lifecycle: generates the
    entire plan upfront (after thinking, if enabled, concludes), executes
    it in concurrent batches with no further LLM interaction.

    **Design**:

    1. **Thinking** (optional, shared ``Agent2`` machinery) — zero or more
       rounds of categorized thoughts before any plan exists.
    2. **Planning** (first ``_prepare_next_batch``/``_aprepare_next_batch``
       call, once ``task.keep_thinking`` is ``False``) — one LLM call
       produces the full JSON plan; normalized (return moved to end, added
       if missing) and validated (tool existence, dependency graph,
       ``await_step`` ordering, ``tool_calls_limit`` budget, cache
       reference validity), with generation retries on parse/validation
       failure. Compiled into topologically-sorted concurrent batches.
    3. **Execution** (``execute()``, shared/final on ``ToolAgent2``) — each
       batch runs concurrently; the return step always lands in its own
       final batch.

    Advantages
    ~~~~~~~~~~
    - **No replanning**: full plan known upfront, no latency per iteration.
    - **Concurrency-friendly**: topological compilation maximizes
      parallelism.
    - **Deterministic**: same inputs produce the same plan every time.

    Limitations
    ~~~~~~~~~~~
    - **No adaptivity**: can't branch based on intermediate results.
    - **Plan quality**: entirely dependent on the single planning turn.
    - **Error recovery**: a failed step fails the whole plan under
      ``fail_fast=True`` (no dynamic replanning).
    """

    _THINK_PROMPT: PromptConfig = PLANACT_THINKING_PROMPT

    def __init__(
        self,
        name: str,
        namespace: str,
        description: str,
        llm_engine: LLMEngine,
        filter_extraneous_inputs: Optional[bool] = None,
        context_enabled: bool = False,
        tool_calls_limit: int | None = None,
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
        Initialize a PlanActAgent2.

        No ``*`` keyword-only separator — every parameter is
        positional-or-keyword, matching ``ToolAgent2``'s own constructor
        style (a deliberate departure from v1 PlanActAgent, which had one
        before ``tool_calls_limit``). ``"plan_first"`` is the key under
        which the built-in planning prompt is registered in
        ``self._system_prompts``. Thinking knobs pass straight through to
        ``super().__init__()``; ``thinking_instructions`` is not exposed
        here (``ToolAgent2``'s own posture, see its docstring).
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
        self._system_prompts["plan_first"] = PLANNER_PROMPT

    # ------------------------------------------------------------------ #
    # Initialization / validation
    # ------------------------------------------------------------------ #
    def _normalize_planned_slots(
        self,
        planned_slots: list[BlackboardSlot],
    ) -> list[BlackboardSlot] | str:
        """
        Normalize a generated PlanAct slot list into final running-blackboard order.

        Normalization policy
        --------------------
        - At most one return slot may be present.
        - If a return slot is present, it is moved to the end.
        - If no return slot is present, `return(None)` is appended.
        - Final list positions become authoritative step indices.
        - The final return slot is forced to depend on all prior slots so completion
          represents the whole plan, not just the value in return args.
        - The final return slot cannot have await_step.

        Parameters
        ----------
        planned_slots : list[BlackboardSlot]
            Generated planned slots before return-position normalization.

        Returns
        -------
        list[BlackboardSlot]
            Normalized planned slots.
        str
            LLM-facing feedback if the plan contains multiple return steps.
        """
        slots: list[BlackboardSlot] = [slot.copy() for slot in planned_slots]

        return_name = RETURN_TOOL_FULL_NAME
        return_positions = [
            i for i, slot in enumerate(slots)
            if slot.tool == return_name
        ]

        if len(return_positions) > 1:
            return (
                f"plan contains multiple return steps at positions {return_positions!r}. "
                "Include at most one return step."
            )

        if len(return_positions) == 1:
            return_slot = slots.pop(return_positions[0])
            slots.append(return_slot)
        else:
            slots.append(
                BlackboardSlot(
                    step=len(slots),
                    tool=return_name,
                    args={RETURN_VALUE_FIELD: None},
                    resolved_args=NO_VAL,
                    result=NO_VAL,
                    error=NO_VAL,
                    status=BlackboardSlot.PLANNED,
                    step_dependencies=tuple(),
                    await_step=NO_VAL,
                )
            )

        for i, slot in enumerate(slots):
            slot.step = i
            slot.resolved_args = NO_VAL
            slot.result = NO_VAL
            slot.error = NO_VAL
            slot.status = BlackboardSlot.PLANNED

        return_idx = len(slots) - 1
        return_slot = slots[return_idx]

        # Return is a synthetic finalization step, not a normal data-only step.
        # Force it to depend on every prior step so completion represents the whole plan.
        # This makes the blackboard invariant explicit even though batch compilation also
        # isolates return as the final batch.
        return_slot.step_dependencies = tuple(range(return_idx))
        return_slot.await_step = NO_VAL
        return_slot.status = BlackboardSlot.PLANNED

        return slots

    def _validate_planned_slots(
        self,
        *,
        planned_slots: list[BlackboardSlot],
        cache_blackboard: list[BlackboardSlot],
        valid_cache_indices: frozenset[int],
        failed_cache_indices: frozenset[int],
    ) -> str | None:
        """
        Validate a normalized PlanAct planned-slot list.

        Checks only properties that ``_normalize_planned_slots`` does not
        guarantee: tool existence, step dependency graph, await_step ordering,
        tool_calls_limit, and cache reference validity.

        Parameters
        ----------
        planned_slots : list[BlackboardSlot]
            Normalized planned slots.

        cache_blackboard : list[BlackboardSlot]
            Blackboard entries to validate cache references against
            (``self._blackboard`` at the call site — see
            ``_process_plan_output``).

        valid_cache_indices : frozenset[int]
            Cache indices from the current conversation that completed successfully.

        failed_cache_indices : frozenset[int]
            Cache indices from the current conversation that failed; cannot be
            referenced by a new plan.

        Returns
        -------
        str
            LLM-facing feedback string describing the first invariant violation
            found. No class/name prefix.
        None
            All invariants satisfied.

        Raises
        ------
        ToolAgentError
            If ``planned_slots`` or ``cache_blackboard`` are not lists (entry
            guard — these are caller-contract violations, not LLM output errors).
        """
        if not isinstance(planned_slots, list) or not planned_slots:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: planned_slots must be a non-empty list."
            )

        if not isinstance(cache_blackboard, list):
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: cache_blackboard must be a list."
            )

        return_name = RETURN_TOOL_FULL_NAME
        cache_len = len(cache_blackboard)

        for i, slot in enumerate(planned_slots):
            if not self.has_tool(slot.tool):
                return (
                    f"plan step {i} uses an unknown tool: {slot.tool!r}. "
                    "Use only tools from the available list."
                )

            # Return slot step_dependencies are synthetic (overwritten by _normalize_planned_slots
            # to all prior steps); re-extract from args to catch stale or wrong-namespace refs.
            if slot.tool == return_name:
                step_refs = extract_dependencies(slot.args, placeholder_pattern=self.STEP_REF_PATTERN)
                bad_step_refs = [ref for ref in step_refs if ref < 0 or ref >= i]
                if bad_step_refs:
                    return (
                        f"return step has invalid step references {sorted(set(bad_step_refs))!r} "
                        f"in args; step refs must be earlier steps in the current plan (< {i})."
                    )
            else:
                bad_step_deps = [dep for dep in slot.step_dependencies if dep < 0 or dep >= i]
                if bad_step_deps:
                    return (
                        f"plan step {i} has invalid step dependencies {sorted(set(bad_step_deps))!r}; "
                        f"all deps must be earlier steps (< {i})."
                    )

            if slot.await_step is not NO_VAL and slot.await_step >= i:
                return (
                    f"plan step {i} has an invalid await_step {slot.await_step!r}; "
                    f"await_step must reference an earlier step (< {i})."
                )

            cache_refs = extract_dependencies(slot.args, placeholder_pattern=self.CACHE_REF_PATTERN)

            # Category 1: index outside the cache entirely.
            out_of_range = [idx for idx in cache_refs if idx < 0 or idx >= cache_len]
            if out_of_range:
                return (
                    f"plan step {i} references cache indices that do not exist: "
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
                    f"plan step {i} references cache entries that failed in this "
                    f"conversation and cannot be used: {details}."
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
                    f"plan step {i} references cache indices not part of this "
                    f"conversation: {sorted(set(out_of_conv))!r}."
                )

        limit = self.tool_calls_limit
        if limit is not None:
            non_return = sum(1 for slot in planned_slots if slot.tool != return_name)
            if non_return > limit:
                return (
                    f"plan exceeds the tool_calls_limit of {limit} "
                    f"(planned {non_return} non-return steps). Reduce the number of tool calls."
                )

        return None

    def _process_plan_output(
        self,
        *,
        parsed: Any,
        cache_blackboard: list[BlackboardSlot],
        valid_cache_indices: frozenset[int],
        failed_cache_indices: frozenset[int],
    ) -> list[BlackboardSlot] | str:
        """
        Parse, normalize, and validate a pre-extracted plan value into planned slots.

        Returns ``list[BlackboardSlot]`` on success. Returns a ``str`` feedback
        message on any structural or spec-validation failure; the string is written
        for LLM consumption and is injected as a correction turn on retry.
        """
        if not isinstance(parsed, list) or not parsed:
            return "The plan must be a non-empty JSON array."

        planned_slots: list[BlackboardSlot] = []

        for i, item in enumerate(parsed):
            if not isinstance(item, Mapping):
                return f"Step {i} must be a JSON object (dict), not {type(item).__name__!r}."

            step_dict = self._validate_tool_step_dict(
                item,
                expected_step=i,
                allowed_fields=PLAN_FIELDS,
                required_fields=REQUIRED_PLAN_FIELDS,
                context="plan step",
            )
            if isinstance(step_dict, str):
                return step_dict

            slot = self._tool_step_dict_to_slot(
                step_dict,
                step=i,
                allowed_fields=PLAN_FIELDS,
                context="plan step",
            )
            planned_slots.append(slot)

        normalized = self._normalize_planned_slots(planned_slots)
        if isinstance(normalized, str):
            return normalized

        feedback = self._validate_planned_slots(
            planned_slots=normalized,
            cache_blackboard=cache_blackboard,
            valid_cache_indices=valid_cache_indices,
            failed_cache_indices=failed_cache_indices,
        )
        if feedback is not None:
            return feedback

        return normalized

    # ------------------------------------------------------------------ #
    # Render pipeline
    # ------------------------------------------------------------------ #
    def _render_system_message(self, task: PlanActTask) -> list[dict[str, str]]:
        """Dispatch ``"plan_first"`` locally; delegate everything else
        (i.e. ``"think"``) to ``Agent2``'s own implementation.

        The empty base context is sufficient — ``ToolAgent2._render_system_prompt``
        already injects ``TOOLS``/``CONSTANTS``/``TOOL_CALLS_LIMIT``
        automatically, no manual ``render_context`` construction needed here
        (a real simplification over v1's own ``render_task``).
        """
        if task.system_prompt_name != "plan_first":
            return super()._render_system_message(task)

        rendered = self._render_system_prompt(task, {})
        return [{"role": "system", "content": rendered}]

    def _render_task_messages(self, task: PlanActTask) -> list[dict[str, str]]:
        """Build this phase's task messages, "think" and "plan_first" alike.

        Delegates the "think" branch to ``Agent2``'s shared
        ``_render_thinking_task_messages`` entirely. Owns the "plan_first"
        branch: a banner-wrapped decompose-into-JSON request, preceded by a
        ``PRIOR THINKING`` splice of ``task.thoughts`` when thinking
        actually ran this task. Without this splice, thinking's conclusions
        would be computed and then discarded — the one-shot plan-generation
        call renders from a hard-cleared ``task_messages`` and has no other
        path back to them (unlike ``ReActAgent2``, which re-surfaces
        ``task.thoughts`` into every action-decision round). When
        ``task.thoughts`` is empty (thinking disabled), collapses back to
        v1's original single-message form.
        """
        if task.task_messages:
            return task.task_messages

        if task.system_prompt_name == "think":
            task.task_messages = self._render_thinking_task_messages(task)
            return task.task_messages

        banner = self._render_task_banner(task)
        instruction = (
            "Using the current task above, your prior thinking, and the prior "
            "chat history, construct a valid JSON array that decomposes it "
            "into tool-call steps."
            if task.thoughts
            else (
                "Using the current task above and the prior chat history, "
                "construct a valid JSON array that decomposes it into "
                "tool-call steps."
            )
        )

        if task.thoughts:
            task.task_messages = [
                {"role": "user", "content": banner},
                {
                    "role": "assistant",
                    "content": f"PRIOR THINKING:\n\n{self._format_thoughts(task.thoughts)}",
                },
                {"role": "user", "content": instruction},
            ]
        else:
            task.task_messages = [{
                "role": "user",
                "content": f"{banner}\n\n{instruction}",
            }]
        return task.task_messages

    # ------------------------------------------------------------------ #
    # Plan generation
    # ------------------------------------------------------------------ #
    def _generate_plan(self, *, task: PlanActTask) -> list[BlackboardSlot]:
        """
        Generate, parse, and validate a complete PlanAct running blackboard, with
        optional retry on generation failures.

        Lifecycle
        ---------
        1. ``additional_messages`` starts empty.
        2. Loop:
           a. Render this attempt's send payload via ``render_task``.
           b. Call the LLM engine; capture ``engine_result``.
           c. Append an ``LLMRecord`` (``messages=list(task.task_messages)``,
              which ``render_task`` has already extended for this attempt)
              to ``task.llm_records``.
           d. Try JSON extraction (``_extract_from_json_string``). On
              failure: check ``task.retries_used`` against
              ``self._generation_retries``; if exhausted raise; else inject
              JSON-error feedback as ``additional_messages`` for the next
              attempt, increment ``task.retries_used``, continue.
           e. Try spec validation (``_process_plan_output``), passing
              ``cache_blackboard=self._blackboard`` (the live, persisted
              blackboard — no per-task snapshot exists anymore). On
              failure: same budget check/feedback-injection pattern as (d).
           f. On success: return the validated planned slots.

        Parameters
        ----------
        task : PlanActTask
            Current task — supplies ``valid_cache_indices``/
            ``failed_cache_indices`` for validation and accumulates
            ``llm_records``/``retries_used`` directly.

        Returns
        -------
        list[BlackboardSlot]
            Fully normalized and validated planned slots (not yet assigned
            onto ``task`` — the caller finalizes that inline).

        Raises
        ------
        ToolAgentError
            If generation output cannot be parsed or validated after all
            allowed attempts are exhausted.
        """
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
                    "Produce a correctly formatted JSON array."
                )
                if task.retries_used >= self._generation_retries:
                    raise ToolAgentError(
                        f"{type(self).__name__}.{self.name}: generation retry budget exhausted "
                        f"after {task.retries_used + 1} attempt(s). Last error is JSONDecodeError: {exc}"
                    )
                additional_messages = [
                    {"role": "assistant", "content": raw_output},
                    {"role": "user", "content": feedback},
                ]
                task.retries_used += 1
                continue

            # Spec validation
            result = self._process_plan_output(
                parsed=parsed,
                cache_blackboard=self._blackboard,
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
                plan_repr = json.dumps(parsed, indent=2)
                additional_messages = [
                    {"role": "assistant", "content": plan_repr},
                    {"role": "user", "content": feedback},
                ]
                task.retries_used += 1
                continue

            return result

    async def _agenerate_plan(self, *, task: PlanActTask) -> list[BlackboardSlot]:
        """Async mirror of ``_generate_plan``: uses ``async_invoke`` for each LLM call.
        Retry logic, feedback injection, and return type are identical."""
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
                if task.retries_used >= self._generation_retries:
                    raise ToolAgentError(
                        f"{type(self).__name__}.{self.name}: generation retry budget exhausted "
                        f"after {task.retries_used + 1} attempt(s). Last error is JSONDecodeError: {exc}"
                    )
                additional_messages = [
                    {"role": "assistant", "content": raw_output},
                    {"role": "user", "content": feedback},
                ]
                task.retries_used += 1
                continue

            # Spec validation
            result = self._process_plan_output(
                parsed=parsed,
                cache_blackboard=self._blackboard,
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
                plan_repr = json.dumps(parsed, indent=2)
                additional_messages = [
                    {"role": "assistant", "content": plan_repr},
                    {"role": "user", "content": feedback},
                ]
                task.retries_used += 1
                continue

            return result

    # ------------------------------------------------------------------ #
    # Task-lifecycle hooks
    # ------------------------------------------------------------------ #
    def _initialize_task(
        self,
        *,
        turns: list[AgentRecord],
        prompt: str,
        inputs: dict,
    ) -> PlanActTask:
        """
        Build this invocation's PlanActTask. No plan generation here
        anymore — thinking (if enabled) must conclude first, so plan
        generation moves to the first ``_prepare_next_batch``/
        ``_aprepare_next_batch`` call instead.

        1. Compute ``valid_cache_indices``/``failed_cache_indices`` via
           ``_compute_cache_index_sets(turns)``.
        2. Return a bare ``PlanActTask`` seeded with
           ``system_prompt_name="think"`` (the only phase base ``Agent2``
           itself renders) and ``keep_thinking = self._thinking_rounds != 0``.
           ``batches``/``running_blackboard``/``batch_index`` stay at their
           dataclass defaults (``[]``, ``[]``, ``0``) until the plan is
           generated.
        """
        valid_cache_indices, failed_cache_indices = self._compute_cache_index_sets(turns)
        task = PlanActTask(
            turns=turns,
            inputs=inputs,
            user_prompt=prompt,
            system_prompt_name="think",
            valid_cache_indices=valid_cache_indices,
            failed_cache_indices=failed_cache_indices,
        )
        task.keep_thinking = self._thinking_rounds != 0
        return task

    def _prepare_next_batch(self, task: PlanActTask) -> PlanActTask:
        """
        Generate the plan on first entry (once thinking has concluded),
        then prepare the next pre-compiled batch.

        1. If ``task.batches`` is empty (plan not yet generated):
           stamp ``system_prompt_name="plan_first"``, clear
           ``task_messages``, call ``_generate_plan``, assign
           ``task.running_blackboard``/``task.batches``
           (``_compile_batches_from_deps``)/``task.batch_index=0`` from the
           result, clear ``task_messages`` again (matches v1's own
           ``_initialize_task`` tail).
        2. Delegate to ``_prepare_compiled_batch`` unconditionally — reads
           the batch at ``task.batch_index`` regardless of whether it was
           just compiled this call or already existed from an earlier one.

        No ``if task.prepared_steps: raise`` guard — structurally
        unreachable given ``execute()`` always clears ``prepared_steps``
        before the next ``prepare()`` call.
        """
        if not task.batches:
            task.system_prompt_name = "plan_first"
            task.task_messages = []

            planned_slots = self._generate_plan(task=task)
            task.running_blackboard = planned_slots
            task.batches = self._compile_batches_from_deps(
                planned_slots=planned_slots,
                return_idx=len(planned_slots) - 1,
            )
            task.batch_index = 0

            task.task_messages.clear()

        return self._prepare_compiled_batch(task)

    async def _aprepare_next_batch(self, task: PlanActTask) -> PlanActTask:
        """Async mirror of ``_prepare_next_batch``: uses ``_agenerate_plan``
        for the plan-generation LLM call rather than the inherited
        ``asyncio.to_thread`` default — thread-offloading a sync LLM call
        would be a quality regression, exactly what v1's own
        ``_async_initialize_task`` override existed to avoid; that concern
        moves with the generation call, not away from it."""
        if not task.batches:
            task.system_prompt_name = "plan_first"
            task.task_messages = []

            planned_slots = await self._agenerate_plan(task=task)
            task.running_blackboard = planned_slots
            task.batches = self._compile_batches_from_deps(
                planned_slots=planned_slots,
                return_idx=len(planned_slots) - 1,
            )
            task.batch_index = 0

            task.task_messages.clear()

        return self._prepare_compiled_batch(task)

    def _prepare_compiled_batch(self, task: PlanActTask) -> PlanActTask:
        """
        Prepare the next pre-compiled batch for execution.

        Shared tail for both ``_prepare_next_batch``/``_aprepare_next_batch``,
        once ``task.batches`` is guaranteed non-empty. Reads the next batch
        indices, resolves placeholders, marks those slots prepared, and
        populates ``prepared_steps``.

        Execution
        ~~~~~~~~~
        1. **Bounds check**: ``batch_index`` must be within ``task.batches``.
        2. **Read next batch**: ``task.batches[task.batch_index]``.
        3. **Validate non-empty**: batch must have at least one step
           (internal-error guard on the compiler's own output).
        4. **For each step in batch**:
           - Validate bounds, slot-step match, not already
             executed/prepared/failed, is planned, tool name is set.
           - **Cascade check** (``fail_fast=False`` only): if any
             ``step_dependencies`` entry is FAILED, the return tool raises
             immediately; non-return steps are marked FAILED and skipped
             (not added to ``prepared_steps``).
           - Resolve placeholders via ``_resolve_placeholders``; store in
             ``slot.resolved_args``; mark ``status=PREPARED``.
        5. **Set prepared_steps**: indices that passed the cascade check
           and were prepared; may be empty if every step in the batch
           cascade-failed.
        6. **Advance cursor**: increment ``task.batch_index``.

        Concurrency
        ~~~~~~~~~~~
        All steps in the batch can execute concurrently since the
        topological sort guarantees no step in a batch depends on another
        step in the same batch.

        Raises
        ------
        ToolAgentError
            On batch_index out of bounds, batch validation failure, or
            placeholder resolution failure.
        """
        if task.batch_index >= len(task.batches):
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: no remaining batches to prepare (batch_index={task.batch_index})."
            )

        batch = task.batches[task.batch_index]
        if not batch:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: internal error: encountered empty batch at index {task.batch_index}."
            )

        board = task.running_blackboard
        board_len = len(board)

        # Resolve args for all steps in the batch; resolver enforces readiness.
        prepared_in_batch: list[int] = []  # excludes cascade-failed steps; may yield empty prepared_steps
        for i in batch:
            if not isinstance(i, int):
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: batch contains non-int index: {i!r}."
                )
            if i < 0 or i >= board_len:
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: batch index {i} out of range (plan length={board_len})."
                )

            slot = board[i]

            # Invariant: slot.step is plan-local index.
            if slot.step != i:
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: running slot step mismatch at index {i}: slot.step={slot.step}."
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
                    f"{type(self).__name__}.{self.name}: step {i} has invalid tool name in running_blackboard."
                )

            # Cascade-fail: when fail_fast=False, propagate failures through arg dependencies.
            if not self._fail_fast and self._check_cascade_failure(slot, board):
                continue

            # Resolve placeholders using base resolver (checks cache + executed step readiness).
            slot.resolved_args = self._resolve_placeholders(slot.args, task=task)

            if slot.result is not NO_VAL:
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: internal error: step {i} result was set during preparation."
                )

            slot.error = NO_VAL
            slot.status = BlackboardSlot.PREPARED
            prepared_in_batch.append(i)

        task.prepared_steps = sorted(prepared_in_batch)  # empty when all steps cascade-failed
        task.batch_index += 1
        logger.info(
            f"{self.full_name}: Prepared batch {task.batch_index}/{len(task.batches)} "
            f"with steps {task.prepared_steps}."
        )
        return task
