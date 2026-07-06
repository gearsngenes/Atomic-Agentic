"""
PlanActAgent: One-Shot LLM Planner with Concurrent Batch Execution

This module provides ``PlanActAgent``, a concrete ``ToolAgent`` subclass that
implements a **static planning** strategy: the LLM is queried once per invoke
to produce a complete JSON plan, which is then compiled into topologically-sorted
concurrent batches and executed without further LLM interaction.

Planning Model
--------------
A single LLM call at the start of each ``invoke()`` emits the full plan as a
JSON array of tool-call steps. Each step specifies a tool name, argument map
(optionally containing ``<<__sN__>>`` step-ref or ``<<__cN__>>`` cache-ref
placeholders), an optional ``await`` scheduling barrier, and an optional
``return`` terminator.

Compilation
-----------
After the plan is normalized and validated, a dependency graph is derived from
``<<__sN__>>`` placeholder references and explicit ``await`` fields. A topological
level-assignment produces concurrent batches: all steps at the same dependency
level execute together.

Execution
---------
The base ``ToolAgent`` template-method loop calls ``_prepare_next_batch`` once
per batch to resolve placeholders and mark steps prepared; the loop executes
each prepared batch concurrently and then advances the cursor.

Contrast
--------
For adaptive, step-by-step iteration see ``agents/react.py`` (``ReActAgent``).
For the shared iteration loop, blackboard management, and tool registry see
``agents/toolagent.py`` (``ToolAgent``).
"""

from __future__ import annotations
import json
from typing import Any, Callable, Mapping, Optional

import logging

from .toolagent import ToolAgent
from .prompts import PLANNER_PROMPT
from ..constants.agents import (
    RETURN_TOOL_FULL_NAME,
    RETURN_VALUE_FIELD,
    PLAN_FIELDS,
    REQUIRED_PLAN_FIELDS,
)
from ..core import AtomicInvokable
from ..constants.core import NO_VAL
from ..engines.LLMEngines import LLMEngine
from ..exceptions import ToolAgentError
from ..models.agents import PlanActRunState
from ..models.agents import BlackboardSlot
from ..models.agents.records import AgentRecord, LLMRecord
from ..models.agents.prompts import PromptConfig
from ..models.parameters import ParamSpec
from ..utils.agents import extract_dependencies

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# PlanAct Agent
# --------------------------------------------------------------------------- #
class PlanActAgent(ToolAgent):
    """
    One-shot planner agent: generates entire plan upfront, executes in batches.

    **Design**: PlanActAgent implements a **static planning** strategy:

    1. **Initialization** (``_initialize_run_state``)
       - LLM generates complete plan as a JSON array of steps (one-shot)
       - Each step: ``{"tool": "<name>", "args": {...}}``, optionally with "await"
       - Plan is normalized (return moved to end, added if missing)
       - Compiled into topologically-sorted batches using dependency analysis
       - Running blackboard allocated with slots for all planned steps

    2. **Compilation**
       - Each step's args are scanned for ``<<__sN__>>`` placeholders to extract
         plan-local dependencies
       - Topological sort produces concurrent batches (steps with identical dependency
         level execute together)
       - Return step is always isolated as the final batch

    3. **Execution**
       - ``_prepare_next_batch()`` reads next batch from ``state.batches[state.batch_index]``
       - Resolves placeholders in parallel-executable steps
       - Increments batch_index; loop continues until all batches consumed

    Advantages
    ~~~~~~~~~~
    - **No replanning**: Full plan is known upfront; no latency per iteration
    - **Concurrency-friendly**: Topological compilation enables maximal parallelism
    - **Deterministic**: Same inputs produce identical execution plan every time

    Limitations
    ~~~~~~~~~~~
    - **No adaptivity**: Cannot branch based on intermediate results
    - **Plan quality**: Entirely dependent on LLM's single planning turn
    - **Error recovery**: If a step fails, entire plan fails (no dynamic replanning)

    Parameters (construction)
    ~~~~~~~~~~~~~~~~~~~~~~~~
    Same as ToolAgent, with ``tool_calls_limit`` being the max non-return steps
    in any single plan.
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
        context_properties: list[str] | list[ParamSpec] | None = None,
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
    ) -> None:
        """
        Initialize a PlanActAgent.

        ``"plan_first"`` is the key under which the built-in planning prompt is
        registered in ``self._system_prompts``. All other parameters are
        forwarded verbatim to ``ToolAgent.__init__``.
        """
        super().__init__(
            name=name,
            namespace=namespace,
            description=description,
            llm_engine=llm_engine,
            filter_extraneous_inputs=filter_extraneous_inputs,
            context_enabled=context_enabled,
            context_properties=context_properties,
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
        self._system_prompts["plan_first"] = PLANNER_PROMPT

    # ------------------------------------------------------------------ #
    # Prompt update guard
    # ------------------------------------------------------------------ #
    def update_prompt(self, key: str, config: PromptConfig) -> None:
        """Guard the built-in planning instruction prompt.

        Raises ``ToolAgentError`` when ``key`` is ``'plan_first'`` — that prompt
        is operational machinery and must not be replaced post-construction.
        All other keys are forwarded to the base implementation.
        """
        if key.strip() == "plan_first":
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: "
                "'plan_first' is the built-in planning instruction prompt and cannot "
                "be replaced via update_prompt."
            )
        super().update_prompt(key, config)

    # ------------------------------------------------------------------ #
    # Initialization
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
            Runtime snapshot of persisted cache entries, used for cache-reference
            range validation.

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

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #
    def _setup_plan_init(
        self,
        *,
        messages: list[dict[str, str]],
    ) -> tuple[list[dict[str, str]], list[BlackboardSlot]]:
        """Validate messages, copy to working list, snapshot cache blackboard."""
        if not messages:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: messages must be non-empty."
            )
        working_messages = [dict(m) for m in messages]
        cache_blackboard: list[BlackboardSlot] = (
            [slot.copy() for slot in self._blackboard] if self.context_enabled else []
        )
        return working_messages, cache_blackboard

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

        Returns ``list[BlackboardSlot]`` on success.  Returns a ``str`` feedback
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

    def _build_planact_run_state(
        self,
        *,
        planned_slots: list[BlackboardSlot],
        working_messages: list[dict[str, str]],
        cache_blackboard: list[BlackboardSlot],
        llm_records: list[LLMRecord],
        valid_cache_indices: frozenset[int],
        failed_cache_indices: frozenset[int],
    ) -> PlanActRunState:
        """Validate plan structure, compile batches, and construct PlanActRunState."""
        if not planned_slots:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: internal error: generated plan is empty."
            )

        return_idx = len(planned_slots) - 1
        if planned_slots[return_idx].tool != RETURN_TOOL_FULL_NAME:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: internal error: generated plan does not end with return tool."
            )

        batches = self._compile_batches_from_deps(
            planned_slots=planned_slots,
            return_idx=return_idx,
        )

        return PlanActRunState(
            messages=working_messages,
            cache_blackboard=cache_blackboard,
            running_blackboard=planned_slots,
            executed_steps=set(),
            prepared_steps=[],
            tool_calls_used=0,
            llm_records=list(llm_records),
            valid_cache_indices=valid_cache_indices,
            failed_cache_indices=failed_cache_indices,
            is_done=False,
            return_value=NO_VAL,
            batches=batches,
            batch_index=0,
        )

    def _generate_plan(
        self,
        *,
        messages: list[dict[str, str]],
        cache_blackboard: list[BlackboardSlot],
        valid_cache_indices: frozenset[int],
        failed_cache_indices: frozenset[int],
    ) -> tuple[list[BlackboardSlot], list[LLMRecord]]:
        """
        Generate, parse, and validate a complete PlanAct running blackboard, with
        optional retry on generation failures.

        Lifecycle
        ---------
        1. Validate that ``messages`` is non-empty.
        2. Initialize ``working_messages`` as a mutable copy of ``messages``; initialize
           ``llm_records`` list and ``retries_used`` counter.
        3. Loop:
           a. Call the LLM engine with ``working_messages``; capture ``engine_result``.
           b. Construct an ``LLMRecord`` from the last working message and
              ``engine_result``; append to ``llm_records``.
           c. Try JSON extraction (``_extract_from_json_string``). On failure: check
              budget; if exhausted re-raise; else inject JSON-error feedback and
              continue.
           d. Try spec validation (``_process_plan_output``). On failure: check budget;
              if exhausted re-raise; else inject spec-error feedback and continue.
           e. On success: return ``(planned_slots, llm_records)``.

        Parameters
        ----------
        messages : list[dict[str, str]]
            LLM-facing messages already built by the base Agent message pipeline.
        cache_blackboard : list[BlackboardSlot]
            Snapshot of persisted blackboard entries available to this invoke.

        Returns
        -------
        tuple[list[BlackboardSlot], list[LLMRecord]]
            Fully normalized and validated planned slots, plus one ``LLMRecord`` per
            generation attempt (including failed attempts).

        Raises
        ------
        ToolAgentError
            If generation output cannot be parsed or validated after all allowed
            attempts are exhausted, or if ``messages`` is empty.
        """
        if not messages:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: messages must be non-empty."
            )

        working_messages: list[dict[str, str]] = list(messages)
        llm_records: list[LLMRecord] = []
        retries_used: int = 0

        while True:
            engine_result = self._llm_engine.invoke({"messages": [dict(m) for m in working_messages]})
            raw_output: str = engine_result.result

            # PlanAct delta: the last working message (the current user turn) only.
            # ReAct uses a 3-element delta (task + snapshot + step-request); PlanAct
            # is simpler because the system prompt already carries planning context.
            llm_records.append(LLMRecord(
                messages=[working_messages[-1]],
                llm_result=engine_result,
                system_prompt_name="plan_first",
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

            # Spec validation
            result = self._process_plan_output(
                parsed=parsed,
                cache_blackboard=cache_blackboard,
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
                plan_repr = json.dumps(parsed, indent=2)
                working_messages.append({"role": "assistant", "content": plan_repr})
                working_messages.append({"role": "user", "content": feedback})
                retries_used += 1
                continue

            return result, llm_records

    async def _agenerate_plan(
        self,
        *,
        messages: list[dict[str, str]],
        cache_blackboard: list[BlackboardSlot],
        valid_cache_indices: frozenset[int],
        failed_cache_indices: frozenset[int],
    ) -> tuple[list[BlackboardSlot], list[LLMRecord]]:
        """Async mirror of ``_generate_plan``: uses ``async_invoke`` for each LLM call.
        Retry logic, feedback injection, and return type are identical."""
        if not messages:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: messages must be non-empty."
            )

        working_messages: list[dict[str, str]] = list(messages)
        llm_records: list[LLMRecord] = []
        retries_used: int = 0

        while True:
            engine_result = await self._llm_engine.async_invoke({"messages": [dict(m) for m in working_messages]})
            raw_output: str = engine_result.result

            llm_records.append(LLMRecord(
                messages=[working_messages[-1]],
                llm_result=engine_result,
                system_prompt_name="plan_first",
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

            # Spec validation
            result = self._process_plan_output(
                parsed=parsed,
                cache_blackboard=cache_blackboard,
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
                plan_repr = json.dumps(parsed, indent=2)
                working_messages.append({"role": "assistant", "content": plan_repr})
                working_messages.append({"role": "user", "content": feedback})
                retries_used += 1
                continue

            return result, llm_records

    def _initialize_run_state(
        self,
        *,
        turns: list[AgentRecord],
        prompt: str,
        valid_cache_indices: frozenset[int],
        failed_cache_indices: frozenset[int],
    ) -> PlanActRunState:
        """
        One-shot plan generation and compilation into concurrent batches.

        Renders the planning system prompt from instance state, builds the full
        message list, then delegates plan generation/normalization/validation to
        ``_generate_plan(...)`` and compiles the resulting planned slots into
        topologically-sorted concurrent batches.

        Execution Steps
        ~~~~~~~~~~~~~~~
        1. Render ``self._system_prompts["plan_first"]`` with TOOLS/LIMIT/CONSTANTS
           assembled from instance state; call ``build_messages`` to produce messages.
        2. ``_setup_plan_init``: validate messages; copy to working list; snapshot
           cache blackboard.
        3. ``_generate_plan``: generate the plan via LLM, returning
           ``(planned_slots, llm_records)``.
        4. ``_build_planact_run_state``: compile batches and construct the
           ``PlanActRunState`` with the generated slots and LLM records.

        Returns
        -------
        PlanActRunState
            Initialized state ready for the base template-method loop.

        Raises
        ------
        ToolAgentError
            On any of: empty messages, empty or invalid plan, multiple return steps,
            unknown tool references, out-of-range placeholder references, invalid plan
            dependencies, or budget exceeded.
        """
        limit_text = "unlimited" if self._tool_calls_limit is None else str(self._tool_calls_limit)
        render_context = {
            ToolAgent.TOOLS_FIELD: self.actions_context(),
            ToolAgent.LIMIT_FIELD: limit_text,
            ToolAgent.CONSTANTS_FIELD: self.constants_context(),
        }
        system = self._system_prompts["plan_first"].render(render_context)
        messages = self.build_messages(system, turns, prompt)
        working_messages, cache_blackboard = self._setup_plan_init(messages=messages)
        planned_slots, llm_records = self._generate_plan(
            messages=working_messages,
            cache_blackboard=cache_blackboard,
            valid_cache_indices=valid_cache_indices,
            failed_cache_indices=failed_cache_indices,
        )
        return self._build_planact_run_state(
            planned_slots=planned_slots,
            working_messages=working_messages,
            cache_blackboard=cache_blackboard,
            llm_records=llm_records,
            valid_cache_indices=valid_cache_indices,
            failed_cache_indices=failed_cache_indices,
        )

    async def _ainitialize_run_state(
        self,
        *,
        turns: list[AgentRecord],
        prompt: str,
        valid_cache_indices: frozenset[int],
        failed_cache_indices: frozenset[int],
    ) -> PlanActRunState:
        """
        Async override: uses ``_agenerate_plan`` so the planning LLM call
        goes through ``async_invoke`` rather than a worker thread.
        """
        limit_text = "unlimited" if self._tool_calls_limit is None else str(self._tool_calls_limit)
        render_context = {
            ToolAgent.TOOLS_FIELD: self.actions_context(),
            ToolAgent.LIMIT_FIELD: limit_text,
            ToolAgent.CONSTANTS_FIELD: self.constants_context(),
        }
        system = self._system_prompts["plan_first"].render(render_context)
        messages = self.build_messages(system, turns, prompt)
        working_messages, cache_blackboard = self._setup_plan_init(messages=messages)
        planned_slots, llm_records = await self._agenerate_plan(
            messages=working_messages,
            cache_blackboard=cache_blackboard,
            valid_cache_indices=valid_cache_indices,
            failed_cache_indices=failed_cache_indices,
        )
        return self._build_planact_run_state(
            planned_slots=planned_slots,
            working_messages=working_messages,
            cache_blackboard=cache_blackboard,
            llm_records=llm_records,
            valid_cache_indices=valid_cache_indices,
            failed_cache_indices=failed_cache_indices,
        )

    def _compile_batches_from_deps(
        self,
        *,
        planned_slots: list[BlackboardSlot],
        return_idx: int,
    ) -> list[list[int]]:
        """
        Compile concurrent batches from plan-local scheduling dependencies.

        For non-return step i:
          scheduling_deps[i] = step_dependencies + await_step if present
          level[i] = 0 if scheduling_deps are empty else
          1 + max(level[d] for d in scheduling_deps)

        step_dependencies represent data dependencies extracted from <<__sN__>>
        placeholders. await_step is an explicit scheduling barrier and is folded into
        dependencies only locally while compiling execution batches.

        Return step is always isolated as its own final batch [return_idx].
        """
        if not planned_slots:
            raise ToolAgentError(f"{type(self).__name__}.{self.name}: cannot compile empty plan.")

        if (
            return_idx != len(planned_slots) - 1
            or planned_slots[return_idx].tool != RETURN_TOOL_FULL_NAME
        ):
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: internal error: return_idx mismatch during batch compilation."
            )

        levels: dict[int, int] = {}

        # Non-return only.
        for i in range(return_idx):
            slot = planned_slots[i]

            scheduling_deps: set[int] = set(slot.step_dependencies)
            if slot.await_step is not NO_VAL:
                scheduling_deps.add(slot.await_step)

            if not scheduling_deps:
                levels[i] = 0
            else:
                levels[i] = 1 + max(levels[d] for d in scheduling_deps)

        buckets: dict[int, list[int]] = {}
        for i in range(return_idx):
            lvl = levels.get(i, 0)
            buckets.setdefault(lvl, []).append(i)

        batches: list[list[int]] = []
        for lvl in sorted(buckets):
            batch = sorted(buckets[lvl])
            if batch:
                batches.append(batch)

        batches.append([return_idx])
        return batches

    # ------------------------------------------------------------------ #
    # Prepare next batch
    # ------------------------------------------------------------------ #
    def _prepare_next_batch(self, state: PlanActRunState) -> PlanActRunState:
        """
        Prepare the next pre-compiled batch for execution.

        PlanActAgent uses pre-compiled batches created during initialization. This method
        reads the next batch indices, resolves placeholders, marks those slots prepared,
        and populates the prepared_steps list.

        Execution
        ~~~~~~~~~
        1. **Validate state**: prepared_steps must be empty
        2. **Read next batch**: Get batch indices from ``state.batches[state.batch_index]``
        3. **Validate non-empty**: Batch must have at least one step
        4. **For each step in batch**:
           - Validate bounds: index must be within running_blackboard
           - Validate not already executed or prepared
           - Validate slot is currently planned
           - Validate tool name is set
           - **Cascade check** (``fail_fast=False`` only): if any ``step_dependencies``
             entry is FAILED in the running blackboard, the return tool raises immediately;
             non-return steps are marked FAILED and skipped (not added to prepared_steps)
           - Call ``_resolve_placeholders(slot.args, state=state)``
           - Store resolved args in ``slot.resolved_args``
           - Mark slot ``status="prepared"``
        5. **Set prepared_steps**: Indices of steps that passed the cascade check and were
           prepared; may be empty if all steps in the batch were cascade-failed
        6. **Advance cursor**: Increment ``state.batch_index`` for next iteration

        Concurrency
        ~~~~~~~~~~~
        All steps in the batch can execute concurrently since the topological sort
        guarantee ensures no step in a batch depends on another step in the same batch.

        Parameters
        ----------
        state : PlanActRunState
            Current run state with initialized batches and batch_index cursor

        Returns
        -------
        PlanActRunState
            Updated state with prepared_steps populated, batch_index incremented

        Raises
        ------
        ToolAgentError
            On any of:
            - prepared_steps not empty
            - batch_index out of bounds
            - Batch validation failure
            - Placeholder resolution failure
        """
        if state.prepared_steps:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: cannot prepare next batch while prepared_steps is non-empty."
            )

        if state.batch_index >= len(state.batches):
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: no remaining batches to prepare (batch_index={state.batch_index})."
            )

        batch = state.batches[state.batch_index]
        if not batch:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: internal error: encountered empty batch at index {state.batch_index}."
            )

        board = state.running_blackboard
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
            # Use extract_dependencies(slot.args) rather than slot.step_dependencies because
            # the return slot's step_dependencies is forced to include ALL prior steps for
            # scheduling; only steps actually referenced in args need to resolve successfully.
            if not self._fail_fast:
                failed_arg_deps = sorted(
                    d
                    for d in extract_dependencies(slot.args, placeholder_pattern=self.STEP_REF_PATTERN)
                    if board[d].is_failed()
                )
                if failed_arg_deps:
                    dep_str = ", ".join(str(d) for d in failed_arg_deps)
                    if slot.tool == RETURN_TOOL_FULL_NAME:
                        raise ToolAgentError(
                            f"{type(self).__name__}.{self.name}: return step {i} cannot execute; "
                            f"dependency step(s) {dep_str} failed."
                        )
                    slot.error = ToolAgentError(
                        f"{type(self).__name__}.{self.name}: step {i} skipped — "
                        f"dependency step(s) {dep_str} failed."
                    )
                    slot.status = BlackboardSlot.FAILED
                    continue

            # Resolve placeholders using base resolver (checks cache + executed step readiness).
            slot.resolved_args = self._resolve_placeholders(slot.args, state=state)

            if slot.result is not NO_VAL:
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: internal error: step {i} result was set during preparation."
                )

            slot.error = NO_VAL
            slot.status = BlackboardSlot.PREPARED
            prepared_in_batch.append(i)

        state.prepared_steps = sorted(prepared_in_batch)  # empty when all steps cascade-failed
        state.batch_index += 1
        logger.info(
            f"{self.full_name}: Prepared batch {state.batch_index}/{len(state.batches)} "
            f"with steps {state.prepared_steps}."
        )
        return state
