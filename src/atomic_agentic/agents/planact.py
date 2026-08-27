"""
PlanActAgent: One-Shot LLM Planner with Concurrent Batch Execution

This module provides ``PlanActAgent``, a concrete ``ToolAgent`` subclass that
implements a **static planning** strategy: the LLM is queried once per invoke
to produce a complete JSON plan, which is then compiled into topologically-sorted
concurrent batches and executed without further LLM interaction.

Planning Model
--------------
A single LLM call at the start of each ``invoke()`` emits the full plan,
either as a JSON array of tool-call steps (``generation_format="json"``,
the default) or as a sequence of ``[CALL]``/``[RETURN]`` tag blocks
(``generation_format="regex"``) -- a construction-time, fixed-topology
choice (see ``ToolAgent.__init__``). Each step specifies a tool name,
argument map (optionally containing ``|STEP.N|`` step-ref or ``|CACHE.N|``
cache-ref placeholders), an optional ``await`` scheduling barrier, and an
optional ``return`` terminator.

Compilation
-----------
After the plan is normalized and validated, a dependency graph is derived from
``|STEP.N|`` placeholder references and explicit ``await`` fields. A topological
level-assignment produces concurrent batches: all steps at the same dependency
level execute together.

Execution
---------
Plan generation and validation happen once, in ``think()``, gated on
``task.generated_plan``. Compilation into concurrent batches happens once,
on ``prepare()``'s first call, gated on ``task.batches``. Every round after
that, ``prepare()`` resolves placeholders for the next compiled batch and
marks its steps prepared; ``act()`` (base ``ToolAgent``, final) executes
the batch concurrently and advances the cursor.

Contrast
--------
For adaptive, step-by-step iteration see ``agents/react.py`` (``ReActAgent``).
For the shared iteration loop, blackboard management, and tool registry see
``agents/toolagent.py`` (``ToolAgent``).
"""

from __future__ import annotations
from typing import Any, Callable, Literal, Mapping, Optional

import json
import logging

from .toolagent import ToolAgent
from .prompts import PLANNER_PROMPT, REGEX_PLANNER_PROMPT
from ..constants.agents import (
    RETURN_TOOL_FULL_NAME,
    RETURN_VALUE_FIELD,
    RETURN_TOOL_REASON_TEXT,
    PLAN_FIELDS,
    REQUIRED_PLAN_FIELDS,
    REGEX_PLAN_FIELDS,
    REQUIRED_REGEX_PLAN_FIELDS,
    REASON_FIELD,
)
from ..core import AtomicInvokable
from ..constants.core import NO_VAL
from ..llm.base import LLMEngine
from ..exceptions import ToolAgentError
from ..models.agents.tasks import PlanActTask
from ..models.agents import BlackboardSlot
from ..models.agents.records import AgentRecord
from ..utils.agents import extract_dependencies, format_generation_issues

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# PlanAct Agent
# --------------------------------------------------------------------------- #
class PlanActAgent(ToolAgent):
    """
    One-shot planner agent: generates entire plan upfront, executes in batches.

    **Design**: PlanActAgent implements a **static planning** strategy:

    1. **Planning** (``think()``, once — no-op on later rounds)
       - LLM generates complete plan as a JSON array of steps (one-shot)
       - Each step: ``{"tool": "<name>", "args": {...}}``, optionally with "await"
       - Plan is normalized (return moved to end, added if missing) and validated
       - Validated plan stored on ``task.generated_plan``

    2. **Compilation** (``prepare()``, first call only)
       - Each step's args are scanned for ``|STEP.N|`` placeholders to extract
         plan-local dependencies
       - Topological sort produces concurrent batches (steps with identical dependency
         level execute together)
       - Return step is always isolated as the final batch
       - ``running_blackboard``/``batches``/``batch_index`` assigned onto the task

    3. **Execution** (``prepare()``/``act()``, every round)
       - ``prepare()`` reads next batch from ``task.batches[task.batch_index]``
       - Resolves placeholders in parallel-executable steps
       - ``act()`` (base ``ToolAgent``, final) runs the batch concurrently;
         sets ``task.complete`` when the return step executes
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
        context_enabled: bool = False,
        *,
        tool_calls_limit: int | None = None,
        fail_fast: bool = True,
        generation_retries: int = 0,
        generation_format: Literal["json", "regex"] = "json",
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
        Initialize a PlanActAgent.

        ``"plan_first"`` is the key under which the built-in planning prompt is
        registered in ``self._system_prompts``. All other parameters are
        forwarded verbatim to ``ToolAgent.__init__`` — no extra_parameters
        keyword is passed at all (``ToolAgent.__init__`` accepts none).
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
            generation_format=generation_format,
            peek_at_cache=peek_at_cache,
            response_preview_limit=response_preview_limit,
            blackboard_preview_limit=blackboard_preview_limit,
            pre_invoke=pre_invoke,
            post_invoke=post_invoke,
            post_result_key=post_result_key,
            records_window=records_window,
            assistant_response_source=assistant_response_source,
        )
        self._system_prompts["plan_first"] = PLANNER_PROMPT
        self._system_prompts["plan_first_regex"] = REGEX_PLANNER_PROMPT

    # ------------------------------------------------------------------ #
    # Initialization
    # ------------------------------------------------------------------ #
    def _normalize_planned_slots(
        self,
        planned_slots: list[BlackboardSlot],
    ) -> list[BlackboardSlot]:
        """
        Normalize a generated PlanAct slot list into final running-blackboard order.

        Normalization policy
        --------------------
        - At most one return slot may be present -- guaranteed by the caller
          (``_validate_generation_output``'s tier-1 loop) before this method
          is ever invoked; reaching more than one here is an internal-contract
          violation, not an LLM-facing problem.
        - If a return slot is present, it is moved to the end.
        - If no return slot is present, `return(None)` is appended, stamped
          with the same synthesized ``reason`` a desugared ``[RETURN]`` block
          gets, for consistency.
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

        Raises
        ------
        ToolAgentError
            If more than one return slot is present -- ``_validate_generation_output``
            must catch and report this to the LLM before normalization runs.
        """
        slots: list[BlackboardSlot] = [slot.copy() for slot in planned_slots]

        return_name = RETURN_TOOL_FULL_NAME
        return_positions = [
            i for i, slot in enumerate(slots)
            if slot.tool == return_name
        ]

        if len(return_positions) > 1:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: internal error: "
                f"_normalize_planned_slots received a plan with multiple "
                f"return steps at positions {return_positions!r}; "
                "_validate_generation_output's tier-1 loop must reject this "
                "before normalization runs."
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
                    reason=RETURN_TOOL_REASON_TEXT,
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
    ) -> list[str]:
        """
        Validate a normalized PlanAct planned-slot list.

        Checks only properties that ``_normalize_planned_slots`` does not
        guarantee: tool existence, step dependency graph, await_step ordering,
        tool_calls_limit, and cache reference validity. Every applicable
        check, across every slot, is evaluated independently and
        accumulated -- none short-circuits the rest.

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
        list[str]
            LLM-facing feedback strings, one per invariant violation found.
            Empty when every invariant is satisfied.

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
        issues: list[str] = []

        for i, slot in enumerate(planned_slots):
            if not self.has_tool(slot.tool):
                issues.append(
                    f"plan step {i} uses an unknown tool: {slot.tool!r}. "
                    "Use only tools from the available list."
                )

            # Return slot step_dependencies are synthetic (overwritten by _normalize_planned_slots
            # to all prior steps); re-extract from args to catch stale or wrong-namespace refs.
            if slot.tool == return_name:
                step_refs = extract_dependencies(slot.args, placeholder_pattern=self.STEP_REF_PATTERN)
                bad_step_refs = [ref for ref in step_refs if ref < 0 or ref >= i]
                if bad_step_refs:
                    issues.append(
                        f"return step has invalid step references {sorted(set(bad_step_refs))!r} "
                        f"in args; step refs must be earlier steps in the current plan (< {i})."
                    )
            else:
                bad_step_deps = [dep for dep in slot.step_dependencies if dep < 0 or dep >= i]
                if bad_step_deps:
                    issues.append(
                        f"plan step {i} has invalid step dependencies {sorted(set(bad_step_deps))!r}; "
                        f"all deps must be earlier steps (< {i})."
                    )

            if slot.await_step is not NO_VAL and slot.await_step >= i:
                issues.append(
                    f"plan step {i} has an invalid await_step {slot.await_step!r}; "
                    f"await_step must reference an earlier step (< {i})."
                )

            cache_refs = extract_dependencies(slot.args, placeholder_pattern=self.CACHE_REF_PATTERN)

            # Category 1: index outside the cache entirely.
            out_of_range = [idx for idx in cache_refs if idx < 0 or idx >= cache_len]
            if out_of_range:
                issues.append(
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
                issues.append(
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
                issues.append(
                    f"plan step {i} references cache indices not part of this "
                    f"conversation: {sorted(set(out_of_conv))!r}."
                )

        limit = self.tool_calls_limit
        if limit is not None:
            non_return = sum(1 for slot in planned_slots if slot.tool != return_name)
            if non_return > limit:
                issues.append(
                    f"plan exceeds the tool_calls_limit of {limit} "
                    f"(planned {non_return} non-return steps). Reduce the number of tool calls."
                )

        return issues

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #
    def _validate_generation_output(self, raw_output: str, *, task: PlanActTask) -> list[BlackboardSlot] | str:
        """
        Concrete implementation of ``ToolAgent._validate_generation_output``
        for PlanAct. Owns decoding/extraction (mode-gated) and the entire
        plan-validation pipeline; supersedes the old ``_process_plan_output``.

        Whole-plan comprehensive: every step's own structural/type problems
        are collected across the *entire* plan before returning (not just
        the first offending step), and every cross-step semantic problem
        (``_validate_planned_slots``) is likewise collected in full.

        1. Mode-gated extraction. ``regex``: ``self._extract_from_regex_string``;
           non-empty ``pre_issues`` returns syntax-category feedback
           immediately (tier-1/2 never entered this attempt); empty
           ``candidate_dicts`` returns shape-category feedback immediately.
           ``json``: ``self._extract_from_json_string``; ``json.JSONDecodeError``
           or a non-list/empty result returns the existing decode/shape
           feedback, unchanged. Selects the mode-appropriate field-set
           constants (``REGEX_PLAN_FIELDS``/``REQUIRED_REGEX_PLAN_FIELDS`` vs
           ``PLAN_FIELDS``/``REQUIRED_PLAN_FIELDS``).
        2. For each step: structural/type-validate via
           ``_validate_tool_step_dict``; accumulate every step's issues
           across the whole plan rather than stopping at the first. In
           regex-mode only, pop/validate/attach ``reason`` per step (mirrors
           ``ReActAgent``'s existing json-mode pattern) -- ``reason``'s
           presence is already guaranteed by the required-fields check
           above, only its type/emptiness needs a manual check here. Tally
           every successfully-built slot whose tool is the return tool.
        3. If more than one return-tool slot was tallied: append the
           multi-return issue (relocated verbatim from the old
           ``_normalize_planned_slots`` fail-fast check -- now accumulated
           alongside any other structural issue in the same attempt instead
           of only surfacing on a separate retry turn). Applies to both
           modes.
        4. If any issues accumulated: return the joined feedback --
           normalization and cross-step validation both assume a fully
           structurally-valid slot list, which isn't true yet. Regex-mode
           gets a structural-category header text; json-mode's wording
           stays byte-identical to before (bare joined issues) -- this pass
           only changes json-mode's *timing* for the multi-return case
           (accumulated here instead of a separate retry turn), never its
           wording.
        5. Normalize (``_normalize_planned_slots``) -- no longer a
           string-returning fail-fast gate; multi-return is now impossible
           to reach here (caught in step 3).
        6. Cross-step validation (``_validate_planned_slots``) -- accumulates
           across every slot; return the joined feedback if non-empty
           (regex-mode: semantic-category header text; json-mode: bare
           joined issues, unchanged).
        7. Return the normalized, validated slot list.
        """
        if self._generation_format == "regex":
            candidate_dicts, pre_issues = self._extract_from_regex_string(raw_output)
            if pre_issues:
                return (
                    "Your last output contained syntax errors: address the "
                    "following and resubmit your full plan:\n"
                    + format_generation_issues(pre_issues)
                )
            if not candidate_dicts:
                return (
                    "Your last output failed to parse: no correctly-formatted "
                    "[CALL] or [RETURN] blocks were found. Every plan must "
                    "contain at least one [CALL] block, and must end with "
                    "exactly one [RETURN] block. Re-emit your full plan using "
                    "the exact tag format described in your instructions."
                )
            allowed_fields, required_fields = REGEX_PLAN_FIELDS, REQUIRED_REGEX_PLAN_FIELDS
        else:
            try:
                candidate_dicts = self._extract_from_json_string(raw_output)
            except json.JSONDecodeError as exc:
                return format_generation_issues(
                    [f"Your output could not be parsed as valid JSON. Decoder error: {exc}"]
                )

            if not isinstance(candidate_dicts, list) or not candidate_dicts:
                return format_generation_issues(["The plan must be a non-empty JSON array."])
            allowed_fields, required_fields = PLAN_FIELDS, REQUIRED_PLAN_FIELDS

        issues: list[str] = []
        planned_slots: list[BlackboardSlot] = []
        return_step_indices: list[int] = []

        for i, item in enumerate(candidate_dicts):
            if not isinstance(item, Mapping):
                issues.append(f"Step {i} must be a JSON object (dict), not {type(item).__name__!r}.")
                continue

            step_dict = self._validate_tool_step_dict(
                item,
                expected_step=i,
                allowed_fields=allowed_fields,
                required_fields=required_fields,
                context="plan step",
            )
            if isinstance(step_dict, list):
                issues.extend(step_dict)
                continue

            reason: str | None = None
            if self._generation_format == "regex":
                reason = step_dict.pop(REASON_FIELD)
                if isinstance(reason, str):
                    reason = reason.strip() or None
                elif reason is not None:
                    issues.append(f"plan step {i} 'reason' must be a string; got {type(reason).__name__!r}.")
                    continue

            slot = self._tool_step_dict_to_slot(
                step_dict,
                step=i,
                allowed_fields=PLAN_FIELDS,
                context="plan step",
            )
            if self._generation_format == "regex":
                slot.reason = reason

            if slot.tool == RETURN_TOOL_FULL_NAME:
                return_step_indices.append(i)

            planned_slots.append(slot)

        if len(return_step_indices) > 1:
            issues.append(
                f"plan contains multiple return steps at positions {return_step_indices!r}. "
                "Include at most one return step."
            )

        if issues:
            if self._generation_format == "regex":
                return (
                    "Your last output could not be validated: address the "
                    "following structural issue(s) and resubmit your full plan:\n"
                    + format_generation_issues(issues)
                )
            return format_generation_issues(issues)

        normalized = self._normalize_planned_slots(planned_slots)

        issues = self._validate_planned_slots(
            planned_slots=normalized,
            cache_blackboard=self._blackboard,
            valid_cache_indices=task.valid_cache_indices,
            failed_cache_indices=task.failed_cache_indices,
        )
        if issues:
            if self._generation_format != "regex":
                return format_generation_issues(issues)
            return (
                "Your plan's tags all parsed correctly, but the following "
                "semantic issue(s) must be corrected before it can run:\n"
                + format_generation_issues(issues)
            )

        return normalized

    # ------------------------------------------------------------------ #
    # Task-lifecycle hooks
    # ------------------------------------------------------------------ #
    def _render_task_messages(self, task: PlanActTask) -> list[dict[str, str]]:
        """
        Build this invocation's single decompose-into-plan request.

        Build-once contract: returns ``task.task_messages`` as-is if
        already non-empty. Otherwise combines
        ``self._render_task_banner(task)``'s content with the decompose
        instruction into one user message — the exact text today's
        ``render_task`` produced, now composed through the shared base
        ``Agent.render_task`` pipeline (1a) instead of a full
        reimplementation. The instruction sentence is mode-gated:
        regex-mode describes constructing a tag-block sequence, json-mode
        keeps today's "valid JSON array" wording verbatim.
        """
        if task.task_messages:
            return task.task_messages

        banner = self._render_task_banner(task)
        if self._generation_format == "regex":
            instruction = (
                "Using the current task above and the prior chat history, "
                "construct a sequence of [CALL]/[RETURN] tag blocks (per the "
                "tag format described in your instructions) that decomposes "
                "the task into a series of executable steps."
            )
        else:
            instruction = (
                "Using the current task above and the prior chat history, "
                "construct a valid JSON array that decomposes it into "
                "tool-call steps."
            )
        task.task_messages = [{
            "role": "user",
            "content": f"{banner['content']}\n\n{instruction}",
        }]
        return task.task_messages

    def _generate_plan(self, *, task: PlanActTask) -> list[BlackboardSlot]:
        """
        Generate, parse, and validate a complete PlanAct running blackboard,
        via the shared ``ToolAgent._run_generation_retry_loop``.

        Returns
        -------
        list[BlackboardSlot]
            Fully normalized and validated planned slots (not yet assigned
            onto ``task`` — the caller (``think()``/``async_think()``)
            compiles batches and assigns them).
        """
        return self._run_generation_retry_loop(task=task)

    async def _agenerate_plan(self, *, task: PlanActTask) -> list[BlackboardSlot]:
        """Async mirror of ``_generate_plan``, via
        ``ToolAgent._arun_generation_retry_loop``."""
        return await self._arun_generation_retry_loop(task=task)

    def _initialize_task(
        self,
        *,
        turns: list[AgentRecord],
        prompt: str,
        inputs: dict,
    ) -> PlanActTask:
        """
        Build a bare ``PlanActTask``. No LLM call here — plan generation
        moved to ``think()``, gated on ``task.batches`` being empty, so it
        never re-runs once the plan exists.

        Computes ``valid_cache_indices``/``failed_cache_indices`` via
        ``_compute_cache_index_sets(turns)`` — this hook's shared base
        signature doesn't receive them. ``running_blackboard``/``batches``/
        ``batch_index`` stay at their dataclass defaults (``[]``, ``[]``,
        ``0``) until ``think()`` populates them.
        """
        valid_cache_indices, failed_cache_indices = self._compute_cache_index_sets(turns)
        return PlanActTask(
            turns=turns,
            inputs=inputs,
            user_prompt=prompt,
            system_prompt_name="plan_first" if self._generation_format == "json" else "plan_first_regex",
            valid_cache_indices=valid_cache_indices,
            failed_cache_indices=failed_cache_indices,
        )

    def think(self, task: PlanActTask) -> PlanActTask:
        """
        Generate and validate the whole plan, once.

        No-op once ``task.generated_plan`` is set — a one-shot planner has
        nothing further to decide after its single generation call.
        ``_generate_plan`` already retries internally on parse/validation
        failure, so a validated plan is guaranteed by the time this
        returns. Compilation into batches is deliberately not done here —
        that's deterministic bookkeeping, ``prepare()``'s job (its first
        call compiles from ``task.generated_plan``).
        """
        if task.generated_plan is not NO_VAL:
            return task

        task.generated_plan = self._generate_plan(task=task)
        task.task_messages.clear()
        return task

    async def async_think(self, task: PlanActTask) -> PlanActTask:
        """Async mirror of ``think``: uses ``_agenerate_plan`` so the
        planning LLM call goes through ``async_invoke`` rather than a
        worker thread."""
        if task.generated_plan is not NO_VAL:
            return task

        task.generated_plan = await self._agenerate_plan(task=task)
        task.task_messages.clear()
        return task

    # ------------------------------------------------------------------ #
    # Prepare next batch
    # ------------------------------------------------------------------ #
    def prepare(self, task: PlanActTask) -> PlanActTask:
        """
        Prepare the next pre-compiled batch for execution.

        Compiles the validated plan into topologically-sorted batches on
        first entry (``task.batches`` empty), reading ``task.generated_plan``
        (set by ``think()``). Every later round's call skips straight to
        batch resolution below — compilation is a one-time job, not
        repeated per round. This method then reads the next batch indices,
        resolves placeholders, marks those slots prepared, and populates
        the prepared_steps list.

        Execution
        ~~~~~~~~~
        0. **Compile on first entry**: if ``task.batches`` is empty, compile
           ``task.generated_plan`` via ``_compile_batches_from_deps`` and
           assign ``running_blackboard``/``batches``/``batch_index=0``
        1. **Read next batch**: Get batch indices from ``task.batches[task.batch_index]``
        2. **Validate non-empty**: Batch must have at least one step
        3. **For each step in batch**:
           - Validate bounds: index must be within running_blackboard
           - Validate not already executed or prepared
           - Validate slot is currently planned
           - Validate tool name is set
           - **Cascade check** (``fail_fast=False`` only): if any ``step_dependencies``
             entry is FAILED in the running blackboard, the return tool raises immediately;
             non-return steps are marked FAILED and skipped (not added to prepared_steps)
           - Call ``_resolve_placeholders(slot.args, task=task)``
           - Store resolved args in ``slot.resolved_args``
           - Mark slot ``status="prepared"``
        4. **Set prepared_steps**: Indices of steps that passed the cascade check and were
           prepared; may be empty if all steps in the batch were cascade-failed
        5. **Advance cursor**: Increment ``task.batch_index`` for next iteration

        Concurrency
        ~~~~~~~~~~~
        All steps in the batch can execute concurrently since the topological sort
        guarantee ensures no step in a batch depends on another step in the same batch.

        Note: no ``if task.prepared_steps: raise`` re-entry guard here —
        dead by the same construction as base ``ToolAgent.act()``'s dropped
        guard (1c): ``act()`` always leaves ``task.prepared_steps`` empty
        by the time the next round's ``prepare()`` runs.

        Parameters
        ----------
        task : PlanActTask
            Current task with initialized batches and batch_index cursor

        Returns
        -------
        PlanActTask
            Updated task with prepared_steps populated, batch_index incremented

        Raises
        ------
        ToolAgentError
            On any of:
            - batch_index out of bounds
            - Batch validation failure
            - Placeholder resolution failure
        """
        if not task.batches:
            planned_slots = task.generated_plan
            task.running_blackboard = planned_slots
            task.batches = self._compile_batches_from_deps(
                planned_slots=planned_slots,
                return_idx=len(planned_slots) - 1,
            )
            task.batch_index = 0

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

    async def async_prepare(self, task: PlanActTask) -> PlanActTask:
        """Async mirror of ``prepare``. Direct passthrough — ``prepare``
        has no I/O of its own to justify a thread offload."""
        return self.prepare(task)
