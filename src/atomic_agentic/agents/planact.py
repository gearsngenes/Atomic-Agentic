"""
PlanActAgent: One-Shot output_structure Planner with Concurrent Batch Execution

This module provides ``PlanActAgent``, a concrete ``JsonToolAgent`` subclass
that implements a **static planning** strategy: the LLM is queried once per
invoke, via provider-native structured output (``LLMEngine.output_structure``,
``PLANACT_OUTPUT_SCHEMA``), to produce a complete plan of tool calls, which is
then compiled into topologically-sorted concurrent batches and executed
without further LLM interaction.

Design lineage: this class deliberately mirrors ``DagAgent`` closely --
``$name``-sigil argument resolution, ``DagToolCall``-based batches,
``utils/dag.py``'s ``compile_batches``/``resolve_call_args`` reused directly
-- minus everything that exists only to support ``DagAgent``'s multi-round
continuation (``remaining_work``, ``planning_rounds_limit``, the
continuation-snapshot render path). A one-shot planner has no second round
to fall back to, which is also why it needs one thing ``DagAgent`` doesn't:
cascade-failure handling (``fail_fast=False``) that lets independent
branches of the same plan keep running when part of it fails, since there's
no continuation round to hand the problem to instead.

Planning Model
--------------
A single ``output_structure`` call at the start of each ``invoke()`` produces
``summary``/``plan``/``return`` (``PLANACT_OUTPUT_SCHEMA`` -- no
``remaining_work`` field exists in this schema at all, and ``return`` is
non-nullable: a one-shot plan must always produce a return value,
structurally guaranteed by the schema itself, not a separate post-hoc
validation check). Each ``plan`` entry names a registered tool (by its
alias/bare name, never the dotted ``full_name``), its arguments (literals or
``$name`` references/interpolations), and an optional ``result_name`` other
calls may reference.

Compilation
-----------
``parse_generation`` (``utils/dag.py``) normalizes the validated payload into
a flat ``DagToolCall`` sequence, synthesizing a trailing ``RETURN_ALIAS`` call
from the (always-present) ``return`` value. ``compile_batches`` groups that
sequence into concurrency batches by ``$name`` dependency, with the
``RETURN_ALIAS`` call always isolated into its own final batch.

Execution
---------
Generation and validation happen once, in ``think()``. ``prepare()`` resolves
each batch's ``$name`` references via ``resolve_call_args`` as it's reached.
``act()``/``async_act()`` dispatch each batch concurrently and apply results.
On a resolution or execution failure: ``fail_fast=True`` raises immediately;
``fail_fast=False`` (cascade) skips only the calls that transitively depend
on the failure (``find_cascade_failures``), letting independent branches
finish -- the overall invocation still fails, but only if the failure
prevents the plan's own ``return`` call from ever executing.

Contrast
--------
For adaptive, step-by-step iteration see ``agents/react.py`` (``ReActAgent``).
For the shared tool/constant registry, execution knobs, and lifecycle
contract see ``agents/json_tool_agent.py`` (``JsonToolAgent``).
"""

from __future__ import annotations

import asyncio
import copy
import json
from datetime import datetime
from typing import Any, Callable, ClassVar, Optional

from .json_tool_agent import JsonToolAgent
from .prompts import PLANNER_PROMPT
from ..constants.agents import RETURN_ALIAS, RETURN_VALUE_FIELD
from ..core import AtomicInvokable
from ..llm.base import LLMEngine
from ..exceptions import ToolAgentError, ToolInvocationError
from ..mcp import MCPClientHub
from ..a2a import A2AClientHub, PyA2AtomicClient
from ..models.agents.tasks import PlanActTask
from ..models.agents.records import AgentRecord, JsonToolAgentRecord, LLMRecord
from ..models.results.agents import JsonToolAgentResult, ToolUsageRecord
from ..utils.core import run_coro_sync
from ..utils.dag import (
    build_planact_schema,
    compile_batches,
    find_cascade_failures,
    find_sigil_refs,
    is_dispatched_call,
    parse_generation,
    render_completed_as_json,
    resolve_call_args,
    validate_calls,
)


# --------------------------------------------------------------------------- #
# PlanAct Agent
# --------------------------------------------------------------------------- #
class PlanActAgent(JsonToolAgent):
    """
    One-shot planner agent: generates an entire plan upfront via
    ``output_structure``, executes it in concurrent batches, never
    replans.

    **Design**: mirrors ``DagAgent`` closely (see module docstring for the
    full lineage). The one genuinely new mechanism relative to ``DagAgent``
    is cascade-failure handling -- see ``_apply_batch_results`` below.

    Advantages
    ~~~~~~~~~~
    - **No replanning**: the whole plan is known upfront; no latency per
      iteration.
    - **Concurrency-friendly**: dependency-batch compilation enables maximal
      parallelism.
    - **Deterministic**: same inputs produce an identical execution plan
      every time (modulo the LLM's own generation).
    - **Schema-guaranteed shape**: ``output_structure`` strict mode makes a
      malformed-shape generation structurally impossible -- the
      regeneration-retry loop only ever fires for semantic issues
      (``validate_calls``), never a parse failure.

    Limitations
    ~~~~~~~~~~~
    - **No adaptivity**: cannot branch based on intermediate results.
    - **Plan quality**: entirely dependent on the LLM's single planning
      turn.
    - **Error recovery**: a failure that reaches the plan's own ``return``
      call (directly, or by cascading through a dependency) ends the whole
      invocation -- there's no second round to repair it in.
    """

    _ATOMIC_IMMUTABLE_TYPES: ClassVar[tuple[type, ...]] = (
        str, int, float, bool, complex, bytes, type(None),
    )

    def __init__(
        self,
        name: str,
        namespace: str,
        description: str,
        llm_engine: LLMEngine,
        context_enabled: bool = False,
        *,
        tool_calls_limit: Optional[int] = None,
        regeneration_limit: int = 5,
        tool_concurrency_limit: Optional[int] = None,
        fail_fast: bool = True,
        response_preview_limit: Optional[int] = None,
        pre_invoke: Optional[AtomicInvokable | Callable[..., Any]] = None,
        post_invoke: Optional[AtomicInvokable | Callable[..., Any]] = None,
        post_result_key: Optional[str] = None,
        records_window: Optional[int] = None,
        tools: Optional[list[AtomicInvokable | Callable | MCPClientHub | A2AClientHub | PyA2AtomicClient]] = None,
        constants: Optional[list[Any]] = None,
        constant_aliases: Optional[list[Optional[str]]] = None,
        constant_descriptions: Optional[list[Optional[str]]] = None,
    ) -> None:
        """
        Every parameter forwards verbatim to ``JsonToolAgent.__init__`` --
        no ``extra_parameters`` keyword, matching that base class's own
        signature exactly (it accepts none). ``"plan_first"`` is the key
        under which the built-in planning prompt is registered in
        ``self._system_prompts``.
        """
        super().__init__(
            name=name,
            namespace=namespace,
            description=description,
            llm_engine=llm_engine,
            context_enabled=context_enabled,
            fail_fast=fail_fast,
            tool_calls_limit=tool_calls_limit,
            regeneration_limit=regeneration_limit,
            tool_concurrency_limit=tool_concurrency_limit,
            response_preview_limit=response_preview_limit,
            pre_invoke=pre_invoke,
            post_invoke=post_invoke,
            post_result_key=post_result_key,
            records_window=records_window,
            tools=tools,
            constants=constants,
            constant_aliases=constant_aliases,
            constant_descriptions=constant_descriptions,
        )
        self._system_prompts["plan_first"] = PLANNER_PROMPT

    # ------------------------------------------------------------------ #
    # Shared per-invocation helpers
    # ------------------------------------------------------------------ #
    @classmethod
    def _copy_for_task_namespace(cls, value: Any) -> Any:
        """
        Ported verbatim from ``DagAgent._copy_for_task_namespace`` -- own
        private copy, not shared on ``JsonToolAgent``, matching this
        release's Stage A/B duplication posture for family-specific
        initialization helpers. Returns ``value`` unchanged if it's a known
        atomic-immutable type; otherwise a deep copy, so a mutating call on
        a cached/constant value can never reach the real, shared object.
        """
        if isinstance(value, cls._ATOMIC_IMMUTABLE_TYPES):
            return value
        return copy.deepcopy(value)

    def _initialize_task(
        self,
        *,
        turns: list[AgentRecord],
        prompt: str,
        inputs: dict,
    ) -> PlanActTask:
        """
        Ported from ``DagAgent._initialize_task``: build a bare
        ``PlanActTask``, seed its ``cache`` with every visible prior turn's
        result under ``task_result_{i}`` (unconditional over whatever
        ``turns`` contains -- every committed turn is definitionally a
        completed success under this model), and seed ``constant_values``
        from every registered constant. Both copied via
        ``_copy_for_task_namespace`` exactly once here.
        """
        task = PlanActTask(
            turns=turns, inputs=inputs, user_prompt=prompt, system_prompt_name="plan_first",
        )
        for turn in turns:
            task.cache[f"task_result_{self._turn_position(turn)}"] = self._copy_for_task_namespace(
                turn.generated_response
            )
        for spec in self._constants.values():
            task.constant_values[spec.name] = self._copy_for_task_namespace(spec.value)
        return task

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #
    def _render_current_task_message(self, task: PlanActTask) -> dict[str, str]:
        """
        The "what is the task" user message. Unlike ``DagAgent``'s own
        version, this one carries the ``tool_calls_limit`` text -- moved
        out of the system prompt entirely (see
        ``JsonToolAgent._render_system_message``) since it's a
        per-invocation fact, not a standing instruction, and showing it
        here avoids spending system-prompt tokens on it every construction
        even though this family only ever renders one round.
        """
        content = f"CURRENT TASK:\n{task.user_prompt}"
        if self.tool_calls_limit is not None:
            plural = "s" if self.tool_calls_limit != 1 else ""
            content += f"\n\nYou may make at most {self.tool_calls_limit} tool call{plural} total."
        return {"role": "user", "content": content}

    def _render_task_messages(self, task: PlanActTask) -> list[dict[str, str]]:
        """
        Build-once contract. No continuation branch exists at all -- this
        family never re-enters ``think()`` after its one real generation
        call, unlike ``DagAgent``'s own version. Regen-repair feedback (on
        a semantic validation failure) is injected by the retry loop
        itself as ``additional_messages``, never through this method --
        same separation ``DagAgent._run_planning_retry_loop`` keeps.
        """
        if task.task_messages:
            return task.task_messages

        banner = self._render_current_task_message(task)
        task.task_messages = [{
            "role": "user",
            "content": f"{banner['content']}\n\nWrite a plan to accomplish this task now.",
        }]
        return task.task_messages

    # ------------------------------------------------------------------ #
    # Generation (think())
    # ------------------------------------------------------------------ #
    def _process_generation_output(
        self, raw_output: dict[str, Any], task: PlanActTask,
    ) -> list[list[Any]] | str:
        """
        Pure-computation validate callback for the planning retry loop:
        parse, validate semantics + remaining tool-call budget, and compile
        into batches. Returns the compiled batches on success, or a
        feedback string describing every problem found on failure.

        Ported from ``DagAgent._process_generation_output``, deltas:
        ``remaining_work`` is discarded (``parse_generation`` always
        returns ``None`` for it against this schema) and hardcoded ``None``
        into ``validate_calls`` -- both of that function's
        ``remaining_work``-keyed checks become permanent no-ops. No
        final-round-defer check -- there is no rounds concept to bound.
        """
        calls, _ = parse_generation(raw_output)

        remaining_budget = self._tool_calls_limit
        known_names = frozenset(task.cache) | frozenset(task.constant_values)
        issues = validate_calls(calls, None, remaining_budget, known_names)

        if issues:
            issues_msg = "\n".join(f"{i + 1}. {m}" for i, m in enumerate(issues))
            # TODO(smoke-test aid, mirrors DagAgent's own): remove once
            # cross-provider output_structure reliability is confirmed.
            print(f"[PlanActAgent DEBUG] plan rejected, issues:\n{issues_msg}")
            return issues_msg

        return compile_batches(
            calls,
            max_concurrency=self._tool_concurrency_limit,
            start_batch_index=0,
        )

    def _run_planning_retry_loop(self, *, task: PlanActTask) -> list[list[Any]]:
        """
        Render, call the engine (with ``output_structure``), record the
        attempt, validate/compile via ``_process_generation_output``, and
        retry with injected feedback on failure until success or the
        regeneration budget (``self._regeneration_limit``, tracked via
        ``task.regenerations_used``) is exhausted. Ported from
        ``DagAgent._run_planning_retry_loop``, including its smoke-test-aid
        debug print of every raw generated plan (see
        ``_process_generation_output`` for the matching rejected-issues
        print).
        """
        additional_messages: list[dict[str, str]] = []

        while True:
            messages = self.render_task(task, additional_messages=additional_messages)
            schema = build_planact_schema(self._toolbox.keys())
            engine_result = self._llm_engine.invoke(
                {"messages": messages, "output_structure": schema}
            )
            raw_output: dict[str, Any] = engine_result.result
            # TODO(smoke-test aid, mirrors DagAgent's own): remove once
            # cross-provider output_structure reliability is confirmed.
            print(f"[PlanActAgent DEBUG] generated plan:\n{json.dumps(raw_output, indent=2)}")

            task.llm_records.append(LLMRecord(
                messages=list(task.task_messages),
                llm_result=engine_result,
                system_prompt_name=task.system_prompt_name,
            ))

            result = self._process_generation_output(raw_output, task)
            if isinstance(result, str):
                if task.regenerations_used >= self._regeneration_limit:
                    raise ToolAgentError(
                        f"{type(self).__name__}.{self.name}: regeneration budget "
                        f"exhausted after {task.regenerations_used + 1} attempt(s). "
                        f"Last feedback: {result}"
                    )
                additional_messages = [
                    {"role": "assistant", "content": json.dumps(raw_output)},
                    {"role": "user", "content": (
                        f"Your plan could not be used:\n\n{result}\n\n"
                        "Produce a corrected plan."
                    )},
                ]
                task.regenerations_used += 1
                continue

            return result

    async def _arun_planning_retry_loop(self, *, task: PlanActTask) -> list[list[Any]]:
        """Async mirror of ``_run_planning_retry_loop``, using
        ``async_invoke`` for the engine call. Same smoke-test-aid debug
        print of every raw generated plan."""
        additional_messages: list[dict[str, str]] = []

        while True:
            messages = self.render_task(task, additional_messages=additional_messages)
            schema = build_planact_schema(self._toolbox.keys())
            engine_result = await self._llm_engine.async_invoke(
                {"messages": messages, "output_structure": schema}
            )
            raw_output: dict[str, Any] = engine_result.result
            # TODO(smoke-test aid, mirrors DagAgent's own): remove once
            # cross-provider output_structure reliability is confirmed.
            print(f"[PlanActAgent DEBUG] generated plan:\n{json.dumps(raw_output, indent=2)}")

            task.llm_records.append(LLMRecord(
                messages=list(task.task_messages),
                llm_result=engine_result,
                system_prompt_name=task.system_prompt_name,
            ))

            result = self._process_generation_output(raw_output, task)
            if isinstance(result, str):
                if task.regenerations_used >= self._regeneration_limit:
                    raise ToolAgentError(
                        f"{type(self).__name__}.{self.name}: regeneration budget "
                        f"exhausted after {task.regenerations_used + 1} attempt(s). "
                        f"Last feedback: {result}"
                    )
                additional_messages = [
                    {"role": "assistant", "content": json.dumps(raw_output)},
                    {"role": "user", "content": (
                        f"Your plan could not be used:\n\n{result}\n\n"
                        "Produce a corrected plan."
                    )},
                ]
                task.regenerations_used += 1
                continue

            return result

    def think(self, task: PlanActTask) -> PlanActTask:
        """
        Generate and validate the whole plan, once. No-op on any later
        call -- a one-shot planner has nothing further to decide once
        ``task.pending``/``task.completed`` exist (or the task is already
        complete).
        """
        if task.pending or task.completed or task.complete:
            return task

        task.pending = self._run_planning_retry_loop(task=task)
        task.task_messages.clear()
        return task

    async def async_think(self, task: PlanActTask) -> PlanActTask:
        """Async mirror of ``think``."""
        if task.pending or task.completed or task.complete:
            return task

        task.pending = await self._arun_planning_retry_loop(task=task)
        task.task_messages.clear()
        return task

    # ------------------------------------------------------------------ #
    # Prepare next batch
    # ------------------------------------------------------------------ #
    def prepare(self, task: PlanActTask) -> PlanActTask:
        """
        Resolve the next pending batch's args.

        Ported from ``DagAgent.prepare``, with the failure branch replaced
        entirely -- ``DagAgent`` has a continuation round to fall back to
        on a resolution failure; this family doesn't:

        1. If ``task.pending`` is empty: reset ``task.resolved_args`` and
           return unchanged (no ``continue_planning`` branch to check --
           if the return call never executed, ``act()``'s own final-batch
           check is what raises, not this method).
        2. Resolve every call's args in ``task.pending[0]`` against
           ``{**task.cache, **task.constant_values}``, collecting failures
           rather than stopping at the first.
        3. If any resolution issues:
           - ``fail_fast=True``: raise ``ToolAgentError`` immediately,
             listing every issue.
           - ``fail_fast=False``: for each call that failed to resolve, set
             ``call.exception = ToolAgentError(<issue text>)`` and append it
             to ``task.failed_statements``; collect the identifiers of every
             such call; call ``find_cascade_failures`` against
             ``task.pending[1:]`` to find every later-batch call that
             transitively depends on one of them; remove every call (from
             the *current* batch and every later batch) whose args/kwargs
             reference a poisoned name -- cascade-skipped calls are never
             added to ``completed``/``failed_statements`` (never attempted).
        4. Resolve the surviving calls in this batch (already-resolved
           values re-used, not re-computed) into ``task.resolved_args`` --
           may be empty if the whole batch was cascade-affected; ``act()``
           treats that as a legal no-op.
        """
        if not task.pending:
            task.resolved_args = []
            return task

        batch = task.pending[0]
        resolution_namespace = {**task.cache, **task.constant_values}

        resolved_by_index: dict[int, tuple[list[Any], dict[str, Any]]] = {}
        issues: list[str] = []
        failed_identifiers: set[str] = set()

        for i, call in enumerate(batch):
            label = call.identifier if call.identifier is not None else "(unassigned)"
            try:
                positional, keyword = resolve_call_args(call, resolution_namespace)
            except Exception as e:
                issue = f"{label}: could not resolve argument value(s): {e!r}"
                issues.append(issue)
                if not self._fail_fast:
                    call.exception = ToolAgentError(issue)
                    task.failed_statements.append(call)
                    if call.identifier is not None:
                        failed_identifiers.add(call.identifier)
                continue
            resolved_by_index[i] = (positional, keyword)

        if issues and self._fail_fast:
            issues_msg = "\n".join(f"{i + 1}. {m}" for i, m in enumerate(issues))
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: batch could not be resolved:\n{issues_msg}"
            )

        if failed_identifiers:
            poisoned = find_cascade_failures(failed_identifiers, task.pending[1:])
            self._drop_poisoned_calls(task.pending, poisoned)
            # The current batch's own surviving entries may also reference
            # a poisoned name from a sibling failure within this same
            # batch -- filter resolved_by_index accordingly.
            for i, call in enumerate(batch):
                if i not in resolved_by_index:
                    continue
                refs: set[str] = set()
                for value in call.args:
                    refs |= find_sigil_refs(value)
                for value in call.kwargs.values():
                    refs |= find_sigil_refs(value)
                if refs & poisoned:
                    del resolved_by_index[i]

        resolved: list[dict[str, Any]] = []
        surviving_batch: list[Any] = []
        for i, call in enumerate(batch):
            if i not in resolved_by_index:
                continue
            positional, keyword = resolved_by_index[i]
            if call.tool == RETURN_ALIAS:
                resolved.append({RETURN_VALUE_FIELD: keyword[RETURN_VALUE_FIELD]})
            else:
                tool = self.get_tool(call.tool)
                resolved.append(tool._args_kwargs_to_dict(*positional, **keyword))
            surviving_batch.append(call)

        task.pending[0] = surviving_batch
        task.resolved_args = resolved
        return task

    async def async_prepare(self, task: PlanActTask) -> PlanActTask:
        """Direct passthrough -- ``prepare`` has no I/O of its own."""
        return self.prepare(task)

    @staticmethod
    def _drop_poisoned_calls(pending: list[list[Any]], poisoned: set[str]) -> None:
        """
        Remove, in place, every call from every batch in ``pending`` whose
        ``args``/``kwargs`` reference (via ``find_sigil_refs``) any name in
        ``poisoned`` -- named or not. ``poisoned`` is a filter key
        (``find_cascade_failures``'s own return contract), not a list of
        calls to remove directly.
        """
        for batch in pending:
            survivors = []
            for call in batch:
                refs: set[str] = set()
                for value in call.args:
                    refs |= find_sigil_refs(value)
                for value in call.kwargs.values():
                    refs |= find_sigil_refs(value)
                if not (refs & poisoned):
                    survivors.append(call)
            batch[:] = survivors

    # ------------------------------------------------------------------ #
    # Execute prepared batch
    # ------------------------------------------------------------------ #
    async def _gather_batch_results(
        self, batch: list[Any], resolved: list[dict[str, Any]],
    ) -> list[Any]:
        """Ported verbatim from ``DagAgent._gather_batch_results``."""
        coros: list[Any] = []
        dispatch_map: dict[int, int] = {}
        for i, call in enumerate(batch):
            if is_dispatched_call(call):
                dispatch_map[i] = len(coros)
                tool = self.get_tool(call.tool)
                coros.append(tool.async_invoke(resolved[i]))

        gathered = await asyncio.gather(*coros, return_exceptions=True) if coros else []
        return [
            gathered[dispatch_map[i]] if i in dispatch_map else resolved[i][RETURN_VALUE_FIELD]
            for i in range(len(batch))
        ]

    def _apply_batch_results(
        self,
        task: PlanActTask,
        batch: list[Any],
        resolved: list[dict[str, Any]],
        raw_results: list[Any],
    ) -> PlanActTask:
        """
        Shared post-gather bookkeeping for ``act``/``async_act``. Ported
        from ``DagAgent._apply_batch_results``, with the failure branch
        replaced entirely -- this is the one real behavioral divergence
        from ``DagAgent``'s own version, since ``DagAgent`` has no
        fail-fast/cascade fork at all (every failure there triggers a
        forced continuation; this family has no continuation to fall back
        to).

        1. Partition ``raw_results`` into successes/failures (same
           ``is_dispatched_call`` gate, same ``BaseException`` check as
           ``DagAgent``'s own version).
        2. No failures: apply successes into ``completed``/``cache``; if
           the ``RETURN_ALIAS`` call succeeded, set
           ``generated_response``/``complete``; pop the batch.
        3. Failures present:
           - ``fail_fast=True``: raise immediately (preserve
             ``ToolInvocationError`` as-is; wrap anything else in
             ``ToolAgentError``).
           - ``fail_fast=False``: record each failure onto
             ``failed_statements``, cascade-skip every later-batch call
             that transitively depends on one of the failed identifiers
             (``find_cascade_failures`` + ``_drop_poisoned_calls``), apply
             this batch's own surviving successes, pop the batch.
        4. After popping: if ``task.pending`` is now empty and
           ``task.complete`` is still ``False`` -- the return call never
           executed (failed directly, or was cascade-skipped). Raise
           ``ToolAgentError`` summarizing every failure recorded in
           ``task.failed_statements``. Otherwise return ``task`` normally.
        """
        triples = list(zip(batch, resolved, raw_results))
        failures = [
            (call, kwargs, raw) for call, kwargs, raw in triples
            if is_dispatched_call(call) and isinstance(raw, BaseException)
        ]
        successes = [
            (call, kwargs, raw) for call, kwargs, raw in triples
            if not (is_dispatched_call(call) and isinstance(raw, BaseException))
        ]

        if failures:
            if self._fail_fast:
                call, _kwargs, raw_error = failures[0]
                if isinstance(raw_error, ToolInvocationError):
                    raise raw_error
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: tool call failed "
                    f"for {call.tool!r} (identifier={call.identifier!r}): {raw_error}"
                ) from raw_error

            failed_identifiers: set[str] = set()
            for call, _kwargs, raw_error in failures:
                call.exception = raw_error
                task.failed_statements.append(call)
                if call.identifier is not None:
                    failed_identifiers.add(call.identifier)

            poisoned = find_cascade_failures(failed_identifiers, task.pending[1:])
            self._drop_poisoned_calls(task.pending[1:], poisoned)

        for call, kwargs, value in successes:
            task.completed.append(call)
            if call.tool != RETURN_ALIAS:
                call.result = value
                unwrapped = value.result
            else:
                unwrapped = value
            if call.identifier is not None:
                task.cache[call.identifier] = unwrapped
            if call.tool == RETURN_ALIAS:
                task.generated_response = kwargs[RETURN_VALUE_FIELD]
                task.complete = True

        task.pending.pop(0)
        task.resolved_args = []
        self._check_plan_exhausted(task)
        return task

    def _check_plan_exhausted(self, task: PlanActTask) -> None:
        """
        Raise iff ``task.pending`` has drained (no batches left, empty or
        otherwise) and ``task.complete`` is still ``False`` -- the return
        call never executed, whether because it failed directly or was
        cascade-skipped via a failed dependency. Shared by ``act``'s
        no-op branch (below -- a batch that cascade-filtering emptied out
        entirely, including the return call's own isolated batch, still
        needs this check once it's popped) and the tail of
        ``_apply_batch_results`` above.
        """
        if not task.pending and not task.complete:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: plan finished without "
                "producing a return value -- the return call either failed "
                "directly or depended on a call that failed. Failures:\n"
                f"{render_completed_as_json(task.failed_statements)}"
            )

    def act(self, task: PlanActTask) -> PlanActTask:
        """
        Execute the currently prepared batch, or -- unlike ``DagAgent.act``,
        which never needs this -- consume a batch that cascade-filtering
        emptied out entirely before it ever reached dispatch (`DagAgent`
        always clears the *whole* of `pending` on any failure, never
        partially filters one batch, so this situation never arises there).
        Without popping it here, `prepare()` would keep re-processing the
        same now-empty batch forever -- this is a real bug this
        implementation found and fixed via live testing, not a
        theoretical concern.
        """
        if not task.resolved_args:
            if task.pending and not task.pending[0]:
                task.pending.pop(0)
            self._check_plan_exhausted(task)
            return task

        batch = task.pending[0]
        resolved = task.resolved_args
        raw_results = run_coro_sync(self._gather_batch_results(batch, resolved))
        return self._apply_batch_results(task, batch, resolved, raw_results)

    async def async_act(self, task: PlanActTask) -> PlanActTask:
        """Async mirror of ``act`` -- same empty-batch consumption fix."""
        if not task.resolved_args:
            if task.pending and not task.pending[0]:
                task.pending.pop(0)
            self._check_plan_exhausted(task)
            return task

        batch = task.pending[0]
        resolved = task.resolved_args
        raw_results = await self._gather_batch_results(batch, resolved)
        return self._apply_batch_results(task, batch, resolved, raw_results)

    # ------------------------------------------------------------------ #
    # Record / result construction
    # ------------------------------------------------------------------ #
    def _build_record_from_task(
        self,
        task: PlanActTask,
        turns: list[AgentRecord],
    ) -> JsonToolAgentRecord:
        """
        Assemble a completed ``JsonToolAgentRecord`` from a finished
        ``PlanActTask``. No agent-level global blackboard to persist into
        -- each record owns its own calls outright, mirroring
        ``DagAgent._build_record_from_task``'s identical shape.
        """
        prev = turns[-1] if turns else None
        return JsonToolAgentRecord(
            user_prompt=task.user_prompt,
            generated_response=task.generated_response,
            inputs=task.inputs,
            llm_records=tuple(task.llm_records),
            prev=prev,
            statements=tuple(task.completed),
            failed_statements=tuple(task.failed_statements),
            regenerations_used=task.regenerations_used,
        )

    def build_result_from_record(
        self,
        record: JsonToolAgentRecord,
        *,
        result: Any,
        started_at: datetime,
        ended_at: datetime,
    ) -> JsonToolAgentResult:
        """
        Construct this agent's ``JsonToolAgentResult`` envelope directly
        from a completed ``JsonToolAgentRecord``. No ``DagAgent`` override
        exists to port from (confirmed -- ``DagAgent`` uses base
        ``Agent.build_result_from_record``'s plain ``AgentResult``
        unmodified); follows that base method's own documented contract
        shape instead (``base.py``'s ``build_result_from_record``).

        Derives ``tool_usage`` from ``record.statements`` (per-tool call
        counts, ``RETURN_ALIAS`` excluded, ordered by first-call order) and
        ``failed_call_count`` from ``len(record.failed_statements)``.
        """
        llm_token_usage = tuple(r.llm_result.token_usage for r in record.llm_records)
        llm_model_data = record.llm_records[-1].llm_result.model_data

        counts: dict[str, int] = {}
        for call in record.statements:
            if is_dispatched_call(call):
                counts[call.tool] = counts.get(call.tool, 0) + 1
        tool_usage = tuple(
            ToolUsageRecord(tool_name=name, call_count=count)
            for name, count in counts.items()
        )

        return self._make_result(
            result=result,
            started_at=started_at,
            ended_at=ended_at,
            result_cls=JsonToolAgentResult,
            llm_token_usage=llm_token_usage,
            llm_model_data=llm_model_data,
            tool_usage=tool_usage,
            failed_call_count=len(record.failed_statements),
            regenerations_used=record.regenerations_used,
        )
