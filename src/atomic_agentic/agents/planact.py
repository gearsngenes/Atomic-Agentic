"""
PlanActAgent: One-Shot output_structure Planner with Concurrent Batch Execution

This module provides ``PlanActAgent``, a concrete ``ToolAgent`` subclass
that implements a **static planning** strategy: the LLM is queried once per
invoke, via provider-native structured output (``LLMEngine.output_structure``,
``PLANACT_OUTPUT_SCHEMA``), to produce a complete plan of tool calls, which is
then compiled into topologically-sorted concurrent batches and executed
without further LLM interaction.

Design: uses ``$name``-sigil argument resolution, translated into real
``ast.expr`` nodes on a shared ``ToolStatement`` (``utils/sigils.py``'s
``translate_calls``), and ``ToolStatement``-based batch compilation
(``utils/agents.py``'s ``compile_batches``/``resolve_statement_args``,
reused directly). Values live in ``task.context``: pre-existing context
variables are referenced as ``$name``, and each ``result_name`` becomes a
context variable of the task (persisted only by the usual conversation
rules). No multi-round continuation of any kind -- a one-shot
planner decides everything in a single generation, which
is also why it needs cascade-failure handling (``fail_fast=False``): with no
continuation round to hand a failure to, independent branches of the same
plan must be able to keep running on their own when part of it fails.

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
``parse_generation`` (``utils/sigils.py``) normalizes the validated payload
into a flat list of draft calls; ``translate_calls`` then validates and
translates those drafts into a flat ``ToolStatement`` sequence (synthesizing
a trailing ``RETURN_ALIAS`` call from the (always-present) ``return``
value). ``compile_batches`` groups that sequence into concurrency batches by
``$name`` dependency, with the ``RETURN_ALIAS`` call always isolated into
its own final batch.

Execution
---------
Generation and validation happen once, in ``think()``; the plan is checked
(including a dry bind of every tool call's arguments) before anything runs.
``prepare()`` resolves each batch's ``$name`` references against
``task.context`` via ``resolve_statement_args`` as it's reached.
``act()``/``async_act()`` dispatch each batch concurrently and apply results.
On a resolution or execution failure: ``fail_fast=True`` raises immediately;
``fail_fast=False`` (cascade) skips only the calls that transitively depend
on the failure (``find_cascade_failures``), letting independent branches
finish -- the overall invocation still fails, but only if the failure
prevents the plan's own ``return`` call from ever executing.

Contrast
--------
For adaptive, step-by-step iteration see ``agents/react.py`` (``ReActAgent``).
For the shared tool registry, execution knobs, and lifecycle
contract see ``agents/toolagent.py`` (``ToolAgent``).
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Iterable
from typing import Any, Callable, ClassVar, Mapping, Optional

from .toolagent import ToolAgent
from .prompts import PLANNER_PROMPT
from .tools import make_dict, make_sequence
from ..constants.agents import RETURN_ALIAS, RETURN_VALUE_FIELD
from ..core import AtomicInvokable
from ..llm.base import LLMEngine
from ..exceptions import ToolAgentError, ToolInvocationError
from ..mcp import MCPClientHub
from ..a2a import A2AClientHub, PyA2AtomicClient
from ..models.agents.blackboard_models import ToolStatement
from ..models.agents.prompts import PromptConfig
from ..models.agents.tasks import PlanActTask
from ..models.agents.records import ToolAgentRecord, LLMRecord
from ..models.results.agents import ToolAgentResult
from ..utils.agents import (
    build_usage_report,
    check_call_bindings,
    compile_batches,
    extract_identifiers,
    is_dispatched,
    resolve_statement_args,
)
from ..utils.core import run_coro_sync
from ..utils.sigils import (
    build_planact_schema,
    find_cascade_failures,
    parse_generation,
    render_failed_as_json,
    translate_calls,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# PlanAct Agent
# --------------------------------------------------------------------------- #
class PlanActAgent(ToolAgent):
    """
    One-shot planner agent: generates an entire plan upfront via
    ``output_structure``, executes it in concurrent batches, never
    replans.

    **Design**: see the module docstring for the full lineage. Cascade-
    failure handling (see ``_apply_batch_results`` below) is this class's
    own distinguishing mechanism.

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
      (``translate_calls``), never a parse failure.

    Limitations
    ~~~~~~~~~~~
    - **No adaptivity**: cannot branch based on intermediate results.
    - **Plan quality**: entirely dependent on the LLM's single planning
      turn.
    - **Error recovery**: a failure that reaches the plan's own ``return``
      call (directly, or by cascading through a dependency) ends the whole
      invocation -- there's no second round to repair it in.
    """

    # A class attribute (active before __init__ runs) -- the two
    # composite-value-building utility tools this family's schema text
    # tells the model to use. No return_tool entry -- PlanActAgent never
    # registers a real return tool (its return value comes from the
    # synthesized RETURN_ALIAS call instead). No get_item entry: too niche
    # relative to make_sequence/make_dict/return, and its presence in
    # AVAILABLE TOOLS pulls live generations toward container-shaped
    # detours a flat call sequence doesn't need.
    _RESERVED_TOOL_NAMES: ClassVar[frozenset[str]] = frozenset(
        {"make_sequence", "make_dict"}
    )

    # This family's $name-sigil grammar (utils/sigils.py): context entries
    # render as "$name", history labels as "$task_result_i".
    _CONTEXT_NAME_PREFIX: str = "$"

    _TASK_CLS = PlanActTask
    _INITIAL_PROMPT_NAME = "plan_first"

    def __init__(
        self,
        name: str,
        namespace: str,
        description: str,
        llm_engine: LLMEngine,
        record_enabled: bool = False,
        *,
        tool_calls_limit: Optional[int] = None,
        regeneration_limit: int = 5,
        tool_concurrency_limit: Optional[int] = None,
        fail_fast: bool = True,
        pre_invoke: Optional[AtomicInvokable | Callable[..., Any]] = None,
        post_invoke: Optional[AtomicInvokable | Callable[..., Any]] = None,
        inputs_to_context: Optional[Iterable[str] | Mapping[str, str]] = None,
        response_to_context: Optional[str | Iterable[str] | Mapping[Any, str]] = None,
        response_to_post: Optional[str | Iterable[str] | Mapping[Any, str]] = None,
        use_runtime_context: bool = False,
        records_window: Optional[int] = None,
        tools: Optional[list[AtomicInvokable | Callable | MCPClientHub | A2AClientHub | PyA2AtomicClient]] = None,
        instructions: Optional[str | PromptConfig] = None,
    ) -> None:
        """
        Every parameter except ``fail_fast`` forwards verbatim to
        ``ToolAgent.__init__`` (``instructions`` is the optional standing
        user guidance, framed into the system message by the base class).
        There are no constants: pre-existing values are context variables
        (remembered on the conversation, bound from inputs, or runtime).
        ``fail_fast`` has no shared home on ``ToolAgent`` (see the
        ``fail_fast`` property below) -- validated and stored locally
        instead. ``"plan_first"`` is the key under which the built-in
        planning prompt is registered in ``self._system_prompts``.
        """
        super().__init__(
            name=name,
            namespace=namespace,
            description=description,
            llm_engine=llm_engine,
            record_enabled=record_enabled,
            tool_calls_limit=tool_calls_limit,
            regeneration_limit=regeneration_limit,
            tool_concurrency_limit=tool_concurrency_limit,
            pre_invoke=pre_invoke,
            post_invoke=post_invoke,
            inputs_to_context=inputs_to_context,
            response_to_context=response_to_context,
            response_to_post=response_to_post,
            use_runtime_context=use_runtime_context,
            records_window=records_window,
            tools=tools,
            instructions=instructions,
        )

        if not isinstance(fail_fast, bool):
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: fail_fast must be a bool."
            )
        self._fail_fast = fail_fast

        self._system_prompts["plan_first"] = PLANNER_PROMPT
        # Seeded directly, bypassing register_tool -- both are reserved
        # (self._RESERVED_TOOL_NAMES), and register_tool now rejects any
        # attempt to register something under a reserved id.
        self._seed_reserved_tool(make_sequence, "make_sequence")
        self._seed_reserved_tool(make_dict, "make_dict")

    @property
    def fail_fast(self) -> bool:
        """
        When ``True`` (default), the first tool call failure immediately
        raises and aborts the run. When ``False``, execution cascades:
        calls that depend, directly or transitively, on a failed call's
        result are skipped rather than attempted; independent branches
        still run to completion. Declared directly on this class -- not
        shared with ``ReActAgent``, whose own ``fail_fast`` governs a
        different granularity (a single per-round call, not a whole batch).
        """
        return self._fail_fast

    def to_dict(self) -> dict[str, Any]:
        """Extends ``ToolAgent.to_dict()`` with this class's own ``fail_fast``."""
        d = super().to_dict()
        d["fail_fast"] = self._fail_fast
        return d

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #
    def _render_steps(self, task: PlanActTask) -> str:
        """Always ``""``: a one-shot planner has no completed work or repair
        to show before its only generation, so the base emits no STEPS block."""
        return ""

    def _render_directive(self, task: PlanActTask) -> str:
        """The plan-now instruction, plus the total ``tool_calls_limit``
        line when a limit is set."""
        directive = "Write a plan to accomplish this task now."
        limit = self.tool_calls_limit
        if limit is not None:
            plural = "s" if limit != 1 else ""
            directive += f"\n\nYou may make at most {limit} tool call{plural} total."
        return directive

    # ------------------------------------------------------------------ #
    # Generation (think())
    # ------------------------------------------------------------------ #
    def _process_generation_output(
        self, raw_output: dict[str, Any], task: PlanActTask,
    ) -> list[list[ToolStatement]] | str:
        """
        Pure-computation validate callback for the planning retry loop:

        1. ``parse_generation``: raw JSON -> drafts.
        2. ``translate_calls`` against the names in ``task.context`` (and its
           readonly subset) and the tool-call budget -> ``ToolStatement``s.
        3. ``check_call_bindings``: dry-bind each registered-tool call.
        4. Any issues from 2-3 are returned as one numbered feedback string;
           otherwise the calls are compiled into dependency batches.
        """
        drafts = parse_generation(raw_output)

        context_names = frozenset(task.context)
        readonly_names = frozenset(n for n, v in task.context.items() if v.readonly)
        issues, calls = translate_calls(
            drafts, self._tool_calls_limit, context_names, readonly_names
        )
        issues += check_call_bindings(calls, self._toolbox)

        if issues:
            issues_msg = "\n".join(f"{i + 1}. {m}" for i, m in enumerate(issues))
            logger.debug(f"PlanActAgent.{self.name}: plan rejected, issues:\n{issues_msg}")
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
        ``task.regenerations_used``) is exhausted. Logs every raw generated
        plan at ``DEBUG`` level (see ``_process_generation_output`` for the
        matching rejected-issues log).
        """
        additional_messages: list[dict[str, str]] = []

        while True:
            messages = self.render_task(task, additional_messages=additional_messages)
            schema = build_planact_schema(self._toolbox.keys())
            engine_result = self._llm_engine.invoke(
                {"messages": messages, "output_structure": schema}
            )
            raw_output: dict[str, Any] = engine_result.result
            logger.debug(
                f"PlanActAgent.{self.name}: generated plan:\n{json.dumps(raw_output, indent=2)}"
            )

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
        ``async_invoke`` for the engine call. Same ``DEBUG``-level logging of
        every raw generated plan."""
        additional_messages: list[dict[str, str]] = []

        while True:
            messages = self.render_task(task, additional_messages=additional_messages)
            schema = build_planact_schema(self._toolbox.keys())
            engine_result = await self._llm_engine.async_invoke(
                {"messages": messages, "output_structure": schema}
            )
            raw_output: dict[str, Any] = engine_result.result
            logger.debug(
                f"PlanActAgent.{self.name}: generated plan:\n{json.dumps(raw_output, indent=2)}"
            )

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
        ``task.pending`` exists or the task is already complete.
        ``task.completed`` alone can never independently distinguish this:
        whenever ``task.pending`` drains to empty, either ``task.complete``
        becomes ``True`` (the return call executed) or ``_check_plan_exhausted``
        already raised -- ``act()`` never returns in an intermediate state
        where ``completed`` is non-empty but ``pending``/``complete`` both
        say "keep going".
        """
        if task.pending or task.complete:
            return task

        task.pending = self._run_planning_retry_loop(task=task)
        task.task_messages.clear()
        return task

    async def async_think(self, task: PlanActTask) -> PlanActTask:
        """Async mirror of ``think``."""
        if task.pending or task.complete:
            return task

        task.pending = await self._arun_planning_retry_loop(task=task)
        task.task_messages.clear()
        return task

    # ------------------------------------------------------------------ #
    # Prepare next batch
    # ------------------------------------------------------------------ #
    def _record_failures(
        self, task: PlanActTask, failed: list[ToolStatement], *, executing: bool,
    ) -> None:
        """
        The one failure policy for ``prepare`` (``executing=False``) and
        ``_apply_batch_results`` (``executing=True``). The caller has already
        set ``call.exception`` on every call in ``failed`` (never empty).

        1. ``fail_fast=True``: raise. While executing, the first failure's
           ``ToolInvocationError`` is re-raised as-is, anything else is
           wrapped in ``ToolAgentError``; while resolving, one
           ``ToolAgentError`` lists every failure.
        2. ``fail_fast=False``: record onto ``task.failed_statements`` and
           cascade-skip every later-batch call that transitively depends on a
           failed identifier. Skipped calls are never added to ``completed``
           or ``failed_statements``.
        """
        if self._fail_fast:
            if executing:
                first = failed[0]
                if isinstance(first.exception, ToolInvocationError):
                    raise first.exception
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: tool call failed "
                    f"for {first.tool!r} (identifier={first.identifier!r}): {first.exception}"
                ) from first.exception
            issues_msg = "\n".join(f"{i + 1}. {c.exception}" for i, c in enumerate(failed))
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: batch could not be resolved:\n{issues_msg}"
            )

        task.failed_statements.extend(failed)
        failed_identifiers = {c.identifier for c in failed if c.identifier is not None}
        poisoned = find_cascade_failures(failed_identifiers, task.pending[1:])
        self._drop_poisoned_calls(task.pending[1:], poisoned)

    def prepare(self, task: PlanActTask) -> PlanActTask:
        """
        Resolve the next pending batch's args. This family has no
        continuation round to fall back to on a resolution failure:

        1. If ``task.pending`` is empty: reset ``task.resolved_args`` and
           return unchanged (``act()``'s own final-batch check raises if the
           return call never executed).
        2. For every call in ``task.pending[0]``: resolve its args against
           the values in ``task.context`` *and* bind them against the target
           tool's real parameter contract (``tool._args_kwargs_to_dict``). A
           failure sets ``call.exception`` and joins ``failed``; a success
           keeps the call and its bound dict.
        3. Any failures go through ``_record_failures`` (raises under
           ``fail_fast``; otherwise records and cascade-skips later batches).
        4. The surviving calls and bound dicts become ``pending[0]`` and
           ``task.resolved_args`` -- possibly empty, which ``act()`` treats
           as a legal no-op.
        """
        if not task.pending:
            task.resolved_args = []
            return task

        batch = task.pending[0]
        namespace = {n: v.value for n, v in task.context.items()}

        survivors: list[ToolStatement] = []
        resolved: list[dict[str, Any]] = []
        failed: list[ToolStatement] = []

        for call in batch:
            label = call.identifier if call.identifier is not None else "(unassigned)"
            try:
                positional, keyword = resolve_statement_args(call, namespace)
                if call.tool == RETURN_ALIAS:
                    bound = {RETURN_VALUE_FIELD: keyword[RETURN_VALUE_FIELD]}
                else:
                    bound = self.get_tool(call.tool)._args_kwargs_to_dict(*positional, **keyword)
            except Exception as e:
                call.exception = ToolAgentError(
                    f"{label}: could not resolve or bind argument(s): {e!r}"
                )
                failed.append(call)
                continue
            survivors.append(call)
            resolved.append(bound)

        if failed:
            self._record_failures(task, failed, executing=False)

        task.pending[0] = survivors
        task.resolved_args = resolved
        return task

    async def async_prepare(self, task: PlanActTask) -> PlanActTask:
        """Direct passthrough -- ``prepare`` has no I/O of its own."""
        return self.prepare(task)

    @staticmethod
    def _drop_poisoned_calls(pending: list[list[ToolStatement]], poisoned: set[str]) -> None:
        """
        Remove, in place, every call from every batch in ``pending`` whose
        ``args``/``kwargs`` reference (via ``extract_identifiers``) any name
        in ``poisoned`` -- named or not. ``poisoned`` is a filter key
        (``find_cascade_failures``'s own return contract), not a list of
        calls to remove directly.
        """
        for batch in pending:
            survivors = []
            for call in batch:
                refs = set(extract_identifiers(call.args)) | set(extract_identifiers(call.kwargs))
                if not (refs & poisoned):
                    survivors.append(call)
            batch[:] = survivors

    # ------------------------------------------------------------------ #
    # Execute prepared batch
    # ------------------------------------------------------------------ #
    async def _gather_batch_results(
        self, batch: list[Any], resolved: list[dict[str, Any]],
    ) -> list[Any]:
        """Dispatch every dispatched call in ``batch`` concurrently via
        ``asyncio.gather``; a non-dispatched (``RETURN_ALIAS``) entry's
        result is already its resolved ``val``, no dispatch needed."""
        coros: list[Any] = []
        dispatch_map: dict[int, int] = {}
        for i, call in enumerate(batch):
            if is_dispatched(call):
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
        Shared post-gather bookkeeping for ``act``/``async_act``:

        1. Partition ``raw_results`` into failures (a dispatched call whose
           result is a ``BaseException``; ``call.exception`` is set) and
           successes.
        2. Successes join ``task.completed``; each named result is bound, and
           a ``RETURN_ALIAS`` success sets ``generated_response`` and
           ``complete``.
        3. Bound names are written to ``task.context`` in one
           ``_write_context`` (creating new variables), before the failure
           branch so a partly failed batch keeps its successes.
        4. Failures go through ``_record_failures``.
        5. Pop the batch; if ``task.pending`` is now empty and the task is not
           complete, ``_check_plan_exhausted`` raises.
        """
        failed: list[ToolStatement] = []
        bound: dict[str, Any] = {}
        for call, kwargs, raw in zip(batch, resolved, raw_results):
            if is_dispatched(call) and isinstance(raw, BaseException):
                call.exception = raw
                failed.append(call)
                continue
            task.completed.append(call)
            if call.tool != RETURN_ALIAS:
                call.result = raw
                unwrapped = raw.result
            else:
                unwrapped = raw
            if call.identifier is not None:
                bound[call.identifier] = unwrapped
            if call.tool == RETURN_ALIAS:
                task.generated_response = kwargs[RETURN_VALUE_FIELD]
                task.complete = True

        self._write_context(task, bound, create=True)

        if failed:
            self._record_failures(task, failed, executing=True)

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
                f"{render_failed_as_json(task.failed_statements)}"
            )

    def act(self, task: PlanActTask) -> PlanActTask:
        """
        Execute the currently prepared batch, or consume a batch that
        cascade-filtering emptied out entirely before it ever reached
        dispatch. Without popping it here, `prepare()` would keep
        re-processing the same now-empty batch forever.
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
    def _build_record_from_task(self, task: PlanActTask) -> ToolAgentRecord:
        """
        Assemble a completed ``ToolAgentRecord`` from a finished
        ``PlanActTask``. No agent-level global blackboard to persist into
        -- each record owns its own calls outright.
        """
        return ToolAgentRecord(
            **self._record_base_fields(task),
            statements=tuple(task.completed),
            failed_statements=tuple(task.failed_statements),
            regenerations_used=task.regenerations_used,
        )

    def build_result_from_task(self, task: PlanActTask, result: Any) -> ToolAgentResult:
        """
        Construct this agent's ``ToolAgentResult`` envelope directly from a
        finished ``PlanActTask`` (independent of the record). Derives
        ``usage_report`` via ``build_usage_report`` and ``failed_call_count``
        from ``len(task.failed_statements)``.
        """
        llm_token_usage = tuple(r.llm_result.token_usage for r in task.llm_records)
        llm_model_data = self._llm_engine.model_data

        return self._make_result(
            result=result,
            started_at=task.started_at,
            ended_at=task.ended_at,
            result_cls=ToolAgentResult,
            run_id=task.run_id,
            llm_token_usage=llm_token_usage,
            llm_model_data=llm_model_data,
            usage_report=build_usage_report(task.completed, task.failed_statements),
            failed_call_count=len(task.failed_statements),
            regenerations_used=task.regenerations_used,
        )
