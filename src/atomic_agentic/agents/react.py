"""
ReActAgent: one Python statement per round.

Each round the model writes a statement in a fenced ``python`` block; only the
first statement is used (later ones are dropped and logged, never
regenerated). The agent validates it, runs it, and shows the outcome next
round through the runtime-context block; the run ends only when the model
writes ``return``. A cousin of ``PlanActAgent`` (one generated plan, many
statements): it shares that family's statement parser, allow-lists, validator
and renderers (``utils/script.py``), collapsed to one statement per round,
with no sigil code.

Statement forms: a bare ``call(...)``, ``x = <expression>``, ``return`` and
``return <expression>``. An expression may contain calls (a registered tool,
an allowed builtin, a method on any value) anywhere, including inside other
calls, alongside literals, names, subscripts, attribute reads, container
literals, f-strings, binary and unary operators, ``not`` and comparisons.
``and``/``or``, ternaries, comprehensions, lambdas, ``await`` and control flow
are rejected.

A statement's calls are hoisted and run in order within the round, innermost
first and left to right; every call runs, nothing short-circuits. Hoisted
results live in a per-statement scratch namespace and never reach the context;
a bare call's non-``None`` result is auto-named ``__rN__``. A failure stops the
statement: calls already run stay committed, the rest are dropped.

``tool_calls_limit`` counts actions: one per call run (a failed call included),
and one for a call-free ``x = <expression>``; the expression around calls and
a ``return`` are free. Failures cost a regeneration before a statement runs
(parse, reference, binding and budget errors, inside ``think()``) and a
recorded failed statement after (a call, resolution or evaluation error);
``fail_fast=False`` shows it next round, ``True`` raises at the failure site.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import replace
from typing import Any, Callable, Mapping, Optional

from .toolagent import ToolAgent
from .tools import attr_call_tool, builtin_call_tool
from ..a2a import A2AClientHub
from ..constants.agents import (
    ATTR_CALL_ALIAS,
    PY_BUILTIN_ALIAS,
    RETURN_ALIAS,
    RHS_ASSIGN_ALIAS,
    SUB_NAME_PREFIX,
)
from ..constants.prompts import BUILTINS_PREAMBLE, DEFAULT_PRE_INVOKE, REACTOR_PROMPT
from ..exceptions import (
    AgentInvocationError,
    BlackboardParseError,
    ToolAgentError,
    ToolInvocationError,
)
from ..invokables import AtomicInvokable
from ..llm.base import LLMEngine
from ..mcp import MCPClientHub
from ..models.agents.blackboard_models import ToolStatement
from ..models.agents.records import LLMRecord, ToolAgentRecord
from ..models.agents.tasks import ReActTask
from ..utils.agents import (
    check_call_bindings,
    extract_identifiers,
    is_dispatched,
    resolve_statement_args,
)
from ..utils.script import (
    check_script_tool_name,
    dispatch_inputs,
    parse_first_statement,
    rewrite_builtin_calls,
    script_context_clashes,
    validate_references,
)

logger = logging.getLogger(__name__)


class ReActAgent(ToolAgent):
    """
    Iterative agent that runs one Python statement per round (see the module
    docstring for the grammar and the budget rule).
    """

    _TASK_CLS = ReActTask
    TOOL_INSTRUCTIONS = REACTOR_PROMPT

    def __init__(
        self,
        name: str,
        description: str,
        llm_engine: LLMEngine,
        instructions: Optional[str | Callable[..., Any] | AtomicInvokable] = None,
        tools: Optional[list[AtomicInvokable | Callable | MCPClientHub | A2AClientHub]] = None,
        pre_invoke: str | Callable[..., Any] | AtomicInvokable = DEFAULT_PRE_INVOKE,
        post_invoke: Optional[str | Callable[..., Any] | AtomicInvokable] = None,
        inputs_to_context: Iterable[str] | Mapping[str, str] = (),
        response_to_context: str | Iterable[str] | Mapping[Any, str] = (),
        response_to_post: str | Iterable[str] | Mapping[Any, str] = (),
        use_runtime_context: bool = False,
        record_enabled: bool = True,
        records_window: Optional[int] = None,
        *,
        tool_calls_limit: Optional[int] = 25,
        regeneration_limit: int = 5,
        fail_fast: bool = False,
    ) -> None:
        """
        Everything but ``fail_fast`` forwards to ``ToolAgent.__init__``; the
        concurrency limit is fixed at ``1`` (one statement per round).
        ``tool_calls_limit`` defaults to ``25`` (the only run-level stop, since
        ``regeneration_limit`` restarts each round) and ``fail_fast`` to
        ``False`` (redirecting on failure is this family's point).
        """
        super().__init__(
            name=name,
            description=description,
            llm_engine=llm_engine,
            record_enabled=record_enabled,
            tool_calls_limit=tool_calls_limit,
            regeneration_limit=regeneration_limit,
            tool_concurrency_limit=1,
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

    @property
    def fail_fast(self) -> bool:
        """``True`` raises at the first failed statement; ``False`` (default)
        records it and shows it next round so the model can redirect."""
        return self._fail_fast

    def to_dict(self) -> dict[str, Any]:
        """Extends ``ToolAgent.to_dict()`` with ``fail_fast``."""
        d = super().to_dict()
        d["fail_fast"] = self._fail_fast
        return d

    # ------------------------------------------------------------------ #
    # Shared Python-grammar hooks (same checks as PlanActAgent)
    # ------------------------------------------------------------------ #
    def _render_actions(self) -> str:
        """The tools frame body plus the permitted Python builtins text."""
        return f"{super()._render_actions()}\n\n{BUILTINS_PREAMBLE}"

    def _validate_tool_name(self, name: str) -> None:
        """Rejects names a statement could never reach."""
        super()._validate_tool_name(name)
        check_script_tool_name(name)

    def _seed_task(self, task: ReActTask) -> None:
        """Adds the ``task_result_i`` entries, then rejects any context
        variable name a statement could not use unambiguously."""
        super()._seed_task(task)
        clashes = script_context_clashes(task.context, self._toolbox)
        if clashes:
            detail = "; ".join(f"{n!r} is {why}" for n, why in clashes.items())
            raise AgentInvocationError(
                f"{type(self).__name__}.{self.name}: context variable name(s) "
                f"unusable in a script: {detail}. Rename or forget the "
                "variable, or rename the tool with to_tool."
            )

    @staticmethod
    def _action_cost(slot: ToolStatement) -> int:
        """
        Actions one slot spends. A dispatched slot (tool, builtin, method)
        costs 1. A pure assignment costs 1 only when its value references no
        hoisted ``_SUB_`` name (a statement with calls is charged on its call
        slots, so the expression around them is free). A ``return`` is free.
        """
        if is_dispatched(slot):
            return 1
        if slot.tool == RHS_ASSIGN_ALIAS:
            names = extract_identifiers(slot.kwargs["val"])
            return 0 if any(n.startswith(SUB_NAME_PREFIX) for n in names) else 1
        return 0

    def _actions_used(self, task: ReActTask) -> int:
        """Actions spent this run: ``_action_cost`` over every completed and
        failed slot."""
        return sum(
            self._action_cost(s) for s in (*task.completed, *task.failed_statements)
        )

    def _resolve_dispatch_tool(self, tool_id: str) -> AtomicInvokable:
        """The invokable that runs a dispatched slot: a builtin or attribute
        dispatcher, else the registered tool."""
        if tool_id == PY_BUILTIN_ALIAS:
            return builtin_call_tool
        if tool_id == ATTR_CALL_ALIAS:
            return attr_call_tool
        return self.get_tool(tool_id)

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #
    def _render_steps(self, task: ReActTask) -> str:
        """
        The statements log: every completed statement as the model wrote it
        and, only when the last one failed, its failure note. ``""`` on round
        1. Values are not rendered here; they are in the runtime-context block.
        """
        if not task.history and not task.last_call_failed:
            return ""
        completed = "\n".join(task.history) or "(nothing completed yet)"
        steps = f"# STATEMENTS COMPLETED SO FAR:\n{completed}"
        if task.last_call_failed:
            steps += f"\n\n# YOUR LAST STATEMENT FAILED:\n{task.failure_note}"
        return steps

    def _render_directive(self, task: ReActTask) -> str:
        """The next-statement request (failure wording after a failure), then
        the live remaining-calls line when a limit is set."""
        base = (
            "Write the NEXT single statement for the current task, or write "
            "`return` with the answer if the task is complete."
        )
        directive = (
            "Your last statement failed -- see YOUR LAST STATEMENT FAILED above "
            f"for why. Adjust your approach. {base}"
            if task.last_call_failed
            else base
        )
        limit = self.tool_calls_limit
        if limit is not None:
            remaining = max(limit - self._actions_used(task), 0)
            line = f"Actions remaining: {remaining} of {limit}."
            if remaining == 0:
                line += " Only a return with no call in it is allowed now."
            directive += f"\n\n{line}"
        return directive

    # ------------------------------------------------------------------ #
    # Generation (think())
    # ------------------------------------------------------------------ #
    def _process_generation_output(
        self, raw_text: str, task: ReActTask,
    ) -> tuple[list[ToolStatement], str] | str:
        """
        Validate callback for the retry loop: parse the first statement into
        its slot queue, rewrite builtin calls, enforce the action budget
        against the whole queue, then check references and call bindings.
        Returns ``(slots, source)``, or a feedback string for regeneration.
        """
        try:
            slots, source, dropped = parse_first_statement(
                raw_text, start_index=len(task.completed) + len(task.failed_statements),
            )
        except BlackboardParseError as e:
            return str(e)
        if dropped:
            logger.debug(
                f"ReActAgent.{self.name}: ignored {dropped} statement(s) after the first"
            )

        rewrite_builtin_calls(slots)

        limit = self._tool_calls_limit
        if limit is not None:
            cost = sum(self._action_cost(s) for s in slots)
            remaining = max(limit - self._actions_used(task), 0)
            if cost > remaining:
                if remaining == 0:
                    return (
                        "no actions remain in the budget; the only statement "
                        "allowed now is a return with no call in it."
                    )
                return (
                    f"this statement needs {cost} action(s) but {remaining} "
                    "remain; split it into smaller statements, or finish with "
                    "a return that has no call in it."
                )

        readonly_names = frozenset(n for n, v in task.context.items() if v.readonly)
        issues = (
            validate_references(
                slots, frozenset(self._toolbox), frozenset(task.context), readonly_names, None,
            )
            + check_call_bindings(slots, self._toolbox)
        )
        if issues:
            return "\n".join(f"{i + 1}. {m}" for i, m in enumerate(issues))

        return slots, source

    def _feedback_messages(self, raw_output: str, feedback: str) -> list[dict[str, str]]:
        """The assistant/user pair that asks for a corrected statement."""
        return [
            {"role": "assistant", "content": raw_output},
            {"role": "user", "content": (
                f"Your statement could not be used:\n\n{feedback}\n\n"
                "Write a corrected statement."
            )},
        ]

    def _run_step_retry_loop(self, *, task: ReActTask) -> tuple[list[ToolStatement], str]:
        """
        Render, call the engine, record the attempt, validate and resolve via
        ``_process_generation_output``, and retry with injected feedback until
        success or the regeneration budget (``self._regeneration_limit``,
        counted per generation; ``task.regenerations_used`` keeps the run
        total) is exhausted.
        """
        additional_messages: list[dict[str, str]] = []
        rejections = 0

        while True:
            messages = self.render_task(task, additional_messages=additional_messages)
            engine_result = self._llm_engine.invoke({"messages": messages})
            raw_output: str = engine_result.result
            logger.debug(f"ReActAgent.{self.name}: generated statement:\n{raw_output}")

            task.llm_records.append(LLMRecord(
                messages=list(task.task_messages),
                llm_result=engine_result,
            ))

            result = self._process_generation_output(raw_output, task)
            if isinstance(result, str):
                logger.debug(
                    f"ReActAgent.{self.name}: statement rejected ({rejections + 1}):\n"
                    f"{raw_output.strip()}\n-> {result}"
                )
                if rejections >= self._regeneration_limit:
                    raise ToolAgentError(
                        f"{type(self).__name__}.{self.name}: regeneration budget "
                        f"exhausted after {rejections + 1} attempt(s). "
                        f"Last feedback: {result}"
                    )
                additional_messages = self._feedback_messages(raw_output, result)
                rejections += 1
                task.regenerations_used += 1
                continue

            return result

    async def _arun_step_retry_loop(self, *, task: ReActTask) -> tuple[list[ToolStatement], str]:
        """Async mirror of ``_run_step_retry_loop``: ``async_invoke`` for the
        engine call, otherwise identical."""
        additional_messages: list[dict[str, str]] = []
        rejections = 0

        while True:
            messages = self.render_task(task, additional_messages=additional_messages)
            engine_result = await self._llm_engine.async_invoke({"messages": messages})
            raw_output: str = engine_result.result
            logger.debug(f"ReActAgent.{self.name}: generated statement:\n{raw_output}")

            task.llm_records.append(LLMRecord(
                messages=list(task.task_messages),
                llm_result=engine_result,
            ))

            result = self._process_generation_output(raw_output, task)
            if isinstance(result, str):
                logger.debug(
                    f"ReActAgent.{self.name}: statement rejected ({rejections + 1}):\n"
                    f"{raw_output.strip()}\n-> {result}"
                )
                if rejections >= self._regeneration_limit:
                    raise ToolAgentError(
                        f"{type(self).__name__}.{self.name}: regeneration budget "
                        f"exhausted after {rejections + 1} attempt(s). "
                        f"Last feedback: {result}"
                    )
                additional_messages = self._feedback_messages(raw_output, result)
                rejections += 1
                task.regenerations_used += 1
                continue

            return result

    def think(self, task: ReActTask) -> ReActTask:
        """Generate and validate this round's statement without running it;
        stashes its slot queue and source for ``act()``."""
        task.pending, task.source = self._run_step_retry_loop(task=task)
        task.task_messages.clear()
        return task

    async def async_think(self, task: ReActTask) -> ReActTask:
        """Async mirror of ``think``."""
        task.pending, task.source = await self._arun_step_retry_loop(task=task)
        task.task_messages.clear()
        return task

    def prepare(self, task: ReActTask) -> ReActTask:
        """No-op: validation and resolution already happened in ``think()``."""
        return task

    async def async_prepare(self, task: ReActTask) -> ReActTask:
        """Direct passthrough to ``prepare``."""
        return self.prepare(task)

    # ------------------------------------------------------------------ #
    # Execute (act())
    # ------------------------------------------------------------------ #
    def _prepare_slot(
        self, task: ReActTask, slot: ToolStatement, scratch: dict[str, Any],
    ) -> tuple[Optional[AtomicInvokable], Any]:
        """Resolve ``slot``'s arguments against the context plus the statement's
        scratch results. Returns ``(tool, dispatch inputs)`` for a call, or
        ``(None, value)`` for an expression or ``return``."""
        namespace = {n: v.value for n, v in task.context.items()} | scratch
        positional, keyword = resolve_statement_args(slot, namespace)
        if not is_dispatched(slot):
            return None, keyword["val"]
        tool = self._resolve_dispatch_tool(slot.tool)
        return tool, dispatch_inputs(slot, tool, positional, keyword)

    def _on_slot_success(
        self, task: ReActTask, slot: ToolStatement, raw: Any, scratch: dict[str, Any],
    ) -> Any:
        """Commit one successful slot and return its value: route it (scratch
        for a hoisted result, ``__rN__`` for an unnamed non-``None`` call
        result, the context for a named one), log it, and finish the run on a
        ``return``."""
        dispatched = is_dispatched(slot)
        if dispatched:
            slot.result = raw
            value = raw.result
        else:
            value = raw
        if slot.identifier is None and dispatched and value is not None:
            slot.identifier = f"__r{len(task.history)}__"
        task.completed.append(slot)
        if slot.identifier is not None:
            if slot.identifier.startswith(SUB_NAME_PREFIX):
                scratch[slot.identifier] = value
            else:
                self._write_context(task, {slot.identifier: value}, create=True)
        logger.info(
            f"ReActAgent.{self.name}: ran {slot.to_code()}"
            f"{'' if slot.identifier is None else f'  -> {value!r:.80}'}"
        )
        if slot.tool == RETURN_ALIAS:
            task.generated_response = value
            task.complete = True
        return value

    def _on_slot_failure(
        self,
        task: ReActTask,
        slot: ToolStatement,
        error: BaseException,
        ran: list[tuple[ToolStatement, Any]],
        total: int,
    ) -> None:
        """Record a failed slot: ``fail_fast`` raises; otherwise the slot joins
        ``failed_statements``, the failure note is built and the rest of the
        queue is dropped."""
        if self._fail_fast:
            if isinstance(error, ToolInvocationError):
                raise error
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: statement failed at "
                f"{slot.to_code()!r}: {error}"
            ) from error
        slot.exception = error
        task.failed_statements.append(slot)
        task.last_call_failed = True
        if total == 1:
            task.failure_note = f"{task.source}  # FAILED: {error!r}"
        else:
            note = (
                f"{task.source}\n# sub-step `{replace(slot, identifier=None).to_code()}` "
                f"failed: {error!r}"
            )
            if ran:
                done = ", ".join(
                    f"`{replace(s, identifier=None).to_code()}` -> {v!r:.80}" for s, v in ran
                )
                note += f"\n# sub-steps that already ran: {done}"
            task.failure_note = note
        task.pending.clear()
        logger.info(f"ReActAgent.{self.name}: FAILED {slot.to_code()}  # {error!r}")

    def _finish_statement(self, task: ReActTask) -> None:
        """A statement whose whole queue succeeded joins ``history``."""
        task.history.append(task.source)
        task.last_call_failed = False
        task.failure_note = ""
        task.source = ""

    def act(self, task: ReActTask) -> ReActTask:
        """
        Run this round's slot queue, front to back: resolve each slot against
        the context plus this statement's scratch results, dispatch it (or
        evaluate its expression), and commit or fail it. A failure stops the
        statement; a clean finish records it in ``history``.
        """
        scratch: dict[str, Any] = {}
        ran: list[tuple[ToolStatement, Any]] = []
        total = len(task.pending)
        while task.pending:
            slot = task.pending.pop(0)
            try:
                tool, inputs = self._prepare_slot(task, slot, scratch)
                raw = inputs if tool is None else tool.invoke(inputs)
            except Exception as e:
                self._on_slot_failure(task, slot, e, ran, total)
                return task
            ran.append((slot, self._on_slot_success(task, slot, raw, scratch)))
        self._finish_statement(task)
        return task

    async def async_act(self, task: ReActTask) -> ReActTask:
        """Async mirror of ``act``: ``async_invoke`` for the dispatch."""
        scratch: dict[str, Any] = {}
        ran: list[tuple[ToolStatement, Any]] = []
        total = len(task.pending)
        while task.pending:
            slot = task.pending.pop(0)
            try:
                tool, inputs = self._prepare_slot(task, slot, scratch)
                raw = inputs if tool is None else await tool.async_invoke(inputs)
            except Exception as e:
                self._on_slot_failure(task, slot, e, ran, total)
                return task
            ran.append((slot, self._on_slot_success(task, slot, raw, scratch)))
        self._finish_statement(task)
        return task

    # ------------------------------------------------------------------ #
    # Record construction
    # ------------------------------------------------------------------ #
    def _build_record_from_task(self, task: ReActTask) -> ToolAgentRecord:
        """Assemble the completed ``ToolAgentRecord`` from a finished task."""
        return ToolAgentRecord(
            **self._record_base_fields(task),
            statements=tuple(task.completed),
            failed_statements=tuple(task.failed_statements),
            regenerations_used=task.regenerations_used,
        )
