"""
ReAct2Agent: one Python statement per round.

Each round the model writes exactly one statement in a fenced ``python`` block.
The agent validates it, executes it, and shows the outcome next round through
the runtime-context block; the run ends only when the model writes ``return``.
A cousin of ``ScriptActAgent`` (one generated plan, many statements): it shares
that family's statement parser, allow-lists, validator and renderers
(``utils/script.py``), collapsed to one statement per round, with no sigil
code.

Statement shapes: ``x = call(...)``, a bare ``call(...)``, ``x = <expression>``
and ``return [<expression>]``. At most one call per statement, and it must be
the whole right-hand side or the whole statement; the call is a registered
tool, an allowed builtin or an allowed attribute method. A bare call's
non-``None`` result is auto-named ``__rN__`` so it stays referenceable.

``tool_calls_limit`` counts statements executed (pure assignments included,
``return`` excluded), unlike the dispatched-calls-only count the inherited
``tool_calls_used`` reports. Failures cost a regeneration before a statement
runs (parse, reference, binding and resolution errors, inside ``think()``) and
a recorded failed statement after (the tool raised); ``fail_fast=False``
shows it next round, ``True`` raises at the failure site.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any, Callable, Mapping, Optional

from .toolagent import ToolAgent
from .tools import attr_call_tool, builtin_call_tool
from ..a2a import A2AClientHub
from ..constants.agents import ATTR_CALL_ALIAS, PY_BUILTIN_ALIAS, RETURN_ALIAS, RHS_ASSIGN_ALIAS
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
from ..utils.agents import check_call_bindings, is_dispatched, resolve_statement_args
from ..utils.script import (
    check_script_tool_name,
    dispatch_inputs,
    parse_one_statement,
    render_completed_as_python,
    render_failed_as_python,
    rewrite_builtin_calls,
    script_context_clashes,
    validate_references,
)

logger = logging.getLogger(__name__)


class ReAct2Agent(ToolAgent):
    """
    Iterative agent that runs one Python statement per round (see the module
    docstring for the grammar and the budget rule). Placeholder name: Pass 13
    renames it ``ReActAgent`` when the JSON agent is deleted.
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
    # Shared Python-grammar hooks (same checks as ScriptActAgent)
    # ------------------------------------------------------------------ #
    def _render_actions(self) -> str:
        """The tools frame body plus the permitted Python builtins text."""
        return f"{super()._render_actions()}\n\n{BUILTINS_PREAMBLE}"

    def _validate_tool_name(self, name: str) -> None:
        """Reserved names, plus names a statement could never reach."""
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
    def _statements_used(task: ReActTask) -> int:
        """Executed statements this run, ``return`` excluded."""
        return sum(
            1 for s in (*task.completed, *task.failed_statements) if s.tool != RETURN_ALIAS
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
        The statements log: every statement completed so far as Python source
        and, only when the last one failed, that one failure with its error.
        ``""`` on round 1. Values are not rendered here; they are in the
        runtime-context block.
        """
        if not task.completed and not task.last_call_failed:
            return ""
        completed = render_completed_as_python(task.completed) or "(nothing completed yet)"
        steps = f"# STATEMENTS COMPLETED SO FAR:\n{completed}"
        if task.last_call_failed:
            steps += (
                "\n\n# YOUR LAST STATEMENT FAILED:\n"
                f"{render_failed_as_python([task.failed_statements[-1]])}"
            )
        return steps

    def _render_directive(self, task: ReActTask) -> str:
        """The next-statement request (failure wording after a failure), then
        the live remaining-statements line when a limit is set."""
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
            remaining = max(limit - self._statements_used(task), 0)
            line = f"Statements remaining: {remaining} of {limit}."
            if remaining == 0:
                line += " Only a return statement is allowed now."
            directive += f"\n\n{line}"
        return directive

    # ------------------------------------------------------------------ #
    # Generation (think())
    # ------------------------------------------------------------------ #
    def _process_generation_output(
        self, raw_text: str, task: ReActTask,
    ) -> tuple[ToolStatement, dict[str, Any]] | str:
        """
        Validate-and-resolve callback for the retry loop: parse the one
        statement, rewrite a builtin call, enforce the statement budget, check
        references and call bindings, then resolve the arguments. Returns
        ``(statement, dispatch inputs)``, or a feedback string for
        regeneration.
        """
        try:
            slot = parse_one_statement(raw_text)
        except BlackboardParseError as e:
            return str(e)

        rewrite_builtin_calls([slot])

        limit = self._tool_calls_limit
        if limit is not None and limit - self._statements_used(task) <= 0 and slot.tool != RETURN_ALIAS:
            return "no statements remain in the budget; the only statement allowed now is return."

        readonly_names = frozenset(n for n, v in task.context.items() if v.readonly)
        issues = (
            validate_references(
                [slot], frozenset(self._toolbox), frozenset(task.context), readonly_names, None,
            )
            + check_call_bindings([slot], self._toolbox)
        )
        if issues:
            return "\n".join(f"{i + 1}. {m}" for i, m in enumerate(issues))

        label = slot.identifier if slot.identifier is not None else "(unassigned)"
        try:
            namespace = {n: v.value for n, v in task.context.items()}
            positional, keyword = resolve_statement_args(slot, namespace)
            if slot.tool in (RHS_ASSIGN_ALIAS, RETURN_ALIAS):
                inputs = {"val": keyword["val"]}
            else:
                tool = self._resolve_dispatch_tool(slot.tool)
                inputs = dispatch_inputs(slot, tool, positional, keyword)
        except Exception as e:
            # Fed back as regeneration feedback; the traceback stays at DEBUG
            # so a genuine framework bug is not lost.
            logger.debug(
                f"ReAct2Agent.{self.name}: could not resolve or bind statement", exc_info=True
            )
            return f"{label}: could not resolve or bind argument(s): {e!r}"

        return slot, inputs

    def _feedback_messages(self, raw_output: str, feedback: str) -> list[dict[str, str]]:
        """The assistant/user pair that asks for a corrected statement."""
        return [
            {"role": "assistant", "content": raw_output},
            {"role": "user", "content": (
                f"Your statement could not be used:\n\n{feedback}\n\n"
                "Write a corrected statement."
            )},
        ]

    def _run_step_retry_loop(self, *, task: ReActTask) -> tuple[ToolStatement, dict[str, Any]]:
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
            logger.debug(f"ReAct2Agent.{self.name}: generated statement:\n{raw_output}")

            task.llm_records.append(LLMRecord(
                messages=list(task.task_messages),
                llm_result=engine_result,
            ))

            result = self._process_generation_output(raw_output, task)
            if isinstance(result, str):
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

    async def _arun_step_retry_loop(self, *, task: ReActTask) -> tuple[ToolStatement, dict[str, Any]]:
        """Async mirror of ``_run_step_retry_loop``: ``async_invoke`` for the
        engine call, otherwise identical."""
        additional_messages: list[dict[str, str]] = []
        rejections = 0

        while True:
            messages = self.render_task(task, additional_messages=additional_messages)
            engine_result = await self._llm_engine.async_invoke({"messages": messages})
            raw_output: str = engine_result.result
            logger.debug(f"ReAct2Agent.{self.name}: generated statement:\n{raw_output}")

            task.llm_records.append(LLMRecord(
                messages=list(task.task_messages),
                llm_result=engine_result,
            ))

            result = self._process_generation_output(raw_output, task)
            if isinstance(result, str):
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
        stashes it and its dispatch inputs for ``act()``."""
        slot, inputs = self._run_step_retry_loop(task=task)
        task.generated_step = slot
        task.resolved_args = inputs
        task.task_messages.clear()
        return task

    async def async_think(self, task: ReActTask) -> ReActTask:
        """Async mirror of ``think``."""
        slot, inputs = await self._arun_step_retry_loop(task=task)
        task.generated_step = slot
        task.resolved_args = inputs
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
    def _apply_statement_result(
        self, task: ReActTask, slot: ToolStatement, raw: Any,
    ) -> ReActTask:
        """
        Post-execution bookkeeping shared by ``act``/``async_act``.

        1. A dispatched call that raised: ``fail_fast`` re-raises a
           ``ToolInvocationError`` as is and wraps anything else in
           ``ToolAgentError``; otherwise the statement joins
           ``task.failed_statements`` and ``task.last_call_failed`` is set.
        2. Otherwise a dispatched call's envelope is stored on the slot and its
           payload taken; a bare call with a non-``None`` payload is auto-named
           ``__rN__``; the statement joins ``task.completed`` and a named value
           is written to ``task.context``.
        3. A ``return`` sets the response and completes the task.
        """
        dispatched = is_dispatched(slot)
        if dispatched and isinstance(raw, BaseException):
            if self._fail_fast:
                if isinstance(raw, ToolInvocationError):
                    raise raw
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: tool call failed "
                    f"for {slot.tool!r} (identifier={slot.identifier!r}): {raw}"
                ) from raw
            slot.exception = raw
            task.failed_statements.append(slot)
            task.last_call_failed = True
        else:
            if dispatched:
                slot.result = raw
                value = raw.result
            else:
                value = raw
            if slot.identifier is None and dispatched and value is not None:
                slot.identifier = f"__r{len(task.completed)}__"
            task.completed.append(slot)
            if slot.identifier is not None:
                self._write_context(task, {slot.identifier: value}, create=True)
            task.last_call_failed = False
            if slot.tool == RETURN_ALIAS:
                task.generated_response = value
                task.complete = True

        task.generated_step = None
        task.resolved_args = None
        return task

    def act(self, task: ReActTask) -> ReActTask:
        """Run this round's one statement: dispatch the call, or take the
        already-resolved value of a pure expression or ``return``."""
        slot = task.generated_step
        inputs = task.resolved_args
        if is_dispatched(slot):
            tool = self._resolve_dispatch_tool(slot.tool)
            try:
                raw = tool.invoke(inputs)
            except Exception as e:
                raw = e
        else:
            raw = inputs["val"]
        return self._apply_statement_result(task, slot, raw)

    async def async_act(self, task: ReActTask) -> ReActTask:
        """Async mirror of ``act``."""
        slot = task.generated_step
        inputs = task.resolved_args
        if is_dispatched(slot):
            tool = self._resolve_dispatch_tool(slot.tool)
            try:
                raw = await tool.async_invoke(inputs)
            except Exception as e:
                raw = e
        else:
            raw = inputs["val"]
        return self._apply_statement_result(task, slot, raw)

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
