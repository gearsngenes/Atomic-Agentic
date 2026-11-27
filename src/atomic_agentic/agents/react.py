"""
ReActAgent: one Python statement per round.

Each round the model writes a statement in a fenced ``python`` block; only the
first statement is used (later ones are dropped and logged, never
regenerated). The agent validates it, runs it, and shows the outcome next
round through the runtime-context block; the run ends only when the model
writes ``return``. A cousin of ``PlanActAgent`` (one generated plan, many
statements): it shares that family's statement core (``utils/statements.py``:
compile, check, run, cost and render) collapsed to one statement per round.

Statement forms: a bare ``call(...)``, ``x = <expression>``, ``return`` and
``return <expression>``. An expression may contain calls (a registered tool,
an allowed builtin, a method on any value) anywhere, including inside other
calls, alongside literals, names, subscripts, attribute reads, container
literals, f-strings, binary and unary operators, ``not`` and comparisons.
``and``/``or``, ternaries, comprehensions, lambdas, ``await`` and control flow
are rejected.

A statement's calls are steps that run in order within the round, innermost
first and left to right; every call runs, nothing short-circuits. Hoisted
results live in a per-statement scratch namespace and never reach the context;
a bare call's non-``None`` result is auto-named ``__rN__``. A failure stops the
statement: the steps already run keep their results in the recorded failed
statement, the rest stay not run.

``tool_calls_limit`` counts actions: a statement costs ``max(1, tool steps)``
(a failed call counts), and a ``return`` is exempt from the minimum, so an
agent at its limit can still finish. Failures cost a regeneration before a
statement runs (parse, reference, binding and budget errors, inside
``think()``) and a recorded failed statement after (a call, resolution or
evaluation error); ``fail_fast=False`` shows it next round, ``True`` records it
and then raises at the failure site.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any, Callable, Mapping, Optional

from .toolagent import ToolAgent
from ..a2a import A2AClientHub
from ..constants.prompts import DEFAULT_PRE_INVOKE, REACTOR_PROMPT
from ..exceptions import (
    BlackboardParseError,
    ToolAgentError,
    ToolInvocationError,
)
from ..invokables import AtomicInvokable
from ..llm.base import LLMEngine
from ..mcp import MCPClientHub
from ..models.agents.blackboard_models import Statement
from ..models.agents.records import LLMRecord, ToolAgentRecord
from ..models.agents.tasks import ReActTask
from ..utils.statements import (
    async_run_statement,
    check_statements,
    compile_statements,
    planned_cost,
    render_statement_failure,
    run_statement,
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
        Everything forwards to ``ToolAgent.__init__`` (one statement runs per
        round, so there is no concurrency knob). ``tool_calls_limit`` defaults
        to ``25`` (the only run-level stop, since ``regeneration_limit``
        restarts each round) and ``fail_fast`` to ``False`` (redirecting on
        failure is this family's point).
        """
        super().__init__(
            name=name,
            description=description,
            llm_engine=llm_engine,
            record_enabled=record_enabled,
            tool_calls_limit=tool_calls_limit,
            regeneration_limit=regeneration_limit,
            fail_fast=fail_fast,
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

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #
    def _render_steps(self, task: ReActTask) -> str:
        """
        The statements log: every completed statement as the model wrote it
        and, only when the last one failed, its failure note. ``""`` on round
        1. Values are not rendered here; they are in the runtime-context block.
        """
        if not task.completed and not task.failure_note:
            return ""
        completed = "\n".join(statement.source for statement in task.completed)
        steps = f"# STATEMENTS COMPLETED SO FAR:\n{completed or '(nothing completed yet)'}"
        if task.failure_note:
            steps += f"\n\n# YOUR LAST STATEMENT FAILED:\n{task.failure_note}"
        return steps

    def _render_directive(self, task: ReActTask) -> str:
        """The next-statement request (failure wording after a failure), then
        the live remaining-actions line when a limit is set."""
        base = (
            "Write the NEXT single statement for the current task, or write "
            "`return` with the answer if the task is complete."
        )
        directive = (
            "Your last statement failed -- see YOUR LAST STATEMENT FAILED above "
            f"for why. Adjust your approach. {base}"
            if task.failure_note
            else base
        )
        limit = self.tool_calls_limit
        if limit is not None:
            remaining = max(limit - task.tool_calls_used, 0)
            line = f"Actions remaining: {remaining} of {limit}."
            if remaining == 0:
                line += " Only a return with no call in it is allowed now."
            directive += f"\n\n{line}"
        return directive

    # ------------------------------------------------------------------ #
    # Generation (think())
    # ------------------------------------------------------------------ #
    def _process_generation_output(self, raw_text: str, task: ReActTask) -> Statement | str:
        """
        Validate callback for the retry loop. Returns the compiled
        ``Statement``, or a feedback string for regeneration.

        1. Compile the first statement (``compile_statements(limit=1)``).
        2. Enforce the action budget against the statement's planned cost.
        3. Check references and call bindings (``check_statements``).
        """
        try:
            statements, dropped = compile_statements(
                raw_text, limit=1,
                start_ordinal=len(task.completed) + len(task.failed_statements),
            )
        except BlackboardParseError as e:
            return str(e)
        statement = statements[0]
        if dropped:
            logger.debug(
                f"ReActAgent.{self.name}: ignored {dropped} statement(s) after the first"
            )

        limit = self._tool_calls_limit
        if limit is not None:
            cost = planned_cost(statement)
            remaining = max(limit - task.tool_calls_used, 0)
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
        issues = check_statements(
            [statement], self._toolbox, frozenset(task.context), readonly_names, None,
        )
        if issues:
            return "\n".join(f"{i + 1}. {m}" for i, m in enumerate(issues))

        return statement

    def _feedback_messages(self, raw_output: str, feedback: str) -> list[dict[str, str]]:
        """The assistant/user pair that asks for a corrected statement."""
        return [
            {"role": "assistant", "content": raw_output},
            {"role": "user", "content": (
                f"Your statement could not be used:\n\n{feedback}\n\n"
                "Write a corrected statement."
            )},
        ]

    def _run_step_retry_loop(self, *, task: ReActTask) -> Statement:
        """
        Render, call the engine, record the attempt, validate via
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

    async def _arun_step_retry_loop(self, *, task: ReActTask) -> Statement:
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
        stashes it on the task for ``act()``."""
        task.statement = self._run_step_retry_loop(task=task)
        task.task_messages.clear()
        return task

    async def async_think(self, task: ReActTask) -> ReActTask:
        """Async mirror of ``think``."""
        task.statement = await self._arun_step_retry_loop(task=task)
        task.task_messages.clear()
        return task

    def prepare(self, task: ReActTask) -> ReActTask:
        """No-op: validation already happened in ``think()``; arguments are
        resolved at execution."""
        return task

    async def async_prepare(self, task: ReActTask) -> ReActTask:
        """Direct passthrough to ``prepare``."""
        return self.prepare(task)

    # ------------------------------------------------------------------ #
    # Execute (act())
    # ------------------------------------------------------------------ #
    @staticmethod
    def _namespace(task: ReActTask) -> dict[str, Any]:
        """The context's current values, the snapshot a statement runs against."""
        return {name: variable.value for name, variable in task.context.items()}

    def _commit_statement(self, task: ReActTask, statement: Statement) -> None:
        """
        Commit one executed statement (sync and async share it).

        1. A failed statement joins ``failed_statements``; ``fail_fast``
           raises (a ``ToolInvocationError`` as is, anything else wrapped);
           otherwise ``failure_note`` is built from it.
        2. A done statement joins ``completed`` and clears the note; its
           target (or ``__rN__`` for a bare call with a non-``None`` value) is
           written to the context.
        3. A ``return`` sets the response and completes the task.
        """
        task.statement = None

        if statement.status == "failed":
            task.failed_statements.append(statement)
            step = next(s for s in statement.steps if s.status == "failed")
            logger.info(f"ReActAgent.{self.name}: FAILED {statement.source}  # {step.error!r}")
            if self._fail_fast:
                if isinstance(step.error, ToolInvocationError):
                    raise step.error
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: statement failed at "
                    f"{step.text!r}: {step.error}"
                ) from step.error
            task.failure_note = render_statement_failure(statement)
            return

        value = statement.value
        target = statement.target
        if target is None and statement.kind == "call" and value is not None:
            target = f"__r{len(task.completed)}__"
        task.completed.append(statement)
        task.failure_note = ""
        if target is not None:
            self._write_context(task, {target: value}, create=True)
        logger.info(
            f"ReActAgent.{self.name}: ran {statement.source}"
            f"{'' if target is None else f'  -> {value!r:.80}'}"
        )
        if statement.kind == "return":
            task.generated_response = value
            task.complete = True

    def act(self, task: ReActTask) -> ReActTask:
        """Run this round's statement against the context snapshot and
        commit it."""
        statement = run_statement(task.statement, self._namespace(task), self._resolve_dispatch_tool)
        self._commit_statement(task, statement)
        return task

    async def async_act(self, task: ReActTask) -> ReActTask:
        """Async mirror of ``act``: ``async_invoke`` for the dispatch."""
        statement = await async_run_statement(
            task.statement, self._namespace(task), self._resolve_dispatch_tool,
        )
        self._commit_statement(task, statement)
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
