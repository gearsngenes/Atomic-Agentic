"""
ReActAgent: Iterative LLM Actor with Per-Step Reactive Planning

Generates, resolves, and dispatches exactly one registered-tool call per
round via provider-native structured output (``REACT_OUTPUT_SCHEMA``),
observes the result, and repeats until the model calls the registered
``return`` tool. Design lineage: mirrors ``PlanActAgent``'s own
``$name``-sigil machinery (``utils/sigils.py``, reused directly) collapsed to
a single call per round instead of a whole batched plan -- the per-step
sibling to ``PlanActAgent``'s one-shot multi-call plan. See
``agents/planact.py``'s own module docstring for the shared design lineage
this mirrors.

Values live in ``task.context``: pre-existing context variables are
referenced as ``$name``, and each successful call's ``result_name`` (or an
auto-assigned ``__rN__``) is written there, so the next round sees the
observation as an ordinary context entry in the runtime-context block. The
steps block shows only the call log and the last failure, never values.

Distinguishing features relative to ``PlanActAgent``:
- ``return`` is an ordinary, really-dispatched tool call (``return_tool``,
  registered under its own bare name) -- never a synthesized,
  non-dispatched ``RETURN_ALIAS`` sentinel, and ``act()``/``async_act()``
  never gate *dispatch* on ``is_dispatched`` (every call, including
  ``return``, is always actually invoked). That claim is scoped to dispatch
  specifically: ``RETURN_TOOL_NAME`` and ``RETURN_ALIAS`` are the identical
  literal string ``"return"`` (`constants/agents.py`), so anywhere else
  ``is_dispatched`` is consulted for *accounting* purposes (see
  ``build_result_from_task`` below), a real, successful ``return`` call
  is still indistinguishable from the sentinel and gets excluded --
  correctly, per ``ToolUsageRecord.call_count``'s own "non-return
  executions" contract, but via the same string coincidence, not a
  deliberate discriminator.
- ``think()`` owns validation AND argument resolution in one retry loop;
  ``prepare()`` is a documented no-op. Exactly one call is generated per
  round and every value it may reference already exists at generation time,
  so a bad resolution (an unbound reference, a value of the wrong type for
  the target tool) is cheap regeneration feedback -- no round and no
  tool-call budget is spent on it.
- No batching, no gathering, no cascade machinery -- exactly one call is
  dispatched per round, and a tolerated failure (``fail_fast=False``) is
  simply information the model sees and redirects around next round.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from typing import Any, Callable, ClassVar, Mapping, Optional

from .toolagent import ToolAgent
from .prompts import REACT_PROMPT
from .tools import make_dict, make_sequence, return_tool
from ..constants.agents import RETURN_TOOL_NAME
from ..core import AtomicInvokable
from ..llm.base import LLMEngine
from ..exceptions import ToolAgentError, ToolInvocationError
from ..mcp import MCPClientHub
from ..a2a import A2AClientHub, PyA2AtomicClient
from ..models.agents.blackboard_models import ToolStatement
from ..models.agents.prompts import PromptConfig
from ..models.agents.tasks import ReActTask
from ..models.agents.records import ToolAgentRecord, LLMRecord
from ..models.results.agents import ToolAgentResult
from ..utils.agents import build_usage_report, resolve_statement_args
from ..utils.sigils import (
    build_react_schema,
    parse_react_call,
    render_completed_as_json,
    render_failed_as_json,
    translate_calls,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# ReAct Agent
# --------------------------------------------------------------------------- #
class ReActAgent(ToolAgent):
    """
    Iterative agent with reactive step-by-step planning (ReAct-style
    architecture): one call generated, resolved, and dispatched per round,
    observed before the next round's decision.

    Advantages
    ~~~~~~~~~~
    - Fully adaptive: each step can react to prior tool results, including
      failures -- ``fail_fast=False`` (this family's own default) makes a
      failure simply a redirect signal, not an abort.
    - Schema-guaranteed shape: ``output_structure`` strict mode makes a
      malformed-shape generation structurally impossible.

    Limitations
    ~~~~~~~~~~~
    - Higher latency: one LLM call per step, no concurrency.
    - Step quality depends on the model's ability to select the next best
      action from the rendered history alone.
    - ``tool_calls_limit=None`` removes the only run-level stop: nothing else
      ends a run whose model never calls ``return`` (``regeneration_limit``
      restarts each round, so it does not bound the run). The default is 25;
      pass ``None`` only when the model is trusted to finish.
    """

    # A class attribute (active before __init__ runs) — return_tool plus the
    # two composite-value-building utility tools this family's schema text
    # tells the model to use. Reserved so no caller can register a different
    # tool under any of these ids, or remove/replace them once seeded.
    _RESERVED_TOOL_NAMES: ClassVar[frozenset[str]] = frozenset(
        {RETURN_TOOL_NAME, "make_sequence", "make_dict"}
    )

    # This family's $name-sigil grammar (utils/sigils.py): context entries
    # render as "$name", history labels as "$task_result_i".
    _CONTEXT_NAME_PREFIX: str = "$"

    _TASK_CLS = ReActTask
    TOOL_INSTRUCTIONS = REACT_PROMPT

    def __init__(
        self,
        name: str,
        namespace: str,
        description: str,
        llm_engine: LLMEngine,
        record_enabled: bool = False,
        *,
        tool_calls_limit: Optional[int] = 25,
        regeneration_limit: int = 5,
        fail_fast: bool = False,
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
        There is no ``tool_concurrency_limit`` parameter: exactly one call is
        dispatched per round, so the base's limit is fixed at ``1``.
        ``fail_fast`` has no shared home on ``ToolAgent`` (see the
        ``fail_fast`` property below) -- validated and stored locally
        instead. Two defaults deliberately diverge from ``ToolAgent``'s own:
        ``tool_calls_limit`` defaults to ``25`` (not ``None`` -- this family
        has no second round-ceiling knob the way ``ScriptActAgent`` has
        ``replanning_limit``, so an unbounded default would have zero
        structural backstop), and ``fail_fast`` defaults to ``False`` (not
        ``True`` -- redirecting on failure is this family's entire reason
        for existing; a caller who genuinely wants one failure to be fatal
        can still pass ``fail_fast=True`` explicitly).
        """
        super().__init__(
            name=name,
            namespace=namespace,
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

        # Seeded directly, bypassing register_tool -- all three are reserved
        # (self._RESERVED_TOOL_NAMES), and register_tool now rejects any
        # attempt to register something under a reserved id, including this
        # one. return_tool's own bare name already equals RETURN_TOOL_NAME;
        # naming it explicitly here decouples seeding from that incidental
        # fact.
        self._seed_reserved_tool(return_tool, RETURN_TOOL_NAME)
        self._seed_reserved_tool(make_sequence, "make_sequence")
        self._seed_reserved_tool(make_dict, "make_dict")

    @property
    def fail_fast(self) -> bool:
        """
        A failure with ``fail_fast=True`` raises immediately, aborting the
        run. With ``fail_fast=False`` (this family's own default) it is
        recorded and the model sees it next round to redirect around --
        this family's entire reason for existing. Declared directly on
        this class -- not shared with ``PlanActAgent``, whose own
        ``fail_fast`` governs a whole batch's cascade behavior, not a
        single per-round call.
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
    def _render_steps(self, task: ReActTask) -> str:
        """
        The call log: every call completed so far as wire-schema JSON and,
        only when the last call failed, that one failed call with its error.
        ``""`` on round 1 (nothing completed, no failure), so the base emits
        no steps block. Values are not rendered here; they are in the
        runtime-context block.

        Only the single most recent failure is shown, never the accumulated
        history: ``task.last_call_failed`` answers "is there anything to
        react to" precisely (False the instant a later call succeeds), so
        the render is gated on it, not on ``failed_statements`` being
        non-empty.
        """
        if not task.completed and not task.last_call_failed:
            return ""
        completed = render_completed_as_json(task.completed) or "(nothing completed yet)"
        steps = f"# STEPS COMPLETED SO FAR:\n{completed}"
        if task.last_call_failed:
            steps += (
                "\n\n# YOUR LAST CALL FAILED:\n"
                f"{render_failed_as_json([task.failed_statements[-1]])}"
            )
        return steps

    def _render_directive(self, task: ReActTask) -> str:
        """
        The closing instruction: the next-call request (with failure wording
        after a tolerated failure), then the live remaining-budget line when
        a limit is set. Recomputed every round, since the figure depletes.
        Enforcement itself is structural (the schema enum narrows to
        ``return`` only at the boundary, see ``_run_step_retry_loop``); the
        text only lets the model see it coming.
        """
        base = (
            "Produce the NEXT BEST single tool call for the current task, "
            "or call the return tool if the task is complete."
        )
        directive = (
            "Your last call failed -- see YOUR LAST CALL FAILED above for "
            f"why. Adjust your approach. {base}"
            if task.last_call_failed
            else base
        )
        limit = self.tool_calls_limit
        if limit is not None:
            remaining = max(limit - task.tool_calls_used, 0)
            line = f"Tool calls remaining: {remaining} of {limit}."
            if remaining == 0:
                line += " Only the return tool is available now."
            directive += f"\n\n{line}"
        return directive

    # ------------------------------------------------------------------ #
    # Generation (think())
    # ------------------------------------------------------------------ #
    def _process_generation_output(
        self, raw_output: dict[str, Any], task: ReActTask,
    ) -> tuple[ToolStatement, dict[str, Any]] | str:
        """
        Pure-computation validate-and-resolve callback for the step retry
        loop: parse (``parse_react_call``, raw JSON -> draft), translate
        (``translate_calls``, draft -> validated ``ToolStatement`` against
        the names in ``task.context``, the registered tool ids and the remaining
        tool-call budget),
        then attempt argument resolution against the context values -- the
        one failure category ``translate_calls`` structurally can't catch (a
        resolved value's real type mismatching the target tool's parameter
        contract). Returns ``(call, resolved_kwargs)`` on success, or a
        feedback string describing the problem on failure.
        """
        draft = parse_react_call(raw_output)

        remaining_budget = (
            None if self._tool_calls_limit is None
            else self._tool_calls_limit - task.tool_calls_used
        )
        context_names = frozenset(task.context)
        readonly_names = frozenset(n for n, v in task.context.items() if v.readonly)
        issues, calls = translate_calls(
            [draft], remaining_budget, context_names, readonly_names,
            frozenset(self._toolbox),
        )

        if issues:
            issues_msg = "\n".join(f"{i + 1}. {m}" for i, m in enumerate(issues))
            logger.debug(f"ReActAgent.{self.name}: step rejected, issues:\n{issues_msg}")
            return issues_msg

        call = calls[0]
        try:
            namespace = {n: v.value for n, v in task.context.items()}
            positional, keyword = resolve_statement_args(call, namespace)
            tool = self.get_tool(call.tool)
            resolved = tool._args_kwargs_to_dict(*positional, **keyword)
        except Exception as e:
            # Fed back to the model as regeneration feedback; the traceback is
            # kept at DEBUG so a genuine framework bug is not lost.
            logger.debug(
                f"ReActAgent.{self.name}: could not resolve or bind step", exc_info=True
            )
            label = call.identifier if call.identifier is not None else "(unassigned)"
            return f"{label}: could not resolve or bind argument(s): {e!r}"

        return call, resolved

    def _run_step_retry_loop(self, *, task: ReActTask) -> tuple[ToolStatement, dict[str, Any]]:
        """
        Render, compute the schema's tool set, call the engine (with
        ``output_structure``), record the attempt, validate+resolve via
        ``_process_generation_output``, and retry with injected feedback on
        failure until success or the regeneration budget
        (``self._regeneration_limit``, counted per
        generation; ``task.regenerations_used`` keeps the run total) is exhausted. Mirrors ``PlanActAgent``'s
        own ``DEBUG``-level logging of every raw generated step (see
        ``_process_generation_output`` for the matching rejected-issues log).
        """
        additional_messages: list[dict[str, str]] = []
        rejections = 0

        while True:
            messages = self.render_task(task, additional_messages=additional_messages)

            tool_names = (
                [RETURN_TOOL_NAME]
                if (
                    self._tool_calls_limit is not None
                    and task.tool_calls_used >= self._tool_calls_limit
                )
                else self._toolbox.keys()
            )
            schema = build_react_schema(tool_names)

            engine_result = self._llm_engine.invoke(
                {"messages": messages, "output_structure": schema}
            )
            raw_output: dict[str, Any] = engine_result.result
            logger.debug(
                f"ReActAgent.{self.name}: generated step:\n{json.dumps(raw_output, indent=2)}"
            )

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
                additional_messages = [
                    {"role": "assistant", "content": json.dumps(raw_output)},
                    {"role": "user", "content": (
                        f"Your step could not be used:\n\n{result}\n\n"
                        "Produce a corrected step."
                    )},
                ]
                rejections += 1
                task.regenerations_used += 1
                continue

            return result

    async def _arun_step_retry_loop(self, *, task: ReActTask) -> tuple[ToolStatement, dict[str, Any]]:
        """Async mirror of ``_run_step_retry_loop``, using
        ``async_invoke`` for the engine call. Same regeneration-budget
        check, feedback-message construction, and ``DEBUG``-level logging
        of every raw generated step."""
        additional_messages: list[dict[str, str]] = []
        rejections = 0

        while True:
            messages = self.render_task(task, additional_messages=additional_messages)

            tool_names = (
                [RETURN_TOOL_NAME]
                if (
                    self._tool_calls_limit is not None
                    and task.tool_calls_used >= self._tool_calls_limit
                )
                else self._toolbox.keys()
            )
            schema = build_react_schema(tool_names)

            engine_result = await self._llm_engine.async_invoke(
                {"messages": messages, "output_structure": schema}
            )
            raw_output: dict[str, Any] = engine_result.result
            logger.debug(
                f"ReActAgent.{self.name}: generated step:\n{json.dumps(raw_output, indent=2)}"
            )

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
                additional_messages = [
                    {"role": "assistant", "content": json.dumps(raw_output)},
                    {"role": "user", "content": (
                        f"Your step could not be used:\n\n{result}\n\n"
                        "Produce a corrected step."
                    )},
                ]
                rejections += 1
                task.regenerations_used += 1
                continue

            return result

    def think(self, task: ReActTask) -> ReActTask:
        """
        Generate and validate this round's single call, without dispatching
        it. Stashes the resolved decision onto ``task.generated_step``/
        ``task.resolved_args`` for ``act()`` to run.
        """
        call, resolved = self._run_step_retry_loop(task=task)
        task.generated_step = call
        task.resolved_args = resolved
        task.task_messages.clear()
        return task

    async def async_think(self, task: ReActTask) -> ReActTask:
        """Async mirror of ``think``, using ``_arun_step_retry_loop``."""
        call, resolved = await self._arun_step_retry_loop(task=task)
        task.generated_step = call
        task.resolved_args = resolved
        task.task_messages.clear()
        return task

    # ------------------------------------------------------------------ #
    # Prepare (no-op for this family)
    # ------------------------------------------------------------------ #
    def prepare(self, task: ReActTask) -> ReActTask:
        """
        Documented no-op -- all validation/resolution already happened
        inside ``think()``'s own retry loop (see the module docstring for
        why); nothing is deferred to this hook for this family. Unlike
        ``PlanActAgent``, whose ``prepare()`` resolves a batch compiled by an
        earlier ``think()`` call, this family's ``think()`` and ``prepare()``
        are logically inseparable -- one call, resolved immediately.
        """
        return task

    async def async_prepare(self, task: ReActTask) -> ReActTask:
        """Direct passthrough -- ``prepare`` has no I/O of its own to
        justify a thread offload."""
        return self.prepare(task)

    # ------------------------------------------------------------------ #
    # Execute (act())
    # ------------------------------------------------------------------ #
    def _apply_call_result(
        self, task: ReActTask, call: ToolStatement, raw_result: Any,
    ) -> ReActTask:
        """
        Shared post-dispatch bookkeeping for ``act``/``async_act``.

        A failure with ``fail_fast=True`` raises immediately (preserving
        ``ToolInvocationError`` as-is, wrapping anything else in
        ``ToolAgentError``); with ``fail_fast=False`` it is recorded into
        ``task.failed_statements`` and ``task.last_call_failed`` is set, so
        the model sees it next round and redirects -- no cascade machinery
        of any kind. Every successful call, including a call to
        ``return_tool``, is unwrapped and recorded identically -- there is
        no synthesized, never-dispatched sentinel in this family to skip.

        Two identifier normalizations happen on a successful call, before
        it's appended to ``task.completed``: a ``return_tool`` call's
        identifier is always forced to ``None`` (matches ``RETURN_ALIAS``'s
        own always-``None`` convention on ``PlanActAgent`` -- a ``return``
        call ends the task, so there is no later round for any
        name of its to ever be referenced in); any other call whose
        identifier is still ``None`` and whose real result isn't ``None``
        gets auto-named ``__r{N}__`` (``N`` = count of successful calls
        so far), so a result the model forgot to name stays referenceable
        instead of becoming permanently unrecoverable the moment this round
        ends -- this family has no lookahead across rounds the way a
        one-shot planner does, so "forgot to name it" would otherwise never
        be repairable.

        A named result is written to ``task.context`` (creating the
        variable, or updating a writable existing one); persistence follows
        the normal destination rule.
        """
        if isinstance(raw_result, BaseException):
            if self._fail_fast:
                if isinstance(raw_result, ToolInvocationError):
                    raise raw_result
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: tool call failed "
                    f"for {call.tool!r} (identifier={call.identifier!r}): {raw_result}"
                ) from raw_result

            call.exception = raw_result
            task.failed_statements.append(call)
            task.last_call_failed = True
        else:
            call.result = raw_result
            unwrapped = raw_result.result

            if call.tool == RETURN_TOOL_NAME:
                call.identifier = None
            elif call.identifier is None and unwrapped is not None:
                call.identifier = f"__r{len(task.completed)}__"

            task.completed.append(call)
            if call.identifier is not None:
                self._write_context(task, {call.identifier: unwrapped}, create=True)
            task.last_call_failed = False

            if call.tool == RETURN_TOOL_NAME:
                task.generated_response = unwrapped
                task.complete = True

        task.generated_step = None
        task.resolved_args = None
        return task

    def act(self, task: ReActTask) -> ReActTask:
        """
        Dispatch this round's one resolved call for real -- no gathering,
        no ``asyncio``/``run_coro_sync`` of any kind, since exactly one
        call is ever dispatched per round.
        """
        call = task.generated_step
        resolved = task.resolved_args
        tool = self.get_tool(call.tool)
        try:
            raw_result = tool.invoke(resolved)
        except Exception as e:
            raw_result = e
        return self._apply_call_result(task, call, raw_result)

    async def async_act(self, task: ReActTask) -> ReActTask:
        """Async mirror of ``act``."""
        call = task.generated_step
        resolved = task.resolved_args
        tool = self.get_tool(call.tool)
        try:
            raw_result = await tool.async_invoke(resolved)
        except Exception as e:
            raw_result = e
        return self._apply_call_result(task, call, raw_result)

    # ------------------------------------------------------------------ #
    # Record / result construction
    # ------------------------------------------------------------------ #
    def _build_record_from_task(self, task: ReActTask) -> ToolAgentRecord:
        """
        Assemble a completed ``ToolAgentRecord`` from a finished
        ``ReActTask``. Mirrors ``PlanActAgent._build_record_from_task``
        exactly -- no agent-level global blackboard to persist into.
        """
        return ToolAgentRecord(
            **self._record_base_fields(task),
            statements=tuple(task.completed),
            failed_statements=tuple(task.failed_statements),
            regenerations_used=task.regenerations_used,
        )

    def build_result_from_task(self, task: ReActTask, result: Any) -> ToolAgentResult:
        """
        Construct this agent's ``ToolAgentResult`` envelope directly from a
        finished ``ReActTask``. Mirrors ``PlanActAgent.build_result_from_task``
        exactly.
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
