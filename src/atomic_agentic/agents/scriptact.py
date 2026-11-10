from __future__ import annotations

import asyncio
import builtins
import keyword
from collections.abc import Iterable
from typing import Any, Callable, Mapping, Optional

from ..mcp.MCPClientHub import MCPClientHub
from ..a2a.A2AClientHub import A2AClientHub
from ..a2a.PyA2AtomicClient import PyA2AtomicClient

from .toolagent import ToolAgent
from .prompts import ONESHOT_PLANNER_PROMPT
from .tools import attr_call_tool, builtin_call_tool
from ..core.Invokable import AtomicInvokable
from ..llm.base import LLMEngine
from ..models.agents.blackboard_models import ToolStatement
from ..models.agents.prompts import PromptConfig
from ..models.agents.records import LLMRecord, ScriptActAgentRecord
from ..models.agents.tasks import ScriptActAgentTask
from ..models.results.agents import ScriptActAgentResult
from ..constants.core import NO_VAL
from ..constants.agents import (
    ATTR_CALL_ALIAS,
    BUILTINS_PREAMBLE,
    EXCLUDED_PY_BUILTINS,
    PY_BUILTIN_ALIAS,
    RETURN_ALIAS,
    RHS_ASSIGN_ALIAS,
)
from ..exceptions import (
    AgentInvocationError,
    BlackboardParseError,
    ToolAgentError,
    ToolInvocationError,
    ToolRegistrationError,
)
from ..utils.agents import build_usage_report, check_call_bindings, compile_batches, is_dispatched, resolve_statement_args
from ..utils.core import run_coro_sync
from ..utils.script import (
    parse_generation,
    render_completed_as_python,
    render_failed_as_python,
    rewrite_builtin_calls,
    validate_references,
)


class ScriptActAgent(ToolAgent):
    """
    One-shot-planning tool-invoking agent -- a direct ``ToolAgent`` sibling
    of ``PlanActAgent``/``ReActAgent``. Writes native-grammar, Python-style
    statements toward a task from a single generated plan. There is no
    separate decomposition, orchestration, or synthesis call, and no
    construction-time mode knob.

    Failure-triggered repair only: a generation terminates at its first
    ``return`` (the grammar forbids conditionals outright; a model that
    writes one has its generation rejected and fed back for a corrected
    full plan -- never silently truncated). The plan is requested as a
    single fenced ``python`` block; prose around exactly one block is
    ignored. There is no voluntary continuation mechanism of any
    kind -- a model never decides to pause; the only way a second round can
    ever start is a real resolution failure (``prepare()``) or execution
    failure (``act()``). On either: ``fail_fast=True`` raises immediately,
    at the failure site, naming what failed; ``fail_fast=False`` (default)
    grants up to ``replanning_limit`` repair rounds first, each seeing a
    Python-source snapshot of the work already done this invoke plus
    exactly what failed and why in the triggering batch, and only raises
    (the same way ``fail_fast=True`` would have) once that budget is
    exhausted without recovering -- repair only ever buys a chance to avoid
    raising, never a way to return degraded output instead. `tool_calls_limit`
    is an optional, fully independent budget on total dispatched calls
    (tools and builtins counted identically) across every generation round
    in one invoke -- `None` (the default) means no such cap. `replanning_limit`
    (always a plain `int`, never `None`, minimum `0`, default `2`) separately
    bounds how many repair rounds may be granted -- the free initial plan
    never counts against it. `regeneration_limit` (always a plain `int`,
    never `None`, default 5) independently bounds how many times a single
    round's malformed/invalid draft may be regenerated before raising --
    distinct from `replanning_limit`, which governs genuine repair rounds
    after a real failure, not within-round mistake recovery ("how many
    second chances does one attempt get").

    Values live in ``task.context``: a plan may reference any context
    variable by name (conversation, runtime and input-bound values, a prior
    turn's result as the read-only ``task_result_i``, and every name an
    earlier statement of this invoke bound). Each executed batch writes its
    result names (and ``_SUB_n`` temporaries) into ``task.context``; a
    writable name is updated in place, and only declared destinations
    persist (see ``Agent.record``).

    Tool registration, ``instructions``, shared rendering, and the
    tool_calls_limit/tool_concurrency_limit/regeneration_limit knobs are
    all inherited from ``ToolAgent`` unchanged -- this class owns only its
    own generation/execution grammar and the ``replanning_limit``/
    ``fail_fast`` knobs, neither of which has an equivalent on the shared
    base (``fail_fast`` is independently declared here, same pattern as
    ``PlanActAgent``'s/``ReActAgent``'s own, different semantics).
    """

    _TASK_CLS = ScriptActAgentTask
    _INITIAL_PROMPT_NAME = "planner"
    _BUILTINS_PREAMBLE = BUILTINS_PREAMBLE

    def __init__(
        self,
        name: str,
        namespace: str,
        description: str,
        llm_engine: LLMEngine,
        record_enabled: bool = False,
        *,
        regeneration_limit: int = 5,
        tool_calls_limit: Optional[int] = None,
        replanning_limit: int = 2,
        fail_fast: bool = False,
        tool_concurrency_limit: Optional[int] = None,
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
        ``super().__init__`` reaches ``ToolAgent`` (tool registry init,
        ``instructions``, regeneration_limit/tool_calls_limit/
        tool_concurrency_limit validation+storage, construction-time tools
        registration all happen there) -- this ``__init__`` only handles what's
        genuinely local to this class: ``replanning_limit``, ``fail_fast``,
        and the ``"planner"`` system prompt. ``replanning_limit`` defaults
        to ``2`` -- a repairs-only budget never has to cover the free
        initial plan, so it doesn't need a large ceiling.
        ``fail_fast`` defaults to ``False`` -- repair is the normal path
        this class exists to offer; ``True`` is the opt-out for strict
        one-shot-or-die behavior.
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

        self.replanning_limit = replanning_limit

        if not isinstance(fail_fast, bool):
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: fail_fast must be a bool."
            )
        self._fail_fast = fail_fast

        self._system_prompts["planner"] = ONESHOT_PLANNER_PROMPT

    # ------------------------------------------------------------------ #
    # Construction-time / mutable knobs
    # ------------------------------------------------------------------ #
    @property
    def replanning_limit(self) -> int:
        """Max number of framework-granted repair rounds after the free
        initial plan -- always a plain ``int``, never ``None``: an
        unbounded repair budget on a failure that's already proven itself
        unrecoverable once is a runaway-cost risk, not a legitimate use
        case, now that continuation is never voluntary. Minimum ``0`` --
        a real, legal value meaning "one-shot only, zero tolerance for
        failure," not an edge case to special-case around. Mutable, like
        ``tool_calls_limit``/``tool_concurrency_limit`` on ``ToolAgent``."""
        return self._replanning_limit

    @replanning_limit.setter
    def replanning_limit(self, value: int) -> None:
        if type(value) is not int or value < 0:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: replanning_limit "
                f"must be an int >= 0; got {value!r}."
            )
        self._replanning_limit = value

    @property
    def fail_fast(self) -> bool:
        """``True``: the first resolution or execution failure raises
        immediately, at the failure site -- no repair attempted,
        ``replanning_limit`` irrelevant in this mode. ``False`` (default):
        up to ``replanning_limit`` repair rounds are attempted first; if
        the budget is exhausted without recovering, raises the same way
        ``True`` would have -- there is no silent partial-success path
        either way. Read-only after construction, same shape as
        ``PlanActAgent``'s/``ReActAgent``'s own independently-declared
        ``fail_fast`` properties (no setter on either) -- different
        semantics (this one governs repair-round budget exhaustion, not
        batch cascade or single-call tolerance)."""
        return self._fail_fast

    def to_dict(self) -> dict[str, Any]:
        """Extends ``ToolAgent.to_dict()`` with this class's own
        ``fail_fast``/``replanning_limit``."""
        d = super().to_dict()
        d["fail_fast"] = self._fail_fast
        d["replanning_limit"] = self._replanning_limit
        return d

    # ------------------------------------------------------------------ #
    # Tool registration -- naming-constraint hook override only
    # ------------------------------------------------------------------ #
    def _validate_effective_tool_id(self, effective_id: str) -> None:
        """
        Extends ``ToolAgent``'s default reserved-name check with this
        family's own sentinel/real-Python-builtin collision rules. Never a
        reserved parser sentinel (``RHS_ASSIGN_ALIAS``/``RETURN_ALIAS``/
        ``PY_BUILTIN_ALIAS``/``ATTR_CALL_ALIAS``), and never a real,
        non-excluded Python builtin name -- a builtin always resolves first
        (see ``utils/script.py``'s ``rewrite_builtin_calls``), so a tool
        registered under a colliding name would be permanently, silently
        unreachable rather than raising here. Nor a Python keyword: a script
        could never call it.
        """
        super()._validate_effective_tool_id(effective_id)

        if effective_id in (RHS_ASSIGN_ALIAS, RETURN_ALIAS, PY_BUILTIN_ALIAS, ATTR_CALL_ALIAS):
            raise ToolRegistrationError(
                f"effective id {effective_id!r} is reserved for the parser's "
                "own sentinel tool names and cannot be used as a registered "
                "tool alias or name."
            )
        if hasattr(builtins, effective_id) and effective_id not in EXCLUDED_PY_BUILTINS:
            raise ToolRegistrationError(
                f"effective id {effective_id!r} collides with a real Python "
                "builtin of the same name -- registering a tool under this "
                "name would make it permanently unreachable (the builtin "
                "always resolves first). Choose a different alias."
            )
        if keyword.iskeyword(effective_id):
            raise ToolRegistrationError(
                f"effective id {effective_id!r} is a Python keyword; a "
                "script could never call it. Choose a different alias."
            )

    def _seed_task(self, task: ScriptActAgentTask) -> None:
        """
        Adds the hidden ``task_result_i`` entries (``ToolAgent``), then
        rejects any context variable a script could not use unambiguously: a
        Python keyword (cannot be written as a name) or a registered tool id
        (a name in call position would resolve to the tool). Builtin and
        sentinel names are not clashes (value and call positions differ).
        Runs inside ``create_task``, before any model call.
        """
        super()._seed_task(task)
        clashes = {
            n: ("a Python keyword" if keyword.iskeyword(n) else "a registered tool id")
            for n in sorted(task.context)
            if keyword.iskeyword(n) or n in self._toolbox
        }
        if clashes:
            detail = "; ".join(f"{n!r} is {why}" for n, why in clashes.items())
            raise AgentInvocationError(
                f"{type(self).__name__}.{self.name}: context variable name(s) "
                f"unusable in a script: {detail}. Rename or forget the "
                "variable, or re-alias the tool."
            )

    # ------------------------------------------------------------------ #
    # Record construction
    # ------------------------------------------------------------------ #
    def _build_record_from_task(self, task: ScriptActAgentTask) -> ScriptActAgentRecord:
        """
        Assemble a completed ``ScriptActAgentRecord`` from a finished
        ``ScriptActAgentTask``. Each record owns its own slots outright, so
        this is a direct field copy.
        """
        return ScriptActAgentRecord(
            **self._record_base_fields(task),
            statements=tuple(task.completed),
            failed_statements=tuple(task.failed_statements),
            regenerations_used=task.regenerations_used,
            repair_rounds_used=task.repair_rounds_used,
        )

    def build_result_from_task(self, task: ScriptActAgentTask, result: Any) -> ScriptActAgentResult:
        """
        Construct this agent's ``ScriptActAgentResult`` envelope directly from
        a finished ``ScriptActAgentTask`` -- surfaces ``regenerations_used``/
        ``repair_rounds_used`` past the ephemeral task. Also populates
        ``usage_report`` (via ``build_usage_report``) and ``failed_call_count``,
        both required fields inherited from ``ToolAgentResult``.
        """
        llm_token_usage = tuple(r.llm_result.token_usage for r in task.llm_records)
        llm_model_data = self._llm_engine.model_data

        return self._make_result(
            result=result,
            started_at=task.started_at,
            ended_at=task.ended_at,
            result_cls=ScriptActAgentResult,
            run_id=task.run_id,
            llm_token_usage=llm_token_usage,
            llm_model_data=llm_model_data,
            usage_report=build_usage_report(task.completed, task.failed_statements),
            failed_call_count=len(task.failed_statements),
            regenerations_used=task.regenerations_used,
            repair_rounds_used=task.repair_rounds_used,
        )

    # ------------------------------------------------------------------ #
    # Task-lifecycle hooks
    # ------------------------------------------------------------------ #
    def _render_steps(self, task: ScriptActAgentTask) -> str:
        """
        Round 1 has no steps block. A repair round shows the completed work
        as Python source plus only the failed slice of the triggering batch
        (``task.failed_statements[task.repair_batch_start:]``), never the
        whole failure history. Bound values are not rendered here; they are
        in the context block.
        """
        if not task.needs_repair:
            return ""
        completed = render_completed_as_python(task.completed) or "(nothing completed yet)"
        failed = render_failed_as_python(task.failed_statements[task.repair_batch_start:])
        steps = f"# WORK COMPLETED SO FAR:\n{completed}"
        if failed:
            steps += f"\n\n# THIS BATCH FAILED:\n{failed}"
        return steps

    def _render_directive(self, task: ScriptActAgentTask) -> str:
        """
        Branches on ``task.needs_repair`` (reliably ``True`` for every repair
        round, including a failure on the very first batch). Round 1 states
        the total ``tool_calls_limit``; a repair round the remaining budget,
        recomputed fresh since it depletes.
        """
        limit = self.tool_calls_limit
        if not task.needs_repair:
            directive = "Write a plan to accomplish this task now."
            if limit is not None:
                plural = "s" if limit != 1 else ""
                directive += f"\n\nYou may make at most {limit} tool call{plural} total."
            return directive
        directive = (
            "Continue planning the rest of this task, fixing what failed "
            "above. Use the existing work done to guide you on what the "
            "next steps should be."
        )
        if limit is not None:
            directive += f" Tool calls remaining: {limit - task.tool_calls_used} of {limit}."
        return directive

    # ------------------------------------------------------------------ #
    # Generation (think())
    # ------------------------------------------------------------------ #
    def _process_generation_output(
        self, raw_text: str, task: ScriptActAgentTask,
    ) -> list[list[ToolStatement]] | str:
        """
        Pure-computation validate callback for the planning retry loop:
        parse, rewrite builtins, validate references + remaining tool-call
        budget, dry-bind registered-tool calls, and compile into batches. Returns the compiled result on success, or a
        feedback string describing every problem found on failure -- a
        ``BlackboardParseError`` from parsing is converted here, not
        propagated, so the retry loop can inject it as corrective feedback.
        """
        try:
            flat_slots = parse_generation(raw_text)
        except BlackboardParseError as e:
            return str(e)

        # Rewrite eligible builtin calls before whole-sequence validation,
        # so a rewritten slot is validated as PY_BUILTIN_ALIAS, not as an
        # unregistered tool; excluded-but-real builtin names are reported
        # here with a specific message instead of validate_references'
        # generic "unregistered tool" one.
        builtin_issues = rewrite_builtin_calls(flat_slots)
        context_names = frozenset(task.context)
        readonly_names = frozenset(n for n, v in task.context.items() if v.readonly)
        remaining_budget = (
            None if self._tool_calls_limit is None
            else self._tool_calls_limit - task.tool_calls_used
        )
        issues = (
            builtin_issues
            + validate_references(
                flat_slots, frozenset(self._toolbox), context_names,
                readonly_names, remaining_budget,
            )
            + check_call_bindings(flat_slots, self._toolbox)
        )

        if issues:
            issues_msg = "\n".join(f"{i + 1}. {m}" for i, m in enumerate(issues))
            return issues_msg

        pending = compile_batches(
            flat_slots,
            max_concurrency=self._tool_concurrency_limit,
            start_batch_index=task.batch_counter,
        )
        task.batch_counter += len(pending)
        return pending

    def _run_planning_retry_loop(
        self, *, task: ScriptActAgentTask,
    ) -> list[list[ToolStatement]]:
        """
        Render, call the engine, record the attempt, validate/compile via
        ``_process_generation_output``, and retry with injected feedback on
        failure until success or the regeneration budget
        (``self._regeneration_limit``, tracked via
        ``task.regenerations_used``) is exhausted. ``regeneration_limit``
        is always a plain ``int`` (never ``None``), so this check is a
        direct comparison -- no ``None``-guard needed, unlike
        ``tool_calls_limit`` elsewhere in this class.
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

            result = self._process_generation_output(raw_output, task)
            if isinstance(result, str):
                if task.regenerations_used >= self._regeneration_limit:
                    raise ToolAgentError(
                        f"{type(self).__name__}.{self.name}: regeneration budget "
                        f"exhausted after {task.regenerations_used + 1} attempt(s). "
                        f"Last feedback: {result}"
                    )
                additional_messages = [
                    {"role": "assistant", "content": raw_output},
                    {"role": "user", "content": (
                        f"Your plan could not be used:\n\n{result}\n\n"
                        "Produce a corrected plan."
                    )},
                ]
                task.regenerations_used += 1
                continue

            return result

    async def _arun_planning_retry_loop(
        self, *, task: ScriptActAgentTask,
    ) -> list[list[ToolStatement]]:
        """Async mirror of ``_run_planning_retry_loop``: uses
        ``async_invoke`` for the engine call, otherwise identical."""
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

            result = self._process_generation_output(raw_output, task)
            if isinstance(result, str):
                if task.regenerations_used >= self._regeneration_limit:
                    raise ToolAgentError(
                        f"{type(self).__name__}.{self.name}: regeneration budget "
                        f"exhausted after {task.regenerations_used + 1} attempt(s). "
                        f"Last feedback: {result}"
                    )
                additional_messages = [
                    {"role": "assistant", "content": raw_output},
                    {"role": "user", "content": (
                        f"Your plan could not be used:\n\n{result}\n\n"
                        "Produce a corrected plan."
                    )},
                ]
                task.regenerations_used += 1
                continue

            return result

    def _finalize_without_continuation(self, task: ScriptActAgentTask) -> ScriptActAgentTask:
        """A clean drain with no failure and no repair needed -- the
        absence of a ``return`` is not an invitation to keep planning.
        Infers ``None`` if nothing was ever returned and marks the task
        complete. Called only from ``act`` (an empty
        generation) and ``_apply_batch_results`` (the last batch of a plan
        draining) -- never from ``think`` or ``prepare``."""
        if task.generated_response is NO_VAL:
            task.generated_response = None
        task.complete = True
        return task

    def think(self, task: ScriptActAgentTask) -> ScriptActAgentTask:
        """
        Generate, validate, and compile the next segment of the plan --
        either the unconditional first generation, or (once a prior round's
        failure was granted a repair) a fresh continuation. No-op whenever
        there is still pending work to drain, or the task is already fully
        complete -- a repair round is requested by re-entering this same
        hook, not a separate mechanism.

        This hook has no budget awareness of any kind anymore: no ceiling
        check, no round counter of its own. ``replanning_limit`` is checked
        and ``task.repair_rounds_used`` is incremented entirely in
        ``_grant_repair_or_raise`` (via ``prepare()``/``_apply_batch_results()``) -- the same two places
        that decide whether a repair round is granted in the first place --
        so by the time this hook is ever re-entered, that decision has
        already been made; it just generates.

        ``task.needs_repair`` is cleared here, after generation, not before
        -- ``_render_steps``/``_render_directive`` (reached through
        ``render_task`` in ``_run_planning_retry_loop``) still needs to see it ``True`` to render this round's repair
        framing. Once a fresh round has actually been generated, the flag's
        job is done: whatever ``prepare()``/``_apply_batch_results`` do with
        this new content next is a normal round, not a still-pending repair
        -- leaving it set would make ``act`` a permanent no-op for the rest of
        the invoke, the moment a single repair round is ever granted.
        """
        if task.pending or task.complete:
            return task

        pending = self._run_planning_retry_loop(task=task)
        task.pending = pending
        task.needs_repair = False
        task.task_messages.clear()
        return task

    async def async_think(self, task: ScriptActAgentTask) -> ScriptActAgentTask:
        """Async mirror of ``think``, using ``_arun_planning_retry_loop``. See
        ``think()``'s own docstring for why ``needs_repair`` is cleared here,
        after generation."""
        if task.pending or task.complete:
            return task

        pending = await self._arun_planning_retry_loop(task=task)
        task.pending = pending
        task.needs_repair = False
        task.task_messages.clear()
        return task

    def _resolve_dispatch_tool(self, tool_id: str) -> AtomicInvokable:
        """Resolve a slot's ``tool`` id to the actual invokable to dispatch
        -- the shared ``PY_BUILTIN_ALIAS``/``ATTR_CALL_ALIAS``-vs-registered-
        tool selection used by ``prepare()`` and ``_gather_batch_results``."""
        if tool_id == PY_BUILTIN_ALIAS:
            return builtin_call_tool
        if tool_id == ATTR_CALL_ALIAS:
            return attr_call_tool
        return self.get_tool(tool_id)

    # ------------------------------------------------------------------ #
    # Prepare next batch
    # ------------------------------------------------------------------ #
    def prepare(self, task: ScriptActAgentTask) -> ScriptActAgentTask:
        """
        Resolution only: resolve the next pending batch's args against the
        context values; never finalizes. With nothing pending,
        ``task.resolved_args`` is reset and the task returned (``act``
        decides what an empty plan means).

        Otherwise resolves every slot's args in ``task.pending[0]``,
        collecting every failure (not stopping at the first). Any collected
        failure abandons this batch and every batch still queued after it
        (they may depend on bindings this batch was supposed to produce);
        ``_grant_repair_or_raise`` then grants a repair round or raises, per
        ``fail_fast``/``replanning_limit``. Nothing here was ever
        dispatched, so this does not consume ``tool_calls_used``.
        """
        if not task.pending:
            task.resolved_args = []
            return task

        batch = task.pending[0]
        resolved: list[dict[str, Any]] = []
        failed_slots: list[ToolStatement] = []
        resolution_namespace = {name: var.value for name, var in task.context.items()}
        for slot in batch:
            label = slot.identifier if slot.identifier is not None else "(unassigned)"

            try:
                positional, keyword = resolve_statement_args(slot, resolution_namespace)
            except Exception as e:
                slot.exception = ToolAgentError(
                    f"{label}: could not resolve argument value(s): {e!r}"
                )
                failed_slots.append(slot)
                continue

            if slot.tool in (RHS_ASSIGN_ALIAS, RETURN_ALIAS):
                resolved.append({"val": keyword["val"]})
                continue

            try:
                tool = self._resolve_dispatch_tool(slot.tool)
                # PY_BUILTIN_ALIAS/ATTR_CALL_ALIAS pack the real target
                # call's own trailing positional args/kwargs as opaque
                # tuple/dict values instead of splatting them -- splatting
                # would let a real call's own argument name (e.g. `name`,
                # `obj`, `method_name`) collide with the dispatcher's own
                # same-named parameter during binding (see agents/tools.py).
                if slot.tool == PY_BUILTIN_ALIAS:
                    resolved.append(
                        tool._args_kwargs_to_dict(positional[0], tuple(positional[1:]), keyword)
                    )
                elif slot.tool == ATTR_CALL_ALIAS:
                    resolved.append(
                        tool._args_kwargs_to_dict(
                            positional[0], positional[1], tuple(positional[2:]), keyword
                        )
                    )
                else:
                    resolved.append(tool._args_kwargs_to_dict(*positional, **keyword))
            except Exception as e:
                slot.exception = ToolAgentError(
                    f"{label}: argument(s) do not match {slot.tool!r}'s "
                    f"parameter contract: {e!r}"
                )
                failed_slots.append(slot)

        if failed_slots:
            return self._grant_repair_or_raise(task, failed_slots, executing=False)

        task.resolved_args = resolved
        return task

    async def async_prepare(self, task: ScriptActAgentTask) -> ScriptActAgentTask:
        """Direct passthrough to ``prepare`` -- no I/O of its own."""
        return self.prepare(task)

    # ------------------------------------------------------------------ #
    # Execute prepared batch
    # ------------------------------------------------------------------ #
    async def _gather_batch_results(
        self, batch: list[ToolStatement], resolved: list[dict[str, Any]],
    ) -> list[Any]:
        """
        Dispatch every dispatched-call slot in ``batch`` (registered tool
        or approved builtin, via ``is_dispatched``) concurrently;
        ``rhs_assign``/``return`` slots need no dispatch, their result is
        already the resolved ``"val"`` value. Shared by ``act``/
        ``async_act`` -- both differ only in how the resulting coroutine is
        driven.
        """
        coros: list[Any] = []
        dispatch_map: dict[int, int] = {}
        for i, slot in enumerate(batch):
            if is_dispatched(slot):
                dispatch_map[i] = len(coros)
                tool = self._resolve_dispatch_tool(slot.tool)
                coros.append(tool.async_invoke(resolved[i]))

        gathered = await asyncio.gather(*coros, return_exceptions=True) if coros else []
        return [
            gathered[dispatch_map[i]] if i in dispatch_map else resolved[i]["val"]
            for i in range(len(batch))
        ]

    def _apply_batch_results(
        self,
        task: ScriptActAgentTask,
        batch: list[ToolStatement],
        resolved: list[dict[str, Any]],
        raw_results: list[Any],
    ) -> ScriptActAgentTask:
        """
        Shared post-gather bookkeeping for ``act``/``async_act``: record
        results, bind every result name into ``task.context``, handle a
        terminal ``return`` slot, pop the consumed batch -- or, if any real
        call in this batch failed, abandon the rest of this round and either
        grant a repair round or raise (``_grant_repair_or_raise``).

        The context write happens once per batch, before the failure branch,
        so the successes of a partly failed batch stay bound for the repair
        round. ``task.tool_calls_used`` reflects ``completed``/
        ``failed_statements`` automatically.
        """
        # A raised exception only ever appears here for a slot that was
        # actually dispatched (asyncio.gather(..., return_exceptions=True)
        # is the only source of a bare BaseException in raw_results) -- an
        # rhs_assign/return slot's raw_results entry is always its plain
        # resolved value, which may itself legitimately BE a BaseException
        # instance (e.g. a context variable holding an exception object).
        # Gating on is_dispatched prevents misclassifying that value as an
        # execution failure.
        failed_slots: list[ToolStatement] = []
        bound: dict[str, Any] = {}

        for slot, value in zip(batch, raw_results):
            if is_dispatched(slot) and isinstance(value, BaseException):
                slot.exception = value
                failed_slots.append(slot)
                continue
            task.completed.append(slot)
            # A dispatched tool call's raw_results entry is a full
            # AtomicResult envelope (slot.result stores it verbatim);
            # rhs_assign/return slots were never dispatched, so their value
            # is already the plain resolved Python value.
            if slot.tool not in (RHS_ASSIGN_ALIAS, RETURN_ALIAS):
                slot.result = value
                unwrapped = value.result
            else:
                unwrapped = value
            if slot.identifier is not None:
                bound[slot.identifier] = unwrapped

        self._write_context(task, bound, create=True)

        if failed_slots:
            return self._grant_repair_or_raise(task, failed_slots, executing=True)

        for slot, kwargs in zip(batch, resolved):
            if slot.tool == RETURN_ALIAS:
                task.generated_response = kwargs["val"]
                task.complete = True

        task.pending.pop(0)
        task.resolved_args = []

        # The plan just drained with no failure and no `return`: finalize
        # here, the natural point `task.pending` reaches empty, instead of
        # letting `think()` regenerate an uninvited round on the next pass.
        if not task.pending and not task.complete:
            return self._finalize_without_continuation(task)

        return task

    def _grant_repair_or_raise(
        self,
        task: ScriptActAgentTask,
        failed_slots: list[ToolStatement],
        *,
        executing: bool,
    ) -> ScriptActAgentTask:
        """
        Shared failure branch of ``prepare`` (resolution, ``executing=False``)
        and ``_apply_batch_results`` (execution, ``executing=True``).

        Records ``failed_slots``. With ``fail_fast`` or the repair budget
        spent, raises: a lone execution failure that is a
        ``ToolInvocationError`` is re-raised as-is, anything else becomes a
        ``ToolAgentError`` naming what failed. Otherwise grants a repair
        round: consumes budget, marks where this batch's failures start
        (``repair_batch_start``), abandons the queued batches and sets
        ``needs_repair`` for ``think`` to regenerate.
        """
        start = len(task.failed_statements)
        task.failed_statements.extend(failed_slots)
        if self._fail_fast or task.repair_rounds_used >= self._replanning_limit:
            if (
                executing
                and len(failed_slots) == 1
                and isinstance(failed_slots[0].exception, ToolInvocationError)
            ):
                raise failed_slots[0].exception
            what = "batch execution failed" if executing else "batch could not be resolved"
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: {what} and no repair is "
                f"available ({'fail_fast=True' if self._fail_fast else 'repair budget exhausted'}): "
                f"{render_failed_as_python(failed_slots)}"
            )
        task.repair_rounds_used += 1
        task.repair_batch_start = start
        task.pending.clear()
        task.resolved_args = []
        task.needs_repair = True
        return task

    def act(self, task: ScriptActAgentTask) -> ScriptActAgentTask:
        """
        Execute the resolved batch. A just-granted repair is a no-op
        (``think`` regenerates next); nothing pending means an empty
        generation, which finalizes with a ``None`` result.
        """
        if task.needs_repair:
            return task
        if not task.pending:
            return self._finalize_without_continuation(task)

        batch = task.pending[0]
        resolved = task.resolved_args
        raw_results = run_coro_sync(self._gather_batch_results(batch, resolved))
        return self._apply_batch_results(task, batch, resolved, raw_results)

    async def async_act(self, task: ScriptActAgentTask) -> ScriptActAgentTask:
        """Async mirror of ``act``; awaits ``_gather_batch_results``
        directly rather than ``run_coro_sync``-wrapping it."""
        if task.needs_repair:
            return task
        if not task.pending:
            return self._finalize_without_continuation(task)

        batch = task.pending[0]
        resolved = task.resolved_args
        raw_results = await self._gather_batch_results(batch, resolved)
        return self._apply_batch_results(task, batch, resolved, raw_results)
