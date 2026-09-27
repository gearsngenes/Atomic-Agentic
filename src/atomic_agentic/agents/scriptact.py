from __future__ import annotations

import asyncio
import builtins
from datetime import datetime
from typing import Any, Callable, Optional

from ..mcp.MCPClientHub import MCPClientHub
from ..a2a.A2AClientHub import A2AClientHub
from ..a2a.PyA2AtomicClient import PyA2AtomicClient

from .toolagent import ToolAgent
from .prompts import ONESHOT_PLANNER_PROMPT
from .tools import attr_call_tool, builtin_call_tool
from ..core.Invokable import AtomicInvokable
from ..llm.base import LLMEngine
from ..models.agents.blackboard_models import CodeStatement
from ..models.agents.records import AgentRecord, LLMRecord, ScriptActAgentRecord
from ..models.agents.tasks import ScriptActAgentTask
from ..models.results.agents import ScriptActAgentResult
from ..constants.core import NO_VAL
from ..constants.agents import (
    ATTR_CALL_ALIAS,
    EXCLUDED_PY_BUILTINS,
    PY_BUILTIN_ALIAS,
    RETURN_ALIAS,
    RHS_ASSIGN_ALIAS,
)
from ..exceptions import (
    BlackboardParseError,
    ToolAgentError,
    ToolInvocationError,
    ToolRegistrationError,
)
from ..utils.core import run_coro_sync
from ..utils.script import (
    compile_batches,
    is_dispatched_slot,
    parse_generation,
    render_cache_snapshot,
    render_completed_as_python,
    render_failed_as_python,
    resolve_slot_args,
    rewrite_builtin_calls,
    validate_references,
)


class ScriptActAgent(ToolAgent):
    """
    One-shot-planning tool-invoking agent (renamed from ``ScriptAgent`` --
    now a direct ``ToolAgent`` sibling of ``PlanActAgent``/``ReActAgent``,
    closing the naming asymmetry now that all three sit at the same tier).
    Writes native-grammar, Python-style statements toward a task from a
    single generated plan. There is no separate decomposition,
    orchestration, or synthesis call, and no construction-time mode knob.

    Failure-triggered repair only (Pass 8): a generation always terminates
    at the first of a ``return`` or a defensively-pruned ``if`` statement
    (the grammar forbids conditionals outright; a model that writes one
    anyway always has its generation rejected as a regen-repair issue, fed
    back for a corrected full plan -- there is no silent-truncation
    tolerance for it). There is no voluntary continuation mechanism of any
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

    Cross-invocation result addressing is implemented: a prior turn's
    result is seeded into `task.cache` and labeled in rendered history as
    `task_result_i`, a fixed, read-only reference a later plan can use by
    name.

    Tool/constant registration, shared rendering, and the
    tool_calls_limit/tool_concurrency_limit/regeneration_limit knobs are
    all inherited from ``ToolAgent`` unchanged -- this class owns only its
    own generation/execution grammar and the ``replanning_limit``/
    ``fail_fast`` knobs, neither of which has an equivalent on the shared
    base (``fail_fast`` is independently declared here, same pattern as
    ``PlanActAgent``'s/``ReActAgent``'s own, different semantics).
    """

    def __init__(
        self,
        name: str,
        namespace: str,
        description: str,
        llm_engine: LLMEngine,
        context_enabled: bool = False,
        *,
        regeneration_limit: int = 5,
        tool_calls_limit: Optional[int] = None,
        replanning_limit: int = 2,
        fail_fast: bool = False,
        tool_concurrency_limit: Optional[int] = None,
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
        ``super().__init__`` reaches ``ToolAgent`` (tool/constant registry
        init, regeneration_limit/tool_calls_limit/tool_concurrency_limit
        validation+storage, construction-time tools/constants registration
        all happen there) -- this ``__init__`` only handles what's
        genuinely local to this class: ``replanning_limit``, ``fail_fast``,
        and the ``"planner"`` system prompt. ``replanning_limit`` defaults
        to ``2`` (not the old ``planning_rounds_limit``'s ``25`` -- a
        repairs-only budget doesn't need anywhere near that many chances,
        now that it no longer also has to cover the free initial plan).
        ``fail_fast`` defaults to ``False`` -- repair is the normal path
        this class exists to offer; ``True`` is the opt-out for strict
        one-shot-or-die behavior.
        """
        super().__init__(
            name=name,
            namespace=namespace,
            description=description,
            llm_engine=llm_engine,
            context_enabled=context_enabled,
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
        ``fail_fast``/``replanning_limit`` -- ``ScriptActAgent`` had no
        override at all before this pass."""
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
        unreachable rather than raising here.
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

    # ------------------------------------------------------------------ #
    # Record construction
    # ------------------------------------------------------------------ #
    def _build_record_from_task(
        self,
        task: ScriptActAgentTask,
        turns: list[AgentRecord],
    ) -> ScriptActAgentRecord:
        """
        Assemble a completed ``ScriptActAgentRecord`` from a finished
        ``ScriptActAgentTask``. No agent-level global blackboard to persist
        into (unlike v1 ``JsonToolAgent``'s span-tracking
        ``update_blackboard`` append) -- each record owns its own slots
        outright, so this is a direct field copy.
        """
        prev = turns[-1] if turns else None
        return ScriptActAgentRecord(
            user_prompt=task.user_prompt,
            generated_response=task.generated_response,
            inputs=task.inputs,
            llm_records=tuple(task.llm_records),
            prev=prev,
            statements=tuple(task.completed),
            failed_statements=tuple(task.failed_statements),
            regenerations_used=task.regenerations_used,
            repair_rounds_used=task.repair_rounds_used,
        )

    def build_result_from_record(
        self,
        record: ScriptActAgentRecord,
        *,
        result: Any,
        started_at: datetime,
        ended_at: datetime,
    ) -> ScriptActAgentResult:
        """
        Construct this agent's ``ScriptActAgentResult`` envelope directly
        from a completed ``ScriptActAgentRecord`` -- surfaces
        ``regenerations_used``/``repair_rounds_used`` past the ephemeral
        task, mirroring ``PlanActAgent.build_result_from_record``'s own
        pattern for ``regenerations_used``. ``ScriptActAgent`` had no
        override of this hook at all before now (a plain ``AgentResult``
        was returned, with neither counter visible past the task).
        """
        llm_token_usage = tuple(r.llm_result.token_usage for r in record.llm_records)
        llm_model_data = record.llm_records[-1].llm_result.model_data

        return self._make_result(
            result=result,
            started_at=started_at,
            ended_at=ended_at,
            result_cls=ScriptActAgentResult,
            llm_token_usage=llm_token_usage,
            llm_model_data=llm_model_data,
            regenerations_used=record.regenerations_used,
            repair_rounds_used=record.repair_rounds_used,
        )

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #
    def _extra_system_context(self) -> dict[str, str]:
        """
        Adds ``EXCLUDED_PY_BUILTINS`` on top of ``ToolAgent``'s shared
        ``{TOOLS}``/``{CONSTANTS}`` context -- the one piece of this
        family's system prompt that isn't generic across every ``ToolAgent``
        subclass.
        """
        return {"EXCLUDED_PY_BUILTINS": ", ".join(sorted(EXCLUDED_PY_BUILTINS))}

    # ------------------------------------------------------------------ #
    # Task-lifecycle hooks
    # ------------------------------------------------------------------ #
    def _initialize_task(
        self,
        *,
        turns: list[AgentRecord],
        prompt: str,
        inputs: dict,
    ) -> ScriptActAgentTask:
        """Return a ``ScriptActAgentTask`` with the planner system prompt
        active, its ``cache`` pre-seeded with every visible prior turn's
        result under ``task_result_{i}`` -- unconditional over whatever
        ``turns`` contains (empty when there's nothing to seed; already
        gated upstream by conversation-resolution/``context_enabled``/
        ``records_window``) -- and its ``constant_values`` pre-seeded from
        every registered constant. Both are copied via
        ``_copy_for_task_namespace`` exactly once here, not re-derived
        later, so a mutation is visible for the rest of this invocation's
        own rounds but never reaches the real constant or a future
        invocation. No other field needs seeding -- completed/pending/
        resolved_args/needs_repair/repair_rounds_used/tool_calls_used/
        repair_batch_start/failed_statements all start at their dataclass
        defaults."""
        task = ScriptActAgentTask(
            turns=turns, inputs=inputs, user_prompt=prompt, system_prompt_name="planner",
        )
        for turn in turns:
            task.cache[f"task_result_{self._turn_position(turn)}"] = self._copy_for_task_namespace(
                turn.generated_response
            )
        for spec in self._constants.values():
            task.constant_values[spec.name] = self._copy_for_task_namespace(spec.value)
        return task

    def _render_current_task_message(self, task: ScriptActAgentTask) -> dict[str, str]:
        """Bare "what is the task" user message -- reused verbatim for
        round 1 and every repair round's opening message. Mirrors the old
        ``JsonToolAgent._render_task_banner``'s role (dedup a repeated
        banner across every round) scoped to this family's own established
        wording (no ``===== ... =====`` markers -- that's JsonToolAgent-
        family styling, this family never used it). No "translate this
        into a plan" framing -- ``ONESHOT_PLANNER_PROMPT``'s OBJECTIVE
        section already states that once; repeating it every round would
        be redundant. Deliberately carries no ``tool_calls_limit`` text of
        its own -- round 1 and a repair round need different wording
        (total vs. remaining), so each of ``_render_task_messages``'s own
        two branches appends its own variant after calling this."""
        return {"role": "user", "content": f"CURRENT TASK:\n{task.user_prompt}"}

    def _render_task_messages(self, task: ScriptActAgentTask) -> list[dict[str, str]]:
        """Build-once contract per base ``Agent``'s documented pattern.
        Mirrors ``ReActAgent._render_task_messages``'s own 3-part
        organization (banner / assistant-authored state snapshot / user
        instruction) instead of cramming everything into one user message.

        Branches on ``task.needs_repair`` rather than ``task.completed`` --
        ``needs_repair`` is the one flag reliably ``True`` for every real
        repair round, including the edge case where a resolution/execution
        failure hits on the very first batch of round 1
        (``prepare()``/``_apply_batch_results`` set it directly, before
        anything ever lands in ``completed``); checking ``completed`` alone
        would silently drop that failure's reason and misrender it as a
        fresh round-1 call.

        Round 1 (``needs_repair`` still ``False``): the banner plus a
        trailing imperative ("Write a plan to accomplish this task now."),
        with a ``tool_calls_limit``-total line appended when set (``None``
        renders nothing).

        A repair round: banner, then an assistant-role state message with
        three sections -- the completed-work snapshot (reconstructed code,
        flat, no batch grouping, via ``render_completed_as_python``), the
        cache snapshot (``render_cache_snapshot``), and (new this pass)
        exactly what failed in the triggering batch, sliced from
        ``task.failed_statements[task.repair_batch_start:]`` and rendered
        via ``render_failed_as_python`` -- never the whole accumulated
        failure history, mirroring ``ReActTask.last_call_failed``'s own
        "show what you're reacting to" precedent. Then a fixed user
        instruction, with a ``tool_calls_limit``-*remaining* line appended
        when set (recomputed fresh each round, since it depletes -- unlike
        round 1's static total)."""
        if task.task_messages:
            return task.task_messages

        banner = self._render_current_task_message(task)

        if not task.needs_repair:
            content = (
                banner["content"]
                + "\n\nWrite a plan to accomplish this task now."
            )
            if self.tool_calls_limit is not None:
                plural = "s" if self.tool_calls_limit != 1 else ""
                content += (
                    f"\n\nYou may make at most {self.tool_calls_limit} "
                    f"tool call{plural} total."
                )
            task.task_messages = [{"role": "user", "content": content}]
            return task.task_messages

        snapshot = render_completed_as_python(task.completed) or "(nothing completed yet)"
        cache_snapshot = render_cache_snapshot(
            task.completed, task.cache, self._response_preview_limit
        )
        cache_section = f"\n\n{cache_snapshot}" if cache_snapshot else ""
        fresh_failures = task.failed_statements[task.repair_batch_start:]
        failure_text = render_failed_as_python(fresh_failures)
        failure_section = f"\n\n# THIS BATCH FAILED:\n{failure_text}" if failure_text else ""
        state_message = {
            "role": "assistant",
            "content": f"# WORK COMPLETED SO FAR:\n{snapshot}{cache_section}{failure_section}",
        }

        instruction = (
            "Continue planning the rest of this task, fixing what failed "
            "above. Use the existing work done to guide you on what the "
            "next steps should be."
        )
        if self.tool_calls_limit is not None:
            remaining = self.tool_calls_limit - task.tool_calls_used
            instruction += (
                f" Tool calls remaining: {remaining} of {self.tool_calls_limit}."
            )

        task.task_messages = [banner, state_message, {"role": "user", "content": instruction}]
        return task.task_messages

    # ------------------------------------------------------------------ #
    # Generation (think())
    # ------------------------------------------------------------------ #
    def _process_generation_output(
        self, raw_text: str, task: ScriptActAgentTask,
    ) -> list[list[CodeStatement]] | str:
        """
        Pure-computation validate callback for the planning retry loop:
        parse, validate references + remaining tool-call budget, and
        compile into batches. Returns the compiled result on success, or a
        feedback string describing every problem found on failure -- a
        ``BlackboardParseError`` from parsing is converted here, not
        propagated, so the retry loop can inject it as corrective feedback.
        """
        try:
            flat_slots = parse_generation(raw_text)
        except BlackboardParseError as e:
            return str(e)

        known_tools = frozenset(self._toolbox.keys())
        # Rewrite eligible builtin calls before whole-sequence validation,
        # so a rewritten slot is validated as PY_BUILTIN_ALIAS, not as an
        # unregistered tool; excluded-but-real builtin names are reported
        # here with a specific message instead of validate_references'
        # generic "unregistered tool" one.
        builtin_issues = rewrite_builtin_calls(flat_slots)
        # Derived from each ConstantSpec's own .name (already correctly
        # K_-prefixed exactly once for both the auto-named and aliased
        # registration paths -- see register_constant/register_constants),
        # never reconstructed by re-prefixing the internal dict key itself:
        # an auto-named key is already "K_0"-shaped, so blindly prepending
        # "K_" again would double-prefix it ("K_K_0"), silently mismatching
        # the name actually shown to the model via constants_context().
        known_constants = frozenset(spec.name for spec in self._constants.values())
        # Every key already in task.cache is safe to reference by name here:
        # cross-invocation task_result_i entries (seeded in
        # _initialize_task) AND, on a continuation round, every identifier
        # bound by an earlier round's own completed slots this same
        # invoke -- the exact names the "work completed so far" snapshot
        # (_render_task_messages) shows the model. Not scoped to the
        # TASK_RESULT_PREFIX any more; that was only ever correct back when
        # every generation was a fresh, single-round invocation.
        known_history = frozenset(task.cache.keys())
        remaining_budget = (
            None if self._tool_calls_limit is None
            else self._tool_calls_limit - task.tool_calls_used
        )
        issues = builtin_issues + validate_references(
            flat_slots, known_tools, known_constants, known_history, remaining_budget
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
    ) -> list[list[CodeStatement]]:
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
    ) -> list[list[CodeStatement]]:
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
        complete. Called from wherever ``task.pending`` actually reaches
        empty with ``needs_repair`` still ``False``: ``prepare`` (an
        already-empty round, or a same-round empty generation) and
        ``_apply_batch_results`` (the last batch of a plan draining) --
        never from ``think``, which only ever reads ``needs_repair``,
        never decides based on it."""
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
        ``prepare()``/``_apply_batch_results()`` -- the same two places
        that decide whether a repair round is granted in the first place --
        so by the time this hook is ever re-entered, that decision has
        already been made; it just generates.

        ``task.needs_repair`` is cleared here, after generation, not before
        -- ``_render_task_messages`` (invoked by ``_run_planning_retry_loop``
        below) still needs to see it ``True`` to render this round's repair
        framing. Once a fresh round has actually been generated, the flag's
        job is done: whatever ``prepare()``/``_apply_batch_results`` do with
        this new content next is a normal round, not a still-pending repair
        -- leaving it set would permanently block the natural-completion
        checks in both of those methods (they read ``needs_repair`` to tell
        "still mid-repair" apart from "genuinely finished") for the rest of
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
        Resolve the next pending batch's args, or short-circuit completion
        (or a granted repair) if nothing remains.

        If ``task.pending`` is empty: reset ``task.resolved_args``, then
        either leave the round as-is if ``task.needs_repair`` is already
        set (nothing to prepare -- ``think()`` will regenerate the repair
        round) or, otherwise, infer an implicit ``return None`` if no
        executed ``return`` slot already set ``task.generated_response``
        and mark the task complete -- covers both a genuinely empty
        generation and the natural end-of-plan drain, so ``act()`` needs
        only a bare no-op guard, not a second check.

        Otherwise: resolves every slot's args in ``task.pending[0]``,
        collecting every failure (not stopping at the first). Any collected
        failure abandons this batch and every batch still queued after it
        (they may depend on bindings this batch was supposed to produce).
        With ``fail_fast=True``, or once ``replanning_limit`` repair rounds
        are already spent, this raises immediately instead of granting
        another repair -- no silent partial-success path. Nothing here was
        ever dispatched, so this does not consume ``tool_calls_used``.
        """
        if not task.pending:
            task.resolved_args = []
            if task.needs_repair:
                return task
            return self._finalize_without_continuation(task)

        batch = task.pending[0]
        resolved: list[dict[str, Any]] = []
        failed_slots: list[CodeStatement] = []
        # Constants are validated as known references (validate_references'
        # known_constants) and rendered to the model (constants_context()),
        # but their actual runtime values live in task.constant_values --
        # a per-invocation copy seeded once by _initialize_task, never
        # re-derived from self._constants here (that would hand out the
        # live, shared constant object fresh every batch, defeating the
        # "a mutation stays visible for the rest of this invocation, never
        # leaks to another" guarantee). Merged in here, once per batch, so
        # a correct K_NAME reference actually resolves instead of raising
        # NameError; task.cache last so a real bound/history name would win
        # on the (never expected) collision.
        resolution_namespace = {**task.constant_values, **task.cache}
        for slot in batch:
            label = slot.identifier if slot.identifier is not None else "(unassigned)"

            try:
                positional, keyword = resolve_slot_args(slot, resolution_namespace)
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
            start = len(task.failed_statements)
            task.failed_statements.extend(failed_slots)
            if self._fail_fast or task.repair_rounds_used >= self._replanning_limit:
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: batch could not be "
                    "resolved and no repair is available "
                    f"({'fail_fast=True' if self._fail_fast else 'repair budget exhausted'}): "
                    f"{render_failed_as_python(failed_slots)}"
                )
            task.repair_rounds_used += 1
            task.repair_batch_start = start
            task.pending.clear()
            task.resolved_args = []
            task.needs_repair = True
            return task

        task.resolved_args = resolved
        return task

    async def async_prepare(self, task: ScriptActAgentTask) -> ScriptActAgentTask:
        """Direct passthrough to ``prepare`` -- no I/O of its own."""
        return self.prepare(task)

    # ------------------------------------------------------------------ #
    # Execute prepared batch
    # ------------------------------------------------------------------ #
    async def _gather_batch_results(
        self, batch: list[CodeStatement], resolved: list[dict[str, Any]],
    ) -> list[Any]:
        """
        Dispatch every dispatched-call slot in ``batch`` (registered tool
        or approved builtin, via ``is_dispatched_slot``) concurrently;
        ``rhs_assign``/``return`` slots need no dispatch, their result is
        already the resolved ``"val"`` value. Shared by ``act``/
        ``async_act`` -- both differ only in how the resulting coroutine is
        driven.
        """
        coros: list[Any] = []
        dispatch_map: dict[int, int] = {}
        for i, slot in enumerate(batch):
            if is_dispatched_slot(slot):
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
        batch: list[CodeStatement],
        resolved: list[dict[str, Any]],
        raw_results: list[Any],
    ) -> ScriptActAgentTask:
        """
        Shared post-gather bookkeeping for ``act``/``async_act``: apply
        results, update completed/cache, handle a terminal ``return`` slot,
        pop the consumed batch -- or, if any real call in this batch
        failed, record whichever succeeded, abandon the rest of this round,
        and either grant a repair round or raise, per ``fail_fast``/
        ``replanning_limit`` (same budget check as ``prepare()``'s own
        resolution-failure branch).

        Every dispatched call in ``batch`` (registered tool or approved
        builtin, via ``is_dispatched_slot``) was actually dispatched via
        ``asyncio.gather`` regardless of whether any of them failed, so all
        of them count against ``tool_calls_used`` unconditionally, before
        checking for failures.
        """
        real_call_count = sum(1 for slot in batch if is_dispatched_slot(slot))
        task.tool_calls_used += real_call_count

        # A raised exception only ever appears here for a slot that was
        # actually dispatched (asyncio.gather(..., return_exceptions=True)
        # is the only source of a bare BaseException in raw_results) -- an
        # rhs_assign/return slot's raw_results entry is always its plain
        # resolved value (see _gather_batch_results), which may itself
        # legitimately BE a BaseException instance (e.g. a registered
        # constant holding an exception object as data). Gating on
        # is_dispatched_slot prevents misclassifying that legitimate value
        # as an execution failure.
        failed_slots: list[CodeStatement] = []

        for slot, value in zip(batch, raw_results):
            if is_dispatched_slot(slot) and isinstance(value, BaseException):
                slot.exception = value
                failed_slots.append(slot)
                continue
            task.completed.append(slot)
            # A dispatched tool call's raw_results entry is a full
            # AtomicResult envelope (slot.result stores it verbatim,
            # matching v1's board[idx].result precedent); rhs_assign/return
            # slots were never dispatched, so their value is already the
            # plain resolved Python value -- no envelope to unwrap.
            if slot.tool not in (RHS_ASSIGN_ALIAS, RETURN_ALIAS):
                slot.result = value
                unwrapped = value.result
            else:
                unwrapped = value
            if slot.identifier is not None:
                task.cache[slot.identifier] = unwrapped

        if failed_slots:
            start = len(task.failed_statements)
            task.failed_statements.extend(failed_slots)
            if self._fail_fast or task.repair_rounds_used >= self._replanning_limit:
                if len(failed_slots) == 1 and isinstance(failed_slots[0].exception, ToolInvocationError):
                    raise failed_slots[0].exception
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: batch execution "
                    "failed and no repair is available "
                    f"({'fail_fast=True' if self._fail_fast else 'repair budget exhausted'}): "
                    f"{render_failed_as_python(failed_slots)}"
                )
            task.repair_rounds_used += 1
            task.repair_batch_start = start
            task.pending.clear()
            task.resolved_args = []
            task.needs_repair = True
            return task

        for slot, kwargs in zip(batch, resolved):
            if slot.tool == RETURN_ALIAS:
                task.generated_response = kwargs["val"]
                task.complete = True

        task.pending.pop(0)
        task.resolved_args = []

        # This batch just drained the plan. If nothing needs repairing and
        # nothing already completed it (no `return` above), finalize right
        # here -- the natural point `task.pending` actually reaches empty --
        # instead of leaving it for a future `prepare()` call that `think()`
        # would otherwise reach first on the next loop iteration and
        # regenerate an uninvited round.
        if not task.pending and not task.complete and not task.needs_repair:
            return self._finalize_without_continuation(task)

        return task

    def act(self, task: ScriptActAgentTask) -> ScriptActAgentTask:
        """
        Execute the currently resolved batch, or no-op if ``prepare``
        produced nothing to run this round (covers a short-circuited round
        from ``prepare``'s empty-``pending`` guard).
        """
        if not task.resolved_args:
            return task

        batch = task.pending[0]
        resolved = task.resolved_args
        raw_results = run_coro_sync(self._gather_batch_results(batch, resolved))
        return self._apply_batch_results(task, batch, resolved, raw_results)

    async def async_act(self, task: ScriptActAgentTask) -> ScriptActAgentTask:
        """Async mirror of ``act``; awaits ``_gather_batch_results``
        directly rather than ``run_coro_sync``-wrapping it."""
        if not task.resolved_args:
            return task

        batch = task.pending[0]
        resolved = task.resolved_args
        raw_results = await self._gather_batch_results(batch, resolved)
        return self._apply_batch_results(task, batch, resolved, raw_results)
