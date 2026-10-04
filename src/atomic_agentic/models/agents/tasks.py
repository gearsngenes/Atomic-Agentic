from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from ...constants.core import NO_VAL
from .blackboard_models import ToolStatement
from .records import AgentRecord, LLMRecord

__all__ = [
    "AgentTask",
    "ToolAgentTask",
    "ScriptActAgentTask",
    "PlanActTask",
    "ReActTask",
    "ThinkingTask",
]


@dataclass(slots=True)
class AgentTask:
    """
    Base run-task contract for one Agent invocation.

    Drives an invocation through ``_initialize_task`` then
    ``think``/``prepare``/``act`` each round. ``BasicAgent`` uses this base
    class directly; it needs nothing beyond these fields.

    Fields
    ------
    turns : list[AgentRecord]
        The selected conversation turns for this invocation (same list
        ``_initialize_task`` received). Used for ``prev`` linkage when
        building the completed record.

    inputs : dict[str, Any]
        The full, untouched ``inputs`` mapping from the base ``Agent``
        invocation lifecycle.

    user_prompt : str
        The fully-resolved prompt string produced by ``pre_invoke`` for this
        invocation (matches ``AgentRecord.user_prompt`` 1:1).

    system_prompt_name : str | None
        Identifies which entry in ``self._system_prompts`` governs the
        current phase — read by ``render_task``/``_render_system_message``
        to select and render the active system prompt. Required (no
        default): every concrete subclass must decide it explicitly.
        ``None`` is a legitimate value meaning "render no system message at
        all," not a not-yet-set placeholder — though no concrete subclass
        shipped today ever constructs a task with ``None``.

    llm_records : list[LLMRecord]
        Accumulator for every LLM generation made while producing this
        invocation's result. Read exactly once, at loop-close, to populate
        the completed record's ``llm_records`` tuple.

    complete : bool
        Loop termination flag. The base lifecycle loop is
        ``while not task.complete: task = think(task); task = prepare(task);
        task = act(task)``.

    generated_response : Any
        The record's produced-response equivalent — raw LLM text for
        ``BasicAgent``, the executed return-tool value for ``JsonToolAgent``
        and its subclasses. ``NO_VAL`` until ``act`` sets it on the round
        that completes the task.

    historic_messages : list[dict[str, str]]
        Rendered ``turns``, built lazily once per invoke by
        ``Agent._render_historic_messages`` and reused for the rest of it.
        Never rebuilt mid-invoke — turn-rendering is phase-invariant,
        governed only by ``response_preview_limit``, itself static
        per-agent config.

    task_messages : list[dict[str, str]]
        Phase-scoped LLM-facing content, lazily built by each subclass's
        ``render_task`` when empty and grown by ``additional_messages``
        across generation retries within one phase. Cleared by the owning
        subclass at its own phase boundary (e.g. ``PlanActAgent`` once
        planning finishes; ``ReActAgent`` after each step commits).
    """
    turns: list[AgentRecord]
    inputs: dict[str, Any]
    user_prompt: str
    system_prompt_name: str | None

    llm_records: list[LLMRecord] = field(default_factory=list)
    complete: bool = False
    generated_response: Any = NO_VAL
    historic_messages: list[dict[str, str]] = field(default_factory=list)
    task_messages: list[dict[str, str]] = field(default_factory=list)


@dataclass(slots=True)
class ToolAgentTask(AgentTask):
    """
    Base task shape shared by every concrete ``ToolAgent`` family
    (``ScriptActAgent``/``PlanActAgent``/``ReActAgent``), renamed from this
    class's prior name -- that class tier was removed from the agent
    hierarchy in an earlier pass (``ScriptActAgent`` is a direct
    ``ToolAgent`` sibling now, not a separate family), so the model name
    never caught up until this pass. Promotes every field genuinely shared
    by all three concrete agents today, closing the gap where
    ``ScriptActAgentTask`` used to independently redeclare these same six
    fields as a bare ``AgentTask`` sibling instead of inheriting them.

    Fields
    ------
    completed : list[ToolStatement]
        Every call that executed successfully so far this run, in commit
        order. Becomes the owning record's ``statements`` tuple verbatim at
        commit time.

    failed_statements : list[ToolStatement]
        Every call whose dispatch (or, for ``ScriptActAgent``, resolution)
        actually failed this run, in the order observed.

    cache : dict[str, Any]
        identifier -> resolved value for every call in ``completed``, plus
        ``task_result_i`` entries seeded once at ``_initialize_task``.

    constant_values : dict[str, Any]
        Registered-constant name -> value, populated once by each concrete
        agent's own ``_initialize_task`` and never touched again.

    regenerations_used : int
        Cumulative regeneration attempts consumed across every generation
        call this run -- irreducible; a rejected/regenerated draft leaves
        no artifact anywhere else to derive this from (contrast
        ``tool_calls_used``, fully derivable from ``completed``/
        ``failed_statements``). Renamed from the prior ``retries_used``
        since this family's generation model is ``output_structure`` (or,
        for ``ScriptActAgent``, free-form source) plus a regen-retry loop,
        not the old free-text-JSON retry loop.
    """
    completed: list[ToolStatement] = field(default_factory=list)
    failed_statements: list[ToolStatement] = field(default_factory=list)
    cache: dict[str, Any] = field(default_factory=dict)
    constant_values: dict[str, Any] = field(default_factory=dict)
    regenerations_used: int = 0

    @property
    def tool_calls_used(self) -> int:
        """
        Derived, not stored -- ``completed``/``failed_statements`` are the
        single source of truth for every concrete subclass. Counts every
        dispatched (non-``RETURN_ALIAS``/``RHS_ASSIGN_ALIAS``) call in
        either list, whether it ultimately succeeded or raised.

        This is the one deliberate behavior change in this pass: previously
        ``ScriptActAgentTask`` manually incremented a counter only in
        ``_apply_batch_results`` (a real dispatch attempt), excluding a
        resolution failure caught earlier in ``prepare()`` from the budget
        count. This shared formula counts it instead -- ``is_dispatched`` is
        purely a function of ``call.tool``, not of whether dispatch was ever
        actually attempted -- harmonizing onto ``PlanActTask``'s/
        ``ReActTask``'s already-existing rule: a call the plan committed to
        spends budget whether or not the tool itself ever ran.
        """
        from ...utils.agents import is_dispatched
        return (
            sum(1 for c in self.completed if is_dispatched(c))
            + sum(1 for c in self.failed_statements if is_dispatched(c))
        )


@dataclass(slots=True)
class ScriptActAgentTask(ToolAgentTask):
    """
    ScriptActAgent-flavored task -- now a real ``ToolAgentTask`` subclass
    (was a bare ``AgentTask`` sibling independently redeclaring the same six
    fields ``ToolAgentTask`` now owns). ``completed``/``failed_statements``/
    ``cache``/``constant_values``/``regenerations_used``/``tool_calls_used``
    are all inherited; only genuinely ``ScriptActAgent``-specific fields
    remain declared here.

    No __post_init__ -- matches AgentTask's own family-wide convention of
    zero constructor-time validation (an in-flight, internal-only object,
    not a real boundary, per 01-overview.md Section 4).

    Fields
    ------
    pending : list[list[ToolStatement]]
        Every not-yet-executed dependency batch compiled from the current
        generation -- the whole one-shot draft (or, after a granted repair,
        the whole freshly regenerated tail) is parsed and batch-compiled in
        a single pass, so this can hold several batches at once even though
        there is only ever one generation per round. The front batch
        (``pending[0]``) is the next one act() runs; once it fully
        executes, its slots move into ``completed`` and it is popped from
        this list.

    needs_repair : bool
        Set only by ``prepare()``/``_apply_batch_results()``, only on a
        real resolution or execution failure that ``fail_fast``/
        ``replanning_limit`` grants a repair round for -- there is no
        model-authored way to set this anymore (no ``# PAUSE``, no
        if-cutoff). Cleared by ``think()``/``async_think()``, right after
        generating the granted repair round's fresh content -- the flag's
        only job is telling ``prepare()``'s and ``_apply_batch_results``'s
        own empty-``pending`` checks "stopped, needs a repair round" (leave
        the task incomplete for ``think()`` to regenerate next) apart from
        "genuinely finished" (finalize via ``_finalize_without_continuation``
        right there); once that regeneration has actually happened, leaving
        it set would permanently block both methods' natural-completion
        checks for the rest of the invoke.

    repair_rounds_used : int
        Count of framework-granted repair rounds so far this invoke --
        never counts the free initial plan (contrast the old
        ``planning_rounds_used``, which counted round 1 too). Incremented
        only in ``prepare()``/``_apply_batch_results()``, at the exact
        point a repair round is granted -- never in ``think()``, which has
        no budget awareness at all anymore. Checked against
        ``self._replanning_limit`` at those same two sites before granting
        another.

    repair_batch_start : int
        Index into ``failed_statements`` marking where the batch that most
        recently triggered (or attempted to trigger) a repair round began.
        Set alongside ``needs_repair = True``, in the same
        ``prepare()``/``_apply_batch_results()`` branch that extends
        ``failed_statements`` with this round's failures. Consumed (read,
        not cleared) by ``_render_task_messages`` to slice
        ``failed_statements[repair_batch_start:]`` -- exactly this round's
        fresh failures, not the whole accumulated history -- mirrors
        ``ReActTask.last_call_failed``'s own "show what you're reacting to"
        precedent. Replaces the old ``continuation_note`` field entirely;
        there is no separately-maintained note string anymore, rendering
        derives everything from this slice plus each slot's own
        ``.exception``.

    resolved_args : list[dict[str, Any]]
        Positionally matched to ``pending[0]``'s slots -- the resolved
        kwargs ``prepare()`` computed for the batch ``act()`` is about to
        run. Reset to ``[]`` by ``act()`` once that batch is fully consumed
        (or by ``prepare()``'s empty-``pending`` guard). Empty whenever
        there is nothing currently prepared to execute.

    batch_counter : int
        Running total of batches compiled so far this invoke, across every
        generation round -- never reset mid-run. Passed to
        ``compile_batches`` as ``start_batch_index`` each time it's called,
        then advanced by the number of batches that call produced, so
        ``ToolStatement.batch_index`` values stay globally unique across a
        whole invoke. Note: ``failed_statements`` -- populated by BOTH
        ``prepare()`` (a resolution/binding failure, ``slot.exception`` set
        to a synthesized ``ToolAgentError``, mirroring ``PlanActTask``'s
        own resolution-failure pattern) and ``_apply_batch_results`` (a
        real dispatch failure, ``slot.exception`` set to the actual raised
        value) -- is now inherited from ``ToolAgentTask``; see that class's
        own docstring. ``repair_batch_start`` marks where each triggering
        batch's own slice into it begins.
    """
    pending: list[list[ToolStatement]] = field(default_factory=list)
    resolved_args: list[dict[str, Any]] = field(default_factory=list)
    needs_repair: bool = False
    repair_rounds_used: int = 0
    repair_batch_start: int = 0
    batch_counter: int = 0
    # REMOVED: completed, failed_statements, cache, constant_values,
    # regenerations_used, tool_calls_used -- all now inherited from
    # ToolAgentTask.


@dataclass(slots=True)
class PlanActTask(ToolAgentTask):
    """
    PlanActAgent-flavored task -- ``ToolStatement``-based batch-execution
    fields, with no continuation-related fields at all: ``think()`` runs
    exactly once, ever, per invoke, so there is nothing analogous to a
    voluntary multi-round continuation flag, round counter, or continuation
    note to carry. ``completed``/``failed_statements``/``cache``/
    ``constant_values``/``tool_calls_used`` are all inherited from
    ``ToolAgentTask`` -- only ``pending``/``resolved_args`` are genuinely
    ``PlanActAgent``-specific.

    Fields
    ------
    pending : list[list[ToolStatement]]
        Dependency batches compiled once (via ``compile_batches``) from the
        single generated plan. ``pending[0]`` is the next batch ``act()``
        runs. A cascade-failure may remove calls from batches here without
        popping them (see ``find_cascade_failures``) -- only ``act()``
        pops a batch once it's been executed.

    resolved_args : list[dict[str, Any]]
        Positionally matched to ``pending[0]``'s surviving calls -- the
        resolved kwargs ``prepare()`` computed for the batch ``act()`` is
        about to run.

    No ``batch_counter`` field -- a single ``compile_batches`` call per
    invoke needs no cross-round ``ToolStatement.batch_index`` uniqueness
    tracking; a local variable in ``think()`` suffices. The inherited
    ``tool_calls_used`` formula (``ToolAgentTask``) already matches this
    class's own prior override for every real case here: a cascade-skipped
    call is never dispatched, so it never enters ``completed`` or
    ``failed_statements`` either way.
    """
    pending: list[list[ToolStatement]] = field(default_factory=list)
    resolved_args: list[dict[str, Any]] = field(default_factory=list)
    # REMOVED: completed, failed_statements, cache, constant_values (now
    # inherited), and the tool_calls_used @property override (now
    # inherited -- same formula).


@dataclass(slots=True)
class ReActTask(ToolAgentTask):
    """
    ReActAgent-flavored task: one registered-tool call generated, resolved,
    and dispatched per round, via provider-native structured output
    (``REACT_OUTPUT_SCHEMA``) -- the per-step sibling to ``PlanActTask``'s
    one-shot multi-call plan. ``completed``/``failed_statements``/``cache``/
    ``constant_values``/``tool_calls_used`` are all inherited from
    ``ToolAgentTask`` -- only ``generated_step``/``resolved_args``/
    ``last_call_failed`` are genuinely ``ReActAgent``-specific.

    No fixed-size preallocated board and no observability-decay window
    (both dropped from the pre-rewrite shape, along with ``next_step_index``/
    ``step_meta``) -- every round renders a full snapshot of ``completed``/
    ``cache`` instead (``utils.sigils.render_completed_as_json``/
    ``render_cache_snapshot``, reused unmodified). No
    ``pending: list[list[ToolStatement]]`` batch field either -- unlike
    ``PlanActTask``, this family dispatches exactly one call per round,
    never a concurrency batch. No ``__post_init__`` -- matches every other
    ``*Task`` in this family: an in-flight, internal-only object, not a real
    construction-time boundary.

    A resolution failure (a ``$name`` reference that doesn't resolve, or a
    resolved value's type mismatching the target tool's parameter contract)
    never reaches ``failed_statements`` at all under this family's design --
    both are caught and fed back for regeneration inside ``think()``'s own
    retry loop, before a call is ever accepted as this round's decision.
    Only a real dispatch failure (the tool itself raised once actually
    invoked) lands there -- which is also why the inherited
    ``tool_calls_used`` formula (gating both ``completed`` and
    ``failed_statements`` through ``is_dispatched``) already matches this
    class's own prior override for every real case here: every entry in
    either list is guaranteed to be a real dispatch attempt regardless.

    Fields
    ------
    generated_step : ToolStatement | None
        The one call ``think()`` validated and resolved this round -- set
        only once both ``utils.sigils.translate_calls`` and
        ``utils.agents.resolve_statement_args`` succeed against it, ``None``
        before that and after ``act()`` consumes it. Retyped from the pre-rewrite
        shape's ``Any = NO_VAL`` (which held a ``(BlackboardSlot, int,
        str)`` tuple under the old duration/observability design) --
        ``None`` is the idiomatic "not yet decided" value for a field
        genuinely typed ``Optional[ToolStatement]``, so no ``NO_VAL``
        sentinel is needed here.

    resolved_args : dict[str, Any] | None
        The resolved keyword-argument dict for ``generated_step``
        (``tool._args_kwargs_to_dict``-ready), computed inside ``think()``'s
        own retry loop -- never a later ``prepare()`` step, since
        ``prepare()`` is a documented no-op for this family (both
        ``think()`` and ``prepare()`` operate on the same single call;
        nothing is deferred to a later loop iteration the way a multi-batch
        plan requires). Singular, not ``PlanActTask.resolved_args``'s
        ``list[dict[str, Any]]`` -- there is no batch to index into.

    last_call_failed : bool
        Set by ``act()``/``async_act()`` at the end of every round --
        ``True`` on a tolerated (``fail_fast=False``) dispatch failure,
        ``False`` on success. Exists because ``completed``/
        ``failed_statements`` are two separate append-only lists with no
        shared chronological index between them -- without this flag, a
        later round's rendering can't tell whether the round that just
        finished succeeded or failed, only that some historical failure
        exists somewhere in ``failed_statements``. The failed call itself
        is always ``task.failed_statements[-1]`` when this is ``True`` --
        no duplicate reference stored.
    """
    generated_step: Optional[ToolStatement] = None
    resolved_args: Optional[dict[str, Any]] = None
    last_call_failed: bool = False
    # REMOVED: completed, failed_statements, cache, constant_values (now
    # inherited), and the tool_calls_used @property override (now
    # inherited -- same formula, this class's own version was already
    # byte-for-byte the formula being promoted).


@dataclass(slots=True)
class ThinkingTask(AgentTask):
    """
    ThinkingAgent-flavored task.

    No ``phase`` field: ``AgentTask.system_prompt_name`` doubles as the
    phase discriminator. The name ``"role"`` is reserved for the reply
    phase; any other value (``ThinkingAgent.THINKING_PROMPT_NAME``) means a
    thinking round is still active. This is why the field is required (no
    default) on the base ``AgentTask`` — every concrete subclass must
    decide it explicitly, and here that decision *is* the phase.

    No retry-budget field: a thinking round either produces a non-empty
    stripped string (or structured value, once ``thinking_schema`` exists)
    or ``think()`` raises outright — there is no malformed-output case to
    retry against, since there is no parsing step left to fail partially.

    Fields
    ------
    thoughts : list[str | int | float | bool | list | dict | None]
        Task-local accumulator, one raw value per completed round (widened
        type declared for ``thinking_schema`` support). Persisted onto the
        completed ``ThinkingAgentRecord.thoughts`` tuple verbatim at
        ``_build_record_from_task`` time -- there is no agent-level
        accumulator; the record is the only place this content survives
        past the task's own lifetime. Its own length doubles as the
        completed-round count — no separate counter field is kept.

    thinking_rounds : int
        Validated per-invocation round budget, resolved once by
        ``ThinkingAgent._initialize_task`` from the reserved
        ``thinking_rounds`` runtime parameter and never changed for the
        rest of this task's lifetime. Declared with a default of ``1``
        only because Python dataclass field ordering requires every field
        following ``AgentTask``'s own defaulted fields to carry one too —
        the real value is always passed explicitly at construction
        (mirrors ``ToolAgentTask.regenerations_used: int = 0``'s identical
        precedent). ``think()``/``async_think()`` read this field directly
        instead of any agent-level attribute — there is no construction-time
        equivalent left; the round budget is purely per-invocation now.
    """
    thoughts: list[str | int | float | bool | list | dict | None] = field(default_factory=list)
    thinking_rounds: int = 1
