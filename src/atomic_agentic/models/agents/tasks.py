from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from ...constants.core import NO_VAL
from .blackboard_models import CodeStatement, DagToolCall
from .records import AgentRecord, LLMRecord

__all__ = [
    "AgentTask",
    "JsonToolAgentTask",
    "ScriptAgentTask",
    "DagAgentTask",
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
class JsonToolAgentTask(AgentTask):
    """
    JsonToolAgent-flavored task -- narrowed to the one field genuinely
    shared across every JsonToolAgent family regardless of execution shape
    (batch-cursor vs. step-cursor). Everything else this class used to
    carry (``running_blackboard``/``executed_steps``/``prepared_steps``/
    ``valid_cache_indices``/``failed_cache_indices``/``tool_calls_used``)
    was blackboard/placeholder-grammar-era and is removed outright, not
    adapted -- see the `json-tool-agent-rename` design record's
    lifecycle-slimming addendum for the base-class side of this same cut.

    Fields
    ------
    llm_records : list[LLMRecord]
        Inherited from ``AgentTask``. Seeded at construction time —
        non-empty for subclasses that generate up front (e.g. PlanAct's
        one-shot plan), empty for subclasses that generate lazily during the
        loop (e.g. ReAct's per-step planning) — and appended to as further
        generations occur.

    regenerations_used : int
        Cumulative regeneration attempts consumed across every generation
        call this run -- irreducible; a rejected/regenerated draft leaves
        no artifact anywhere else to derive this from (contrast
        ``tool_calls_used``, declared per-subclass now since it's fully
        derivable from whatever call-history fields that subclass's own
        task shape carries). Renamed from the prior ``retries_used`` to
        match ``DagAgentTask``'s own terminology, since this family's
        generation model now mirrors DagAgent's (``output_structure`` +
        regen-retry loop) rather than the old free-text-JSON retry loop.
    """
    regenerations_used: int = 0


@dataclass(slots=True)
class ScriptAgentTask(AgentTask):
    """
    ScriptAgent-flavored task -- a sibling to JsonToolAgentTask, not a subclass
    (ScriptAgent is a new agent family, not a JsonToolAgent subclass, per this
    branch's established convention).

    No __post_init__ -- matches AgentTask's own family-wide convention of
    zero constructor-time validation (an in-flight, internal-only object,
    not a real boundary, per 01-overview.md Section 4).

    Fields
    ------
    completed : list[CodeStatement]
        Every slot that executed successfully so far this run, in commit
        order. Purely historical -- nothing here is ever mutated once a
        slot lands in this list. Becomes ScriptAgentRecord.statements
        verbatim (normalized to a tuple) at commit time. A slot whose
        dispatch raised is never appended here -- see failed_statements.

    pending : list[list[CodeStatement]]
        Every not-yet-executed dependency batch compiled so far for the
        current plan -- not scoped to just the next pause. The whole
        one-shot draft (or, after a replan, the whole freshly regenerated
        tail) is parsed and batch-compiled in a single pass, so this can
        span multiple pauses' worth of batches at once. The front batch
        (``pending[0]``) is the next one act() runs; once it fully
        executes, its slots move into ``completed`` and it is popped from
        this list.

    continue_planning : bool
        Set by ``parse_generation`` (via ``think()``) when the round itself
        asked to continue (a ``# PAUSE``, or a valid if-cutoff), or by
        ``prepare()``/``_apply_batch_results()`` when a batch's args failed
        to resolve or a real tool call failed. Read by ``prepare()``'s and
        ``_apply_batch_results``'s own empty-``pending`` checks -- whichever
        one actually drains ``pending`` to empty -- to tell "stopped, needs
        a continuation" (leave the task incomplete for ``think()`` to
        regenerate next) apart from "genuinely finished" (finalize via
        ``_finalize_without_continuation`` right there, before ``think()``
        ever gets a turn to regenerate an uninvited round).

    planning_rounds_used : int
        Count of planning generations made so far this invoke, including
        the first -- total-count semantics, not "extra chances beyond a
        free first attempt" (contrast ``regenerations_used``, which
        specifically counts second chances within one round).
        Incremented unconditionally by ``think()``/``async_think()`` on
        every real call, then checked against
        ``self._planning_rounds_limit`` before requesting another.

    tool_calls_used : int
        Cumulative count of dispatched (non-``rhs_assign``/``return``)
        calls actually dispatched so far this invoke, across every
        generation round -- registered-tool and approved-builtin calls
        counted identically (no per-category exemption), never decremented.
        Incremented only in ``_apply_batch_results`` (a real dispatch
        happened, whether it succeeded or failed); never in ``prepare()``
        (a resolution failure means nothing was ever dispatched). Read by
        ``_process_generation_output`` to compute the remaining budget
        passed into ``validate_references``.

    continuation_note : Optional[str]
        Framework-authored (never model-authored) reason text, set only by
        ``prepare()``'s resolution-failure branch or
        ``_apply_batch_results``'s execution-failure branch -- a
        multi-line block combining the failed batch's own rendered source
        (via ``render_completed_as_python``) with its labeled issue/failure
        list, so the next continuation round sees both what it wrote and
        specifically what went wrong with it. An explicit ``# PAUSE`` or
        if-cutoff continuation never sets this (bare sentinel, no note of
        any kind); consulted and cleared back to ``None`` in the same read
        by ``_render_task_messages`` on the next generation, so a stale note
        from an already-addressed failure never leaks into a later round.

    cache : dict[str, Any]
        identifier -> resolved value for every slot in ``completed``, kept
        in sync as slots complete. Shaped to be passed directly as
        utils/script.py's ``resolve_slot_args(statement, resolved)``'s
        ``resolved`` argument -- an O(1) lookup instead of scanning
        ``completed``.

    constant_values : dict[str, Any]
        Registered-constant name -> value, populated exactly once by
        ``ScriptAgent._initialize_task`` (deep-copied per constant except
        for known atomic-immutable types) and never touched again after
        that. Every batch's resolution namespace reads this instead of
        re-deriving values from the agent's own ``self._constants`` each
        time, so a mutating attribute/method call in one round is visible
        to a later round of the *same* invocation (same copy, shared for
        this task's lifetime) but never reaches a different invocation (a
        fresh task gets a fresh copy).

    regenerations_used : int
        Cumulative regeneration attempts consumed across every generation
        call this run (initial plan, planner repair calls) -- structural/
        syntax/validation failures only. Unlike tool-call budget
        accounting, not derivable from ``completed`` (regenerations are
        generation attempts, not slots), so this stays an explicit counter.
        Checked against ``self._regeneration_limit`` (always a plain
        non-negative ``int``, never ``None``) before permitting another
        attempt within one round.

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
        ``CodeStatement.batch_index`` values stay globally unique across a
        whole invoke.

    failed_statements : list[CodeStatement]
        Every slot whose dispatch raised, in the order the failure was
        observed, across every round this invoke -- the permanent
        counterpart to ``continuation_note``'s ephemeral text (which is
        consulted and cleared on the very next render). Appended by
        ``_apply_batch_results`` at the same point a failure is detected,
        with ``slot.exception`` set to the raised value first. Never
        cleared or mutated once appended. Becomes
        ScriptAgentRecord.failed_statements verbatim (normalized to a
        tuple) at commit time.
    """
    completed: list[CodeStatement] = field(default_factory=list)
    pending: list[list[CodeStatement]] = field(default_factory=list)
    cache: dict[str, Any] = field(default_factory=dict)
    constant_values: dict[str, Any] = field(default_factory=dict)
    regenerations_used: int = 0
    resolved_args: list[dict[str, Any]] = field(default_factory=list)
    continue_planning: bool = False
    planning_rounds_used: int = 0
    tool_calls_used: int = 0
    continuation_note: Optional[str] = None
    batch_counter: int = 0
    failed_statements: list[CodeStatement] = field(default_factory=list)


@dataclass(slots=True)
class DagAgentTask(AgentTask):
    """
    DagAgent-flavored task -- a sibling to ScriptAgentTask, not a subclass
    (DagAgent is a new agent family, not a ScriptAgent subclass, matching
    ScriptAgentTask's own precedent relative to JsonToolAgentTask).

    Field-for-field structural copy of ScriptAgentTask, CodeStatement
    swapped for DagToolCall everywhere it appears -- no new fields. (An
    earlier design draft added a return_value staging field for the round's
    raw return payload; superseded once return was decided to reuse
    RETURN_ALIAS/the normal batch pipeline instead of a bespoke
    resolve-outside-the-batch-compiler mechanism.) No __post_init__ --
    matches AgentTask's family-wide convention of zero constructor-time
    validation (an in-flight, internal-only object, not a real boundary).

    Fields
    ------
    completed : list[DagToolCall]
        Every call that executed successfully so far this run, in commit
        order (the RETURN_ALIAS call, once it executes, lands here too --
        it is not excluded). Becomes DagAgentRecord.statements verbatim
        (normalized to a tuple) at commit time.

    pending : list[list[DagToolCall]]
        Every not-yet-executed dependency batch compiled so far for the
        current plan. pending[0] is the next batch act() runs.

    continue_planning : bool
        Unified continuation trigger, set either by the model's own
        remaining_work schema signal or by the framework itself on a
        resolution/execution failure -- same single reactive-continuation
        path either way, no split between "continuation" and "repair"
        handling. Same role ScriptAgentTask.continue_planning already has.

    cache : dict[str, Any]
        identifier -> resolved value for every call in completed, kept in
        sync as calls complete, plus task_result_i entries seeded once at
        _initialize_task. Precedence on lookup: constant_values and this
        dict, together, always outrank a plan-local result_name name on a
        name collision -- though a real collision is structurally
        impossible by construction, since result_name/task_result_* names can
        never start with K_ and constant names always do (enforced by
        validate_calls).

    constant_values : dict[str, Any]
        Registered-constant name -> value, populated once by
        DagAgent._initialize_task and never touched again after that.

    regenerations_used, resolved_args, planning_rounds_used,
    tool_calls_used, continuation_note, batch_counter, failed_statements
        Identical role and shape to ScriptAgentTask's same-named fields --
        see that class's own docstring for the authoritative description;
        nothing about them changes for DagAgentTask.
    """
    completed: list[DagToolCall] = field(default_factory=list)
    pending: list[list[DagToolCall]] = field(default_factory=list)
    cache: dict[str, Any] = field(default_factory=dict)
    constant_values: dict[str, Any] = field(default_factory=dict)
    regenerations_used: int = 0
    resolved_args: list[dict[str, Any]] = field(default_factory=list)
    continue_planning: bool = False
    planning_rounds_used: int = 0
    tool_calls_used: int = 0
    continuation_note: Optional[str] = None
    batch_counter: int = 0
    failed_statements: list[DagToolCall] = field(default_factory=list)


@dataclass(slots=True)
class PlanActTask(JsonToolAgentTask):
    """
    PlanActAgent-flavored task -- ``DagAgentTask``'s batch-execution field
    shape, minus every field that exists purely to support ``DagAgent``'s
    multi-round continuation (``continue_planning``/``planning_rounds_used``/
    ``continuation_note``/``batch_counter``) -- none of that applies to a
    one-shot planner: ``think()`` runs exactly once, ever, per invoke.
    ``DagToolCall`` in place of ``CodeStatement``, matching
    ``DagAgentTask``'s own precedent for the same swap.

    Fields
    ------
    completed : list[DagToolCall]
        Every call that executed successfully this run, in commit order
        (the ``RETURN_ALIAS`` call included once it executes -- same
        convention as ``DagAgentTask.completed``).

    pending : list[list[DagToolCall]]
        Dependency batches compiled once (via ``compile_batches``) from the
        single generated plan. ``pending[0]`` is the next batch ``act()``
        runs. A cascade-failure may remove calls from batches here without
        popping them (see ``find_cascade_failures``) -- only ``act()``
        pops a batch once it's been executed.

    cache : dict[str, Any]
        identifier -> resolved value for every call in ``completed``, plus
        ``task_result_i`` entries seeded once at ``_initialize_task``.

    constant_values : dict[str, Any]
        Registered-constant name -> value, populated once by
        ``PlanActAgent._initialize_task``.

    resolved_args : list[dict[str, Any]]
        Positionally matched to ``pending[0]``'s surviving calls -- the
        resolved kwargs ``prepare()`` computed for the batch ``act()`` is
        about to run.

    failed_statements : list[DagToolCall]
        Every call whose dispatch actually raised this run -- not
        cascade-skipped calls, which are never attempted and never appear
        here (see ``find_cascade_failures``'s own contract). Becomes
        ``JsonToolAgentRecord.failed_statements`` verbatim at commit time.

    No ``batch_counter`` field -- a single ``compile_batches`` call per
    invoke needs no cross-round ``DagToolCall.batch_index`` uniqueness
    tracking; a local variable in ``think()`` suffices.
    """
    completed: list[DagToolCall] = field(default_factory=list)
    pending: list[list[DagToolCall]] = field(default_factory=list)
    cache: dict[str, Any] = field(default_factory=dict)
    constant_values: dict[str, Any] = field(default_factory=dict)
    resolved_args: list[dict[str, Any]] = field(default_factory=list)
    failed_statements: list[DagToolCall] = field(default_factory=list)

    @property
    def tool_calls_used(self) -> int:
        """
        Derived, not stored: every dispatched (non-``RETURN_ALIAS``) call in
        ``completed``, plus every call that was actually attempted and
        raised (``failed_statements``) -- ``completed``/``failed_statements``
        are already the single source of truth, so a separate incrementing
        counter would just be a second, driftable copy of the same fact.
        Cascade-skipped calls (never dispatched at all) are correctly
        excluded, since they never enter either list. As a side benefit,
        this stays correct for free if a future plan-repair mechanism ever
        adds a second generation round to this family.
        """
        from ...utils.dag import is_dispatched_call
        return (
            sum(1 for c in self.completed if is_dispatched_call(c))
            + len(self.failed_statements)
        )


@dataclass(slots=True)
class ReActTask(JsonToolAgentTask):
    """
    ReActAgent-flavored task: one registered-tool call generated, resolved,
    and dispatched per round, via provider-native structured output
    (``REACT_OUTPUT_SCHEMA``) -- the per-step sibling to ``PlanActTask``'s
    one-shot multi-call plan.

    No fixed-size preallocated board and no observability-decay window
    (both dropped from the pre-rewrite shape, along with ``next_step_index``/
    ``step_meta``) -- every round renders a full snapshot of ``completed``/
    ``cache`` instead (``utils.dag.render_completed_as_json``/
    ``render_cache_snapshot``, reused unmodified). No
    ``pending: list[list[DagToolCall]]`` batch field either -- unlike
    ``PlanActTask``, this family dispatches exactly one call per round,
    never a concurrency batch. No ``__post_init__`` -- matches every other
    ``*Task`` in this family: an in-flight, internal-only object, not a real
    construction-time boundary.

    Fields
    ------
    completed : list[DagToolCall]
        Every call actually dispatched and successful so far this run, in
        commit order. Becomes ``JsonToolAgentRecord.statements`` verbatim
        (normalized to a tuple) at commit time -- same convention as
        ``PlanActTask.completed``.

    failed_statements : list[DagToolCall]
        Every call actually dispatched that raised, in the order observed.
        A resolution failure (a ``$name`` reference that doesn't resolve, or
        a resolved value's type mismatching the target tool's parameter
        contract) never reaches this list at all under this family's
        design -- both are caught and fed back for regeneration inside
        ``think()``'s own retry loop, before a call is ever accepted as
        this round's decision. Only a real dispatch failure (the tool
        itself raised once actually invoked) lands here.

    cache : dict[str, Any]
        identifier -> resolved value for every call in ``completed``, plus
        ``task_result_i`` entries seeded once at ``_initialize_task`` from
        prior turns. Same role as ``PlanActTask.cache``.

    constant_values : dict[str, Any]
        Registered-constant name -> value, populated once by
        ``ReActAgent._initialize_task`` and never touched again. Same role
        as ``PlanActTask.constant_values``.

    generated_step : DagToolCall | None
        The one call ``think()`` validated and resolved this round -- set
        only once both ``utils.dag.validate_calls`` and
        ``utils.dag.resolve_call_args`` succeed against it, ``None`` before
        that and after ``act()`` consumes it. Retyped from the pre-rewrite
        shape's ``Any = NO_VAL`` (which held a ``(BlackboardSlot, int,
        str)`` tuple under the old duration/observability design) --
        ``None`` is the idiomatic "not yet decided" value for a field
        genuinely typed ``Optional[DagToolCall]``, so no ``NO_VAL``
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
    completed: list[DagToolCall] = field(default_factory=list)
    failed_statements: list[DagToolCall] = field(default_factory=list)
    cache: dict[str, Any] = field(default_factory=dict)
    constant_values: dict[str, Any] = field(default_factory=dict)
    generated_step: Optional[DagToolCall] = None
    resolved_args: Optional[dict[str, Any]] = None
    last_call_failed: bool = False

    @property
    def tool_calls_used(self) -> int:
        """
        Derived, not stored -- ``completed``/``failed_statements`` are
        already the single source of truth. Deliberately gates **both**
        halves through ``is_dispatched_call``, a real, intentional
        divergence from ``PlanActTask.tool_calls_used``'s bare
        ``len(self.failed_statements)``: a call to ``return_tool`` is a
        genuine dispatch in this family (never a synthesized,
        non-dispatched ``RETURN_ALIAS`` sentinel the way ``PlanActTask``'s
        return call is), so a *failed* return call should be exempt from
        the budget count the same way a successful one already is, for
        symmetry. Since a resolution failure never reaches
        ``failed_statements`` at all under this family's design (see that
        field's own docstring above), every entry here is guaranteed to be
        a real dispatch attempt regardless.
        """
        from ...utils.dag import is_dispatched_call
        return (
            sum(1 for c in self.completed if is_dispatched_call(c))
            + sum(1 for c in self.failed_statements if is_dispatched_call(c))
        )


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
        (mirrors ``JsonToolAgentTask.tool_calls_used: int = 0``'s identical
        precedent). ``think()``/``async_think()`` read this field directly
        instead of any agent-level attribute — there is no construction-time
        equivalent left; the round budget is purely per-invocation now.
    """
    thoughts: list[str | int | float | bool | list | dict | None] = field(default_factory=list)
    thinking_rounds: int = 1
