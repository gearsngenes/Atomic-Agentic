from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from ...constants.core import NO_VAL
from .blackboard_models import ToolStatement
from .records import AgentRecord, LLMRecord

__all__ = [
    "AgentTask",
    "JsonToolAgentTask",
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
        task shape carries). Renamed from the prior ``retries_used`` since
        this family's generation model is ``output_structure`` +
        regen-retry loop, not the old free-text-JSON retry loop.
    """
    regenerations_used: int = 0


@dataclass(slots=True)
class ScriptActAgentTask(AgentTask):
    """
    ScriptActAgent-flavored task -- a sibling to JsonToolAgentTask, not a
    subclass (ScriptActAgent is a new agent family, not a JsonToolAgent
    subclass, per this branch's established convention).

    No __post_init__ -- matches AgentTask's own family-wide convention of
    zero constructor-time validation (an in-flight, internal-only object,
    not a real boundary, per 01-overview.md Section 4).

    Fields
    ------
    completed : list[ToolStatement]
        Every slot that executed successfully so far this run, in commit
        order. Purely historical -- nothing here is ever mutated once a
        slot lands in this list. Becomes ScriptActAgentRecord.statements
        verbatim (normalized to a tuple) at commit time. A slot whose
        dispatch raised is never appended here -- see failed_statements.

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

    tool_calls_used : int
        Cumulative count of dispatched (non-``rhs_assign``/``return``)
        calls actually dispatched so far this invoke, across every
        generation round -- registered-tool and approved-builtin calls
        counted identically (no per-category exemption), never decremented.
        Incremented only in ``_apply_batch_results`` (a real dispatch
        happened, whether it succeeded or failed); never in ``prepare()``
        (a resolution failure means nothing was ever dispatched). Read by
        ``_process_generation_output`` to compute the remaining budget
        passed into ``validate_references``, and by
        ``_render_task_messages`` to show the model its own remaining
        ``tool_calls_limit`` on a repair round.

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

    cache : dict[str, Any]
        identifier -> resolved value for every slot in ``completed``, kept
        in sync as slots complete. Shaped to be passed directly as
        utils/agents.py's ``resolve_statement_args(statement, resolved)``'s
        ``resolved`` argument -- an O(1) lookup instead of scanning
        ``completed``.

    constant_values : dict[str, Any]
        Registered-constant name -> value, populated exactly once by
        ``ScriptActAgent._initialize_task`` (deep-copied per constant except
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
        ``ToolStatement.batch_index`` values stay globally unique across a
        whole invoke.

    failed_statements : list[ToolStatement]
        Every slot that failed, in the order the failure was observed,
        across every round this invoke -- populated by BOTH
        ``prepare()`` (a resolution/binding failure, ``slot.exception`` set
        to a synthesized ``ToolAgentError``, mirroring ``PlanActTask``'s
        own resolution-failure pattern) and ``_apply_batch_results``
        (a real dispatch failure, ``slot.exception`` set to the actual
        raised value) -- unlike the old shape, where only
        ``_apply_batch_results`` ever appended here. Never cleared or
        mutated once appended; ``repair_batch_start`` marks where each
        triggering batch's own slice begins. Becomes
        ScriptActAgentRecord.failed_statements verbatim (normalized to a
        tuple) at commit time.
    """
    completed: list[ToolStatement] = field(default_factory=list)
    pending: list[list[ToolStatement]] = field(default_factory=list)
    cache: dict[str, Any] = field(default_factory=dict)
    constant_values: dict[str, Any] = field(default_factory=dict)
    regenerations_used: int = 0
    resolved_args: list[dict[str, Any]] = field(default_factory=list)
    needs_repair: bool = False
    repair_rounds_used: int = 0
    tool_calls_used: int = 0
    repair_batch_start: int = 0
    batch_counter: int = 0
    failed_statements: list[ToolStatement] = field(default_factory=list)


@dataclass(slots=True)
class PlanActTask(JsonToolAgentTask):
    """
    PlanActAgent-flavored task -- ``ToolStatement``-based batch-execution
    fields (``completed``/``pending``/``cache``/``constant_values``/
    ``resolved_args``/``failed_statements``), with no continuation-related
    fields at all: ``think()`` runs exactly once, ever, per invoke, so there
    is nothing analogous to a voluntary multi-round continuation flag,
    round counter, or continuation note to carry.

    Fields
    ------
    completed : list[ToolStatement]
        Every call that executed successfully this run, in commit order
        (the ``RETURN_ALIAS`` call included once it executes).

    pending : list[list[ToolStatement]]
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

    failed_statements : list[ToolStatement]
        Every call whose dispatch actually raised this run -- not
        cascade-skipped calls, which are never attempted and never appear
        here (see ``find_cascade_failures``'s own contract). Becomes
        ``JsonToolAgentRecord.failed_statements`` verbatim at commit time.

    No ``batch_counter`` field -- a single ``compile_batches`` call per
    invoke needs no cross-round ``ToolStatement.batch_index`` uniqueness
    tracking; a local variable in ``think()`` suffices.
    """
    completed: list[ToolStatement] = field(default_factory=list)
    pending: list[list[ToolStatement]] = field(default_factory=list)
    cache: dict[str, Any] = field(default_factory=dict)
    constant_values: dict[str, Any] = field(default_factory=dict)
    resolved_args: list[dict[str, Any]] = field(default_factory=list)
    failed_statements: list[ToolStatement] = field(default_factory=list)

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
        from ...utils.agents import is_dispatched
        return (
            sum(1 for c in self.completed if is_dispatched(c))
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
    ``cache`` instead (``utils.sigils.render_completed_as_json``/
    ``render_cache_snapshot``, reused unmodified). No
    ``pending: list[list[ToolStatement]]`` batch field either -- unlike
    ``PlanActTask``, this family dispatches exactly one call per round,
    never a concurrency batch. No ``__post_init__`` -- matches every other
    ``*Task`` in this family: an in-flight, internal-only object, not a real
    construction-time boundary.

    Fields
    ------
    completed : list[ToolStatement]
        Every call actually dispatched and successful so far this run, in
        commit order. Becomes ``JsonToolAgentRecord.statements`` verbatim
        (normalized to a tuple) at commit time -- same convention as
        ``PlanActTask.completed``.

    failed_statements : list[ToolStatement]
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
    completed: list[ToolStatement] = field(default_factory=list)
    failed_statements: list[ToolStatement] = field(default_factory=list)
    cache: dict[str, Any] = field(default_factory=dict)
    constant_values: dict[str, Any] = field(default_factory=dict)
    generated_step: Optional[ToolStatement] = None
    resolved_args: Optional[dict[str, Any]] = None
    last_call_failed: bool = False

    @property
    def tool_calls_used(self) -> int:
        """
        Derived, not stored -- ``completed``/``failed_statements`` are
        already the single source of truth. Deliberately gates **both**
        halves through ``is_dispatched``, a real, intentional
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
        from ...utils.agents import is_dispatched
        return (
            sum(1 for c in self.completed if is_dispatched(c))
            + sum(1 for c in self.failed_statements if is_dispatched(c))
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
