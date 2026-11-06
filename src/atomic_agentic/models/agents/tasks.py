from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from ...constants.core import NO_VAL
from .blackboard_models import ToolStatement
from .context import ContextVariable
from .records import AgentRecord, LLMRecord

__all__ = [
    "AgentTask",
    "ToolAgentTask",
    "ScriptActAgentTask",
    "PlanActTask",
    "ReActTask",
    "ThinkingTask",
    "ScratchpadTask",
]


@dataclass(slots=True)
class AgentTask:
    """
    Base run-task contract for one Agent invocation.

    Drives an invocation through ``create_task`` then
    ``think``/``prepare``/``act`` each round. ``BasicAgent`` uses this base
    class directly; it needs nothing beyond these fields.

    Fields
    ------
    turns : list[AgentRecord]
        The selected conversation turns for this invocation (same list
        ``create_task`` pinned). Used for ``prev`` linkage when
        building the completed record (``turns[-1]``, if any).

    inputs : dict[str, Any]
        The full, untouched ``inputs`` mapping from the base ``Agent``
        invocation lifecycle.

    user_prompt : str
        The fully-resolved prompt string produced by ``pre_invoke`` for this
        invocation (matches ``AgentRecord.user_prompt`` 1:1). Defaults to
        ``""`` so a task can exist before ``pre_process`` runs; ``pre_process``
        always overwrites it.

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
        ``BasicAgent``, the executed return-tool value for ``ToolAgent``
        and its subclasses. ``NO_VAL`` until ``act`` sets it on the round
        that completes the task.

    historic_messages : list[dict[str, str]]
        Rendered ``turns``, built lazily once per invoke by
        ``Agent.summarize`` and reused for the rest of it.
        Never rebuilt mid-invoke — turn-rendering is phase-invariant,
        governed only by the bound conversation's preview limits, read from
        the conversation (``task.conversation``) when it is built.

    task_messages : list[dict[str, str]]
        The task-message list most recently built by ``_render_task_messages``
        (what ``LLMRecord`` copies right after a render). A family that retries
        within a phase builds it once per phase and grows it with
        ``additional_messages``, clearing it at its own phase boundary (e.g.
        ``PlanActAgent`` once planning finishes). A family that never retries
        (``ThinkingAgent``) rebuilds it on every call.

    conversation : str | None
        Registry key of the conversation this invocation is bound to. ``None``
        until ``create_task`` populates it.

    context : dict[str, ContextVariable]
        Per-call working copy of the bound conversation's context, seeded by
        ``create_task`` (a shallow copy; the variables are shared frozen
        objects) and written back by ``record`` -- see its docstring for the
        rules. ``post_process`` replaces the value of existing writable variables
        from ``bind_response_to``; any other writer (subclass loop steps) must skip
        or raise on a ``readonly`` variable itself. This dict is not guarded.

    pre_inputs, post_inputs, agentic_inputs : dict[str, Any]
        Views of ``inputs`` sliced by consumer (pre-invoke, post-invoke,
        agent-specific). A name consumed by more than one goes into every
        bucket that declares it. ``post_process`` then overlays the response
        pieces bound by ``bind_response_to`` onto ``post_inputs`` (only names
        the post stage declares), so afterwards it is exactly what
        ``post_invoke`` received.

    run_id, started_at, ended_at : str | None, datetime | None, datetime | None
        Identity and timing shared by the result and the record built from
        this task. ``None`` until ``create_task`` mints them.
    """
    turns: list[AgentRecord]
    inputs: dict[str, Any]
    system_prompt_name: str | None

    user_prompt: str = ""
    llm_records: list[LLMRecord] = field(default_factory=list)
    complete: bool = False
    generated_response: Any = NO_VAL
    historic_messages: list[dict[str, str]] = field(default_factory=list)
    task_messages: list[dict[str, str]] = field(default_factory=list)

    conversation: str | None = None
    context: dict[str, ContextVariable] = field(default_factory=dict)
    pre_inputs: dict[str, Any] = field(default_factory=dict)
    post_inputs: dict[str, Any] = field(default_factory=dict)
    agentic_inputs: dict[str, Any] = field(default_factory=dict)
    run_id: str | None = None
    started_at: datetime | None = None
    ended_at: datetime | None = None


@dataclass(slots=True)
class ToolAgentTask(AgentTask):
    """
    Base task shape shared by every concrete ``ToolAgent`` family
    (``ScriptActAgent``/``PlanActAgent``/``ReActAgent``). Promotes every
    field genuinely shared by all three concrete agents.

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
        ``task_result_i`` entries seeded once by ``_seed_task``.

    constant_values : dict[str, Any]
        Registered-constant name -> value, populated once by each concrete
        agent's own ``_seed_task`` and never touched again.

    regenerations_used : int
        Cumulative regeneration attempts consumed across every generation
        call this run -- irreducible; a rejected/regenerated draft leaves
        no artifact anywhere else to derive this from (contrast
        ``tool_calls_used``, fully derivable from ``completed``/
        ``failed_statements``).
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
        either list, whether it ultimately succeeded or raised --
        ``is_dispatched`` is purely a function of ``call.tool``, not of
        whether dispatch was ever actually attempted, so a call the plan
        committed to spends budget whether or not the tool itself ever ran,
        regardless of which lifecycle stage (``prepare()``'s resolution, or
        a real dispatch attempt) caught the failure.
        """
        from ...utils.agents import is_dispatched
        return (
            sum(1 for c in self.completed if is_dispatched(c))
            + sum(1 for c in self.failed_statements if is_dispatched(c))
        )


@dataclass(slots=True)
class ScriptActAgentTask(ToolAgentTask):
    """
    ScriptActAgent-flavored task, a ``ToolAgentTask`` subclass.
    ``completed``/``failed_statements``/``cache``/``constant_values``/
    ``regenerations_used``/``tool_calls_used`` are all inherited; only
    genuinely ``ScriptActAgent``-specific fields remain declared here.

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
        never counts the free initial plan. Incremented
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
        precedent. There is no separately-maintained note string; rendering
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
        whole invoke. Note: ``failed_statements`` (inherited from
        ``ToolAgentTask``; see that class's own docstring) is populated by
        BOTH ``prepare()`` (a resolution/binding failure, ``slot.exception``
        set to a synthesized ``ToolAgentError``, mirroring ``PlanActTask``'s
        own resolution-failure pattern) and ``_apply_batch_results`` (a
        real dispatch failure, ``slot.exception`` set to the actual raised
        value). ``repair_batch_start`` marks where each triggering batch's
        own slice into it begins.
    """
    pending: list[list[ToolStatement]] = field(default_factory=list)
    resolved_args: list[dict[str, Any]] = field(default_factory=list)
    needs_repair: bool = False
    repair_rounds_used: int = 0
    repair_batch_start: int = 0
    batch_counter: int = 0


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
    ``tool_calls_used`` formula (``ToolAgentTask``) holds correctly here
    too: a cascade-skipped call is never dispatched, so it never enters
    ``completed`` or ``failed_statements`` either way.
    """
    pending: list[list[ToolStatement]] = field(default_factory=list)
    resolved_args: list[dict[str, Any]] = field(default_factory=list)


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

    No fixed-size preallocated board and no observability-decay window --
    every round renders a full snapshot of ``completed``/``cache`` instead
    (``utils.sigils.render_completed_as_json``/``render_cache_snapshot``,
    reused unmodified). No ``pending: list[list[ToolStatement]]`` batch
    field either -- unlike ``PlanActTask``, this family dispatches exactly
    one call per round, never a concurrency batch. No ``__post_init__`` --
    matches every other ``*Task`` in this family: an in-flight,
    internal-only object, not a real construction-time boundary.

    A resolution failure (a ``$name`` reference that doesn't resolve, or a
    resolved value's type mismatching the target tool's parameter contract)
    never reaches ``failed_statements`` at all under this family's design --
    both are caught and fed back for regeneration inside ``think()``'s own
    retry loop, before a call is ever accepted as this round's decision.
    Only a real dispatch failure (the tool itself raised once actually
    invoked) lands there -- which is also why the inherited
    ``tool_calls_used`` formula (gating both ``completed`` and
    ``failed_statements`` through ``is_dispatched``) holds correctly here
    too: every entry in either list is guaranteed to be a real dispatch
    attempt regardless.

    Fields
    ------
    generated_step : ToolStatement | None
        The one call ``think()`` validated and resolved this round -- set
        only once both ``utils.sigils.translate_calls`` and
        ``utils.agents.resolve_statement_args`` succeed against it, ``None``
        before that and after ``act()`` consumes it. ``None`` is the
        idiomatic "not yet decided" value for a field genuinely typed
        ``Optional[ToolStatement]``, so no ``NO_VAL`` sentinel is needed
        here.

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


@dataclass(slots=True)
class ThinkingTask(AgentTask):
    """
    ThinkingAgent-family task.

    No phase field: ``AgentTask.system_prompt_name`` is the phase indicator. The
    name ``"role"`` means the reply phase; any other value (including ``None``)
    means thinking is still in progress. A strategy may use several thinking
    prompt names (for example one per step). The base ``ThinkingAgent`` switches
    the name to ``"role"`` when its ``_thinking_done`` hook turns true.

    Fields
    ------
    thoughts : list[Any]
        Task-local list of thoughts, appended by the strategy's
        ``_generate_thoughts``. Each element may be any type, including nested
        structures; how they are rendered and interpreted is the strategy's
        (``_render_thoughts``). Persisted onto the completed
        ``ThinkingAgentRecord.thoughts`` tuple verbatim by
        ``_build_record_from_task``. There is no round counter or budget here:
        those, if any, belong to the strategy.
    """
    thoughts: list[Any] = field(default_factory=list)


@dataclass(slots=True)
class ScratchpadTask(ThinkingTask):
    """
    ScratchpadAgent task.

    Fields
    ------
    more_thinking_required : bool
        The model's latest "keep thinking" decision under ``stop_early``.
        ``True`` (the default) means the model has not ended thinking;
        ``ScratchpadAgent._generate_thoughts`` overwrites it after every
        thinking call when ``stop_early`` is on and never touches it
        otherwise.
    """
    more_thinking_required: bool = True
