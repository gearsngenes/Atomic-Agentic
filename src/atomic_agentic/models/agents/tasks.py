from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ...constants.core import NO_VAL
from .blackboard_models import BlackboardSlot
from .records import AgentRecord, LLMRecord
from .thought_models import AgentThought
from .thought_models2 import AgentThought2

__all__ = [
    "AgentTask",
    "ToolAgentTask",
    "PlanActTask",
    "ReActTask",
    "ReActStepMeta",
    "ThinkingTask",
]


@dataclass(slots=True)
class AgentTask:
    """
    Base run-task contract for one Agent invocation.

    Drives an invocation through ``_initialize_task``/``think``/``prepare``/
    ``execute``. ``BasicAgent2`` uses this base class directly; it needs
    nothing beyond these fields.

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
        task = execute(task)``.

    generated_response : Any
        The record's produced-response equivalent — raw LLM text for
        ``BasicAgent``, the executed return-tool value for ``ToolAgent`` and
        its subclasses. ``NO_VAL`` until ``execute``/``async_execute`` sets
        it on the round that completes the task.

    final_response : Any
        ``post_invoke``'s transformation of ``generated_response`` — the
        value that becomes ``AgentResult.result``. ``NO_VAL`` until
        ``Agent2.invoke``/``async_invoke`` sets it, after the task loop
        finishes and ``post_invoke`` has run, right before
        ``_commit_emit`` builds the final record/result pair.

    historic_messages : list[dict[str, str]]
        Rendered ``turns``, built lazily once per invoke by
        ``Agent2._render_history_messages`` and reused for the rest of it.
        Never rebuilt mid-invoke — turn-rendering is phase-invariant,
        governed only by ``assistant_response_source``/
        ``response_preview_limit``, both static per-agent config.

    task_messages : list[dict[str, str]]
        Phase-scoped LLM-facing content, lazily built by each subclass's
        ``render_task`` when empty and grown by ``additional_messages``
        across generation retries within one phase. Cleared by the owning
        subclass at its own phase boundary (e.g. ``PlanActAgent`` once
        planning finishes; ``ReActAgent`` after each step commits).

    thoughts : list[list[AgentThought2]]
        ``Agent2`` thinking-round accumulator, one inner list per
        completed round. Round count is always ``len(self.thoughts)`` --
        no separate counter field exists, so it can never drift from what
        was actually recorded. Empty for agents that never call
        ``Agent2.think`` or that have thinking disabled.

    keep_thinking : bool
        Whether another thinking round may run. Seeded at
        ``Agent2._initialize_task`` as ``self._thinking_rounds != 0``.
        Flipped to ``False`` by ``Agent2.think`` when the
        ``|STOP_THINKING|`` sentinel is seen or the round cap is reached.
        Defaults to ``True`` here as a neutral placeholder -- ``Agent2``
        always overwrites it at initialization time, so the default is
        never observed there. The old (pre-``Agent2``) family sharing this
        base field -- ``ThinkingTask``/``ToolAgentTask``/``PlanActTask``/
        ``ReActTask`` -- never reads or writes it at all; it simply sits at
        this default forever for those tasks, unobserved only because
        nothing looks at it, not because it's overwritten.
    """
    turns: list[AgentRecord]
    inputs: dict[str, Any]
    user_prompt: str
    system_prompt_name: str | None

    llm_records: list[LLMRecord] = field(default_factory=list)
    complete: bool = False
    generated_response: Any = NO_VAL
    final_response: Any = NO_VAL
    historic_messages: list[dict[str, str]] = field(default_factory=list)
    task_messages: list[dict[str, str]] = field(default_factory=list)
    thoughts: list[list[AgentThought2]] = field(default_factory=list)
    keep_thinking: bool = True


@dataclass(slots=True)
class ToolAgentTask(AgentTask):
    """
    ToolAgent-flavored task.

    Fields
    ------
    cache_blackboard : list[BlackboardSlot]
        Snapshot of the persisted ``self._blackboard`` at invoke start when
        context is enabled. Previous invocation results are available here
        and can be referenced via ``<<__cN__>>`` placeholders.

    running_blackboard : list[BlackboardSlot]
        Plan-local slots (0-based indices) created during this invoke.
        Populated by ``_initialize_task()``, planned by
        ``_prepare_next_batch()``, and executed by
        ``_execute_prepared_batch()``. If ``context_enabled=True``, executed
        slots are persisted and merged into ``self._blackboard`` by
        ``_build_record_from_task``.

    Placeholder Semantics & Resolvability
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    **Cached Placeholder** (``<<__cN__>>``)
        Resolvable iff ``0 <= N < len(cache_blackboard)`` AND
        ``cache_blackboard[N].is_executed() == True``.

    **Step Placeholder** (``<<__sN__>>``)
        Resolvable iff ``0 <= N < len(running_blackboard)`` AND
        ``running_blackboard[N].is_executed() == True``.

    executed_steps : set[int]
        Running plan indices that have been executed.

    prepared_steps : list[int]
        Running plan indices ready for execution in the next batch. Must be
        set by ``_prepare_next_batch()`` and consumed by ``_progress``.

    tool_calls_used : int
        Count of non-return tool calls executed so far.

    llm_records : list[LLMRecord]
        Inherited from ``AgentTask``. Seeded at construction time —
        non-empty for subclasses that generate up front (e.g. PlanAct's
        one-shot plan), empty for subclasses that generate lazily during the
        loop (e.g. ReAct's per-step planning) — and appended to as further
        generations occur.

    valid_cache_indices : frozenset[int]
        Cache-blackboard indices reachable in this conversation — entries
        that are EXECUTED and belong to a record in the current ``turns``
        chain. Computed once in ``_initialize_task`` from ``turns`` via
        ``_compute_cache_index_sets``; empty when ``context_enabled=False``.

    failed_cache_indices : frozenset[int]
        Cache-blackboard indices that belong to this conversation but whose
        slots have FAILED status. Disjoint with ``valid_cache_indices``.
        Referenced during generation to produce targeted LLM feedback.

    retries_used : int
        Cumulative retry attempts consumed across all step generations in
        this run. ``PlanActAgent``'s retry budget is a local counter inside
        ``_generate_plan``, not stored on the task; only ``ReActAgent``
        actually uses this field. Declared here (rather than on
        ``ReActTask``) so it's available uniformly to every subclass,
        including ``PlanActTask``.
    """
    cache_blackboard: list[BlackboardSlot] = field(default_factory=list)
    running_blackboard: list[BlackboardSlot] = field(default_factory=list)

    executed_steps: set[int] = field(default_factory=set)
    prepared_steps: list[int] = field(default_factory=list)

    tool_calls_used: int = 0

    valid_cache_indices: frozenset[int] = field(default_factory=frozenset)
    failed_cache_indices: frozenset[int] = field(default_factory=frozenset)

    retries_used: int = 0


@dataclass(slots=True)
class PlanActTask(ToolAgentTask):
    """
    PlanActAgent-flavored task.

    Fields
    ------
    batches : list[list[int]]
        Pre-compiled topologically-sorted batches. Each batch is a list of
        plan-local indices that can execute concurrently. Created during
        ``_initialize_task()`` via ``_compile_batches_from_deps()``.

        Example: ``[[0, 1], [2, 3], [4]]`` means:
        - Batch 0: steps 0 and 1 execute together
        - Batch 1: steps 2 and 3 execute together (after batch 0)
        - Batch 2: step 4 executes (after batch 1; typically the return step)

    batch_index : int
        Cursor pointing to the next batch to prepare. Starts at 0; incremented
        after each batch is prepared. Task completes when
        ``batch_index >= len(batches)`` and the return step has executed.

    Workflow
    ~~~~~~~~
    1. ``_initialize_task()`` creates batches and sets ``batch_index=0``.
    2. Each ``_progress`` round:
       - ``_prepare_next_batch()`` reads ``batches[batch_index]``, resolves
         placeholders for that batch.
       - ``_execute_prepared_batch()`` runs the batch concurrently.
       - ``batch_index`` incremented for the next round.
    3. When ``batch_index >= len(batches)``, ``_prepare_next_batch()`` raises
       — in practice this is never reached, since the final batch always
       contains the return step, which sets ``task.complete = True`` and
       ends the ``_progress`` loop first.
    """
    batches: list[list[int]] = field(default_factory=list)
    batch_index: int = 0


@dataclass(slots=True)
class ReActStepMeta:
    """
    Per-slot metadata for a single ReAct step.

    Pairs the raw-result visibility counter with the one-sentence description
    for the same slot.

    Fields
    ------
    observable : int
        Remaining prepare-turns during which this step's raw result is shown
        as ``observable_result`` in the running-plan snapshot. Decremented
        after each successful generation turn. ``0`` means not visible.

    description : str
        One-sentence intent summary rendered in the running-plan snapshot so
        the LLM understands what the step was intended to do without needing
        raw result visibility.
    """
    observable: int = 0
    description: str = ""


@dataclass(slots=True)
class ReActTask(ToolAgentTask):
    """
    ReActAgent-flavored task. Tracks cursor and per-slot metadata for
    step-by-step reactive planning.

    Fields
    ------
    next_step_index : int
        Cursor for the next plan-local ``running_blackboard`` slot to fill.
        Starts at 0 and increments after each step is prepared.

        Dual role:
        1. Allocation cursor: determines which slot index gets the next
           prepared step.
        2. Dependency cutoff: any ``<<__sN__>>`` placeholder in newly
           prepared args must satisfy ``N < next_step_index``.

    step_meta : list[ReActStepMeta]
        Per-slot metadata for each slot in ``running_blackboard``. Always the
        same length as ``running_blackboard``. Both fields are written by
        ``_prepare_next_batch``/``_aprepare_next_batch`` at the index of the
        slot being prepared.

    ``retries_used`` is declared on ``ToolAgentTask`` (see that class) — its
    behavior originates here: incremented by ``_generate_next_step`` on each
    failed attempt; checked against ``self._generation_retries`` before
    permitting a retry.

    Workflow
    ~~~~~~~~
    Each ``_progress`` round:

    1. ``_prepare_next_batch`` (a single step):
       - Build a fresh temporary copy of the static base messages.
       - Append a running-plan snapshot and a step-request message.
       - Request the next step from the LLM.
       - Validate ``idx == next_step_index``; fill ``running_blackboard[idx]``;
         write ``step_meta[idx]``; set ``prepared_steps=[idx]``; increment
         ``next_step_index``.
    2. ``_execute_prepared_batch`` (base ``ToolAgent``, shared):
       - Run the prepared single-step batch; store the result in
         ``running_blackboard[idx]``.
    3. Continue until the return tool executes, setting ``task.complete``.
    """
    next_step_index: int = 0
    step_meta: list[ReActStepMeta] = field(default_factory=list)


@dataclass(slots=True)
class ThinkingTask(AgentTask):
    """
    ThinkingAgent-flavored task.

    No ``phase`` field: ``AgentTask.system_prompt_name`` doubles as the
    phase discriminator. The name ``"role"`` is reserved for the reply
    phase across the whole family; any other value means a thinking round
    is still active. This is why the field is required (no default) on the
    base ``AgentTask`` — every concrete subclass must decide it explicitly,
    and here that decision *is* the phase.

    Fields
    ------
    retries_used : int
        Cumulative generation-retry attempts across all thinking rounds in
        this run. Mirrors ``ToolAgentTask.retries_used``.

    rounds_used : int
        Cumulative successful thinking rounds completed. Checked by
        subclass ``_think`` implementations against
        ``self._max_thinking_rounds`` to force the transition to the reply
        phase regardless of any self-declared continuation signal.

    thoughts : list[AgentThought]
        Task-local accumulator, mirrors ``ToolAgentTask.running_blackboard``
        — merged into the agent-level persisted ``self._thoughts`` only at
        ``_build_record_from_task`` time, never appended to the agent-level
        list mid-run.
    """
    retries_used: int = 0
    rounds_used: int = 0
    thoughts: list[AgentThought] = field(default_factory=list)
