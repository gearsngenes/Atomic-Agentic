from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ...constants.core import NO_VAL
from .blackboard_models import BlackboardSlot
from .records import AgentRecord, LLMRecord
from .thought_models import AgentThought

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
        ``BasicAgent``, the executed return-tool value for ``ToolAgent`` and
        its subclasses. ``NO_VAL`` until ``act`` sets it on the round that
        completes the task.

    historic_messages : list[dict[str, str]]
        Rendered ``turns``, built lazily once per invoke by
        ``Agent._render_historic_messages`` and reused for the rest of it.
        Never rebuilt mid-invoke — turn-rendering is phase-invariant,
        governed only by ``assistant_response_source``/
        ``response_preview_limit``, both static per-agent config.

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
    ToolAgent-flavored task.

    Fields
    ------
    running_blackboard : list[BlackboardSlot]
        Plan-local slots (0-based indices) created during this invoke.
        Populated by ``_initialize_task()``, planned by ``prepare()``, and
        executed by ``act()``. If ``context_enabled=True``, executed slots
        are persisted and merged into ``self._blackboard`` by
        ``_build_record_from_task``.

    Placeholder Semantics & Resolvability
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    **Cached Placeholder** (``<<__cN__>>``)
        Resolvable iff ``0 <= N < len(self._blackboard)`` AND
        ``self._blackboard[N].is_executed() == True`` — resolved directly
        against the owning ``ToolAgent``'s live, always-persisted
        blackboard, not a task-local snapshot.

    **Step Placeholder** (``<<__sN__>>``)
        Resolvable iff ``0 <= N < len(running_blackboard)`` AND
        ``running_blackboard[N].is_executed() == True``.

    executed_steps : set[int]
        Running plan indices that have been executed.

    prepared_steps : list[int]
        Running plan indices ready for execution in the next batch. Must be
        set by ``prepare()`` and consumed by ``act()``.

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
        Cumulative retry attempts consumed across all generation attempts in
        this run. Both ``PlanActAgent`` (``_generate_plan``/
        ``_agenerate_plan``) and ``ReActAgent`` (``_generate_next_step``/
        ``_agenerate_next_step``) read and increment this field directly on
        the task — neither keeps a separate local counter. Declared here
        (rather than on ``ReActTask``) so it's available uniformly to every
        subclass, including ``PlanActTask``.
    """
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
    generated_plan : Any
        Holds the validated (but not yet compiled) plan — the
        ``list[BlackboardSlot]`` ``think()``/``async_think()`` produce —
        until ``prepare()``'s first call compiles it into
        ``running_blackboard``/``batches``/``batch_index``. Unlike
        ``ReActTask.generated_step``, never reset back to ``NO_VAL``: it's
        this hook's own one-time-generation marker (``think()`` no-ops once
        it's set), not a per-round handoff.

    batches : list[list[int]]
        Pre-compiled topologically-sorted batches. Each batch is a list of
        plan-local indices that can execute concurrently. Compiled from
        ``generated_plan`` during ``prepare()``'s first call via
        ``_compile_batches_from_deps()``.

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
    1. ``think()`` generates and validates the whole plan, once, storing it
       on ``generated_plan``; every later round's ``think()`` is a no-op.
    2. ``prepare()``'s first call (``batches`` still empty) compiles
       ``generated_plan`` into batches and sets ``batch_index=0``.
    3. Each round after that:
       - ``prepare()`` reads ``batches[batch_index]``, resolves placeholders
         for that batch.
       - ``act()`` runs the batch concurrently.
       - ``batch_index`` incremented for the next round.
    4. When ``batch_index >= len(batches)``, ``prepare()`` raises — in
       practice this is never reached, since the final batch always
       contains the return step, which sets ``task.complete = True`` and
       ends the loop first.
    """
    batches: list[list[int]] = field(default_factory=list)
    batch_index: int = 0
    generated_plan: Any = NO_VAL


@dataclass(slots=True)
class ReActStepMeta:
    """
    Per-slot metadata for a single ReAct step.

    Fields
    ------
    observable : int
        Remaining prepare-turns during which this step's raw result is shown
        as ``observable_result`` in the running-plan snapshot. Decremented
        after each successful generation turn. ``0`` means not visible.

    The one-sentence intent summary previously carried here as ``description``
    now lives directly on ``BlackboardSlot.reason``, set once at
    slot-construction time by generation rather than stamped separately here.
    """
    observable: int = 0


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
        ``prepare``/``async_prepare`` at the index of the slot being
        prepared.

    generated_step : Any
        Holds the ``(BlackboardSlot, int)`` tuple ``think()``/
        ``async_think()`` produces each round (the freshly-generated,
        not-yet-applied step and its observability duration), until
        ``prepare()`` unpacks it and resets this back to ``NO_VAL``. The
        step's reason lives on the ``BlackboardSlot`` itself (set at
        generation time), not as a separate tuple element. Needed because
        ``think()`` and ``prepare()`` are independent top-level calls from
        the base loop — there's no local Python scope to pass the decision
        through directly the way a single fused method could.

    ``retries_used`` is declared on ``ToolAgentTask`` (see that class) — its
    behavior originates here: incremented by ``_generate_next_step`` on each
    failed attempt; checked against ``self._generation_retries`` before
    permitting a retry.

    Workflow
    ~~~~~~~~
    Each ``think``/``prepare``/``act`` round:

    1. ``think()`` (a single step):
       - Validate cursor/step_meta bookkeeping (``_validate_react_prepare_state``).
       - Build a fresh temporary copy of the static base messages.
       - Append a running-plan snapshot and a step-request message.
       - Request the next step from the LLM; validate it end-to-end.
       - Stash the validated step onto ``task.generated_step``.
    2. ``prepare()``:
       - Unpack ``task.generated_step``, reset it to ``NO_VAL``.
       - Cascade-check dependencies; resolve placeholders.
       - Fill ``running_blackboard[idx]``; write ``step_meta[idx]``; set
         ``prepared_steps=[idx]``; increment ``next_step_index``.
    3. ``act()`` (base ``ToolAgent``, final):
       - Run the prepared single-step batch; store the result in
         ``running_blackboard[idx]``.
    4. Continue until the return tool executes, setting ``task.complete``.
    """
    next_step_index: int = 0
    step_meta: list[ReActStepMeta] = field(default_factory=list)
    generated_step: Any = NO_VAL


@dataclass(slots=True)
class ThinkingTask(AgentTask):
    """
    SelfAskAgent-flavored task.

    No ``phase`` field: ``AgentTask.system_prompt_name`` doubles as the
    phase discriminator. The name ``"role"`` is reserved for the reply
    phase; any other value (``SelfAskAgent.SELF_ASK_PROMPT_NAME``) means a
    thinking round is still active. This is why the field is required (no
    default) on the base ``AgentTask`` — every concrete subclass must
    decide it explicitly, and here that decision *is* the phase.

    No retry-budget field: the free-flowing category-marker parser
    (``parse_thoughts``) degrades unmarked text to a single ``OTHER``
    thought rather than failing, so a thinking round either produces at
    least one thought or ``think()`` raises outright — there is no
    malformed-output case to retry.

    Fields
    ------
    thoughts : list[list[AgentThought]]
        Task-local accumulator, one inner list per completed round (a round
        may produce more than one thought, up to ``thoughts_per_round``).
        Mirrors ``ToolAgentTask.running_blackboard`` — merged into the
        agent-level persisted ``self._thoughts`` only at
        ``_build_record_from_task`` time, never appended to the agent-level
        list mid-run. Its own length doubles as the completed-round count —
        no separate counter field is kept.
    """
    thoughts: list[list[AgentThought]] = field(default_factory=list)
