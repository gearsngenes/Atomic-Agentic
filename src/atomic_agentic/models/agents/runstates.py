from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ...constants.core import NO_VAL
from .blackboard_models import BlackboardSlot
from .records import LLMRecord

__all__ = [
    "ToolAgentRunState",
    "PlanActRunState",
    "ReActRunState",
    "ReActStepMeta",
]


@dataclass(slots=True)
class ToolAgentRunState:
    """
    Base run state contract for ``ToolAgent`` invocations.

    This dataclass encapsulates all mutable state during a single ``invoke()`` call,
    enabling subclasses to extend it with domain-specific planning/execution fields.

    Blackboard Architecture
    -----------------------
    **cache_blackboard** : list[BlackboardSlot]
        Snapshot of the persisted ``self._blackboard`` at invoke start when context is
        enabled. Previous invocation results are available here and can be referenced
        via ``<<__cN__>>`` placeholders.

    **running_blackboard** : list[BlackboardSlot]
        Plan-local slots (0-based indices) created during this invoke. Populated by
        ``_initialize_run_state()``, planned by ``_prepare_next_batch()``, and
        executed by ``_execute_prepared_batch()``.
        If ``context_enabled=True``, executed slots are persisted and merged into
        ``self._blackboard`` after ``invoke()`` completes.

    Placeholder Semantics & Resolvability
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    **Cached Placeholder** (``<<__cN__>>``)
        Resolvable iff ``0 <= N < len(cache_blackboard)`` AND
        ``cache_blackboard[N].is_executed() == True``.

    **Step Placeholder** (``<<__sN__>>``)
        Resolvable iff ``0 <= N < len(running_blackboard)`` AND
        ``running_blackboard[N].is_executed() == True``.

    Execution State Machine
    -----------------------
    **messages** : list[dict[str, str]]
        Run-local LLM-facing messages used during this invocation. ReAct may augment
        this list with current-run observations. These messages are not the canonical
        stored memory format; completed invocations are stored as turns.

    **executed_steps** : set[int]
        Running plan indices that have been executed.

    **prepared_steps** : list[int]
        Running plan indices ready for execution in the next batch. Must be set by
        ``_prepare_next_batch()`` and consumed by the base execution path.

    **tool_calls_used** : int
        Count of non-return tool calls executed so far.

    **is_done** : bool
        Loop termination flag. Set to True when the ``return`` tool executes.

    **return_value** : Any | NO_VAL
        Agent's raw final output. Set when the return tool executes. This becomes the
        raw response passed into the base Agent post-processing and turn pipeline.

    **llm_records** : list[LLMRecord]
        Accumulator for every LLM generation made while producing this invocation's
        result. Seeded at construction time — non-empty for subclasses that generate
        up front (e.g. PlanAct's one-shot plan), empty for subclasses that generate
        lazily during the loop (e.g. ReAct's per-step planning) — and appended to as
        further generations occur. Read exactly once, at loop-close, to populate the
        draft ``ToolAgentRecord``'s ``llm_records`` tuple.
    """
    messages: list[dict[str, str]]

    cache_blackboard: list[BlackboardSlot]
    running_blackboard: list[BlackboardSlot]

    # Plan-local indices whose results are available (optional fast-path bookkeeping).
    executed_steps: set[int] = field(default_factory=set)

    # Exact plan-local indices to execute next; must be set by _prepare_next_batch.
    prepared_steps: list[int] = field(default_factory=list)

    # Bookkeeping:
    tool_calls_used: int = 0  # non-return calls executed in this run
    llm_records: list[LLMRecord] = field(default_factory=list)

    # Completion:
    is_done: bool = False
    return_value: Any = NO_VAL  # NO_VAL by default; set once return tool executes


@dataclass(slots=True)
class PlanActRunState(ToolAgentRunState):
    """
    PlanActAgent-specific run state extending ToolAgentRunState.

    Adds planning-specific fields for managing pre-compiled concurrent batches.

    Fields
    ------
    batches : list[list[int]]
        Pre-compiled topologically-sorted batches. Each batch is a list of plan-local
        indices that can execute concurrently. Created during ``_initialize_run_state()``
        via ``_compile_batches_from_deps()``.

        Example: ``[[0, 1], [2, 3], [4]]`` means:
        - Batch 0: steps 0 and 1 execute together
        - Batch 1: steps 2 and 3 execute together (after batch 0)
        - Batch 2: step 4 executes (after batch 1; typically the return step)

    batch_index : int
        Cursor pointing to the next batch to prepare. Starts at 0 during initialization;
        incremented after each batch is prepared. Loop exits when batch_index >= len(batches).

    Workflow
    ~~~~~~~~
    1. ``_initialize_run_state()`` creates batches and sets batch_index=0
    2. Each iteration of the base loop:
       - ``_prepare_next_batch()`` reads batches[batch_index]
       - Resolves placeholders for that batch
       - Base ``_execute_prepared_batch()`` runs the batch concurrently
       - batch_index incremented for next iteration
    3. When batch_index >= len(batches), _prepare_next_batch() raises (loop exits)
    """
    batches: list[list[int]] = field(default_factory=list)
    batch_index: int = 0


@dataclass(slots=True)
class ReActStepMeta:
    """
    Per-slot metadata for a single ReAct step.

    Pairs the raw-result visibility counter with the one-sentence description
    for the same slot, replacing two parallel lists on ``ReActRunState``.

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
class ReActRunState(ToolAgentRunState):
    """
    ReActAgent-specific run state extending ToolAgentRunState.

    Tracks cursor and per-slot metadata for step-by-step reactive planning.

    Fields
    ------
    next_step_index : int
        Cursor for the next plan-local running_blackboard slot to fill. Starts at 0
        and increments after each step is prepared.

        Dual role:
        1. Allocation cursor: determines which slot index gets the next prepared step.
        2. Dependency cutoff: any <<__sN__>> placeholder in newly prepared args must
           satisfy N < next_step_index.

    step_meta : list[ReActStepMeta]
        Per-slot metadata for each slot in ``running_blackboard``. Always the same
        length as ``running_blackboard``. Initialized to
        ``[ReActStepMeta() for _ in running_blackboard]`` at construction.

        ``step_meta[i].observable`` — raw-result visibility counter for slot i.
        Zero means not visible; positive means the result is shown as
        ``observable_result`` for that many future successful generation turns.

        ``step_meta[i].description`` — one-sentence intent summary for slot i,
        rendered in the running-plan snapshot to preserve semantic context.

        Both fields are written by ``_prepare_next_batch`` /
        ``_aprepare_next_batch`` at the index of the slot being prepared.

    Workflow
    ~~~~~~~~
    Iteration k:

    1. prepare():
       - Build a fresh temporary copy of the static base messages.
       - Append one assistant message containing the current running-plan snapshot.
       - Append one small user request asking for the next single step.
       - Request the next step from the LLM.
       - Parse step object plus ReAct metadata:
         {"step": idx, "tool": ..., "args": {...}, "duration": ..., "description": ...}
       - Validate idx == next_step_index.
       - Fill running_blackboard[idx].
       - Write step_meta[idx].observable and step_meta[idx].description.
       - Set prepared_steps=[idx].
       - Increment next_step_index.

    2. execute():
       - Run the prepared single-step batch.
       - Store result in running_blackboard[idx].

    3. Continue until the return tool executes.
    """
    next_step_index: int = 0
    step_meta: list[ReActStepMeta] = field(default_factory=list)
