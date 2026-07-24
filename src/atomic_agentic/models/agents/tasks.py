from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ...constants.core import NO_VAL
from .blackboard_models import BlackboardSlot
from .records import AgentRecord, LLMRecord
from .runstates import ReActStepMeta  # reused as-is, not duplicated

__all__ = [
    "AgentTask",
    "ToolAgentTask",
    "PlanActTask",
    "ReActTask",
]


@dataclass(slots=True)
class AgentTask:
    """
    Base run-task contract for one Agent invocation.

    Replaces the role ``ToolAgentRunState`` used to play, generalized up to
    base ``Agent`` so any subclass — not just ``ToolAgent`` — can drive its
    invocation through ``_initialize_task``/``_progress`` rather than a
    single ``_invoke`` call. ``BasicAgent`` uses this base class directly;
    it needs nothing beyond these six fields.

    ``turns`` and ``user_prompt`` have no precedent on
    ``ToolAgentRunState`` — that family never stored either; both were bare
    ``_invoke`` parameters that ``invoke()`` re-injected into the draft
    record afterward via ``dataclasses.replace``. Storing them here is what
    lets ``_build_record_from_task`` assemble a complete record in one call.

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

    llm_records : list[LLMRecord]
        Accumulator for every LLM generation made while producing this
        invocation's result. Read exactly once, at loop-close, to populate
        the completed record's ``llm_records`` tuple.

    complete : bool
        Loop termination flag. The base lifecycle loop is
        ``while not task.complete: task = progress(task)``.

    generated_response : Any
        The record's produced-response equivalent — raw LLM text for
        ``BasicAgent``, the executed return-tool value for ``ToolAgent`` and
        its subclasses. ``NO_VAL`` until ``_progress`` sets it on the round
        that completes the task.
    """
    turns: list[AgentRecord]
    inputs: dict[str, Any]
    user_prompt: str

    llm_records: list[LLMRecord] = field(default_factory=list)
    complete: bool = False
    generated_response: Any = NO_VAL


@dataclass(slots=True)
class ToolAgentTask(AgentTask):
    """
    ToolAgent-flavored task, generalized from ``ToolAgentRunState``.

    Every field beyond the three inherited required fields carries a
    default — required by dataclass inheritance ordering (once any earlier
    field in the chain has a default, every later field must too), not a
    relaxation of ``ToolAgentRunState``'s current all-required shape for
    these particular fields: ``_initialize_task`` always populates all of
    them with real values.

    ``retries_used`` has no precedent at this level — today it lives only
    on ``ReActRunState`` (cumulative retry-attempt budget shared across all
    step generations in one run); ``PlanActAgent``'s retry budget is
    currently a local counter inside ``_generate_plan``, not stored on
    state at all. Promoting it here makes it available uniformly to every
    subclass, including ``PlanActTask``, which never carried an equivalent
    field on ``PlanActRunState``.

    See ``ToolAgentRunState`` (``runstates.py``) for the authoritative
    per-field behavioral description of every other field; this class is
    otherwise a structural rename/re-parent (plus the two new base fields on
    ``AgentTask`` above), not a redesign.
    """
    messages: list[dict[str, str]] = field(default_factory=list)

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
    PlanActAgent-flavored task. See ``PlanActRunState`` for the behavior of
    ``batches``/``batch_index``. Inherits ``retries_used`` from
    ``ToolAgentTask``, which has no ``PlanActRunState`` precedent — see
    that class's docstring.
    """
    batches: list[list[int]] = field(default_factory=list)
    batch_index: int = 0


@dataclass(slots=True)
class ReActTask(ToolAgentTask):
    """
    ReActAgent-flavored task. See ``ReActRunState`` for the behavior of
    ``next_step_index``/``step_meta``/``retries_used`` — the latter is
    declared on ``ToolAgentTask`` now (not here), but its behavioral
    description still originates from ``ReActRunState``.
    """
    next_step_index: int = 0
    step_meta: list[ReActStepMeta] = field(default_factory=list)
