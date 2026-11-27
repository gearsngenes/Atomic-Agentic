from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from ...constants.core import NO_VAL
from .blackboard_models import Statement
from .context import ContextVariable
from .records import AgentRecord, LLMRecord

__all__ = [
    "AgentTask",
    "ToolAgentTask",
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

    thinking_complete : bool
        ``True`` when the task needs no further thinking before ``act``.
        Defaults to ``True`` (a single-phase agent has nothing to think
        about). ``ThinkingTask`` defaults it to ``False`` and
        ``ThinkingAgent.think`` is its only writer. Tool families do not use
        it.

    instructions : str
        The main instructions rendered once per invoke by the agent's
        ``_render_instructions`` hook (the last step of ``pre_process``).
        Empty until then, and empty for an agent with no instructions.
        The system message reads it; nothing else writes it.

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
        objects). ``Agent._write_context`` is its only writer: ``create_task``
        replaces or creates variables from the call's ``runtime_context`` dict
        (when the agent's ``use_runtime_context`` is on) and then from
        ``inputs_to_context``, and ``post_process`` replaces or creates
        variables from ``response_to_context``. Any other writer (subclass loop
        steps) must call ``_write_context``, or skip or raise on a ``readonly``
        variable itself. This dict is not guarded. Persistence beyond the call
        is ``record``'s: see ``changed_context``.

    changed_context : set[str]
        Names ``_write_context`` wrote during this call (created or replaced).
        ``record`` offers only these to the conversation, so a name the call
        never wrote is never persisted and never overwrites a newer value.
        Empty until a write happens.

    pre_inputs, post_inputs, agentic_inputs : dict[str, Any]
        Views of ``inputs`` sliced by consumer (pre-invoke, post-invoke,
        agent-specific -- the names the agent's parameter sources declare,
        i.e. the instructions). ``agentic_inputs`` is the union over all
        sources; each source takes only its own names from it. A name
        consumed by more than one goes into every
        bucket that declares it. ``post_process`` then overlays the response
        pieces bound by ``response_to_post`` onto ``post_inputs`` (always names
        the post stage declares), so afterwards it is exactly what
        ``post_invoke`` received.

    run_id : str | None
        Identity shared by the result and the record built from this task.
        ``None`` when the agent does not record (``record_enabled`` False).
        Set by ``create_task``.
    """
    turns: list[AgentRecord]
    inputs: dict[str, Any]

    user_prompt: str = ""
    llm_records: list[LLMRecord] = field(default_factory=list)
    complete: bool = False
    generated_response: Any = NO_VAL
    historic_messages: list[dict[str, str]] = field(default_factory=list)
    task_messages: list[dict[str, str]] = field(default_factory=list)

    conversation: str | None = None
    context: dict[str, ContextVariable] = field(default_factory=dict)
    changed_context: set[str] = field(default_factory=set)
    pre_inputs: dict[str, Any] = field(default_factory=dict)
    post_inputs: dict[str, Any] = field(default_factory=dict)
    agentic_inputs: dict[str, Any] = field(default_factory=dict)
    run_id: str | None = None
    thinking_complete: bool = True
    instructions: str = ""


@dataclass(slots=True)
class ToolAgentTask(AgentTask):
    """
    Base task shape shared by every concrete ``ToolAgent`` family
    (``PlanActAgent``/``ReActAgent``). Promotes every field genuinely
    shared by both concrete agents.

    Fields
    ------
    completed : list[Statement]
        Every statement that ran to the end this run (status ``done``), in
        commit order. Becomes the owning record's ``statements`` tuple
        verbatim at commit time.

    failed_statements : list[Statement]
        Every statement that failed this run (status ``failed``), in the order
        observed. A failed statement keeps the steps that ran and the failing
        step's ``error``. A statement that never ran is held by the family
        (``PlanActTask.pending``) and is never recorded.

    regenerations_used : int
        Cumulative regeneration attempts consumed across every generation
        call this run -- irreducible; a rejected/regenerated draft leaves
        no artifact anywhere else to derive this from (contrast
        ``tool_calls_used``, fully derivable from the statements).

    Result bindings and referenceable prior results live in ``task.context``
    (``task_result_i`` entries seeded by ``ToolAgent._seed_task``; run-local
    bindings written through ``Agent._write_context``), not on this task.
    """
    completed: list[Statement] = field(default_factory=list)
    failed_statements: list[Statement] = field(default_factory=list)
    regenerations_used: int = 0

    @property
    def tool_calls_used(self) -> int:
        """
        Actions spent this run, derived, not stored: ``actions_used`` over
        ``completed`` and ``failed_statements``. A statement costs
        ``max(1, tool steps that ran or failed)``, and a ``return`` is exempt
        from the minimum. This is what ``tool_calls_limit`` caps.
        """
        from ...utils.statements import actions_used
        return actions_used([*self.completed, *self.failed_statements])


@dataclass(slots=True)
class PlanActTask(ToolAgentTask):
    """
    PlanActAgent-flavored task, a ``ToolAgentTask`` subclass.
    ``completed``/``failed_statements``/``regenerations_used``/
    ``tool_calls_used`` are all inherited; only genuinely
    ``PlanActAgent``-specific fields remain declared here. Result
    bindings live in ``task.context`` as context variables, not on the task.

    No __post_init__ -- matches AgentTask's own family-wide convention of
    zero constructor-time validation (an in-flight, internal-only object,
    not a real boundary, per 01-overview.md Section 4).

    Fields
    ------
    pending : list[Statement]
        The statements compiled from the current generation that have not run
        yet -- the whole one-shot draft (or, after a granted repair, the
        whole freshly regenerated tail). Statements leave it as they run:
        done ones move to ``completed``, failed ones to ``failed_statements``.
        Whatever is left after a failure is reported as not run.

    needs_repair : bool
        Set only on a real resolution or execution failure that
        ``fail_fast``/``replanning_limit`` grants a repair round for -- there
        is no model-authored way to set this. Cleared by ``think()``/
        ``async_think()`` right after generating the granted repair round's
        fresh content; its only job is telling "stopped, needs a repair
        round" apart from "genuinely finished" in the empty-``pending`` checks.

    repair_rounds_used : int
        Count of framework-granted repair rounds so far this invoke --
        never counts the free initial plan. Checked against
        ``replanning_limit`` before granting another.

    repair_batch_start : int
        Index into ``failed_statements`` marking where the failures of the
        round that most recently triggered (or attempted to trigger) a repair
        began; the repair message renders ``failed_statements[repair_batch_start:]``
        -- exactly this round's fresh failures, not the accumulated history.
    """
    pending: list[Statement] = field(default_factory=list)
    needs_repair: bool = False
    repair_rounds_used: int = 0
    repair_batch_start: int = 0


@dataclass(slots=True)
class ReActTask(ToolAgentTask):
    """
    ReActAgent-flavored task: one Python statement generated, validated and
    run per round -- the per-step sibling to ``PlanActTask``'s one-shot plan.
    ``completed``/``failed_statements``/``tool_calls_used`` are inherited from
    ``ToolAgentTask``; only the fields below are ``ReActAgent``-specific.
    Result bindings live in ``task.context`` (written through
    ``Agent._write_context``).

    No fixed-size preallocated board and no observability-decay window --
    every round renders the sources of ``completed`` (plus the last failure)
    as the steps block, and the bound values through the context block. No
    ``__post_init__`` -- an in-flight, internal-only object, not a real
    construction-time boundary.

    Fields
    ------
    statement : Statement | None
        The statement this round compiled (``compile_statements(limit=1)``).
        ``think()`` sets it; ``act()`` runs it, files the result under
        ``completed`` or ``failed_statements`` and clears it. ``None``
        between rounds.

    failure_note : str
        The model-facing description of the last failed statement (the
        statement, the failing sub-step and the sub-steps that already ran);
        ``""`` when the last statement succeeded, so a non-empty note means
        the last statement failed.
    """
    statement: Optional[Statement] = None
    failure_note: str = ""


@dataclass(slots=True)
class ThinkingTask(AgentTask):
    """
    ThinkingAgent-family task.

    Thinking is in progress while ``thinking_complete`` is ``False`` (this task's
    default, unlike the base task's). The base ``ThinkingAgent.think`` sets it
    from its ``_thinking_done`` hook at the start of every pass; nothing else
    writes it. Once true, ``act`` makes the reply call.

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
    thinking_complete: bool = False
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

    thinking_instructions : str
        The thinking instructions rendered once per invoke by the agent's
        ``_render_instructions`` hook (the last step of ``pre_process``).
        Empty until then, and empty when ``thinking_rounds`` is 0 at
        pre-process time. The system message reads it; nothing else writes it.
    """
    more_thinking_required: bool = True
    thinking_instructions: str = ""
