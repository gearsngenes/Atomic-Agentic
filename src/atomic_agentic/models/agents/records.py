from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

from ..results.agents import ToolUsageReport
from ..results.llm import LLMResult
from .blackboard_models import Statement

__all__ = [
    "LLMRecord",
    "AgentRecord",
    "ToolAgentRecord",
    "PlanActAgentRecord",
    "ThinkingAgentRecord",
]


@dataclass(frozen=True, slots=True)
class LLMRecord:
    """
    Canonical memory record for one completed LLM generation made during an
    Agent invocation.

    An Agent invocation may involve one or more LLM generations (e.g. a
    ``PlanActAgent``'s planning loop). Each generation is preserved here so
    that rendering, debugging, and accounting aren't constrained by what's
    kept.

    Fields
    ------
    messages:
        The messages appended on top of the rendered conversation history
        immediately before this LLM call — the delta that is new for this
        specific generation. The system message and rendered prior turns are
        excluded; they are already captured by the enclosing AgentRecord /
        ToolAgentRecord.

        For base Agent: a one-element tuple containing the current user
        prompt message. For the tool agents: the task frame, the STEPS
        frame (when there is work to show) and the directive.

    llm_result:
        The complete LLMResult produced by this generation, including token
        usage, model identity, timing, and run identity.
    """

    messages: tuple[dict[str, str], ...]
    llm_result: LLMResult

    def __post_init__(self) -> None:
        if isinstance(self.messages, (str, bytes)) or not isinstance(self.messages, (list, tuple)):
            raise TypeError(
                "LLMRecord.messages must be a list or tuple of dict[str, str]; "
                f"got {type(self.messages).__name__}."
            )
        normalized = tuple(self.messages)
        if not normalized:
            raise ValueError("LLMRecord.messages must be non-empty.")
        for index, msg in enumerate(normalized):
            if not isinstance(msg, dict):
                raise TypeError(
                    f"LLMRecord.messages[{index}] must be a dict; got {type(msg).__name__}."
                )
            if not msg:
                raise ValueError(f"LLMRecord.messages[{index}] must not be empty.")
            for key, value in msg.items():
                if not isinstance(key, str):
                    raise TypeError(
                        f"LLMRecord.messages[{index}] has a non-string key: {key!r}."
                    )
                if not isinstance(value, str):
                    raise TypeError(
                        f"LLMRecord.messages[{index}] key {key!r} has a non-string value: "
                        f"{type(value).__name__}."
                    )
        object.__setattr__(self, "messages", normalized)
        if not isinstance(self.llm_result, LLMResult):
            raise TypeError(
                "LLMRecord.llm_result must be an LLMResult instance, "
                f"got {type(self.llm_result).__name__}."
            )

    def to_dict(self) -> dict[str, Any]:
        """Return the explicit serialized dictionary representation."""
        return {
            "messages": list(self.messages),
            "llm_result": self.llm_result.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class AgentRecord:
    """
    Canonical memory record for one completed Agent invocation.

    A record stores the lifecycle artifacts needed to reconstruct future
    LLM-facing context. It carries its own identity (``run_id``) directly --
    no attached ``AgentResult``.
    A record's processed, post-``post_invoke`` output is deliberately NOT
    recoverable from history: only ``run_id`` and the raw
    ``generated_response`` persist. A caller wanting
    the processed result must capture it from ``invoke()``'s live return
    value at call time.

    Records are linked in both directions via ``prev``/``children``: each
    committed record points backward to the most recent record that was used
    as context when it was created (``prev``), and that target record points
    forward to every record that was ever committed on top of it
    (``children``). ``prev=None`` marks a chain root (first invocation or a
    fresh start). Walking ``prev`` backward from any record reconstructs the
    exact conversation branch that produced it; a target with more than one
    child marks a branch point where more than one conversation (or racing
    call) continued from the same record. Under a shared conversation a
    record may gain sibling ``children`` from different agents, while the
    conversation's list order stays commit order. Provenance only: turn
    selection never reads it.

    Fields
    ------
    user_prompt:
        The fully-resolved prompt string produced by ``pre_invoke`` for this
        invocation. Already rendered — ``render_turn`` uses it verbatim, with
        no further templating or context lookup.

    inputs:
        The full filtered top-level inputs mapping for this invocation (post
        ``filter_inputs``, pre pre/post-invoke slicing) — the same dict the
        task's ``create_task`` pinned. Retained for
        provenance/observability only; not consumed when reconstructing this
        turn's rendered content (``render_turn`` uses ``user_prompt``
        verbatim).

    generated_response:
        Raw post-engine response material for this invocation, prior to
        ``post_invoke`` processing.

    run_id:
        This record's identity; required. Minted by ``Agent.create_task``
        (records exist only when the agent records).

    agent_name:
        The writing agent's ``full_name``; ``None`` for bare construction.
        Provenance only -- history replay does not distinguish writers.

    llm_records:
        Complete record of every LLM generation that contributed to this
        invocation, populated by ``_build_record_from_task`` from the
        task's accumulated ``llm_records``.

    prev:
        The most recent ``AgentRecord`` that was used as context for this
        invocation, or ``None`` if no prior context was used.

    children:
        Every ``AgentRecord`` (across every conversation) that named this
        record as its ``prev`` — the forward-pointing counterpart to
        ``prev``, making the overall structure a doubly-linked tree rather
        than a singly-linked chain. Mutable-in-place despite this being a
        frozen dataclass — ``hash=False, compare=False`` mirrors ``inputs``'s
        existing treatment, so appending to it post-construction doesn't
        disturb hashing/equality. Deliberately a list, not a tuple: later
        commits append to it in place. Never reassigned after construction;
        expected to always start empty (a record cannot have children before
        it exists).
    """

    user_prompt: str
    generated_response: Any
    run_id: str = field(kw_only=True)
    agent_name: str | None = field(default=None, kw_only=True)
    inputs: dict = field(default_factory=dict, hash=False, compare=False)
    llm_records: tuple[LLMRecord, ...] = ()
    prev: AgentRecord | None = None
    children: list["AgentRecord"] = field(default_factory=list, hash=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.user_prompt, str):
            raise TypeError(
                f"AgentRecord.user_prompt must be a str, "
                f"got {type(self.user_prompt).__name__}."
            )
        object.__setattr__(self, "inputs", dict(self.inputs))

        if not isinstance(self.llm_records, (tuple, list)) or isinstance(self.llm_records, (str, bytes)):
            raise TypeError(
                "AgentRecord.llm_records must be a sequence of LLMRecord instances, "
                f"got {type(self.llm_records).__name__}."
            )
        object.__setattr__(self, "llm_records", tuple(self.llm_records))

    def to_dict(self) -> Dict[str, Any]:
        """Return the explicit serialized dictionary representation."""
        return {
            "user_prompt": self.user_prompt,
            "inputs": self.inputs,
            "generated_response": self.generated_response,
            "run_id": self.run_id,
            "agent_name": self.agent_name,
            "llm_records": [r.to_dict() for r in self.llm_records],
            "prev_run_id": self.prev.run_id if self.prev is not None else None,
            "child_ids": [c.run_id for c in self.children],
        }


@dataclass(frozen=True, slots=True)
class ToolAgentRecord(AgentRecord):
    """
    Canonical memory record for one completed ToolAgent invocation.

    ``statements``/``failed_statements`` as ``Statement`` tuples, no
    blackboard span of any kind. Shared directly by ``ReActAgent``
    (no further subclass needed -- it would add zero fields beyond what's
    here); ``PlanActAgentRecord`` is a real subclass of this class.

    Fields
    ------
    statements : tuple[Statement, ...]
        Every statement the task's ``completed`` accumulated this run
        (status ``done``, ``return`` included once it ran).

    failed_statements : tuple[Statement, ...]
        Every statement that failed this run (status ``failed``), with the
        steps that ran and the failing step's error. Statements that never ran
        are not recorded.

    regenerations_used : int
        Total regeneration attempts consumed across this run's generation
        call(s) -- carried over from ``task.regenerations_used`` verbatim
        at commit time.
    """

    statements: tuple[Statement, ...] = ()
    failed_statements: tuple[Statement, ...] = ()
    regenerations_used: int = 0

    def __post_init__(self) -> None:
        # Explicit two-argument super() -- @dataclass(slots=True) rebuilds
        # the class object to add __slots__, which invalidates the
        # zero-arg super()'s implicit __class__ closure cell (a documented
        # CPython gotcha for slotted-dataclass inheritance chains).
        super(ToolAgentRecord, self).__post_init__()

        for label in ("statements", "failed_statements"):
            value = getattr(self, label)
            if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
                raise TypeError(
                    f"ToolAgentRecord.{label} must be a list or tuple of "
                    f"Statement instances; got {type(value).__name__!r}."
                )
            for index, statement in enumerate(value):
                if not isinstance(statement, Statement):
                    raise TypeError(
                        f"ToolAgentRecord.{label}[{index}] must be a "
                        f"Statement instance; got {type(statement).__name__!r}."
                    )
            object.__setattr__(self, label, tuple(value))

    def to_dict(self) -> dict[str, Any]:
        """Return the explicit serialized dictionary representation."""
        d = super(ToolAgentRecord, self).to_dict()
        d.update({
            "statements": [s.to_dict() for s in self.statements],
            "failed_statements": [s.to_dict() for s in self.failed_statements],
            "regenerations_used": self.regenerations_used,
        })
        return d

    def render_as_code(self) -> str:
        """
        Reconstruct this run's statements as source-formatted text, in source
        order (done and failed statements merged by ordinal; a failed one
        carries its failure note). A standalone, human-only view. Named
        generically ("code", not "python") since the underlying grammar
        isn't guaranteed to stay Python-syntax-specific forever.
        """
        from ...utils.statements import render_statements

        ordered = sorted((*self.statements, *self.failed_statements), key=lambda st: st.ordinal)
        return render_statements(ordered)

    def usage_report(self) -> ToolUsageReport:
        """
        Per-tool call-count accounting for this run -- one ``ToolUsageRecord``
        per distinct real tool identity among the tool steps that ran or
        failed (``build_usage_report``), plus ``total_dispatched`` and
        ``total_failed`` counts, bundled into a ``ToolUsageReport``.
        """
        from ...utils.agents import build_usage_report

        return build_usage_report((*self.statements, *self.failed_statements))


@dataclass(frozen=True, slots=True)
class PlanActAgentRecord(ToolAgentRecord):
    """
    Canonical memory record for one completed PlanActAgent invocation, a
    ``ToolAgentRecord`` subclass. ``statements``/``failed_statements``/
    ``regenerations_used``/``render_as_code()`` are all inherited; only
    ``repair_rounds_used`` remains genuinely ``PlanActAgent``-specific.

    No agent-level global blackboard for this family -- each record owns
    its own statements outright, with no blackboard_start/blackboard_end span
    to index into.

    Fields
    ------
    repair_rounds_used : int
        Total framework-granted repair rounds actually consumed this run --
        carried over from ``task.repair_rounds_used`` verbatim at commit
        time. ``0`` means the plan finished without ever needing one.
    """

    repair_rounds_used: int = 0

    def __post_init__(self) -> None:
        # Explicit two-argument super() -- @dataclass(slots=True) rebuilds
        # the class object to add __slots__, which invalidates the
        # zero-arg super()'s implicit __class__ closure cell (a documented
        # CPython gotcha for slotted-dataclass inheritance chains; confirmed
        # live: bare super() raises "obj must be an instance or subtype of
        # type" here). Targets ToolAgentRecord now -- it already validates/
        # normalizes statements/failed_statements, inherited unchanged.
        super(PlanActAgentRecord, self).__post_init__()

    def to_dict(self) -> dict[str, Any]:
        """Return the explicit serialized dictionary representation --
        extends the inherited ``ToolAgentRecord.to_dict()`` (already
        includes ``statements``/``failed_statements``/``regenerations_used``)
        with just ``repair_rounds_used``."""
        # Explicit two-argument super() -- same slotted-dataclass gotcha
        # __post_init__ documents above; bare super() raises here too.
        d = super(PlanActAgentRecord, self).to_dict()
        d["repair_rounds_used"] = self.repair_rounds_used
        return d


@dataclass(frozen=True, slots=True)
class ThinkingAgentRecord(AgentRecord):
    """
    Canonical memory record for one completed thinking-agent invocation
    (any ``ThinkingAgent`` subclass).

    In addition to the base AgentRecord lifecycle artifacts, a
    ThinkingAgentRecord stores the full thought content produced by the
    invocation directly -- one value per thought, in order. Mirrors
    ``AgentRecord.llm_records``'s own pattern (the full artifact lives on the
    record, not a span into some agent-level accumulator). The tuple is
    shallow: the thought values are shared with the task, not copied.
    """

    thoughts: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        # Explicit two-argument super() -- @dataclass(slots=True) rebuilds
        # the class object to add __slots__, which invalidates the
        # zero-arg super()'s implicit __class__ closure cell (same
        # slotted-dataclass-subclass gotcha PlanActAgentRecord.__post_init__
        # already documents above).
        super(ThinkingAgentRecord, self).__post_init__()

        if isinstance(self.thoughts, (str, bytes)) or not isinstance(self.thoughts, (list, tuple)):
            raise TypeError(
                "ThinkingAgentRecord.thoughts must be a list or tuple, "
                f"got {type(self.thoughts).__name__}."
            )
        object.__setattr__(self, "thoughts", tuple(self.thoughts))

    def to_dict(self) -> dict[str, Any]:
        """Return the explicit serialized dictionary representation."""
        return {
            **super(ThinkingAgentRecord, self).to_dict(),
            "thoughts": list(self.thoughts),
        }
