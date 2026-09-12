from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

from ...constants.agents import ATTR_CALL_ALIAS, PY_BUILTIN_ALIAS, RHS_ASSIGN_ALIAS
from ..results.agents import AgentResult
from ..results.llm import LLMResult
from ...utils.script import is_dispatched_slot, render_completed_as_python
from .blackboard_models import CodeStatement

__all__ = [
    "LLMRecord",
    "AgentRecord",
    "ToolAgentRecord",
    "ScriptAgentRecord",
    "ScriptAgentToolUsage",
    "ThinkingAgentRecord",
]


@dataclass(frozen=True, slots=True)
class LLMRecord:
    """
    Canonical memory record for one completed LLM generation made during an
    Agent invocation.

    An Agent invocation may involve one or more LLM generations (e.g. a
    ToolAgent's planning loop). Each generation is preserved here so that
    future rendering, debugging, and accounting are not constrained by what
    an earlier pass chose to keep.

    Fields
    ------
    messages:
        The messages appended on top of the rendered conversation history
        immediately before this LLM call — the delta that is new for this
        specific generation. The system message and rendered prior turns are
        excluded; they are already captured by the enclosing AgentRecord /
        ToolAgentRecord.

        For base Agent: a one-element tuple containing the current user
        prompt message. For PlanActAgent: the same — one new user message.
        For ReActAgent: a three-element tuple — the original user task
        (user), the running-plan snapshot (assistant), and the step-request
        stub (user).

    llm_result:
        The complete LLMResult produced by this generation, including token
        usage, model identity, timing, and run identity.

    system_prompt_name:
        Key in the producing agent's ``system_prompts`` dict used to generate
        the system prompt for this LLM call. ``None`` for static or legacy
        callers that do not participate in the ``system_prompts`` API.
    """

    messages: tuple[dict[str, str], ...]
    llm_result: LLMResult
    system_prompt_name: str | None = None

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
        if self.system_prompt_name is not None and not isinstance(self.system_prompt_name, str):
            raise TypeError(
                "LLMRecord.system_prompt_name must be a str or None, "
                f"got {type(self.system_prompt_name).__name__}."
            )

    def to_dict(self) -> dict[str, Any]:
        """Return the explicit serialized dictionary representation."""
        return {
            "messages": list(self.messages),
            "llm_result": self.llm_result.to_dict(),
            "system_prompt_name": self.system_prompt_name,
        }


@dataclass(frozen=True, slots=True)
class AgentRecord:
    """
    Canonical memory record for one completed Agent invocation.

    A record stores the lifecycle artifacts needed to reconstruct future
    LLM-facing context. It is related to, but distinct from, AgentResult:
    AgentResult is the public successful-invocation envelope carrying LLM
    accounting; AgentRecord is the memory/rendering record. The completed
    record points to its AgentResult via ``final_result``.

    Records form a doubly-linked tree via ``prev``/``children``: each
    committed record points backward to the most recent record that was used
    as context when it was created (``prev``), and that target record points
    forward to every record that was ever committed on top of it
    (``children``). ``prev=None`` marks a chain root (first invocation or a
    fresh start). Walking ``prev`` backward from any record reconstructs the
    exact conversation branch that produced it; a target with more than one
    child marks a fork point where more than one conversation continued from
    the same record.

    Fields
    ------
    user_prompt:
        The fully-resolved prompt string produced by ``pre_invoke`` for this
        invocation. Already rendered — ``render_turn`` uses it verbatim, with
        no further templating or context lookup.

    inputs:
        The full filtered top-level inputs mapping for this invocation (post
        ``filter_inputs``, pre pre/post-invoke slicing) — the same dict the
        task's ``_initialize_task`` received. Retained for
        provenance/observability only; not consumed when reconstructing this
        turn's rendered content (``render_turn`` uses ``user_prompt``
        verbatim).

    generated_response:
        Raw post-engine response material for this invocation, prior to
        ``post_invoke`` processing.

    final_result:
        The completed ``AgentResult`` for this invocation. ``None`` during
        the draft phase (between ``_build_record_from_task`` and
        ``build_result_from_record`` completion); an ``AgentResult``
        instance on all stored records.

    llm_records:
        Complete record of every LLM generation that contributed to this
        invocation. Empty tuple during the draft phase; populated by
        ``_build_record_from_task`` from the task's accumulated
        ``llm_records``.

    prev:
        The most recent ``AgentRecord`` that was used as context for this
        invocation, or ``None`` if no prior context was used. Always points
        to a completed (non-draft) record on any record committed to history.

    children:
        Every ``AgentRecord`` (across every conversation) that named this
        record as its ``prev`` — the forward-pointing counterpart to
        ``prev``, making the overall structure a doubly-linked tree rather
        than a singly-linked chain. Mutable-in-place despite this being a
        frozen dataclass — ``hash=False, compare=False`` mirrors ``inputs``'s
        existing treatment, so appending to it post-construction doesn't
        disturb hashing/equality. Never reassigned after construction, only
        ever mutated via ``.append(...)``; expected to always start empty at
        construction time (a record cannot have children before it exists).
    """

    user_prompt: str
    generated_response: Any
    inputs: dict = field(default_factory=dict, hash=False, compare=False)
    final_result: AgentResult | None = None
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
        normalized = tuple(self.llm_records)
        for index, record in enumerate(normalized):
            if not isinstance(record, LLMRecord):
                raise TypeError(
                    "AgentRecord.llm_records must contain only LLMRecord instances; "
                    f"item {index} is {type(record).__name__}."
                )
        object.__setattr__(self, "llm_records", normalized)

        if self.prev is not None:
            if not isinstance(self.prev, AgentRecord):
                raise TypeError(
                    f"AgentRecord.prev must be an AgentRecord or None, "
                    f"got {type(self.prev).__name__}."
                )
            if self.prev.final_result is None:
                raise ValueError(
                    "AgentRecord.prev must point to a completed record "
                    "(final_result is not None); cannot link to a draft."
                )

        if not isinstance(self.children, (list, tuple)) or isinstance(self.children, (str, bytes)):
            raise TypeError(
                "AgentRecord.children must be a list of AgentRecord instances, "
                f"got {type(self.children).__name__}."
            )
        for index, child in enumerate(self.children):
            if not isinstance(child, AgentRecord):
                raise TypeError(
                    "AgentRecord.children must contain only AgentRecord instances; "
                    f"item {index} is {type(child).__name__}."
                )
        # Deliberately NOT normalized to tuple -- children is the one field
        # exempt from this class's "normalize sequences to tuple" convention,
        # exactly like the existing inputs dict exemption. It must stay a
        # mutable list so later commits can append to it in place.

    def to_dict(self) -> Dict[str, Any]:
        """Return the explicit serialized dictionary representation."""
        return {
            "user_prompt": self.user_prompt,
            "inputs": self.inputs,
            "generated_response": self.generated_response,
            "final_result": self.final_result.to_dict() if self.final_result is not None else None,
            "llm_records": [r.to_dict() for r in self.llm_records],
            "prev_run_id": self.prev.final_result.run_id if self.prev is not None else None,
            "child_ids": [c.final_result.run_id for c in self.children],
        }


@dataclass(frozen=True, slots=True)
class ToolAgentRecord(AgentRecord):
    """
    Canonical memory record for one completed ToolAgent invocation.

    In addition to the base AgentRecord lifecycle artifacts, a ToolAgentRecord
    stores the half-open span of persisted blackboard entries produced by
    the invocation. The ToolAgent renders that span into future LLM-facing
    context when building messages.
    """

    blackboard_start: int | None = None
    blackboard_end: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the explicit serialized dictionary representation."""
        return {
            **super(ToolAgentRecord, self).to_dict(),
            "blackboard_start": self.blackboard_start,
            "blackboard_end": self.blackboard_end,
        }


@dataclass(frozen=True, slots=True)
class ScriptAgentToolUsage:
    """
    Five orthogonal counts derived from one completed ``ScriptAgentRecord``'s
    ``statements``, for debugging/observability only -- never used to
    reconstruct or re-plan. ``registered_tool_calls`` + ``builtin_calls`` +
    ``attribute_calls`` is the same total already enforced (silently)
    against ``tool_calls_limit``; this just slices it three ways instead of
    one. ``binop_count`` is a subset of ``rhs_assignment_count``, not
    double-counted against the other three.

    Fields
    ------
    registered_tool_calls : int
        Dispatched slots that are neither a builtin nor an attribute/method
        call -- a real registered-tool invocation.

    builtin_calls : int
        Slots dispatched through the approved-Python-builtin path
        (``PY_BUILTIN_ALIAS``).

    attribute_calls : int
        Slots dispatched through the attribute/method-call path
        (``ATTR_CALL_ALIAS``).

    binop_count : int
        Bare-expression (``RHS_ASSIGN_ALIAS``) slots whose stored value is
        still an ``ast.expr`` containing an ``ast.BinOp`` anywhere --
        constant expressions are never folded before commit (only
        dry-run-validated at parse time), so this is a plain post-hoc walk,
        not a count tracked separately during parsing.

    rhs_assignment_count : int
        Total ``RHS_ASSIGN_ALIAS`` slot count, independent of whether it
        contains a binop.
    """

    registered_tool_calls: int
    builtin_calls: int
    attribute_calls: int
    binop_count: int
    rhs_assignment_count: int

    def to_dict(self) -> dict[str, int]:
        """Return the explicit serialized dictionary representation."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ScriptAgentRecord(AgentRecord):
    """
    Canonical memory record for one completed ScriptAgent invocation -- a
    sibling to ToolAgentRecord, not a subclass (ScriptAgent is a new agent
    family, not a ToolAgent subclass).

    Unlike the Task family, Record types in this codebase validate at
    construction (AgentRecord.__post_init__ already checks user_prompt/
    llm_records/prev) -- a persisted/serialized/rendered record sits closer
    to a real boundary than an in-flight task does. This class's own
    __post_init__ follows that precedent for its own new fields.

    No more agent-level global blackboard for this family -- each record
    owns its own slots outright. There is no blackboard_start/
    blackboard_end span to index into, unlike ToolAgentRecord (v1).

    Fields
    ------
    statements : tuple[CodeStatement, ...]
        Every slot ScriptAgentTask.completed accumulated this run, carried
        over at commit time (normalized to a tuple here, mirroring
        llm_records' existing list-or-tuple-in, tuple-stored normalization).
        A bare reasoning string the model wrote is never a slot in its own
        right (parse_generation treats it as an inert, unstored no-op,
        legal anywhere in a generation) -- there is no separate annotation
        record of it.
    """

    statements: tuple[CodeStatement, ...] = ()

    def __post_init__(self) -> None:
        # Explicit two-argument super() -- @dataclass(slots=True) rebuilds
        # the class object to add __slots__, which invalidates the
        # zero-arg super()'s implicit __class__ closure cell (a documented
        # CPython gotcha for slotted-dataclass inheritance chains; confirmed
        # live: bare super() raises "obj must be an instance or subtype of
        # type" here).
        super(ScriptAgentRecord, self).__post_init__()

        # 1. statements must be a list/tuple of CodeStatement instances.
        if isinstance(self.statements, (str, bytes)) or not isinstance(self.statements, (list, tuple)):
            raise TypeError(
                "ScriptAgentRecord.statements must be a list or tuple of "
                f"CodeStatement instances; got {type(self.statements).__name__!r}."
            )
        for index, slot in enumerate(self.statements):
            if not isinstance(slot, CodeStatement):
                raise TypeError(
                    f"ScriptAgentRecord.statements[{index}] must be a "
                    f"CodeStatement instance; got {type(slot).__name__!r}."
                )

        # 2. normalize to a tuple -- object.__setattr__ required, the
        # dataclass is frozen (mirrors llm_records/inputs normalization
        # above).
        object.__setattr__(self, "statements", tuple(self.statements))

    def render_as_code(self) -> str:
        """
        Reconstruct this run's statements as source-formatted text, grouped
        by concurrent batch (``show_batches=True`` -- a standalone,
        human-only view; the live continuation-message path this mirrors,
        ``ScriptAgent._render_task_messages``, deliberately never shows
        batch grouping to the model itself). Named generically ("code",
        not "python") since the underlying grammar isn't guaranteed to
        stay Python-syntax-specific forever.
        """
        return render_completed_as_python(self.statements, show_batches=True)

    def tool_usage(self) -> ScriptAgentToolUsage:
        """
        Compute a ``ScriptAgentToolUsage`` snapshot from ``self.statements``
        in one pass. Pure/derived -- not stored, recomputed on each call.
        """
        registered_tool_calls = 0
        builtin_calls = 0
        attribute_calls = 0
        binop_count = 0
        rhs_assignment_count = 0

        for slot in self.statements:
            if slot.tool == PY_BUILTIN_ALIAS:
                builtin_calls += 1
            elif slot.tool == ATTR_CALL_ALIAS:
                attribute_calls += 1
            elif slot.tool == RHS_ASSIGN_ALIAS:
                rhs_assignment_count += 1
                value = slot.kwargs["val"]
                if isinstance(value, ast.expr) and any(
                    isinstance(node, ast.BinOp) for node in ast.walk(value)
                ):
                    binop_count += 1
            elif is_dispatched_slot(slot):
                registered_tool_calls += 1

        return ScriptAgentToolUsage(
            registered_tool_calls=registered_tool_calls,
            builtin_calls=builtin_calls,
            attribute_calls=attribute_calls,
            binop_count=binop_count,
            rhs_assignment_count=rhs_assignment_count,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the explicit serialized dictionary representation."""
        # Explicit two-argument super() -- same slotted-dataclass gotcha
        # __post_init__ documents above; bare super() raises here too.
        d = super(ScriptAgentRecord, self).to_dict()
        d.update({
            "statements": [s.to_dict() for s in self.statements],
        })
        return d

@dataclass(frozen=True, slots=True)
class ThinkingAgentRecord(AgentRecord):
    """
    Canonical memory record for one completed thinking-capable agent
    invocation (currently only ``SelfAskAgent``).

    In addition to the base AgentRecord lifecycle artifacts, a
    ThinkingAgentRecord stores the half-open span of persisted thoughts
    produced by the invocation. ``SelfAskAgent`` renders that span into
    future LLM-facing context when building messages.
    """

    thoughts_start: int | None = None
    thoughts_end: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the explicit serialized dictionary representation."""
        return {
            **super(ThinkingAgentRecord, self).to_dict(),
            "thoughts_start": self.thoughts_start,
            "thoughts_end": self.thoughts_end,
        }
