from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

from ..results.agents import AgentResult
from ..results.llm import LLMResult

__all__ = [
    "LLMRecord",
    "AgentRecord",
    "ToolAgentRecord",
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

    Records form a singly-linked list via ``prev``: each committed record
    points to the most recent record that was used as context when it was
    created. ``prev=None`` marks a chain root (first invocation or a fresh
    start). Walking ``prev`` backward
    from any record reconstructs the exact conversation branch that produced
    it.

    Fields
    ------
    user_prompt:
        The fully-resolved prompt string produced by ``pre_invoke`` for this
        invocation. Already rendered — ``render_turn`` uses it verbatim, with
        no further templating or context lookup.

    inputs:
        The full filtered top-level inputs mapping for this invocation (post
        ``filter_inputs``, pre pre/post-invoke slicing) — the same dict passed
        to ``_invoke``/``_ainvoke``. Retained for provenance/observability only;
        not consumed when reconstructing this turn's rendered content
        (``render_turn`` uses ``user_prompt`` verbatim).

    generated_response:
        Raw post-engine response material for this invocation, prior to
        ``post_invoke`` processing.

    final_result:
        The completed ``AgentResult`` for this invocation. ``None`` during
        the draft phase (between ``_invoke`` return and ``make_result``
        completion); an ``AgentResult`` instance on all stored records.

    llm_records:
        Complete record of every LLM generation that contributed to this
        invocation. Empty tuple during the draft phase; populated in the
        completion ``replace`` step after ``make_result`` runs.

    prev:
        The most recent ``AgentRecord`` that was used as context for this
        invocation, or ``None`` if no prior context was used. Always points
        to a completed (non-draft) record on any record committed to history.
    """

    user_prompt: str
    generated_response: Any
    inputs: dict = field(default_factory=dict, hash=False, compare=False)
    final_result: AgentResult | None = None
    llm_records: tuple[LLMRecord, ...] = ()
    prev: AgentRecord | None = None

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

    def to_dict(self) -> Dict[str, Any]:
        """Return the explicit serialized dictionary representation."""
        return {
            "user_prompt": self.user_prompt,
            "inputs": self.inputs,
            "generated_response": self.generated_response,
            "final_result": self.final_result.to_dict() if self.final_result is not None else None,
            "llm_records": [r.to_dict() for r in self.llm_records],
            "prev_run_id": self.prev.final_result.run_id if self.prev is not None else None,
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
