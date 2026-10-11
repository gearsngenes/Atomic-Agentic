from __future__ import annotations

from dataclasses import dataclass, field

from .records import AgentRecord

__all__ = ["Conversation"]


@dataclass(frozen=True, slots=True)
class Conversation:
    """
    One named conversation's forward-ordered record history.

    Plain data container, no Agent-registry awareness -- mirrors
    AgentRecord's own stance (a dataclass with no owning-Agent awareness).
    ``records`` is a mutable list despite the frozen dataclass -- same
    treatment as ``AgentRecord.children``/``.inputs``: identity (``name``)
    is frozen; the list's own contents are appended to in place by
    ``Agent._commit_emit`` and read via direct indexing/slicing elsewhere.

    Fields
    ------
    name:
        This conversation's key, as stored in ``Agent._conversations``. Not
        itself pattern-validated here (alphanumeric/underscore,
        no-trailing-``_digits`` shape) -- that's a user-facing-boundary
        concern owned by ``Agent.create_conversation``/``clone_conversation``,
        not this passive container's constructor.
    records:
        Forward-ordered (oldest-first) list of ``AgentRecord``. A single
        ``AgentRecord`` may legitimately appear in more than one
        ``Conversation``'s list (cloning is a reference-copy, not a deep
        copy).
    """

    name: str
    records: list[AgentRecord] = field(default_factory=list, hash=False, compare=False)

    def __post_init__(self) -> None:
        # Shape-only validation -- element-type checking of each record is
        # AgentRecord's own construction's job, not this container's.
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(
                f"Conversation.name must be a non-empty str, got {self.name!r}."
            )
        if not isinstance(self.records, list):
            raise ValueError(
                f"Conversation.records must be a list, got {type(self.records)!r}."
            )
