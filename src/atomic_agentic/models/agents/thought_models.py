from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["AgentThought"]


@dataclass(slots=True)
class AgentThought:
    """
    One self-questioning thinking step produced by a ``ThinkingAgent``.

    Always constructed already-answered, in one validated round — unlike
    ``BlackboardSlot`` there is no in-progress/planned status to represent,
    since retries happen before a thought is ever appended (never after).

    Fields
    ------
    observation : str | None
        Optional context that prompted the question. ``None`` when the
        producing subclass's schema doesn't distinguish an observation from
        the question itself.
    question : str
        The self-asked question this thought answers.
    answer : str
        The answer produced for ``question``.
    """

    observation: str | None
    question: str
    answer: str

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable representation of this thought."""
        return {
            "observation": self.observation,
            "question": self.question,
            "answer": self.answer,
        }
