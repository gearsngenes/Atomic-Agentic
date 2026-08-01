from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["AgentThought"]


@dataclass(slots=True)
class AgentThought:
    """
    One categorized thought produced during a ``SelfAskAgent`` thinking
    round.

    Parsed from free-flowing ``[CATEGORY] content`` marked text
    (``parse_thoughts``, ``utils/agents.py``) rather than a structured JSON
    schema — a round's raw output degrades to a single ``OTHER``-category
    thought when no marker is present, so parsing itself never fails; only
    genuinely empty output does.

    Fields
    ------
    category : str
        One of ``constants.agents.THOUGHT_CATEGORIES`` (uppercased at parse
        time). Not validated against that set here — ``parse_thoughts``'s
        own regex only ever matches a category listed there, so an
        out-of-set value can't reach this constructor through the normal
        path.
    content : str
        The thought's text, stripped of surrounding whitespace.
    """

    category: str
    content: str

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable representation of this thought."""
        return {
            "category": self.category,
            "content": self.content,
        }
