from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["AgentThought2"]


@dataclass(slots=True)
class AgentThought2:
    """
    One categorized line of thinking-phase output produced by ``Agent2``.

    Parallel-copy counterpart to ``AgentThought`` (``thought_models.py``) --
    a new file/name rather than an in-place shape change, since the old
    ``{observation, question, answer}`` dataclass is still actively
    constructed by name in the shipped ``ThinkingAgent``/``SelfAskAgent``/
    ``PlanAskAgent`` family. Renamed back to ``AgentThought`` at cutover.

    No ``__post_init__`` validation -- constructed exclusively by
    ``utils.agents.parse_thoughts``, which is responsible for producing
    well-formed instances.

    Fields
    ------
    category : str
        One of ``constants.agents.THOUGHT_CATEGORIES``, always upper-cased
        by the parser.
    content : str
        Free text following the category marker, stripped.
    """

    category: str
    content: str

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable representation of this thought."""
        return {
            "category": self.category,
            "content": self.content,
        }
