from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ...constants.agents import RENDER_MODES
from ...constants.core import IDENTIFIER_PATTERN

__all__ = ["ContextVariable"]


@dataclass(frozen=True, slots=True)
class ContextVariable:
    """
    Named, mutable-by-replacement runtime value.

    Successor to the old ToolAgent-only, ``inline_limit``-bearing
    ``ConstantSpec`` -- base-``Agent``-scoped context/memory work builds on
    this. ``name`` is fixed for life; ``value``/``description`` are updated
    by constructing a new instance via ``dataclasses.replace`` (never
    in-place mutation, never a dedicated method on this class -- see
    ``type``'s own docstring for why no update methods live here).

    Fields
    ------
    name : str
        Safe identifier-like name. Must be non-empty, alphanumeric/
        underscore, not starting with a digit. Validated and normalized
        (stripped) in ``__post_init__``. Immutable after construction --
        direct assignment raises ``FrozenInstanceError``.

    value : Any
        Runtime value bound to this name. No type restriction.

    description : str | None
        Optional human-readable context. Normalized: blank/whitespace-only
        collapses to ``None`` in ``__post_init__``. Non-string, non-``None``
        raises ``TypeError``.

    readonly : bool
        Never overwritten: ``Conversation.remember``/``remember_many``/
        ``describe`` and a task's write-back all refuse or skip a readonly
        variable. It can still be forgotten. Must be a ``bool``
        (``TypeError`` otherwise).

    render_mode : str
        How ``value`` is formatted when rendered into a prompt: one of
        ``RENDER_MODES`` (``s`` str, ``r`` repr, ``a`` ascii, ``j`` json,
        ``p`` pformat). Defaults to ``"s"``; anything outside ``RENDER_MODES``
        (including ``None``) raises ``ValueError``. Kept by
        ``dataclasses.replace``.

    preview_limit : int | None
        Optional character ceiling for this variable's rendered value. ``None``
        defers to the conversation's context limit; otherwise the smaller of
        the two applies (a variable can only tighten the conversation's budget,
        never loosen it). ``0`` hides the value. Must be ``None`` or an
        ``int >= 0`` (``bool`` rejected), else ``ValueError``.
    """

    name: str
    value: Any
    description: str | None = None
    readonly: bool = False
    render_mode: str = "s"
    preview_limit: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("ContextVariable.name must be a non-empty string.")

        normalized_name = self.name.strip()
        if not IDENTIFIER_PATTERN.fullmatch(normalized_name):
            raise ValueError(
                "ContextVariable.name must be alphanumeric/underscore and not start "
                f"with a digit; got {self.name!r}."
            )

        normalized_description: str | None
        if self.description is None:
            normalized_description = None
        else:
            if not isinstance(self.description, str):
                raise TypeError(
                    "ContextVariable.description must be a string or None."
                )
            normalized_description = self.description.strip() or None

        if not isinstance(self.readonly, bool):
            raise TypeError("ContextVariable.readonly must be a bool.")

        if self.render_mode not in RENDER_MODES:
            raise ValueError(
                f"ContextVariable.render_mode must be one of {RENDER_MODES}, "
                f"got {self.render_mode!r}."
            )

        if self.preview_limit is not None and (
            type(self.preview_limit) is not int or self.preview_limit < 0
        ):
            raise ValueError(
                "ContextVariable.preview_limit must be None or an int >= 0, "
                f"got {self.preview_limit!r}."
            )

        object.__setattr__(self, "name", normalized_name)
        object.__setattr__(self, "description", normalized_description)

    @property
    def type(self) -> str:
        """
        Runtime type name of ``value``, computed fresh on every access --
        never stored. A ``dataclasses.replace(cv, value=...)`` update is
        reflected here with no separate bookkeeping step, which is also why
        this class defines no ``update_value``/``update_description``
        methods of its own: ``replace()`` already re-runs ``__post_init__``
        validation on the new instance, and ``type`` just re-derives itself
        from whatever ``value`` that instance ends up with. A validated/
        orchestrated update API belongs to whatever owns a
        ``ContextVariable`` instance, not to this passive data model.
        """
        return type(self.value).__name__

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation of this context variable.

        Built explicitly rather than via ``dataclasses.asdict`` -- that
        helper only sees real dataclass fields, and ``type`` is a property.
        """
        return {
            "name": self.name,
            "value": self.value,
            "description": self.description,
            "readonly": self.readonly,
            "render_mode": self.render_mode,
            "preview_limit": self.preview_limit,
            "type": self.type,
        }
