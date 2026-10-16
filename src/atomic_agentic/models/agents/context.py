from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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
    """

    name: str
    value: Any
    description: str | None = None

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
            "type": self.type,
        }
