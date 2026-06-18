"""Parameter shape specification for callables.

This module provides ``ParamSpec``, the canonical schema atom used by
``AtomicInvokable``, ``Tool``, ``Agent``, and ``Workflow`` objects to describe
one declared input parameter.

Utility functions that *use* ``ParamSpec`` (``extract_io``, ``to_paramspec_list``,
``is_valid_parameter_order``) live in ``utils/parameters.py``.

``ParamSpec`` is intentionally object-first in v2: live instances expose attribute
access and explicit ``to_dict()``/``from_dict()`` serialization, but no longer
behave as dict or Mapping instances.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Mapping

from ..constants.core import IDENTIFIER_PATTERN, NO_VAL

__all__ = ["ParamSpec"]


@dataclass(frozen=True, slots=True)
class ParamSpec:
    """Typed parameter specification for callable parameters.

    ``ParamSpec`` is the canonical schema atom used by AtomicInvokable, Tool,
    Agent, and Workflow objects to describe one declared input parameter.

    Each instance is self-sufficient and contains:

    - ``name``: parameter name
    - ``index``: parameter position in signature order
    - ``kind``: parameter kind, e.g. ``POSITIONAL_ONLY`` or ``KEYWORD_ONLY``
    - ``type``: human-readable type annotation string
    - ``default``: explicit default value, or ``NO_VAL`` when no default exists

    Contract
    --------
    ``ParamSpec`` is a frozen dataclass in v2. It supports attribute access:

        spec.name
        spec.index
        spec.kind
        spec.type
        spec.default

    It does not support mapping-style access. Use :meth:`to_dict` when a concrete
    dictionary representation is needed, and :meth:`from_dict` when rebuilding a
    ``ParamSpec`` from serialized metadata.
    """

    POSITIONAL_ONLY: ClassVar[str] = "POSITIONAL_ONLY"
    POSITIONAL_OR_KEYWORD: ClassVar[str] = "POSITIONAL_OR_KEYWORD"
    VAR_POSITIONAL: ClassVar[str] = "VAR_POSITIONAL"
    KEYWORD_ONLY: ClassVar[str] = "KEYWORD_ONLY"
    VAR_KEYWORD: ClassVar[str] = "VAR_KEYWORD"

    _VALID_KINDS: ClassVar[tuple[str, ...]] = (
        POSITIONAL_ONLY,
        POSITIONAL_OR_KEYWORD,
        VAR_POSITIONAL,
        KEYWORD_ONLY,
        VAR_KEYWORD,
    )

    name: str
    index: int
    kind: str
    type: str
    default: Any = NO_VAL

    def __post_init__(self) -> None:
        """Validate and normalize dataclass fields after initialization."""
        cleaned_name, validated_index, validated_kind, cleaned_type = (
            self._validate_init_args(
                name=self.name,
                index=self.index,
                kind=self.kind,
                type=self.type,
            )
        )

        object.__setattr__(self, "name", cleaned_name)
        object.__setattr__(self, "index", validated_index)
        object.__setattr__(self, "kind", validated_kind)
        object.__setattr__(self, "type", cleaned_type)

    @classmethod
    def _validate_init_args(
        cls,
        *,
        name: str,
        index: int,
        kind: str,
        type: str,
    ) -> tuple[str, int, str, str]:
        """Validate and normalize constructor fields before state is finalized."""
        if not isinstance(name, str):
            raise TypeError(
                f"ParamSpec.name must be a str, got {name.__class__.__name__}"
            )

        cleaned_name = name.strip()
        if not cleaned_name:
            raise ValueError("ParamSpec.name must be a non-empty string")

        if not IDENTIFIER_PATTERN.fullmatch(cleaned_name):
            raise ValueError(
                f"ParamSpec.name {cleaned_name!r} is not a valid identifier"
            )

        if not isinstance(index, int) or isinstance(index, bool):
            raise TypeError(
                f"ParamSpec.index must be an int, got {index.__class__.__name__}"
            )

        if index < 0:
            raise ValueError("ParamSpec.index must be >= 0")

        if not isinstance(kind, str):
            raise TypeError(
                f"ParamSpec.kind must be a str, got {kind.__class__.__name__}"
            )

        if kind not in cls._VALID_KINDS:
            raise ValueError(
                "ParamSpec.kind must be one of: "
                f"{', '.join(cls._VALID_KINDS)}; got {kind!r}"
            )

        if not isinstance(type, str):
            raise TypeError(
                f"ParamSpec.type must be a str, got {type.__class__.__name__}"
            )

        cleaned_type = type.strip()
        if not cleaned_type:
            raise ValueError("ParamSpec.type must be a non-empty string")

        return cleaned_name, index, kind, cleaned_type

    def to_dict(self) -> dict[str, Any]:
        """Return the explicit serialized dictionary representation."""
        d: dict[str, Any] = {
            "name": self.name,
            "index": self.index,
            "kind": self.kind,
            "type": self.type,
        }
        if self.default is not NO_VAL:
            d["default"] = self.default
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> ParamSpec:
        """Create a ParamSpec from a serialized mapping.

        The mapping must contain ``name`` (str), ``index`` (int), ``kind`` (str),
        and ``type`` (str). ``default`` is optional and treated as an explicit
        default only when present.
        """
        if not isinstance(d, Mapping):
            raise TypeError("ParamSpec.from_dict expects a mapping")

        name = d.get("name")
        idx = d.get("index")
        kind = d.get("kind")
        type_str = d.get("type")
        default = d.get("default", NO_VAL)

        if not all(
            isinstance(v, t)
            for v, t in [
                (name, str),
                (idx, int),
                (kind, str),
                (type_str, str),
            ]
        ):
            raise TypeError(
                "ParamSpec.from_dict expects 'name' (str), 'index' (int), "
                "'kind' (str), and 'type' (str)"
            )

        return cls(name=name, index=idx, kind=kind, type=type_str, default=default)
