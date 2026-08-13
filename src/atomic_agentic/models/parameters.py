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
from ..constants.parameters import (
    POSITIONAL_ONLY,
    POSITIONAL_OR_KEYWORD,
    VAR_POSITIONAL,
    KEYWORD_ONLY,
    VAR_KEYWORD,
    VALID_KINDS,
)

__all__ = ["ParamSpec", "ParameterReport"]


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

    # Shims onto constants/parameters.py -- that module is the real source of
    # truth (constants/ sits below models/ in the dependency topology, so
    # utils/parameters.py's kind-priority table can share it directly); these
    # class attributes exist only to keep ParamSpec.POSITIONAL_ONLY (etc.)
    # working unchanged for every existing caller.
    POSITIONAL_ONLY: ClassVar[str] = POSITIONAL_ONLY
    POSITIONAL_OR_KEYWORD: ClassVar[str] = POSITIONAL_OR_KEYWORD
    VAR_POSITIONAL: ClassVar[str] = VAR_POSITIONAL
    KEYWORD_ONLY: ClassVar[str] = KEYWORD_ONLY
    VAR_KEYWORD: ClassVar[str] = VAR_KEYWORD

    _VALID_KINDS: ClassVar[tuple[str, ...]] = VALID_KINDS

    name: str
    index: int
    kind: str
    type: tuple[str, ...]
    default: Any = NO_VAL
    description: str | None = None

    def __post_init__(self) -> None:
        """Validate and normalize dataclass fields after initialization."""
        cleaned_name, validated_index, validated_kind, cleaned_type, validated_description = (
            self._validate_init_args(
                name=self.name,
                index=self.index,
                kind=self.kind,
                type=self.type,
                description=self.description,
            )
        )

        object.__setattr__(self, "name", cleaned_name)
        object.__setattr__(self, "index", validated_index)
        object.__setattr__(self, "kind", validated_kind)
        object.__setattr__(self, "type", cleaned_type)
        object.__setattr__(self, "description", validated_description)

    @classmethod
    def _validate_init_args(
        cls,
        *,
        name: str,
        index: int,
        kind: str,
        type: str | tuple[str, ...] | list[str],
        description: str | None = None,
    ) -> tuple[str, int, str, tuple[str, ...], str | None]:
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

        if isinstance(type, str):
            type_items: tuple[Any, ...] = (type,)
        elif isinstance(type, (tuple, list)):
            type_items = tuple(type)
        else:
            raise TypeError(
                f"ParamSpec.type must be a str, tuple[str, ...], or list[str], "
                f"got {type.__class__.__name__}"
            )

        if not type_items:
            raise ValueError("ParamSpec.type must be non-empty")

        cleaned_items: list[str] = []
        for item in type_items:
            if not isinstance(item, str):
                raise TypeError(
                    f"ParamSpec.type members must be str, got {item.__class__.__name__}"
                )
            stripped_item = item.strip()
            if not stripped_item:
                raise ValueError("ParamSpec.type members must be non-empty strings")
            cleaned_items.append(stripped_item)

        cleaned_type = tuple(sorted(set(cleaned_items)))

        if description is not None and not isinstance(description, str):
            raise TypeError(
                f"ParamSpec.description must be a str or None, got {description.__class__.__name__}"
            )
        if isinstance(description, str):
            description = description.strip() or None

        return cleaned_name, index, kind, cleaned_type, description

    def to_dict(self) -> dict[str, Any]:
        """Return the explicit serialized dictionary representation."""
        d: dict[str, Any] = {
            "name": self.name,
            "index": self.index,
            "kind": self.kind,
            "type": list(self.type),
        }
        if self.default is not NO_VAL:
            d["default"] = self.default
        if self.description is not None:
            d["description"] = self.description
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> ParamSpec:
        """Create a ParamSpec from a serialized mapping.

        The mapping must contain ``name`` (str), ``index`` (int), ``kind`` (str),
        and ``type`` (str, list[str], or tuple[str, ...]). ``default`` is
        optional and treated as an explicit default only when present.
        """
        if not isinstance(d, Mapping):
            raise TypeError("ParamSpec.from_dict expects a mapping")

        name = d.get("name")
        idx = d.get("index")
        kind = d.get("kind")
        type_val = d.get("type")
        default = d.get("default", NO_VAL)
        description = d.get("description", None)

        if not all(
            isinstance(v, t)
            for v, t in [
                (name, str),
                (idx, int),
                (kind, str),
            ]
        ) or not isinstance(type_val, (str, list, tuple)):
            raise TypeError(
                "ParamSpec.from_dict expects 'name' (str), 'index' (int), "
                "'kind' (str), and 'type' (str, list[str], or tuple[str, ...])"
            )

        return cls(name=name, index=idx, kind=kind, type=type_val, default=default, description=description)


@dataclass(frozen=True, slots=True)
class ParameterReport:
    """One parameter name's cross-source reconciliation outcome.

    Produced by ``utils.parameters.build_parameter_report``/
    ``build_parameter_reports`` -- carries the actual computed
    reconciliation result (witness types, kind compatibility, winning
    source, identity), not just raw aggregated observations. The caller
    still owns the raise/warn policy decision, but doesn't need to
    recompute witness-set/kind compatibility itself.

    Fields
    ------
    parameter_name:
        The parameter name this report covers.
    witness_types:
        The full set of type tokens compatible with every opinionated
        source's declared type (see ``utils.parameters.n_way_type_witness``).
        Empty means no compatible witness exists -- a genuine type conflict.
    winner_source:
        Index into ``observations`` of the highest-priority source that
        actually declares this name (first non-``None`` entry). Supplies
        kind/default/description for the reconciled parameter.
    kind_compatible:
        Whether every declaring source's ``kind`` is jointly compatible
        (see ``utils.parameters.n_way_kind_compatible``).
    observations:
        Dense, one slot per source in the same priority order the caller
        assembled -- ``None`` where that source doesn't declare this name.
        Positionally aligned with whatever priority-ordered source list
        produced this report; the caller (not this dataclass) knows what
        each index means.
    is_identical:
        Whether every present observation is ``semantically_identical`` to
        the winner (vacuously ``True`` when only one source declares the
        name).
    """

    parameter_name: str
    witness_types: frozenset[str]
    winner_source: int
    kind_compatible: bool
    observations: tuple[ParamSpec | None, ...]
    is_identical: bool
