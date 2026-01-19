from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, Dict
import re

from .sentinels import NO_VAL

class ParamSpec(dict):
    """Typed parameter specification for callable parameters.

    Behaves like a read-only mapping and a self-contained typed container. This design
    makes ``ParamSpec`` instances JSON-serializable by default (they are dicts),
    while also exposing convenient attribute access for internal code.

    Each ParamSpec is a complete, self-sufficient atom of information containing:
      - name: str (parameter name)
      - index: int (parameter position in signature order)
      - kind: str (parameter kind, e.g. POSITIONAL_ONLY, KEYWORD_ONLY)
      - type: str (human-readable type name)
      - default: Any or ``NO_VAL`` sentinel when no default is present

    Notes:
      - Instances are intentionally immutable (attempts to set items will raise).
      - Use :meth:`to_dict()` for an explicit dict representation.
    """

    __slots__ = ("_name", "_index", "_kind", "_type", "_default")

    def __init__(self, name: str, index: int, kind: str, type: str, default: Any = NO_VAL) -> None:
        # Initialize both mapping contents and attribute storage
        dict.__init__(self, name=name, index=index, kind=kind, type=type)
        if default is not NO_VAL:
            dict.__setitem__(self, "default", default)
        self._name = name
        self._index = index
        self._kind = kind
        self._type = type
        self._default = default

    # Attribute accessors
    @property
    def name(self) -> str:
        return self._name

    @property
    def index(self) -> int:
        return self._index

    @property
    def kind(self) -> str:
        return self._kind

    @property
    def type(self) -> str:
        return self._type

    @property
    def default(self) -> Any:
        return self._default

    # Read-only mapping (prevent mutation)
    def __setitem__(self, key, value):  # pragma: no cover - trivial immutability
        raise TypeError("ParamSpec is immutable")

    def __delitem__(self, key):  # pragma: no cover - trivial immutability
        raise TypeError("ParamSpec is immutable")

    # Convenience
    def to_dict(self) -> dict:
        """Return a JSON-serializable dict representation of this ParamSpec."""
        d = {"name": self._name, "index": self._index, "kind": self._kind, "type": self._type}
        if self._default is not NO_VAL:
            d["default"] = self._default
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ParamSpec":
        """Create a ParamSpec from a mapping produced by :meth:`to_dict()`.

        The mapping must contain ``name`` (str), ``index`` (int), ``kind`` (str), and ``type`` (str).
        ``default`` is optional and treated as an explicit default when present.
        """
        if not isinstance(d, Mapping):
            raise TypeError("ParamSpec.from_dict expects a mapping")
        name = d.get("name")
        idx = d.get("index")
        kind = d.get("kind")
        type_str = d.get("type")
        if not all(isinstance(v, t) for v, t in [(name, str), (idx, int), (kind, str), (type_str, str)]):
            raise TypeError("ParamSpec.from_dict expects 'name' (str), 'index' (int), 'kind' (str), and 'type' (str)")
        default = d.get("default", NO_VAL)
        return cls(name=name, index=idx, kind=kind, type=type_str, default=default)

# Canonical mapping of parameter name -> ParamSpec (legacy, for backward compat)
ParameterMap = dict[str, ParamSpec]

# Legacy aliases for backward compatibility
ArgumentMap = ParameterMap  # Deprecated: use ParameterMap instead
ArgSpec = ParamSpec  # Deprecated: use ParamSpec instead

# Valid name: like a Python identifier (letters, digits, underscores), must not start with a digit
_VALID_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class AtomicInvokable(ABC):
    """
    Basal invokable contract for Tools, Agents, and Workflows.

    Overview
    --------
    `AtomicInvokable` defines the minimal, language-level contract that all
    executable primitives in this codebase must satisfy. It standardises:

    - **identity**: validated `name` and `description` (human-friendly strings).
    - **interface**: a single execution entrypoint ``invoke(inputs: Mapping[str, Any])``.
    - **parameters & return type**: declared at construction time as concrete
      `parameters: list[ParamSpec]` and `return_type: str`.
    - **persistibility**: a small overrideable heuristic ``_compute_is_persistible()`` that
      indicates whether the instance can be reliably rehydrated from metadata.

    Parameters and schema
    ---------------------
    Parameters are passed as a list of ``ParamSpec`` objects at construction time.
    Each ``ParamSpec`` contains complete information:
      - ``name``: parameter name
      - ``index``: position in signature order
      - ``kind``: one of ``POSITIONAL_ONLY``, ``POSITIONAL_OR_KEYWORD``, ``KEYWORD_ONLY``,
        ``VAR_POSITIONAL``, ``VAR_KEYWORD``
      - ``type``: human-readable type string
      - ``default``: default value, or ``NO_VAL`` if not present

    The list order defines the parameter sequence. ``index`` is stored for redundancy
    (self-sufficiency) and should match list position.

    Invocation and inputs
    ---------------------
    ``invoke(inputs)`` accepts a ``Mapping[str, Any]`` where keys correspond to
    parameter names. Implementations should raise clear, typed exceptions on invalid inputs.

    Persistibility
    ---------------
    Subclasses override ``_compute_is_persistible()`` to indicate whether instances
    can be reliably rehydrated from their metadata (name, description, parameters, return_type).

    Notes
    -----
    - The class intentionally does not expose a human-readable ``signature`` inside
      ``to_dict()`` to reduce churn when persisting metadata; ``signature`` remains
      available as a convenience property for logging and UIs.
    - Legacy ``arguments_map`` property is available for backward compatibility.
    """

    def __init__(
        self,
        *,
        name: str,
        description: str,
        parameters: list[ParamSpec],
        return_type: str,
    ) -> None:
        # setters include validation
        self.name = name
        self.description = description

        # Validate parameters
        if not isinstance(parameters, list):
            raise TypeError(
                f"{type(self).__name__}: parameters must be a list[ParamSpec], got {type(parameters)!r}"
            )
        if not all(isinstance(p, ParamSpec) for p in parameters):
            raise TypeError(
                f"{type(self).__name__}: all parameters must be ParamSpec instances"
            )

        # Validate parameter names are unique and valid
        param_names = [p.name for p in parameters]
        if len(param_names) != len(set(param_names)):
            raise TypeError(
                f"{type(self).__name__}: duplicate parameter names detected: {param_names}"
            )
        for p in parameters:
            if not isinstance(p.name, str) or not p.name:
                raise TypeError(
                    f"{type(self).__name__}: parameter names must be non-empty strings"
                )
            if not _VALID_NAME.match(p.name):
                raise ValueError(
                    f"{type(self).__name__}: parameter name {p.name!r} is not a valid identifier"
                )

        # Validate indices are consistent with list position
        for i, p in enumerate(parameters):
            if p.index != i:
                raise TypeError(
                    f"{type(self).__name__}: parameter at position {i} has mismatched index {p.index}"
                )

        # Validate return type
        if not isinstance(return_type, str):
            raise TypeError(
                f"{type(self).__name__}: return_type must be a str, got {type(return_type)!r}"
            )

        self._parameters: list[ParamSpec] = parameters
        self._return_type: str = return_type
        self._is_persistible: bool = self._compute_is_persistible()

    # ---------------------------------------------------------------- #
    # Name + description with validation
    # ---------------------------------------------------------------- #
    @property
    def name(self) -> str:
        """The canonical name of this invokable."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("name must be a non-empty string")
        if not _VALID_NAME.match(value):
            raise ValueError(
                f"name must be alphanumeric/underscore and not start with a digit; got {value!r}"
            )
        self._name = value

    @property
    def description(self) -> str:
        """Human-friendly description."""
        return self._description

    @description.setter
    def description(self, value: str) -> None:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("description must be a non-empty string")
        self._description = value.strip()

    @property
    def is_persistible(self) -> bool:
        return self._is_persistible
    
    # ---------------------------------------------------------------- #
    # Parameters and return type (primary API)
    # ---------------------------------------------------------------- #
    @property
    def parameters(self) -> list[ParamSpec]:
        """Primary parameter specification as an ordered list of ParamSpec objects.

        Returns a list of ``ParamSpec`` in signature order. The list is copied
        to prevent external mutation.
        """
        return list(self._parameters)

    @property
    def return_type(self) -> str:
        """Return type (string) of this invokable."""
        return self._return_type

    # ---------------------------------------------------------------- #
    # Legacy backward compatibility properties
    # ---------------------------------------------------------------- #
    @property
    def parameters_map(self) -> ParameterMap:
        """Legacy alternative viewing mechanism: parameters as a dict mapping name -> ParamSpec.

        This property is provided for backward compatibility. New code should prefer
        the ``parameters`` property which provides the ordered list directly.
        """
        return {spec.name: spec for spec in self._parameters}

    @property
    def arguments_map(self) -> ArgumentMap:
        """Deprecated legacy property. Use ``parameters_map`` or ``parameters`` instead."""
        return self.parameters_map

    # ---------------------------------------------------------------- #
    # Signature helper
    # ---------------------------------------------------------------- #
    @property
    def signature(self) -> str:
        """
        Returns a signature like:

            ClassName.name(param1: Type, param2: Type = default) -> return_type
        """
        params = []
        for spec in self._parameters:
            ptype = spec.type or "Any"
            default_marker = ""
            if spec.default is not NO_VAL:
                default_marker = f" = {spec.default!r}"
            
            if spec.kind == "VAR_POSITIONAL":
                params.append(f"*{spec.name}: {ptype}{default_marker}")
            elif spec.kind == "VAR_KEYWORD":
                params.append(f"**{spec.name}: {ptype}{default_marker}")
            else:
                params.append(f"{spec.name}: {ptype}{default_marker}")
        
        params_str = ", ".join(params)
        return f"{type(self).__name__}.{self.name}({params_str}) -> {self.return_type}"

    # ---------------------------------------------------------------- #
    # Abstract contract
    # ---------------------------------------------------------------- #
    @abstractmethod
    def _compute_is_persistible(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def invoke(self, inputs: Mapping[str, Any]) -> Any:
        """Perform work."""
        raise NotImplementedError

    # ---------------------------------------------------------------- #
    # Default metadata serialization
    # ---------------------------------------------------------------- #
    def to_dict(self) -> Dict[str, Any]:
        """
        Minimal metadata serialization.

        Does *not* include `signature` by default to reduce churn in persisted metadata.
        """
        return {
            "type": type(self).__name__,
            "name": self.name,
            "description": self.description,
            "parameters": [spec.to_dict() for spec in self._parameters],
            "return_type": self.return_type,
        }

    # ---------------------------------------------------------------- #
    # Unified repr/str
    # ---------------------------------------------------------------- #
    def __str__(self) -> str:
        return f"<{self.signature} - {self.description}>"

    def __repr__(self) -> str:
        return f"{self.signature}: {self.description}"
