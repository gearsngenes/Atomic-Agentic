from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, Dict
import re

from .sentinels import NO_VAL

class ArgSpec(dict):
    """Typed argument specification for parameters returned by build_args_returns().

    Behaves like a read-only mapping and a small typed container. This design
    makes ``ArgSpec`` instances JSON-serializable by default (they are dicts),
    while also exposing convenient attribute access for internal code.

    Fields:
      - index: int (parameter position)
      - kind: str (parameter kind, e.g. POSITIONAL_ONLY)
      - type: str (human-readable type name)
      - default: Any or ``NO_VAL`` sentinel when no default is present

    Notes:
      - Instances are intentionally immutable (attempts to set items will raise).
      - Use :meth:`to_dict()` for an explicit dict representation (identical to the
        mapping view produced by the instance itself).
    """

    __slots__ = ("_index", "_kind", "_type", "_default")

    def __init__(self, index: int, kind: str, type: str, default: Any = NO_VAL) -> None:
        # Initialize both mapping contents and attribute storage
        dict.__init__(self, index=index, kind=kind, type=type)
        if default is not NO_VAL:
            dict.__setitem__(self, "default", default)
        self._index = index
        self._kind = kind
        self._type = type
        self._default = default

    # Attribute accessors
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
        raise TypeError("ArgSpec is immutable")

    def __delitem__(self, key):  # pragma: no cover - trivial immutability
        raise TypeError("ArgSpec is immutable")

    # Convenience
    def to_dict(self) -> dict:
        """Return a JSON-serializable dict representation of this ArgSpec."""
        d = {"index": self._index, "kind": self._kind, "type": self._type}
        if self._default is not NO_VAL:
            d["default"] = self._default
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ArgSpec":
        """Create an ArgSpec from a mapping produced by :meth:`to_dict()`.

        The mapping must contain ``index`` (int), ``kind`` (str), and ``type`` (str).
        ``default`` is optional and treated as an explicit default when present.
        """
        if not isinstance(d, Mapping):
            raise TypeError("ArgSpec.from_dict expects a mapping")
        idx = d.get("index")
        kind = d.get("kind")
        type_str = d.get("type")
        if not isinstance(idx, int) or not isinstance(kind, str) or not isinstance(type_str, str):
            raise TypeError("ArgSpec.from_dict expects 'index' (int), 'kind' (str), and 'type' (str)")
        default = d.get("default", NO_VAL)
        return cls(index=idx, kind=kind, type=type_str, default=default)

# Canonical mapping of argument name -> ArgSpec
ArgumentMap = dict[str, ArgSpec]

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
    - **interface**: a schema-producing hook ``build_args_returns() -> (ArgumentMap, str)``
      and a single execution entrypoint ``invoke(inputs)``.
    - **metadata**: read-only ``arguments_map`` (mapping of argument name -> :class:`ArgSpec`)
      and ``return_type`` which are derived from ``build_args_returns()``.
    - **persistibility**: a small overrideable heuristic ``_compute_is_persistible()`` that
      indicates whether the instance can be reliably rehydrated from metadata.

    Key expectations and invariants
    -------------------------------
    - ``build_args_returns()`` must return a mapping (``dict``) whose values
      are either :class:`ArgSpec` instances or dict-like metadata that can be coerced into
      an :class:`ArgSpec` (fields: ``index``, ``kind``, ``type``, optional ``default``).
      Indices must be unique integers and define deterministic argument order. Callers that
      require ordering should sort by ``ArgSpec.index``.
    - ``invoke(inputs)`` accepts a ``Mapping[str, Any]`` and performs the component's work.
      Implementations should raise clear, typed exceptions on invalid inputs.
    - ``to_dict()`` returns a minimal diagnostic snapshot intended for logging/inspection.
      It serialises :class:`ArgSpec` values into plain dicts and omits defaults marked with
      the shared ``NO_VAL`` sentinel (allowing ``None`` to be an explicit default value).
    - Subclasses that depend on instance attributes to build their argument schema should
      initialise those attributes *before* calling ``super().__init__`` so that
      ``build_args_returns()`` can inspect a correctly-initialised object.

    Notes
    -----
    - The class intentionally does not expose a human-readable ``signature`` inside
      ``to_dict()`` to reduce churn when persisting metadata; ``signature`` remains
      available as a convenience property for logging and UIs.
    """

    def __init__(
        self,
        *,
        name: str,
        description: str,
    ) -> None:
        # setters include validation
        self.name = name
        self.description = description

        args, ret = self.build_args_returns()

        # normalize to dict
        args = dict(args)

        # Accept mapping-like results (dict) and normalize to ArgumentMap.
        if not isinstance(args, (dict, Mapping)) or any(not isinstance(arg, ArgSpec) for arg in args.values()):
            raise TypeError(
                f"{type(self).__name__}.build_args_returns must return a mapping of argument metadata"
            )

        # Validate indices are unique
        indices = [s.index for s in args.values()]
        if len(indices) != len(set(indices)):
            raise TypeError(f"{type(self).__name__}.build_args_returns: duplicate argument indices detected")

        self._arguments_map: ArgumentMap = args
        if not isinstance(ret, str):
            raise TypeError(
                f"{type(self).__name__}.build_args_returns must return str for return_type"
            )
        self._return_type: str = ret
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
    # Exposed read-only derived properties
    # ---------------------------------------------------------------- #
    @property
    def arguments_map(self) -> ArgumentMap:
        """Public view of argument metadata.

        Returns a mapping of argument name -> :class:`ArgSpec`. ``ArgSpec`` objects
        behave like read-only mappings and are JSON-serializable (they produce
        the same dict shape as before: keys ``index``, ``kind``, ``type`` and
        optional ``default`` when present). This keeps the public API simple
        (there is no separate typed/serializable property) while preserving
        backward-compatible JSON behaviour.
        """
        return dict(self._arguments_map)

    @property
    def return_type(self) -> str:
        """Return type (string) produced by build_args_returns()."""
        return self._return_type

    # ---------------------------------------------------------------- #
    # Signature helper
    # ---------------------------------------------------------------- #
    @property
    def signature(self) -> str:
        """
        Returns a signature like:

            name(arg1: Type, arg2: Type = default) -> return_type
        """
        params = []
        for param, spec in sorted(self._arguments_map.items(), key=lambda kv: kv[1].index):
            ptype = spec.type or "Any"
            if spec.default is not NO_VAL:
                params.append(f"{param}: {ptype} = {spec.default!r}")
            else:
                params.append(f"{param}: {ptype}")
        params_str = ", ".join(params)
        return f"{type(self).__name__}.{self.name}({params_str}) -> {self.return_type}"

    # ---------------------------------------------------------------- #
    # Abstract contract
    # ---------------------------------------------------------------- #
    @abstractmethod
    def _compute_is_persistible(self) -> bool:
        raise NotImplementedError
    
    @abstractmethod
    def build_args_returns(self) -> tuple[ArgumentMap, str]:
        """Return (arguments_map, return_type)."""
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

        Does *not* include `signature` by default, per request.
        """
        return {
            "type": type(self).__name__,
            "name": self.name,
            "description": self.description,
            "arguments_map": {
                name: spec.to_dict()
                for name, spec in self._arguments_map.items()
                },
            "return_type": self.return_type,
        }

    # ---------------------------------------------------------------- #
    # Unified repr/str
    # ---------------------------------------------------------------- #
    def __str__(self) -> str:
        return f"<{self.signature} - {self.description}>"

    def __repr__(self) -> str:
        return f"{self.signature}: {self.description}"
