"""Parameter specification and input/output schema extraction for callables.

This module provides:
- ParamSpec: A self-contained specification of a single callable parameter
- extract_io: Universal function to extract parameters and return type from any callable
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Mapping, get_args, get_origin

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


def _format_annotation(ann: Any) -> str:
    """Convert a type annotation into a readable string.

    Behaviour:
    - If missing/empty → 'Any'.
    - If already a string → returned as-is.
    - If a parameterized / generic type (e.g. List[Dict[str, int]] or dict[str, int]):
      builds the full nested structure string.
    - If a plain class → its name (e.g. 'int', 'MyModel').
    - Otherwise → best-effort str(ann).
    """

    # Missing / unknown annotation
    if ann is inspect._empty or ann is None:
        return "Any"

    # Forward reference or explicit string annotation
    if isinstance(ann, str):
        return ann

    # typing / generic / PEP 585 parameterized types
    origin = get_origin(ann)
    if origin is not None:
        # Recursively format origin and args
        origin_str = _format_annotation(origin)
        args = get_args(ann)
        if not args:
            return origin_str
        args_str = ", ".join(_format_annotation(a) for a in args)
        return f"{origin_str}[{args_str}]"

    # Plain classes / types
    module = getattr(ann, "__module__", None)
    name = getattr(ann, "__name__", None)
    if module == "builtins" and name:
        # int, str, dict, list, etc.
        return name
    if name:
        # Custom or library class
        return name

    # Fallback: best-effort string representation
    return str(ann)


def extract_io(function: Callable) -> tuple[list[ParamSpec], str]:
    """Extract parameter specifications and return type from a callable.

    This is a universal, reusable function that builds an ordered list of ParamSpec
    objects from a function's signature. Each ParamSpec is self-sufficient and
    contains all necessary information (name, index, kind, type, default).

    Parameters
    ----------
    function : Callable
        Any Python callable (function, method, lambda, etc.)

    Returns
    -------
    tuple[list[ParamSpec], str]
        A tuple of:
        - List of ParamSpec objects in signature order
        - Return type as a human-readable string

    Raises
    ------
    TypeError
        If function is not callable
    """
    if not callable(function):
        raise TypeError(f"extract_io expects a callable, got {type(function)!r}")

    sig = inspect.signature(function)
    parameters: list[ParamSpec] = []

    for index, (name, param) in enumerate(sig.parameters.items()):
        kind_name = param.kind.name  # e.g. "POSITIONAL_ONLY"
        ann = param.annotation
        default = param.default

        # Decide the source of the type information:
        # 1) annotation if present,
        # 2) otherwise the default's type if present,
        # 3) otherwise "Any".
        if ann is not inspect._empty:
            raw_type = ann
        elif default is not inspect._empty:
            raw_type = type(default)
        else:
            raw_type = inspect._empty

        type_str = _format_annotation(raw_type)

        # Handle default value
        default_val = default if default is not inspect._empty else NO_VAL

        # Create self-sufficient ParamSpec with name
        spec = ParamSpec(
            name=name,
            index=index,
            kind=kind_name,
            type=type_str,
            default=default_val,
        )
        parameters.append(spec)

    # Return type: annotation if present, else 'Any'
    ret_ann = sig.return_annotation
    return_type = _format_annotation(ret_ann)

    return parameters, return_type
