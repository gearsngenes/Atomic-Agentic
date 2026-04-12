"""Parameter specification and input/output schema extraction for callables.

This module provides:
- ParamSpec: A self-contained specification of a single callable parameter
- extract_io: Universal function to extract parameters and return type from any callable
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Mapping, Optional, get_args, get_origin, get_type_hints

from .sentinels import NO_VAL
from .Exceptions import SchemaError

__all__ = ["ParamSpec", "extract_io", "to_paramspec_list", "is_valid_parameter_order"]

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
    
    POSITIONAL_ONLY = "POSITIONAL_ONLY"
    POSITIONAL_OR_KEYWORD = "POSITIONAL_OR_KEYWORD"
    VAR_POSITIONAL = "VAR_POSITIONAL"
    KEYWORD_ONLY = "KEYWORD_ONLY"
    VAR_KEYWORD = "VAR_KEYWORD"

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
    """Convert a type annotation into a readable string representation.
    
    This internal helper normalizes type annotations from function signatures
    into human-readable strings suitable for serialization and display.
    
    Handles all annotation styles including plain types, forward references,
    PEP 585 generic types (e.g. ``dict[str, int]``), and ``typing`` module types
    (e.g. ``List[Dict[str, int]]``).
    
    Parameters
    ----------
    ann : Any
        A type annotation object (from inspect.signature or typing module).
    
    Returns
    -------
    str
        Human-readable type string. Behavior:
        
        - ``'Any'`` – if annotation is missing or empty (``inspect._empty`` or None)
        - string as-is – if annotation is already a string (forward reference)
        - nested structure – for parameterized types (e.g. ``List[str]`` → ``\"list[str]\"``)
        - class name – for plain classes (e.g. ``int`` → ``\"int\"``, ``MyClass`` → ``\"MyClass\"``)
        - best-effort ``str(ann)`` – fallback for unknown types
    
    Examples
    --------
    >>> _format_annotation(str)
    'str'
    >>> _format_annotation(dict[str, int])
    'dict[str, int]'
    >>> _format_annotation(inspect._empty)
    'Any'
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


def is_valid_parameter_order(parameters: list[ParamSpec]) -> bool:
    """
    Validate that parameters follow Python-compatible signature rules.

    Validates:
    1. No duplicate parameter names.
    2. Parameter kinds appear in valid order:
       POSITIONAL_ONLY -> POSITIONAL_OR_KEYWORD -> VAR_POSITIONAL
       -> KEYWORD_ONLY -> VAR_KEYWORD
    3. VAR_POSITIONAL and VAR_KEYWORD appear at most once.
    4. Defaults in the positional-capable section follow Python's trailing-default rule:
       once a POSITIONAL_ONLY or POSITIONAL_OR_KEYWORD parameter has a default,
       all later parameters in that same section must also have defaults.
    5. KEYWORD_ONLY parameters may be mixed required/optional in any order.
    6. VAR_POSITIONAL / VAR_KEYWORD must not declare defaults.

    Parameters
    ----------
    parameters : list[ParamSpec]
        List of parameter specifications to validate.

    Returns
    -------
    bool
        True if valid ordering. Raises SchemaError if invalid.

    Raises
    ------
    SchemaError
        If parameters are malformed, out of order, duplicated, or violate
        default-placement rules.
    """
    if not isinstance(parameters, list):
        raise TypeError(
            f"is_valid_parameter_order expects list[ParamSpec], got {type(parameters)!r}"
        )

    if not all(isinstance(spec, ParamSpec) for spec in parameters):
        raise TypeError("All items in parameters must be ParamSpec instances")

    # ------------------------------------------------------------------
    # Duplicate names
    # ------------------------------------------------------------------
    seen_names: set[str] = set()
    duplicate_names: list[str] = []

    for spec in parameters:
        if spec.name in seen_names and spec.name not in duplicate_names:
            duplicate_names.append(spec.name)
        seen_names.add(spec.name)

    if duplicate_names:
        raise SchemaError(f"Duplicate parameter names: {duplicate_names}")

    # ------------------------------------------------------------------
    # Kind ordering
    # ------------------------------------------------------------------
    kind_order = {
        ParamSpec.POSITIONAL_ONLY: 0,
        ParamSpec.POSITIONAL_OR_KEYWORD: 1,
        ParamSpec.VAR_POSITIONAL: 2,
        ParamSpec.KEYWORD_ONLY: 3,
        ParamSpec.VAR_KEYWORD: 4,
    }

    last_priority = -1
    last_kind: str | None = None
    seen_varpos = False
    seen_varkw = False

    for index, spec in enumerate(parameters):
        kind = spec.kind

        if kind not in kind_order:
            raise SchemaError(f"Unknown parameter kind: {kind!r} at index {index}")

        priority = kind_order[kind]
        if priority < last_priority:
            raise SchemaError(
                f"Invalid parameter order at index {index}: "
                f"{kind} comes after {last_kind}"
            )

        if kind == ParamSpec.VAR_POSITIONAL:
            if seen_varpos:
                raise SchemaError("Only one VAR_POSITIONAL parameter is allowed")
            seen_varpos = True
            if spec.default is not NO_VAL:
                raise SchemaError(
                    f"VAR_POSITIONAL parameter {spec.name!r} cannot have a default"
                )

        elif kind == ParamSpec.VAR_KEYWORD:
            if seen_varkw:
                raise SchemaError("Only one VAR_KEYWORD parameter is allowed")
            seen_varkw = True
            if spec.default is not NO_VAL:
                raise SchemaError(
                    f"VAR_KEYWORD parameter {spec.name!r} cannot have a default"
                )

        last_priority = priority
        last_kind = kind

    # ------------------------------------------------------------------
    # Default placement
    # ------------------------------------------------------------------
    #
    # Python's trailing-default rule applies only to parameters that can be
    # passed positionally:
    #   POSITIONAL_ONLY + POSITIONAL_OR_KEYWORD
    #
    # KEYWORD_ONLY parameters are a separate section and may be mixed
    # required/optional in any order.
    # ------------------------------------------------------------------
    saw_default_in_positional_section = False

    for index, spec in enumerate(parameters):
        kind = spec.kind
        has_default = spec.default is not NO_VAL

        if kind in (ParamSpec.POSITIONAL_ONLY, ParamSpec.POSITIONAL_OR_KEYWORD):
            if has_default:
                saw_default_in_positional_section = True
            elif saw_default_in_positional_section:
                raise SchemaError(
                    f"Required parameter {spec.name!r} at index {index} cannot follow "
                    "a defaulted positional parameter"
                )

        elif kind in (ParamSpec.VAR_POSITIONAL, ParamSpec.KEYWORD_ONLY, ParamSpec.VAR_KEYWORD):
            # Separate section; no positional trailing-default rule applies here.
            continue

    return True


def _is_typed_dict_class(obj: Any) -> bool:
    """Return whether ``obj`` appears to be a TypedDict class."""
    return (
        isinstance(obj, type)
        and issubclass(obj, dict)
        and hasattr(obj, "__annotations__")
        and hasattr(obj, "__total__")
    )


def to_paramspec_list(
    schema: Optional[type | list[str] | tuple[str, ...] | set[str] | list[ParamSpec]],
) -> list[ParamSpec]:
    """Normalize supported schema inputs into a fresh canonical ``list[ParamSpec]``.

    Accepted inputs
    ---------------
    - ``None`` -> empty list
    - ``TypedDict`` class
    - ``list[str]``
    - ``tuple[str, ...]``
    - ``set[str]`` using snapshot iteration order
    - ``list[ParamSpec]``

    Returns
    -------
    list[ParamSpec]
        Fresh ParamSpec objects with canonical sequential indices.

    Raises
    ------
    SchemaError
        If the schema input is unsupported or invalid.
    """
    # ------------------------------------------------------------------
    # None -> empty schema
    # ------------------------------------------------------------------
    if schema is None:
        normalized: list[ParamSpec] = []
        is_valid_parameter_order(normalized)
        return normalized

    # ------------------------------------------------------------------
    # TypedDict class -> use field annotations as ParamSpec.type
    # ------------------------------------------------------------------
    if _is_typed_dict_class(schema):
        hints = get_type_hints(schema)
        normalized = [
            ParamSpec(
                name=name,
                index=index,
                kind=ParamSpec.POSITIONAL_OR_KEYWORD,
                type=_format_annotation(annotation),
                default=NO_VAL,
            )
            for index, (name, annotation) in enumerate(hints.items())
        ]
        is_valid_parameter_order(normalized)
        return normalized

    # ------------------------------------------------------------------
    # list[str] / tuple[str, ...] / set[str]
    # Snapshot current iteration order for tuples/sets as provided.
    # ------------------------------------------------------------------
    if isinstance(schema, (list, tuple, set)):
        items = list(schema)

        if not items:
            normalized = []
            is_valid_parameter_order(normalized)
            return normalized

        if all(isinstance(item, str) for item in items):
            normalized = [
                ParamSpec(
                    name=item,
                    index=index,
                    kind=ParamSpec.POSITIONAL_OR_KEYWORD,
                    type="Any",
                    default=NO_VAL,
                )
                for index, item in enumerate(items)
            ]
            is_valid_parameter_order(normalized)
            return normalized

        if isinstance(schema, list) and all(isinstance(item, ParamSpec) for item in items):
            normalized = [
                ParamSpec(
                    name=item.name,
                    index=index,
                    kind=item.kind,
                    type=item.type,
                    default=item.default,
                )
                for index, item in enumerate(items)
            ]
            is_valid_parameter_order(normalized)
            return normalized

        raise SchemaError(
            "Schema sequences must be one of: list[str], tuple[str, ...], "
            "set[str], or list[ParamSpec]."
        )

    # ------------------------------------------------------------------
    # Unsupported input
    # ------------------------------------------------------------------
    raise SchemaError(
        "Unsupported schema type. Expected one of: None, TypedDict class, "
        "list[str], tuple[str, ...], set[str], or list[ParamSpec]."
    )
