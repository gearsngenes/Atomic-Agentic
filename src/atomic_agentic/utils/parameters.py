"""Parameter utility functions for schema normalization, extraction, and validation.

This module provides the utility layer that operates on ``ParamSpec`` instances.
It sits one tier above ``models/parameters.py`` in the dependency topology
(``utils/`` depends on ``models/``) and is the canonical home for all logic that
*uses* ``ParamSpec`` shapes rather than defining them.

Public surface
--------------
- ``extract_io`` — derive a ``list[ParamSpec]`` and return-type string from any
  callable's signature.
- ``to_paramspec_list`` — normalize any supported schema input (``None``,
  ``TypedDict``, string sequences, ``list[ParamSpec]``) into a fresh canonical
  ``list[ParamSpec]``.
- ``is_valid_parameter_order`` — bool predicate: ``True`` if the list satisfies
  Python-compatible ordering rules, ``False`` on ``SchemaError``, propagates
  ``TypeError``.

Private helpers
---------------
- ``_validate_parameter_order`` — raise-or-return-None enforcement of the same
  rules; used internally by ``to_paramspec_list`` and directly by call sites that
  want the error (``Invokable``, workflow constructors).
- ``_format_annotation``, ``_is_typed_dict_class``, ``_validate_schema_name``,
  ``_paramspec_list_from_strings`` — low-level helpers consumed by the public
  functions above.
- ``_normalize_prompt_template``, ``_try_parse_clean_field`` — template-string
  field discovery/escaping for ``PromptConfig``
  (``models/agents/prompts.py``). Imported directly by that module — a
  one-directional exception to the usual ``models -> utils`` ordering, mirroring
  the existing ``models/results/llm.py -> utils/core.py`` precedent; no cycle
  results since this module has no dependency on ``models/agents/``.
"""

from __future__ import annotations

import inspect
import re
from typing import Annotated, Any, Callable, Optional, get_args, get_origin, get_type_hints

from ..constants.core import IDENTIFIER_PATTERN, IDENTIFIER_PATTERN_TEXT, NO_VAL
from ..exceptions import SchemaError
from ..models.parameters import ParamSpec

__all__ = ["extract_io", "to_paramspec_list", "is_valid_parameter_order"]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _format_annotation(ann: Any) -> str:
    """Convert a type annotation into a readable string representation.

    Normalizes type annotations from function signatures into human-readable
    strings suitable for serialization and display.

    Handles all annotation styles: plain types, forward references, PEP 585
    generic types (e.g. ``dict[str, int]``), and ``typing`` module types
    (e.g. ``List[Dict[str, int]]``).

    Returns ``'Any'`` for missing/empty annotations, the string as-is for
    forward references, a nested ``origin[args]`` form for parameterized types,
    the class name for plain types, and ``str(ann)`` as a fallback.
    """
    # Missing / unknown annotation
    if ann is inspect._empty or ann is None:
        return "Any"

    # NoneType -> "None" to match Python -> None annotation convention
    if ann is type(None):
        return "None"

    # Forward reference or explicit string annotation
    if isinstance(ann, str):
        return ann

    # typing / generic / PEP 585 parameterized types
    origin = get_origin(ann)
    if origin is not None:
        # Recursively format origin and args.
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
        # Custom or library class.
        return name

    # Fallback: best-effort string representation.
    return str(ann)


def _is_typed_dict_class(obj: Any) -> bool:
    """Return whether ``obj`` appears to be a TypedDict class."""
    return (
        isinstance(obj, type)
        and issubclass(obj, dict)
        and hasattr(obj, "__annotations__")
        and hasattr(obj, "__total__")
    )


def _unwrap_annotated(ann: Any) -> tuple[Any, str | None]:
    """Separate the base type and optional description from any annotation.

    If ``ann`` is ``Annotated[T, ...]``, returns ``(T, description)`` where
    ``description`` is the first ``str`` item in the metadata (stripped;
    empty-after-strip becomes ``None``), or ``None`` if no string is present.
    For all other annotations returns ``(ann, None)`` unchanged.
    """
    if get_origin(ann) is not Annotated:
        return ann, None
    args = get_args(ann)
    base = args[0]
    description = next((m for m in args[1:] if isinstance(m, str)), None)
    if description is not None:
        description = description.strip() or None
    return base, description


def _validate_schema_name(name: str) -> str:
    """Validate and normalize one parameter name from a string schema."""
    if not isinstance(name, str):
        raise SchemaError(
            f"Schema parameter names must be strings, got {type(name)!r}"
        )

    cleaned = name.strip()
    if not cleaned:
        raise SchemaError("Schema parameter names must be non-empty strings")

    if not IDENTIFIER_PATTERN.fullmatch(cleaned):
        raise SchemaError(
            f"Schema parameter name {cleaned!r} is not a valid identifier"
        )

    return cleaned


def _paramspec_list_from_strings(items: list[str]) -> list[ParamSpec]:
    """Parse a string schema into a canonical ``list[ParamSpec]``.

    Supported string grammar
    ------------------------
    - ``"name"``     -> POSITIONAL_OR_KEYWORD before keyword-only mode,
                        KEYWORD_ONLY after ``"*"`` or ``"*args"``
    - ``"/"``        -> marker converting previous positional-or-keyword names
                        to POSITIONAL_ONLY
    - ``"*"``        -> marker starting keyword-only mode
    - ``"*args"``    -> VAR_POSITIONAL and starts keyword-only mode
    - ``"**kwargs"`` -> VAR_KEYWORD and must be final
    """
    normalized: list[ParamSpec] = []

    saw_slash = False
    keyword_only_mode = False
    saw_bare_star = False
    saw_varargs = False
    saw_varkwargs = False
    saw_keyword_only_name_after_bare_star = False

    for raw_index, raw_item in enumerate(items):
        item = raw_item.strip()

        if item == "/":
            if saw_slash:
                raise SchemaError("String schema may contain '/' at most once")
            if keyword_only_mode or saw_varargs or saw_bare_star or saw_varkwargs:
                raise SchemaError(
                    "'/' marker must appear before keyword-only or variadic markers"
                )
            if not normalized:
                raise SchemaError("'/' marker requires at least one preceding parameter")

            for spec in normalized:
                if spec.kind != ParamSpec.POSITIONAL_OR_KEYWORD:
                    raise SchemaError(
                        "'/' marker can only convert prior positional-or-keyword parameters"
                    )

            normalized = [
                ParamSpec(
                    name=spec.name,
                    index=index,
                    kind=ParamSpec.POSITIONAL_ONLY,
                    type=spec.type,
                    default=spec.default,
                )
                for index, spec in enumerate(normalized)
            ]
            saw_slash = True
            continue

        if item == "*":
            if saw_bare_star or saw_varargs:
                raise SchemaError(
                    "String schema may contain only one '*' or '*args' marker"
                )
            if saw_varkwargs:
                raise SchemaError("'*' marker cannot appear after '**kwargs'")
            saw_bare_star = True
            keyword_only_mode = True
            continue

        if item.startswith("**"):
            if saw_varkwargs:
                raise SchemaError(
                    "String schema may contain only one '**kwargs' parameter"
                )
            if raw_index != len(items) - 1:
                raise SchemaError(
                    "'**kwargs' style parameter must be the final schema item"
                )

            name = _validate_schema_name(item[2:])
            normalized.append(
                ParamSpec(
                    name=name,
                    index=len(normalized),
                    kind=ParamSpec.VAR_KEYWORD,
                    type="Any",
                    default=NO_VAL,
                )
            )
            saw_varkwargs = True
            continue

        if item.startswith("*"):
            if saw_bare_star or saw_varargs:
                raise SchemaError(
                    "String schema may contain only one '*' or '*args' marker"
                )
            if saw_varkwargs:
                raise SchemaError(
                    "'*args' style parameter cannot appear after '**kwargs'"
                )

            name = _validate_schema_name(item[1:])
            normalized.append(
                ParamSpec(
                    name=name,
                    index=len(normalized),
                    kind=ParamSpec.VAR_POSITIONAL,
                    type="Any",
                    default=NO_VAL,
                )
            )
            saw_varargs = True
            keyword_only_mode = True
            continue

        if saw_varkwargs:
            raise SchemaError("No schema items may appear after '**kwargs'")

        name = _validate_schema_name(item)
        kind = (
            ParamSpec.KEYWORD_ONLY
            if keyword_only_mode
            else ParamSpec.POSITIONAL_OR_KEYWORD
        )

        if saw_bare_star and kind == ParamSpec.KEYWORD_ONLY:
            saw_keyword_only_name_after_bare_star = True

        normalized.append(
            ParamSpec(
                name=name,
                index=len(normalized),
                kind=kind,
                type="Any",
                default=NO_VAL,
            )
        )

    if saw_bare_star and not saw_keyword_only_name_after_bare_star:
        raise SchemaError(
            "Bare '*' marker must be followed by at least one keyword-only parameter"
        )

    return normalized


def _validate_parameter_order(parameters: list[ParamSpec]) -> None:
    """Enforce Python-compatible parameter ordering rules, raising on any violation.

    This is the primary enforcement function. Call sites that want errors
    (``Invokable`` construction, workflow constructors, ``to_paramspec_list``)
    call this directly. ``is_valid_parameter_order`` wraps it as a bool predicate.

    1. Raise TypeError if ``parameters`` is not a list.
    2. Raise TypeError if any item is not a ``ParamSpec`` instance.
    3. Collect duplicate names; raise ``SchemaError`` if any found.
    4. Walk parameters in order; raise ``SchemaError`` if kind priority decreases
       (invalid ordering).
    5. Raise ``SchemaError`` if VAR_POSITIONAL appears more than once.
    6. Raise ``SchemaError`` if a VAR_POSITIONAL parameter has a default.
    7. Raise ``SchemaError`` if VAR_KEYWORD appears more than once.
    8. Raise ``SchemaError`` if a VAR_KEYWORD parameter has a default.
    9. Walk positional-capable parameters (POSITIONAL_ONLY, POSITIONAL_OR_KEYWORD);
       raise ``SchemaError`` if a required parameter follows a defaulted one.
    10. Return None on success.
    """
    if not isinstance(parameters, list):
        raise TypeError(
            f"_validate_parameter_order expects list[ParamSpec], got {type(parameters)!r}"
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

        elif kind in (
            ParamSpec.VAR_POSITIONAL,
            ParamSpec.KEYWORD_ONLY,
            ParamSpec.VAR_KEYWORD,
        ):
            # Separate section; no positional trailing-default rule applies here.
            continue


def _try_parse_clean_field(inner: str) -> str | None:
    """Return the identifier name if ``inner`` is a clean, self-contained field.

    ``inner`` is the text strictly between one outer ``'{'`` and its matching
    top-level ``'}'``, already confirmed by the caller to contain no further
    nested braces. Accepts ``name``, ``name!conv`` (``conv`` one of ``r``/
    ``s``/``a``), ``name:spec``, or ``name!conv:spec``. Anything else (empty,
    non-identifier leading text, invalid conversion flag, trailing garbage
    after a recognized piece) returns ``None``.
    """
    match = re.match(IDENTIFIER_PATTERN_TEXT, inner)
    if not match:
        return None

    name = match.group(0)
    rest = inner[match.end():]

    if rest.startswith("!"):
        if len(rest) < 2 or rest[1] not in ("r", "s", "a"):
            return None
        rest = rest[2:]

    if rest.startswith(":"):
        rest = ""

    if rest:
        return None

    return name


def _normalize_prompt_template(template: str) -> tuple[str, list[str]]:
    """Normalize a raw prompt template and discover its identifier fields.

    Scans ``template`` once. Clean, self-contained identifier fields (see
    ``_try_parse_clean_field``) are kept single-braced and their names are
    discovered in first-appearance order, deduplicated. Every other brace
    region — non-identifier shapes (``{}``, ``{0}``, ``{obj.attr}``,
    ``{x[0]}``), fields nested inside other fields, unbalanced/stray braces —
    is escaped into inert literal text (whole-span, not partially recovered).
    Already-doubled literal escapes (``{{``/``}}``) pass through unchanged,
    so re-normalizing an already-normalized template is a no-op. Never raises
    regardless of input shape.
    """
    i = 0
    n = len(template)
    out: list[str] = []
    discovered: list[str] = []
    seen: set[str] = set()

    while i < n:
        ch = template[i]

        if ch == "{":
            if i + 1 < n and template[i + 1] == "{":
                out.append("{{")
                i += 2
                continue

            # Find the matching top-level close, tracking nested depth.
            depth = 1
            j = i + 1
            nested = False
            while j < n and depth > 0:
                if template[j] == "{":
                    depth += 1
                    nested = True
                elif template[j] == "}":
                    depth -= 1
                j += 1

            if depth != 0:
                # Unbalanced: no matching close found before end of string.
                # Escape just this one stray brace and resume scanning.
                out.append("{{")
                i += 1
                continue

            span = template[i:j]
            if not nested:
                name = _try_parse_clean_field(template[i + 1 : j - 1])
                if name is not None:
                    out.append(span)
                    if name not in seen:
                        seen.add(name)
                        discovered.append(name)
                    i = j
                    continue

            # Not a clean self-contained identifier field: escape the whole
            # span (including any nested content) into inert literal text.
            out.append(span.replace("{", "{{").replace("}", "}}"))
            i = j
            continue

        if ch == "}":
            if i + 1 < n and template[i + 1] == "}":
                out.append("}}")
                i += 2
                continue
            # Orphan close brace with no opener: escape it alone.
            out.append("}}")
            i += 1
            continue

        out.append(ch)
        i += 1

    return "".join(out), discovered


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def extract_io(function: Callable[..., Any]) -> tuple[list[ParamSpec], str]:
    """Extract parameter specifications and return type from a callable.

    Builds an ordered list of ``ParamSpec`` objects from a function's signature.
    Each ``ParamSpec`` is self-sufficient (name, index, kind, type, default,
    description).

    Annotation source: ``get_type_hints(function, include_extras=True)`` is
    attempted first so that ``Annotated[T, "desc"]`` is preserved intact and
    forward references are resolved. On ``NameError`` (unresolvable forward
    reference) the call falls back to ``{}``; each parameter then uses
    ``param.annotation`` from ``inspect.signature`` directly.

    ``Annotated`` handling: ``_unwrap_annotated`` separates the base type from
    metadata. The first ``str`` item in the metadata becomes
    ``ParamSpec.description`` (stripped; empty-after-strip → ``None``).
    ``_format_annotation`` always receives the unwrapped base type.

    Type resolution priority per parameter:
    1. Unwrapped annotation base type if present.
    2. ``type(default)`` if annotation absent but default present.
    3. ``"Any"`` otherwise.

    Parameters
    ----------
    function : Callable
        Any Python callable (function, method, lambda, etc.)

    Returns
    -------
    tuple[list[ParamSpec], str]
        - List of ``ParamSpec`` objects in signature order.
        - Return type as a human-readable string (``Annotated`` unwrapped).

    Raises
    ------
    TypeError
        If ``function`` is not callable.
    """
    if not callable(function):
        raise TypeError(f"extract_io expects a callable, got {type(function)!r}")

    sig = inspect.signature(function)

    # Prefer resolved hints; fall back to {} on unresolvable forward references.
    try:
        hints = get_type_hints(function, include_extras=True)
    except NameError:
        hints = {}

    parameters: list[ParamSpec] = []

    for index, (name, param) in enumerate(sig.parameters.items()):
        kind_name = param.kind.name
        default   = param.default

        # Annotation source: resolved hint > raw sig annotation.
        ann = hints.get(name, param.annotation)
        base_ann, description = _unwrap_annotated(ann)

        # Type resolution: base_ann > type(default) > "Any".
        if base_ann is not inspect._empty:
            raw_type = base_ann
        elif default is not inspect._empty:
            raw_type = type(default)
        else:
            raw_type = inspect._empty

        type_str    = _format_annotation(raw_type)
        default_val = default if default is not inspect._empty else NO_VAL

        parameters.append(ParamSpec(
            name=name,
            index=index,
            kind=kind_name,
            type=type_str,
            default=default_val,
            description=description,
        ))

    # Return type: resolved hint > raw sig annotation; Annotated unwrapped.
    ret_ann = hints.get("return", sig.return_annotation)
    base_ret, _ = _unwrap_annotated(ret_ann)
    return_type = _format_annotation(base_ret)

    return parameters, return_type


def to_paramspec_list(
    schema: Optional[type | list[str] | tuple[str, ...] | set[str] | list[ParamSpec]],
) -> list[ParamSpec]:
    """Normalize supported schema inputs into a fresh canonical ``list[ParamSpec]``.

    Accepted inputs
    ---------------
    - ``None`` -> empty list
    - ``TypedDict`` class
    - ``list[str]``, ``tuple[str, ...]``, or ``set[str]`` using marker grammar:
      - ``"name"`` -> normal parameter, or keyword-only after ``"*"`` / ``"*args"``
      - ``"/"`` -> converts preceding normal parameters to positional-only
      - ``"*"`` -> starts keyword-only section
      - ``"*args"`` -> var positional and starts keyword-only section
      - ``"**kwargs"`` -> var keyword and must be final
    - ``list[ParamSpec]``

    Returns
    -------
    list[ParamSpec]
        Fresh ``ParamSpec`` objects with canonical sequential indices.

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
        _validate_parameter_order(normalized)
        return normalized

    # ------------------------------------------------------------------
    # TypedDict class -> use field annotations as ParamSpec.type
    # ------------------------------------------------------------------
    if _is_typed_dict_class(schema):
        hints = get_type_hints(schema, include_extras=True)
        normalized = []
        for index, (name, annotation) in enumerate(hints.items()):
            base_ann, description = _unwrap_annotated(annotation)
            normalized.append(ParamSpec(
                name=name,
                index=index,
                kind=ParamSpec.POSITIONAL_OR_KEYWORD,
                type=_format_annotation(base_ann),
                default=NO_VAL,
                description=description,
            ))
        _validate_parameter_order(normalized)
        return normalized

    # ------------------------------------------------------------------
    # list[str] / tuple[str, ...] / set[str]
    # Snapshot current iteration order for tuples/sets as provided.
    # ------------------------------------------------------------------
    if isinstance(schema, (list, tuple, set)):
        items = list(schema)

        if not items:
            normalized = []
            _validate_parameter_order(normalized)
            return normalized

        if all(isinstance(item, str) for item in items):
            normalized = _paramspec_list_from_strings(items)
            _validate_parameter_order(normalized)
            return normalized

        if isinstance(schema, list) and all(isinstance(item, ParamSpec) for item in items):
            normalized = [
                ParamSpec(
                    name=item.name,
                    index=index,
                    kind=item.kind,
                    type=item.type,
                    default=item.default,
                    description=item.description,
                )
                for index, item in enumerate(items)
            ]
            _validate_parameter_order(normalized)
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


def is_valid_parameter_order(parameters: list[ParamSpec]) -> bool:
    """Bool predicate for Python-compatible parameter ordering.

    Wraps ``_validate_parameter_order``: returns ``True`` on success,
    ``False`` if a ``SchemaError`` is raised, and propagates ``TypeError``
    to the caller unchanged.

    1. Call ``_validate_parameter_order(parameters)``.
    2. If ``SchemaError`` is raised, return ``False``.
    3. If ``TypeError`` is raised, propagate to caller.
    4. Otherwise return ``True``.
    """
    try:
        _validate_parameter_order(parameters)
        return True
    except SchemaError:
        return False
