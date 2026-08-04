"""Callable/AtomicInvokable schema introspection.

Lives in ``core/`` rather than ``utils/`` because ``extract_io`` needs to
recognize ``AtomicInvokable`` directly, and ``utils/`` sits *below* ``core/``
in the layered dependency topology (``core/Invokable.py`` already imports
from ``utils/parameters.py``) — an ``AtomicInvokable``-aware ``extract_io``
living in ``utils/`` would be a circular import, not just a layering
preference.
"""
from __future__ import annotations

import inspect
from typing import Any, Callable, get_type_hints

from ..constants.core import NO_VAL
from ..models.parameters import ParamSpec
from ..utils.parameters import _format_annotation, _unwrap_annotated
from .Invokable import AtomicInvokable

__all__ = ["extract_io"]


def extract_io(
    function: AtomicInvokable | Callable[..., Any],
) -> tuple[list[ParamSpec], str]:
    """Extract parameter specifications and return type from a callable or
    an ``AtomicInvokable``.

    Annotation source: ``get_type_hints(function, include_extras=True)`` is
    attempted first so that ``Annotated[T, "desc"]`` is preserved intact and
    forward references are resolved. On ``NameError`` (unresolvable forward
    reference) the call falls back to ``{}``; each parameter then uses
    ``param.annotation`` from ``inspect.signature`` directly.

    ``Annotated`` handling: ``_unwrap_annotated`` separates the base type from
    metadata. The first ``str`` item in the metadata becomes
    ``ParamSpec.description`` (stripped; empty-after-strip -> ``None``).
    ``_format_annotation`` always receives the unwrapped base type.

    Type resolution priority per parameter:
    1. Unwrapped annotation base type if present.
    2. ``type(default)`` if annotation absent but default present.
    3. ``"Any"`` otherwise.

    Parameters
    ----------
    function : AtomicInvokable | Callable
        An ``AtomicInvokable`` instance, or any Python callable (function,
        method, lambda, etc.)

    Returns
    -------
    tuple[list[ParamSpec], str]
        - List of ``ParamSpec`` objects in signature order.
        - Return type as a human-readable string (``Annotated`` unwrapped).

    Raises
    ------
    TypeError
        If ``function`` is neither an ``AtomicInvokable`` nor callable.
    """
    if isinstance(function, AtomicInvokable):
        return function.parameters, function.return_type

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
