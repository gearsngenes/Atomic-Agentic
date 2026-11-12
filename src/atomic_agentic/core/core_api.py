"""Callable/AtomicInvokable schema introspection and cross-source
parameter comparison.

Lives in ``core/`` rather than ``utils/`` because these functions need to
recognize ``AtomicInvokable`` directly (via ``extract_io``), and ``utils/``
sits *below* ``core/`` in the layered dependency topology
(``core/Invokable.py`` already imports from ``utils/parameters.py``) — an
``AtomicInvokable``-aware function living in ``utils/`` would be a
circular import, not just a layering preference. ``extract_io`` was here
first (a25); ``parameter_overlap``/``parameter_collisions`` joined it here
for the identical reason once their public signature moved from
``Sequence[Sequence[ParamSpec]]`` to ``Sequence[Callable]``.
"""
from __future__ import annotations

import inspect
from typing import Any, Callable, Sequence, get_type_hints

from ..constants.core import NO_VAL
from ..models.parameters import ParamSpec
from ..utils.parameters import (
    _format_annotation_tuple,
    _unwrap_annotated,
    build_parameter_reports,
)
from .Invokable import AtomicInvokable

__all__ = ["extract_io", "parameter_overlap", "parameter_collisions"]


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
    ``_format_annotation_tuple`` always receives the unwrapped base type.

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

        type_tuple  = _format_annotation_tuple(raw_type)
        default_val = default if default is not inspect._empty else NO_VAL

        parameters.append(ParamSpec(
            name=name,
            index=index,
            kind=kind_name,
            type=type_tuple,
            default=default_val,
            description=description,
        ))

    # Return type: resolved hint > raw sig annotation; Annotated unwrapped.
    # Stays a single joined string, not a tuple -- nothing compares it
    # structurally the way parameter types get merged/reconciled.
    ret_ann = hints.get("return", sig.return_annotation)
    base_ret, _ = _unwrap_annotated(ret_ann)
    return_type = " | ".join(_format_annotation_tuple(base_ret))

    return parameters, return_type


def parameter_overlap(
    sources: Sequence[Callable], *, unanimous_only: bool = False
) -> list[str]:
    """Names shared across ``sources`` that are jointly compatible.

    1. Extract each source's own parameter list via ``extract_io`` (covers
       plain callables and ``AtomicInvokable`` uniformly -- the latter
       implements ``__call__``).
    2. Build one ``ParameterReport`` per distinct name via
       ``build_parameter_reports``.
    3. A name qualifies if declared by at least ``len(sources)`` sources
       when ``unanimous_only`` is ``True``, or at least 2 sources
       otherwise, AND its report has a non-empty ``witness_types`` and
       ``kind_compatible`` is ``True``.
    4. Return qualifying names in ``reports``' own order (first-seen
       across sources).

    Complements ``parameter_collisions`` -- together they partition every
    name meeting the same shared-ness threshold; a name below that
    threshold (too few declaring sources) is neither.
    """
    specs = [extract_io(s)[0] for s in sources]
    reports = build_parameter_reports(specs)
    threshold = len(sources) if unanimous_only else 2
    return [
        r.parameter_name
        for r in reports
        if sum(1 for o in r.observations if o is not None) >= threshold
        and r.witness_types
        and r.kind_compatible
    ]


def parameter_collisions(sources: Sequence[Callable]) -> list[str]:
    """Names shared across 2+ of ``sources`` that are NOT jointly compatible.

    Same extraction/report steps as ``parameter_overlap``, inverted
    qualifying condition, and deliberately no ``unanimous_only`` --
    this is a safety check; narrowing it to only unanimous disagreement
    would hide a real collision between just two of many sources.
    """
    specs = [extract_io(s)[0] for s in sources]
    reports = build_parameter_reports(specs)
    return [
        r.parameter_name
        for r in reports
        if sum(1 for o in r.observations if o is not None) >= 2
        and not (r.witness_types and r.kind_compatible)
    ]
