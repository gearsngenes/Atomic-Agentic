from __future__ import annotations

from typing import Any, Callable

from ..invokables.atomic_invokable import AtomicInvokable
from ..invokables.prompt_template import PromptTemplate
from .toolification import toolify

__all__ = ["to_invokable"]


def to_invokable(
    obj: str | Callable[..., Any] | AtomicInvokable,
    *,
    name: str,
    description: str,
) -> AtomicInvokable:
    """
    Coerce one construction-time value into an invokable.

    1. An ``AtomicInvokable`` is returned unchanged (no wrapper, even for a
       ``Tool``; ``name``/``description`` are ignored).
    2. A ``str`` becomes a ``PromptTemplate`` with discovered parameters. The
       string is used verbatim (no ``.strip()``); a blank string raises
       ``PromptTemplate``'s ``ValueError``.
    3. Any other callable goes through ``toolify`` with ``name`` and
       ``description``.
    4. Anything else (``None`` included) raises ``TypeError`` naming ``name``
       and the type.

    Invokables are callable, so they are tested first; ``str`` is not callable.
    Used for instructions and the pre/post stages; an agent is an invokable, so
    recursion is allowed and unguarded.
    """
    if isinstance(obj, AtomicInvokable):
        return obj
    if isinstance(obj, str):
        return PromptTemplate(template=obj, name=name, description=description)
    if callable(obj):
        return toolify(obj, name=name, description=description)
    raise TypeError(
        f"{name} must be a str, callable or AtomicInvokable; got {type(obj).__name__}."
    )