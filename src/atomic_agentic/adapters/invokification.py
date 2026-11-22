from __future__ import annotations

from typing import Any, Callable, Sequence

from ..a2a import A2AClientHub
from ..invokables.atomic_invokable import AtomicInvokable
from ..invokables.prompt_template import PromptTemplate
from ..mcp import MCPClientHub
from .toolification import batch_toolify, toolify

__all__ = ["to_invokable", "batch_to_invokable"]


def to_invokable(
    obj: str | Callable[..., Any] | AtomicInvokable,
    *,
    name: str | None = None,
    description: str | None = None,
) -> AtomicInvokable:
    """
    Coerce one value into an invokable.

    1. An ``AtomicInvokable`` is returned unchanged (no wrapper, even for a
       ``Tool``; ``name``/``description`` are ignored).
    2. A ``str`` becomes a ``PromptTemplate`` with discovered parameters, used
       verbatim (no ``.strip()``; a blank string raises ``PromptTemplate``'s
       ``ValueError``). A template must be named: ``TypeError`` if ``name`` or
       ``description`` is ``None``, naming the missing argument(s).
    3. Any other callable goes through ``toolify`` with ``name`` and
       ``description``; a ``None`` falls back to the callable's ``__name__``
       / ``__doc__`` (``toolify``'s own inference).
    4. Anything else (``None`` included) raises ``TypeError``: with a ``name``,
       "<name> must be a str, callable or AtomicInvokable; got <type>."; without
       one, "unsupported component type <type!r>. Expected str, callable or
       AtomicInvokable."

    Invokables are callable, so they are tested first; ``str`` is not callable.
    Used for instructions, the pre/post stages and composite entries (names
    given), and for toolbox items (names inferred). An agent is an invokable,
    so recursion is allowed and unguarded.
    """
    if isinstance(obj, AtomicInvokable):
        return obj
    if isinstance(obj, str):
        missing = [
            label
            for label, value in (("name", name), ("description", description))
            if value is None
        ]
        if missing:
            raise TypeError(
                "a str becomes a PromptTemplate and needs both `name` and "
                f"`description`; missing: {', '.join(missing)}."
            )
        return PromptTemplate(template=obj, name=name, description=description)
    if callable(obj):
        return toolify(obj, name=name, description=description)
    if name is None:
        raise TypeError(
            f"unsupported component type {type(obj).__name__!r}. "
            "Expected str, callable or AtomicInvokable."
        )
    raise TypeError(
        f"{name} must be a str, callable or AtomicInvokable; got {type(obj).__name__}."
    )


def batch_to_invokable(
    objs: Sequence[Any] | None = None,
) -> list[AtomicInvokable]:
    """
    Convert every item of ``objs`` into an invokable, in order; the batch form
    of ``to_invokable`` for a toolbox.

    1. An ``MCPClientHub`` or ``A2AClientHub`` expands in place through
       ``batch_toolify([hub])`` (MCP in ``list_tools()`` order; A2A skills
       first, then the one generic tool).
    2. A ``str`` raises ``TypeError`` naming its index: nothing can name the
       template in a batch.
    3. Any other item goes through ``to_invokable(item)`` with no name or
       description (an invokable passes through unchanged; a callable becomes a
       ``Tool`` named from ``__name__``). A ``TypeError`` from it is re-raised
       with the index appended; a ``ToolDefinitionError`` from ``toolify``
       propagates unchanged.
    4. ``None`` is an empty batch. Returns the flat list.
    """
    converted: list[AtomicInvokable] = []
    for index, obj in enumerate(objs or []):
        if isinstance(obj, (MCPClientHub, A2AClientHub)):
            converted.extend(batch_toolify([obj]))
        elif isinstance(obj, str):
            raise TypeError(
                f"unsupported component type 'str' at index {index}: "
                "a str cannot be named in a batch."
            )
        else:
            try:
                converted.append(to_invokable(obj))
            except TypeError as exc:
                raise TypeError(f"{exc} (at index {index})") from exc
    return converted
