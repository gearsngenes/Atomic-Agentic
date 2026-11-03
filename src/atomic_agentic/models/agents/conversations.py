from __future__ import annotations

import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any

from ...constants.agents import RENDER_MODES
from .context import ContextVariable
from .records import AgentRecord

__all__ = ["Conversation"]


@dataclass(slots=True)
class Conversation:
    """
    One conversation's forward-ordered record history and durable context.

    Plain data container, no Agent-registry awareness -- mirrors
    AgentRecord's own stance (a dataclass with no owning-Agent awareness).
    ``records`` is a mutable list whose contents are appended to in place by
    ``commit`` and read via direct indexing/slicing elsewhere. The
    dataclass is not frozen so the preview limits can be assigned directly
    (``conv.response_preview_limit = 500``); ``agent_type``, ``records`` and
    ``context`` are not meant to be reassigned, and ``context`` changes only
    through its mutators.

    A conversation has no name: the key under which an Agent registers it is
    the only name, and one object may sit under different keys in different
    agents. The object may be shared by several agents of the exact same
    ``agent_type`` (``Agent.import_conversation`` registers the same object,
    never a copy); ``slice`` is how an independent copy is made. The context
    mutators, ``commit`` and ``slice`` each run under the conversation's own
    ``_lock`` (a re-entrant lock, never compared, copied or shown in the
    repr), so they are atomic with respect to one another.

    Equality is by value over every field (context values are compared with
    ``==``, so a value with an unusual ``__eq__`` surfaces naturally).
    Instances are unhashable: the contents change over time.

    Fields
    ------
    agent_type:
        The ``Agent`` class this conversation was created for; required, no
        default. ``Agent.import_conversation`` accepts a conversation only on
        an exact match. Trusted -- records are not re-validated against it.
        (Typed ``type`` because the model layer never imports ``Agent``.)
    records:
        Forward-ordered (oldest-first) list of ``AgentRecord``. A single
        ``AgentRecord`` may legitimately appear in more than one
        ``Conversation``'s list (``slice`` is a reference-copy, not a deep
        copy).
    context:
        Name -> ``ContextVariable``, this conversation's durable context.
        Always a read-only ``MappingProxyType`` view: ``conv.context[k]``
        reads, ``conv.context[k] = x`` raises ``TypeError``. Storage is
        copy-on-write -- every mutator below builds a new dict and swaps in
        a new view, so a previously obtained view never changes. Constructor
        input is copied, never aliased; the frozen ``ContextVariable``
        objects are shared and stored values are shared by reference, like
        reference-copied records.
    response_preview_limit, prompt_preview_limit, context_preview_limit:
        Character budgets applied when history responses, history prompts and
        context-variable values are rendered into a prompt. ``None`` is
        unlimited, ``0`` hides the text, a positive ``int`` truncates. Owned
        by the conversation (not the agent), copied by ``slice``, and
        validated on every assignment, including construction (``None`` or an
        ``int >= 0``; ``bool`` is rejected).
    render_mode:
        How prior responses are formatted when this conversation's history is
        replayed into a prompt: one of ``RENDER_MODES``, default ``"s"``
        (``"j"`` suits structured-output agents). Prompts are always plain
        strings. Unrelated to a ``ContextVariable``'s own ``render_mode``.
        Validated on every assignment (``ValueError``) and copied by
        ``slice``.
    """

    agent_type: type
    records: list[AgentRecord] = field(default_factory=list)
    context: Mapping[str, ContextVariable] = field(default_factory=dict)
    response_preview_limit: int | None = None
    prompt_preview_limit: int | None = None
    context_preview_limit: int | None = None
    render_mode: str = "s"
    _lock: Any = field(
        default_factory=threading.RLock, init=False, repr=False, compare=False
    )

    def __setattr__(self, name: str, value: Any) -> None:
        if name.endswith("_preview_limit") and value is not None and (
            type(value) is not int or value < 0
        ):
            raise ValueError(
                f"Conversation.{name} must be None or an int >= 0, got {value!r}."
            )
        if name == "render_mode" and value not in RENDER_MODES:
            raise ValueError(
                f"Conversation.render_mode must be one of {RENDER_MODES}, got {value!r}."
            )
        object.__setattr__(self, name, value)

    def __post_init__(self) -> None:
        # Shape-only validation -- element-type checking of each record is
        # AgentRecord's own construction's job, not this container's.
        if not isinstance(self.agent_type, type):
            raise TypeError(
                f"Conversation.agent_type must be a class, got {self.agent_type!r}."
            )
        if not isinstance(self.records, list):
            raise ValueError(
                f"Conversation.records must be a list, got {type(self.records)!r}."
            )
        if not isinstance(self.context, Mapping):
            raise ValueError(
                f"Conversation.context must be a Mapping, got {type(self.context)!r}."
            )
        for key, variable in self.context.items():
            if not isinstance(variable, ContextVariable):
                raise TypeError(
                    f"Conversation.context[{key!r}] must be a ContextVariable, "
                    f"got {type(variable)!r}."
                )
            if key != variable.name:
                raise ValueError(
                    f"Conversation.context key {key!r} must equal its "
                    f"variable's name {variable.name!r}."
                )
        self._swap(dict(self.context))

    def _swap(self, new: dict[str, ContextVariable]) -> None:
        object.__setattr__(self, "context", MappingProxyType(new))

    def remember(
        self,
        key: str,
        value: Any,
        description: str | None = None,
        *,
        readonly: bool = False,
        render_mode: str = "s",
        preview_limit: int | None = None,
    ) -> None:
        """
        Add a NEW variable. Raises ``ValueError`` if ``key`` already exists
        (readonly or not); change an existing one with ``update`` /
        ``describe``, or ``forget`` it first.
        """
        variable = ContextVariable(
            name=key,
            value=value,
            description=description,
            readonly=readonly,
            render_mode=render_mode,
            preview_limit=preview_limit,
        )
        with self._lock:
            if variable.name in self.context:
                raise ValueError(f"Context variable {variable.name!r} already exists.")
            self._swap({**self.context, variable.name: variable})

    def update(self, key: str, value: Any) -> bool:
        """
        Replace only ``key``'s value, keeping its description, flags and
        limits. Raises ``KeyError`` on an unknown key. A readonly variable is
        left untouched: returns ``False``; otherwise ``True``.
        """
        with self._lock:
            existing = self.context[key]
            if existing.readonly:
                return False
            self._swap({**self.context, key: replace(existing, value=value)})
            return True

    def describe(self, key: str, description: str | None) -> bool:
        """
        Replace only ``key``'s description, keeping its value and flags.
        Raises ``KeyError`` on an unknown key. A readonly variable is left
        untouched: returns ``False``; otherwise ``True``.
        """
        with self._lock:
            existing = self.context[key]
            if existing.readonly:
                return False
            self._swap({**self.context, key: replace(existing, description=description)})
            return True

    def forget(self, key: str) -> None:
        """Remove ``key``. Raises ``KeyError`` on an unknown key; ``readonly`` has no effect."""
        with self._lock:
            if key not in self.context:
                raise KeyError(key)
            new = dict(self.context)
            del new[key]
            self._swap(new)

    def remember_many(self, variables: Iterable[ContextVariable]) -> None:
        """
        Atomic add of several NEW variables. Raises on a duplicate name or a
        name that already exists, leaving the context untouched.
        """
        items = list(variables)
        with self._lock:
            seen: set[str] = set()
            for variable in items:
                if not isinstance(variable, ContextVariable):
                    raise TypeError(
                        f"remember_many expects ContextVariable items, got {type(variable)!r}."
                    )
                if variable.name in seen:
                    raise ValueError(f"Duplicate context variable name {variable.name!r}.")
                if variable.name in self.context:
                    raise ValueError(f"Context variable {variable.name!r} already exists.")
                seen.add(variable.name)
            self._swap({**self.context, **{v.name: v for v in items}})

    def forget_many(self, keys: Iterable[str]) -> None:
        """Atomic removal. Raises ``KeyError`` if any key is unknown, removing nothing."""
        keys = list(keys)
        with self._lock:
            missing = [k for k in keys if k not in self.context]
            if missing:
                raise KeyError(missing)
            drop = set(keys)
            self._swap({k: v for k, v in self.context.items() if k not in drop})

    def slice(self, start: int | None = None, end: int | None = None) -> Conversation:
        """
        A new, independent Conversation over ``records[start:end]``.

        Plain Python slice semantics (``end`` exclusive, negatives allowed,
        out-of-range clamped, an empty result is legal; no step). Records are
        reference-copied; context is copied into a distinct container
        (``ContextVariable``s shared, values by reference) and is the LATEST
        context, not its state at the sliced records. The preview limits and
        ``render_mode`` are copied; the new object has its own lock. Never
        mutates ``self``.
        """
        with self._lock:
            records = self.records[start:end]
            return Conversation(
                agent_type=self.agent_type,
                records=records,
                context=self.context,
                response_preview_limit=self.response_preview_limit,
                prompt_preview_limit=self.prompt_preview_limit,
                context_preview_limit=self.context_preview_limit,
                render_mode=self.render_mode,
            )

    def commit(
        self, record: AgentRecord, task_context: Mapping[str, ContextVariable]
    ) -> None:
        """
        Atomically commit one finished call.

        1. Append ``record`` to ``records``.
        2. Append it to ``record.prev.children`` when ``prev`` is set.
        3. Context write-back, in one swap: for every name in ``task_context``
           that is in ``self.context`` and not ``readonly`` there, the VALUE
           is replaced (description and flags are kept). Scratch keys, keys
           forgotten mid-call and readonly destinations are skipped silently;
           every eligible key is written, changed or not.

        No type validation; ``record_enabled`` is the caller's concern.
        """
        with self._lock:
            self.records.append(record)
            if record.prev is not None:
                record.prev.children.append(record)

            current = self.context
            self._swap({
                **current,
                **{
                    name: replace(current[name], value=variable.value)
                    for name, variable in task_context.items()
                    if name in current and not current[name].readonly
                },
            })
