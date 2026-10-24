from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any

from .context import ContextVariable
from .records import AgentRecord

__all__ = ["Conversation"]


@dataclass(frozen=True, slots=True)
class Conversation:
    """
    One named conversation's forward-ordered record history and durable
    context.

    Plain data container, no Agent-registry awareness -- mirrors
    AgentRecord's own stance (a dataclass with no owning-Agent awareness).
    ``records`` is a mutable list despite the frozen dataclass -- same
    treatment as ``AgentRecord.children``/``.inputs``: the field bindings are
    frozen; the list's own contents are appended to in place by
    ``Agent.record`` and read via direct indexing/slicing elsewhere.

    Equality is by value over every field (context values are compared with
    ``==``, so a value with an unusual ``__eq__`` surfaces naturally).
    Instances are unhashable: the contents change over time.

    Fields
    ------
    name:
        This conversation's key, as stored in ``Agent._conversations``. Not
        itself pattern-validated here (alphanumeric/underscore,
        no-trailing-``_digits`` shape) -- that's a user-facing-boundary
        concern owned by ``Agent.create_conversation``/``clone_conversation``,
        not this passive container's constructor.
    agent_type:
        The ``Agent`` class this conversation was created for; required, no
        default. ``Agent.import_conversation`` accepts a conversation only on
        an exact match. Trusted -- records are not re-validated against it.
        (Typed ``type`` because the model layer never imports ``Agent``.)
    records:
        Forward-ordered (oldest-first) list of ``AgentRecord``. A single
        ``AgentRecord`` may legitimately appear in more than one
        ``Conversation``'s list (cloning is a reference-copy, not a deep
        copy).
    context:
        Name -> ``ContextVariable``, this conversation's durable context.
        Always a read-only ``MappingProxyType`` view: ``conv.context[k]``
        reads, ``conv.context[k] = x`` raises ``TypeError``. Storage is
        copy-on-write -- every mutator below builds a new dict and swaps in
        a new view, so a previously obtained view never changes. The
        mutators are NOT thread-safe on their own; ``Agent``'s wrappers hold
        ``_invoke_lock``. Constructor input is copied, never aliased; the
        frozen ``ContextVariable`` objects are shared and stored values are
        shared by reference, like reference-copied records.
    """

    name: str
    agent_type: type
    records: list[AgentRecord] = field(default_factory=list)
    context: Mapping[str, ContextVariable] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Shape-only validation -- element-type checking of each record is
        # AgentRecord's own construction's job, not this container's.
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(
                f"Conversation.name must be a non-empty str, got {self.name!r}."
            )
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

    def _check_writable(self, name: str) -> None:
        existing = self.context.get(name)
        if existing is not None and existing.readonly:
            raise ValueError(
                f"Context variable {name!r} is readonly and cannot be overwritten."
            )

    def remember(
        self,
        key: str,
        value: Any,
        description: str | None = None,
        *,
        readonly: bool = False,
    ) -> None:
        """
        Upsert ``key``. A full replace: ``description=None`` clears any old
        description. Raises if the existing variable is readonly.
        """
        variable = ContextVariable(
            name=key, value=value, description=description, readonly=readonly
        )
        self._check_writable(variable.name)
        self._swap({**self.context, variable.name: variable})

    def describe(self, key: str, description: str | None) -> None:
        """
        Replace only ``key``'s description, keeping its value and readonly
        flag. Raises ``KeyError`` on an unknown key, ``ValueError`` if the
        variable is readonly.
        """
        existing = self.context[key]
        self._check_writable(key)
        self._swap({**self.context, key: replace(existing, description=description)})

    def forget(self, key: str) -> None:
        """Remove ``key``. Raises ``KeyError`` on an unknown key; ``readonly`` has no effect."""
        if key not in self.context:
            raise KeyError(key)
        new = dict(self.context)
        del new[key]
        self._swap(new)

    def remember_many(self, variables: Iterable[ContextVariable]) -> None:
        """
        Atomic wholesale replace of each variable under its own name. Raises
        on a duplicate name or a readonly collision, leaving the context
        untouched.
        """
        items = list(variables)
        seen: set[str] = set()
        for variable in items:
            if not isinstance(variable, ContextVariable):
                raise TypeError(
                    f"remember_many expects ContextVariable items, got {type(variable)!r}."
                )
            if variable.name in seen:
                raise ValueError(f"Duplicate context variable name {variable.name!r}.")
            self._check_writable(variable.name)
            seen.add(variable.name)
        self._swap({**self.context, **{v.name: v for v in items}})

    def forget_many(self, keys: Iterable[str]) -> None:
        """Atomic removal. Raises ``KeyError`` if any key is unknown, removing nothing."""
        keys = list(keys)
        missing = [k for k in keys if k not in self.context]
        if missing:
            raise KeyError(missing)
        drop = set(keys)
        self._swap({k: v for k, v in self.context.items() if k not in drop})
