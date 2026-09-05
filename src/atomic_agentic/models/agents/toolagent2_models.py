from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ...constants.core import IDENTIFIER_PATTERN
from ..results import AtomicResult

__all__ = [
    "BlackboardSlotV2",
]


@dataclass(slots=True)
class BlackboardSlotV2:
    """
    One slot in a ToolAgent2 subtask's per-invocation blackboard.

    Every generated statement normalizes to this one call-shaped record --
    a real tool call (``tool`` = the call's dotted name) or a bare
    expression (``tool`` = ``RHS_ASSIGN_ALIAS``, ``args = {"val": <expr>}``).
    ``args`` values are mixed: a dependency-free expression is evaluated
    eagerly at parse time and stored as a plain Python value; an expression
    referencing another slot's identifier is stored unresolved as the raw
    ``ast.expr`` node, pending a future ``resolve_slot_args`` call. Mutable
    (not frozen) -- ``result``/``exception`` are populated after
    construction by a future ``act()``-phase caller.

    Fields
    ------
    identifier : str
        This slot's bound name -- the statement's LHS, or a synthesized
        ``_HOIST_N`` name for an auto-hoisted nested call.

    tool : str
        Dotted call name (e.g. ``"Type.namespace.name"``), or
        ``RHS_ASSIGN_ALIAS`` (``"rhs_assign"``) for a bare-expression
        statement. ``rhs_assign`` calls never count against
        tools-used/tool-call budget accounting (enforced by a future,
        out-of-scope caller).

    args : dict[str, Any]
        Keyword arguments (real tool call) or ``{"val": <expr>}``
        (``rhs_assign``). Each value is either an already-resolved Python
        literal or an unresolved ``ast.expr`` node -- see class docstring.

    awaited : bool
        True iff this statement's RHS was ``await <call>`` before hoisting
        (this slot's own tool call must be a real, unhoisted top-level call
        for this to ever be True). Forward-barrier batch-scheduling
        semantics; interpreted by a future, out-of-scope
        ``prepare()``/``act()`` caller, not by anything in this slice.

    result : AtomicResult | None
        Full result envelope from a future ``act()``-phase caller. ``None``
        until executed.

    exception : Exception | None
        Live exception object from a failed execution attempt, or a
        ``DependencyFailedError`` if a dependency itself failed. ``None``
        until a failure occurs. Not stringified.
    """

    identifier: str
    tool: str
    args: dict[str, Any]
    awaited: bool = False
    result: AtomicResult | None = None
    exception: Exception | None = None

    def __post_init__(self) -> None:
        # 1. identifier must be a non-empty, Python-identifier-legal string.
        if not isinstance(self.identifier, str) or not IDENTIFIER_PATTERN.fullmatch(
            self.identifier
        ):
            raise ValueError(
                "BlackboardSlotV2.identifier must be a non-empty, "
                f"Python-identifier-legal string; got {self.identifier!r}."
            )

        # 2. tool must be a non-empty string (dotted call names are not
        # bare identifiers, so IDENTIFIER_PATTERN does not apply here).
        if not isinstance(self.tool, str) or not self.tool.strip():
            raise ValueError(
                f"BlackboardSlotV2.tool must be a non-empty string; got {self.tool!r}."
            )

        # 3. args must be a dict.
        if not isinstance(self.args, dict):
            raise TypeError(
                f"BlackboardSlotV2.args must be a dict; got {type(self.args).__name__!r}."
            )

        # 4. awaited must be exactly a bool (bool is an int subclass; guard
        # against e.g. 0/1 silently passing an isinstance check).
        if type(self.awaited) is not bool:
            raise TypeError(
                f"BlackboardSlotV2.awaited must be a bool; got {type(self.awaited).__name__!r}."
            )

        # 5. result/exception are set internally by future, not-yet-built
        # lifecycle code, not derived from external/LLM input -- not
        # defensively validated here, per 01-overview.md Section 4's
        # boundary-only-validation rule.
