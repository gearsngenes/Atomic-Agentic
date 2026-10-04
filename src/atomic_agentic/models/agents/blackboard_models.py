from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from ...constants.agents import (
    ATTR_CALL_ALIAS,
    KWARGS_UNPACK_KEY,
    PY_BUILTIN_ALIAS,
    RETURN_ALIAS,
    RETURN_VALUE_FIELD,
    RHS_ASSIGN_ALIAS,
)
from ...constants.core import IDENTIFIER_PATTERN
from ..results import AtomicResult

__all__ = [
    "ConstantSpec",
    "ToolStatement",
]


@dataclass(frozen=True, slots=True)
class ConstantSpec:
    """
    Read-only named runtime value registered on a ToolAgent.

    Constants are stable symbolic bindings that can be exposed to an LLM by
    name/type/description and later resolved by the ToolAgent runtime. The
    actual value is stored here, but prompt-facing renderers should avoid
    displaying it unless deliberately designed to do so.

    Fields
    ------
    name : str
        Safe constant name. Must be identifier-like: letters/underscore first,
        then letters/numbers/underscore.

    value : Any
        Runtime value bound to this constant.

    description : str | None
        Optional human-readable context for what the constant represents.

    inline_limit : int | None
        Optional character limit for future inline string substitution. None
        means no limit. If provided, must be an int > 0.

    type : str
        Derived automatically from ``type(value).__name__``.
    """

    name: str
    value: Any
    description: str | None = None
    inline_limit: int | None = None
    type: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("ConstantSpec.name must be a non-empty string.")

        normalized_name = self.name.strip()
        if not IDENTIFIER_PATTERN.fullmatch(normalized_name):
            raise ValueError(
                "ConstantSpec.name must be alphanumeric/underscore and not start "
                f"with a digit; got {self.name!r}."
            )

        normalized_description: str | None
        if self.description is None:
            normalized_description = None
        else:
            if not isinstance(self.description, str):
                raise TypeError(
                    "ConstantSpec.description must be a string or None."
                )
            normalized_description = self.description.strip() or None

        if self.inline_limit is not None:
            if type(self.inline_limit) is not int or self.inline_limit <= 0:
                raise ValueError(
                    "ConstantSpec.inline_limit must be None or an int > 0."
                )

        object.__setattr__(self, "name", normalized_name)
        object.__setattr__(self, "description", normalized_description)
        object.__setattr__(self, "type", type(self.value).__name__)

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation of this constant spec."""
        return asdict(self)


@dataclass(slots=True)
class ToolStatement:
    """
    One slot in a per-invocation call/statement sequence -- the single,
    unified representation shared by ``ScriptActAgent`` (built directly from
    real ``ast.parse`` output), ``PlanActAgent``, and ``ReActAgent`` (built
    by ``utils/sigils.py``'s ``translate_calls`` from a JSON wire payload).
    Supersedes the prior per-grammar subclass split
    (``toolstatement-rename`` pass) -- there is now exactly one concrete
    representation, no ABC, no per-grammar hook.

    ``args``/``kwargs`` are always real ``ast.expr`` nodes, regardless of
    origin: for ``ScriptActAgent`` that's unchanged (its parser already
    builds real ast nodes directly); for ``PlanActAgent``/``ReActAgent`` the
    JSON wire payload's plain scalars are translated into ``ast.Constant``/
    ``ast.Name``/``ast.JoinedStr`` nodes by ``translate_calls`` before a
    ``ToolStatement`` is ever constructed. No ``.dependencies``/
    ``.resolved_refs`` field exists here, ever -- dependencies are always
    derived on demand via ``extract_identifiers`` (``utils/agents.py``) over
    the real ``ast.expr`` tree.

    Mutable (not frozen) -- ``result``/``exception`` are populated after
    construction by a future ``act()``-phase caller.

    Fields
    ------
    identifier : str | None
        This slot's bound name. ``None`` for a bare (unassigned) call or a
        terminal ``return``. A JSON-sourced identifier has already been
        normalized (stripped, ``$``-prefix removed, blank-collapsed to
        ``None``) by ``translate_calls`` *before* this constructor ever
        runs; a ``ScriptActAgent``-sourced identifier is always already
        ast-sourced-legal. One uniform legality check applies here either
        way -- no per-grammar leniency hook.

    tool : str
        The call's identity -- a dotted call name, a registered tool's
        alias/``full_name``, or a grammar-specific sentinel (``RHS_ASSIGN_
        ALIAS``/``RETURN_ALIAS``/``PY_BUILTIN_ALIAS``/``ATTR_CALL_ALIAS`` for
        a ``ScriptActAgent``-sourced statement; ``RETURN_ALIAS`` is the only
        sentinel a ``PlanActAgent``/``ReActAgent``-sourced statement ever
        carries). Must be a non-empty string.

    args : tuple[ast.expr, ...]
        Positional call arguments, in source order. Normalized to a tuple
        from whatever tuple/list was supplied. Always real ``ast.expr``
        (or ``ast.Starred``, itself an ``ast.expr`` subtype) nodes.

    kwargs : dict[str, ast.expr]
        Keyword call arguments. Same element-type guarantee as ``args``.

    batch_index : int | None
        Which concurrently-dispatched batch this slot belongs to, stamped
        once by the shared batch compiler (``utils/agents.py``'s
        ``compile_batches``) when the batch closes. ``None`` until then. Not
        defensively validated -- set internally by framework lifecycle code,
        not derived from external/LLM input.

    result : AtomicResult | None
        Full result envelope from a future ``act()``-phase caller. ``None``
        until executed.

    exception : Exception | None
        Live exception object from a failed execution attempt. ``None``
        until a failure occurs. Not stringified.
    """

    identifier: Optional[str]
    tool: str
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)
    batch_index: Optional[int] = None
    result: AtomicResult | None = None
    exception: Exception | None = None

    def __post_init__(self) -> None:
        # 1. identifier: if not None, must be a non-empty, IDENTIFIER_
        # PATTERN-legal string -- hard raise otherwise. Safe unconditionally
        # now: a ScriptActAgent-sourced identifier is always already
        # ast-sourced-legal; a JSON-sourced identifier has already been
        # normalized (stripped, $-prefix removed, blank-collapsed to None)
        # by translate_calls before this constructor ever runs -- no
        # leniency needed here.
        if self.identifier is not None and (
            not isinstance(self.identifier, str)
            or not IDENTIFIER_PATTERN.fullmatch(self.identifier)
        ):
            raise ValueError(
                "ToolStatement.identifier must be None or a non-empty, "
                f"Python-identifier-legal string; got {self.identifier!r}."
            )

        # 2. tool must be a non-empty string.
        if not isinstance(self.tool, str) or not self.tool.strip():
            raise ValueError(
                f"ToolStatement.tool must be a non-empty string; got {self.tool!r}."
            )

        # 3. args must be a tuple or list (no per-element validation).
        # Normalized to a tuple below.
        if isinstance(self.args, (str, bytes)) or not isinstance(self.args, (tuple, list)):
            raise TypeError(
                f"ToolStatement.args must be a tuple or list; "
                f"got {type(self.args).__name__!r}."
            )
        self.args = tuple(self.args)

        # 4. kwargs must be a dict.
        if not isinstance(self.kwargs, dict):
            raise TypeError(
                f"ToolStatement.kwargs must be a dict; got {type(self.kwargs).__name__!r}."
            )

        # 5. batch_index/result/exception are set internally by framework
        # lifecycle code, not derived from external/LLM input -- not
        # defensively validated here, per 01-overview.md Section 4's
        # boundary-only-validation rule.

    def to_dict(self) -> dict[str, Any]:
        """
        Return the explicit serialized dictionary representation, for
        debugging/observability only -- never used to reconstruct or
        re-plan. Inlines the old per-subclass ``_render_value`` hook (only
        one behavior exists now that args/kwargs are always ``ast.expr``):
        render a value via ``ast.unparse(value)`` if it's an ``ast.expr``,
        else pass it through unchanged.
        """
        def render_value(value: Any) -> Any:
            return ast.unparse(value) if isinstance(value, ast.expr) else value

        return {
            "identifier": self.identifier,
            "tool": self.tool,
            "args": [render_value(value) for value in self.args],
            "kwargs": {key: render_value(value) for key, value in self.kwargs.items()},
            "batch_index": self.batch_index,
            "result": self.result.to_dict() if self.result is not None else None,
            "exception": repr(self.exception) if self.exception is not None else None,
        }

    def to_code(self) -> str:
        """
        Render this statement as one line of real Python source -- verbatim
        port of the prior per-grammar ``to_code()`` body. No behavior
        change for any ``ScriptActAgent``-sourced statement. For a
        ``PlanActAgent``/``ReActAgent``-sourced statement (``tool`` is a
        real registered tool or ``RETURN_ALIAS``, ``args``/``kwargs`` are
        ``Name``/``Constant``/``JoinedStr`` only), this produces real Python
        source with zero extra code -- the renderer only ever inspects
        ``ast.expr``-ness, never which grammar produced the node.

        ``RETURN_ALIAS`` needs its value read from either ``kwargs`` or
        ``args``, not ``kwargs`` alone: ``RETURN_TOOL_NAME`` (``ReActAgent``'s
        real, dispatched ``return`` tool) and ``RETURN_ALIAS``
        (``PlanActAgent``'s synthesized, non-dispatched sentinel) are the
        identical string ``"return"`` (see ``agents/react.py``'s own module
        docstring). ``PlanActAgent``'s sentinel is always framework-built
        with ``kwargs={"val": ...}``, but ``ReActAgent``'s real call is
        model-authored like any other tool call -- its one argument may be
        given positionally (``"name": null``) just as legally as by keyword,
        landing in ``args`` instead.
        """

        def render_value(value: Any) -> str:
            return ast.unparse(value) if isinstance(value, ast.expr) else repr(value)

        if self.tool == RETURN_ALIAS:
            value = self.kwargs[RETURN_VALUE_FIELD] if RETURN_VALUE_FIELD in self.kwargs else self.args[0]
            return f"return {render_value(value)}"

        prefix = f"{self.identifier} = " if self.identifier is not None else ""
        if self.tool == RHS_ASSIGN_ALIAS:
            return f"{prefix}{render_value(self.kwargs['val'])}"

        # A py_builtin slot renders back as the original natural call syntax
        # (`len(x)`), never the internal rewritten form (`py_builtin('len',
        # x)`) -- unsplice the builtin name before falling into the same
        # generic rendering as any real tool call. An attr_call slot gets the
        # same treatment: `obj.method(args)`, not the internal (obj, "method",
        # *args) shape.
        if self.tool == PY_BUILTIN_ALIAS:
            call_name, call_args = self.args[0], self.args[1:]
        elif self.tool == ATTR_CALL_ALIAS:
            call_name, call_args = f"{render_value(self.args[0])}.{self.args[1]}", self.args[2:]
        else:
            call_name, call_args = self.tool, self.args

        positional_tokens = [
            f"*{render_value(entry.value)}" if isinstance(entry, ast.Starred) else render_value(entry)
            for entry in call_args
        ]
        keyword_tokens = [
            f"**{render_value(value)}" if name == KWARGS_UNPACK_KEY else f"{name}={render_value(value)}"
            for name, value in self.kwargs.items()
        ]
        args_source = ", ".join(positional_tokens + keyword_tokens)
        return f"{prefix}{call_name}({args_source})"
