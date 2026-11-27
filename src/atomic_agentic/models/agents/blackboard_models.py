from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any, Optional

from ...constants.agents import (
    ATTR_CALL_ALIAS,
    KWARGS_UNPACK_KEY,
    PY_BUILTIN_ALIAS,
    RETURN_ALIAS,
    RETURN_VALUE_FIELD,
    RHS_ASSIGN_ALIAS,
)
from ...constants.core import IDENTIFIER_PATTERN, NO_VAL
from ...utils.core import render_value as _render
from ..results import AtomicResult

__all__ = ["ToolStatement", "Step", "Statement"]


@dataclass(slots=True)
class ToolStatement:
    """
    One slot in a per-invocation call/statement sequence -- the single,
    unified representation shared by ``PlanActAgent`` and ``ReActAgent``
    (built from real ``ast.parse`` output by ``utils/script.py``). Exactly one
    concrete representation, no ABC, no per-grammar hook.

    ``args``/``kwargs`` are real ``ast.expr`` nodes, except the plain ``str``
    values ``rewrite_builtin_calls`` splices in. No ``.dependencies``/
    ``.resolved_refs`` field exists here, ever -- dependencies are always
    derived on demand via ``extract_identifiers`` (``utils/agents.py``) over
    the real ``ast.expr`` tree.

    Mutable (not frozen) -- ``result``/``exception`` are populated after
    construction by a future ``act()``-phase caller.

    Fields
    ------
    identifier : str | None
        This slot's bound name. ``None`` for a bare (unassigned) call or a
        terminal ``return``. An identifier is always already ast-sourced-legal; one
        legality check applies here, no per-grammar leniency hook.

    tool : str
        The call's identity -- a dotted call name, a registered tool's
        name, or a grammar-specific sentinel (``RHS_ASSIGN_
        ALIAS``/``RETURN_ALIAS``/``PY_BUILTIN_ALIAS``/``ATTR_CALL_ALIAS``). Must be a non-empty string.

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
        # PATTERN-legal string -- hard raise otherwise. An identifier is
        # always already ast-sourced-legal, so no leniency is needed here.
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
        re-plan. Renders a value via ``ast.unparse(value)`` if it's an
        ``ast.expr``, else passes it through unchanged -- the one rendering
        rule needed now that args/kwargs are always ``ast.expr``.
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
        Render this statement as one line of real Python source.
        """

        def render_value(value: Any) -> str:
            return ast.unparse(value) if isinstance(value, ast.expr) else repr(value)

        if self.tool == RETURN_ALIAS:
            value = self.kwargs[RETURN_VALUE_FIELD]
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


@dataclass(frozen=True, slots=True)
class Step:
    """
    One sub-operation of a ``Statement``: a call (a registered tool, an
    allowed builtin or a method) or an expression evaluation. Frozen: compile
    sets the planned fields, ``run_statement`` returns a new ``Step`` with the
    run fields filled in (``dataclasses.replace``).

    Planned fields
    --------------
    text : str
        The step as source, without any binding name (``add(_SUB_0, 3)``).
    kind : str
        ``"tool"`` (dispatched: a tool, builtin or method call) or ``"eval"``
        (an expression evaluated in place, no tool).
    tool : str
        The dispatch id: a registered tool id, ``PY_BUILTIN_ALIAS`` or
        ``ATTR_CALL_ALIAS`` for a tool step; ``RHS_ASSIGN_ALIAS`` or
        ``RETURN_ALIAS`` for an eval step.
    name : str | None
        The scratch name a later step of the same statement reads
        (``_SUB_n``); ``None`` for the statement's own (last) step.
    args, kwargs
        The parsed argument expressions (``ast.expr`` nodes, plus the spliced
        builtin-name ``str``), exactly as the parser built them.

    Run fields (empty until the step runs)
    --------------------------------------
    The arguments are resolved at execution time and not stored: the planned
    ``args``/``kwargs`` are the step's representation, and a failed step
    carries its ``error``.

    output : Any
        What the step produced, ``NO_VAL`` until it runs: the full
        ``AtomicResult`` envelope for a tool step, the plain value for an eval
        step. Read it through ``result`` and ``value``.
    error : Exception | None
        The live exception of a failed step.
    status : str
        ``"not_run"``, ``"ran"`` or ``"failed"``.
    """

    text: str
    kind: str
    tool: str
    name: Optional[str] = None
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)
    output: Any = NO_VAL
    error: Optional[Exception] = None
    status: str = "not_run"

    @property
    def result(self) -> Optional[AtomicResult]:
        """The full result envelope of a tool step that ran; else ``None``."""
        return self.output if self.kind == "tool" and self.status == "ran" else None

    @property
    def value(self) -> Any:
        """The plain value (``result.result`` for a tool step); ``NO_VAL`` until it runs."""
        if self.status != "ran":
            return NO_VAL
        return self.output.result if self.kind == "tool" else self.output

    def to_dict(self) -> dict[str, Any]:
        """Debugging/observability snapshot; never used to rebuild a step."""
        return {
            "text": self.text,
            "kind": self.kind,
            "tool": self.tool,
            "name": self.name,
            "args": [_render(v) for v in self.args],
            "kwargs": {k: _render(v) for k, v in self.kwargs.items()},
            "result": None if self.result is None else self.result.to_dict(),
            "value": None if self.value is NO_VAL else self.value,
            "error": None if self.error is None else repr(self.error),
            "status": self.status,
        }


@dataclass(frozen=True, slots=True)
class Statement:
    """
    One statement as the model wrote it, with the steps it takes: hoisted
    calls first (innermost, left to right), its own step last. The top-level
    unit of execution and concurrency; a failure is local to it. Frozen:
    ``run_statement`` returns a new ``Statement`` whose steps carry the run.

    Fields
    ------
    ordinal : int
        Position among the statements of the run.
    source : str
        The statement as written (``ast.unparse`` of the parsed node).
    kind : str
        ``"assign"`` (binds ``target``), ``"call"`` (a bare call) or
        ``"return"``.
    target : str | None
        The name an assignment binds; ``None`` for a bare call or a return.
    steps : tuple[Step, ...]

    Derived from the steps (never stored)
    -------------------------------------
    status : str
        ``"failed"`` if any step failed, ``"done"`` if the own (last) step
        ran, else ``"not_run"``.
    value : Any
        The own step's value once done; ``NO_VAL`` otherwise.
    """

    ordinal: int
    source: str
    kind: str
    target: Optional[str]
    steps: tuple[Step, ...]

    @property
    def status(self) -> str:
        if any(step.status == "failed" for step in self.steps):
            return "failed"
        return "done" if self.steps[-1].status == "ran" else "not_run"

    @property
    def value(self) -> Any:
        return self.steps[-1].value if self.status == "done" else NO_VAL

    def to_dict(self) -> dict[str, Any]:
        """Debugging/observability snapshot; never used to rebuild a statement."""
        return {
            "ordinal": self.ordinal,
            "source": self.source,
            "kind": self.kind,
            "target": self.target,
            "steps": [step.to_dict() for step in self.steps],
            "status": self.status,
            "value": None if self.value is NO_VAL else self.value,
        }
