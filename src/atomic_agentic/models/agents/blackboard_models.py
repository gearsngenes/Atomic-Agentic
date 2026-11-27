from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any, Optional

from ...constants.agents import ATTR_CALL_ALIAS, KWARGS_UNPACK_KEY, PY_BUILTIN_ALIAS
from ...constants.core import NO_VAL
from ...utils.core import render_value as _render
from ..results import AtomicResult

__all__ = ["Step", "Statement"]


def _source(value: Any) -> str:
    """Source of an argument: ``ast.unparse`` of an expression, else ``repr``."""
    return ast.unparse(value) if isinstance(value, ast.expr) else repr(value)


@dataclass(frozen=True, slots=True)
class Step:
    """
    One sub-operation of a ``Statement``: a call (a registered tool, an
    allowed builtin or a method) or an expression evaluation. Frozen: compile
    sets the planned fields, ``run_statement`` returns a new ``Step`` with the
    run fields filled in (``dataclasses.replace``).

    Planned fields
    --------------
    kind : str
        ``"tool"`` (dispatched: a tool, builtin or method call) or ``"eval"``
        (an expression evaluated in place, no tool).
    tool : str | None
        The dispatch id of a tool step: a registered tool id,
        ``PY_BUILTIN_ALIAS`` or ``ATTR_CALL_ALIAS``. ``None`` for an eval step.
    name : str | None
        The scratch name a later step of the same statement reads
        (``_SUB_n``); ``None`` for the statement's own (last) step.
    args, kwargs
        The parsed argument expressions (``ast.expr`` nodes, plus the spliced
        builtin or method name ``str``), exactly as the parser built them. An
        eval step carries its expression as ``args[0]``. A builtin step is
        ``(builtin name, *call args)``; a method step is
        ``(object expr, method name, *call args)``.

    ``text`` renders the step as source (``add(_SUB_0, 3)``, ``x.upper()``,
    ``a + 1``), without any binding name.

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

    kind: str
    tool: Optional[str] = None
    name: Optional[str] = None
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)
    output: Any = NO_VAL
    error: Optional[Exception] = None
    status: str = "not_run"

    @property
    def text(self) -> str:
        """The step as source, without any binding name."""
        if self.kind == "eval":
            return _source(self.args[0])
        if self.tool == PY_BUILTIN_ALIAS:
            call_name, call_args = self.args[0], self.args[1:]
        elif self.tool == ATTR_CALL_ALIAS:
            call_name, call_args = f"{_source(self.args[0])}.{self.args[1]}", self.args[2:]
        else:
            call_name, call_args = self.tool, self.args
        tokens = [
            f"*{_source(entry.value)}" if isinstance(entry, ast.Starred) else _source(entry)
            for entry in call_args
        ]
        tokens += [
            f"**{_source(value)}" if key == KWARGS_UNPACK_KEY else f"{key}={_source(value)}"
            for key, value in self.kwargs.items()
        ]
        return f"{call_name}({', '.join(tokens)})"

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
