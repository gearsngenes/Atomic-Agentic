from __future__ import annotations

import ast
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from ...constants.agents import RETURN_ALIAS
from ...constants.core import IDENTIFIER_PATTERN, NO_VAL
from ..results import AtomicResult

__all__ = [
    "ConstantSpec",
    "CallSlot",
    "CodeStatement",
    "DagToolCall",
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
class CallSlot(ABC):
    """
    Shared ancestor for ``CodeStatement`` (``ScriptActAgent``'s Python-
    statement grammar) and ``DagToolCall`` (``PlanActAgent``/``ReActAgent``/
    ``DagAgent``'s JSON-wire grammar) -- one slot in a per-invocation
    call/statement sequence.

    Owns the field shape and validation the two grammars already share
    identically: the seven fields below, and the ``tool``/``args``/
    ``kwargs`` checks in ``__post_init__``. What genuinely differs between
    the two -- identifier-legality strictness (``CodeStatement``'s
    identifiers are always ast-sourced and therefore already
    grammar-legal; ``DagToolCall``'s come from a free-form wire string with
    no such guarantee) and how an ``args``/``kwargs`` value renders for
    ``to_dict()`` (``CodeStatement`` may still hold an unresolved
    ``ast.expr``; ``DagToolCall``'s are always already JSON-plain) -- stays
    on two small per-subclass hooks. Resolution, compilation, and
    prompt-rendering logic stay entirely in ``utils/dag.py``/
    ``utils/script.py``, untouched by this shared ancestor -- this class
    unifies *data shape* only, never grammar-specific behavior.

    Mutable (not frozen) -- ``result``/``exception`` are populated after
    construction by a future ``act()``-phase caller.

    Fields
    ------
    identifier : str | None
        This slot's bound name. ``None`` for a bare (unassigned) call or a
        terminal ``return``. Subclass-specific legality/normalization rules
        apply via ``_validate_identifier()``.

    tool : str
        The call's identity -- a dotted call name, a registered tool's
        alias/``full_name``, or a grammar-specific sentinel (e.g.
        ``RHS_ASSIGN_ALIAS``/``RETURN_ALIAS``/``PY_BUILTIN_ALIAS``/
        ``ATTR_CALL_ALIAS`` for ``CodeStatement``; ``RETURN_ALIAS`` for
        ``DagToolCall``). Must be a non-empty string.

    args : tuple[Any, ...]
        Positional call arguments, in source order. Normalized to a tuple
        from whatever tuple/list was supplied. Element type is
        subclass-specific (``ast.expr``/``ast.Starred`` nodes for
        ``CodeStatement``; already-typed JSON scalars for ``DagToolCall``).

    kwargs : dict[str, Any]
        Keyword call arguments. Element type is subclass-specific, same
        split as ``args``.

    batch_index : int | None
        Which concurrently-dispatched batch this slot belongs to, stamped
        once by the owning family's batch compiler when the batch closes.
        ``None`` until then. Not defensively validated -- set internally by
        framework lifecycle code, not derived from external/LLM input.

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
        # 1. identifier legality/normalization is entirely subclass-owned --
        # called first, matching both current classes' own field order.
        self._validate_identifier()

        # 2. tool must be a non-empty string -- identical check, both
        # subclasses today; type(self).__name__ substitution produces the
        # exact same message text either concrete class already hardcodes.
        if not isinstance(self.tool, str) or not self.tool.strip():
            raise ValueError(
                f"{type(self).__name__}.tool must be a non-empty string; got {self.tool!r}."
            )

        # 3. args must be a tuple or list (no per-element validation --
        # values may be anything, including raw ast nodes or plain
        # scalars). Normalized to a tuple below.
        if isinstance(self.args, (str, bytes)) or not isinstance(self.args, (tuple, list)):
            raise TypeError(
                f"{type(self).__name__}.args must be a tuple or list; "
                f"got {type(self.args).__name__!r}."
            )
        self.args = tuple(self.args)

        # 4. kwargs must be a dict.
        if not isinstance(self.kwargs, dict):
            raise TypeError(
                f"{type(self).__name__}.kwargs must be a dict; got {type(self.kwargs).__name__!r}."
            )

        # 5. batch_index/result/exception are set internally by framework
        # lifecycle code, not derived from external/LLM input -- not
        # defensively validated here, per 01-overview.md Section 4's
        # boundary-only-validation rule.

    @abstractmethod
    def _validate_identifier(self) -> None:
        """
        Validate (and, if the subclass's rules call for it, normalize in
        place) ``self.identifier``. Required, no shared default -- the two
        concrete grammars disagree on how strict this should be, and a
        future third grammar must decide this explicitly rather than
        silently inherit either one's answer.
        """
        ...

    def _render_value(self, value: Any) -> Any:
        """
        Render one ``args``/``kwargs`` value for ``to_dict()``. Default:
        identity passthrough -- correct for ``DagToolCall``, whose values
        are already JSON-plain scalars from construction onward.
        ``CodeStatement`` overrides this to unparse a still-unresolved
        ``ast.expr``/``ast.Starred`` back to source text.
        """
        return value

    def to_dict(self) -> dict[str, Any]:
        """
        Return the explicit serialized dictionary representation, for
        debugging/observability only -- never used to reconstruct or
        re-plan. Shared shape for both subclasses; only the per-value
        rendering (``_render_value``) differs.
        """
        return {
            "identifier": self.identifier,
            "tool": self.tool,
            "args": [self._render_value(value) for value in self.args],
            "kwargs": {key: self._render_value(value) for key, value in self.kwargs.items()},
            "batch_index": self.batch_index,
            "result": self.result.to_dict() if self.result is not None else None,
            "exception": repr(self.exception) if self.exception is not None else None,
        }


@dataclass(slots=True)
class CodeStatement(CallSlot):
    """
    One slot in a ScriptActAgent subtask's per-invocation statement
    sequence.

    Every generated statement normalizes to this one call-shaped record --
    a real tool call (``tool`` = the call's dotted name, optionally a bare
    unassigned call with ``identifier=None``), a bare expression (``tool``
    = ``RHS_ASSIGN_ALIAS``, ``args = {"val": <expr>}``), a terminal
    ``return`` statement (``tool`` = ``RETURN_ALIAS``, ``identifier=None``),
    or a rewritten Python builtin call (``tool`` = ``PY_BUILTIN_ALIAS``,
    structurally a real tool call with the builtin's name spliced into
    ``args[0]`` as a plain ``str`` by ``rewrite_builtin_calls`` -- the one
    exception to the rule below, since it's a post-parse rewrite, not
    something the model itself wrote as an expression). Every other
    ``args``/``kwargs`` value is always the original ``ast.expr`` node the
    model wrote, whether or not the expression has a dependency on
    another slot's identifier -- a dependency-free expression is only
    dry-run evaluated at parse time (catching a guaranteed-bad constant
    expression early), never folded into a plain value; the sole
    ``resolve_slot_args`` call at prepare time is where every such value,
    dependency-bearing or not, actually resolves to a plain Python value.

    No new fields beyond ``CallSlot``'s own seven -- only
    ``_validate_identifier``/``_render_value`` are overridden below.
    """

    def _validate_identifier(self) -> None:
        # identifier, if not None, must be a non-empty, Python-identifier-
        # legal string. None means a bare call or return. Always already
        # ast-sourced by the time a real CodeStatement is constructed, so a
        # hard raise here is safe -- never a correctable regen-repair issue
        # the way DagToolCall's free-form wire string is.
        if self.identifier is not None and (
            not isinstance(self.identifier, str)
            or not IDENTIFIER_PATTERN.fullmatch(self.identifier)
        ):
            raise ValueError(
                "CodeStatement.identifier must be None or a non-empty, "
                f"Python-identifier-legal string; got {self.identifier!r}."
            )

    def _render_value(self, value: Any) -> Any:
        return ast.unparse(value) if isinstance(value, ast.expr) else value


@dataclass(slots=True)
class DagToolCall(CallSlot):
    """
    One slot in a ``PlanActAgent``/``ReActAgent``/``DagAgent`` round's
    per-invocation call sequence.

    ``args``/``kwargs`` entries here are always plain, already-typed
    scalars (``str | int | float | bool | None``) straight from the wire
    payload, from construction onward -- never an ``ast.expr``, and never a
    "parse pending/failed" half-state. A ``str`` entry may carry zero or
    more ``$name`` sigil references (whole-string or embedded); resolving
    those happens once, at prepare time, via ``utils.dag.resolve_call_args``
    -- never here, never at construction, never more than once.

    Every ``DagToolCall`` represents a real registered-tool call authored by
    the model, or the framework-synthesized call representing a round's
    resolved ``return`` value (``tool == RETURN_ALIAS``) -- there is no
    other kind. No ``rhs_assign``/builtin/attribute-call branch: this
    format has no inline-computation grammar at all.

    No new fields beyond ``CallSlot``'s own seven -- only
    ``_validate_identifier`` is overridden below; ``_render_value`` uses
    ``CallSlot``'s default identity passthrough, since every value here is
    already JSON-plain.
    """

    def _validate_identifier(self) -> None:
        # 1. type-check only -- content-level legality (pattern match,
        # reserved-prefix collision, dunder-shape) is deliberately left to
        # utils.dag.validate_calls as a regen-repair-eligible issue, not a
        # construction-time crash, since this string comes straight from a
        # free-form wire payload with no grammatical guarantee.
        if self.identifier is not None:
            if not isinstance(self.identifier, str):
                raise TypeError(
                    "DagToolCall.identifier must be None or a string; got "
                    f"{type(self.identifier).__name__!r}."
                )
            stripped = self.identifier.strip()
            # 1c. Exactly one leading '$' is stripped, silently -- no legal
            # identifier can start with '$', so this is pure recovery from
            # a model blending reference-syntax ("$total") with
            # definition-syntax ("total"), never a collision with an
            # intended name. Not repeated: "$$total" becomes "$total",
            # still pattern-illegal, left for validate_calls to catch.
            if stripped.startswith("$"):
                stripped = stripped[1:].strip()
            # 1d. Blank after both strips (originally "", whitespace-only,
            # or a lone "$") collapses to None -- indistinguishable from no
            # name supplied at all, rather than surviving as "" to be
            # rejected downstream as an invalid identifier.
            self.identifier = stripped or None

    def serialize(self) -> dict[str, Any]:
        """
        Return this call reconstructed in the **wire schema's own shape**
        (``call``/``arguments: [{"name", "value"}, ...]``/``result_name``),
        the model's own generation vocabulary -- distinct from
        ``to_dict()``'s internal-field debug view. This is the one
        authoritative home for that reconstruction:
        ``utils.dag.render_completed_as_json`` is a thin wrapper calling
        this once per call, rather than rebuilding the shape itself.

        Every args/kwargs value is already JSON-plain -- used directly, no
        rendering step needed.
        """
        return {
            "call": self.tool,
            "arguments": (
                [{"name": None, "value": value} for value in self.args]
                + [{"name": key, "value": value} for key, value in self.kwargs.items()]
            ),
            "result_name": self.identifier,
        }

    def to_python_code(self) -> str:
        """
        Reconstruct this call as one line of real Python source, via
        ``repr()`` on each value (a value is already the real Python object
        it represents -- there is no ``ast.expr`` left to unparse). A
        ``RETURN_ALIAS`` call renders as ``return <val!r>``; any other call
        renders as ``<identifier> = <tool>(<args>)`` (bare ``<tool>(<args>)``
        when ``identifier`` is ``None``). Still produces one line of real,
        syntactically valid Python per call -- an unresolved ``$name``-
        bearing string just repr's as an ordinary quoted string (e.g.
        ``x = '$total'``), which is exactly what it is pre-resolution.
        ``repr()`` never raises for any value this class's own
        ``__post_init__`` already accepted.
        """
        if self.tool == RETURN_ALIAS:
            return f"return {self.kwargs['val']!r}"

        prefix = f"{self.identifier} = " if self.identifier is not None else ""
        args_source = ", ".join(
            [repr(value) for value in self.args]
            + [f"{key}={value!r}" for key, value in self.kwargs.items()]
        )
        return f"{prefix}{self.tool}({args_source})"
