from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from ...constants.agents import RETURN_ALIAS
from ...constants.core import IDENTIFIER_PATTERN, NO_VAL
from ..results import AtomicResult

__all__ = [
    "ConstantSpec",
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
class CodeStatement:
    """
    One slot in a ScriptAgent subtask's per-invocation statement sequence.

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
    Mutable (not frozen) -- ``result``/``exception`` are populated after
    construction by a future ``act()``-phase caller.

    Fields
    ------
    identifier : str | None
        This slot's bound name -- the statement's LHS, or a synthesized
        ``_SUB_N`` name for an auto-hoisted nested call. ``None`` for a
        bare (unassigned) call or a ``return`` statement -- never
        resolvable by name, and never written into
        ``ScriptAgentTask.cache``.

    tool : str
        Dotted call name (e.g. ``"Type.namespace.name"``), or
        ``RHS_ASSIGN_ALIAS`` (``"rhs_assign"``) for a bare-expression
        statement. ``rhs_assign``/``RETURN_ALIAS`` calls never count
        against tool-call budget accounting -- they're never dispatched at
        all. ``PY_BUILTIN_ALIAS`` (``"py_builtin"``) marks a rewritten
        approved-builtin call -- dispatches through a real ``Tool``
        (``agents.tools.builtin_call_tool``) and counts toward tool-call
        budget accounting identically to a real registered-tool call (no
        exemption).

    args : tuple[Any, ...]
        Positional call arguments, in source order. Each entry is an
        ``ast.expr`` (dependency-bearing or not -- see class docstring) or
        an ``ast.Starred`` (a ``*expr`` unpack, its Starred-ness preserved
        regardless of whether its own inner expr has dependencies, so it
        survives to resolve time) -- except ``args[0]`` on a
        ``PY_BUILTIN_ALIAS`` slot, a plain ``str`` (see class docstring).
        Empty for a keyword-only call, or for the ``rhs_assign``/
        ``return`` sentinel shape (which lives entirely in ``kwargs``).

    kwargs : dict[str, Any]
        Keyword call arguments (real tool call), or ``{"val": <expr>}``
        (``rhs_assign``/``return``). Each value is an ``ast.expr`` node --
        see class docstring. A ``**expr`` unpack is stored under the
        reserved key ``constants.agents.KWARGS_UNPACK_KEY`` (``"**"``,
        never a valid Python identifier, so it never collides with a real
        parameter name) -- at most one per statement.

    batch_index : int | None
        Which concurrently-dispatched batch this slot belongs to, stamped
        once by ``compile_batches`` when the batch closes. ``None`` until
        then -- never observed externally in that state, since only
        committed slots (always already batch-stamped) ever reach
        ``.completed`` or a rendered record. Not defensively validated in
        ``__post_init__``, matching ``result``/``exception``'s treatment
        below: set internally by framework lifecycle code, not derived
        from external/LLM input.

    result : AtomicResult | None
        Full result envelope from a future ``act()``-phase caller. ``None``
        until executed.

    exception : Exception | None
        Live exception object from a failed execution attempt, or a
        ``DependencyFailedError`` if a dependency itself failed. ``None``
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
        # 1. identifier, if not None, must be a non-empty, Python-
        # identifier-legal string. None means a bare call or return.
        if self.identifier is not None and (
            not isinstance(self.identifier, str)
            or not IDENTIFIER_PATTERN.fullmatch(self.identifier)
        ):
            raise ValueError(
                "CodeStatement.identifier must be None or a non-empty, "
                f"Python-identifier-legal string; got {self.identifier!r}."
            )

        # 2. tool must be a non-empty string (dotted call names are not
        # bare identifiers, so IDENTIFIER_PATTERN does not apply here).
        if not isinstance(self.tool, str) or not self.tool.strip():
            raise ValueError(
                f"CodeStatement.tool must be a non-empty string; got {self.tool!r}."
            )

        # 3. args must be a tuple or list (no per-element validation --
        # values may be literally anything, including raw ast nodes).
        # Normalized to a tuple below.
        if isinstance(self.args, (str, bytes)) or not isinstance(self.args, (tuple, list)):
            raise TypeError(
                f"CodeStatement.args must be a tuple or list; got {type(self.args).__name__!r}."
            )
        self.args = tuple(self.args)

        # 4. kwargs must be a dict.
        if not isinstance(self.kwargs, dict):
            raise TypeError(
                f"CodeStatement.kwargs must be a dict; got {type(self.kwargs).__name__!r}."
            )

        # 5. batch_index/result/exception are set internally by framework
        # lifecycle code, not derived from external/LLM input -- not
        # defensively validated here, per 01-overview.md Section 4's
        # boundary-only-validation rule.

    def to_dict(self) -> dict[str, Any]:
        """
        Return the explicit serialized dictionary representation, for
        debugging/observability only -- never used to reconstruct or
        re-plan. An args/kwargs value still an unresolved ``ast.expr``/
        ``ast.Starred`` (this slot depended on another slot's identifier,
        never folded back per ``resolve_slot_args``'s own docstring) is
        rendered as its source text via ``ast.unparse`` rather than the
        raw AST node, which is not JSON-serializable.
        """

        def render(value: Any) -> Any:
            return ast.unparse(value) if isinstance(value, ast.expr) else value

        return {
            "identifier": self.identifier,
            "tool": self.tool,
            "args": [render(value) for value in self.args],
            "kwargs": {key: render(value) for key, value in self.kwargs.items()},
            "batch_index": self.batch_index,
            "result": self.result.to_dict() if self.result is not None else None,
            "exception": repr(self.exception) if self.exception is not None else None,
        }


@dataclass(slots=True)
class DagToolCall:
    """
    One slot in a DagAgent round's per-invocation call sequence.

    Sibling to ``CodeStatement``, not a subclass or shared-base retrofit --
    ``args``/``kwargs`` entries here are always plain, already-typed
    scalars (``str | int | float | bool | None``) straight from the wire
    payload, from construction onward -- never an ``ast.expr``, and never a
    "parse pending/failed" half-state (that state doesn't exist at all;
    there is no separate parse step anymore, fallible or otherwise). A
    ``str`` entry may carry zero or more ``$name`` sigil references
    (whole-string or embedded); resolving those happens once, at prepare
    time, via ``utils.dag.resolve_call_args``/``resolve_sigil_value`` --
    never here, never at construction, never more than once. The real logic
    (validation/``to_dict``/resolution) doesn't share anything with
    ``CodeStatement`` regardless -- different enough front ends and
    grammars (``DagAgent`` permits no function/method calls at all;
    ``ScriptAgent`` does) that a shared base class isn't worth it.

    Every ``DagToolCall`` represents a real registered-tool call authored by
    the model into the wire schema's ``plan`` array, or the framework-
    synthesized call representing a round's resolved ``return`` value (see
    ``tool`` below) -- there is no other kind. Unlike ``CodeStatement``,
    there is no ``rhs_assign``/builtin/attribute-call branch: this format has
    no inline-computation grammar at all.

    Mutable (not frozen) -- ``result``/``exception`` are populated after
    construction by a future ``act()``-phase caller, exactly like
    ``CodeStatement``.

    Fields
    ------
    identifier : str | None
        This call's bound name -- the wire schema's ``result_name`` value
        (stripped, never otherwise validated at this layer -- see
        ``__post_init__``), or ``None`` for a bare unassigned call. Also
        always ``None`` for the framework-synthesized ``RETURN_ALIAS`` call
        (a ``return`` value is never resolvable by name from a later round
        -- there is no later round). A non-``None`` value has exactly one
        leading ``$`` stripped (after whitespace stripping) before any
        further use -- ``__post_init__`` step 1c, below -- since no legal
        identifier can start with ``$``, this is pure recovery from a model
        blending reference-syntax with definition-syntax, never a
        collision with an intended name. Beyond that stripping, still
        accepted here without complaint if not actually identifier-pattern-
        legal or if it collides with a reserved ``K_*``/``task_result_*``
        prefix -- ``utils.dag.validate_calls`` is what catches that, as a
        regen-repair-eligible issue rather than a construction-time crash.

    tool : str
        Either a real registered tool's ``full_name`` (the wire schema's
        ``call`` value -- the model can only ever have picked a name that
        was already in the schema's dynamic ``call`` enum at generation
        time), or the reserved ``RETURN_ALIAS`` sentinel (imported from
        ``constants.agents``, reused verbatim -- not redefined). The model
        itself never writes ``RETURN_ALIAS`` into ``plan`` -- it is
        synthesized by the framework, post-parse, from the wire schema's
        separate top-level ``return`` field, exactly the way
        ``ScriptAgent``'s ``rewrite_builtin_calls`` synthesizes
        ``PY_BUILTIN_ALIAS`` post-parse. A ``RETURN_ALIAS`` call is never
        dispatched through a real tool -- its ``.result``/``.exception``
        stay ``None`` for its entire lifetime; its resolved value lives in
        ``kwargs["val"]`` instead (mirrors ``CodeStatement``'s own
        ``RETURN_ALIAS``/``RHS_ASSIGN_ALIAS`` treatment).

    args : tuple[Any, ...]
        Positional call arguments, in wire-schema order (the ``arguments``
        list's ``name: null`` entries). Each entry is exactly the value
        ``DAG_OUTPUT_SCHEMA``'s own value union produced -- ``str | int |
        float | bool | None`` -- verbatim from construction onward. A
        ``str`` entry may contain zero or more ``$name`` sigil references
        (whole-string or embedded); resolving those happens once, at
        prepare time, via ``utils.dag.resolve_call_args``/
        ``resolve_sigil_value`` -- never here, never at construction, never
        more than once.

    kwargs : dict[str, Any]
        Keyword call arguments (the ``arguments`` list's ``name: "<str>"``
        entries), or ``{"val": <value>}`` for the synthesized
        ``RETURN_ALIAS`` call. Same already-typed-scalar shape as ``args``.

    batch_index : int | None
        Which concurrently-dispatched batch this call belongs to, stamped
        once by the batch compiler when its batch closes. ``None`` until
        then.

    result : AtomicResult | None
        Full result envelope from a future ``act()``-phase caller. ``None``
        until executed, and permanently ``None`` for a ``RETURN_ALIAS`` call
        (see ``tool`` above).

    exception : Exception | None
        Live exception object from a failed dispatch attempt. ``None`` until
        a failure occurs, and permanently ``None`` for a ``RETURN_ALIAS``
        call.
    """

    identifier: Optional[str]
    tool: str
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)
    batch_index: Optional[int] = None
    result: AtomicResult | None = None
    exception: Exception | None = None

    def __post_init__(self) -> None:
        # 1. identifier, if not None, must be a string -- type-level check
        # only, stripped in place. Whether the (now-stripped) result is
        # actually identifier-pattern-legal, or collides with a reserved
        # K_*/task_result_* prefix, is content-level and belongs entirely
        # to validate_calls' regen-repair-eligible issue collection
        # (utils/dag.py), not a hard construction-time raise -- unlike
        # CodeStatement.identifier, which is always already guaranteed
        # identifier-shaped by ast's own grammar before it ever reaches
        # that class, DagToolCall.identifier comes straight from a
        # free-form wire-schema string with no such guarantee, so treating
        # illegality as fatal here would crash the whole invoke() on a
        # model mistake instead of feeding back a correctable issue.
        if self.identifier is not None:
            if not isinstance(self.identifier, str):
                raise TypeError(
                    "DagToolCall.identifier must be None or a string; got "
                    f"{type(self.identifier).__name__!r}."
                )
            self.identifier = self.identifier.strip()
            # 1c. Exactly one leading '$' is stripped, silently -- no legal
            # identifier can start with '$', so this is pure recovery from
            # a model blending reference-syntax ("$total") with
            # definition-syntax ("total"), never a collision with an
            # intended name. Not repeated: "$$total" becomes "$total",
            # still pattern-illegal, left for validate_calls to catch.
            if self.identifier.startswith("$"):
                self.identifier = self.identifier[1:].strip()

        # 2. tool must be a non-empty string. No further constraint here --
        # dotted full_names and the bare RETURN_ALIAS string are both legal;
        # enum enforcement against the registered toolset happens at the
        # schema/generation layer, not here.
        if not isinstance(self.tool, str) or not self.tool.strip():
            raise ValueError(
                f"DagToolCall.tool must be a non-empty string; got {self.tool!r}."
            )

        # 3. args must be a tuple or list. Normalized to a tuple below.
        if isinstance(self.args, (str, bytes)) or not isinstance(self.args, (tuple, list)):
            raise TypeError(
                f"DagToolCall.args must be a tuple or list; got {type(self.args).__name__!r}."
            )
        self.args = tuple(self.args)

        # 4. kwargs must be a dict.
        if not isinstance(self.kwargs, dict):
            raise TypeError(
                f"DagToolCall.kwargs must be a dict; got {type(self.kwargs).__name__!r}."
            )

        # 5. batch_index/result/exception are set internally by framework
        # lifecycle code, not derived from external/LLM input -- not
        # defensively validated here, per 01-overview.md Section 4's
        # boundary-only-validation rule.

    def to_dict(self) -> dict[str, Any]:
        """
        Return the explicit serialized dictionary representation, for
        debugging/observability only -- never used to reconstruct or
        re-plan. Every args/kwargs value is already JSON-plain (see class
        docstring), so no rendering/transformation is needed -- unlike
        ``CodeStatement.to_dict()``'s ``ast.unparse``-or-passthrough
        fallback, values are used directly.
        """
        return {
            "identifier": self.identifier,
            "tool": self.tool,
            "args": list(self.args),
            "kwargs": dict(self.kwargs),
            "batch_index": self.batch_index,
            "result": self.result.to_dict() if self.result is not None else None,
            "exception": repr(self.exception) if self.exception is not None else None,
        }

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
