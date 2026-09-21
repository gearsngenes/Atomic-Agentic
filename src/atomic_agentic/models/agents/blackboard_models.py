from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar, Mapping, Optional

from ...constants.agents import (
    ARGS_FIELD,
    AWAIT_FIELD,
    RETURN_ALIAS,
    STEP_FIELD,
    TOOL_FIELD,
)
from ...constants.core import IDENTIFIER_PATTERN, NO_VAL
from ..results import AtomicResult

__all__ = [
    "ConstantSpec",
    "BlackboardSlot",
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
class BlackboardSlot:
    """
    One indexed slot in the run blackboard, representing a single tool invocation.

    Each slot tracks the complete lifecycle of a tool call from planning through execution.
    State transitions are tracked explicitly through ``status`` while unset field values
    continue to use the shared ``NO_VAL`` marker:

    State Lifecycle
    ~~~~~~~~~~~~~~~
    1. **Empty**: ``status="empty"``
       - Slot allocated but not yet planned

    2. **Planned**: ``status="planned"``
       - Slot assigned a tool and raw arguments, but not yet ready for execution

    3. **Prepared**: ``status="prepared"``
       - Slot assigned resolved arguments; ready for execution
       - Placeholder dependencies have been resolved to concrete values

    4. **Executed**: ``status="executed"``
       - Tool has been invoked successfully; result is stored
       - Slot is now available for subsequent steps' placeholder resolution

    5. **Failed**: ``status="failed"``
       - Tool invocation failed; error is stored

    Fields
    ------
    step : int
        Global blackboard index (0-based). Always matches the slot's position in the
        containing blackboard list during planning. After persistence to cache, this
        index becomes globally unique (incremented from previous cache length).

    tool : str | NO_VAL
        Tool name (``Tool.full_name``). Set at prepare time; must reference a
        registered tool or invoke will raise.

    args : Any (typically dict)
        Raw, unresolved arguments. May contain placeholders (``<<__sN__>>``,
        ``<<__cN__>>``). Immutable after prepare time.

    resolved_args : Any (typically dict) | NO_VAL
        Arguments after placeholder resolution. Created at prepare time by
        ``_resolve_placeholders(args, state=...)``. Passed to ``tool.invoke()``.

    result : AtomicResult | NO_VAL
        Full result envelope (``ToolResult``, an ``AtomicResult`` subclass)
        produced by a successful tool invocation — preserved whole for richer
        tracing (timing, run identity, invoker). Set by ``act()``.
        Consumers that need the caller-facing
        value (placeholder resolution, previews, ``return_value``) read
        ``result.result`` directly; every such site is already gated by an
        ``is_executed()`` check, so the envelope is guaranteed present there.

    error : Any | NO_VAL
        Exception captured during execution (if any). Set only on failure.
        Result remains ``NO_VAL`` if error is set.

    status : str
        Explicit lifecycle status. Must be one of:
        ``"empty"``, ``"planned"``, ``"prepared"``, ``"executed"``, or ``"failed"``.

    step_dependencies : tuple[int, ...]
        Plan-local step dependencies used by planning/batch compilation logic.
        This field is stored directly and is not inferred from ``args``.

    await_step : int | NO_VAL
        Optional explicit scheduling barrier from a planner ``"await"`` field.
        Defaults to ``NO_VAL`` when no await barrier is present.
    """
    EMPTY: ClassVar[str] = "empty"
    PLANNED: ClassVar[str] = "planned"
    PREPARED: ClassVar[str] = "prepared"
    EXECUTED: ClassVar[str] = "executed"
    FAILED: ClassVar[str] = "failed"

    VALID_STATUSES: ClassVar[frozenset[str]] = frozenset(
        {
            EMPTY,
            PLANNED,
            PREPARED,
            EXECUTED,
            FAILED,
        }
    )

    RESOLVED_ARGS_FIELD: ClassVar[str] = "resolved_args"
    RESULT_FIELD: ClassVar[str] = "result"
    ERROR_FIELD: ClassVar[str] = "error"
    STATUS_FIELD: ClassVar[str] = "status"
    STEP_DEPENDENCIES_FIELD: ClassVar[str] = "step_dependencies"
    AWAIT_STEP_FIELD: ClassVar[str] = "await_step"

    FROM_DICT_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            STEP_FIELD,
            TOOL_FIELD,
            ARGS_FIELD,
            RESOLVED_ARGS_FIELD,
            RESULT_FIELD,
            ERROR_FIELD,
            STATUS_FIELD,
            STEP_DEPENDENCIES_FIELD,
            AWAIT_FIELD,
            AWAIT_STEP_FIELD,
        }
    )

    step: int

    tool: str | Any = NO_VAL
    args: Any = NO_VAL
    resolved_args: Any = NO_VAL
    result: AtomicResult | Any = NO_VAL
    error: Any = NO_VAL
    status: str = EMPTY
    step_dependencies: tuple[int, ...] = ()
    await_step: int | Any = NO_VAL

    def __post_init__(self) -> None:
        self._validate_step(self.step)
        self._validate_status(self.status)
        self.step_dependencies = self._normalize_step_dependencies(
            self.step_dependencies
        )
        self._validate_await_step(self.await_step)

    @staticmethod
    def _validate_step(value: Any) -> None:
        if type(value) is not int or value < 0:
            raise ValueError("BlackboardSlot.step must be an int >= 0.")

    @classmethod
    def _validate_status(cls, value: Any) -> None:
        if not isinstance(value, str) or value not in cls.VALID_STATUSES:
            raise ValueError(
                "BlackboardSlot.status must be one of: "
                f"{', '.join(sorted(cls.VALID_STATUSES))}."
            )

    @staticmethod
    def _normalize_step_dependencies(value: Any) -> tuple[int, ...]:
        if value is NO_VAL or value is None:
            return tuple()

        if isinstance(value, int) and not isinstance(value, bool):
            raw_values = [value]
        elif isinstance(value, (list, tuple, set, frozenset)):
            raw_values = list(value)
        else:
            raise ValueError(
                "BlackboardSlot.step_dependencies must be an int or an iterable of ints."
            )

        normalized: set[int] = set()
        for dep in raw_values:
            if type(dep) is not int or dep < 0:
                raise ValueError(
                    "BlackboardSlot.step_dependencies must contain only ints >= 0."
                )
            normalized.add(dep)

        return tuple(sorted(normalized))

    @staticmethod
    def _validate_await_step(value: Any) -> None:
        if value is NO_VAL:
            return
        if type(value) is not int or value < 0:
            raise ValueError("BlackboardSlot.await_step must be NO_VAL or an int >= 0.")

    def is_empty(self) -> bool:
        return self.status == self.EMPTY

    def is_planned(self) -> bool:
        return self.status == self.PLANNED

    def is_prepared(self) -> bool:
        return self.status == self.PREPARED

    def is_executed(self) -> bool:
        return self.status == self.EXECUTED

    def is_failed(self) -> bool:
        return self.status == self.FAILED

    def copy(self) -> "BlackboardSlot":
        """Return a shallow copy of this blackboard slot."""
        return BlackboardSlot(
            step=self.step,
            tool=self.tool,
            args=self.args,
            resolved_args=self.resolved_args,
            result=self.result,
            error=self.error,
            status=self.status,
            step_dependencies=self.step_dependencies,
            await_step=self.await_step,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BlackboardSlot":
        """
        Construct a blackboard slot from a mapping.

        This method does not inspect ``args`` or infer dependencies. Dependency metadata
        should be passed directly through ``step_dependencies`` by the caller.

        The planner-facing key ``"await"`` is accepted as an alias for ``await_step``.
        """
        if not isinstance(data, Mapping):
            raise TypeError(
                f"BlackboardSlot.from_dict requires a mapping; got {type(data).__name__!r}."
            )

        extra = set(data.keys()) - cls.FROM_DICT_FIELDS
        if extra:
            raise ValueError(
                f"BlackboardSlot.from_dict received unsupported keys: {sorted(extra)!r}."
            )

        if STEP_FIELD not in data:
            raise ValueError(
                f"BlackboardSlot.from_dict missing required key: {STEP_FIELD!r}."
            )

        if AWAIT_FIELD in data and cls.AWAIT_STEP_FIELD in data:
            raise ValueError(
                f"BlackboardSlot.from_dict received both {AWAIT_FIELD!r} and "
                f"{cls.AWAIT_STEP_FIELD!r}; provide only one."
            )

        await_step = data.get(
            cls.AWAIT_STEP_FIELD,
            data.get(AWAIT_FIELD, NO_VAL),
        )

        return cls(
            step=data[STEP_FIELD],
            tool=data.get(TOOL_FIELD, NO_VAL),
            args=data.get(ARGS_FIELD, NO_VAL),
            resolved_args=data.get(cls.RESOLVED_ARGS_FIELD, NO_VAL),
            result=data.get(cls.RESULT_FIELD, NO_VAL),
            error=data.get(cls.ERROR_FIELD, NO_VAL),
            status=data.get(cls.STATUS_FIELD, cls.EMPTY),
            step_dependencies=data.get(cls.STEP_DEPENDENCIES_FIELD, tuple()),
            await_step=await_step,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            STEP_FIELD: self.step,
            TOOL_FIELD: self.tool,
            ARGS_FIELD: self.args,
            self.RESOLVED_ARGS_FIELD: self.resolved_args,
            self.RESULT_FIELD: self.result,
            self.ERROR_FIELD: self.error,
            self.STATUS_FIELD: self.status,
            self.STEP_DEPENDENCIES_FIELD: self.step_dependencies,
            self.AWAIT_STEP_FIELD: self.await_step,
        }


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
    the fields are structurally close (both ultimately hold parsed
    ``ast.expr`` nodes in ``args``/``.kwargs``, evaluated exactly once at
    prepare time via the shared ``utils.agents.evaluate_expr``), but they
    arrive there through different front ends: ``CodeStatement`` parses one
    whole Python statement at once (``ScriptAgent``'s own text-based
    generation); a ``DagToolCall``'s ``args``/``.kwargs`` entries each start
    as their own independent wire-schema *string* (``output_structure``
    strict-mode JSON), parsed one value at a time by
    ``utils.dag.parse_call_expressions``, not one statement at a time. The
    real logic (validation/``to_dict``/resolution) still doesn't share
    anything regardless -- different enough front ends and grammars
    (``DagAgent`` permits no function/method calls at all; ``ScriptAgent``
    does) that a shared base class isn't worth it.

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
        This call's bound name -- the wire schema's ``assign_to`` value
        (stripped, never otherwise validated at this layer -- see
        ``__post_init__``), or ``None`` for a bare unassigned call. Also
        always ``None`` for the framework-synthesized ``RETURN_ALIAS`` call
        (a ``return`` value is never resolvable by name from a later round
        -- there is no later round). A non-``None`` value that isn't
        actually identifier-pattern-legal, or collides with a reserved
        ``K_*``/``task_result_*`` prefix, is accepted here without
        complaint -- ``utils.dag.validate_calls`` is what catches that,
        as a regen-repair-eligible issue rather than a construction-time
        crash.

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
        list's ``name: null`` entries). Each entry starts, at construction
        time, as the raw Python-source *string* the wire schema's
        ``value`` field carries (no decoding of any kind happens in
        ``parse_generation``) -- ``utils.dag.parse_call_expressions``
        parses it, exactly once, into an ``ast.expr`` node (mirrors
        ``CodeStatement``'s own "parsed-once, never re-folded" contract),
        replacing the raw string in place. An entry that fails to parse
        (empty after fence-stripping, a ``SyntaxError``, or a rejected
        form -- see ``reject_unsupported_forms``) stays the original raw
        string -- but a round with any such entry never proceeds past
        ``validate_calls``' combined issue list, so no other method ever
        needs to handle that half-parsed state.

    kwargs : dict[str, Any]
        Keyword call arguments (the ``arguments`` list's ``name: "<str>"``
        entries), or ``{"val": <value>}`` for the synthesized
        ``RETURN_ALIAS`` call. Same raw-string-then-parsed-``ast.expr``
        shape as ``args``.

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
        re-plan. An args/kwargs value still an unparsed raw string (this
        call's own parse failed, or -- transiently -- hasn't run yet) is
        rendered as-is; an already-parsed ``ast.expr`` is rendered as its
        source text via ``ast.unparse`` (not the raw AST node, which isn't
        JSON-serializable) -- same fallback shape ``CodeStatement.to_dict()``
        uses.
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

    def serialize(self) -> dict[str, Any]:
        """
        Return this call reconstructed in the **wire schema's own shape**
        (``call``/``assign_to``/``arguments: [{"name", "value"}, ...]``),
        the model's own generation vocabulary -- distinct from
        ``to_dict()``'s internal-field debug view. This is the one
        authoritative home for that reconstruction:
        ``utils.dag.render_completed_as_json`` is a thin wrapper calling
        this once per call, rather than rebuilding the shape itself.

        Same ``ast.unparse``-if-parsed-else-passthrough rendering
        ``to_dict()`` uses -- a wire-shape ``value`` is always a string
        either way (Python source), matching ``DAG_OUTPUT_SCHEMA``'s own
        ``value``/``return`` domain.
        """

        def render(value: Any) -> str:
            return ast.unparse(value) if isinstance(value, ast.expr) else value

        return {
            "call": self.tool,
            "assign_to": self.identifier,
            "arguments": (
                [{"name": None, "value": render(value)} for value in self.args]
                + [{"name": key, "value": render(value)} for key, value in self.kwargs.items()]
            ),
        }

    def to_python_code(self) -> str:
        """
        Reconstruct this call as one line of real Python source -- possible
        now that every args/kwargs value is a genuinely parsed ``ast.expr``,
        not a JSON scalar. A ``RETURN_ALIAS`` call renders as
        ``return <val>``; any other call renders as
        ``<identifier> = <tool>(<args>)`` (bare ``<tool>(<args>)`` when
        ``identifier`` is ``None``). Assumes every value here is already an
        ``ast.expr`` -- a round with any unparsed/rejected value never
        reaches a call site that calls this method (see ``args``/``kwargs``
        docstrings above); ``ast.unparse`` raises naturally on a plain
        string, uncaught, if that assumption is ever violated.
        """
        if self.tool == RETURN_ALIAS:
            return f"return {ast.unparse(self.kwargs['val'])}"

        prefix = f"{self.identifier} = " if self.identifier is not None else ""
        args_source = ", ".join(
            [ast.unparse(value) for value in self.args]
            + [f"{key}={ast.unparse(value)}" for key, value in self.kwargs.items()]
        )
        return f"{prefix}{self.tool}({args_source})"
