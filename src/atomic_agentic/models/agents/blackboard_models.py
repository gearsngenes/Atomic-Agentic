from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar, Mapping, Optional

from ...constants.agents import (
    ARGS_FIELD,
    AWAIT_FIELD,
    STEP_FIELD,
    TOOL_FIELD,
)
from ...constants.core import IDENTIFIER_PATTERN, NO_VAL
from ..results import AtomicResult

__all__ = [
    "ConstantSpec",
    "BlackboardSlot",
    "CodeStatement",
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
    ``args[0]``, dispatched through ``agents.tools.builtin_call_tool``
    rather than a registered tool). ``args`` values are mixed: a
    dependency-free expression is evaluated
    eagerly at parse time and stored as a plain Python value; an expression
    referencing another slot's identifier is stored unresolved as the raw
    ``ast.expr`` node, pending a future ``resolve_slot_args`` call. Mutable
    (not frozen) -- ``result``/``exception`` are populated after
    construction by a future ``act()``-phase caller.

    Fields
    ------
    identifier : str | None
        This slot's bound name -- the statement's LHS, or a synthesized
        ``_HOIST_N`` name for an auto-hoisted nested call. ``None`` for a
        bare (unassigned) call or a ``return`` statement -- never
        resolvable by name, and never written into
        ``ScriptAgentTask.cache``.

    tool : str
        Dotted call name (e.g. ``"Type.namespace.name"``), or
        ``RHS_ASSIGN_ALIAS`` (``"rhs_assign"``) for a bare-expression
        statement. ``rhs_assign`` calls never count against
        tools-used/tool-call budget accounting (enforced by a future,
        out-of-scope caller). ``PY_BUILTIN_ALIAS`` (``"py_builtin"``) marks
        a rewritten approved-builtin call -- also exempt from tool-call
        budget accounting like ``rhs_assign``/``return``, but (unlike those
        two) still dispatches through a real ``Tool``
        (``agents.tools.builtin_call_tool``).

    args : tuple[Any, ...]
        Positional call arguments, in source order. Each entry is a plain
        resolved literal, a pending ``ast.expr`` (an ordinary
        dependency-bearing positional value), or a pending ``ast.Starred``
        (a ``*expr`` unpack -- never eagerly constant-folded regardless of
        whether its own inner expr has dependencies, so its Starred-ness
        survives to resolve time). Empty for a keyword-only call, or for
        the ``rhs_assign``/``return`` sentinel shape (which lives entirely
        in ``kwargs``).

    kwargs : dict[str, Any]
        Keyword call arguments (real tool call), or ``{"val": <expr>}``
        (``rhs_assign``/``return``). Each value is either an
        already-resolved Python literal or an unresolved ``ast.expr`` node
        -- see class docstring. A ``**expr`` unpack is stored under the
        reserved key ``constants.agents.KWARGS_UNPACK_KEY`` (``"**"``,
        never a valid Python identifier, so it never collides with a real
        parameter name) -- at most one per statement.

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

    identifier: Optional[str]
    tool: str
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)
    awaited: bool = False
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

        # 5. awaited must be exactly a bool (bool is an int subclass; guard
        # against e.g. 0/1 silently passing an isinstance check).
        if type(self.awaited) is not bool:
            raise TypeError(
                f"CodeStatement.awaited must be a bool; got {type(self.awaited).__name__!r}."
            )

        # 6. result/exception are set internally by future, not-yet-built
        # lifecycle code, not derived from external/LLM input -- not
        # defensively validated here, per 01-overview.md Section 4's
        # boundary-only-validation rule.
