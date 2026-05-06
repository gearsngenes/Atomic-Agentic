from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any, ClassVar, Dict, Mapping

from ..core.sentinels import NO_VAL

__all__ = ["AgentTurn", "ToolAgentTurn", "BlackboardSlot", "ConstantSpec"]


_VALID_CONSTANT_NAME: re.Pattern[str] = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


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
        if not _VALID_CONSTANT_NAME.fullmatch(normalized_name):
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
class AgentTurn:
    """
    Canonical memory record for one completed Agent invocation.

    A turn stores the important lifecycle artifacts needed to reconstruct
    future LLM-facing context without storing provider-facing message dicts
    as the canonical memory format.
    """
    prompt: str
    raw_response: Any
    final_response: Any

    def to_dict(self) -> Dict[str, Any]:
        """Return a shallow dictionary representation of this turn."""
        return asdict(self)


@dataclass(slots=True)
class ToolAgentTurn(AgentTurn):
    """
    Canonical memory record for one completed ToolAgent invocation.

    In addition to the base AgentTurn lifecycle artifacts, a ToolAgentTurn
    stores the half-open span of persisted blackboard entries produced by
    the invocation. The ToolAgent renders that span into future LLM-facing
    context when building messages.
    """
    blackboard_start: int | None = None
    blackboard_end: int | None = None


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

    result : Any | NO_VAL
        Tool execution result. Set by ``_execute_prepared_batch()`` on success.
        Used for placeholder resolution in dependent steps.

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
    VALID_STATUSES: ClassVar[set[str]] = {
        "empty",
        "planned",
        "prepared",
        "executed",
        "failed",
    }

    step: int

    tool: str | Any = NO_VAL
    args: Any = NO_VAL
    resolved_args: Any = NO_VAL
    result: Any = NO_VAL
    error: Any = NO_VAL
    status: str = "empty"
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
        return self.status == "empty"

    def is_planned(self) -> bool:
        return self.status == "planned"

    def is_prepared(self) -> bool:
        return self.status == "prepared"

    def is_executed(self) -> bool:
        return self.status == "executed"

    def is_failed(self) -> bool:
        return self.status == "failed"

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

        allowed = {
            "step",
            "tool",
            "args",
            "resolved_args",
            "result",
            "error",
            "status",
            "step_dependencies",
            "await",
            "await_step",
        }
        extra = set(data.keys()) - allowed
        if extra:
            raise ValueError(
                f"BlackboardSlot.from_dict received unsupported keys: {sorted(extra)!r}."
            )

        if "step" not in data:
            raise ValueError("BlackboardSlot.from_dict missing required key: 'step'.")

        if "await" in data and "await_step" in data:
            raise ValueError(
                "BlackboardSlot.from_dict received both 'await' and 'await_step'; "
                "provide only one."
            )

        await_step = data.get("await_step", data.get("await", NO_VAL))

        return cls(
            step=data["step"],
            tool=data.get("tool", NO_VAL),
            args=data.get("args", NO_VAL),
            resolved_args=data.get("resolved_args", NO_VAL),
            result=data.get("result", NO_VAL),
            error=data.get("error", NO_VAL),
            status=data.get("status", "empty"),
            step_dependencies=data.get("step_dependencies", tuple()),
            await_step=await_step,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "tool": self.tool,
            "args": self.args,
            "resolved_args": self.resolved_args,
            "result": self.result,
            "error": self.error,
            "status": self.status,
            "step_dependencies": self.step_dependencies,
            "await_step": self.await_step,
        }