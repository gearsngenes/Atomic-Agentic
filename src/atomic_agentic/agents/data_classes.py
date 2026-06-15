from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar, Dict, Mapping

from .constants import (
    ARGS_FIELD,
    AWAIT_FIELD,
    STEP_FIELD,
    TOOL_FIELD,
)
from ..core.constants import IDENTIFIER_PATTERN, NO_VAL
from ..models.results import AtomicResult, LLMResult

__all__ = [
    "AgentRecord",
    "LLMRecord",
    "ToolAgentRecord",
    "BlackboardSlot",
    "ConstantSpec",
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

@dataclass(frozen=True, slots=True)
class LLMRecord:
    """
    Canonical memory record for one completed LLM generation made during an
    Agent invocation.

    An Agent invocation may involve one or more LLM generations (e.g. a
    ToolAgent's planning loop). Each generation that contributes to an
    invocation is preserved here in full — not distilled — so that future
    rendering, debugging, and accounting needs are not constrained by what an
    earlier pass chose to keep.

    Fields
    ------
    user_prompt:
        User-facing prompt text that triggered this specific LLM generation.
        May differ from the AgentRecord's overall ``user_prompt`` when an
        invocation makes multiple LLM calls with distinct prompts.

    llm_result:
        The complete LLMResult produced by this generation, including its
        token usage, model identity, timing, and run identity.
    """

    user_prompt: str
    llm_result: LLMResult

    def __post_init__(self) -> None:
        if not isinstance(self.user_prompt, str):
            raise TypeError(
                f"LLMRecord.user_prompt must be a str, got {type(self.user_prompt).__name__}."
            )

        if not isinstance(self.llm_result, LLMResult):
            raise TypeError(
                "LLMRecord.llm_result must be an LLMResult instance, "
                f"got {type(self.llm_result).__name__}."
            )

    def to_dict(self) -> Dict[str, Any]:
        """Return the explicit serialized dictionary representation."""
        return {
            "user_prompt": self.user_prompt,
            "llm_result": self.llm_result.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class AgentRecord:
    """
    Canonical memory record for one completed Agent invocation.

    A record stores the important lifecycle artifacts needed to reconstruct
    future LLM-facing context without storing provider-facing message dicts
    as the canonical memory format. It is related to, but distinct from,
    AgentResult: AgentResult is the public successful-invocation envelope,
    while AgentRecord is the memory/rendering record. Neither wraps the
    other; ``run_id`` links a record to the AgentResult of the same
    invocation.

    Fields
    ------
    user_prompt:
        Prompt string produced by ``pre_invoke`` for this invocation.

    generated_response:
        Raw post-engine response material for this invocation, prior to
        ``post_invoke`` processing.

    final_response:
        Final Agent output for this invocation, after ``post_invoke``
        processing has completed.

    llm_records:
        Non-empty record of every LLM generation that contributed to this
        invocation. Stored as an immutable tuple after construction.

    run_id:
        Identifier linking this record to the AgentResult produced by the
        same invocation.
    """
    user_prompt: str
    generated_response: Any
    final_response: Any
    llm_records: tuple[LLMRecord, ...]
    run_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.user_prompt, str):
            raise TypeError(
                f"AgentRecord.user_prompt must be a str, got {type(self.user_prompt).__name__}."
            )

        normalized_llm_records = self._normalize_llm_records(self.llm_records)
        normalized_run_id = self._normalize_run_id(self.run_id)

        object.__setattr__(self, "llm_records", normalized_llm_records)
        object.__setattr__(self, "run_id", normalized_run_id)

    @staticmethod
    def _normalize_llm_records(value: Any) -> tuple[LLMRecord, ...]:
        """Validate and normalize the invocation's LLM generation records."""
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
            raise TypeError(
                "AgentRecord.llm_records must be a non-empty sequence of "
                f"LLMRecord instances, got {type(value).__name__}."
            )

        normalized = tuple(value)
        if not normalized:
            raise ValueError("AgentRecord.llm_records must not be empty.")

        for index, record in enumerate(normalized):
            if not isinstance(record, LLMRecord):
                raise TypeError(
                    "AgentRecord.llm_records must contain only LLMRecord instances; "
                    f"item {index} is {type(record).__name__}."
                )

        return normalized

    @staticmethod
    def _normalize_run_id(value: Any) -> str:
        """Validate and normalize the linked AgentResult run identifier."""
        if not isinstance(value, str):
            raise TypeError(
                f"AgentRecord.run_id must be a str, got {type(value).__name__}."
            )

        normalized = value.strip()
        if not normalized:
            raise ValueError("AgentRecord.run_id must be a non-empty string.")

        return normalized

    def to_dict(self) -> Dict[str, Any]:
        """Return the explicit serialized dictionary representation."""
        return {
            "user_prompt": self.user_prompt,
            "generated_response": self.generated_response,
            "final_response": self.final_response,
            "llm_records": [record.to_dict() for record in self.llm_records],
            "run_id": self.run_id,
        }


@dataclass(frozen=True, slots=True)
class ToolAgentRecord(AgentRecord):
    """
    Canonical memory record for one completed ToolAgent invocation.

    In addition to the base AgentRecord lifecycle artifacts, a ToolAgentRecord
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

    result : AtomicResult | NO_VAL
        Full result envelope (``ToolResult``, an ``AtomicResult`` subclass)
        produced by a successful tool invocation — preserved whole for richer
        tracing (timing, run identity, invoker). Set by
        ``_execute_prepared_batch()``. Consumers that need the caller-facing
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
    def from_dict(cls, data: Mapping[str, Any]) -> BlackboardSlot:
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