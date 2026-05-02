from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict
from ..core.sentinels import NO_VAL

__all__ = ["AgentTurn", "ToolAgentTurn"]


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
    State transitions are **sentinel-driven** using the ``NO_VAL`` marker:

    State Lifecycle
    ~~~~~~~~~~~~~~~
    1. **Empty** (initial): ``tool=NO_VAL, resolved_args=NO_VAL, result=NO_VAL``
       - Slot allocated but not yet planned

    2. **Prepared**: ``tool≠NO_VAL, resolved_args≠NO_VAL, result=NO_VAL``
       - Slot assigned a tool and resolved arguments; ready for execution
       - Placeholder dependencies have been resolved to concrete values

    3. **Executed**: ``result≠NO_VAL`` (or ``error≠NO_VAL`` on failure)
       - Tool has been invoked; result (or exception) stored
       - Slot is now available for subsequent steps' placeholder resolution

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
    """
    step: int

    tool: str | Any = NO_VAL
    args: Any = NO_VAL
    resolved_args: Any = NO_VAL
    result: Any = NO_VAL
    error: Any = NO_VAL

    def is_empty(self) -> bool:
        return self.tool is NO_VAL and self.result is NO_VAL and self.resolved_args is NO_VAL

    def is_prepared(self) -> bool:
        return self.tool is not NO_VAL and self.resolved_args is not NO_VAL and self.result is NO_VAL

    def is_executed(self) -> bool:
        return self.result is not NO_VAL

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "tool": self.tool,
            "args": self.args,
            "resolved_args": self.resolved_args,
            "result": self.result,
            "error": self.error,
            "completed": bool(self.is_executed()),
        }