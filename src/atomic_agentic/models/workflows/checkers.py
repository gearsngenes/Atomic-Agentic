from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ...core.Invokable import AtomicInvokable

__all__ = ["CheckerSpec"]


@dataclass(frozen=True, slots=True)
class CheckerSpec:
    """One early-exit gate for IterativeFlow, tied to a body-step position.

    Fields
    ------
    index:
        Body-step position this checker fires at.
    judge:
        Invoked with that step's own (mapping-shaped) result when this
        position is reached.
    approval_value:
        Compared via ``==`` against the judge's unwrapped result; a match
        stops the loop immediately.
    """

    index: int
    judge: AtomicInvokable
    approval_value: Any
