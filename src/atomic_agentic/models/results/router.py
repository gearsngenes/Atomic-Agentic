from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .atomic import AtomicResult

__all__ = ["RouterResult"]


@dataclass(frozen=True, slots=True)
class RouterResult(AtomicResult):
    """Result of one ``Router`` run.

    Fields
    ------
    route:
        The selector's response, the value that was matched against the route keys.
    branch:
        ``name`` of the branch invokable that ran. ``result`` is that branch's
        raw response.
    """

    route: Any = field(default=None, kw_only=True)
    branch: str = field(default="", kw_only=True)
