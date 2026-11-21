from __future__ import annotations

from dataclasses import dataclass, field

from .atomic import AtomicResult

__all__ = ["GraphResult"]


@dataclass(frozen=True, slots=True)
class GraphResult(AtomicResult):
    """Result of one ``Graph`` run.

    Fields
    ------
    trace:
        Node names fired per superstep (declaration order within a superstep),
        always populated. Routers do not appear.
    """

    trace: tuple[tuple[str, ...], ...] = field(default=(), kw_only=True)
