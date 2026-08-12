from __future__ import annotations

from dataclasses import dataclass

__all__ = ["StatePolicySpec"]


@dataclass(frozen=True, slots=True)
class StatePolicySpec:
    """Collision-resolution policy for one accumulated-state key.

    Fields
    ------
    key:
        The state key this policy governs.
    raise_on_collision:
        If True, a same-superstep write collision on this key raises
        ``ValidationError`` instead of being resolved by priority/tiebreak.
        Cross-superstep writes to the same key are never governed by this
        policy -- they're ordinary sequential overwrite.
    tiebreak:
        ``"first"`` or ``"last"`` -- direction used to break equal-priority
        collisions on this key. Must be ``None`` when ``raise_on_collision``
        is ``True``.
    """

    key: str
    raise_on_collision: bool
    tiebreak: str | None = None

    def __post_init__(self) -> None:
        """Validate field shape and the raise/tiebreak mutual exclusion."""
        if not isinstance(self.key, str) or not self.key:
            raise TypeError(
                f"StatePolicySpec.key must be a non-empty str, got {self.key!r}"
            )
        if not isinstance(self.raise_on_collision, bool):
            raise TypeError(
                "StatePolicySpec.raise_on_collision must be a bool, got "
                f"{type(self.raise_on_collision)!r}"
            )
        if self.raise_on_collision and self.tiebreak is not None:
            raise ValueError(
                "StatePolicySpec.tiebreak must be None when raise_on_collision is True"
            )
        if self.tiebreak is not None and self.tiebreak not in ("first", "last"):
            raise ValueError(
                f"StatePolicySpec.tiebreak must be 'first', 'last', or None, "
                f"got {self.tiebreak!r}"
            )
