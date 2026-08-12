from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from atomic_agentic.models.workflows.graph import StatePolicySpec


class TestStatePolicySpec:
    def test_construction_stores_all_fields(self) -> None:
        spec = StatePolicySpec(key="x", raise_on_collision=False, tiebreak="first")

        assert spec.key == "x"
        assert spec.raise_on_collision is False
        assert spec.tiebreak == "first"

    def test_tiebreak_defaults_to_none(self) -> None:
        spec = StatePolicySpec(key="x", raise_on_collision=False)
        assert spec.tiebreak is None

    def test_key_must_be_non_empty_str(self) -> None:
        with pytest.raises(TypeError, match="key must be a non-empty str"):
            StatePolicySpec(key="", raise_on_collision=False)

        with pytest.raises(TypeError, match="key must be a non-empty str"):
            StatePolicySpec(key=123, raise_on_collision=False)  # type: ignore[arg-type]

    def test_raise_on_collision_must_be_bool(self) -> None:
        with pytest.raises(TypeError, match="raise_on_collision must be a bool"):
            StatePolicySpec(key="x", raise_on_collision="yes")  # type: ignore[arg-type]

    def test_raise_on_collision_true_with_tiebreak_raises(self) -> None:
        with pytest.raises(
            ValueError, match="tiebreak must be None when raise_on_collision is True"
        ):
            StatePolicySpec(key="x", raise_on_collision=True, tiebreak="first")

    def test_raise_on_collision_true_with_no_tiebreak_is_valid(self) -> None:
        spec = StatePolicySpec(key="x", raise_on_collision=True)
        assert spec.tiebreak is None

    def test_invalid_tiebreak_value_raises(self) -> None:
        with pytest.raises(ValueError, match="tiebreak must be 'first', 'last', or None"):
            StatePolicySpec(key="x", raise_on_collision=False, tiebreak="middle")

    def test_is_frozen(self) -> None:
        spec = StatePolicySpec(key="x", raise_on_collision=False)
        with pytest.raises(FrozenInstanceError):
            spec.key = "y"  # type: ignore[misc]
