from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone

import pytest

from atomic_agentic.models.results.atomic import AtomicResult


def make_result(**overrides) -> AtomicResult:
    started_at = datetime.now(timezone.utc)
    kwargs = dict(
        result="payload",
        invoker_id="Tool.name",
        started_at=started_at,
        ended_at=started_at + timedelta(seconds=2),
    )
    kwargs.update(overrides)
    return AtomicResult(**kwargs)


class TestAtomicResultConstruction:
    def test_valid_result_exposes_fields(self) -> None:
        started_at = datetime.now(timezone.utc)
        ended_at = started_at + timedelta(seconds=1)
        result = AtomicResult(
            result=[1, 2],
            invoker_id="Tool.name",
            started_at=started_at,
            ended_at=ended_at,
        )
        assert result.result == [1, 2]
        assert result.invoker_id == "Tool.name"
        assert result.started_at == started_at
        assert result.ended_at == ended_at

    def test_invoker_id_is_stripped(self) -> None:
        assert make_result(invoker_id="  Tool.name  ").invoker_id == "Tool.name"

    def test_datetimes_normalized_to_utc(self) -> None:
        tz = timezone(timedelta(hours=5))
        started_at = datetime(2026, 1, 1, 12, 0, tzinfo=tz)
        result = make_result(started_at=started_at, ended_at=started_at + timedelta(seconds=1))
        assert result.started_at.utcoffset() == timedelta(0)
        assert result.started_at == started_at
        assert result.ended_at.utcoffset() == timedelta(0)

    def test_naive_started_at_raises(self) -> None:
        naive = datetime(2026, 1, 1, 12, 0)
        with pytest.raises(ValueError, match="started_at"):
            make_result(started_at=naive, ended_at=datetime.now(timezone.utc))

    def test_naive_ended_at_raises(self) -> None:
        with pytest.raises(ValueError, match="ended_at"):
            make_result(ended_at=datetime(2026, 1, 1, 12, 0))

    def test_ended_before_started_raises(self) -> None:
        started_at = datetime.now(timezone.utc)
        with pytest.raises(ValueError, match="ended_at"):
            make_result(started_at=started_at, ended_at=started_at - timedelta(seconds=1))

    @pytest.mark.parametrize("bad", ["", "   "])
    def test_empty_invoker_id_raises(self, bad: str) -> None:
        with pytest.raises(ValueError, match="invoker_id"):
            make_result(invoker_id=bad)

    def test_non_str_invoker_id_raises(self) -> None:
        with pytest.raises(TypeError, match="invoker_id"):
            make_result(invoker_id=123)

    def test_non_datetime_raises(self) -> None:
        with pytest.raises(TypeError, match="started_at"):
            make_result(started_at="now")

    def test_run_id_keyword_is_rejected(self) -> None:
        with pytest.raises(TypeError, match="run_id"):
            make_result(run_id="r1")


class TestAtomicResultElapsedTime:
    def test_elapsed_time_is_derived_from_timestamps(self) -> None:
        started_at = datetime.now(timezone.utc)
        ended_at = started_at + timedelta(seconds=2, milliseconds=500)
        result = make_result(started_at=started_at, ended_at=ended_at)
        assert result.elapsed_time == (ended_at - started_at).total_seconds()
        assert result.elapsed_time == 2.5

    def test_elapsed_time_has_no_setter(self) -> None:
        result = make_result()
        with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):  # frozen+slots dataclass raises TypeError for non-field names
            result.elapsed_time = 1.0  # type: ignore[misc]

    def test_old_fields_are_gone(self) -> None:
        result = make_result()
        assert not hasattr(result, "elapsed_s")
        assert not hasattr(result, "run_id")

    def test_is_frozen(self) -> None:
        result = make_result()
        with pytest.raises(FrozenInstanceError):
            result.result = "other"  # type: ignore[misc]


class TestAtomicResultToDict:
    def test_to_dict_keys(self) -> None:
        d = make_result().to_dict()
        assert set(d) == {"invoker_id", "started_at", "ended_at", "elapsed_time", "result"}

    def test_to_dict_values(self) -> None:
        result = make_result()
        d = result.to_dict()
        assert d["invoker_id"] == "Tool.name"
        assert d["started_at"] == result.started_at.isoformat()
        assert d["ended_at"] == result.ended_at.isoformat()
        assert d["elapsed_time"] == 2.0
        assert d["result"] == "payload"
