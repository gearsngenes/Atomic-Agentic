import pytest

from atomic_agentic.utils.core import normalize_timeout_seconds


@pytest.mark.parametrize("value, expected", [(5, 5.0), (0.5, 0.5), (600, 600.0)])
def test_positive_numbers_become_floats(value, expected) -> None:
    result = normalize_timeout_seconds(value, "t")

    assert result == expected
    assert isinstance(result, float)


@pytest.mark.parametrize("value", [0, -1, float("nan"), True, "5", None, [1]])
def test_invalid_values_raise_naming_the_parameter(value) -> None:
    with pytest.raises(ValueError, match="my_timeout must be a positive number"):
        normalize_timeout_seconds(value, "my_timeout")
