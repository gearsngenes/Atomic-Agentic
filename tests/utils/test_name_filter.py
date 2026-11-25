from __future__ import annotations

import pytest

from atomic_agentic.utils.core import apply_name_filter, validate_name_filter


class TestValidateNameFilterStripsNames:
    def test_include_names_are_stored_stripped(self) -> None:
        include, exclude = validate_name_filter(["  a "], None)

        assert include == frozenset({"a"})
        assert exclude is None

    def test_exclude_names_are_stored_stripped(self) -> None:
        include, exclude = validate_name_filter(None, [" b", "c  "])

        assert include is None
        assert exclude == frozenset({"b", "c"})

    def test_padded_include_matches_the_bare_name(self) -> None:
        include, exclude = validate_name_filter([" a"], None)

        assert apply_name_filter({"a": 1, "b": 2}, include, exclude) == {"a": 1}

    def test_overlap_is_detected_after_stripping(self) -> None:
        with pytest.raises(ValueError, match="overlap"):
            validate_name_filter(["a"], [" a "])

    def test_blank_entries_still_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty strings"):
            validate_name_filter(["  "], None)
