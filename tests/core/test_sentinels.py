from __future__ import annotations

from atomic_agentic.core.sentinels import NO_VAL


class TestNoValSentinel:
    def test_no_val_repr_is_stable(self) -> None:
        assert repr(NO_VAL) == "NO_VAL"

    def test_no_val_is_singleton_import_identity(self) -> None:
        from atomic_agentic.core.sentinels import NO_VAL as imported_again

        assert imported_again is NO_VAL

    def test_no_val_is_not_none(self) -> None:
        assert NO_VAL is not None

    def test_no_val_is_identity_checkable_in_containers(self) -> None:
        payload = {"missing": NO_VAL}

        assert payload["missing"] is NO_VAL

    def test_no_val_does_not_compare_equal_to_common_absence_values(self) -> None:
        assert NO_VAL != None  # noqa: E711
        assert NO_VAL != False  # noqa: E712
        assert NO_VAL != 0
        assert NO_VAL != ""
        assert NO_VAL != []
        assert NO_VAL != {}