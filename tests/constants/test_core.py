from __future__ import annotations

from atomic_agentic.constants.core import NO_VAL


class TestNoValSentinel:
    def test_no_val_repr_is_stable(self) -> None:
        assert repr(NO_VAL) == "NO_VAL"

    def test_no_val_is_singleton_import_identity(self) -> None:
        from atomic_agentic.constants.core import NO_VAL as imported_again

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


class TestNoValSurvivesCopying:
    """`is NO_VAL` is the contract; copying or pickling must not break it."""

    def test_copy_and_deepcopy_return_the_sentinel(self) -> None:
        import copy

        from atomic_agentic.constants.core import NO_VAL

        assert copy.copy(NO_VAL) is NO_VAL
        assert copy.deepcopy(NO_VAL) is NO_VAL

    def test_pickle_round_trip_returns_the_sentinel(self) -> None:
        import pickle

        from atomic_agentic.constants.core import NO_VAL

        assert pickle.loads(pickle.dumps(NO_VAL)) is NO_VAL

    def test_deepcopied_paramspec_keeps_its_no_default(self) -> None:
        import copy

        from atomic_agentic.constants.core import NO_VAL
        from atomic_agentic.models.parameters import ParamSpec

        spec = ParamSpec(name="q", index=0, kind=ParamSpec.KEYWORD_ONLY, type="str")
        clone = copy.deepcopy(spec)

        assert clone == spec
        assert clone.default is NO_VAL
