from __future__ import annotations

import pytest

from atomic_agentic.utils.binding import normalize_binding, slice_response


def _norm(value, **kw):
    return normalize_binding(value, label="route", **kw)


class TestNormalizeBinding:
    def test_none_passes_through(self):
        assert _norm(None) is None

    def test_str(self):
        assert _norm("raw") == "raw"

    def test_str_is_stripped(self):
        assert _norm("  raw  ") == "raw"

    def test_list(self):
        assert _norm(["a", "b"]) == {"a": "a", "b": "b"}

    def test_tuple(self):
        assert _norm(("b", "a")) == {"b": "b", "a": "a"}

    def test_set_is_sorted(self):
        assert list(_norm({"c", "a", "b"})) == ["a", "b", "c"]

    def test_dict_view(self):
        assert _norm({"x": 1, "y": 2}.keys()) == {"x": "x", "y": "y"}

    def test_generator(self):
        assert _norm(n for n in ("p", "q")) == {"p": "p", "q": "q"}

    def test_mapping(self):
        assert _norm({"k": "dest", "j": "other"}) == {"k": "dest", "j": "other"}

    def test_mapping_with_non_str_selector(self):
        assert _norm({0: "first", 2: "third"}) == {0: "first", 2: "third"}

    @pytest.mark.parametrize("empty", [[], (), {}, set()])
    def test_empty_collections_are_empty_dict_not_none(self, empty):
        result = _norm(empty)
        assert result == {}
        assert result is not None

    def test_names_are_stripped(self):
        assert _norm([" a ", "b "]) == {"a": "a", "b": "b"}
        assert _norm({"k": " d "}) == {"k": "d"}

    def test_non_str_name_passes_through(self):
        assert _norm([1, "a"]) == {1: 1, "a": "a"}

    def test_duplicate_in_list(self):
        with pytest.raises(ValueError, match="duplicate"):
            _norm(["a", "a"])

    def test_duplicate_in_mapping(self):
        with pytest.raises(ValueError, match="duplicate"):
            _norm({"k1": "d", "k2": "d"})

    def test_duplicate_after_stripping(self):
        with pytest.raises(ValueError, match="duplicate"):
            _norm(["a", " a"])

    def test_bare_str_rejected_when_not_allowed(self):
        with pytest.raises(TypeError, match="bare str"):
            _norm("x", allow_str=False)

    def test_collection_allowed_when_str_not_allowed(self):
        assert _norm(["x"], allow_str=False) == {"x": "x"}

    @pytest.mark.parametrize("bad", [5, b"abc", bytearray(b"x"), 1.5])
    def test_wrong_type_raises_type_error(self, bad):
        with pytest.raises(TypeError, match="route"):
            _norm(bad)

    def test_wrong_type_message_names_type(self):
        with pytest.raises(TypeError, match="int"):
            _norm(5)


def _slice(response, binding, on_missing="skip"):
    return slice_response(response, binding, label="route", on_missing=on_missing)


class TestSliceResponseSkip:
    @pytest.mark.parametrize("response", ["text", b"bytes", 5, None, [1], {"a": 1}])
    def test_str_binding_binds_whole_response(self, response):
        assert _slice(response, "raw") == {"raw": response}

    def test_dict_response_present(self):
        assert _slice({"a": 1, "b": 2}, {"a": "x", "b": "y"}) == {"x": 1, "y": 2}

    def test_dict_response_missing_key_skipped(self):
        assert _slice({"a": 1}, {"a": "x", "z": "y"}) == {"x": 1}

    def test_list_response_present(self):
        assert _slice([10, 20], {1: "second", 0: "first"}) == {"second": 20, "first": 10}

    def test_list_response_missing_index_skipped(self):
        assert _slice([10], {0: "first", 5: "far"}) == {"first": 10}

    def test_present_none_is_bound(self):
        assert _slice({"a": None}, {"a": "x"}) == {"x": None}

    def test_empty_binding(self):
        assert _slice({"a": 1}, {}) == {}

    @pytest.mark.parametrize("response", ["text", b"bytes"])
    def test_str_bytes_response_with_mapping_raises(self, response):
        with pytest.raises(TypeError, match="str/bytes"):
            _slice(response, {0: "x"})

    def test_non_subscriptable_response_natural_type_error(self):
        with pytest.raises(TypeError, match="subscriptable"):
            _slice(5, {"a": "x"})

    def test_inputs_not_mutated(self):
        response = {"a": 1}
        binding = {"a": "x", "z": "y"}
        _slice(response, binding)
        assert response == {"a": 1}
        assert binding == {"a": "x", "z": "y"}


class TestSliceResponseRaise:
    def test_missing_key_raises_key_error(self):
        with pytest.raises(KeyError):
            _slice({"a": 1}, {"z": "y"}, on_missing="raise")

    def test_missing_index_raises_index_error(self):
        with pytest.raises(IndexError):
            _slice([1], {3: "y"}, on_missing="raise")

    def test_present_values_bind(self):
        assert _slice({"a": 1, "b": None}, {"a": "x", "b": "y"}, on_missing="raise") == {
            "x": 1,
            "y": None,
        }

    def test_str_binding_still_binds_whole_response(self):
        assert _slice(None, "raw", on_missing="raise") == {"raw": None}


class TestOnMissingArgument:
    def test_required(self):
        with pytest.raises(TypeError):
            slice_response({"a": 1}, {"a": "x"}, label="route")  # type: ignore[call-arg]

    def test_keyword_only(self):
        with pytest.raises(TypeError):
            slice_response({"a": 1}, {"a": "x"}, "route", "skip")  # type: ignore[misc]

    def test_unknown_value_raises(self):
        with pytest.raises(ValueError, match="on_missing"):
            _slice({"a": 1}, {"a": "x"}, on_missing="ignore")
