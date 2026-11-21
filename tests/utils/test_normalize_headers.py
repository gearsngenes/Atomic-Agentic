from __future__ import annotations

import pytest

from atomic_agentic.utils.core import normalize_headers


class TestNormalizeHeaders:
    def test_none_returns_none(self) -> None:
        assert normalize_headers(None) is None

    def test_scalar_values_are_normalized(self) -> None:
        headers = normalize_headers(
            {
                "X-Int": 1,
                "X-Float": 1.5,
                "X-Bool": True,
                "X-Bytes": b"abc",
                "X-ByteArray": bytearray(b"xyz"),
            }  # type: ignore[dict-item]
        )

        assert headers == {
            "X-Int": "1",
            "X-Float": "1.5",
            "X-Bool": "true",
            "X-Bytes": "abc",
            "X-ByteArray": "xyz",
        }

    def test_result_is_read_only(self) -> None:
        headers = normalize_headers({"Authorization": "Bearer secret"})

        with pytest.raises(TypeError):
            headers["Authorization"] = "changed"  # type: ignore[index]

    @pytest.mark.parametrize(
        ("headers", "match"),
        [
            ("bad", "headers must be a mapping"),
            ({1: "ok"}, "header names must be strings"),
            ({"bad": None}, "must not be None"),
            ({"bad": {"nested": "value"}}, "must not be a mapping"),
            ({"bad": ["value"]}, "must not be a collection"),
            ({"bad": object()}, "must be str, int, float, bool, bytes, or bytearray"),
            ({"bad": b"\xff"}, "ASCII-decodable"),
            ({"bad\n": "value"}, "forbidden character"),
            ({"bad": "line\nbreak"}, "forbidden character"),
        ],
    )
    def test_invalid_headers_raise_value_error(self, headers: object, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            normalize_headers(headers)  # type: ignore[arg-type]
