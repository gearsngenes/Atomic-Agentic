"""Shared binding grammar: shape normalization and response slicing.

``normalize_binding`` applies only the shape rules (str / iterable of names /
mapping, empty, stripping, duplicates). Identifier and reserved-name rules,
selector contents and destination checks belong to the caller. ``None`` is
passed through so a caller can give it a meaning distinct from the empty
binding ``{}``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Set as AbstractSet
from typing import Any

from ..models.binding import Binding, MissingPolicy


def normalize_binding(
    value: Any,
    *,
    label: str,
    allow_str: bool = True,
) -> Binding | None:
    """Normalize one binding value by shape only.

    1. ``None`` -> ``None`` (distinct from empty; the caller decides).
    2. ``str`` -> the whole-response form (``TypeError`` when ``allow_str`` is
       false).
    3. ``Mapping`` -> ``{selector: destination}``; any other non-bytes iterable
       of names -> ``{n: n}`` (sorted when unordered); anything else raises
       ``TypeError``.
    4. No names -> ``{}``; names are stripped; duplicate names raise
       ``ValueError``.
    """
    if value is None:
        return None

    if isinstance(value, str):
        if not allow_str:
            raise TypeError(
                f"{label} does not accept a bare str; pass a collection of "
                "input names or a {input: destination} mapping."
            )
        selectors, names = None, [value]
    elif isinstance(value, Mapping):
        selectors, names = list(value), list(value.values())
    elif isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
        selectors, names = None, list(value)
        if isinstance(value, AbstractSet) and all(isinstance(n, str) for n in names):
            names.sort()  # unordered: store deterministically
    else:
        raise TypeError(
            f"{label} must be None, "
            + ("a str, " if allow_str else "")
            + "a collection of str, or a mapping of "
            + ("selector" if allow_str else "input name")
            + f" -> str; got {type(value).__name__}."
        )

    if not names:
        return {}

    names = [n.strip() if isinstance(n, str) else n for n in names]
    if any(n in names[:i] for i, n in enumerate(names)):  # names may be unhashable
        raise ValueError(f"{label} has duplicate destinations.")

    if isinstance(value, str):
        return names[0]
    try:
        return dict(zip(selectors if selectors is not None else names, names))
    except TypeError as exc:  # an unhashable selector or name
        raise TypeError(f"{label} contains an unhashable entry: {exc}.") from exc


def slice_response(
    response: Any,
    binding: Binding,
    *,
    label: str,
    on_missing: MissingPolicy,
) -> dict[str, Any]:
    """Resolve ``binding`` against ``response``; pure, mutates neither.

    1. A ``str`` binding returns ``{binding: response}`` (any response type);
       an empty binding returns ``{}``.
    2. A ``str``/``bytes`` response with selectors raises ``TypeError``.
    3. Each selector is indexed in order. ``on_missing="skip"`` leaves a
       ``KeyError``/``IndexError`` selector out of the result; ``"raise"`` lets
       it propagate. Any other error propagates. A present ``None`` is bound.
    """
    if on_missing not in ("skip", "raise"):
        raise ValueError(f"on_missing must be 'skip' or 'raise'; got {on_missing!r}.")
    if isinstance(binding, str):
        return {binding: response}
    if not binding:
        return {}
    if isinstance(response, (str, bytes, bytearray)):
        raise TypeError(
            f"{label} selectors cannot index a str/bytes response; "
            "use the str form to bind the whole response."
        )
    bound = {}
    for selector, destination in binding.items():
        try:
            bound[destination] = response[selector]
        except (KeyError, IndexError):
            if on_missing == "raise":
                raise
    return bound


__all__ = ["normalize_binding", "slice_response"]
