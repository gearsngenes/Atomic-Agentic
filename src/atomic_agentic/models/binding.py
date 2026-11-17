"""Binding types shared by agents and, later, workflows and graph nodes.

A binding says how to pull named pieces out of a response (or write inputs to
named destinations). The grammar functions live in ``utils/binding.py``.
"""

from __future__ import annotations

from typing import Any, Literal, TypeAlias

# A normalized response/input binding: a ``str`` is the whole response under
# one name; a dict is ``{selector: destination}``; ``{}`` means no binding.
Binding: TypeAlias = "str | dict[Any, str]"

# What slicing does when a selector is absent from the response. The caller
# chooses; there is no default.
MissingPolicy: TypeAlias = Literal["skip", "raise"]

__all__ = ["Binding", "MissingPolicy"]
