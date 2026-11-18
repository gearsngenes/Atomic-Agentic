"""``PromptTemplate``: an invokable that renders a template string."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any

from ..models.parameters import ParamSpec
from ..utils.parameters import _normalize_prompt_template
from .atomic_invokable import AtomicInvokable

__all__ = ["PromptTemplate"]


class PromptTemplate(AtomicInvokable):
    """An invokable that renders a template string and returns ``str``.

    The template is normalized at construction: clean ``{identifier}`` fields
    (optionally ``{name!conv}``, ``{name:spec}``) stay as placeholders and every
    other brace region (``{}``, ``{0}``, ``{a.b}``, ``{x[0]}``, nested or
    unbalanced braces) is escaped into literal text. ``template`` holds that
    normalized form.

    ``parameters=None`` discovers every placeholder as a ``POSITIONAL_OR_KEYWORD``
    parameter of type ``Any``. An explicit list is exclusive: its order is the
    signature order, entries are names (``str``) or full ``ParamSpec`` objects,
    and placeholders that are not listed are escaped into literal text. An empty
    list declares no parameters, so every placeholder renders literally.

    ``result_cls`` is the inherited ``AtomicResult``; ``return_type`` is
    ``("str",)``. Caller values are arguments to ``str.format`` and are never
    re-parsed, so braces in a value pass through verbatim.
    """

    def __init__(
        self,
        template: str,
        name: str,
        description: str,
        parameters: Sequence[str | ParamSpec] | None = None,
    ) -> None:
        """
        1. Validate ``template`` (``str``, not blank) and ``parameters`` shape.
        2. Normalize the template, escaping unlisted placeholders when
           ``parameters`` is given.
        3. Build the ``ParamSpec`` list (discovered, or the explicit entries
           re-indexed to their list position); an unknown listed name raises.
        4. Defer duplicate-name and ordering checks to ``AtomicInvokable``.
        """
        # 1. Validate arguments
        if not isinstance(template, str):
            raise TypeError(
                f"PromptTemplate template must be a str, got {type(template).__name__}"
            )
        if not template.strip():
            raise ValueError("PromptTemplate template must not be empty or blank")
        if parameters is not None:
            if not isinstance(parameters, (list, tuple)):
                raise TypeError(
                    f"PromptTemplate parameters must be a list, tuple or None, "
                    f"got {type(parameters).__name__}"
                )
            for entry in parameters:
                if not isinstance(entry, (str, ParamSpec)):
                    raise TypeError(
                        f"PromptTemplate parameters entries must be str or "
                        f"ParamSpec, got {type(entry).__name__}"
                    )
                if isinstance(entry, ParamSpec) and entry.kind in (
                    ParamSpec.VAR_POSITIONAL,
                    ParamSpec.VAR_KEYWORD,
                ):
                    raise ValueError(
                        f"PromptTemplate parameter {entry.name!r} must not be "
                        f"variadic ({entry.kind})"
                    )

        # 2. Normalize the template against the listed names
        entry_names = (
            None
            if parameters is None
            else [e if isinstance(e, str) else e.name for e in parameters]
        )
        normalized, discovered = _normalize_prompt_template(
            template, None if entry_names is None else set(entry_names)
        )

        # 3. Build parameters
        def default_spec(param_name: str, index: int) -> ParamSpec:
            return ParamSpec(
                name=param_name,
                index=index,
                kind=ParamSpec.POSITIONAL_OR_KEYWORD,
                type="Any",
            )

        if parameters is None:
            built = [default_spec(n, i) for i, n in enumerate(discovered)]
        else:
            absent = [n for n in entry_names if n not in discovered]
            if absent:
                raise ValueError(
                    f"PromptTemplate parameters not found as placeholders in "
                    f"the template: {absent!r}"
                )
            built = [
                default_spec(entry, i)
                if isinstance(entry, str)
                else dataclasses.replace(entry, index=i)
                for i, entry in enumerate(parameters)
            ]

        # 4. Base construction (duplicates and ordering are checked there)
        super().__init__(
            name=name,
            description=description,
            parameters=built,
            return_type="str",
        )
        self._template = normalized

    @property
    def template(self) -> str:
        """The normalized template string (escaped literals, clean fields only)."""
        return self._template

    def _invoke_impl(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Format the template with the filtered inputs; errors propagate unwrapped."""
        return {"result": self._template.format(**inputs)}

    async def _async_invoke_impl(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Pure CPU, so no thread offload."""
        return self._invoke_impl(inputs)

    def to_dict(self) -> dict[str, Any]:
        """Standard invokable metadata plus the normalized ``template``."""
        d = super().to_dict()
        d["template"] = self._template
        return d
