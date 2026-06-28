from __future__ import annotations

import string
from dataclasses import InitVar, dataclass, field
from typing import Any

from ..parameters import ParamSpec
from ...constants.core import NO_VAL

__all__ = ["PromptConfig"]


@dataclass(frozen=True, slots=True)
class PromptConfig:
    """
    Pairs a format-string template with a parameter schema auto-discovered
    from that template.

    Construction only requires ``template`` and ``description``. Named
    ``{placeholder}`` slots are extracted at construction time and become
    ``KEYWORD_ONLY`` ``ParamSpec`` entries in ``parameters``, ordered by
    first appearance. Duplicate placeholders produce a single entry.

    ``defaults`` maps placeholder names to their default values. Any key
    in ``defaults`` that is not found in the template raises ``ValueError``
    at construction.

    Only simple named placeholders are supported. Positional (``{}``,
    ``{0}``) and attribute/index expressions (``{obj.attr}``, ``{x[0]}``)
    raise ``ValueError`` at construction.

    ``description`` is a human-readable label for prompt-handoff and
    selection scenarios — it is not used during rendering.
    """

    template: str
    description: str
    defaults: InitVar[dict[str, Any] | None] = None
    parameters: tuple[ParamSpec, ...] = field(init=False)

    def __post_init__(self, defaults: dict[str, Any] | None) -> None:
        if not isinstance(self.template, str):
            raise TypeError(
                f"PromptConfig.template must be a str, got {type(self.template).__name__}"
            )
        if not isinstance(self.description, str):
            raise TypeError(
                f"PromptConfig.description must be a str, "
                f"got {type(self.description).__name__}"
            )
        if defaults is None:
            defaults = {}
        if not isinstance(defaults, dict):
            raise TypeError(
                f"PromptConfig.defaults must be a dict or None, "
                f"got {type(defaults).__name__}"
            )

        # discover placeholder names in template order, deduplicating
        names: list[str] = []
        seen: set[str] = set()
        for _, field_name, _, _ in string.Formatter().parse(self.template):
            if field_name is None:
                continue
            if not field_name.isidentifier():
                raise ValueError(
                    f"PromptConfig: template contains unsupported placeholder "
                    f"{{{field_name!r}}}. Only simple named keyword placeholders "
                    "are allowed (no positional, attribute, or index expressions)."
                )
            if field_name not in seen:
                names.append(field_name)
                seen.add(field_name)

        extra = set(defaults) - seen
        if extra:
            raise ValueError(
                f"PromptConfig: defaults contains key(s) not found in template: "
                f"{sorted(extra)!r}"
            )

        parameters = tuple(
            ParamSpec(
                name=name,
                index=i,
                kind=ParamSpec.KEYWORD_ONLY,
                type="Any",
                default=defaults.get(name, NO_VAL),
            )
            for i, name in enumerate(names)
        )
        object.__setattr__(self, "parameters", parameters)

    def render(self, inputs: dict[str, Any]) -> str:
        """
        Format the template from domain inputs.

        For each declared parameter: use the caller-supplied value if present;
        fill the declared default if absent and optional; raise ``ValueError``
        if absent and required. Variables not declared in ``self.parameters``
        are ignored even if present in ``inputs``.
        """
        format_dict: dict[str, Any] = {}
        for param in self.parameters:
            if param.name in inputs:
                format_dict[param.name] = inputs[param.name]
            elif param.default is not NO_VAL:
                format_dict[param.name] = param.default
            else:
                raise ValueError(
                    f"PromptConfig.render: required parameter {param.name!r} "
                    "is missing from inputs."
                )
        return self.template.format(**format_dict)
