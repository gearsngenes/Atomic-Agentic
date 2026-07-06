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
    ``{placeholder}`` slots that are valid Python identifiers are extracted at
    construction time and become ``KEYWORD_ONLY`` ``ParamSpec`` entries in
    ``parameters``, ordered by first appearance. Duplicate placeholders produce
    a single entry. Non-identifier expressions (``{}``, ``{0}``,
    ``{obj.attr}``, ``{x[0]}``) are silently skipped and passed through as
    literal text during rendering.

    ``defaults`` maps placeholder names to their default values. Any key
    in ``defaults`` that is not found in the template raises ``ValueError``
    at construction.

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

        # Discover identifier placeholder names in template order, deduplicating.
        # Non-identifier expressions are skipped here and handled in render().
        names: list[str] = []
        seen: set[str] = set()
        for _, field_name, _, _ in string.Formatter().parse(self.template):
            if field_name is None:
                continue
            if not field_name.isidentifier():
                continue
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
        if absent and required. Non-identifier brace expressions not in
        ``self.parameters`` are passed through as literal text.
        """
        format_dict: dict[str, Any] = {}
        param_names = {p.name for p in self.parameters}
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
        parts: list[str] = []
        for literal_text, field_name, format_spec, conversion in string.Formatter().parse(self.template):
            if literal_text:
                parts.append(literal_text)
            if field_name is None:
                continue
            if field_name in param_names:
                val = format_dict[field_name]
                if conversion == "r":
                    val = repr(val)
                elif conversion == "s":
                    val = str(val)
                elif conversion == "a":
                    val = ascii(val)
                parts.append(format(val, format_spec) if format_spec else str(val))
            else:
                expr = field_name
                if conversion:
                    expr += f"!{conversion}"
                if format_spec:
                    expr += f":{format_spec}"
                parts.append("{" + expr + "}")
        return "".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable representation of this config."""
        return {
            "template": self.template,
            "description": self.description,
            "defaults": {
                p.name: p.default
                for p in self.parameters
                if p.default is not NO_VAL
            },
        }
