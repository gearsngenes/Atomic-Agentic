from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from typing import Any

from ..parameters import ParamSpec
from ...constants.core import NO_VAL
from ...utils.parameters import _normalize_prompt_template

__all__ = ["PromptConfig"]


@dataclass(frozen=True, slots=True)
class PromptConfig:
    """
    Pairs a format-string template with a parameter schema auto-discovered
    from that template.

    Construction only requires ``template`` and ``description``. The raw
    ``template`` is normalized at construction time: clean, self-contained
    ``{name}``-style identifier fields (optionally ``{name!conv}``,
    ``{name:spec}``, ``{name!conv:spec}``) become ``KEYWORD_ONLY`` ``ParamSpec``
    entries in ``parameters``, ordered by first appearance and deduplicated.
    Every other brace region — non-identifier shapes (``{}``, ``{0}``,
    ``{obj.attr}``, ``{x[0]}``), a field nested inside another field, or a
    genuinely unbalanced/stray brace — is escaped into inert literal text
    instead. Construction never raises due to template shape.

    ``self.template`` after construction holds this *normalized* form, not
    necessarily a byte-identical copy of the constructor argument — this is
    intentional and is the only stored copy (no separate raw/normalized
    fields).

    ``field_specs`` maps discovered identifier names to a sparse
    ``{"description": str, "type": str, "default": Any}`` dict (any subset).
    A key not found among discovered identifiers raises ``ValueError`` at
    construction; an inner key outside ``{"description", "type", "default"}``
    also raises ``ValueError``. A field is required iff its entry omits
    ``"default"``.

    ``description`` is a human-readable label for prompt-handoff and
    selection scenarios — it is not used during rendering.
    """

    template: str
    description: str
    field_specs: InitVar[dict[str, dict[str, Any]] | None] = None
    parameters: tuple[ParamSpec, ...] = field(init=False)

    def __post_init__(self, field_specs: dict[str, dict[str, Any]] | None) -> None:
        if not isinstance(self.template, str):
            raise TypeError(
                f"PromptConfig.template must be a str, got {type(self.template).__name__}"
            )
        if not isinstance(self.description, str):
            raise TypeError(
                f"PromptConfig.description must be a str, "
                f"got {type(self.description).__name__}"
            )

        # Normalize the raw template in place; discover its identifier fields.
        normalized, discovered = _normalize_prompt_template(self.template)
        object.__setattr__(self, "template", normalized)

        if field_specs is None:
            field_specs = {}
        if not isinstance(field_specs, dict):
            raise TypeError(
                f"PromptConfig.field_specs must be a dict or None, "
                f"got {type(field_specs).__name__}"
            )

        extra = set(field_specs) - set(discovered)
        if extra:
            raise ValueError(
                f"PromptConfig: field_specs contains key(s) not found in "
                f"template: {sorted(extra)!r}"
            )

        allowed_keys = {"description", "type", "default"}
        for name, spec in field_specs.items():
            if not isinstance(spec, dict):
                raise TypeError(
                    f"PromptConfig.field_specs[{name!r}] must be a dict, "
                    f"got {type(spec).__name__}"
                )
            invalid_keys = set(spec) - allowed_keys
            if invalid_keys:
                raise ValueError(
                    f"PromptConfig.field_specs[{name!r}] contains invalid "
                    f"key(s): {sorted(invalid_keys)!r}; allowed keys are "
                    f"'description', 'type', 'default'"
                )

        parameters = tuple(
            ParamSpec(
                name=name,
                index=i,
                kind=ParamSpec.KEYWORD_ONLY,
                type=field_specs.get(name, {}).get("type", "Any"),
                default=field_specs.get(name, {}).get("default", NO_VAL),
                description=field_specs.get(name, {}).get("description"),
            )
            for i, name in enumerate(discovered)
        )
        object.__setattr__(self, "parameters", parameters)

    def render(self, inputs: dict[str, Any]) -> str:
        """
        Format the normalized template from domain inputs.

        For each declared parameter: use the caller-supplied value if present,
        else its declared default. Any parameter with neither is collected and
        raised as one ``ValueError`` naming every missing parameter. Otherwise
        formats via the normalized template directly — non-field brace content
        was already neutralized at construction, so no reconstruction is
        needed here.
        """
        resolved: dict[str, Any] = {}
        missing: list[str] = []
        for param in self.parameters:
            if param.name in inputs:
                resolved[param.name] = inputs[param.name]
            elif param.default is not NO_VAL:
                resolved[param.name] = param.default
            else:
                missing.append(param.name)

        if missing:
            raise ValueError(
                f"PromptConfig.render: required parameter(s) "
                f"{sorted(missing)!r} missing from inputs."
            )

        return self.template.format(**resolved)

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable representation of this config."""
        return {
            "template": self.template,
            "description": self.description,
            "parameters": [p.to_dict() for p in self.parameters],
        }
