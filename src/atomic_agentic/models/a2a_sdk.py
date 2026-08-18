"""Shared per-skill metadata for the a2a-sdk-backed A2A track.

``A2AtomicSkillMetadata`` is the one type both ``a2a.A2AtomicExecutor``
(serializes instances into its published card extension) and
``a2a.A2AClientHub`` (deserializes them back out) build against, so the two
sides of the wire can never independently drift on shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .parameters import ParamSpec

__all__ = ["A2AtomicSkillMetadata"]


@dataclass(frozen=True, slots=True)
class A2AtomicSkillMetadata:
    """One registered Atomic skill's full remote metadata.

    Published inside ``A2AtomicExecutor``'s ``PARAM_SCHEMA_EXT_URI`` card
    extension and reconstructed by ``A2AClientHub.get_atomic_skills()``.

    Deliberately ``AtomicInvokable``-unaware -- mirrors ``ParamSpec``'s own
    purity in this same module. Building an instance *from* an
    ``AtomicInvokable`` is ``A2AtomicExecutor``'s own private
    ``_build_skill_metadata`` helper, not a method here: ``models/`` sits
    below ``core/`` in the dependency topology
    (``{exceptions, constants} -> models -> utils -> core -> {..., a2a}``),
    so this type can't know about ``AtomicInvokable`` without reopening the
    exact ``TYPE_CHECKING``-only-import layering workaround a27 closed for
    good when it removed ``CheckerSpec``/``GraphFlowNode`` from
    ``models/workflows/``.
    """

    remote_name: str
    description: str | None
    extra_description: str
    params: tuple[ParamSpec, ...]
    return_type: str

    def __post_init__(self) -> None:
        """Validate and normalize dataclass fields after initialization."""
        if not isinstance(self.remote_name, str):
            raise TypeError(
                f"A2AtomicSkillMetadata.remote_name must be a str, got {type(self.remote_name).__name__}"
            )
        cleaned_remote_name = self.remote_name.strip()
        if not cleaned_remote_name:
            raise ValueError("A2AtomicSkillMetadata.remote_name must be a non-empty string")

        if self.description is not None and not isinstance(self.description, str):
            raise TypeError(
                f"A2AtomicSkillMetadata.description must be a str or None, got {type(self.description).__name__}"
            )
        cleaned_description = self.description
        if isinstance(cleaned_description, str):
            cleaned_description = cleaned_description.strip() or None

        if not isinstance(self.extra_description, str):
            raise TypeError(
                f"A2AtomicSkillMetadata.extra_description must be a str, got {type(self.extra_description).__name__}"
            )

        if not isinstance(self.params, tuple) or not all(isinstance(p, ParamSpec) for p in self.params):
            raise TypeError("A2AtomicSkillMetadata.params must be a tuple[ParamSpec, ...]")

        if not isinstance(self.return_type, str) or not self.return_type.strip():
            raise ValueError("A2AtomicSkillMetadata.return_type must be a non-empty string")

        object.__setattr__(self, "remote_name", cleaned_remote_name)
        object.__setattr__(self, "description", cleaned_description)
        object.__setattr__(self, "return_type", self.return_type.strip())

    def to_dict(self) -> dict[str, Any]:
        """Return the explicit serialized dictionary representation."""
        d: dict[str, Any] = {
            "remote_name": self.remote_name,
            "extra_description": self.extra_description,
            "params": [p.to_dict() for p in self.params],
            "return_type": self.return_type,
        }
        if self.description is not None:
            d["description"] = self.description
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "A2AtomicSkillMetadata":
        """Create an A2AtomicSkillMetadata from a serialized mapping."""
        if not isinstance(d, Mapping):
            raise TypeError("A2AtomicSkillMetadata.from_dict expects a mapping")

        remote_name = d.get("remote_name")
        description = d.get("description", None)
        extra_description = d.get("extra_description", "")
        params = d.get("params")
        return_type = d.get("return_type")

        if not isinstance(remote_name, str) or not isinstance(return_type, str) or not isinstance(params, list):
            raise TypeError(
                "A2AtomicSkillMetadata.from_dict expects 'remote_name' (str), "
                "'return_type' (str), and 'params' (list of ParamSpec mappings)"
            )

        # A2A-sdk wire boundary quirk, not a ParamSpec.from_dict contract
        # issue: protobuf's Struct (what an AgentExtension.params round-trips
        # through) has no native int type, so MessageToDict always comes back
        # with whole numbers as float (0 -> 0.0). ParamSpec.index is
        # correctly strict about int for direct/local construction; this
        # coercion belongs here, at the one boundary that actually introduces
        # the ambiguity, not loosened onto ParamSpec.from_dict itself.
        coerced_params = []
        for p in params:
            if not isinstance(p, Mapping):
                raise TypeError("A2AtomicSkillMetadata.from_dict: each params entry must be a mapping")
            p = dict(p)
            if isinstance(p.get("index"), float) and p["index"].is_integer():
                p["index"] = int(p["index"])
            coerced_params.append(p)

        return cls(
            remote_name=remote_name,
            description=description,
            extra_description=extra_description,
            params=tuple(ParamSpec.from_dict(p) for p in coerced_params),
            return_type=return_type,
        )
