from __future__ import annotations

import functools
import re
from typing import Any, Dict, Mapping, Optional

from ..a2a.A2AClientHub import A2AClientHub
from ..constants.a2a_sdk import GENERIC_METADATA_DESCRIPTION, GENERIC_PARTS_DESCRIPTION
from ..constants.core import IDENTIFIER_PATTERN, NO_VAL
from ..exceptions import ToolDefinitionError, ToolInvocationError
from ..models.parameters import ParamSpec
from ..models.results.tools import A2AProxyToolResult
from ..utils.a2a import _dict_to_part, _part_to_dict
from .base import Tool

__all__ = ["A2AProxyTool"]

# Collapses runs of '-', '.', or whitespace into one '_' before the
# IDENTIFIER_PATTERN check -- "My  Agent" / "my--agent" / "my.agent" all
# normalize to one clean candidate, not one underscore per separator char.
_NAMESPACE_SEPARATOR_PATTERN = re.compile(r"[-.\s]+")


class A2AProxyTool(Tool):
    """
    Proxy an A2AClientHub as a normal AA Tool.

    Two mutually exclusive modes, selected at construction by ``skill_id``:

    - **Skill mode** (``skill_id`` given): bound to one already-discovered
      Atomic skill; the Tool's typed signature is sourced directly from that
      skill's ``A2AtomicSkillMetadata`` (no re-derivation); ``execute``
      delegates straight to ``client_hub.call_atomic_skill``.
    - **Generic mode** (``skill_id=None``): a fixed two-parameter signature
      (``parts``, ``metadata``) for talking to arbitrary/foreign A2A agents.
      Each ``parts`` item is a plain JSON-safe dict normalized to/from an
      a2a-sdk ``Part`` via ``utils/a2a.py`` -- no a2a-sdk types or heuristic
      type-sniffing ever appear on this Tool's own schema.

    Only ``client_hub``-based construction is supported (no raw-transport-
    param alternative) -- deliberately lighter than ``MCPProxyTool``'s dual-
    path constructor. Has no ``refresh()``/mutation lifecycle of its own --
    same policy as ``PyA2AtomicTool``; call ``self.client_hub.refresh(...)``
    directly to pick up remote-side changes.
    """

    def __init__(
        self,
        client_hub: A2AClientHub,
        skill_id: Optional[str] = None,
        name: Optional[str] = None,
        namespace: Optional[str] = None,
        description: Optional[str] = None,
    ) -> None:
        if not isinstance(client_hub, A2AClientHub):
            raise TypeError(f"client_hub must be an A2AClientHub, got {type(client_hub)!r}.")

        resolved_skill_id: Optional[str] = None
        skill_metadata = None
        if skill_id is not None:
            resolved_skill_id = str(skill_id).strip()
            if not resolved_skill_id:
                raise ValueError("skill_id must be a non-empty string when provided.")
            atomic_skills = client_hub.get_atomic_skills()
            if resolved_skill_id not in atomic_skills:
                raise ToolDefinitionError(
                    f"A2AProxyTool: skill_id {resolved_skill_id!r} not found; "
                    f"available skills: {sorted(atomic_skills)!r}."
                )
            skill_metadata = atomic_skills[resolved_skill_id]

        # Set before super().__init__() -- _build_tool_signature/_get_mod_qual
        # (called from within it) read these attributes.
        self._client_hub: A2AClientHub = client_hub
        self._skill_id: Optional[str] = resolved_skill_id
        self._skill_metadata = skill_metadata

        is_skill_mode = self._skill_id is not None

        resolved_name = str(name or "").strip()
        if not resolved_name:
            resolved_name = self._skill_id if is_skill_mode else "send_parts"

        explicit_namespace = str(namespace or "").strip()
        if explicit_namespace:
            resolved_namespace = explicit_namespace
        else:
            card_name = str(getattr(client_hub.agent_card, "name", "") or "")
            candidate = self._sanitize_namespace_candidate(card_name)
            resolved_namespace = candidate if candidate is not None else "a2a"

        explicit_description = str(description or "").strip()
        card_description = str(getattr(client_hub.agent_card, "description", "") or "").strip()
        if is_skill_mode:
            skill_description = str(skill_metadata.description or "").strip()
            resolved_description = (
                explicit_description
                or skill_description
                or card_description
                or f"A2A atomic skill proxy tool {resolved_name!r}"
            )
            function = functools.partial(client_hub.call_atomic_skill, self._skill_id)
        else:
            resolved_description = (
                explicit_description
                or card_description
                or (
                    f"A2A generic send-parts proxy tool {resolved_name!r} via "
                    f"{client_hub.transport_mode}. Sends one or more Parts to the "
                    "remote agent and returns its reply as a list of dicts in the "
                    "same shape as the 'parts' input (see parameter descriptions "
                    "for the exact dict shape)."
                )
            )
            function = client_hub.send_parts

        super().__init__(
            function=function,
            name=resolved_name,
            namespace=resolved_namespace,
            description=resolved_description,
        )

    @staticmethod
    def _sanitize_namespace_candidate(card_name: str) -> Optional[str]:
        """
        Turn a remote AgentCard's ``name`` into a usable namespace, or
        ``None`` if it doesn't sanitize into one. Case is preserved verbatim
        -- no AA convention forces lowercase namespaces (``tools/prebuilt.py``
        already ships ``"Math"``/``"Console"``). A ``None`` return means the
        caller falls back to ``"a2a"``, never to an empty-string namespace.
        """
        candidate = _NAMESPACE_SEPARATOR_PATTERN.sub("_", card_name.strip())
        return candidate if IDENTIFIER_PATTERN.fullmatch(candidate) else None

    # ------------------------------------------------------------------ #
    # Proxy properties
    # ------------------------------------------------------------------ #
    @property
    def client_hub(self) -> A2AClientHub:
        return self._client_hub

    @property
    def skill_id(self) -> Optional[str]:
        return self._skill_id

    @property
    def transport_mode(self) -> str:
        return self.client_hub.transport_mode

    @property
    def base_url(self) -> str:
        return self.client_hub.base_url

    @property
    def persistent(self) -> bool:
        return self.client_hub.persistent

    # ------------------------------------------------------------------ #
    # Signature Building (Template Method)
    # ------------------------------------------------------------------ #
    def _build_tool_signature(self) -> tuple[list[ParamSpec], str]:
        if self._skill_id is not None:
            return list(self._skill_metadata.params), self._skill_metadata.return_type

        return (
            [
                ParamSpec(
                    name="parts",
                    index=0,
                    kind=ParamSpec.POSITIONAL_OR_KEYWORD,
                    type=("list",),
                    default=NO_VAL,
                    description=GENERIC_PARTS_DESCRIPTION,
                ),
                ParamSpec(
                    name="metadata",
                    index=1,
                    kind=ParamSpec.KEYWORD_ONLY,
                    type=("dict", "NoneType"),
                    default=None,
                    description=GENERIC_METADATA_DESCRIPTION,
                ),
            ],
            "list",
        )

    # ------------------------------------------------------------------ #
    # Tool helpers
    # ------------------------------------------------------------------ #
    def _get_mod_qual(self, function: Any) -> tuple[Optional[str], Optional[str]]:
        if self._skill_id is not None:
            return A2AClientHub.call_atomic_skill.__module__, A2AClientHub.call_atomic_skill.__qualname__
        return A2AClientHub.send_parts.__module__, A2AClientHub.send_parts.__qualname__

    def to_arg_kwarg(
        self,
        inputs: Mapping[str, Any],
    ) -> tuple[tuple[Any, ...], Dict[str, Any]]:
        # A2A calls are always dict-first.
        return tuple(), dict(inputs)

    def _normalize_parts_input(self, kwargs: Dict[str, Any]) -> list:
        parts_input = kwargs.get("parts")
        if not isinstance(parts_input, (list, tuple)) or not parts_input:
            raise ToolInvocationError(
                f"{self.full_name}: 'parts' is required and must be a non-empty "
                f"list of dicts; got {parts_input!r}."
            )
        try:
            return [_dict_to_part(item) for item in parts_input]
        except (ValueError, TypeError) as e:
            raise ToolInvocationError(f"{self.full_name}: malformed parts item: {e}") from e

    def execute(self, args: tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        if args:
            raise ToolInvocationError(
                f"{self.full_name}: A2A tools do not accept positional arguments; got {args!r}."
            )

        if self._skill_id is not None:
            try:
                return self._function(inputs=kwargs)
            except ToolInvocationError:
                raise
            except Exception as e:
                raise ToolInvocationError(f"{self.full_name}: invocation failed: {e}") from e

        parts = self._normalize_parts_input(kwargs)
        metadata = kwargs.get("metadata")
        try:
            result_parts = self._function(parts, metadata=metadata)
        except ToolInvocationError:
            raise
        except Exception as e:
            raise ToolInvocationError(f"{self.full_name}: invocation failed: {e}") from e

        return [_part_to_dict(p) for p in result_parts]

    async def async_execute(
        self,
        args: tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Any:
        if args:
            raise ToolInvocationError(
                f"{self.full_name}: A2A tools do not accept positional arguments; got {args!r}."
            )

        if self._skill_id is not None:
            try:
                return await self.client_hub.async_call_atomic_skill(self._skill_id, kwargs)
            except ToolInvocationError:
                raise
            except Exception as e:
                raise ToolInvocationError(f"{self.full_name}: async invocation failed: {e}") from e

        parts = self._normalize_parts_input(kwargs)
        metadata = kwargs.get("metadata")
        try:
            result_parts = await self.client_hub.async_send_parts(parts, metadata=metadata)
        except ToolInvocationError:
            raise
        except Exception as e:
            raise ToolInvocationError(f"{self.full_name}: async invocation failed: {e}") from e

        return [_part_to_dict(p) for p in result_parts]

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def make_result(
        self,
        result: Any,
        started_at,
        ended_at,
        **result_kwargs: Any,
    ) -> A2AProxyToolResult:
        """Construct an A2AProxyToolResult carrying this proxy's transport identity."""
        return self._make_result(
            result=result,
            started_at=started_at,
            ended_at=ended_at,
            result_cls=A2AProxyToolResult,
            transport_mode=self.transport_mode,
            base_url=self.base_url,
            persistent=self.persistent,
            skill_id=self.skill_id,
            **result_kwargs,
        )

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update(
            {
                "skill_id": self.skill_id,
                "client_hub": self.client_hub.to_dict(),
            }
        )
        return data
