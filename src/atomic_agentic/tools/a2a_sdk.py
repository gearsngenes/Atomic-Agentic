from __future__ import annotations

import functools
from typing import Any, ClassVar, Dict, Mapping, Optional

from ..a2a.a2a_client_hub import A2AClientHub
from ..constants.a2a_sdk import GENERIC_METADATA_DESCRIPTION, GENERIC_PARTS_DESCRIPTION
from ..constants.core import NO_VAL
from ..exceptions import ToolInvocationError
from ..models.parameters import ParamSpec
from ..models.results.tools import A2AProxyToolResult
from ..utils.a2a import _dict_to_part, _part_to_dict
from .base import Tool

__all__ = ["A2AProxyTool"]


class A2AProxyTool(Tool):
    """
    Proxy an A2AClientHub as a normal AA Tool.

    Two mutually exclusive modes, selected at construction by ``name``:

    - **Skill mode** (``name`` given): bound to one already-discovered
      Atomic skill; the Tool's typed signature is sourced directly from that
      skill's ``A2AtomicSkillMetadata`` (no re-derivation); ``execute``
      delegates straight to ``client_hub.call_atomic_skill``.
    - **Generic mode** (``name=None``): a fixed two-parameter signature
      (``parts``, ``metadata``) for talking to arbitrary/foreign A2A agents.
      Each ``parts`` item is a plain JSON-safe dict normalized to/from an
      a2a-sdk ``Part`` via ``utils/a2a.py`` -- no a2a-sdk types or heuristic
      type-sniffing ever appear on this Tool's own schema.

    The Tool name is ``<hub.name>_<name>`` or ``<hub.name>_send_parts``, the
    raw name when the hub is unnamed; ``remote_name`` is the raw skill name
    (``None`` in generic mode). Only hub-based construction is supported. Has
    no ``refresh()`` of its own: ``client_hub.refresh(...)`` updates the hub's
    catalog, but an existing proxy keeps the signature it was built with, so
    rebuild proxies (via ``to_tool``/``to_tools``) after a refresh. Skill
    metadata is read from the hub's cached skills
    via ``client_hub.get_skill(name)``, which raises ``A2AProxyError`` for an
    unknown name; the proxy adds no checks of its own.
    """

    result_cls: ClassVar[type[A2AProxyToolResult]] = A2AProxyToolResult

    def __init__(self, client_hub: A2AClientHub, name: Optional[str] = None) -> None:
        # An unknown skill name fails in the hub's get_skill.
        skill_metadata = client_hub.get_skill(name) if name is not None else None
        remote: Optional[str] = name

        # Set before super().__init__() -- _build_tool_signature/_get_mod_qual
        # (called from within it) read these attributes.
        self._client_hub: A2AClientHub = client_hub
        self._remote_name: Optional[str] = remote
        self._skill_metadata = skill_metadata

        base = remote if remote is not None else "send_parts"
        tool_name = f"{client_hub.name}_{base}" if client_hub.name is not None else base

        card_description = str(getattr(client_hub.agent_card, "description", "") or "").strip()
        if remote is not None:
            skill_description = str(skill_metadata.description or "").strip()
            resolved_description = (
                skill_description
                or card_description
                or f"A2A atomic skill proxy tool {tool_name!r}"
            )
            function = functools.partial(client_hub.call_atomic_skill, remote)
        else:
            resolved_description = (
                card_description
                or (
                    f"A2A generic send-parts proxy tool {tool_name!r} via "
                    f"{client_hub.transport_mode}. Sends one or more Parts to the "
                    "remote agent and returns its reply as a list of dicts in the "
                    "same shape as the 'parts' input (see parameter descriptions "
                    "for the exact dict shape)."
                )
            )
            function = client_hub.send_parts

        super().__init__(
            function=function,
            name=tool_name,
            description=resolved_description,
        )

    # ------------------------------------------------------------------ #
    # Proxy properties
    # ------------------------------------------------------------------ #
    @property
    def client_hub(self) -> A2AClientHub:
        return self._client_hub

    @property
    def remote_name(self) -> Optional[str]:
        return self._remote_name

    # ------------------------------------------------------------------ #
    # Signature Building (Template Method)
    # ------------------------------------------------------------------ #
    def _build_tool_signature(self) -> tuple[list[ParamSpec], tuple[str, ...]]:
        if self._remote_name is not None:
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
        if self._remote_name is not None:
            return A2AClientHub.call_atomic_skill.__module__, A2AClientHub.call_atomic_skill.__qualname__
        return A2AClientHub.send_parts.__module__, A2AClientHub.send_parts.__qualname__

    def _extra_description(self) -> str:
        """Skill mode surfaces the remote invokable's own extra description
        (carried in the skill metadata); generic mode has none."""
        if self._remote_name is not None:
            return self._skill_metadata.extra_description
        return ""

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

    def execute(self, args: tuple[Any, ...], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        if args:
            raise ToolInvocationError(
                f"{self.full_name}: A2A tools do not accept positional arguments; got {args!r}."
            )

        if self._remote_name is not None:
            try:
                result = self._function(inputs=kwargs)
            except ToolInvocationError:
                raise
            except Exception as e:
                raise ToolInvocationError(f"{self.full_name}: invocation failed: {e}") from e
        else:
            parts = self._normalize_parts_input(kwargs)
            metadata = kwargs.get("metadata")
            try:
                result_parts = self._function(parts, metadata=metadata)
            except ToolInvocationError:
                raise
            except Exception as e:
                raise ToolInvocationError(f"{self.full_name}: invocation failed: {e}") from e
            result = [_part_to_dict(p) for p in result_parts]

        return self._payload(result)

    def _payload(self, result: Any) -> Dict[str, Any]:
        """The proxy result payload shared by execute and async_execute."""
        return {
            "transport_mode": self.client_hub.transport_mode,
            "location": self.client_hub.location,
            "remote_name": self.remote_name,
            "result": result,
        }

    async def async_execute(
        self,
        args: tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        if args:
            raise ToolInvocationError(
                f"{self.full_name}: A2A tools do not accept positional arguments; got {args!r}."
            )

        if self._remote_name is not None:
            try:
                result = await self.client_hub.async_call_atomic_skill(self._remote_name, kwargs)
            except ToolInvocationError:
                raise
            except Exception as e:
                raise ToolInvocationError(f"{self.full_name}: async invocation failed: {e}") from e
        else:
            parts = self._normalize_parts_input(kwargs)
            metadata = kwargs.get("metadata")
            try:
                result_parts = await self.client_hub.async_send_parts(parts, metadata=metadata)
            except ToolInvocationError:
                raise
            except Exception as e:
                raise ToolInvocationError(f"{self.full_name}: async invocation failed: {e}") from e
            result = [_part_to_dict(p) for p in result_parts]

        return self._payload(result)

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update(
            {
                "remote_name": self.remote_name,
                "client_hub": self.client_hub.to_dict(),
            }
        )
        return data
