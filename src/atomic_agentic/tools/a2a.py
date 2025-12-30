# Tools.py
from __future__ import annotations
from collections import OrderedDict
import functools
from typing import (
    Any,
    Mapping,
    Callable,
    Optional,
    Dict,
)
# python-a2a imports
from python_a2a import (
    A2AClient,
    Message, MessageRole,
    FunctionCallContent, FunctionParameter,
)

from ..core.Exceptions import ToolDefinitionError, ToolInvocationError
from ..core.Invokable import ArgumentMap, NO_VAL, ArgSpec
from .base import Tool
from ..a2a.A2AtomicHost import A2A_RESULT_KEY

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _ensure_mapping(name: str, obj: Any) -> Mapping[str, Any]:
    if not isinstance(obj, Mapping):
        raise TypeError(f"{name} expects a Mapping[str, Any]")
    return obj  # type: ignore[return-value]

def a2atomic_host_invoker(client, inputs: Mapping[str, Any]) -> Any:  # type: ignore[override]
    _ensure_mapping("A2AProxyAgent.invoke", inputs)

    # Typed FunctionParameter avoids `'dict' object has no attribute 'name'`.
    call = FunctionCallContent(
        name="invoke",
        parameters=[FunctionParameter(name="payload", value=dict(inputs))]
    )
    msg = Message(content=call, role=MessageRole.USER)
    resp = client.send_message(msg).content.response
    result = resp
    if isinstance(resp, Mapping):
        result = resp.get(A2A_RESULT_KEY, result)
    return result

# ───────────────────────────────────────────────────────────────────────────────
# A2A-Proxy Tool
# ───────────────────────────────────────────────────────────────────────────────
class A2AProxyTool(Tool):
    """
    A2AProxyTool
    -----------
    Client-side proxy Tool that forwards a single JSON object (a mapping of
    inputs) to a remote A2A agent via a single function call:

      - name="invoke", parameters=[("payload", <dict(inputs)>)]

    This class intentionally overrides :meth:`to_arg_kwarg` and :meth:`execute`
    to support the transport semantics of A2A-backed tools, while keeping
    :meth:`Tool.invoke` as the single public entrypoint.

    Metadata / schema
    -----------------
    On construction and on :meth:`refresh`, the tool calls the remote function
    "agent_metadata" to populate:

      - arguments_map (remote agent's declared input schema)
      - return_type   (remote agent's declared output type)

    Changing :attr:`url` or :attr:`headers` triggers a full :meth:`refresh`,
    rebinding the underlying client and rebuilding schemas.
    """
    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def __init__(self,
                 url: str,
                 namespace: str | None = None,
                 headers: Any | None = None) -> None:
        self._url = url
        self._client = A2AClient(url, headers=headers)

        agent_card = self._client.get_agent_card()
        super().__init__(
            name=agent_card.name,
            description=agent_card.description,
            namespace=namespace,
            function=functools.partial(a2atomic_host_invoker, client=self._client),
        )

    # ------------------------------------------------------------------ #
    # A2A-Proxy-Tool Properties
    # ------------------------------------------------------------------ #
    @property
    def url(self) -> str:
        return self._url

    # ------------------------------------------------------------------ #
    # Atomic-Invokable Helpers
    # ------------------------------------------------------------------ #
    def build_args_returns(self) -> tuple[ArgumentMap, str]:
        """Construct ``arguments_map`` and ``return_type`` from remote "agent_metadata"."""
        call = FunctionCallContent(name="invokable_metadata", parameters=[])
        msg = Message(content=call, role=MessageRole.USER)

        resp = self._client.send_message(msg)
        if getattr(resp.content, "type", None) != "function_response":
            raise ToolDefinitionError(f"{self.full_name}: failed to fetch invokable_metadata from A2A agent")

        payload = resp.content.response
        if not isinstance(payload, Mapping):
            raise ToolDefinitionError(f"{self.full_name}: invokable_metadata response must be a mapping")

        if "arguments_map" not in payload or "return_type" not in payload:
            raise ToolDefinitionError(f"{self.full_name}: invokable_metadata response missing required keys")

        # Extract components
        raw_args_map = payload["arguments_map"]
        return_type = payload["return_type"]
        # Validate types
        if not isinstance(raw_args_map, Mapping):
            raise ToolDefinitionError(f"{self.full_name}: invokable_metadata.arguments_map must be a mapping")
        if not isinstance(return_type, str):
            raise ToolDefinitionError(f"{self.full_name}: invokable_metadata.return_type must be a str")
        args_map: ArgumentMap = {}
        for key, meta in raw_args_map.items():
            if not isinstance(key, str):
                raise ToolDefinitionError(f"{self.full_name}: invokable_metadata.arguments_map keys must be strings")
            if not isinstance(meta, Mapping):
                raise ToolDefinitionError(f"{self.full_name}: metadata for argument '{key}' must be a mapping")
            meta_dict = dict(meta)
            args_map[key] = ArgSpec.from_dict(meta_dict)

        return args_map, return_type

    def _compute_is_persistible(self) -> bool:
        try:
            agent_card = self._client.get_agent_card()
        except Exception:
            return False
        return agent_card.name == self.name

    # ------------------------------------------------------------------ #
    # Tool Helpers
    # ------------------------------------------------------------------ #
    def _get_mod_qual(
        self,
        function: Callable[..., Any],
    ) -> tuple[Optional[str], Optional[str]]:
        """A2A-backed tools don't map to a stable Python import path."""
        return a2atomic_host_invoker.__module__, a2atomic_host_invoker.__qualname__

    def to_arg_kwarg(self, inputs: Mapping[str, Any]) -> tuple[tuple[Any, ...], Dict[str, Any]]:
        return tuple([]), dict(inputs)

    def execute(self, args: tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        result = self._function(inputs = kwargs)
        return result

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def refresh(self, headers: Any | None = None) -> None:
        """
        Re-fetch the agent card and remote schemas, and rebuild the client and
        function binding. Mirrors the intent of MCPProxyTool.refresh().
        """
        self._client = A2AClient(self._url, headers=headers)

        try:
            agent_card = self._client.get_agent_card()
        except Exception as e:  # pragma: no cover
            raise ToolDefinitionError(f"{self.full_name}: failed to fetch A2A agent card: {e}") from e

        # Update exposed identity/metadata
        name = agent_card.name
        description = agent_card.description
        # Rebind callable + rebuild schemas
        function = functools.partial(a2atomic_host_invoker, client=self._client)
        super().__init__(
            name=name,
            description=description,
            namespace=self.namespace,
            function=function,
        )

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        dict_data = super().to_dict()
        dict_data.update({"url": self._url})
        return dict_data
