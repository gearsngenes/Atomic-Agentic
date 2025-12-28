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
from .base import Tool, ArgumentMap
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
# A2AProxyTool
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

    def __init__(self, url: str, headers: Any | None = None) -> None:
        self._url = url
        self._headers = headers
        self._client = A2AClient(url, headers=headers)

        agent_card = self._client.get_agent_card()
        super().__init__(
            name="invoke",
            description=agent_card.description,
            namespace=agent_card.name,
            function=functools.partial(a2atomic_host_invoker, client=self._client),
        )

    def refresh(self, headers: Any | None = None) -> None:
        """
        Re-fetch the agent card and remote schemas, and rebuild the client and
        function binding. Mirrors the intent of MCPProxyTool.refresh().
        """
        self._headers = headers
        self._client = A2AClient(self._url, headers=self._headers)

        try:
            agent_card = self._client.get_agent_card()
        except Exception as e:  # pragma: no cover
            raise ToolDefinitionError(f"{self.full_name}: failed to fetch A2A agent card: {e}") from e

        # Update exposed identity/metadata
        namespace = agent_card.name
        description = agent_card.description
        # Rebind callable + rebuild schemas
        function = functools.partial(a2atomic_host_invoker, client=self._client)
        super().__init__(
            name="invoke",
            description=description,
            namespace=namespace,
            function=function,
        )

    @property
    def url(self) -> str:
        return self._url

    @url.setter
    def url(self, val: str) -> None:
        self._url = val
        self.refresh(self._headers)

    @property
    def headers(self) -> Any | None:
        return self._headers

    @headers.setter
    def headers(self, val: Any | None) -> None:
        self.refresh(val)

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

        # Coerce the remote arguments_map (which may have been JSON-serialized
        # and thus is a plain dict) into the expected OrderedDict shape and
        # validate parameter metadata.
        raw_args_map = payload["arguments_map"]
        if not isinstance(raw_args_map, Mapping):
            raise ToolDefinitionError(f"{self.full_name}: invokable_metadata.arguments_map must be a mapping")

        args_map: OrderedDict[str, dict] = OrderedDict()
        for key, meta in raw_args_map.items():
            if not isinstance(key, str):
                raise ToolDefinitionError(f"{self.full_name}: invokable_metadata.arguments_map keys must be strings")
            if not isinstance(meta, Mapping):
                raise ToolDefinitionError(f"{self.full_name}: metadata for argument '{key}' must be a mapping")
            meta_dict = dict(meta)
            # Minimal validation of expected fields
            if "index" not in meta_dict or "kind" not in meta_dict or "type" not in meta_dict:
                raise ToolDefinitionError(
                    f"{self.full_name}: metadata for argument '{key}' missing required keys (index, kind, type)"
                )
            args_map[key] = meta_dict

        return_type = payload["return_type"]
        if not isinstance(return_type, str):
            raise ToolDefinitionError(f"{self.full_name}: invokable_metadata.return_type must be a str")
        return OrderedDict(args_map), return_type

    def _get_mod_qual(
        self,
        function: Callable[..., Any],
    ) -> tuple[Optional[str], Optional[str]]:
        """A2A-backed tools don't map to a stable Python import path."""
        return a2atomic_host_invoker.__module__, a2atomic_host_invoker.__qualname__

    def _compute_is_persistible(self) -> bool:
        try:
            agent_card = self._client.get_agent_card()
        except Exception:
            return False
        return bool(agent_card.name and agent_card.description and self._url)

    def to_arg_kwarg(self, inputs: Mapping[str, Any]) -> tuple[tuple[Any, ...], Dict[str, Any]]:
        return tuple([]), dict(inputs)

    def execute(self, args: tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        try:
            result = self._function(inputs = kwargs)
        except Exception as e:  # pragma: no cover
            raise ToolInvocationError(f"{self.full_name}: invocation failed: {e}") from e
        return result

    def to_dict(self) -> OrderedDict[str, Any]:
        dict_data = super().to_dict()
        dict_data.update(OrderedDict(url=self._url))
        return dict_data
