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
    A2AClient, A2AServer, run_server, agent,
    Message, MessageRole,
    FunctionCallContent, FunctionResponseContent, FunctionParameter,
    TextContent
)

from ..core.Exceptions import ToolDefinitionError, ToolInvocationError
from .base import Tool, ArgumentMap
# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
A2A_RESULT_KEY = "__py_A2A_result__"  # server uses this key inside FunctionResponseContent.response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _ensure_mapping(name: str, obj: Any) -> Mapping[str, Any]:
    if not isinstance(obj, Mapping):
        raise TypeError(f"{name} expects a Mapping[str, Any]")
    return obj  # type: ignore[return-value]

def a2agent_host_invoker(client, inputs: Mapping[str, Any]) -> Any:  # type: ignore[override]
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
        result = resp.get(A2A_RESULT_KEY, resp)
    return result

# ───────────────────────────────────────────────────────────────────────────────
# A2AgentTool
# ───────────────────────────────────────────────────────────────────────────────
class A2AgentTool(Tool):
    """
    A2AgentTool
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
            function=functools.partial(a2agent_host_invoker, client=self._client),
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
        function = functools.partial(a2agent_host_invoker, client=self._client)
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

    def _build_io_schemas(self) -> tuple[ArgumentMap, str]:
        """Construct ``arguments_map`` and ``return_type`` from remote "agent_metadata"."""
        call = FunctionCallContent(name="agent_metadata", parameters=[])
        msg = Message(content=call, role=MessageRole.USER)

        resp = self._client.send_message(msg)
        if getattr(resp.content, "type", None) != "function_response":
            raise ToolDefinitionError(f"{self.full_name}: failed to fetch agent_metadata from A2A agent")

        payload = resp.content.response
        if not isinstance(payload, Mapping):
            raise ToolDefinitionError(f"{self.full_name}: agent_metadata response must be a mapping")

        if "arguments_map" not in payload or "return_type" not in payload:
            raise ToolDefinitionError(f"{self.full_name}: agent_metadata response missing required keys")

        args_map = payload["arguments_map"]
        if not isinstance(args_map, Mapping):
            raise ToolDefinitionError(f"{self.full_name}: agent_metadata.arguments_map must be a mapping")
        return_type = str(payload["return_type"])
        return args_map, return_type

    def _get_mod_qual(
        self,
        function: Callable[..., Any],
    ) -> tuple[Optional[str], Optional[str]]:
        """A2A-backed tools don't map to a stable Python import path."""
        return None, None

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
