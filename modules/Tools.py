import inspect
from typing import Any, get_type_hints
import asyncio
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from modules.Agents import Agent
from modules.Plugins import *

class Tool:
    def __init__(self, name, func, type = "function", source = "default", description=""):
        self._type = type
        self._source = source
        self._name = name
        self._func = func
        self._description = description
        self.signature = Tool._build_signature(self.name, func)
    @property
    def type(self):
        return self._type
    @property
    def source(self):
        return self._source
    @property
    def name(self):
        return f"{self._type}.{self._source}.{self._name}"
    @property
    def description(self):
        return self._description
    @property
    def func(self):
        return self._func
    @staticmethod
    def _build_signature(key: str, func: callable) -> str:
        sig = inspect.signature(func)
        hints = get_type_hints(func)
        params = [
            f"{n}: {hints.get(n, Any).__name__}"
            + (f" = {p.default!r}" if p.default is not inspect._empty else "")
            for n, p in sig.parameters.items() if n != "self"
        ]
        rtype = hints.get('return', Any).__name__
        return f"{key}({', '.join(params)}) → {rtype}"

class ToolFactory:
    @staticmethod
    def toolify_function(func: callable, type= "function", source = "default", description: str = "") -> list[Tool]:
        if func.__name__ == "<lambda>":
            raise ValueError("Lambda functions must be given proper names.")
        return [Tool(name=func.__name__, func=func, type=type, source=source, description = description)]
    @staticmethod
    def toolify_agent(agent: Agent) -> list[Tool]:
        invoke_tool = ToolFactory.toolify_function(
            func=agent.invoke,
            type="agent",
            source=agent.name,
            description=f"This method invokes the {agent.name} agent. Agent description: {agent.description}"
        )
        attach_tool = ToolFactory.toolify_function(
            func=agent.attach,
            type="agent",
            source=agent.name,
            description=f"Attaches the specified input file path to {agent.name}'s internal knowledge base. Do NOT use unless there is an explicitly mentioned file path needed to be attached"
        )
        detach_tool = ToolFactory.toolify_function(
            func=agent.detach,
            type="agent",
            source=agent.name,
            description=f"Detaches the specified input file path from {agent.name}'s internal knowledge base. Do NOT use unless there is an explicitly mentioned file path needed to be detached"
        )
        clear_tool = ToolFactory.toolify_function(
            func=agent.clear_memory,
            type="agent",
            source=agent.name,
            description=f"Clears {agent.name}'s internal conversation history."
        )
        return invoke_tool + attach_tool + detach_tool + clear_tool
    @staticmethod
    def toolify_plugin(plugin: Plugin) -> list[Tool]:
        tools = []
        source = plugin.get("name", "unknown")
        tool_map = plugin.get("method_map", {})
        for method_name, method_info in tool_map.items():
            func = method_info.get("callable")
            description = method_info.get("description", "")
            if func.__name__ == "<lambda>":
                func.__name__ = method_name
            if callable(func):
                tools.append(Tool(
                    name=method_name,
                    func=func,
                    type="plugin",
                    source=source,
                    description=description
                ))
        return tools
    @staticmethod
    def toolify_mcp_server(name: str, url_or_base: str) -> list[Tool]:
        """
        Convert a native MCP server into Tool wrappers.
        - name: REQUIRED logical server name (bucket key).
        - url_or_base: '.../mcp' or base URL (we'll append '/mcp' if missing).
        """
        if not name or not isinstance(name, str):
            raise ValueError("toolify_mcp_server requires a non-empty server name.")
        base = url_or_base.rstrip("/")
        mcp_url = base if base.endswith("/mcp") else f"{base}/mcp"

        async def _list_tools(u: str):
            async with streamablehttp_client(u) as (r, w, _sid):
                async with ClientSession(r, w) as session:
                    await session.initialize()
                    resp = await session.list_tools()
                    out = []
                    for t in resp.tools:
                        out.append(
                            {
                                "name": t.name,
                                "description": getattr(t, "description", "") or t.name,
                                "schema": getattr(t, "input_schema", None)
                                or getattr(t, "inputSchema", None)
                                or {},
                            }
                        )
                    return out

        specs = asyncio.run(_list_tools(mcp_url))

        def _schema_props(schema: dict) -> tuple[list[str], list[str]]:
            props, req = [], []
            if isinstance(schema, dict):
                props = list((schema.get("properties") or {}).keys())
                req = schema.get("required") or []
            return props, req

        def _make_wrapper(u: str, tool_name: str):
            async def _acall(**payload):
                # unwrap generative "kwargs" style plans
                if "kwargs" in payload and isinstance(payload["kwargs"], dict) and len(payload) == 1:
                    payload = payload["kwargs"]
                async with streamablehttp_client(u) as (r, w, _sid):
                    async with ClientSession(r, w) as session:
                        await session.initialize()
                        result = await session.call_tool(tool_name, arguments=payload)
                        if getattr(result, "content", None):
                            texts = [getattr(c, "text", None) for c in result.content if getattr(c, "text", None)]
                            if texts:
                                return texts[0]
                        try:
                            return result.model_dump()
                        except Exception:
                            return result

            def _sync(**payload):
                return asyncio.run(_acall(**payload))

            _sync.__name__ = f"mcp_{tool_name}"
            return _sync

        tools: list[Tool] = []
        for spec in specs:
            tool_name = spec["name"]
            desc = spec["description"]
            props, req = _schema_props(spec["schema"])
            fn = _make_wrapper(mcp_url, tool_name)

            # Build an arg summary in the description since the wrapper uses **payload
            if props:
                arg_lines = [f"- {p}" + (" (required)" if p in req else " (optional)") for p in props]
                arg_block = "\nArgs:\n" + "\n".join(arg_lines)
            else:
                arg_block = ""

            tools.append(
                Tool(
                    name=f"mcp_{tool_name}",
                    func=fn,
                    type="mcp",
                    source=name,
                    description=f"Calls MCP tool '{tool_name}' at {mcp_url}. {desc}{arg_block}",
                )
            )
        return tools
    @staticmethod
    def toolify(object: Any, name: str|None = None, description: str = "") -> list[Tool]:
        if inspect.isfunction(object):
            return ToolFactory.toolify_function(func=object, description=description)
        elif isinstance(object, Agent):
            return ToolFactory.toolify_agent(agent=object)
        elif isinstance(object, dict) and "method_map" in object and "name" in object:
            return ToolFactory.toolify_plugin(plugin=object)
        elif isinstance(object, str) and object.endswith("/mcp"):
            return ToolFactory.toolify_mcp_server(name = name, url_or_base = object)
        else:
            raise ValueError("Unsupported object type for toolification.")