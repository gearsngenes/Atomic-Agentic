import re, inspect, requests
from dotenv import load_dotenv
from typing import Any, get_type_hints
import asyncio
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
import logging

logging.basicConfig(level=logging.WARNING)  # Default level; override in examples as needed

load_dotenv()

# internal imports
from modules.Plugins import *
import modules.Prompts as Prompts
from modules.LLMEngines import *

# ────────────────────────────────────────────────────────────────
# 1.  Agent  (LLM responds to prompts)
# ────────────────────────────────────────────────────────────────
class Agent:
    """
    Minimal, stateful Agent that:
      • Keeps optional chat history (when context_enabled=True)
      • Tracks a simple list of attached file paths (string paths only)
      • Delegates all provider-specific handling to the LLM engine

    Flow:
      - attach(path): add a file path to this agent's list (no provider calls here)
      - detach(path): remove a file path from this agent's list
      - invoke(prompt): build messages (system + history + user),
                        and call engine.invoke(messages, file_paths=[...])
    """

    def __init__(self, name, description, llm_engine,
                 role_prompt: str = "", context_enabled: bool = False):
        self._name = name
        self._description = description
        self._llm_engine = llm_engine
        self._role_prompt = role_prompt or ""
        self._context_enabled = bool(context_enabled)

        # Conversation turns (list of {"role": str, "content": str})
        self._history: list[dict] = []

        # Simple list of file paths the Agent has attached
        self._file_paths: list[str] = []

    # ── Properties ──────────────────────────────────────────────
    @property
    def name(self):
        return self._name

    @property
    def description(self):
        return self._description

    @property
    def role_prompt(self):
        return self._role_prompt

    @property
    def context_enabled(self):
        return self._context_enabled

    @property
    def llm_engine(self):
        return self._llm_engine

    @property
    def history(self):
        # return a copy for safety
        return self._history.copy()

    @property
    def file_paths(self):
        # return a copy for safety
        return self._file_paths.copy()

    @name.setter
    def name(self, value: str):
        self._name = value

    @description.setter
    def description(self, value: str):
        self._description = value

    @role_prompt.setter
    def role_prompt(self, value: str):
        self._role_prompt = value or ""

    @context_enabled.setter
    def context_enabled(self, value: bool):
        self._context_enabled = bool(value)

    @llm_engine.setter
    def llm_engine(self, engine):
        self._llm_engine = engine

    # ── Memory controls ─────────────────────────────────────────
    def clear_memory(self):
        """Clear conversation history; file attachments are left as-is."""
        self._history = []

    # ── File path management (no provider calls here) ───────────
    def attach(self, path: str) -> bool:
        """
        Add a local file path to this Agent's attachments list.
        Returns True if added; False if it was already present.
        """
        if path in self._file_paths:
            return False
        self._file_paths.append(path)
        return True

    def detach(self, path: str) -> bool:
        """
        Remove a local file path from this Agent's attachments list.
        Returns True if removed; False if it wasn't present.
        """
        try:
            self._file_paths.remove(path)
            return True
        except ValueError:
            return False

    # ── Inference ───────────────────────────────────────────────
    def invoke(self, prompt: str) -> str:
        """
        Build the message list and delegate to the engine.
        Pass both:
          - messages (system + optional history + current user)
          - file_paths (simple list of strings)
        Engines remain stateless and decide how to handle/clean up uploads.
        """
        messages = []
        if self._role_prompt:
            messages.append({"role": "system", "content": self._role_prompt})
        if self._context_enabled and self._history:
            messages.extend(self._history)
        messages.append({"role": "user", "content": prompt})

        # Delegate to engine: it inspects file_paths and does the right thing per provider
        response = self._llm_engine.invoke(messages, file_paths=self._file_paths.copy()).strip()

        # Persist turns only when context is enabled
        if self._context_enabled:
            self._history.append({"role": "user", "content": prompt})
            self._history.append({"role": "assistant", "content": response})

        return response

# ────────────────────────────────────────────────────────────────
# 2.  Human Agent  (Asks human for input, when provided a prompt)
# ────────────────────────────────────────────────────────────────
class HumanAgent(Agent):
    def __init__(self, name, description, context_enabled:bool = False):
        self._context_enabled = context_enabled
        self._name = name
        self._description = description
        self._llm_engine = None
        self._role_prompt = description
    def attach(self, file_path: str):
        raise NotImplementedError("HumanAgent does not support file attachments.")
    def detach(self, file_path: str):
        raise NotImplementedError("HumanAgent does not support file attachments.")
    def invoke(self, prompt:str):
        response = input(f"{prompt}\n{self.name}'s Response: ")
        if self._context_enabled:
            self._history.append({"role": "user", "content": prompt})
            self._history.append({"role": "assistant", "content": response})
        return response

from abc import ABC, abstractmethod
# ────────────────────────────────────────────────────────────────
# 3.  Abstract ToolAgent  (Uses Tools and Agents to execute tasks)
# ────────────────────────────────────────────────────────────────
class ToolAgent(Agent, ABC):
    def __init__(self, name, description, llm_engine, role_prompt = Prompts.DEFAULT_PROMPT, allow_agentic:bool = False, allow_mcp:bool = False):
        super().__init__(name, description, llm_engine, role_prompt, context_enabled = False)
        self._toolbox:dict[str, dict] = {}
        self._previous_steps: list[dict] = []
        # These flags let subclasses declare what they can register
        self._allow_agent_registration = allow_agentic
        self._allow_mcp_registration = allow_mcp
        self._mcpo_servers = {}  # optional for MCP, added only if enabled
        self._mcpo_counter = 0
        self._mcp_counter = 0
        def _return(val: Any): return val
        self.register(_return, "Returns the passed-in value. Always use this at the end of a plan.")

    def _resolve(self, obj: Any) -> Any:
        """
        Recursively resolve {{stepN}} references using self._previous_steps.
        Ensures the referenced step is completed before use.
        """
        if isinstance(obj, str):
            match = re.fullmatch(r"\{\{step(\d+)\}\}", obj)
            if match:
                idx = int(match.group(1))
                if idx >= len(self._previous_steps) or not self._previous_steps[idx]["completed"]:
                    raise RuntimeError(f"Step {idx} has not been completed yet.")
                return self._previous_steps[idx]["result"]

            return re.sub(
                r"\{\{step(\d+)\}\}",
                lambda m: str(self._previous_steps[int(m.group(1))]["result"])
                if self._previous_steps[int(m.group(1))]["completed"]
                else RuntimeError(f"Step {m.group(1)} has not been completed yet."),
                obj
            )

        if isinstance(obj, list):
            return [self._resolve(x) for x in obj]
        if isinstance(obj, dict):
            return {k: self._resolve(v) for k, v in obj.items()}
        return obj
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
    @abstractmethod
    def strategize(self, prompt:str)->dict:
        pass
    @abstractmethod
    def execute(self, plan:dict)->Any:
        pass
    def _register_callable(self, func: callable, description: str) -> str:
        """
        Register a plain Python function as a dev tool.

        - Source group: "__dev_tools__"
        - Key format:   "__dev_tools__.<func_name>"
        - Description:  "<signature> — <description>"

        Returns:
            str: The fully-qualified toolbox key.
        """
        if not callable(func):
            raise TypeError("func must be callable")

        if not description:
            raise ValueError("Tool functions must include a description.")

        name = getattr(func, "__name__", None) or ""
        if not name or name.startswith("<"):
            raise ValueError("Callable tools must have a valid (non-anonymous) __name__.")

        source = "__dev_tools__"
        key = f"{source}.{name}"
        sig = self._build_signature(key, func)

        if source not in self._toolbox:
            self._toolbox[source] = {}

        # Optional: guard against accidental overwrite; comment out if you prefer overwriting
        if key in self._toolbox[source]:
            raise RuntimeError(f"Tool '{key}' is already registered.")

        self._toolbox[source][key] = {
            "callable": func,
            "description": f"{sig} — {description}",
        }
        return key

    def _register_plugin(self,plugin: Plugin) -> str:
        """
        Registers a Plugin instance into the toolbox as its own SOURCE bucket.

        Returns:
            source (str): the SOURCE key that was created (e.g. "__plugin_MathPlugin__")
        """
        # 1) compute SOURCE name & guard against duplicates
        plugin_name = plugin.__class__.__name__
        source = f"__plugin_{plugin_name}__"
        if source in self._toolbox:
            raise RuntimeError(f"Plugin '{plugin_name}' already registered.")

        # 2) get methods and filter by include/exclude if provided
        methods = plugin.method_map()  # { method_name: {"callable": ..., "description": ...}, ... }

        if not methods:
            raise ValueError(f"No methods to register for plugin '{plugin_name}' after filtering.")

        # 3) build the per-method entries (fully qualified function keys)
        bucket: dict[str, dict] = {}
        for method_name, meta in methods.items():
            fq_name = f"{source}.{method_name}"
            fn = meta["callable"]
            desc = meta.get("description", "").strip()
            sig = self._build_signature(fq_name, fn)
            bucket[fq_name] = {
                "callable": fn,
                "description": f"{sig} — {desc}"
            }

        # 4) add to toolbox and return the source for convenience
        self._toolbox[source] = bucket
        return source

    def _register_agent(self, agent: Agent) -> str:
        """
        Register an Agent instance as a tool source.

        - Requires: self._allow_agent_registration == True
        - Source group: "__agent_<AgentName>__"
        - Exposed method: "<source>.invoke" (calls agent.invoke(prompt: str) -> Any)

        Returns:
            str: Fully-qualified toolbox key for the invoke method.
        """
        if not self._allow_agent_registration:
            raise RuntimeError("Agent registration is not enabled for this agent.")

        source = f"__agent_{agent.name}__"
        if source in self._toolbox:
            raise RuntimeError(f"Agent '{agent.name}' is already registered.")

        key = f"{source}.invoke"
        sig = self._build_signature(key, agent.invoke)
        desc = f"{sig} — Agent description: {agent.description}"

        self._toolbox[source] = {
            key: {
                "callable": agent.invoke,
                "description": desc
            }
        }
        return key

    def _register_mcpo(self, base_url: str) -> str:
        """
        Register an MCP‑O (OpenAPI) proxy server as its own SOURCE bucket.
        Returns the SOURCE name (e.g., "__mcpo_server_1__").
        """
        if not self._allow_mcp_registration:
            raise RuntimeError("MCPO registration is not enabled for this agent.")

        host = base_url.rstrip('/')
        if host in self._mcpo_servers:
            return self._mcpo_servers[host][0]  # already registered → return group

        # Probe OpenAPI
        try:
            openapi = requests.get(f"{host}/openapi.json").json()
        except Exception:
            raise ValueError(f"Invalid MCP‑O server URL or no OpenAPI response: {host}")

        if not isinstance(openapi, dict) or "paths" not in openapi:
            raise ValueError(f"OpenAPI response missing 'paths' at {host}/openapi.json")

        # Create group
        self._mcpo_counter += 1
        group = f"__mcpo_server_{self._mcpo_counter}__"
        paths = {
            path: meta.get('post', {}).get('description', '')
            for path, meta in openapi.get('paths', {}).items()
            if isinstance(meta, dict)
        }

        # Single generic invoker
        def mcpo_invoke(path: str, payload: dict, base=host):
            resp = requests.post(f"{base}{path}", json=payload)
            try:
                return resp.json()
            except Exception:
                return resp.text

        key = f"{group}.mcpo_invoke"
        sig = self._build_signature(key, mcpo_invoke)
        desc = (
            f"{sig} — Calls the MCP‑O server at {host} with path+payload.\n"
            "Available paths:\n" + "\n".join(f"  - {p}: {d}" for p, d in paths.items())
        )

        self._toolbox[group] = {key: {"callable": mcpo_invoke, "description": desc}}
        self._mcpo_servers[host] = (group, paths)
        return group

    # in ToolAgent
    def _register_mcp(self, url_or_base: str) -> str:
        """
        Register a native MCP (Streamable HTTP) server as its own SOURCE bucket.
        Returns the SOURCE name (e.g., "__mcp_server_2__").
        """
        if not self._allow_mcp_registration:
            raise RuntimeError("MCP registration is not enabled for this agent.")

        base = url_or_base.rstrip('/')
        mcp_url = base if base.endswith("/mcp") else f"{base}/mcp"

        # Avoid double-registration for the same base URL
        if base in getattr(self, "_mcpo_servers", {}):
            return self._mcpo_servers[base][0]

        # --- 1) Probe the MCP server for its tool list ---------------------------
        async def _list_tools(u: str):
            async with streamablehttp_client(u) as (r, w, _sid):
                async with ClientSession(r, w) as session:
                    await session.initialize()
                    resp = await session.list_tools()
                    # resp.tools items typically have: name, description, input_schema (JSON schema)
                    tools = []
                    for t in resp.tools:
                        tools.append({
                            "name": t.name,
                            "description": getattr(t, "description", "") or t.name,
                            "schema": getattr(t, "input_schema", None) or getattr(t, "inputSchema", None) or {}
                        })
                    return tools

        try:
            tool_specs = asyncio.run(_list_tools(mcp_url))
        except Exception as e:
            raise ValueError(f"Failed to list MCP tools at '{mcp_url}': {e}")

        # --- 2) Create a SOURCE bucket for this MCP server -----------------------
        self._mcpo_counter += 1
        group = f"__mcp_server_{self._mcpo_counter}__"
        self._toolbox[group] = {}

        # Helper: pretty-print JSON schema into arg listing
        def _schema_to_args(schema: dict) -> tuple[list[str], list[str]]:
            props = []
            required = []
            if isinstance(schema, dict):
                props = list((schema.get("properties") or {}).keys())
                required = schema.get("required") or []
            return props, required

        # Optional: synthesize a "pretty signature" string from schema
        def _pretty_signature_from_schema(key: str, props: list[str], required: list[str]) -> str:
            # e.g., "__mcp_server_1__.mcp_derivative(func: Any [req], x: Any [req]) → Any"
            params = []
            for p in props:
                tag = " [req]" if p in required else ""
                params.append(f"{p}: Any{tag}")
            return f"{key}({', '.join(params)}) → Any"

        # --- 3) One wrapper per tool ---------------------------------------------
        def _make_tool_fn(u: str, tool_name: str):
            async def _acall(**payload):
                # ====== IMPORTANT: unwrap nested {'kwargs': {...}} if present ======
                # (This guards against planners that pass a single 'kwargs' arg.)
                if "kwargs" in payload and isinstance(payload["kwargs"], dict) and len(payload) == 1:
                    payload = payload["kwargs"]

                async with streamablehttp_client(u) as (r, w, _sid):
                    async with ClientSession(r, w) as session:
                        await session.initialize()
                        result = await session.call_tool(tool_name, arguments=payload)

                        # Prefer a simple text return if present; else return structured payload
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

            _sync.__name__ = f"mcp_{tool_name}"   # stable, readable name in toolbox keys
            return _sync

        # Register each tool with schema-aware description to guide planners
        name_to_desc = {}
        for spec in tool_specs:
            tool_name = spec["name"]
            tool_desc = spec["description"]
            props, required = _schema_to_args(spec["schema"])

            fn = _make_tool_fn(mcp_url, tool_name)
            key = f"{group}.{fn.__name__}"

            # Build a description that **exposes real arg names** (so planners avoid "kwargs")
            arg_block = ""
            if props:
                arg_lines = [f"  - {p}" + (" (required)" if p in required else " (optional)") for p in props]
                arg_block = "\nArgs:\n" + "\n".join(arg_lines)

            # Prefer a schema-derived signature over the generic **kwargs signature
            sig = _pretty_signature_from_schema(key, props, required) if props else self._build_signature(key, fn)

            self._toolbox[group][key] = {
                "callable": fn,
                "description": f"{sig} — MCP tool '{tool_name}' from {mcp_url}. {tool_desc}{arg_block}"
            }
            name_to_desc[tool_name] = tool_desc

        # --- 4) Track registration & return source name --------------------------
        self._mcpo_servers[base] = (group, name_to_desc)
        return group


    def register(self, tool: Any, description: str | None = None) -> None:
        """
        Dispatch-only entrypoint. Routes to the correct helper based on 'tool' type:
        - callables  -> self._register_callable(...)
        - plugins    -> self._register_plugin(...)
        - agents     -> self._register_agent(...)
        - URLs (str) -> self._register_mcp(...) or self._register_mcpo(...)
        """

        # 1) Plain Python function (method)
        if callable(tool):
            return self._register_callable(tool, description)

        # 2) Plugin instance
        if isinstance(tool, Plugin):
            return self._register_plugin(tool)

        # 3) Agent instance
        if issubclass(type(tool), Agent):
            if not self._allow_agent_registration:
                raise RuntimeError("Agent registration is not enabled for this agent.")
            return self._register_agent(tool)

        # 4) Server URL string: MCP vs MCP‑O
        if isinstance(tool, str):
            if not self._allow_mcp_registration:
                raise RuntimeError("MCP/MCP‑O registration is not enabled for this agent.")

            url = tool.strip()
            if not (url.startswith("http://") or url.startswith("https://")):
                raise ValueError(f"Expected an HTTP(S) URL for MCP/MCP‑O registration, got: {tool!r}")

            # Heuristic:
            # - Native MCP (Streamable HTTP) usually exposes /mcp
            # - MCP‑O (OpenAPI proxy) usually exposes /openapi.json at the base
            u = url.rstrip("/")
            if u.endswith("/mcp") or "/mcp" in u:
                return self._register_mcp(url)     # expects to create one wrapper per MCP tool
            else:
                return self._register_mcpo(url)    # expects to create a generic mcpo_invoke(...)

        # 5) Unknown
        raise TypeError(f"Cannot register object of type: {type(tool)}")

    def invoke(self, prompt):
        raise NotImplementedError("ToolAgent is abstract; use strategize() and execute() instead.")

    @property # toolbox should not be editable from the outside
    def toolbox(self):
        return self._toolbox.copy()
    @property
    def allow_agent_registration(self):
        return self._allow_agent_registration
    @allow_agent_registration.setter
    def allow_agent_registration(self, val: bool):
        self._allow_agent_registration = val
    @property
    def allow_mcp_registration(self):
        return self._allow_mcp_registration
    @allow_mcp_registration.setter
    def allow_mcp_registration(self, val:bool):
            self._allow_mcp_registration = val