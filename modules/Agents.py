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
    def __init__(self, name, description: str, llm_engine:LLMEngine, role_prompt: str = Prompts.DEFAULT_PROMPT, context_enabled: bool = False):
        self._name = name
        self._llm_engine: LLMEngine = llm_engine
        self._role_prompt = role_prompt
        self._context_enabled = context_enabled
        self._description = description
        self._history = []

    @property
    def name(self):
        return self._name
    
    @property
    def description(self):
        description = f"~~Agent {self.name}~~\nA generic Agent for on Text-Text responses. Description: {self._description}"
        return description

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
        return self._history.copy()

    @context_enabled.setter
    def context_enabled(self, value:bool):
        self._context_enabled = value

    @llm_engine.setter
    def llm_engine(self, value: LLMEngine):
        self._llm_engine = value

    @name.setter
    def name(self, value: str):
        self._name = value
    
    @description.setter
    def description(self, value: str):
        self._description = value

    @role_prompt.setter
    def role_prompt(self, value: str):
        self._role_prompt = value

    def clear_memory(self):
        self._history = []
    
    def invoke(self, prompt: str):
        messages = [{"role": "system", "content": self._role_prompt}]
        if self._context_enabled:
            messages.extend(self._history)  # Include previous messages if context is enabled
        messages.append({"role": "user", "content": prompt})
        response = self._llm_engine.invoke(messages).strip()
        if self._context_enabled:
            self._history.append({"role": "user", "content": prompt})
            self._history.append({"role": "assistant", "content": response})
        return response

# ───────────────────────────────────────────────────────────────────────────────
# 2.  PrePostAgent  (calls methods to preprocess and postprocess an Agent's output
# ───────────────────────────────────────────────────────────────────────────────
class PrePostAgent(Agent):
    def __init__(self, name, description, llm_engine, role_prompt = Prompts.DEFAULT_PROMPT, context_enabled = False):
        Agent.__init__(self, name, description, llm_engine, role_prompt, context_enabled)
        self._preprocessors: list[callable] = []
        self._postprocessors: list[callable] = []
    
    # adds a new tool to the preprocessor chain
    def add_prestep(self, func: callable, index: int = None):
        # Only allow callables that do not return None
        hints = get_type_hints(func)
        rtype = hints.get('return', Any)
        if rtype is type(None):
            raise ValueError("Preprocessor tool cannot have return type None")
        if index is not None:
            self._preprocessors.insert(index, func)
        else:
            self._preprocessors.append(func)

    # adds a new tool to the postprocessor chain
    def add_poststep(self, func: callable, index: int = None):
        # Only allow callables that do not return None
        hints = get_type_hints(func)
        rtype = hints.get('return', Any)
        if rtype is type(None):
            raise ValueError("Preprocessor tool cannot have return type None")
        if index is not None:
            self._postprocessors.insert(index, func)
        else:
            self._postprocessors.append(func)
    def invoke(self, prompt: str):
        # 1. Pass prompt through preprocessor chain
        preprocessed = prompt
        for func in self._preprocessors:
            # Try to match argument count: if func takes >1 arg, pass result as first arg
            sig = inspect.signature(func)
            params = list(sig.parameters.values())
            if len(params) == 1:
                preprocessed = func(preprocessed)
            else:
                # If more than one arg, try to unpack if result is tuple/list
                if isinstance(preprocessed, (tuple, list)) and len(preprocessed) == len(params):
                    preprocessed = func(*preprocessed)
                elif isinstance(preprocessed, (tuple, list)) and len(preprocessed) != len(params):
                    raise ValueError(f"Preprocessor tool {func.__name__} expects {len(params)} args but got {len(preprocessed)}")
                else:
                    preprocessed = func(preprocessed)
        # 2. pass preprocessed result through the LLM
        processed = Agent.invoke(self, str(preprocessed))
        # 3. pass the processed prompt through the postprocessors
        postprocessed = processed
        for func in self._postprocessors:
            # Try to match argument count: if func takes >1 arg, pass result as first arg
            sig = inspect.signature(func)
            params = list(sig.parameters.values())
            if len(params) == 1:
                postprocessed = func(postprocessed)
            else:
                # If more than one arg, try to unpack if result is tuple/list
                if isinstance(postprocessed, (tuple, list)) and len(postprocessed) == len(params):
                    postprocessed = func(*postprocessed)
                elif isinstance(postprocessed, (tuple, list)) and len(postprocessed) != len(params):
                    raise ValueError(f"Preprocessor tool {func.__name__} expects {len(params)} args but got {len(preprocessed)}")
                else:
                    postprocessed = func(postprocessed)
        if self._context_enabled:
            self._history[-2] = {"role":"user", "content":prompt}
            self._history[-1] = {"role":"assistant", "content":str(postprocessed)}
        # 4. return post-processed result
        return postprocessed
    @property
    def description(self):
        desc = f"~~PrePost Agent {self.name}~~\nThis agent preprocesses inputs to the LLM before generating output, and then post-processes the output before returning it.\nDescription:{self._description}"
        return desc
    @description.setter
    def description(self, value: str):
        self._description = value
    @property
    def preprocessors(self):
        return self._preprocessors.copy()
    @preprocessors.setter # should be capable of being set in batches
    def preprocessor(self, value: list[callable]):
        self._preprocessors = value
    @property
    def postprocessors(self):
        return self._postprocessors.copy()
    @postprocessors.setter # should be capable of being set in batches
    def postprocessors(self, value: list[callable]):
        self._postprocessors = value


# ────────────────────────────────────────────────────────────────
# 3.  ChainSequenceAgent  (Invokes a chain of Agents)
# ────────────────────────────────────────────────────────────────
class ChainSequenceAgent(Agent):
    """
    A sequential Chain-of-Agents. Uses a flat internal list.
    Each agent's output is passed as input to the next.
    """
    def __init__(self, name: str, context_enabled: bool = False):
        self._agents:list[Agent] = []
        self._name = name
        self._context_enabled = context_enabled
        self._role_prompt = f"~~ChainSequence Agent {self._name}~~\nThis agent sequentially invokes a list of agents."
        self._history = []
        self._llm_engine = None
    @property
    def agents(self) -> list[Agent]:
        return self._agents.copy()

    @property
    def role_prompt(self):
        return self._role_prompt
    @property
    def description(self):
        desc = f"{self._role_prompt}\nDescription: this agent calls the following agents in order below:\n~~~start~~~{"".join(f"\n{agent._description}" for agent in self._agents)}\n~~~end~~~"
        return desc
    @property
    def llm_engine(self):
        return {a.name:a.llm_engine for a in self._agents}
    
    def add(self, agent:Agent, idx:int|None = None):
        if idx != None:
            self._agents.insert(idx, agent)
        else:
            self._agents.append(agent)
    
    def pop(self, idx:int|None = None) -> Agent:
        if idx != None:
            return self._agents.pop(idx)
        return self._agents.pop()
    
    def invoke(self, prompt: str):
        result = prompt
        if not self._agents:
            raise RuntimeError(f"Agents list is empty for ChainSequenceAgent '{self.name}'")
        if self.context_enabled:
            self._history.append({"role": "user", "content": "User-Input: " + prompt})
        for agent in self._agents:
            result = agent.invoke(str(result))
            if self.context_enabled:
                self._history.append({"role": "assistant", "content": agent.name + " - Partial Result: "+str(result)})
        if self.context_enabled:
            self._history.append({"role": "assistant", "content": self.name + "- Final Result: " + str(result)})
        return result

# ────────────────────────────────────────────────────────────────
# 4.  Human Agent  (Asks human for input, when provided a prompt)
# ────────────────────────────────────────────────────────────────
class HumanAgent(Agent):
    def __init__(self, name, description, context_enabled:bool = False):
        self._context_enabled = context_enabled
        self._name = name
        self._description = description
        self._llm_engine = None
        self._role_prompt = description
    def invoke(self, prompt:str):
        response = input(f"{prompt}\n{self.name}'s Response: ")
        if self._context_enabled:
            self._history.append({"role": "user", "content": prompt})
            self._history.append({"role": "assistant", "content": response})
        return response

from abc import ABC, abstractmethod
# ────────────────────────────────────────────────────────────────
# 5.  Abstract ToolAgent  (Uses Tools and Agents to execute tasks)
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

# ────────────────────────────────────────────────────────────────
# 6.  A2A ProxyAgent  (Invokes agents remotely via A2A protocol)
# ────────────────────────────────────────────────────────────────
from python_a2a import A2AClient
class A2AProxyAgent(Agent):
    def __init__(self, a2a_host: str):
        self._client = A2AClient(a2a_host)
        agent_card = self._client.get_agent_card()
        self._name, self._description = agent_card.name, agent_card.description
        self._context_enabled = False
        self._llm_engine = None
        self._history = []
    def invoke(self, prompt:str):
        response = self._client.ask(prompt)
        return response
    @property
    def description(self):
        return self._description
    @property
    def name(self):
        return self._name