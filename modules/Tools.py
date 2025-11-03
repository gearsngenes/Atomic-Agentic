# Tools.py
from __future__ import annotations

import asyncio
import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, get_type_hints, get_origin, get_args, TypedDict
from collections import OrderedDict

# External integrations (MCP) + local modules
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

# from modules.Agents import Agent
from modules.Plugins import *  # Provides Plugin-shaped dicts (see TypedDict below)


# ───────────────────────────────────────────────────────────────────────────────
# Public API
# ───────────────────────────────────────────────────────────────────────────────
__all__ = ["Tool", "ToolFactory"]

# ───────────────────────────────────────────────────────────────────────────────
# Tool
# ───────────────────────────────────────────────────────────────────────────────
from typing import Any, Mapping, Callable, OrderedDict as OrderedDictType, Dict, List, Optional, Tuple
from collections import OrderedDict
import inspect


class ToolError(Exception):
    """Base exception for Tool-related errors."""


class ToolDefinitionError(ToolError):
    """Raised when a callable is incompatible at Tool construction time."""


class ToolInvocationError(ToolError):
    """Raised when inputs are invalid for invocation or binding fails."""


# Sentinel to mark "no default" for parameters
NO_DEFAULT = object()


class Tool:
    """
    Tool
    ----
    Stateless wrapper around a Python callable with a dict-first invocation API.
    At construction, Tool builds an **arguments map** describing each named
    parameter (annotation, kind, default, and declaration position). During
    `invoke()`, Tool splits the provided mapping into a positional-only list
    (`args`) and a keyword dict (`kwargs`) according to that map, then calls
    `func(*args, **kwargs)`.

    Primary API
    -----------
    __init__(func: Callable, name: str, description: str = "",
             tool_type: str = "python", source: str = "local")

    invoke(inputs: Mapping[str, Any]) -> Any
        Deterministic binding by parameter **names**:
          • Positional-only params → collected into *args by **declaration order**,
            enforcing a contiguous prefix (no gaps).
          • Positional-or-keyword & keyword-only params → placed in **kwargs** by name.
          • If the function declares *args, extra positionals must be provided via
            '_args': list|tuple.
          • If the function declares **kwargs, extra keyword pairs must be provided
            via '_kwargs': Mapping.

    Metadata (tags)
    ---------------
    • type: str      # classification tag (e.g., "python", "agent", "workflow")
    • source: str    # provenance tag (e.g., "local", "remote:mcp", "plugin:my_pkg")

    Arguments Map (constructor-built, cached)
    -----------------------------------------
    arguments_map: OrderedDict[str, ArgSpec]
      Each entry captures:
        - index: int                         # declaration index (0-based)
        - kind: inspect._ParameterKind       # EXACT enum (not a string)
        - ann:  type | Any                   # from annotations; Any if absent
        - has_default: bool
        - default: value | NO_DEFAULT

    Call Plan (cached)
    ------------------
    • posonly_order: List[str]               # positional-only names in declaration order
    • p_or_kw_names: List[str]               # positional-or-keyword names
    • kw_only_names: List[str]               # keyword-only names
    • required_names: set[str]               # required among p_or_kw + kw_only (no defaults)
    • has_varargs: bool                      # whether *args exists
    • has_varkw: bool                        # whether **kwargs exists
    • varargs_name: Optional[str]
    • varkw_name: Optional[str]

    Strictness
    ----------
    - Unknown top-level keys are **errors** (even if the function has **kwargs);
      callers must place extras explicitly under '_kwargs'.
    - Duplicate provision (top-level and '_kwargs' for the same name) is an error.
    - Reserved keys must match required container types:
        '_args' -> list|tuple, '_kwargs' -> Mapping.

    Notes
    -----
    - Tool is **stateless**; it exposes no memory APIs.
    - Descriptions must document the function's purpose and the arguments
      (names/types/defaults) and return shape. Do **not** mention dict transport
      or reserved keys in descriptions.
    """

    # -------------------------
    # Construction & metadata
    # -------------------------
    def __init__(
        self,
        func: Callable,
        name: str,
        description: str = "",
        tool_type: str = "python",
        source: str = "local",
    ) -> None:
        self._func: Callable = func
        self._name: str = name
        self._description: str = description
        self._type: str = tool_type
        self._source: str = source

        # Build signature and call plan once (deterministic & fast at runtime)
        try:
            self._sig: inspect.Signature = inspect.signature(inspect.unwrap(func))
        except Exception as e:
            raise ToolDefinitionError(f"{name}: cannot introspect callable signature: {e}") from e

        (
            self.arguments_map,
            self.posonly_order,
            self.p_or_kw_names,
            self.kw_only_names,
            self.required_names,
            self.has_varargs,
            self.varargs_name,
            self.has_varkw,
            self.varkw_name,
        ) = self._build_arguments_map_and_plan(self._sig)

    # Read-only tags and doc
    @property
    def name(self) -> str:
        """Tool name (read-only)."""
        return self._name

    @property
    def description(self) -> str:
        """Function-centric description (read-only)."""
        return self._description

    @property
    def type(self) -> str:
        """Classification tag (read-only). E.g., 'python', 'agent', 'workflow'."""
        return self._type

    @property
    def source(self) -> str:
        """Provenance tag (read-only). E.g., 'local', 'remote:mcp', 'plugin:my_pkg'."""
        return self._source

    # -------------------------
    # Core invocation
    # -------------------------
    def invoke(self, inputs: Mapping[str, Any]) -> Any:
        """
        Invoke the wrapped callable by splitting `inputs` into:
          - args:  positional-only values (contiguous prefix by declaration order),
                   plus optional extras from '_args' if *args exists.
          - kwargs: all positional-or-keyword and keyword-only parameters by name,
                    plus optional extras from '_kwargs' if **kwargs exists.

        Strict validation:
          - Unknown top-level keys are errors (use '_kwargs' explicitly if **kwargs exists).
          - Reserved keys '_args' and '_kwargs' require correct container types.
          - Duplicate provision across top-level and '_kwargs' is an error.

        Returns:
            Any: result of `self._func(*args, **kwargs)`

        Raises:
            ToolInvocationError: on invalid inputs or binding mistakes.
        """
        if not isinstance(inputs, Mapping):
            raise ToolInvocationError(f"{self._name}: inputs must be a mapping")

        ARGS_KEY = "_args"
        KWARGS_KEY = "_kwargs"

        # Validate reserved keys early
        if ARGS_KEY in inputs:
            if not self.has_varargs:
                raise ToolInvocationError(f"{self._name}: '{ARGS_KEY}' provided but function declares no *args")
            extra_args = inputs[ARGS_KEY]
            if not isinstance(extra_args, (list, tuple)):
                raise ToolInvocationError(f"{self._name}: '{ARGS_KEY}' must be list or tuple")
        else:
            extra_args = ()

        if KWARGS_KEY in inputs:
            if not self.has_varkw:
                raise ToolInvocationError(f"{self._name}: '{KWARGS_KEY}' provided but function declares no **kwargs")
            extra_kwargs = inputs[KWARGS_KEY]
            if not isinstance(extra_kwargs, Mapping):
                raise ToolInvocationError(f"{self._name}: '{KWARGS_KEY}' must be a mapping")
        else:
            extra_kwargs = {}

        provided_names = set(inputs.keys()) - {ARGS_KEY, KWARGS_KEY}
        known_names = set(self.arguments_map.keys())
        unknown = sorted(provided_names - known_names)

        # Strict: unknown top-level keys are not allowed even if **kwargs exists
        if unknown:
            if not self.has_varkw:
                raise ToolInvocationError(f"{self._name}: unexpected keys: {unknown}")
            raise ToolInvocationError(
                f"{self._name}: unexpected keys {unknown}; place extras under '{KWARGS_KEY}' because function accepts **kwargs"
            )

        # Required named parameters must be present (pos-only handled separately)
        missing = sorted(self.required_names - provided_names)
        if missing:
            raise ToolInvocationError(f"{self._name}: missing required keys: {missing}")

        # -------------------------
        # Build args (positional-only), gap-safe
        # -------------------------
        args: List[Any] = []
        seen_gap = False
        for pname in self.posonly_order:
            present = pname in inputs
            if not present:
                seen_gap = True
                continue
            if seen_gap:
                # e.g., 'a' missing but 'b' present (both positional-only) — illegal
                raise ToolInvocationError(
                    f"{self._name}: positional-only gap: '{pname}' supplied after an earlier positional-only was missing"
                )
            args.append(inputs[pname])

        # After computing: extra_args, provided_names, and before building args/kwargs
        if extra_args:
            # How many pos-or-kw will be consumed by extra_args?
            consume = min(len(extra_args), len(self.p_or_kw_names))
            if consume:
                conflicting = [self.p_or_kw_names[i] for i in range(consume) if self.p_or_kw_names[i] in provided_names]
                if conflicting:
                    raise ToolInvocationError(
                        f"{self._name}: '_args' will bind positionally to {conflicting} "
                        "but those are also provided by name; use either '_args' or named keys, not both."
                    )


        # -------------------------
        # Build kwargs (pos_or_kw + kw_only)
        # -------------------------
        kwargs: Dict[str, Any] = {}
        for pname in self.p_or_kw_names + self.kw_only_names:
            if pname in inputs:
                kwargs[pname] = inputs[pname]
            # else: omit; Python applies default if available

        # Merge explicit var-keyword extras if any
        if extra_kwargs:
            # Check duplication between named kwargs and provided _kwargs
            dupes = sorted(set(kwargs.keys()) & set(extra_kwargs.keys()))
            if dupes:
                raise ToolInvocationError(
                    f"{self._name}: duplicate keys supplied both as named inputs and in '{KWARGS_KEY}': {dupes}"
                )
            kwargs.update(extra_kwargs)  # type: ignore[arg-type]

        # Finally call the function
        try:
            return self._func(*args, **kwargs)
        except TypeError as e:
            # Surface actionable context for signature mismatches
            raise ToolInvocationError(f"{self._name}: invocation failed: {e}") from e

    # -------------------------
    # Introspection & serialization
    # -------------------------
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize Tool metadata and signature plan for registries and UIs.
        Notes:
          - 'default' uses sentinel string '<NO_DEFAULT>' when absent.
          - 'ann' stores the annotation's repr for readability.
          - 'kind' is emitted as the enum name (e.g., 'POSITIONAL_ONLY').
        """
        def ann_repr(t: Any) -> str:
            if t is inspect._empty:
                return "Any"
            try:
                return getattr(t, "__name__", repr(t))
            except Exception:
                return repr(t)

        argmap_serialized = OrderedDict()
        for name, spec in self.arguments_map.items():
            kind_enum: inspect._ParameterKind = spec["kind"]
            argmap_serialized[name] = {
                "index": spec["index"],
                "kind": kind_enum.name,
                "ann": ann_repr(spec["ann"]),
                "has_default": spec["has_default"],
                "default": "<NO_DEFAULT>" if spec["default"] is NO_DEFAULT else repr(spec["default"]),
            }

        return {
            "name": self._name,
            "description": self._description,
            "type": self._type,
            "source": self._source,
            "signature": str(self._sig),
            "arguments_map": argmap_serialized,
            "posonly_order": list(self.posonly_order),
            "p_or_kw_names": list(self.p_or_kw_names),
            "kw_only_names": list(self.kw_only_names),
            "required_names": sorted(self.required_names),
            "has_varargs": self.has_varargs,
            "varargs_name": self.varargs_name,
            "has_varkw": self.has_varkw,
            "varkw_name": self.varkw_name,
        }

    # -------------------------
    # Internal: build arg map + call plan (ENUM kinds)
    # -------------------------
    def _build_arguments_map_and_plan(
        self, sig: inspect.Signature
    ) -> Tuple[
        OrderedDictType[str, Dict[str, Any]],  # arguments_map
        List[str],                             # posonly_order
        List[str],                             # p_or_kw_names
        List[str],                             # kw_only_names
        set,                                   # required_names
        bool, Optional[str],                   # has_varargs, varargs_name
        bool, Optional[str],                   # has_varkw,   varkw_name
    ]:
        """
        Partition parameters by kind (ENUM), capture annotations/defaults,
        and precompute call-time ordering and requirements.
        """
        arguments_map: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        posonly_order: List[str] = []
        p_or_kw_names: List[str] = []
        kw_only_names: List[str] = []
        required_names: set = set()

        has_varargs = False
        varargs_name: Optional[str] = None
        has_varkw = False
        varkw_name: Optional[str] = None

        index = 0
        for pname, p in sig.parameters.items():
            if p.kind is p.VAR_POSITIONAL:
                has_varargs = True
                varargs_name = pname
                continue
            if p.kind is p.VAR_KEYWORD:
                has_varkw = True
                varkw_name = pname
                continue

            ann = p.annotation if p.annotation is not inspect._empty else Any
            default = p.default if p.default is not inspect._empty else NO_DEFAULT
            kind_enum = p.kind  # store the EXACT enum

            if kind_enum is p.POSITIONAL_ONLY:
                posonly_order.append(pname)
            elif kind_enum is p.POSITIONAL_OR_KEYWORD:
                p_or_kw_names.append(pname)
            elif kind_enum is p.KEYWORD_ONLY:
                kw_only_names.append(pname)
            else:
                # Should be unreachable (varargs/varkw handled above)
                raise ToolDefinitionError(f"Unexpected parameter kind for {pname}: {kind_enum!r}")

            arguments_map[pname] = {
                "index": index,
                "kind": kind_enum,   # enum stored here
                "ann": ann,
                "has_default": default is not NO_DEFAULT,
                "default": default,
            }
            index += 1

        # Required named = those without defaults among pos_or_kw + kw_only
        for pname in p_or_kw_names + kw_only_names:
            if arguments_map[pname]["default"] is NO_DEFAULT:
                required_names.add(pname)

        return (
            arguments_map,
            posonly_order,
            p_or_kw_names,
            kw_only_names,
            required_names,
            has_varargs,
            varargs_name,
            has_varkw,
            varkw_name,
        )


# ───────────────────────────────────────────────────────────────────────────────
# ToolFactory
# ───────────────────────────────────────────────────────────────────────────────

class ToolFactory:
    """
    Factory helpers to convert common objects into `Tool` instances.

    Supported inputs (polymorphic):
      • Plain function          → one Tool ("function.default.<fn_name>")
      • Agent                   → four Tools (invoke/attach/detach/clear_memory) under type="agent"
      • Plugin-dict             → N Tools (one per method) under type="plugin"
      • MCP server (string URL) → N Tools (one per remote tool) under type="mcp"

    All generated `Tool` objects follow the same fully-qualified key convention
    and expose `.signature` and `.description` suitable for LLM prompting.
    """

    # ── Functions → Tools ──────────────────────────────────────────────────────
    @staticmethod
    def toolify_function(
        func: Callable[..., Any],
        type: str = "function",
        source: str = "default",
        description: str = "",
    ) -> List[Tool]:
        """
        Wrap a named Python function as a single Tool.

        Raises
        ------
        ValueError
            If func.__name__ == '<lambda>' (anonymous lambdas are unstable keys).
        """
        if func.__name__ == "<lambda>":
            raise ValueError("Lambda functions must be given proper names.")
        return [Tool(name=func.__name__, func=func, type=type, source=source, description=description)]

    # ── Agents → Tools ─────────────────────────────────────────────────────────
    @staticmethod
    def toolify_agent(agent: Agent) -> List[Tool]:
        """
        Expose an Agent as four tools under type='agent' and source=<agent.name>:

          agent.<name>.invoke(prompt: str) → str
              Invoke the agent with a prompt. Uses the agent's configured LLM engine.

          agent.<name>.attach(path: str) → bool
              Attach a local file path to the agent's internal attachment list.

          agent.<name>.detach(path: str) → bool
              Detach a local file path from the agent.

          agent.<name>.clear_memory() → None
              Clear only the agent's conversation history (attachments remain).
        """
        invoke_tool = ToolFactory.toolify_function(
            func=agent.invoke,
            type="agent",
            source=agent.name,
            description=f"Invoke the {agent.name} agent. Agent description: {agent.description}",
        )
        attach_tool = ToolFactory.toolify_function(
            func=agent.attach,
            type="agent",
            source=agent.name,
            description=f"Attach a local file path to {agent.name}'s attachments list. "
                        f"Use only when a specific path is explicitly required.",
        )
        detach_tool = ToolFactory.toolify_function(
            func=agent.detach,
            type="agent",
            source=agent.name,
            description=f"Detach a local file path from {agent.name}'s attachments list.",
        )
        clear_tool = ToolFactory.toolify_function(
            func=agent.clear_memory,
            type="agent",
            source=agent.name,
            description=f"Clear {agent.name}'s conversation history (attachments unaffected).",
        )
        return invoke_tool + attach_tool + detach_tool + clear_tool

    # ── Plugin dicts → Tools ───────────────────────────────────────────────────
    @staticmethod
    def toolify_plugin(plugin: Plugin) -> List[Tool]:
        """
        Convert a plugin dict into Tools (one per method in method_map).

        Each generated Tool is:
          type="plugin", source=<plugin['name']>, name=<method_name>

        The callable's __name__ is normalized to the method_name if it was a
        lambda (stable, readable keys for planners/orchestrators).
        """
        tools: List[Tool] = []
        source: str = plugin.get("name", "unknown")
        tool_map: Dict[str, Dict[str, Any]] = plugin.get("method_map", {})

        for method_name, method_info in tool_map.items():
            func = method_info.get("callable")
            description = method_info.get("description", "")
            if not callable(func):
                continue
            if getattr(func, "__name__", "<lambda>") == "<lambda>":
                # Stabilize the callable name to keep a readable fully-qualified key
                try:
                    func.__name__ = method_name  # type: ignore[attr-defined]
                except Exception:
                    # Fallback: best-effort; signature still contains parameter names
                    pass
            tools.append(
                Tool(
                    name=method_name,
                    func=func,
                    type="plugin",
                    source=source,
                    description=description,
                )
            )
        return tools

    # ── MCP server URL → Tools ─────────────────────────────────────────────────
    @staticmethod
    def toolify_mcp_server(name: str, url_or_base: str) -> List[Tool]:
        """
        Introspect a native MCP server and produce Tool wrappers for each remote tool.

        Parameters
        ----------
        name : str
            Logical server name (becomes `source` in '<type>.<source>.<name>').
            Must be stable—this is the key your plans will use.

        url_or_base : str
            Either a full MCP endpoint (endswith '/mcp') or a base URL.
            If base is provided, '/mcp' is appended.

        Returns
        -------
        list[Tool]
            One Tool per remote MCP tool. Each generated Tool is:
              type="mcp", source=<name>, name="mcp_<remote_tool_name>"

        Description format
        ------------------
        The Tool.description includes remote tool description and a compact
        "Args:" block listing schema properties and which are required.

        Invocation
        ----------
        The generated functions accept keyword arguments only (kwargs), matching
        the MCP tool's JSON schema. If a plan passes {"kwargs": {...}} it is
        unwrapped to raw keyword args for convenience.

        Notes
        -----
        - Calls are executed synchronously via `asyncio.run` to preserve
          compatibility with existing Planner/Orchestrator runners.
        - The wrapper attempts to return the first text part from MCP's result;
          if none exists, it falls back to `model_dump()` or the raw result.
        """
        if not name or not isinstance(name, str):
            raise ValueError("toolify_mcp_server requires a non-empty server name.")

        base = url_or_base.rstrip("/")
        mcp_url = base if base.endswith("/mcp") else f"{base}/mcp"

        async def _list_tools(u: str) -> List[Dict[str, Any]]:
            async with streamablehttp_client(u) as (r, w, _sid):
                async with ClientSession(r, w) as session:
                    await session.initialize()
                    resp = await session.list_tools()
                    out: List[Dict[str, Any]] = []
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

        def _schema_props(schema: Dict[str, Any]) -> Tuple[List[str], List[str]]:
            props: List[str] = []
            req: List[str] = []
            if isinstance(schema, dict):
                props = list((schema.get("properties") or {}).keys())
                req = schema.get("required") or []
            return props, req

        def _make_wrapper(u: str, tool_name: str) -> Callable[..., Any]:
            async def _acall(**payload: Any) -> Any:
                # Allow {"kwargs": {...}} shape from planners
                if "kwargs" in payload and isinstance(payload["kwargs"], dict) and len(payload) == 1:
                    payload = payload["kwargs"]

                async with streamablehttp_client(u) as (r, w, _sid):
                    async with ClientSession(r, w) as session:
                        await session.initialize()
                        result = await session.call_tool(tool_name, arguments=payload)
                        # Prefer plain text when available

                        # Fallback to dict-like form
                        try:
                            result_dict = result.model_dump()
                            if "structuredContent" in result_dict and result_dict["structuredContent"] is not None:
                                struct_cont = result_dict["structuredContent"]
                                if isinstance(struct_cont, dict) and "result" in struct_cont:
                                    return struct_cont.get("result")
                                else:
                                    return struct_cont
                            if getattr(result, "content", None):
                                texts = [getattr(c, "text", None) for c in result.content if getattr(c, "text", None)]
                                if texts:
                                    return texts[0]
                        except Exception:
                            return result

            def _sync(**payload: Any) -> Any:
                return asyncio.run(_acall(**payload))

            _sync.__name__ = f"mcp_{tool_name}"
            return _sync

        tools: List[Tool] = []
        for spec in specs:
            tool_name = spec["name"]
            desc = spec.get("description", "") or tool_name
            props, req = _schema_props(spec.get("schema", {}))
            fn = _make_wrapper(mcp_url, tool_name)

            # Include a compact arg summary in description
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

    # ── Polymorphic entrypoint → Tools ─────────────────────────────────────────
    @staticmethod
    def toolify(object: Any, name: Optional[str] = None, description: str = "") -> List[Tool]:
        """
        Polymorphic converter: return a list of Tools from a supported input.

        Accepted inputs
        ---------------
        - function:                returns [Tool] with type="function", source="default"
        - Agent:                   returns [Tool x4] (invoke/attach/detach/clear_memory) with type="agent"
        - Plugin-like dict:        returns [Tool xN] with type="plugin"
        - MCP server URL (str):    returns [Tool xN] with type="mcp" (requires `name`)

        Parameters
        ----------
        object : Any
            One of the accepted input types above.

        name : Optional[str]
            For MCP server URLs, this is REQUIRED and becomes the 'source' part.

        description : str
            Optional description forwarded to function-wrapped Tools.

        Raises
        ------
        ValueError
            If the input type is unsupported or if MCP URL is provided without a name.
        """
        if inspect.isfunction(object):
            return ToolFactory.toolify_function(func=object, description=description)

        if isinstance(object, Agent):
            return ToolFactory.toolify_agent(agent=object)

        if isinstance(object, dict) and "method_map" in object and "name" in object:
            # Treat as Plugin dict
            return ToolFactory.toolify_plugin(plugin=object)  # type: ignore[arg-type]

        if isinstance(object, str) and object.endswith("/mcp"):
            if not name:
                raise ValueError("toolify(object='/mcp', ...) requires a non-empty 'name' to use as source.")
            return ToolFactory.toolify_mcp_server(name=name, url_or_base=object)

        raise ValueError("Unsupported object type for toolification.")
