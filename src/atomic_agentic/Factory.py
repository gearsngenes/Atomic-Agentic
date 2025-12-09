import importlib
from typing import Any, Mapping

from .LLMEngines import *
from .Tools import *
from .Agents import *
from .ToolAgents import *
from .A2Agents import *
from .Exceptions import *

__all__ = [
    "load_llm",
    "load_tool",
    "load_agent",
]

VALID_AGENT_TYPES = ["Agent", "PlannerAgent", "OrchestratorAgent", "A2AProxyAgent", "A2AServerAgent"]
VALID_TOOL_TYPES = ["Tool", "AgentTool", "MCPProxyTool"]

def load_llm(data: Mapping[str, Any], **kwargs) -> LLMEngine:
    """Reconstruct an LLMEngine from a dict snapshot."""
    engine_type = data.get("provider")
    if engine_type == "LlamaCppEngine":
        """Reconstruct a LlamaCppEngine from a dict snapshot."""
        model_path = data.get("model_path")
        repo_id = data.get("repo_id")
        filename = data.get("filename")
        n_ctx = data.get("n_ctx", 2048)
        verbose = data.get("verbose", False)
        engine = LlamaCppEngine(
            model_path=model_path,
            repo_id=repo_id,
            filename=filename,
            n_ctx=n_ctx,
            verbose=verbose)
        engine._attachments = data.get("attachments", {})
        return engine
    elif engine_type == "MistralEngine":
        """Reconstruct a MistralEngine from a dict snapshot."""
        model = data.get("model")
        temperature = data.get("temperature", 0.1)
        inline_cutoff_chars = data.get("inline_cutoff_chars", 200_000)
        retry_sign_attempts = data.get("retry_sign_attempts", 5)
        retry_base_delay = data.get("retry_base_delay", 0.3)
        api_key = kwargs.get("api_key", None)
        engine = MistralEngine(
            model=model,
            temperature=temperature,
            inline_cutoff_chars=inline_cutoff_chars,
            retry_sign_attempts=retry_sign_attempts,
            retry_base_delay=retry_base_delay,
            api_key=api_key,
        )
        engine._attachments = data.get("attachments", {})
        return engine
    elif engine_type == "GeminiEngine":
        """Reconstruct a GeminiEngine from a dict snapshot."""
        model = data.get("model")
        temperature = data.get("temperature", 0.1)
        api_key = kwargs.get("api_key", None)
        engine = GeminiEngine(
            model=model,
            temperature=temperature,
            api_key=api_key)
        engine._attachments = data.get("attachments", {})
        return engine
    elif engine_type == "OpenAIEngine":
        """Reconstruct an OpenAIEngine from a dict snapshot."""
        model = data.get("model")
        temperature = data.get("temperature", 0.1)
        inline_cutoff_chars = data.get("inline_cutoff_chars", 200_000)
        api_key = kwargs.get("api_key", None)
        engine = OpenAIEngine(model=model,
                            temperature=temperature,
                            inline_cutoff_chars=inline_cutoff_chars,
                            api_key=api_key)
        engine._attachments = data.get("attachments", {})
        return engine
    else:
        raise ValueError(f"Unknown LLMEngine provider: {engine_type}")

def load_tool(data: Mapping[str, Any], **kwargs) -> Tool:
    """Reconstruct a Tool from a dict snapshot."""
    tool_type = data.get("tool_type", None)
    if tool_type is None or tool_type not in VALID_TOOL_TYPES:
        raise ToolInvocationError("Tool type must be specified and valid.")
    if tool_type == "Tool":
        """Reconstruct a Tool from a dict snapshot."""
        module = data.get("module")
        qualname = data.get("qualname")
        if module is None or qualname is None:
            raise ToolDefinitionError(
                "Cannot reconstruct Tool: 'module' and 'qualname' are required."
            )
        # Retrieve the callable
        try:
            mod = importlib.import_module(module)
            func = mod
            for attr in qualname.split("."):
                func = getattr(func, attr)
        except Exception as e:
            raise ToolDefinitionError(
                f"Cannot reconstruct Tool: failed to retrieve callable {module}.{qualname}: {e}"
            ) from e

        name = data.get("name", "Unnamed_Tool")
        description = data.get("description", "")
        source = data.get("source", "default")
        return Tool(
            func=func,
            name=name,
            description=description,
            source=source,
        )
    elif tool_type == "AgentTool":
        """
        Reconstruct an AgentTool from a dict snapshot.
        """
        agent_data = data.get("agent", None)
        if not agent_data:
            raise ToolDefinitionError("AgentTool: missing 'agent' data.")
        agent = load_agent(agent_data, **kwargs)
        return AgentTool(agent=agent)
    elif tool_type == "MCPProxyTool":
        """
        Reconstruct an MCPProxyTool from a dict snapshot.
        """
        server_url = data.get("mcp_url")
        server_name = data.get("source")
        tool_name = data.get("name")
        headers = data.get("headers", None)
        description = data.get("description", "")
        if not server_url or not server_name or not tool_name:
            raise ToolDefinitionError("MCPProxyTool: missing required fields.")
        return MCPProxyTool(
            server_url=server_url,
            server_name=server_name,
            tool_name=tool_name,
            headers=headers,
            description=description
        )

def load_agent(data: Mapping[str, Any], **kwargs) -> Agent:
    """Reconstruct an Agent from a dict snapshot."""
    agent_type = data.get("agent_type", None)
    if agent_type is None or agent_type not in VALID_AGENT_TYPES:
        raise AgentError(f"Agent type must be a valid option from {VALID_AGENT_TYPES}.")
    name = data.get("name", "UnnamedAgent")
    description = data.get("description", "")
    role_prompt = data.get("role_prompt", None)
    context_enabled = data.get("context_enabled", True)
    history_window = data.get("history_window", 50)
    pre_invoke_data = data.get("pre_invoke", {})
    pre_invoke = load_tool(pre_invoke_data, **kwargs) if pre_invoke_data else None
    llm_data = data.get("llm", None)
    llm_engine = load_llm(llm_data, **kwargs) if llm_data else None
    
    if agent_type == "Agent":
        agent = Agent(
            name=name,
            description=description,
            llm_engine=llm_engine,
            role_prompt=role_prompt,
            context_enabled=context_enabled,
            pre_invoke=pre_invoke,
            history_window=history_window,
        )
    elif agent_type in ["PlannerAgent", "OrchestratorAgent"]:
        """Deserialize a PlannerAgent from a dict (e.g., log/telemetry)."""
        tools = [load_tool(tool_dict) for tool_dict in data.get("tools", [])]
        name_collision_mode = data.get("name_collision_mode", "raise")
        if agent_type == "PlannerAgent":
            agent = PlannerAgent(
                name=name,
                description=description,
                llm_engine=llm_engine,
                context_enabled=context_enabled,
                pre_invoke=pre_invoke,
                history_window=history_window,
                run_concurrent=data.get("run_concurrent", False),
            )
        else:
            agent = OrchestratorAgent(
                name=name,
                description=description,
                llm_engine=llm_engine,
                context_enabled=context_enabled,
                pre_invoke=pre_invoke,
                history_window=history_window,
                max_steps=data.get("max_steps", 25),
                max_failures=data.get("max_failures", 5),
                context_budget_chars=data.get("context_budget_chars", 15_000),
            )
        agent.batch_register(tools, name_collision_mode=name_collision_mode)
    elif agent_type == "A2AServerAgent":
        seed = load_agent(data["seed"],**kwargs)
        agent = A2AServerAgent(
            seed = seed,
            name = name,
            description=description,
            host = data["host"],
            port = data["port"])
    elif agent_type == "A2AProxyAgent":
        agent = A2AProxyAgent(url = data["url"], name = name, description = description)
    agent._history = data.get("history", [])
    return agent