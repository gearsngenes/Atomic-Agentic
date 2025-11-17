# atomic_ui.py
from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple
from dotenv import load_dotenv
import streamlit as st
load_dotenv()  # take environment variables from .env.
# ---- Atomic-Agentic modules (ensure importable in your env) ----
from modules.Agents import Agent
from modules.ToolAgents import PlannerAgent, OrchestratorAgent
from modules.LLMEngines import OpenAIEngine, GeminiEngine, MistralEngine

# ============================== Persistence ==============================
CONFIG_PATH = "agents.json"

@dataclass
class AgentCfg:
    name: str
    agent_type: str = "basic"      # "basic" | "planner" | "orchestrator"
    provider: str = "openai"       # "openai" | "gemini" | "mistral"
    model: str = "gpt-4o-mini"
    temperature: float = 0.2
    description: str = ""
    # basic-only:
    role_prompt: str = ""
    context_enabled: bool = True
    # planner/orchestrator:
    allowed_tools: List[str] = None
    allowed_agents: List[str] = None   # NEW: sub-agents this agent can call

SUPPORTED_PROVIDERS = ("openai", "gemini", "mistral")
ALWAYS_ON_TOOLS = {"_return"}  # always enabled (not shown)

def _cfg_from_dict(d: Dict[str, Any]) -> AgentCfg:
    allowed_tools = d.get("allowed_tools")
    if not isinstance(allowed_tools, list):
        allowed_tools = []
    allowed_agents = d.get("allowed_agents")
    if not isinstance(allowed_agents, list):
        allowed_agents = []
    return AgentCfg(
        name=(d.get("name") or "").strip() or "unnamed",
        agent_type=(d.get("agent_type") or "basic").lower(),
        provider=(d.get("provider") or "openai").lower(),
        model=d.get("model") or "gpt-4o-mini",
        temperature=float(d.get("temperature", 0.2)),
        description=d.get("description") or "",
        role_prompt=d.get("role_prompt") or "",
        context_enabled=bool(d.get("context_enabled", True)),
        allowed_tools=allowed_tools,
        allowed_agents=allowed_agents,
    )

def load_agent_configs_file() -> List[AgentCfg]:
    if not os.path.exists(CONFIG_PATH):
        return []
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            return []
        return [_cfg_from_dict(x) for x in data]
    except Exception:
        return []

def save_agent_configs_file(configs: List[AgentCfg]) -> None:
    try:
        payload = [asdict(c) for c in configs]
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
    except Exception as e:
        st.toast(f"Failed to save agents.json: {e}", icon="❌")

# ============================== Global tool-calls tape ==============================
import time
def record_tool_call(name: str, **kwargs) -> None:
    avatar = "🤖"
    content = _md_tool_block(name, dict(kwargs))
    with st.chat_message("assistant", avatar=avatar):
        st.markdown(content)
    time.sleep(0.2)

def _md_tool_block(name: str, payload: Dict[str, Any]) -> str:
    body = json.dumps(payload, indent=2, ensure_ascii=False)
    return f"**Tool:** `{name}`\n\n```json\n{body}\n```"

# ============================== Built-in Tools ==============================
def _return(x: Any) -> Any:
    record_tool_call("_return", value=x)
    return x

def add(a: float, b: float) -> float:
    record_tool_call("add", a=a, b=b); return a + b

def sub(a: float, b: float) -> float:
    record_tool_call("sub", a=a, b=b); return a - b

def mul(a: float, b: float) -> float:
    record_tool_call("mul", a=a, b=b); return a * b

def div(a: float, b: float) -> float:
    record_tool_call("div", a=a, b=b)
    if b == 0: raise ZeroDivisionError("b must not be 0")
    return a / b

def sqrt(x: float) -> float:
    record_tool_call("sqrt", x=x)
    if x < 0: raise ValueError("sqrt domain error: x must be >= 0")
    return math.sqrt(x)

def sin(x: float) -> float:
    record_tool_call("sin", x=x); return math.sin(x)

def cos(x: float) -> float:
    record_tool_call("cos", x=x); return math.cos(x)

def tan(x: float) -> float:
    record_tool_call("tan", x=x); return math.tan(x)

def radians_(deg: float) -> float:
    record_tool_call("radians", deg=deg); return math.radians(deg)

def degrees_(rad: float) -> float:
    record_tool_call("degrees", rad=rad); return math.degrees(rad)

# Optional: simple web-search (Tavily via langchain_tavily)
try:
    from langchain_tavily import TavilySearch, TavilyExtract
    searcher = TavilySearch(max_results=3)
    extractor = TavilyExtract()
    def web_search(query: str) -> str:
        record_tool_call("web_search", query=query)
        try:
            URLs = [result["url"] for result in searcher.run(query)["results"]]
            content = [result["raw_content"] for result in extractor.invoke({"urls":URLs})["results"]]
            final_content = "\n\n".join(content)
            return f"Web search results for query: {query}\n\n{final_content}"
        except Exception as e:
            return f"(web_search error: {e})"
    _HAS_WEB = True
except Exception:
    _HAS_WEB = False

BUILTIN_TOOLS: Dict[str, Tuple[Callable[..., Any], str]] = {
    "_return": (_return, "Echo the given value."),
    "add": (add, "Add two numbers a + b"),
    "sub": (sub, "Subtract b from a (a - b)"),
    "mul": (mul, "Multiply two numbers a * b"),
    "div": (div, "Divide a by b (b != 0)"),
    "sqrt": (sqrt, "Square root (x >= 0)"),
    "sin": (sin, "Sine"),
    "cos": (cos, "Cosine"),
    "tan": (tan, "Tangent"),
    "radians": (radians_, "Convert degrees to radians"),
    "degrees": (degrees_, "Convert radians to degrees"),
}
if _HAS_WEB:
    BUILTIN_TOOLS["web_search"] = (web_search, "Search the web and extract top snippets")

# ============================== Toolbox sync (register/remove/rebind for BUILTIN TOOLS only) ==============================
def sync_toolbox_with_selection(tool_agent, selected_tools: List[str]):
    """
    Reconcile BUILTIN tools using public ToolAgent APIs.
    - Register selected tools with source="builtin" (ALWAYS_ON pinned).
    - Remove non-selected builtins (do not touch user-registered tools).
    """
    if not hasattr(tool_agent, "list_tools"):
        return
    selected = set((selected_tools or [])) | set(ALWAYS_ON_TOOLS)

    # Desired full_names
    desired_full = {f"function.builtin.{name}" for name in selected if name in BUILTIN_TOOLS}

    # Register/replace desired builtins
    for name in sorted(selected):
        if name not in BUILTIN_TOOLS:
            continue
        fn, desc = BUILTIN_TOOLS[name]
        try:
            tool_agent.register(
                fn,
                name=name,
                description=desc,
                source="builtin",
                name_collision_mode="replace",
            )
        except Exception:
            pass

    # Prune builtins that are no longer selected (keep agent wrappers / user tools)
    try:
        registry = tool_agent.list_tools()
    except Exception:
        registry = {}
    for full_name in list(registry.keys()):
        if full_name.startswith("function.builtin.") and full_name not in desired_full:
            try:
                tool_agent.remove_tool(full_name)
            except Exception:
                pass

# ============================== Agent-to-Agent wrappers (as tools) ==============================
def _slugify_agent(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip().lower())
    s = re.sub(r"_+", "_", s)
    return s.strip("_") or "agent"

def _get_config_map() -> Dict[str, AgentCfg]:
    return {c.name: c for c in st.session_state.configs}

def _get_instance(name: str) -> Any:
    if name in st.session_state.instances:
        return st.session_state.instances[name]
    cfg_map = _get_config_map()
    if name not in cfg_map:
        raise KeyError(f"No agent config named '{name}'")
    inst = _instantiate_agent(cfg_map[name])
    st.session_state.instances[name] = inst
    return inst

def _make_agent_wrapper(sub_agent_name: str) -> Tuple[str, Callable[[str], str], str]:
    """
    Return (tool_name, callable, description) for a sub-agent wrapper tool.
    We register it via ToolAgent.register(..., name=..., description=..., source="agent").
    """
    tool_name = f"agent.{_slugify_agent(sub_agent_name)}.invoke"
    desc = f"Invoke sub-agent '{sub_agent_name}' with a prompt and return its reply."
    def wrapper(prompt: str) -> str:
        record_tool_call(f"agent_wrapper:{sub_agent_name}", prompt=prompt)
        sub = _get_instance(sub_agent_name)
        return _agent_send(sub, prompt)
    return tool_name, wrapper, desc

def sync_agent_wrappers(tool_agent, allowed_agents, cfg_map):
    """
    Keep sub-agent wrapper tools in sync using public ToolAgent APIs.
    - Register wrappers for allowed agents (except self) with source="agent".
    - Remove wrappers for agents that are no longer allowed/present.
    """
    if not hasattr(tool_agent, "list_tools"):
        return
    self_name = getattr(tool_agent, "name", "")
    allowed = set(a for a in (allowed_agents or []) if a in cfg_map and a != self_name)

    # Current registry
    registry = tool_agent.list_tools()  # OrderedDict[str, Tool]
    current_agent_wrappers = {
        k: v for k, v in registry.items()
        if k.startswith("function.agent.")
    }

    # Desired wrappers
    desired_names = set()
    for a in sorted(allowed):
        tname, fn, desc = _make_agent_wrapper(a)
        desired_names.add(f"function.agent.{tname}")
        tool_agent.register(
            fn,
            name=tname,
            description=desc,
            source="agent",
            name_collision_mode="replace",
        )

    # Prune wrappers that are no longer desired
    for full_name in list(current_agent_wrappers.keys()):
        if full_name not in desired_names:
            try:
                tool_agent.remove_tool(full_name)
            except Exception:
                pass

# ============================== Engines & Agents ==============================
def engine_factory(provider: str, model: str, temperature: float):
    provider = (provider or "openai").lower()
    if provider == "openai":
        return OpenAIEngine(model=model, temperature=temperature)
    if provider == "gemini":
        return GeminiEngine(model=model, temperature=temperature)
    if provider == "mistral":
        return MistralEngine(model=model, temperature=temperature)
    raise ValueError(f"Unsupported provider: {provider}")

def _instantiate_agent(cfg: AgentCfg) -> Any:
    engine = engine_factory(cfg.provider, cfg.model, cfg.temperature)

    if cfg.agent_type == "planner":
        inst = PlannerAgent(
            name=cfg.name,
            description=cfg.description,
            llm_engine=engine,
            run_concurrent=False
        )
        # tools only here; wrappers added by caller after instantiation (requires cfg_map)
        sync_toolbox_with_selection(inst, cfg.allowed_tools)
        return inst

    if cfg.agent_type == "orchestrator":
        inst = OrchestratorAgent(
            name=cfg.name,
            description=cfg.description,
            llm_engine=engine
        )
        sync_toolbox_with_selection(inst, cfg.allowed_tools)
        return inst

    return Agent(
        name=cfg.name,
        description=cfg.description,
        llm_engine=engine,
        role_prompt=cfg.role_prompt,
        context_enabled=cfg.context_enabled,
    )

# ============================== Rerun helper ==============================
def _do_rerun():
    if hasattr(st, "rerun"):
        st.rerun()

# ============================== Transcript helpers ==============================
def _ensure_transcript(name: str):
    st.session_state.transcripts.setdefault(name, [])

def _append_user_msg(name: str, content: str):
    _ensure_transcript(name)
    st.session_state.transcripts[name].append({"role": "user", "content": content})

def _append_assistant_msg(name: str, content: str):
    _ensure_transcript(name)
    st.session_state.transcripts[name].append({"role": "assistant", "content": content})

def _clear_transcript(name: str):
    st.session_state.transcripts[name] = []

# ============================== Agent send/reset (modern contracts) ==============================
def _agent_send(agent: Any, text: str) -> str:
    """
    Current Agents accept **mapping** inputs. The default pre-invoke Tool expects {"prompt": str}.
    Keep legacy fallbacks for older objects only.
    """
    if hasattr(agent, "invoke") and callable(agent.invoke):
        return str(agent.invoke({"prompt": text}))
    for m in ("generate_reply", "step", "respond", "chat"):
        if hasattr(agent, m) and callable(getattr(agent, m)):
            return str(getattr(agent, m)(text))
    raise RuntimeError("Agent does not implement invoke(...) or legacy chat methods.")

def _reset_agent_memory(agent: Any):
    # Prefer modern API; gracefully fall back for older objects.
    try:
        if hasattr(agent, "clear_memory") and callable(agent.clear_memory):
            agent.clear_memory(); return
    except Exception:
        pass
    try:
        if hasattr(agent, "clear_history") and callable(agent.clear_history):
            agent.clear_history(); return
    except Exception:
        pass
    try:
        if hasattr(agent, "reset") and callable(agent.reset):
            agent.reset(); return
    except Exception:
        pass

# ============================== Streamlit App ==============================
st.set_page_config(page_title="Atomic-Agentic UI", page_icon="🧪", layout="wide")
st.title("⚛️ Atomic-Agentic — Demo UI")

# Session boot
if "configs" not in st.session_state:
    st.session_state.configs: List[AgentCfg] = load_agent_configs_file()
    if not st.session_state.configs:
        st.session_state.configs = [
            AgentCfg(
                name="default-basic",
                agent_type="basic",
                provider="openai",
                model="gpt-4o-mini",
                temperature=0.2,
                description="A basic assistant.",
                role_prompt="You are a helpful, concise assistant.",
                context_enabled=True,
                allowed_tools=[],
                allowed_agents=[],
            ),
            AgentCfg(
                name="default-planner",
                agent_type="planner",
                provider="openai",
                model="gpt-4o-mini",
                temperature=0.2,
                description="A planner that can call tools.",
                allowed_tools=["add", "sub", "mul"],
                allowed_agents=[],
            )
        ]
        save_agent_configs_file(st.session_state.configs)

    if "instances" not in st.session_state:
        st.session_state.instances: Dict[str, Any] = {}
    if "transcripts" not in st.session_state:
        st.session_state.transcripts: Dict[str, List[Dict[str, str]]] = {}
    if "selected" not in st.session_state:
        st.session_state.selected: Optional[str] = (st.session_state.configs[0].name if st.session_state.configs else None)
    if "last_error" not in st.session_state:
        st.session_state.last_error: str = ""
    if "input_nonce" not in st.session_state:
        st.session_state.input_nonce = 0
    if "cfg_choice" not in st.session_state:
        st.session_state.cfg_choice = "(New)"

    for cfg in st.session_state.configs:
        _ensure_transcript(cfg.name)

def _config_names() -> List[str]:
    return [c.name for c in st.session_state.configs]

def _config_map() -> Dict[str, AgentCfg]:
    return {c.name: c for c in st.session_state.configs}

# ------------------------------ Tabs ------------------------------
tab_chat, tab_config, tab_diag = st.tabs(["Chat", "Manage", "Diagnostics"])

with tab_chat:
    left, right = st.columns([3, 2], gap="large")

    with left:
        st.subheader("Chat")

        if not st.session_state.configs:
            st.info("Create an agent in the **Manage** tab to begin.")
        else:
            # agent selector
            picked = st.selectbox(
                "Pick an agent",
                _config_names(),
                index=_config_names().index(st.session_state.selected) if st.session_state.selected in _config_names() else 0,
                key="agent_picker",
            )
            if picked != st.session_state.selected:
                st.session_state.selected = picked
                _do_rerun()

            agent = _get_instance(st.session_state.selected)

            # render transcript
            for msg in st.session_state.transcripts.get(agent.name, []):
                avatar = "🙂" if msg["role"] == "user" else "🤖"
                with st.chat_message(msg["role"], avatar=avatar):
                    st.markdown(msg["content"])

            # input row
            with st.container(border=True):
                ci_cols = st.columns([6, 1, 1])
                input_key = f"chat_input__{agent.name}__{st.session_state.input_nonce}"
                user_text = ci_cols[0].text_input("Message", key=input_key, label_visibility="collapsed")
                send_clicked = ci_cols[1].button("Send", use_container_width=True, key=f"send__{agent.name}")
                clear_clicked = ci_cols[2].button("Clear", type="secondary", use_container_width=True, key=f"clear__{agent.name}")

            if clear_clicked:
                try:
                    _clear_transcript(agent.name)
                    _reset_agent_memory(agent)
                finally:
                    st.session_state.input_nonce += 1
                    _do_rerun()

            if send_clicked:
                txt = (user_text or "").strip()
                if not txt:
                    st.toast("Please enter a message.", icon="⚠️")
                else:
                    _append_user_msg(agent.name, txt)
                    with st.chat_message("assistant", avatar="🤖"):
                        try:
                            # Sync tools/wrappers for planner/orchestrator
                            cfg_map = _config_map()
                            cfg = cfg_map.get(agent.name)
                            if cfg and cfg.agent_type in ("planner", "orchestrator"):
                                sync_toolbox_with_selection(agent, cfg.allowed_tools)
                                sync_agent_wrappers(agent, cfg.allowed_agents, cfg_map)

                            reply = _agent_send(agent, txt)

                            st.markdown(reply)
                            _append_assistant_msg(agent.name, reply)
                            st.session_state.last_error = ""
                        except Exception as e:
                            err = f"{type(e).__name__}: {e}"
                            st.session_state.last_error = err
                            st.error(err)

    with right:
        st.subheader("Tools & Sub-Agents")
        if not st.session_state.configs or st.session_state.selected not in _config_map():
            st.info("No agent selected.")
        else:
            cfg = _config_map()[st.session_state.selected]
            is_tooled = cfg.agent_type in ("planner", "orchestrator")

            # Tool toggles (builtins)
            with st.expander("Built-in Tools", expanded=is_tooled):
                if is_tooled:
                    default_tools = set(cfg.allowed_tools or [])
                    tcols = st.columns(3)
                    selected = []
                    names = sorted([n for n in BUILTIN_TOOLS.keys() if n not in ALWAYS_ON_TOOLS])
                    for i, n in enumerate(names):
                        with tcols[i % len(tcols)]:
                            if st.checkbox(n, value=(n in default_tools), key=f"toolchk__{cfg.name}__{n}"):
                                selected.append(n)
                    cfg.allowed_tools = selected
                    # live sync on the instance if present
                    try:
                        inst = _get_instance(cfg.name)
                        sync_toolbox_with_selection(inst, cfg.allowed_tools)
                    except Exception:
                        pass
                else:
                    st.caption("This agent type does not use tools.")

            # Sub-agent wrappers
            with st.expander("Sub-Agents (as tools)", expanded=False):
                if is_tooled:
                    cfg_map = _config_map()
                    anames = [n for n in _config_names() if n != cfg.name]
                    default_agents = set(cfg.allowed_agents or [])
                    acols = st.columns(3)
                    selected_agents = []
                    for i, aname in enumerate(sorted(anames)):
                        with acols[i % len(acols)]:
                            if st.checkbox(aname, value=(aname in default_agents), key=f"agentchk__{cfg.name}__{aname}"):
                                selected_agents.append(aname)
                    cfg.allowed_agents = selected_agents
                    # live sync
                    try:
                        inst = _get_instance(cfg.name)
                        sync_agent_wrappers(inst, cfg.allowed_agents, cfg_map)
                    except Exception:
                        pass
                else:
                    st.caption("This agent type does not use sub-agent tools.")

with tab_diag:
    st.subheader("Diagnostics")
    st.caption("Current config & instance state")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Configs**")
        try:
            st.code(json.dumps([asdict(c) for c in st.session_state.configs], indent=2, ensure_ascii=False), language="json")
        except Exception as e:
            st.error(f"Failed to dump configs: {e}")
    with c2:
        st.markdown("**Instances**")
        try:
            info = {}
            for k, v in st.session_state.instances.items():
                try:
                    info[k] = {
                        "class": v.__class__.__name__,
                        "has_list_tools": hasattr(v, "list_tools"),
                    }
                except Exception:
                    info[k] = {"class": str(type(v))}
            st.code(json.dumps(info, indent=2, ensure_ascii=False), language="json")
        except Exception as e:
            st.error(f"Failed to introspect instances: {e}")

    st.markdown("---")
    st.markdown("**Last Error**")
    if st.session_state.last_error:
        st.error(st.session_state.last_error)
    else:
        st.caption("No errors.")

with tab_config:
    st.subheader("Manage Agents")

    existing = [c.name for c in st.session_state.configs]
    choice = st.selectbox(
        "Select",
        ["(New)"] + existing,
        index=0 if st.session_state.get("cfg_choice","(New)")=="(New)"
              else (existing.index(st.session_state["cfg_choice"])+1 if st.session_state["cfg_choice"] in existing else 0),
        key="cfg_picker"
    )
    if choice != st.session_state.get("cfg_choice","(New)"):
        st.session_state.cfg_choice = choice
        _do_rerun()

    creating_new = choice == "(New)"
    src_cfg = None if creating_new else _get_config_map()[choice]

    # Agent type: RADIO (no callbacks; Streamlit auto-reruns and stays on this tab)
    agent_type = st.radio(
        "Category",
        options=["basic", "planner", "orchestrator"],
        index=(0 if creating_new else {"basic":0, "planner":1, "orchestrator":2}[src_cfg.agent_type]),
        key="edit_agent_type",
        horizontal=True,
    )

    provider = st.selectbox(
        "LLM Provider",
        options=list(SUPPORTED_PROVIDERS),
        index=(0 if creating_new else list(SUPPORTED_PROVIDERS).index(src_cfg.provider)),
        key="edit_provider"
    )

    model = st.text_input(
        "Model",
        value=("gpt-4o-mini" if creating_new else src_cfg.model),
        key="edit_model",
        help="Provider-specific model id"
    )

    temperature = st.slider(
        "Temperature",
        min_value=0.0, max_value=1.5, step=0.1,
        value=(0.2 if creating_new else float(src_cfg.temperature)),
        key="edit_temp"
    )

    name = st.text_input(
        "Name",
        value=("new-agent" if creating_new else src_cfg.name),
        key="edit_name"
    )

    description = st.text_area(
        "Description",
        value=("" if creating_new else src_cfg.description),
        key="edit_description"
    )

    if agent_type == "basic":
        role_prompt = st.text_area(
            "Role/System Prompt",
            value=("You are a helpful assistant." if creating_new else src_cfg.role_prompt),
            key="edit_role_prompt"
        )
        context_enabled = st.toggle(
            "Keep conversation context",
            value=(True if creating_new else bool(src_cfg.context_enabled)),
            key="edit_context_enabled"
        )
    else:
        st.caption("This type uses tools/sub-agents; role prompt & context are not shown.")

    st.markdown("**Built-in Tools**")
    if agent_type in ("planner", "orchestrator"):
        tcols = st.columns(3)
        ns = name or "(new)"
        selected_tools = []
        names = sorted([n for n in BUILTIN_TOOLS.keys() if n not in ALWAYS_ON_TOOLS])
        default_tools = (set([]) if creating_new else set(src_cfg.allowed_tools or []))
        for i, n in enumerate(names):
            with tcols[i % len(tcols)]:
                if st.checkbox(n, value=(n in default_tools), key=f"toolchk__{ns}__{n}__manage"):
                    selected_tools.append(n)
    else:
        selected_tools = []

    st.markdown("**Sub-Agents**")
    if agent_type in ("planner", "orchestrator"):
        cfg_map = _config_map()
        anames = [n for n in _config_names() if (creating_new or n != src_cfg.name)]
        acols = st.columns(3)
        ns = name or "(new)"
        selected_agents = []
        default_agents = (set([]) if creating_new else set(src_cfg.allowed_agents or []))
        for i, aname in enumerate(sorted(anames)):
            with acols[i % len(acols)]:
                if st.checkbox(aname, value=(aname in default_agents), key=f"agentchk__{ns}__{aname}__manage"):
                    selected_agents.append(aname)
    else:
        selected_agents = []

    c1, c2, _ = st.columns([1, 1, 6])
    save_clicked = c1.button("Save", key="edit_save")
    delete_clicked = (False if creating_new else c2.button("Delete", key="edit_delete"))

    if save_clicked:
        nm = (name or "").strip()
        agt_type = agent_type or "basic"
        prov = provider or "openai"
        mdl = (model or "gpt-4o-mini").strip()
        temp = float(temperature or 0.2)
        desc = (description or "").strip()

        if not nm:
            st.toast("Name is required.", icon="⚠️")
        elif prov not in SUPPORTED_PROVIDERS:
            st.toast(f"Unsupported provider: {prov}", icon="⚠️")
        else:
            # Build new config
            if agt_type == "basic":
                rp = st.session_state.get("edit_role_prompt") or ""
                ctx = bool(st.session_state.get("edit_context_enabled", True))
                new_cfg = AgentCfg(
                    name=nm, agent_type=agt_type, provider=prov, model=mdl,
                    temperature=temp, description=desc,
                    role_prompt=rp, context_enabled=ctx,
                    allowed_tools=[], allowed_agents=[]
                )
            else:
                new_cfg = AgentCfg(
                    name=nm, agent_type=agt_type, provider=prov, model=mdl,
                    temperature=temp, description=desc,
                    allowed_tools=selected_tools, allowed_agents=selected_agents
                )

            # Insert/replace in configs
            names = _config_names()
            if nm in names:
                idx = names.index(nm)
                st.session_state.configs[idx] = new_cfg
            else:
                st.session_state.configs.append(new_cfg)

            save_agent_configs_file(st.session_state.configs)
            st.session_state.cfg_choice = nm
            st.session_state.selected = nm

            # Refresh instance immediately
            try:
                inst = _instantiate_agent(new_cfg)
                cfg_map = _config_map()
                if new_cfg.agent_type in ("planner", "orchestrator"):
                    sync_toolbox_with_selection(inst, new_cfg.allowed_tools)
                    sync_agent_wrappers(inst, new_cfg.allowed_agents, cfg_map)
                st.session_state.instances[new_cfg.name] = inst
            except Exception as e:
                st.toast(f"Rebuild instance failed: {e}", icon="⚠️")

            st.toast("Saved.", icon="✅")
            _do_rerun()

    if (not creating_new) and delete_clicked:
        victim = src_cfg.name
        st.session_state.configs = [c for c in st.session_state.configs if c.name != victim]
        st.session_state.instances.pop(victim, None)
        st.session_state.transcripts.pop(victim, None)
        if st.session_state.configs:
            st.session_state.selected = st.session_state.configs[0].name
            st.session_state.cfg_choice = st.session_state.selected
            _ensure_transcript(st.session_state.selected)
        else:
            st.session_state.configs = [
                AgentCfg(
                    name="default",
                    agent_type="basic",
                    provider="openai",
                    model="gpt-4o-mini",
                    temperature=0.2,
                    description="Default basic agent.",
                    role_prompt="",
                    context_enabled=True,
                    allowed_tools=[],
                    allowed_agents=[],
                )
            ]
            st.session_state.selected = st.session_state.configs[0].name
            st.session_state.cfg_choice = st.session_state.selected
            _ensure_transcript(st.session_state.selected)
        save_agent_configs_file(st.session_state.configs)
        st.toast(f"Deleted '{victim}'.", icon="🗑️")
        _do_rerun()
