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
        allowed_tools=[str(x) for x in allowed_tools],
        allowed_agents=[str(x) for x in allowed_agents],
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
    _append_chat_line(agent.name, "assistant", content)

# ============================== Transcript helpers ==============================
def _slugify_agent(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_]+", "_", name).strip("_").lower()
    return s or "agent"

def _make_agent_wrapper(sub_agent_name: str):
    """
    Wrapper callable that:
      - logs to TOOL_CALLS: name="Agent: <sub_agent_name>", args={"prompt": prompt}
      - preflight-syncs the sub-agent's toolbox/wrappers
      - calls sub-agent and returns its result as str
    """
    def wrapper(prompt: str):
        record_tool_call(f"Agent: {sub_agent_name}", prompt=prompt)
        sub = _get_instance(sub_agent_name)
        _preflight_sync_for(sub_agent_name, sub)
        return _agent_send(sub, prompt)
    wrapper.__name__ = f"agent_{_slugify_agent(sub_agent_name)}_invoke"
    return wrapper

def sync_agent_wrappers(tool_agent, allowed_agents, cfg_map):
    """
    Ensure __dev_tools__.agent_<slug>_invoke entries match 'allowed_agents'
    (present in cfg_map and not self).
    - Add/register missing wrappers with sub-agent description.
    - Rebind callables for existing wrappers (fresh closure).
    - Remove wrappers for agents no longer allowed/present.
    """
    if not hasattr(tool_agent, "name"):
        return
    self_name = getattr(tool_agent, "name", "")
    present = [a for a in (allowed_agents or []) if a in cfg_map and a != self_name]

    wanted = {}
    for a in present:
        fn = _make_agent_wrapper(a)
        desc = cfg_map[a].description or f"Call sub-agent '{a}' (invoke)."
        key = f"__dev_tools__.{fn.__name__}"  # __dev_tools__.agent_<slug>_invoke
        wanted[key] = (fn, desc)

    tb = getattr(tool_agent, "_toolbox", None)
    if not isinstance(tb, dict):
        for _, (fn, desc) in wanted.items():
            try:
                tool_agent.register(fn, desc)
            except Exception:
                pass
        return

    bucket = tb.get("__dev_tools__")
    if not isinstance(bucket, dict):
        for _, (fn, desc) in wanted.items():
            try:
                tool_agent.register(fn, desc)
            except Exception:
                pass
        bucket = tb.get("__dev_tools__")
        if not isinstance(bucket, dict):
            return

    # remove wrappers not wanted anymore
    for key in list(bucket.keys()):
        if key.startswith("__dev_tools__.agent_") and key.endswith("_invoke"):
            if key not in wanted:
                bucket.pop(key, None)

    # add or rebind wrappers
    for key, (fn, desc) in wanted.items():
        cell = bucket.get(key)
        if isinstance(cell, dict) and "callable" in cell:
            cell["callable"] = fn
        elif cell is not None and callable(cell):
            bucket[key] = fn
        else:
            try:
                tool_agent.register(fn, desc)
            except Exception:
                pass

def _preflight_sync_for(name: str, inst):
    """If a sub-agent is Planner/Orchestrator, sync its tools and agent-wrappers to the saved config."""
    cfg_map = _get_config_map()
    cfg = cfg_map.get(name)
    if not cfg:
        return
    if cfg.agent_type in ("planner", "orchestrator") and hasattr(inst, "_toolbox"):
        sync_toolbox_with_selection(inst, cfg.allowed_tools)
        sync_agent_wrappers(inst, cfg.allowed_agents, cfg_map)



def _ensure_transcript(name: str) -> None:
    st.session_state.transcripts.setdefault(name, [])

def _append_chat_line(agent_name: str, role: str, content_md: str) -> None:
    _ensure_transcript(agent_name)
    st.session_state.transcripts[agent_name].append({"role": role, "content": content_md})

def _clear_transcript(name: str) -> None:
    st.session_state.transcripts[name] = []

def _md_tool_block(name: str, args: Dict[str, Any]) -> str:
    for arg in args:
        if isinstance(args[arg], str) and len(args[arg]) > 1000:
            args[arg] = args[arg][:300] + "...[argument truncated for brevity]"
    if name.startswith("Agent: "):
        _name = name[7:].strip()
        return f"**Agent:** {_name}\n\n**Prompt:**\n```\n{args.get('prompt','')}\n```"
    return f"**Tool:** {name}\n\n**Args**:\n```json\n{json.dumps(args, indent=2, ensure_ascii=False)}\n```"

# ============================== Built-in tools (explicit, no Streamlit) ==============================
def _return(val: Any) -> Any:
    record_tool_call("_return", val=val); return val

def add(a: float, b: float) -> float:
    record_tool_call("add", a=a, b=b); return a + b

def sub(a: float, b: float) -> float:
    record_tool_call("sub", a=a, b=b); return a - b

def mul(a: float, b: float) -> float:
    record_tool_call("mul", a=a, b=b); return a * b

def div(a: float, b: float) -> float:
    record_tool_call("div", a=a, b=b)
    if b == 0: raise ZeroDivisionError("Division by zero")
    return a / b

def pow_(a: float, b: float) -> float:
    record_tool_call("pow", a=a, b=b); return math.pow(a, b)

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
        return f"Web search failed: {e}"
# match toolbox names
pow_.__name__ = "pow"
radians_.__name__ = "radians"
degrees_.__name__ = "degrees"

BUILTIN_TOOLS: Dict[str, Tuple[Callable[..., Any], str]] = {
    "_return": (_return, "Return the supplied value (terminal step)."),
    "add": (add, "add(a: float, b: float) -> float"),
    "sub": (sub, "sub(a: float, b: float) -> float"),
    "mul": (mul, "mul(a: float, b: float) -> float"),
    "div": (div, "div(a: float, b: float) -> float; raises on b==0"),
    "pow": (pow_, "pow(a: float, b: float) -> float"),
    "sqrt": (sqrt, "sqrt(x: float) -> float; x>=0"),
    "sin": (sin, "sin(x: float) -> float (radians)"),
    "cos": (cos, "cos(x: float) -> float (radians)"),
    "tan": (tan, "tan(x: float) -> float (radians)"),
    "radians": (radians_, "radians(deg: float) -> float"),
    "degrees": (degrees_, "degrees(rad: float) -> float"),
    "web_search": (web_search, "web_search(query: str) -> str; performs a web search and returns the amalgamated research content."),
}

# ============================== Agent wrappers ==============================
def _slugify_agent(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_]+", "_", name).strip("_").lower()
    return s or "agent"


# ============================== Toolbox sync (register/remove/rebind for BUILTIN TOOLS only) ==============================
def sync_toolbox_with_selection(tool_agent: Any, selected: List[str] | None, always_on: set[str] = ALWAYS_ON_TOOLS) -> None:
    """
    Bring tool_agent._toolbox['__dev_tools__'] to match selected ∪ always_on for BUILTIN_TOOLS.
    Important: DOES NOT remove keys for agent wrappers (__dev_tools__.agent_*_invoke).
    """
    target = set(selected or [])
    target |= set(always_on)

    want: Dict[str, Tuple[Callable[..., Any], str]] = {}
    for name in target:
        if name in BUILTIN_TOOLS:
            want[f"__dev_tools__.{name}"] = BUILTIN_TOOLS[name]

    tb = getattr(tool_agent, "_toolbox", None)
    if not isinstance(tb, dict):
        for _, (fn, desc) in want.items():
            try: tool_agent.register(fn, desc)
            except Exception: pass
        return

    bucket = tb.get("__dev_tools__")
    if not isinstance(bucket, dict):
        for _, (fn, desc) in want.items():
            try: tool_agent.register(fn, desc)
            except Exception: pass
        bucket = tb.get("__dev_tools__")
        if not isinstance(bucket, dict):
            return

    # Remove only unselected BUILTIN tool keys; DO NOT touch agent wrappers
    for key in list(bucket.keys()):
        if key.startswith("__dev_tools__.agent_") and key.endswith("_invoke"):
            continue  # leave wrappers alone
        if key.startswith("__dev_tools__."):
            if key not in want:
                bucket.pop(key, None)

    # Add or rebind selected builtin tools
    for key, (fn, desc) in want.items():
        cell = bucket.get(key)
        if isinstance(cell, dict) and "callable" in cell:
            cell["callable"] = fn
        elif cell is not None and callable(cell):
            bucket[key] = fn
        else:
            try: tool_agent.register(fn, desc)
            except Exception: pass

# ============================== Engines & Agents ==============================
def engine_factory(provider: str, model: str, temperature: float):
    p = (provider or "").lower()
    if p == "openai":
        return OpenAIEngine(model=model, temperature=temperature)
    if p == "gemini":
        return GeminiEngine(model=model, temperature=temperature)
    if p == "mistral":
        return MistralEngine(model=model, temperature=temperature)
    raise ValueError(f"Unsupported provider: {provider!r} (expected one of {SUPPORTED_PROVIDERS}).")

def agent_factory(cfg: AgentCfg) -> Any:
    engine = engine_factory(cfg.provider, cfg.model, cfg.temperature)

    if cfg.agent_type == "planner":
        inst = PlannerAgent(
            name=cfg.name,
            description=cfg.description,
            llm_engine=engine,
            is_async=False,     # force sync
            allow_agentic=True,
        )
        # tools only here; wrappers added by caller after instantiation (requires cfg_map)
        sync_toolbox_with_selection(inst, cfg.allowed_tools)
        return inst

    if cfg.agent_type == "orchestrator":
        inst = OrchestratorAgent(
            name=cfg.name,
            description=cfg.description,
            llm_engine=engine,
            allow_agentic=True,
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
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()

# ============================== Session bootstrap ==============================
def _bootstrap_defaults():
    if "configs" not in st.session_state:
        loaded = load_agent_configs_file()
        if loaded:
            st.session_state.configs: List[AgentCfg] = loaded
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

def _get_config_map() -> Dict[str, AgentCfg]:
    return {c.name: c for c in st.session_state.configs}

def _get_instance(name: str) -> Any:
    if name in st.session_state.instances:
        return st.session_state.instances[name]
    cfg_map = _get_config_map()
    if name not in cfg_map:
        raise KeyError(f"No such agent config: {name!r}")
    inst = agent_factory(cfg_map[name])
    st.session_state.instances[name] = inst
    _ensure_transcript(name)
    return inst

def _reset_agent_memory(agent: Any):
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
    for attr in ("history", "_history", "messages", "_messages", "_previous_steps"):
        if hasattr(agent, attr):
            try:
                obj = getattr(agent, attr)
                if isinstance(obj, list): obj.clear()
            except Exception:
                pass

def _agent_send(agent: Any, text: str) -> str:
    if hasattr(agent, "invoke") and callable(agent.invoke):
        return str(agent.invoke(text))
    for m in ("generate_reply", "step", "respond", "chat"):
        if hasattr(agent, m) and callable(getattr(agent, m)):
            return str(getattr(agent, m)(text))
    raise RuntimeError("Agent does not implement invoke/generate_reply/step/respond/chat.")

# ============================== UI ==============================
st.set_page_config(page_title="Atomic-Agentic — Agents & Tools", page_icon="🧰", layout="wide")
_bootstrap_defaults()

tab_chat, tab_config = st.tabs(["💬 Chat", "⚙️ Config"])

with tab_chat:
    left, right = st.columns([0.67, 0.33], gap="large")

    with left:
        st.subheader("Chat")
        names = [c.name for c in st.session_state.configs]
        if not names:
            st.info("No agents yet. Create one in the Config tab.")
        else:
            current = st.session_state.selected or names[0]
            try:
                idx = names.index(current)
            except ValueError:
                idx = 0
                st.session_state.selected = names[0]

            selected = st.selectbox("Select agent", names, index=idx, key="agent_select")
            if selected != st.session_state.selected:
                st.session_state.selected = selected
                _ensure_transcript(selected)
                st.session_state.input_nonce += 1
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
                    try:
                        _append_chat_line(agent.name, "user", txt)

                        # Sync toolbox + wrappers to selected config before running
                        cfg_map = _get_config_map()
                        cfg = cfg_map.get(agent.name)
                        if cfg and cfg.agent_type in ("planner", "orchestrator") and hasattr(agent, "_toolbox"):
                            sync_toolbox_with_selection(agent, cfg.allowed_tools)
                            sync_agent_wrappers(agent, cfg.allowed_agents, cfg_map)

                        reply = _agent_send(agent, txt)

                        # stitch tool/agent call logs then final reply
                        _append_chat_line(agent.name, "assistant", reply)

                        st.session_state.last_error = ""
                    except Exception as e:
                        _append_chat_line(agent.name, "assistant", f"❌ Error: {e}")
                        st.session_state.last_error = str(e)
                    finally:
                        st.session_state.input_nonce += 1
                        _do_rerun()

    with right:
        st.subheader("Agent Info")
        if st.session_state.selected:
            cfg_map = _get_config_map()
            cfg = cfg_map.get(st.session_state.selected)
            if cfg:
                st.write("**Name:**", cfg.name)
                st.write("**Type:**", cfg.agent_type.title())
                st.write("**Provider:**", cfg.provider)
                st.write("**Model:**", cfg.model)
                st.write("**Temperature:**", cfg.temperature)
                st.write("**Description:**"); st.caption(cfg.description or "—")
                if cfg.agent_type == "basic":
                    st.write("**Context Enabled:**", "Yes" if cfg.context_enabled else "No")
                    with st.expander("Role/System Prompt", expanded=False):
                        st.code(cfg.role_prompt or "—", language="markdown")
                else:
                    st.write("**Allowed Tools:**", ", ".join(cfg.allowed_tools or []) or "—")
                    st.write("**Allowed Sub-agents:**", ", ".join(cfg.allowed_agents or []) or "—")

        st.markdown("---")
        st.subheader("Diagnostics")
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

    # Common fields
    name = st.text_input("Name (unique)", value=("" if creating_new else src_cfg.name), key="edit_name")
    provider = st.selectbox(
        "Provider", list(SUPPORTED_PROVIDERS),
        index=(0 if creating_new else list(SUPPORTED_PROVIDERS).index(src_cfg.provider)),
        key="edit_provider"
    )
    model = st.text_input("Model", value=("gpt-4o-mini" if creating_new else src_cfg.model), key="edit_model")
    temperature = st.slider("Temperature", 0.0, 1.0, float(0.2 if creating_new else src_cfg.temperature), step=0.05, key="edit_temperature")
    description = st.text_area("Description", value=("" if creating_new else src_cfg.description), height=80, key="edit_description")

    # Conditional sections update immediately when radio changes
    selected_tools: List[str] = []
    selected_agents: List[str] = []

    if agent_type == "basic":
        role_prompt = st.text_area("Role/System Prompt", value=("" if creating_new else src_cfg.role_prompt), height=140, key="edit_role_prompt")
        context_enabled = st.checkbox("Enable context memory", value=(True if creating_new else src_cfg.context_enabled), key="edit_context_enabled")
    else:
        st.caption("Planner/Orchestrator do not use role prompts nor context memory.")

        # Namespaced keys so flipping entries doesn't collide
        ns = (name or (src_cfg.name if src_cfg else "new")).strip() or "new"

        # Tools (checkboxes)
        default_tools = ([] if creating_new else (src_cfg.allowed_tools or []))
        tool_names = [k for k in BUILTIN_TOOLS.keys() if k not in ALWAYS_ON_TOOLS]
        tcols = st.columns(3)
        for i, tname in enumerate(tool_names):
            with tcols[i % len(tcols)]:
                if st.checkbox(tname, value=(tname in default_tools), key=f"toolchk__{ns}__{tname}"):
                    selected_tools.append(tname)

        st.markdown("---")

        # Sub-agents (checkboxes) — exclude self
        all_agent_names = [c.name for c in st.session_state.configs]
        default_agents = ([] if creating_new else (src_cfg.allowed_agents or []))
        selectable_agents = [a for a in all_agent_names if a != (src_cfg.name if src_cfg else name)]
        acols = st.columns(3)
        for i, aname in enumerate(selectable_agents):
            with acols[i % len(acols)]:
                if st.checkbox(aname, value=(aname in default_agents), key=f"agentchk__{ns}__{aname}"):
                    selected_agents.append(aname)

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
                    role_prompt="", context_enabled=False,
                    allowed_tools=list(dict.fromkeys(selected_tools)),
                    allowed_agents=list(dict.fromkeys(selected_agents)),
                )

            # Replace config entry and re-instantiate (robust updates)
            if not creating_new:
                old = src_cfg.name
                st.session_state.configs = [c for c in st.session_state.configs if c.name != old]
                # migrate transcript on rename
                if old in st.session_state.transcripts:
                    if old != new_cfg.name:
                        st.session_state.transcripts[new_cfg.name] = st.session_state.transcripts.pop(old)
                else:
                    _ensure_transcript(new_cfg.name)
                # drop old instance
                st.session_state.instances.pop(old, None)
                if st.session_state.selected == old:
                    st.session_state.selected = new_cfg.name

            st.session_state.configs.append(new_cfg)
            save_agent_configs_file(st.session_state.configs)

            # Recreate instance NOW with tools + agent-wrappers
            inst = agent_factory(new_cfg)
            # Add wrappers now that we have cfg_map
            cfg_map = _get_config_map()
            if new_cfg.agent_type in ("planner", "orchestrator"):
                sync_toolbox_with_selection(inst, new_cfg.allowed_tools)
                sync_agent_wrappers(inst, new_cfg.allowed_agents, cfg_map)
            st.session_state.instances[new_cfg.name] = inst
            _ensure_transcript(new_cfg.name)

            st.session_state.cfg_choice = new_cfg.name
            st.toast("Saved.", icon="💾")
            _do_rerun()

    if delete_clicked and not creating_new:
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
