# atomic_ui.py — global TOOL_CALLS + toolbox rebinding + agents.json persistence
# - One global TOOL_CALLS list (order-preserving).
# - Tool bodies append {'name': ..., 'args': {...}} to TOOL_CALLS.
# - Before invoke(): refresh toolbox entries to THIS RUN’s function objects.
# - After invoke(): stitch tool-calls into assistant transcript, append final reply, clear TOOL_CALLS.
# - Planner forced synchronous (is_async=False).
# - Agent configs persist to agents.json across app reloads.

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import streamlit as st

# ---- Atomic-Agentic modules (your project must expose these on PYTHONPATH) ----
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

SUPPORTED_PROVIDERS = ("openai", "gemini", "mistral")

def _cfg_from_dict(d: Dict[str, Any]) -> AgentCfg:
    # Defensive: ignore unknown keys; fill defaults for missing keys.
    return AgentCfg(
        name=d.get("name", "").strip() or "unnamed",
        agent_type=(d.get("agent_type") or "basic").lower(),
        provider=(d.get("provider") or "openai").lower(),
        model=d.get("model") or "gpt-4o-mini",
        temperature=float(d.get("temperature", 0.2)),
        description=d.get("description") or "",
        role_prompt=d.get("role_prompt") or "",
        context_enabled=bool(d.get("context_enabled", True)),
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
# Tools append here at call time; UI drains after invoke().
TOOL_CALLS: List[Dict[str, Any]] = []  # each: {"name": str, "args": dict}

def record_tool_call(name: str, **kwargs) -> None:
    """Append a single tool invocation record in call order."""
    TOOL_CALLS.append({"name": name, "args": dict(kwargs)})

def drain_tool_calls() -> List[Dict[str, Any]]:
    """Return and clear all recorded tool calls."""
    global TOOL_CALLS
    calls = TOOL_CALLS
    TOOL_CALLS = []
    return calls

# ============================== Transcript helpers ==============================
def _ensure_transcript(name: str) -> None:
    st.session_state.transcripts.setdefault(name, [])

def _append_chat_line(agent_name: str, role: str, content_md: str) -> None:
    _ensure_transcript(agent_name)
    st.session_state.transcripts[agent_name].append({"role": role, "content": content_md})

def _clear_transcript(name: str) -> None:
    st.session_state.transcripts[name] = []

def _md_tool_block(name: str, args: Dict[str, Any]) -> str:
    return f"**Tool:** {name}\n\n**Args**:\n```json\n{json.dumps(args, indent=2, ensure_ascii=False)}\n```"

# ============================== Built-in tools (explicit, no Streamlit) ==============================

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
pow_.__name__ = "pow"
radians_.__name__ = "radians"
degrees_.__name__ = "degrees"

BUILTIN_TOOLS: Dict[str, Tuple[Callable[..., Any], str]] = {
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
}

def register_builtin_tools(tool_agent: Any) -> None:
    """Initial registration (first instantiation). Duplicate keys are ignored."""
    for _, (fn, desc) in BUILTIN_TOOLS.items():
        try:
            tool_agent.register(fn, desc)
        except Exception:
            pass  # okay if already present

def refresh_builtin_tools(tool_agent: Any) -> None:
    """
    Overwrite existing '__dev_tools__.*' entries so they point to THIS RUN's functions.
    Fixes stale callables captured on previous Streamlit reruns.
    """
    tb = getattr(tool_agent, "_toolbox", None)
    bucket = tb.get("__dev_tools__") if isinstance(tb, dict) else None

    if not isinstance(bucket, dict):
        # No dev bucket yet → attempt full registration.
        register_builtin_tools(tool_agent)
        tb = getattr(tool_agent, "_toolbox", None)
        bucket = tb.get("__dev_tools__") if isinstance(tb, dict) else None

    if not isinstance(bucket, dict):
        return  # nothing else we can do

    for name, (fn, desc) in BUILTIN_TOOLS.items():
        key = f"__dev_tools__.{name}"
        cell = bucket.get(key)
        if isinstance(cell, dict) and "callable" in cell:
            cell["callable"] = fn
        elif cell is not None and callable(cell):
            bucket[key] = fn
        else:
            # missing → register fresh
            try:
                tool_agent.register(fn, desc)
            except Exception:
                pass

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
        # Force synchronous execution so tool bodies run inline.
        inst = PlannerAgent(
            name=cfg.name,
            description=cfg.description,
            llm_engine=engine,
            is_async=False,
            allow_agentic=True,
        )
        register_builtin_tools(inst)
        return inst

    if cfg.agent_type == "orchestrator":
        inst = OrchestratorAgent(
            name=cfg.name,
            description=cfg.description,
            llm_engine=engine,
            allow_agentic=True,
        )
        register_builtin_tools(inst)
        return inst

    # basic
    return Agent(
        name=cfg.name,
        description=cfg.description,
        llm_engine=engine,
        role_prompt=cfg.role_prompt,
        context_enabled=cfg.context_enabled,
    )

# ============================== Session bootstrap ==============================
def _bootstrap_defaults():
    if "configs" not in st.session_state:
        # Load from agents.json if present; else create one default agent and save.
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

    # Ensure transcripts exist for loaded configs
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
                st.rerun()

            agent = _get_instance(st.session_state.selected)

            # Render transcript
            for msg in st.session_state.transcripts.get(agent.name, []):
                avatar = "🙂" if msg["role"] == "user" else "🤖"
                with st.chat_message(msg["role"], avatar=avatar):
                    st.markdown(msg["content"])

            # Input row
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
                    st.rerun()

            if send_clicked:
                txt = (user_text or "").strip()
                if not txt:
                    st.toast("Please enter a message.", icon="⚠️")
                else:
                    try:
                        _append_chat_line(agent.name, "user", txt)

                        # New run: clear any stale tool-calls
                        TOOL_CALLS.clear()

                        # Ensure toolbox uses THIS RUN’s tool functions (avoid stale callables)
                        if hasattr(agent, "_toolbox"):
                            refresh_builtin_tools(agent)

                        # Execute synchronously; tools append into TOOL_CALLS
                        reply = _agent_send(agent, txt)

                        # Stitch tool calls into chat (in order), then final reply
                        for call in TOOL_CALLS:
                            _append_chat_line(agent.name, "assistant", _md_tool_block(call["name"], call["args"]))
                        _append_chat_line(agent.name, "assistant", reply)
                        TOOL_CALLS.clear()

                        st.session_state.last_error = ""
                    except Exception as e:
                        _append_chat_line(agent.name, "assistant", f"❌ Error: {e}")
                        st.session_state.last_error = str(e)
                    finally:
                        st.session_state.input_nonce += 1
                        st.rerun()

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

        st.markdown("---")
        st.subheader("Diagnostics")
        if st.session_state.last_error:
            st.error(st.session_state.last_error)
        else:
            st.caption("No errors.")

with tab_config:
    st.subheader("Manage Agents")

    existing = [c.name for c in st.session_state.configs]
    choice = st.selectbox("Select", ["(New)"] + existing, index=0, key="cfg_picker")
    creating_new = choice == "(New)"

    if creating_new:
        working = AgentCfg(name="")
    else:
        src = _get_config_map()[choice]
        working = AgentCfg(**asdict(src))

    with st.form("cfg_form"):
        agent_type = st.selectbox(
            "Category",
            options=["basic", "planner", "orchestrator"],
            index={"basic": 0, "planner": 1, "orchestrator": 2}.get(working.agent_type, 0),
        )

        name = st.text_input("Name (unique)", value=working.name)
        provider = st.selectbox("Provider", list(SUPPORTED_PROVIDERS),
                                index=(list(SUPPORTED_PROVIDERS).index(working.provider) if working.provider in SUPPORTED_PROVIDERS else 0))
        model = st.text_input("Model", value=working.model)
        temperature = st.slider("Temperature", 0.0, 1.0, float(working.temperature), step=0.05)
        description = st.text_area("Description", value=working.description, height=80)

        role_prompt = working.role_prompt
        context_enabled = working.context_enabled

        if agent_type == "basic":
            role_prompt = st.text_area("Role/System Prompt", value=working.role_prompt, height=140)
            context_enabled = st.checkbox("Enable context memory", value=working.context_enabled)
        else:
            st.caption("Planner/Orchestrator do not use role prompts nor context memory.")

        c1, c2, c3 = st.columns([1, 1, 1])
        save_clicked = c1.form_submit_button("Save")
        delete_clicked = (False if creating_new else c2.form_submit_button("Delete"))

        if save_clicked:
            nm = name.strip()
            if not nm:
                st.toast("Name is required.", icon="⚠️")
            elif provider not in SUPPORTED_PROVIDERS:
                st.toast(f"Unsupported provider: {provider}", icon="⚠️")
            else:
                names = [c.name for c in st.session_state.configs]
                is_rename = (not creating_new) and (nm != working.name)
                if (creating_new and nm in names) or (is_rename and nm in names):
                    st.toast("Agent name must be unique.", icon="⚠️")
                else:
                    new_cfg = AgentCfg(
                        name=nm,
                        agent_type=agent_type,
                        provider=provider,
                        model=model.strip(),
                        temperature=float(temperature),
                        description=description.strip(),
                        role_prompt=(role_prompt if agent_type == "basic" else ""),
                        context_enabled=(bool(context_enabled) if agent_type == "basic" else False),
                    )

                    if creating_new:
                        st.session_state.configs.append(new_cfg)
                        _ensure_transcript(new_cfg.name)
                        st.session_state.selected = new_cfg.name
                    else:
                        old = working.name
                        for i, c in enumerate(st.session_state.configs):
                            if c.name == old:
                                st.session_state.configs[i] = new_cfg
                                break
                        if old in st.session_state.transcripts:
                            st.session_state.transcripts[new_cfg.name] = st.session_state.transcripts.pop(old)
                        else:
                            _ensure_transcript(new_cfg.name)
                        st.session_state.instances.pop(old, None)
                        if st.session_state.selected == old:
                            st.session_state.selected = new_cfg.name

                    # Persist to agents.json
                    save_agent_configs_file(st.session_state.configs)
                    st.toast("Saved.", icon="💾")
                    st.session_state.input_nonce += 1
                    st.rerun()

        if delete_clicked and not creating_new:
            victim = working.name
            st.session_state.configs = [c for c in st.session_state.configs if c.name != victim]
            st.session_state.instances.pop(victim, None)
            st.session_state.transcripts.pop(victim, None)
            if st.session_state.configs:
                st.session_state.selected = st.session_state.configs[0].name
                _ensure_transcript(st.session_state.selected)
            else:
                # No agents left — create a default and save
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
                    )
                ]
                st.session_state.selected = st.session_state.configs[0].name
                _ensure_transcript(st.session_state.selected)
            # Persist after delete
            save_agent_configs_file(st.session_state.configs)
            st.toast(f"Deleted '{victim}'.", icon="🗑️")
            st.session_state.input_nonce += 1
            st.rerun()
