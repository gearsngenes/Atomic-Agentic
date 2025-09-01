# streamlit_app.py
# Two-tab Streamlit UI (Chat / Config) for Atomic-Agentic base Agents
# - Persists agent configs to ./agents.json
# - Imports your classes directly from modules.Agents and modules.LLMEngines
# - Whole responses (no streaming), markdown/code rendering
# - Chat history comes from the Agent instance itself (and can be cleared)
# - Config supports add/edit/duplicate/delete, unique names, import/export
# - Secrets come from environment variables (as your engines expect)

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import streamlit as st

# Ensure repo root is importable (so `modules` resolves)
sys.path.append(os.path.abspath("."))

# ---- Import your framework classes ----
from modules.Agents import Agent  # base Agent
from modules.LLMEngines import OpenAIEngine, GeminiEngine, MistralEngine

# ---- Constants ----
AGENTS_DB_PATH = "agents.json"
PROVIDER_CHOICES = ["openai", "gemini", "mistral"]

# Curated, safe defaults for Pro/Free tiers (+ Custom)
OPENAI_MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4.1-mini",
    "Custom…",
]
GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-pro",
    "Custom…",
]
MISTRAL_MODELS = [
    "open-mixtral-8x7b",
    "open-mistral-7b",
    "mistral-small-latest",
    "Custom…",
]

# ---- Persistence helpers ----
def load_agent_configs() -> List[Dict[str, Any]]:
    if not os.path.exists(AGENTS_DB_PATH):
        return []
    try:
        with open(AGENTS_DB_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception:
        return []

def save_agent_configs(configs: List[Dict[str, Any]]) -> None:
    with open(AGENTS_DB_PATH, "w", encoding="utf-8") as f:
        json.dump(configs, f, indent=2, ensure_ascii=False)

def name_exists(configs: List[Dict[str, Any]], name: str, exclude_name: Optional[str] = None) -> bool:
    for c in configs:
        if c.get("name") == name and name != exclude_name:
            return True
    return False

# ---- Engine / Agent instantiation helpers ----
def _construct_engine(cls, model: str, temperature: float):
    tried = []
    for kwargs in (
        {"model": model, "temperature": temperature},
        {"model_name": model, "temperature": temperature},
        {"model": model},
        {"model_name": model},
        {"config": {"model": model, "temperature": temperature}},
    ):
        try:
            return cls(**kwargs)
        except Exception as e:
            tried.append((kwargs, str(e)))
    try:
        return cls()
    except Exception as e:
        tried.append(("{}", str(e)))
        raise RuntimeError(
            f"Could not construct {cls.__name__} with any known signature. Attempts:\n" +
            "\n".join([f"- {kw}: {err}" for kw, err in tried])
        )

def create_engine(provider: str, model: str, temperature: float):
    provider = provider.lower()
    if provider == "openai":
        return _construct_engine(OpenAIEngine, model, temperature)
    if provider == "gemini":
        return _construct_engine(GeminiEngine, model, temperature)
    if provider == "mistral":
        return _construct_engine(MistralEngine, model, temperature)
    raise ValueError(f"Unknown provider '{provider}'")

def create_agent_instance(cfg: Dict[str, Any]) -> Agent:
    engine = create_engine(cfg["provider"], cfg["model"], float(cfg["temperature"]))
    name = cfg["name"]
    description = cfg.get("description", "")
    role_prompt = cfg.get("role_prompt", "")
    context_enabled = bool(cfg.get("context_enabled", False))

    tried = []
    for kwargs in (
        dict(name=name, description=description, llm_engine=engine,
             role_prompt=role_prompt, context_enabled=context_enabled),
        dict(name=name, description=description, llm_engine=engine, role_prompt=role_prompt),
        dict(name=name, description=description, llm_engine=engine, context_enabled=context_enabled),
        dict(name=name, description=description, llm_engine=engine),
        dict(name=name, description=description, engine=engine, role_prompt=role_prompt,
             context_enabled=context_enabled),
        dict(name=name, description=description, llm_engine=engine, system_prompt=role_prompt,
             context_enabled=context_enabled),
    ):
        try:
            return Agent(**kwargs)
        except Exception as e:
            tried.append((kwargs, str(e)))
    raise RuntimeError(
        "Could not construct Agent with the usual signatures. Attempts:\n" +
        "\n".join([f"- {kw}: {err}" for kw, err in tried])
    )

def reset_agent_history(agent: Agent) -> None:
    try:
        if hasattr(agent, "clear_history") and callable(agent.clear_history):
            agent.clear_history()
            return
    except Exception:
        pass
    try:
        if hasattr(agent, "reset") and callable(agent.reset):
            agent.reset()
            return
    except Exception:
        pass
    for attr in ("history", "_history", "messages", "_messages", "_previous_steps"):
        if hasattr(agent, attr):
            try:
                obj = getattr(agent, attr)
                if isinstance(obj, list):
                    obj.clear()
            except Exception:
                pass

def agent_respond(agent: Agent, user_text: str) -> str:
    for m in ("generate_reply", "invoke", "step", "respond", "chat"):
        if hasattr(agent, m) and callable(getattr(agent, m)):
            try:
                return getattr(agent, m)(user_text)
            except Exception as e:
                raise RuntimeError(f"{agent.__class__.__name__}.{m} failed: {e}")
    raise RuntimeError("No known response method found on Agent (tried: generate_reply/invoke/step/respond/chat).")

def extract_history(agent: Agent) -> List[Dict[str, str]]:
    candidates = []
    for attr in ("history", "_history", "messages", "_messages"):
        if hasattr(agent, attr):
            try:
                v = getattr(agent, attr)
                if isinstance(v, list) and len(v) > 0:
                    candidates.append(v)
            except Exception:
                pass
    if not candidates:
        return []
    hist = max(candidates, key=len)
    norm: List[Dict[str, str]] = []
    for item in hist:
        if isinstance(item, dict):
            role = item.get("role") or item.get("speaker") or item.get("name")
            content = item.get("content") or item.get("text") or item.get("message") or ""
        elif isinstance(item, (tuple, list)) and len(item) >= 2:
            role, content = str(item[0]).lower(), str(item[1])
        else:
            role, content = "assistant", str(item)
        role = (role or "assistant").lower()
        role = role if role in ("user", "assistant", "system") else "assistant"
        norm.append({"role": role, "content": content})
    return norm

# ---- Session bootstrap ----
if "agent_configs" not in st.session_state:
    st.session_state.agent_configs = load_agent_configs()
if "agent_instances" not in st.session_state:
    st.session_state.agent_instances: Dict[str, Agent] = {}
if "selected_agent" not in st.session_state:
    st.session_state.selected_agent = None  # agent name

# ---- UI: Title ----
st.set_page_config(page_title="Atomic-Agentic Chat", page_icon="🤖", layout="wide")
st.title("Atomic-Agentic — Agents & Chat")

tab_chat, tab_config = st.tabs(["💬 Chat", "⚙️ Config"])

# =========================
# Chat Tab
# =========================
with tab_chat:
    configs = st.session_state.agent_configs
    names = [c["name"] for c in configs]

    left, right = st.columns([2, 1])
    with left:
        st.subheader("Chat")
        if not names:
            st.info("No agents yet. Create one in the **Config** tab.")
        else:
            # Selection
            current_index = 0
            if st.session_state.selected_agent in names:
                current_index = names.index(st.session_state.selected_agent)
            current_name = st.selectbox("Choose agent", names, index=current_index, key="selected_agent")

            # ---------- SAFER INSTANTIATION ----------
            if current_name:
                # find config safely (avoid StopIteration)
                cfg = next((c for c in configs if c.get("name") == current_name), None)
                if cfg:
                    if current_name not in st.session_state.agent_instances:
                        try:
                            st.session_state.agent_instances[current_name] = create_agent_instance(cfg)
                        except Exception as e:
                            safe_name = current_name or "(unnamed)"
                            st.toast(f"Could not instantiate agent '{safe_name}': {e}", icon="❌")
                            # Do not stop the app; allow user to fix config and continue
                else:
                    # No matching config yet (transient after save); skip instantiation this frame
                    pass
            # ----------------------------------------

            agent = st.session_state.agent_instances.get(current_name)
            st.markdown("###### Conversation")

            if agent:
                history = extract_history(agent)
                if not history:
                    st.caption("No messages yet.")
                # Right-align user, left-align assistant, with avatars at edges
                for msg in history:
                    role = msg.get("role", "assistant")
                    content = msg.get("content", "")
                    is_user = role == "user"
                    if is_user:
                        spacer, msgcol = st.columns([0.15, 0.85])
                        with msgcol:
                            with st.chat_message("user", avatar="🧑"):
                                st.markdown(content)
                    else:
                        msgcol, spacer = st.columns([0.85, 0.15])
                        with msgcol:
                            with st.chat_message("assistant", avatar="🤖"):
                                st.markdown(content)
            else:
                st.caption("Select a valid agent to start chatting.")

            st.divider()

            # --- Input row: message + Send / Clear (FORM; Option A) ---
            input_key = f"user_text_{current_name or 'active'}"
            with st.form(f"chat-send-{current_name or 'active'}", clear_on_submit=True):
                user_text = st.text_area(
                    "Your message",
                    placeholder="Type a message…",
                    height=100,
                    key=input_key,
                )
                send_col, clear_col, _ = st.columns([0.15, 0.15, 0.7])
                send_clicked = send_col.form_submit_button("Send", use_container_width=True)
                clear_clicked = clear_col.form_submit_button("Clear chat", use_container_width=True)

            if clear_clicked and agent:
                try:
                    reset_agent_history(agent)
                    st.session_state.pop(input_key, None)
                    st.toast("Cleared chat (agent memory reset).", icon="🧹")
                    time.sleep(0.2)
                    st.rerun()
                except Exception as e:
                    st.toast(f"Failed to clear: {e}", icon="❌")

            if send_clicked and user_text and user_text.strip():
                if not agent:
                    st.toast("Please select a valid agent first.", icon="⚠️")
                else:
                    try:
                        _ = agent_respond(agent, user_text.strip())
                        st.rerun()
                    except Exception as e:
                        st.toast(f"Message failed: {e}", icon="❌")

    with right:
        st.subheader("Agent Info")
        if names and st.session_state.selected_agent in st.session_state.agent_instances:
            cfg = next((c for c in configs if c["name"] == st.session_state.selected_agent), None)
            if cfg:
                st.write("**Name:**", cfg["name"])
                st.write("**Provider:**", cfg["provider"])
                st.write("**Model:**", cfg["model"])
                st.write("**Temperature:**", cfg["temperature"])
                st.write("**Context Enabled:**", "Yes" if cfg.get("context_enabled") else "No")
                st.write("**Description:**")
                st.caption(cfg.get("description", ""))

# =========================
# Config Tab
# =========================
with tab_config:
    st.subheader("Manage Agents")

    top_l, top_r = st.columns([0.7, 0.3])
    with top_l:
        existing_names = [c["name"] for c in st.session_state.agent_configs]
        selected_to_edit = st.selectbox("Select existing agent to edit", ["(New)"] + existing_names, index=0)

    with top_r:
        # Export / Import
        if st.button("Export JSON", use_container_width=True):
            data = json.dumps(st.session_state.agent_configs, indent=2, ensure_ascii=False).encode("utf-8")
            st.download_button(
                "Download agents.json",
                data,
                file_name="agents.json",
                mime="application/json",
                use_container_width=True,
            )
        uploaded = st.file_uploader("Import JSON", type=["json"], label_visibility="collapsed")
        if uploaded is not None:
            try:
                imported = json.loads(uploaded.read().decode("utf-8"))
                assert isinstance(imported, list)
                added, skipped = 0, 0
                cur = st.session_state.agent_configs
                for item in imported:
                    if not isinstance(item, dict):
                        continue
                    required = ["name", "provider", "model", "temperature", "description", "role_prompt"]
                    if not all(k in item for k in required):
                        continue
                    if name_exists(cur, item["name"]):
                        skipped += 1
                    else:
                        item.setdefault("context_enabled", False)
                        cur.append(item)
                        added += 1
                save_agent_configs(cur)
                st.toast(f"Imported: {added} added, {skipped} skipped (duplicate names).", icon="📥")
            except Exception as e:
                st.toast(f"Import failed: {e}", icon="❌")

    if selected_to_edit == "(New)":
        working = {
            "name": "",
            "provider": PROVIDER_CHOICES[0],
            "model": OPENAI_MODELS[0],
            "temperature": 0.2,
            "description": "",
            "role_prompt": "",
            "context_enabled": True,
        }
        is_new = True
    else:
        working = next(c for c in st.session_state.agent_configs if c["name"] == selected_to_edit)
        is_new = False

    with st.form("agent-form"):
        name = st.text_input("Name", value=working["name"])
        provider = st.selectbox(
            "Provider",
            PROVIDER_CHOICES,
            index=PROVIDER_CHOICES.index(working["provider"]) if working["provider"] in PROVIDER_CHOICES else 0,
        )
        model_choices = OPENAI_MODELS if provider == "openai" else GEMINI_MODELS if provider == "gemini" else MISTRAL_MODELS
        model_sel = st.selectbox(
            "Model",
            model_choices,
            index=(model_choices.index(working["model"]) if working["model"] in model_choices else len(model_choices) - 1),
        )
        if model_sel == "Custom…":
            model = st.text_input("Custom model id", value=working["model"] if working["model"] not in model_choices else "")
        else:
            model = model_sel

        temperature = st.slider("Temperature", 0.0, 1.0, float(working.get("temperature", 0.2)), step=0.05)
        description = st.text_area("Agent description (metadata)", value=working.get("description", ""), height=80)
        role_prompt = st.text_area("Agent role/system prompt", value=working.get("role_prompt", ""), height=140)
        context_enabled = st.checkbox("Context enabled (remember conversation)", value=bool(working.get("context_enabled", True)))

        colA, colB, colC, colD = st.columns(4)
        save_btn = colA.form_submit_button("Save", use_container_width=True)
        dup_btn = (not is_new) and colB.form_submit_button("Duplicate", use_container_width=True)
        del_btn = (not is_new) and colC.form_submit_button("Delete", use_container_width=True)

    if save_btn:
        if not name.strip():
            st.toast("Name is required.", icon="⚠️")
        elif not provider or not model:
            st.toast("Provider and model are required.", icon="⚠️")
        elif name_exists(st.session_state.agent_configs, name.strip(), exclude_name=None if is_new else working["name"]):
            st.toast("Agent name must be unique.", icon="⚠️")
        else:
            new_cfg = {
                "name": name.strip(),
                "provider": provider,
                "model": model.strip(),
                "temperature": float(temperature),
                "description": description,
                "role_prompt": role_prompt,
                "context_enabled": bool(context_enabled),
            }
            cfgs = st.session_state.agent_configs
            if is_new:
                cfgs.append(new_cfg)
            else:
                for i, c in enumerate(cfgs):
                    if c["name"] == working["name"]:
                        cfgs[i] = new_cfg
                        st.session_state.agent_instances.pop(working["name"], None)
                        break
            save_agent_configs(cfgs)
            st.toast("Saved.", icon="💾")
            time.sleep(0.2)
            st.rerun()

    if dup_btn:
        base = working["name"]
        k = 2
        new_name = f"{base} ({k})"
        while name_exists(st.session_state.agent_configs, new_name):
            k += 1
            new_name = f"{base} ({k})"
        copy_cfg = dict(working)
        copy_cfg["name"] = new_name
        st.session_state.agent_configs.append(copy_cfg)
        save_agent_configs(st.session_state.agent_configs)
        st.toast(f"Duplicated as '{new_name}'.", icon="🧬")
        time.sleep(0.2)
        st.rerun()

    if del_btn:
        cfgs = [c for c in st.session_state.agent_configs if c["name"] != working["name"]]
        st.session_state.agent_configs = cfgs
        save_agent_configs(cfgs)
        st.session_state.agent_instances.pop(working["name"], None)
        st.toast(f"Deleted '{working['name']}'.", icon="🗑️")
        time.sleep(0.2)
        st.rerun()
