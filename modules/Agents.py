from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.WARNING)  # Default level; override in examples as needed

load_dotenv()

# internal imports
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
                 role_prompt: str = Prompts.DEFAULT_PROMPT, context_enabled: bool = False):
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