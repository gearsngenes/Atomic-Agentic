from dotenv import load_dotenv
import logging

"""
Agents
======

Minimal, stateful agents that wrap an LLM engine and (optionally) maintain
conversation history and a list of attached local file paths.

Design
------
- Agents are **stateful**: they can remember prior turns (if `context_enabled=True`)
  and they keep a simple list of attached file paths (strings).
- LLM engines are **stateless**: the engine decides how to handle any file paths
  passed to it for a single request and must perform best-effort cleanup.

Classes
-------
Agent
    General LLM-backed agent that builds messages and delegates to an `LLMEngine`.
HumanAgent
    Thin adapter that prompts a human via `input()` and (optionally) stores history.

Notes
-----
This file improves documentation and readability only. No behavioral changes were made.
"""

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
    Minimal, stateful Agent.

    Responsibilities
    ----------------
    - Optionally keeps chat history when `context_enabled=True`.
    - Tracks a list of attached file paths (strings only).
    - Delegates provider-specific behavior to the injected `LLMEngine`.

    Lifecycle
    ---------
    attach(path)
        Add a local file path to this agent (no provider calls here).
    detach(path)
        Remove a previously attached local file path.
    invoke(prompt)
        Build the message list (system + optional history + user) and call
        `llm_engine.invoke(messages, file_paths=[...])`.
    clear_memory()
        Clear only the conversation history (attachments remain).

    Parameters
    ----------
    name : str
        Human-readable name for this agent.
    description : str
        Short description of the agent’s purpose.
    llm_engine : LLMEngine
        Stateless engine that performs the model call.
    role_prompt : str
        System prompt included as the first message (if non-empty).
    context_enabled : bool
        If True, prior turns are persisted and sent on subsequent invocations.
    """

    def __init__(self, name, description, llm_engine: LLMEngine,
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
        """str: Agent name (read/write)."""
        return self._name

    @property
    def description(self):
        """str: Brief description of what this agent does (read/write)."""
        return self._description

    @property
    def role_prompt(self):
        """str: System prompt prepended to every invocation (read/write)."""
        return self._role_prompt

    @property
    def context_enabled(self):
        """bool: Whether prior turns are remembered and sent on subsequent calls (read/write)."""
        return self._context_enabled

    @property
    def llm_engine(self):
        """LLMEngine: The stateless engine used to perform the model call (read/write)."""
        return self._llm_engine

    @property
    def history(self):
        """list[dict]: A **copy** of the stored conversation turns (role/content)."""
        # return a copy for safety
        return self._history.copy()

    @property
    def file_paths(self):
        """list[str]: A **copy** of the attached local file paths."""
        # return a copy for safety
        return self._file_paths.copy()

    @name.setter
    def name(self, value: str):
        """Set the agent name."""
        self._name = value

    @description.setter
    def description(self, value: str):
        """Set the agent description."""
        self._description = value

    @role_prompt.setter
    def role_prompt(self, value: str):
        """Set the system prompt (empty string disables system message)."""
        self._role_prompt = value or ""

    @context_enabled.setter
    def context_enabled(self, value: bool):
        """Enable/disable conversation memory for this agent."""
        self._context_enabled = bool(value)

    @llm_engine.setter
    def llm_engine(self, engine):
        """Replace the underlying LLM engine."""
        self._llm_engine = engine

    # ── Memory controls ─────────────────────────────────────────
    def clear_memory(self):
        """Clear conversation history; file attachments are left as-is."""
        self._history = []

    # ── File path management (no provider calls here) ───────────
    def attach(self, path: str) -> bool:
        """
        Attach a local file path to this agent.

        Parameters
        ----------
        path : str
            Absolute or relative path to a local file.

        Returns
        -------
        bool
            True if the path was added; False if it was already present.
        """
        if path in self._file_paths:
            return False
        self._file_paths.append(path)
        return True

    def detach(self, path: str) -> bool:
        """
        Detach a previously attached local file path.

        Parameters
        ----------
        path : str
            The exact path to remove.

        Returns
        -------
        bool
            True if the path was removed; False if it was not present.
        """
        try:
            self._file_paths.remove(path)
            return True
        except ValueError:
            return False

    # ── Inference ───────────────────────────────────────────────
    def invoke(self, prompt: str) -> str:
        """
        Invoke the underlying LLM engine with the current context and attachments.

        The method constructs the message list in this order:
        1) Optional system message (when `role_prompt` is non-empty)
        2) Prior turns, if `context_enabled=True`
        3) Current user message containing `prompt`

        Then it delegates to `llm_engine.invoke(messages, file_paths=self.file_paths)` and,
        if `context_enabled=True`, appends both the user turn and the assistant
        response to the internal history.

        Parameters
        ----------
        prompt : str
            The user input for this turn.

        Returns
        -------
        str
            The assistant’s response text (stripped).
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
    """
    Agent variant that proxies the response to a human via `input()`.

    Behavior
    --------
    - `invoke(prompt)` prints the prompt and returns the string the human types.
    - If `context_enabled=True`, it appends the user/assistant turns to history.
    - File attachment methods are **not** supported and raise `NotImplementedError`.

    Parameters
    ----------
    name : str
        Human-readable name for display.
    description : str
        Description (also reused as the role prompt for this agent).
    context_enabled : bool
        If True, prior turns are persisted and sent on subsequent invocations.
    """
    def __init__(self, name, description, context_enabled:bool = False):
        self._context_enabled = context_enabled
        self._name = name
        self._description = description
        self._llm_engine = None
        self._role_prompt = description

    def attach(self, file_path: str):
        """HumanAgent does not support file attachments."""
        raise NotImplementedError("HumanAgent does not support file attachments.")

    def detach(self, file_path: str):
        """HumanAgent does not support file attachments."""
        raise NotImplementedError("HumanAgent does not support file attachments.")

    def invoke(self, prompt:str):
        """
        Prompt a human for input and return the raw string.

        Parameters
        ----------
        prompt : str
            The message to display to the human user.

        Returns
        -------
        str
            The human's response collected via `input()`.
        """
        response = input(f"{prompt}\n{self.name}'s Response: ")
        if self._context_enabled:
            self._history.append({"role": "user", "content": prompt})
            self._history.append({"role": "assistant", "content": response})
        return response