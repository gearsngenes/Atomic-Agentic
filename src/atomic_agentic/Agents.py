"""
Agents
======

An **Agent** is a small, stateful unit of software that talks to a Large Language
Model (LLM) using a chosen persona (a system role-prompt). It accepts a single
**input mapping** (a dict-like object), converts that mapping into a plain
**prompt string** using a **pre-invoke Tool**, and then asks the LLM for a response.

Because LLMs are probabilistic, Agent outputs are **non-deterministic**.

Why schema-first (dict inputs)?
-------------------------------
- A single input shape (`Mapping[str, Any]`) is predictable to call and compose.
- Pre-invoke Tools can adapt *richer schemas and types* (e.g., `{topic, style, audience}`) into
  the final prompt **without** changing the Agent’s method signature.
- Workflows/planners can route dicts between Agents and Tools consistently.

Call lifecycle (what happens on `agent.invoke(inputs)`):
-------------------------------------------------------
1) **Input validation**: `inputs` must be a `Mapping`; otherwise we raise.
2) **Pre-invoke Tool** turns the mapping into a **prompt string**.
   - The default Tool is *strict* and only accepts `{"prompt": str}`.
   - Provide your own Tool to accept richer keys.
3) **Message assembly**:
   - Optional system persona (`role_prompt`) as the first message.
   - If `context_enabled`, we include the **last N turns** from history, where each
     *turn* is a user->assistant pair. Stored history itself is **never trimmed**.
   - Finally, we append the **user prompt** for the current call.
4) **Engine call**: we invoke the configured `LLMEngine` with the messages and, if any,
   local **attachments** (file paths). A small compatibility shim allows engines that
   expect `(messages, file_paths=...)`, `(messages, files=...)`, or just `(messages)`.
5) **Result**: we expect the engine to return a **str**. If `context_enabled` is on,
   we append this turn (user prompt, assistant reply) to the stored history and return
   the text (non-deterministic by nature).

History policy
--------------
- **Stored history is never trimmed.**
- `history_window` is a **send window measured in turns** (pairs of user/assistant
  messages) that limits what is **sent** to the engine, not what is stored.
- Set `history_window=0` to send no previous turns.
- Set `context_enabled=False` to both **avoid sending** previous turns and **avoid
  recording** new turns.

Attachments
-----------
- Use `attach(path)` and `detach(path)` to manage a simple list of file paths.
- The Agent does **not** check for file existence by default.
- Engines decide how (or whether) to use attachments.

Common mistakes
---------------
- Passing a raw string to `invoke` → **error**. Always pass a mapping, e.g.
  `{"prompt": "Summarize this."}`.
- Passing extra keys with the default pre-invoke Tool → **error**. Install your own
  Tool that accepts your keys and returns a prompt string.
- Expecting deterministic outputs from the Agent/LLM → results will vary.

Quickstart
----------
>>> from LLMEngines import OpenAIEngine        # example engine (adjust import path as needed)
>>> from Tools import Tool
>>> agent = Agent(
...     name="Writer",
...     description="Helpful writing assistant",
...     llm_engine=OpenAIEngine(model="gpt-4o-mini"),
...     role_prompt="You are a concise writing assistant.",
... )
>>> agent.invoke({"prompt": "Give me a three-point outline for a blog on testability."})
'1) ...\\n2) ...\\n3) ...'

Custom pre-invoke Tool (accept richer keys):
>>> def to_prompt(topic: str, style: str) -> str:
...     return f"Write about '{topic}' in a {style} style."
...
>>> prompt_tool = Tool(func=to_prompt, name="to_prompt", description="topic/style -> prompt")
>>> agent.pre_invoke = prompt_tool
>>> agent.invoke({"topic": "unit testing", "style": "pragmatic"})
'...'

Thread-safety
-------------
This class is **not thread-safe**. It mutates internal history and attachments.
Create one Agent per concurrent lane, or protect it with external synchronization.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Callable
import logging
from collections import OrderedDict

# Local imports (adjust the module paths if your project structure differs)
from .LLMEngines import LLMEngine
from .Tools import Tool, ToolInvocationError


__all__ = [
    "Agent",
    "AgentError",
    "AgentInvocationError",
]


logger = logging.getLogger(__name__)


# ==== Errors ==================================================================


class AgentError(RuntimeError):
    """Base class for Agent-related errors."""


class AgentInvocationError(AgentError):
    """Raised when an Agent fails to prepare or process an invocation."""


# ==== Agent ===================================================================


class Agent:
    """
    Schema-driven LLM Agent.

    An Agent is a stateful software unit that points to an LLM and carries a persona
    (system role-prompt). It accepts a single **input mapping** and uses a **pre-invoke
    Tool** to convert that mapping into a **prompt string** before invoking the engine.

    Core behavior:
    - `invoke(inputs: Mapping[str, Any]) -> str`
      1) Validate inputs (must be a Mapping).
      2) `pre_invoke.invoke(inputs) -> str` (default strict: requires `{"prompt": str}`).
      3) Build messages: [system?] + [last N turns] + [user(prompt)].
      4) Call `llm_engine.invoke(...)` with a small compatibility shim for attachments.
      5) Expect a `str` result; record the turn if `context_enabled`; return the text.

    Thread-safety:
    - Not thread-safe. Use separate instances for concurrent invokes or add external locks.

    Parameters
    ----------
    name : str
        Logical name for this agent (read-only).
    description : str
        Short, human-readable description (read-only).
    llm_engine : LLMEngine
        Engine used to perform the model call. Must be an instance of `LLMEngine`.
    role_prompt : Optional[str], default None
        Optional system persona. If None or empty, no system message is sent.
    context_enabled : bool, default True
        If True, the agent includes and logs conversation context (turns).
        If False, the agent sends no prior turns and does not log new ones.
    history_window : int, default 50
        Send-window measured in **turns** (user+assistant pairs). `0` sends no turns.
        Stored history is never trimmed.
    pre_invoke : Optional[Tool], default None
        Tool that converts the input mapping to a `str` prompt. If None, a strict
        identity Tool is created that accepts exactly `{"prompt": str}`.

    Properties (selected)
    ---------------------
    name : str (read-only)
    description : str (read-only)
    role_prompt : Optional[str] (read-write)
    llm_engine : LLMEngine (read-write, type-enforced)
    context_enabled : bool (read-write)
    history_window : int (read-write; see semantics above)
    history : List[Dict[str, str]] (read-only view)
    attachments : List[str] (read-only view)
    pre_invoke : Tool (read-write)
    """

    # ----------------------------- construction ------------------------------

    def __init__(
        self,
        name: str,
        description: str,
        llm_engine: LLMEngine,
        role_prompt: Optional[str] = None,
        context_enabled: bool = True,
        *,
        pre_invoke: Optional[Tool|Callable] = None,
        history_window: int = 50,
    ) -> None:
        if not isinstance(name, str) or not name:
            raise AgentError("Agent 'name' must be a non-empty string.")
        if not isinstance(description, str):
            raise AgentError("Agent 'description' must be a string.")
        if not isinstance(llm_engine, (LLMEngine, type(None))):
            raise AgentError("llm_engine must be an instance of LLMEngine.")

        self._name = name
        self._description = description
        self._llm_engine: LLMEngine = llm_engine

        self._role_prompt: Optional[str] = role_prompt or None

        if not isinstance(context_enabled, bool):
            raise AgentError("context_enabled must be a bool.")
        self._context_enabled: bool = context_enabled

        # history_window: strict int semantics (>= 0). No 'None' (send-all) mode.
        if not isinstance(history_window, int) or history_window < 0:
            raise AgentError("history_window must be an int >= 0.")
        self._history_window: int = history_window

        # Stored message history (flat list of role/content dicts).
        # We never trim storage; we only limit what we *send* to engine.
        self._history: List[Dict[str, str]] = []

        # Simple list of file paths. No existence checks, no normalization.
        self._attachments: List[str] = []

        # Pre-invoke Tool: strict identity by default -> requires {"prompt": str}
        if pre_invoke is not None and isinstance(pre_invoke, Callable):
            pre_invoke = Tool(
                func=pre_invoke,
                name="pre_invoke_callable",
                description="Pre-invoke callable adapted to Tool",
                type="function",
                source="default",
            )
        self._pre_invoke: Tool = pre_invoke if pre_invoke is not None else self._make_identity_prompt_tool()

    # ------------------------------- utilities -------------------------------

    @staticmethod
    def _make_identity_prompt_tool() -> Tool:
        """Create a strict Tool that requires exactly {'prompt': str} and returns it."""

        def identity_prompt(*, prompt: str) -> str:
            if not isinstance(prompt, str):
                raise ValueError("prompt must be a string")
            return prompt

        return Tool(
            func=identity_prompt,
            name="identity_prompt",
            description="Strict mapping {'prompt': str} -> prompt",
        )

    # --------------------------------- API -----------------------------------
    @property
    def name(self) -> str:
        """Read-only agent name."""
        return self._name

    @property
    def description(self) -> str:
        """Return description agent description."""
        return self._description
    
    @description.setter
    def description(self, val: str) -> None:
        """Set description"""
        self._description = val or "You are a helpful AI assistant"

    @property
    def role_prompt(self) -> Optional[str]:
        """Optional system persona (first message if set)."""
        return self._role_prompt

    @role_prompt.setter
    def role_prompt(self, value: Optional[str]) -> None:
        self._role_prompt = (value or "You are a helpful AI assistant")

    @property
    def llm_engine(self) -> LLMEngine:
        """Engine used to perform the model call (must be an `LLMEngine`)."""
        return self._llm_engine

    @llm_engine.setter
    def llm_engine(self, value: LLMEngine) -> None:
        if not isinstance(value, LLMEngine):
            raise AgentError("llm_engine must be an instance of LLMEngine.")
        self._llm_engine = value

    @property
    def context_enabled(self) -> bool:
        """If True, send prior turns and record new ones; if False, do neither."""
        return self._context_enabled

    @context_enabled.setter
    def context_enabled(self, enabled: bool) -> None:
        if not isinstance(enabled, bool):
            raise AgentError("context_enabled must be a bool.")
        self._context_enabled = enabled

    @property
    def history_window(self) -> int:
        """
        Send-window measured in **turns** (user+assistant pairs).

        - 0    => send none.
        - N>0  => send last N turns (2N messages).
        """
        return self._history_window

    @history_window.setter
    def history_window(self, value: int) -> None:
        if not isinstance(value, int) or value < 0:
            raise AgentError("history_window must be an int >= 0.")
        self._history_window = value

    @property
    def history(self) -> List[Dict[str, str]]:
        """A shallow copy of the stored message history (never trimmed)."""
        return list(self._history)

    def clear_memory(self) -> None:
        """Clear the stored message history."""
        self._history.clear()

    @property
    def attachments(self) -> List[str]:
        """A shallow copy of the current attachment paths."""
        return list(self._attachments)

    def attach(self, path: str) -> bool:
        """
        Add a file path to attachments if not already present.

        Returns True if added, False if it was already present.
        """
        if not isinstance(path, str) or not path:
            raise AgentError("attach(path): path must be a non-empty string.")
        if path in self._attachments:
            return False
        self._attachments.append(path)
        return True

    def detach(self, path: str) -> bool:
        """
        Remove a file path from attachments.

        Returns True if removed, False if it was not present.
        """
        try:
            self._attachments.remove(path)
            return True
        except ValueError:
            return False

    @property
    def pre_invoke(self) -> Tool:
        """
        Tool that converts the input mapping into a **prompt string**.

        Default: strict identity that only accepts `{"prompt": str}`.
        """
        return self._pre_invoke

    @pre_invoke.setter
    def pre_invoke(self, tool: Tool|Callable) -> None:
        if isinstance(tool, Callable):
            tool = Tool(
                func=tool,
                name="pre_invoke_callable",
                description=tool.__doc__ or f"Pre-invoke callable adapted to Tool for agent {self.name}",
                type="function",
                source="default",
            )
        if not isinstance(tool, Tool):
            raise AgentError("pre_invoke must be a Tools.Tool instance or a callable object.")
        self._pre_invoke = tool

    @property
    def arguments_map(self) -> OrderedDict[str, Any]:
        return self.pre_invoke.arguments_map

    def _invoke(self, prompt: str) -> Any:
        """
        Internal call path used by `invoke`. Assumes `prompt` is already validated
        as a string by the base `invoke()` method.
        """
        # 3) Build messages for the engine
        logger.debug(f"[Agent - {self.name}]._invoke: Building messages")
        messages: List[Dict[str, str]] = []

        # optional system
        if self._role_prompt:
            messages.append({"role": "system", "content": self._role_prompt})

        # prior turns (if context is enabled)
        if self._context_enabled and self._history:
            # window in turns: take last N *pairs* => 2N messages
            prior = self._history[-(self._history_window * 2):] if self._history_window > 0 else []
            messages.extend(prior)

        # current user prompt
        messages.append({"role": "user", "content": prompt})

        # 4) Call engine with a small signature shim
        try:
            logger.debug(f"[Agent - {self.name}]._invoke: Invoking LLM")
            text = self._llm_engine.invoke(messages, self._attachments)
        except Exception as e:  # pragma: no cover - engine-specific failures
            raise AgentInvocationError(f"engine invocation failed: {e}") from e

        # Engine contract: must return a string
        if not isinstance(text, str):
            raise AgentInvocationError(
                f"engine returned non-string (type={type(text)!r}); a string is required"
            )

        # 5) Optionally record the turn in stored history
        if self._context_enabled:
            logger.debug(f"[Agent - {self.name}]._invoke: Saving prompt & response to history")
            self._history.append({"role": "user", "content": prompt})
            self._history.append({"role": "assistant", "content": text})

        return text

    def invoke(self, inputs: Mapping[str, Any]) -> Any:
        """
        Invoke the Agent with a single **input mapping**.

        Steps:
        1) Validate `inputs` is a Mapping.
        2) `prompt = pre_invoke.invoke(inputs)` → must be a `str`.
           - If the Tool raises `ToolInvocationError`, it propagates unchanged.
           - Other exceptions are wrapped as `AgentInvocationError`.
        3) Build messages: [system?] + [last N turns] + [user(prompt)].
        4) Call engine via compatibility shim (attachments supported).
        5) Return the engine's text and (if enabled) record the turn in history.

        Parameters
        ----------
        inputs : Mapping[str, Any]
            Input mapping to be adapted to a prompt string via `pre_invoke`.

        Returns
        -------
        str
            The LLM text response.

        Raises
        ------
        TypeError
            If `inputs` is not a Mapping.
        ToolInvocationError
            If the pre-invoke Tool rejects the inputs (e.g., unknown keys).
        AgentInvocationError
            For engine signature/return issues or other unexpected failures.
        """
        logger.info(f"[Agent - {self.name}].invoke begin")
        if not isinstance(inputs, Mapping):
            raise TypeError("Agent.invoke expects a Mapping[str, Any].")

        # 2) Run pre-invoke Tool to get the prompt (strict by default)
        try:
            prompt = self._pre_invoke.invoke(inputs)  # may raise ToolInvocationError
        except ToolInvocationError:
            # let Tool errors bubble up (they are already precise)
            raise
        except Exception as e:  # pragma: no cover - safety net
            raise AgentInvocationError(f"pre_invoke Tool failed: {e}") from e

        if not isinstance(prompt, str):
            raise AgentInvocationError(
                f"pre_invoke returned non-string (type={type(prompt)!r}); "
                "a prompt string is required"
            )
        result = self._invoke(prompt=prompt)
        logger.info(f"[Agent - {self.name}].invoke end")
        # Delegate to the internal call path
        return result

    # ------------------------------ diagnostics ------------------------------

    def to_dict(self) -> OrderedDict[str, Any]:
        """A minimal diagnostic snapshot of this agent (safe to log/serialize)."""
        return OrderedDict({
            # initialization variables
            "name": self._name,
            "description": self._description,
            "role_prompt": self._role_prompt,
            "pre_invoke": self._pre_invoke.to_dict(),
            "llm": self._llm_engine.to_dict() if self._llm_engine else type(None),
            "context_enabled": self._context_enabled,
            "history_window": self._history_window,
            # Runtime variables
            "history": self._history,
            "attachments": self._attachments,
        })

    def __repr__(self) -> str:
        role_preview = ""
        if self._role_prompt:
            rp = self._role_prompt.strip().replace("\n", " ")
            role_preview = (rp[:32] + "…") if len(rp) > 32 else rp
        turns = sum(1 for m in self._history if m.get("role") == "assistant")
        return (
            f"Agent(name={self._name!r}, "
            f"role_prompt_preview={role_preview!r}, "
            f"history_window={self._history_window!r}, "
            f"turns={turns}, "
            f"attachments={len(self._attachments)})"
        )
