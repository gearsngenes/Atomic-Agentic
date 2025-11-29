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
- Workflows and ToolAdapters can treat Agents as boxes that accept `Mapping[str, Any]`
  and return arbitrary Python objects, while only the pre-invoke Tool and LLMEngine
  know about the concrete prompt structure or model protocols.

Design overview
---------------
- Agents are thin coordination layers around:
  - an `LLMEngine` instance (how to call the model),
  - a `pre_invoke` Tool (how to build the prompt),
  - optional role prompt / persona,
  - stored conversation history (list of messages),
  - external attachments (delegated to the engine).

- Agents do **not** own their own concurrency or dedicated threads; they are
  synchronous call units.

- History is stored as a flat list of `{"role": str, "content": str}` dicts.

How `invoke` works (high-level)
-------------------------------
Given `agent.invoke(inputs)`:
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
>>> from LLMEngines import OpenAIEngine
>>> from Tools import Tool
>>> from Agents import Agent
...
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

from typing import Any, Dict, List, Mapping, Optional, Callable, Union
import logging
from collections import OrderedDict

# Local imports (adjust the module paths if your project structure differs)
from .LLMEngines import *
from .Tools import *
from ._exceptions import *


__all__ = ["Agent", "AgentTool"]


logger = logging.getLogger(__name__)


def identity_prompt(*, prompt: str) -> str:
    if not isinstance(prompt, str):
        raise ValueError("prompt must be a string")
    return prompt

default_pre_invoke = Tool(
    func=identity_prompt,
    name="identity_prompt",
    description="A simple identity tool that returns the 'prompt' argument as is.",
)


# ───────────────────────────────────────────────────────────────────────────────
# Agent
# ───────────────────────────────────────────────────────────────────────────────
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
    pre_invoke : Optional[Tool or Callable], default None
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
    attachments : Dict[str, Dict[str, Any]] (read-only view)
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
        pre_invoke: Optional[Tool | Callable] = None,
        history_window: int = 50,
    ) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("Agent 'name' must be a non-empty string.")
        if not isinstance(description, str):
            raise ValueError("Agent 'description' must be a string.")
        if not isinstance(llm_engine, (LLMEngine, type(None))):
            raise ValueError("llm_engine must be an instance of LLMEngine.")

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
        # We never trim storage; we only limit what we *send* to the engine.
        self._history: List[Dict[str, str]] = []

        # Per-invoke buffer for the newest run. This is populated inside `_invoke`
        # and committed to `_history` by `_update_history` if `context_enabled` is True.
        self._newest_history: List[Dict[str, str]] = []

        # Pre-invoke Tool: strict identity by default -> requires {"prompt": str}
        if pre_invoke is not None and isinstance(pre_invoke, Callable):
            pre_invoke = Tool(
                func=pre_invoke,
                name="pre_invoke",
                description="Pre-invoke callable adapted to Tool",
                source=f"{self.name}",
            )
        self._pre_invoke: Tool = pre_invoke if pre_invoke is not None else default_pre_invoke

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
        self._role_prompt = (value or "You are a helpful AI assistant").strip() or None

    @property
    def llm_engine(self) -> LLMEngine:
        """LLMEngine used for this agent."""
        return self._llm_engine

    @llm_engine.setter
    def llm_engine(self, engine: LLMEngine) -> None:
        if not isinstance(engine, LLMEngine):
            raise AgentError("llm_engine must be an instance of LLMEngine.")
        self._llm_engine = engine

    @property
    def context_enabled(self) -> bool:
        """
        Whether the agent uses conversation context
        (sends prior turns and logs new ones).
        """
        return self._context_enabled

    @context_enabled.setter
    def context_enabled(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise ValueError("context_enabled must be a bool.")
        self._context_enabled = value

    @property
    def history_window(self) -> int:
        """
        Number of *turns* (user+assistant pairs) to include from the tail of the
        conversation history when building messages for the engine.
        """
        return self._history_window

    @history_window.setter
    def history_window(self, value: int) -> None:
        if not isinstance(value, int) or value < 0:
            raise ValueError("history_window must be an int >= 0.")
        self._history_window = value

    @property
    def history(self) -> List[Dict[str, str]]:
        """Return a shallow copy of the stored message history (never trimmed)."""
        return list(self._history)

    def clear_memory(self) -> None:
        """Clear the stored message history."""
        self._history.clear()
        self._newest_history.clear()

    @property
    def attachments(self) -> Dict[str, Dict[str, Any]]:
        """A shallow copy of the current attachment paths."""
        return self.llm_engine.attachments

    @property
    def pre_invoke(self) -> Tool:
        """
        Tool that converts the input mapping into a **prompt string**.

        By default this is a strict identity tool that requires *exactly*:
            {"prompt": <str>}
        and returns that string.

        You can set this to any Tool instance that:
        - accepts a Mapping[str, Any]-like object, and
        - returns a `str` when invoked.

        For convenience, setting it to a plain callable will wrap it in a Tool.
        """
        return self._pre_invoke

    @pre_invoke.setter
    def pre_invoke(self, tool: Union[Tool, Callable[..., str]]) -> None:
        """
        Set the pre-invoke Tool.

        Parameters
        ----------
        tool : Tool or callable
            - If a Tool, it is used as-is.
            - If a callable, it is wrapped as a Tool with a simple signature.
        """
        if isinstance(tool, Callable) and not isinstance(tool, Tool):
            tool = Tool(
                func=tool,
                name=f"{self.name}_pre_invoke",
                description=tool.__doc__ or f"Pre-invoke callable adapted to Tool for agent {self.name}",
                type="function",
                source="default",
            )
        if not isinstance(tool, Tool):
            raise ValueError("pre_invoke must be a Tools.Tool instance or a callable object.")
        self._pre_invoke = tool

    def attach(self, path: str) -> bool:
        """
        Add a file path to attachments via the underlying engine.

        Returns
        -------
        bool
            True if added, False if it was already present.
        """
        return self._llm_engine.attach(path)

    def detach(self, path: str) -> bool:
        """
        Remove a file path from attachments via the underlying engine.

        Returns
        -------
        bool
            True if removed, False if it was not present.
        """
        return self._llm_engine.detach(path)

    @property
    def arguments_map(self) -> OrderedDict[str, Any]:
        """
        Mirror of the pre_invoke Tool's arguments map.

        Exposed primarily so that planners and higher-level workflows can inspect
        the agent's input schema (required names, ordering, etc.).
        """
        return self.pre_invoke.arguments_map

    # ----------------------------- core behavior -----------------------------

    def _invoke(self, prompt: str) -> Any:
        """Internal call path used by :meth:`invoke`.

        This base implementation:
        - Builds messages from the optional role prompt, the windowed history,
          and the current user ``prompt``.
        - Invokes the configured LLM engine.
        - Populates ``_newest_history`` with the *current turn only*
          (user + assistant messages).

        Subclasses may override this method to implement more complex behavior,
        but **must not** mutate ``self._history`` directly. Instead, they should
        either:
        - append messages to ``self._newest_history``, or
        - store richer run data on ``self`` for :meth:`_update_history` to
          summarize and commit.
        """
        logger.debug(f"[Agent - {self.name}]._invoke: Building messages")
        messages: List[Dict[str, str]] = []

        # optional system message
        if self._role_prompt:
            messages.append({"role": "system", "content": self._role_prompt})

        # prior turns (if context is enabled)
        if self._context_enabled and self._history:
            # window in turns: take last N *pairs* => 2N messages
            if self._history_window > 0:
                prior = self._history[-(self._history_window * 2):]
            else:
                prior = self._history
            messages.extend(prior)

        # current user prompt
        user_msg = {"role": "user", "content": prompt}
        messages.append(user_msg)

        # 4) Call engine (attachments are managed by the engine itself)
        try:
            logger.debug(f"[Agent - {self.name}]._invoke: Invoking LLM")
            text = self._llm_engine.invoke(messages)
        except Exception as e:  # pragma: no cover - engine-specific failures
            raise AgentInvocationError(f"engine invocation failed: {e}") from e

        # Engine contract: must return a string
        if not isinstance(text, str):
            raise AgentInvocationError(
                f"engine returned non-string (type={type(text)!r}); a string is required"
            )

        # Record the current turn into the per-invoke buffer.
        # The base Agent simply stores the user prompt and raw assistant text.
        self._newest_history.append(user_msg)
        self._newest_history.append({"role": "assistant", "content": text})

        return text

    def _update_history(self) -> None:
        """Commit the newest run's messages to persistent history.

        This method is called by :meth:`invoke` when ``context_enabled`` is True.
        Only this method is allowed to mutate ``self._history``. Subclasses may
        override it to transform or summarize ``_newest_history`` before calling
        ``super()._update_history()``.
        """
        if not self._newest_history:
            return

        self._history.extend(self._newest_history)
        self._newest_history.clear()

    def invoke(self, inputs: Mapping[str, Any]) -> Any:
        """Invoke the Agent with a single **input mapping**.

        Steps
        -----
        1) Validate ``inputs`` is a :class:`~collections.abc.Mapping`.
        2) ``prompt = pre_invoke.invoke(inputs)`` → must be a ``str``.
           - If the Tool raises :class:`ToolInvocationError`, it propagates unchanged.
           - Other exceptions are wrapped as :class:`AgentInvocationError`.
        3) Delegate to :meth:`_invoke`, which performs the actual engine call and
           populates ``_newest_history`` for this run.
        4) If ``context_enabled`` is True, call :meth:`_update_history` to commit
           the newest turn(s) into persistent history.
        5) Return the engine's text (or subclass-specific result).

        Parameters
        ----------
        inputs : Mapping[str, Any]
            Input mapping to be adapted to a prompt string via ``pre_invoke``.

        Returns
        -------
        Any
            The Agent's response (``str`` for the base Agent).

        Raises
        ------
        TypeError
            If ``inputs`` is not a Mapping.
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
            # Let Tool errors bubble up (they are already precise).
            raise
        except Exception as e:  # pragma: no cover - safety net
            raise AgentInvocationError(f"pre_invoke Tool failed: {e}") from e

        if not isinstance(prompt, str):
            raise AgentInvocationError(
                f"pre_invoke returned non-string (type={type(prompt)!r}); a prompt string is required"
            )

        # Reset the per-invoke buffer before running the core logic.
        self._newest_history.clear()

        # Delegate to the internal call path
        result = self._invoke(prompt=prompt)

        # Centralized history policy: only `_update_history` may touch `_history`.
        if self._context_enabled:
            self._update_history()
        else:
            # Ensure no stale data leaks across invocations.
            self._newest_history.clear()

        logger.info(f"[Agent - {self.name}].invoke end")
        return result

    # ------------------------------ diagnostics ------------------------------

    def to_dict(self) -> OrderedDict[str, Any]:
        """A minimal diagnostic snapshot of this agent (safe to log/serialize)."""
        return OrderedDict(
            agent_type = type(self).__name__,
            name = self._name,
            description = self._description,
            role_prompt = self._role_prompt,
            pre_invoke = self._pre_invoke.to_dict(),
            llm = self._llm_engine.to_dict() if self._llm_engine else None,
            context_enabled = self._context_enabled,
            history_window = self._history_window,
            history = self._history,
        )


# ───────────────────────────────────────────────────────────────────────────────
# AgentTool
# ───────────────────────────────────────────────────────────────────────────────
class AgentTool(Tool):
    """
    Adapter that exposes an Agent as a Tool with schema-driven introspection.

    Metadata
    --------
    - type   = "agent"
    - source = agent.name
    - name   = "invoke"
    - description = agent.description

    Schema exposure
    ---------------
    Mirrors the agent's `pre_invoke` Tool call-plan (arguments map, required sets,
    varargs flags, etc.) so planners see the exact input keys & binding rules.

    Signature string
    ----------------
    The base Tool builds a canonical, schema-derived signature of the form:
        "AgentTool.<agent_name>.invoke({p1:v1, p2:v2, ...}) -> Any"
    We set return_type to "Any" and call `_rebuild_signature_str()` after
    overriding the plan to reflect the agent’s actual output.
    """

    def __init__(self, agent: Agent) -> None:
        if not isinstance(agent, Agent):
            raise ToolDefinitionError("AgentTool requires a valid Agent instance.")
        self.agent = agent

        # Initialize as a Tool with the agent's description.
        super().__init__(
            func=self.agent.invoke,
            name="invoke",
            description=agent.description,
            source=agent.name,
        )

        # Override module/qualname to reflect that this is an adapter.
        self.module = None
        self.qualname = None

        # ---- Mirror the pre_invoke call-plan (shallow copies to avoid aliasing) ----
        pre = agent.pre_invoke
        self._arguments_map = pre.arguments_map
        self.posonly_order = list(pre.posonly_order)
        self.p_or_kw_names = list(pre.p_or_kw_names)
        self.kw_only_names = list(pre.kw_only_names)
        self.required_names = set(pre.required_names)
        self.has_varargs = bool(pre.has_varargs)
        self.varargs_name = pre.varargs_name
        self.has_varkw = bool(pre.has_varkw)
        self.varkw_name = pre.varkw_name

        # Explicit return type for AgentTool (agent responses are arbitrary)
        self._return_type = "Any"
        self._rebuild_signature_str()

    def invoke(self, inputs: Mapping[str, Any]) -> Any:
        """
        Invoke the underlying agent with validated inputs.

        Raises
        ------
        ToolInvocationError
            For invocation errors.
        """
        try:
            result = self.agent.invoke(inputs)
            return result
        except Exception as e:
            raise ToolInvocationError(f"AgentTool.invoke error: {e}") from e

    def to_dict(self) -> OrderedDict[str, Any]:
        """
        Serialize AgentTool to a dict, including agent-specific metadata.
        """
        dict_data = super().to_dict()
        dict_data["agent"] = self.agent.to_dict()
        return dict_data
