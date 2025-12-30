from __future__ import annotations
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Union,
)
import logging
import threading

from ..core.Exceptions import (
    AgentError,
    AgentInvocationError,
    ToolInvocationError,
)
from ..core.Invokable import AtomicInvokable, ArgumentMap
from ..engines.LLMEngines import LLMEngine
from ..tools import Tool, toolify

logger = logging.getLogger(__name__)

# ───────────────────────────────────────────────────────────────────────────────
# Agent
# ───────────────────────────────────────────────────────────────────────────────
def identity_pre(*, prompt: str) -> str:
    if not isinstance(prompt, str):
        raise ValueError("prompt must be a string")
    return prompt

def identity_post(*, result: Any) -> Any:
    """
    Default post-invoke identity function.

    This function accepts a single argument named ``result`` and returns it
    unchanged. It is wrapped as a Tool and used when no explicit ``post_invoke``
    Tool is provided.
    """
    return result


class Agent(AtomicInvokable):
    """
    Schema-driven LLM Agent.

    An Agent is a stateful software unit that points to an LLM and carries a persona
    (system role-prompt). It accepts a single **input mapping** and uses a **pre-invoke
    Tool** to convert that mapping into a **prompt string** before invoking the engine.

    Core behavior:
    - `invoke(inputs: Mapping[str, Any]) -> Any`
      1) Validate inputs (must be a Mapping).
      2) `pre_invoke.invoke(inputs) -> str` (default strict: requires `{"prompt": str}`).
      3) Build messages: [system?] + [last N turns] + [user(prompt)].
      4) Call `llm_engine.invoke(messages)` to obtain a raw result (base Agent expects `str`).
      5) Pass the raw result through `post_invoke` (a single-parameter Tool) to obtain the final output.
      6) Record the turn if `context_enabled`; return the final result.

    Inputs and schema
    -----------------
    - Inputs are always a mapping (`Mapping[str, Any]`).
    - The schema is defined entirely by the `pre_invoke` Tool, which:
      - validates and normalizes the incoming mapping,
      - converts it into a prompt string.

    History and context
    -------------------
    - The Agent keeps an in-memory history of messages as a flat list of dicts:
      `{"role": "user"|"assistant"|"system", "content": str}`.
    - `history_window` controls how many *turns* (user+assistant pairs) from the tail
      of the history are included when building messages.
    - History is append-only; no trimming or summarization is performed by default.

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
    post_invoke : Optional[Tool or Callable], default None
        Tool that converts the raw result from :meth:`_invoke` into the final
        return value. It must accept exactly one parameter. If None, a default
        identity Tool is used that returns its single argument unchanged.

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
    post_invoke : Tool (read-write)
    """

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        name: str,
        description: str,
        llm_engine: LLMEngine,
        role_prompt: Optional[str] = None,
        context_enabled: bool = True,
        *,
        pre_invoke: Optional[AtomicInvokable | Callable] = None,
        post_invoke: Optional[AtomicInvokable | Callable] = None,
        history_window: Optional[int] = None,
    ) -> None:

        # Prepare pre_invoke
        pre_tool = toolify(pre_invoke or identity_pre,
                           name = "pre_invoke",
                           namespace = name,
                           description = f"The tool that preprocesses inputs into a string for Agent {name}")[0]
        if pre_tool.return_type.lower() not in {"any", "str"}:
            raise AgentError("Agent.pre_invoke must return a type 'str'|'any' after updating pre_invoke")
        # Prepare post_invoke
        post_tool = toolify(post_invoke or identity_post,
                           name = "post_invoke",
                           namespace = name,
                           description = f"The tool that postprocesses outputs of Agent {name}")[0]
        required = 0
        if len(post_tool.arguments_map) == 0:
            raise AgentError("Agent.post_invoke must expect least 1 argument")
        if len(post_tool.arguments_map) == 1:
            self._post_param_name = list(post_tool.arguments_map.keys())[0]
        else:
            for arg in post_tool.arguments_map:
                if "default" not in post_tool.arguments_map[arg]:
                    required += 1
                    self._post_param_name = arg
            if required != 1:
                raise AgentError(f"Agent.post_invoke must have exactly 1 required argument, got {required}")
        # Set Pre/Post invoke
        self._pre_invoke = pre_tool
        self._post_invoke = post_tool

        # Set the agent-specific attributes
        self._llm_engine: LLMEngine = llm_engine
        self._role_prompt: str = role_prompt.strip() or "You are a helpful AI assistant"
        self._context_enabled: bool = context_enabled

        # history_window: strict int semantics (>= 0). 'None' means (send-all) mode.
        if history_window is not None and (not isinstance(history_window, int) or history_window < 0):
            raise AgentError("history_window must be an int >= 0 or be 'None'.")
        self._history_window: Optional[int] = history_window

        # Stored message history (flat list of role/content dicts).
        # We never trim storage; we only limit what we *send* to the engine.
        self._history: List[Dict[str, str]] = []

        # Per-invoke buffer for the newest run. This is populated inside `_invoke`
        # and committed to `_history` by `_update_history` if `context_enabled` is True.
        self._newest_history: List[Dict[str, str]] = []
        
        # invoke lock
        self._invoke_lock = threading.RLock()

        # set the core AtomicInvokable attributes
        super().__init__(name=name, description=description)

    # ------------------------------------------------------------------ #
    # Agent Properties
    # ------------------------------------------------------------------ #
    @property
    def role_prompt(self) -> str:
        """Optional system persona (first message if set)."""
        return self._role_prompt

    @role_prompt.setter
    def role_prompt(self, value: Optional[str]) -> None:
        if value is not None and not isinstance(value, str):
            raise TypeError(f"role_prompt must be of type 'str' or 'None', but got {type(value).__name__}")
        self._role_prompt = value.strip() or "You are a helpful AI assistant"
    
    @property
    def llm_engine(self) -> LLMEngine:
        """LLMEngine used for this agent."""
        return self._llm_engine

    @llm_engine.setter
    def llm_engine(self, engine: LLMEngine) -> None:
        if not isinstance(engine, LLMEngine):
            raise TypeError("llm_engine must be an instance of LLMEngine.")
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
    def pre_invoke(self, candidate: Optional[Union[Callable, AtomicInvokable]]) -> None:
        """
        Set the pre-invoke Tool.

        Parameters
        ----------
        candidate : Atomic-Invokable or callable
            - If a Tool, it is used as-is.
            - If a callable, it is wrapped as a Tool with a simple signature.

        Behaviour:
        - Use the centralized helper to create/validate the candidate Tool.
        - Compute the new (arguments_map, return_type) based on the candidate
          and the current `post_invoke` before mutating internal state.
        - If the build would not result in the required types, raise `AgentError`.
        """
        # Prepare pre_invoke
        pre_tool = toolify(candidate or identity_pre,
                           name = "pre_invoke",
                           namespace = self.name,
                           description = f"The tool that preprocesses inputs into a string for Agent {self.name}")[0]
        if pre_tool.return_type.lower() not in {"any", "str"}:
            raise AgentError("Agent.pre_invoke must return a type 'str'|'any' after updating pre_invoke")
        # Apply the candidate and sync internal schema
        self._pre_invoke = pre_tool
        args, ret = self.build_args_returns()
        self._arguments_map, self._return_type = args, ret
        self._is_persistible = self._compute_is_persistible()
    
    @property
    def post_invoke(self) -> Tool:
        """
        Tool that converts the raw result from :meth:`_invoke` into the final
        return value for :meth:`invoke`.

        This Tool must accept exactly one parameter. The base Agent enforces
        this constraint when the Tool is set.
        """
        return self._post_invoke

    @post_invoke.setter
    def post_invoke(self, candidate: Optional[Union[Callable, AtomicInvokable]]) -> None:
        """
        Set the post-invoke Tool.

        Parameters
        ----------
        tool : Tool or callable
            - If a Tool, it is used as-is.
            - If a callable, it is wrapped as a Tool with a simple signature.

        Behaviour:
        - Use the centralized helper to create the candidate Tool.
        - Compute the new (arguments_map, return_type) based on the candidate
          and the current `pre_invoke` before mutating internal state.
        - If the build would not result in the required types, raise `AgentError`.
        """
        # Create candidate tool (do not mutate state yet)
        post_tool = toolify(candidate or identity_post,
                           name = "post_invoke",
                           namespace = self.name,
                           description = f"The tool that postprocesses outputs of Agent {self.name}")[0]
        required = 0
        if len(post_tool.arguments_map) == 0:
            raise AgentError("Agent.post_invoke must expect least 1 argument")
        if len(post_tool.arguments_map) == 1:
            self._post_param_name = list(post_tool.arguments_map.keys())[0]
        else:
            for arg in post_tool.arguments_map:
                if "default" not in post_tool.arguments_map[arg]:
                    required += 1
                    self._post_param_name = arg
            if required != 1:
                raise AgentError(f"Agent.post_invoke must have exactly 1 required argument, got {required}")
        self._post_invoke = post_tool
        self._arguments_map, self._return_type = self.build_args_returns()
        self._is_persistible = self._compute_is_persistible()

    # ------------------------------------------------------------------ #
    # Atomic-Invokable Helpers
    # ------------------------------------------------------------------ #
    def build_args_returns(self) -> tuple[ArgumentMap, str]:
        """Return the Agent's input argument map (mirroring pre_invoke) and the post_invoke return type."""
        return self._pre_invoke.arguments_map, str(self._post_invoke.return_type)

    def _compute_is_persistible(self):
        return self.pre_invoke.is_persistible and self.post_invoke.is_persistible

    # ------------------------------------------------------------------ #
    # Agent Helpers
    # ------------------------------------------------------------------ #
    def _invoke(self, messages: List[Dict[str, str]]) -> Any:
        """Internal call path used by :meth:`invoke`.

        This base implementation:
        - Invokes the configured LLM engine with the provided ``messages``.
        - Populates ``_newest_history`` with the *current turn only*
          (user + assistant messages).

        Subclasses may override this method to implement more complex behavior,
        but **must not** mutate ``self._history`` directly. Instead, they should
        either:
        - append messages to ``self._newest_history``, or
        - store richer run data on ``self`` for :meth:`_update_history` to
          summarize and commit.
        """
        # 1) Call engine (attachments are managed by the engine itself)
        try:
            logger.debug(f"[Agent - {self.name}]._invoke: Invoking LLM")
            text = self._llm_engine.invoke(messages)
        except Exception as e:  # pragma: no cover - engine-specific failures
            raise AgentInvocationError(f"engine invocation failed: {e}") from e

        # 2) Engine contract: base Agent expects a string
        if not isinstance(text, str):
            raise AgentInvocationError(
                f"engine returned non-string (type={type(text)!r}); a string is required"
            )

        # 3) Record the current turn into the per-invoke buffer.
        # The base Agent simply stores the user prompt and raw assistant text.
        user_msg = messages[-1]
        
        self._newest_history.append(user_msg)
        self._newest_history.append({"role": "assistant", "content": text})

        return text

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def attach(self, path: str) -> Mapping[str, Any]:
        """
        Add a file path to attachments via the underlying engine.

        Returns
        -------
        Mapping[str, Any]
            The attachment path and metadata.
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

    def clear_attachments(self) -> None:
        return self.llm_engine.clear_attachments()

    def clear_memory(self) -> None:
        """Clear the stored message history."""
        self._history.clear()
        self._newest_history.clear()
    
    def invoke(self, inputs: Mapping[str, Any]) -> Any:
        """Invoke the Agent with a single **input mapping**.

        Steps
        -----
        1) Validate ``inputs`` is a :class:`~collections.abc.Mapping`.
        2) ``prompt = pre_invoke.invoke(inputs)`` → must be a ``str``.
           - If the Tool raises :class:`ToolInvocationError`, it propagates unchanged.
           - Other exceptions are wrapped as :class:`AgentInvocationError`.
        3) Build the messages list from the optional role prompt, the windowed
           history, and the current user ``prompt``.
        4) Delegate to :meth:`_invoke`, which performs the actual engine call and
           populates ``_newest_history`` for this run.
        5) If ``context_enabled`` is True, call update history with the _newest_history to commit
           the newest turn(s) into persistent history.
        6) Run ``post_invoke`` on the raw result to obtain the final output and
           return it.

        Parameters
        ----------
        inputs : Mapping[str, Any]
            Input mapping to be adapted to a prompt string via ``pre_invoke``.

        Returns
        -------
        Any
            The Agent's response. For the base Agent, this is the result of the
            ``post_invoke`` Tool applied to the raw LLM output.

        Raises
        ------
        TypeError
            If ``inputs`` is not a Mapping.
        ToolInvocationError
            If the pre- or post-invoke Tool rejects the inputs.
        AgentInvocationError
            For unexpected runtime errors in Tools or the engine.
        """
        
        # main invoke lock        
        with self._invoke_lock:
            logger.info(f"[{type(self).__name__}.{self.name}.invoke started]")
            if not isinstance(inputs, Mapping):
                raise TypeError("Agent.invoke expects a Mapping[str, Any].")
            # Preprocess inputs to prompt string
            try:
                logger.debug(f"Agent.{self.name}.pre_invoke preprocessing inputs")
                prompt = self._pre_invoke.invoke(inputs)
            except ToolInvocationError:
                raise
            except Exception as e:  # pragma: no cover
                raise AgentInvocationError(f"pre_invoke Tool failed: {e}") from e

            if not isinstance(prompt, str):
                raise AgentInvocationError(
                    f"pre_invoke returned non-string (type={type(prompt)!r}); a prompt string is required"
                )
            # Empty any prior history buffer
            self._newest_history.clear()

            # Build messages
            logger.debug(f"Agent.{self.name} building messages for class '{type(self).__name__}'")
            messages: List[Dict[str, str]] = [{"role": "system", "content": self.role_prompt}]

            if self._context_enabled and self._history:
                if self._history_window is None:
                    prior = self._history
                elif self._history_window > 0:
                    prior = self._history[-(self._history_window * 2):]
                else:
                    prior = self._history
                messages.extend(prior)

            user_msg = {"role": "user", "content": prompt}
            messages.append(user_msg)

            # Invoke core logic
            logger.debug(f"Agent.{self.name} performing logic for class '{type(self).__name__}'")
            raw_result = self._invoke(messages=messages)

            # Update history if enabled
            if self._context_enabled:
                logger.debug(f"Agent.{self.name} updating history")
                self._history.extend(self._newest_history)
            self._newest_history.clear()

            # Postprocess raw result
            try:
                logger.debug(f"Agent.{self.name}.post_invoke postprocessing result")
                result = self._post_invoke.invoke({self._post_param_name: raw_result})
            except ToolInvocationError:
                raise
            except Exception as e:  # pragma: no cover
                raise AgentInvocationError(f"post_invoke Tool failed: {e}") from e

            # Final logging and return
            logger.info(f"[{type(self).__name__}.{self.name}.invoke finished]")
            return result

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        """A minimal diagnostic snapshot of this agent (safe to log/serialize)."""
        d = super().to_dict()
        d.update({
            "role_prompt": self.role_prompt,
            "pre_invoke": self.pre_invoke.to_dict(),
            "post_invoke": self.post_invoke.to_dict(),
            "llm": self._llm_engine.to_dict(),
            "context_enabled": self.context_enabled,
            "history_window": self.history_window,
            "history": self.history
        })
        return d
