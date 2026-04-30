from __future__ import annotations
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Union,
    Tuple,
)
import logging
import threading
import warnings

from ..core.Exceptions import (
    AgentError,
    AgentInvocationError,
    ToolInvocationError,
)
from ..core.Invokable import AtomicInvokable
from ..core.Parameters import ParamSpec
from ..core.sentinels import NO_VAL
from ..engines.LLMEngines import LLMEngine
from ..tools import Tool, toolify
from .data_classes import AgentTurn

logger = logging.getLogger(__name__)

def identity_pre(*, prompt: str) -> str:
    if not isinstance(prompt, str):
        raise ValueError("prompt must be a string")
    return prompt

identity_pre_tool = Tool(
    function = identity_pre,
    name="identity_pre",
    namespace="base_agent",
    description="Default pre-invoke identity function that requires {'prompt': str} and returns the prompt string.",
    filter_extraneous_inputs=True,)

def identity_post(*, result: Any) -> Any:
    """
    Default post-invoke identity function.

    This function accepts a single argument named ``result`` and returns it
    unchanged. It is wrapped as a Tool and used when no explicit ``post_invoke``
    Tool is provided.
    """
    return result

identity_post_tool = Tool(
    function = identity_post,
    name="identity_post",
    namespace="base_agent",
    description="Default post-invoke identity function that accepts a single argument 'result' and returns it unchanged.",
    filter_extraneous_inputs=True,)

# ───────────────────────────────────────────────────────────────────────────────
# Agent
# ───────────────────────────────────────────────────────────────────────────────
class Agent(AtomicInvokable):
    """
    Schema-driven LLM Agent.

    An Agent is a stateful software unit that points to an LLM and carries a persona
    (system role-prompt). It accepts a single **input mapping** and uses a **pre-invoke
    Tool** to convert that mapping into a **prompt string** before invoking the engine.

    Core behavior:
    - `invoke(inputs: Mapping[str, Any]) -> Any`
    1) Validate/filter inputs through the AtomicInvokable input pipeline.
    2) `pre_invoke.invoke(inputs) -> str` (default strict: requires `{"prompt": str}`).
    3) Build messages: [system?] + [rendered prior turns] + [user(prompt)].
    4) Delegate to `_invoke(messages)` to obtain a raw response plus turn metadata.
    5) Pass the raw response through `post_invoke` to obtain the final output.
    6) Record a canonical turn if `context_enabled`; return the final output.

    Inputs and schema
    -----------------
    - Inputs are always a mapping (`Mapping[str, Any]`).
    - The schema is defined by the `pre_invoke` Tool, which:
    - validates and normalizes the incoming mapping,
    - converts it into a prompt string.

    History and context
    -------------------
    - The Agent keeps an in-memory history of AgentTurn objects.
    - `history_window` controls how many *turns* from the tail of the history
    are rendered into future LLM-facing messages.
    - Stored history is append-only; no trimming or summarization is performed by default.
    - `history` is currently a deprecated rendered compatibility view.
    - `turn_history` is the canonical stored turn view.

    Parameters
    ----------
    name : str
        Logical name for this agent (read-only).
    description : str
        Short, human-readable description (read-only).
    llm_engine : LLMEngine
        Engine used to perform the model call. Must be an instance of `LLMEngine`.
    role_prompt : Optional[str], default None
        Optional system persona. If None or empty, a default assistant persona is used.
    context_enabled : bool, default True
        If True, the agent renders prior turns and records new turns.
        If False, the agent sends no prior turns and does not record new ones.
    history_window : Optional[int], default None
        Send-window measured in **turns**. None sends all stored turns; 0 sends no turns.
        Stored history is never trimmed.
    pre_invoke : Optional[Tool or Callable], default None
        Tool that converts the input mapping to a `str` prompt. If None, a strict
        identity Tool is used that accepts exactly `{"prompt": str}`.
    post_invoke : Optional[Tool or Callable], default None
        Tool that converts the raw result from `_invoke` into the final return value.
        It must accept exactly one required parameter.
    response_preview_limit : Optional[int], default None
        Optional character limit applied only when rendering stored assistant responses
        into future LLM-facing message history. Stored turn values are not mutated.
    assistant_response_source : Literal["raw", "final"], default "raw"
        Whether rendered assistant history should use each turn's raw response or
        final post-processed response.

    Properties (selected)
    ---------------------
    name : str (read-only)
    description : str (read-only)
    role_prompt : str (read-write)
    llm_engine : LLMEngine (read-write, type-enforced)
    context_enabled : bool (read-write)
    history_window : Optional[int] (read-write; see semantics above)
    history : List[Dict[str, str]] (deprecated rendered message view)
    turn_history : List[AgentTurn] (read-only canonical turn view)
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
        filter_extraneous_inputs: Optional[bool] = None,
        role_prompt: Optional[str] = None,
        context_enabled: bool = True,
        *,
        pre_invoke: Optional[AtomicInvokable | Callable] = None,
        post_invoke: Optional[AtomicInvokable | Callable] = None,
        history_window: Optional[int] = None,
        response_preview_limit: Optional[int] = None,
        assistant_response_source: Literal["raw", "final"] = "raw",
    ) -> None:

        # Prepare pre_invoke
        if not pre_invoke:
            pre_tool = identity_pre_tool
        else:
            pre_tool = toolify(pre_invoke,
                            name="pre_invoke",
                            namespace=name,
                            description=f"The tool that preprocesses inputs into a string for Agent {name}")
        if pre_tool.return_type.lower() not in {"any", "str"}:
            raise AgentError("Agent.pre_invoke must return a type 'str'|'any' after updating pre_invoke")

        # Prepare post_invoke
        if not post_invoke:
            post_tool = identity_post_tool
        else:
            post_tool = toolify(post_invoke,
                                name="post_invoke",
                                namespace=name,
                                description=f"The tool that postprocesses outputs of Agent {name}")
        # Validate post_invoke has exactly 1 required parameter
        post_params = post_tool.parameters
        if len(post_params) == 0:
            raise AgentError("Agent.post_invoke must expect at least 1 argument")
        if len(post_params) == 1:
            self._post_param_name = post_params[0].name
        else:
            required_count = 0
            for param in post_params:
                if param.default is NO_VAL:
                    required_count += 1
                    self._post_param_name = param.name
            if required_count != 1:
                raise AgentError(f"Agent.post_invoke must have exactly 1 required argument, got {required_count}")
        # Set Pre/Post invoke
        self._pre_invoke = pre_tool
        self._post_invoke = post_tool

        # Set the agent-specific attributes
        self._llm_engine: LLMEngine = llm_engine
        self._role_prompt: str = "You are a helpful AI assistant"
        if role_prompt is not None and role_prompt.strip():
            self._role_prompt = role_prompt.strip()
        self._context_enabled: bool = context_enabled

        # history_window: strict int semantics (>= 0). 'None' means (send-all) mode.
        if history_window is not None and (not isinstance(history_window, int) or history_window < 0):
            raise AgentError("history_window must be an int >= 0 or be 'None'.")
        self._history_window: Optional[int] = history_window

        # Stored turn history.
        # We never trim storage; we only limit what we *send* to the engine.
        self._history: List[AgentTurn] = []

        self.response_preview_limit = response_preview_limit
        self.assistant_response_source = assistant_response_source

        filter = filter_extraneous_inputs if filter_extraneous_inputs is not None else pre_tool.filter_extraneous_inputs
        # Build schema directly from pre_invoke and post_invoke, then delegate to parent
        super().__init__(name=name,
                         description=description,
                         parameters=self._pre_invoke.parameters,
                         return_type=self._post_invoke.return_type,
                         filter_extraneous_inputs=filter,)

    # ------------------------------------------------------------------ #
    # Agent Properties
    # ------------------------------------------------------------------ #
    @property
    def role_prompt(self) -> str:
        """System persona rendered as the first message."""
        return self._role_prompt

    @role_prompt.setter
    def role_prompt(self, value: Optional[str]) -> None:
        if value is None:
            self._role_prompt = "You are a helpful AI assistant"
            return
        if not isinstance(value, str):
            raise TypeError(
                f"role_prompt must be of type 'str' or 'None', but got {type(value).__name__}"
            )
        cleaned = value.strip()
        self._role_prompt = cleaned or "You are a helpful AI assistant"

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
        Whether the agent uses memory context.

        If True, prior turns are rendered into future messages and completed
        invocations are stored as turns. If False, no prior turns are sent and
        no new turns are recorded.
        """
        return self._context_enabled

    @context_enabled.setter
    def context_enabled(self, value: bool) -> None:
        if type(value) is not bool:
            raise ValueError("context_enabled must be a bool.")
        self._context_enabled = value

    @property
    def history_window(self) -> Optional[int]:
        """
        Number of turns to include from the tail of stored turn history.

        None sends all stored turns. 0 sends no prior turns. Stored history is
        never trimmed by this setting.
        """
        return self._history_window

    @history_window.setter
    def history_window(self, value: Optional[int]) -> None:
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError("history_window must be an int >= 0 or be 'None'.")
        self._history_window = value

    @property
    def response_preview_limit(self) -> Optional[int]:
        """Character limit for rendered assistant responses. None means no truncation."""
        return self._response_preview_limit

    @response_preview_limit.setter
    def response_preview_limit(self, value: Optional[int]) -> None:
        if value is None:
            self._response_preview_limit = None
            return
        if type(value) is not int or value <= 0:
            raise AgentError("response_preview_limit must be None or a positive integer > 0.")
        self._response_preview_limit = value

    @property
    def assistant_response_source(self) -> Literal["raw", "final"]:
        """Whether rendered assistant history uses raw or final turn responses."""
        return self._assistant_response_source

    @assistant_response_source.setter
    def assistant_response_source(self, value: Literal["raw", "final"]) -> None:
        if not isinstance(value, str) or value not in {"raw", "final"}:
            raise AgentError("assistant_response_source must be either 'raw' or 'final'.")
        self._assistant_response_source = value

    @property
    def history(self) -> List[Dict[str, str]]:
        """Return a rendered message history compatibility view."""
        warnings.warn(
            "Agent.history currently returns rendered message dictionaries for compatibility. "
            "Use Agent.turn_history for canonical stored turns. In a future version, "
            "Agent.history may become turn-native.",
            DeprecationWarning,
            stacklevel=2,
        )
        rendered: List[Dict[str, str]] = []
        for turn in self._history:
            rendered.extend(self.render_turn(turn))
        return rendered

    @property
    def turn_history(self) -> List[AgentTurn]:
        """Return a shallow copy of the stored turn history (never trimmed)."""
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
        - Rebuild the Agent's schema based on the candidate pre_invoke Tool.
        - If the Tool does not return 'str' or 'any', raise `AgentError`.
        """
        # Prepare pre_invoke
        pre_tool = toolify(candidate or identity_pre,
                           name="pre_invoke",
                           namespace=self.name,
                           description=f"The tool that preprocesses inputs into a string for Agent {self.name}")
        if pre_tool.return_type.lower() not in {"any", "str"}:
            raise AgentError("Agent.pre_invoke must return a type 'str'|'any' after updating pre_invoke")
        # Apply the candidate and rebuild schema from components
        self._pre_invoke = pre_tool
        self._parameters = self._pre_invoke.parameters

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
        - Rebuild the Agent's schema based on the candidate post_invoke Tool.
        - The Tool must accept exactly one required parameter.
        - If validation fails, raise `AgentError`.
        """
        # Create candidate tool and validate it has exactly 1 required parameter
        post_tool = toolify(candidate or identity_post,
                           name="post_invoke",
                           namespace=self.name,
                           description=f"The tool that postprocesses outputs of Agent {self.name}")
        post_params = post_tool.parameters
        if len(post_params) == 0:
            raise AgentError("Agent.post_invoke must expect at least 1 argument")
        if len(post_params) == 1:
            self._post_param_name = post_params[0].name
        else:
            required_count = 0
            for param in post_params:
                if param.default is NO_VAL:
                    required_count += 1
                    self._post_param_name = param.name
            if required_count != 1:
                raise AgentError(f"Agent.post_invoke must have exactly 1 required argument, got {required_count}")
        # Apply the candidate and rebuild schema from components
        self._post_invoke = post_tool
        self._return_type = self._post_invoke.return_type

    # ------------------------------------------------------------------ #
    # Agent Helpers
    # ------------------------------------------------------------------ #
    def build_messages(self, prompt: str) -> List[Dict[str, str]]:
        """Build LLM-facing message dicts from role prompt, rendered prior turns, and current prompt.

        This method does not mutate stored memory. Prior turns are selected according to
        `history_window` and rendered through `render_turn(...)`, allowing subclasses to
        control how their canonical turn records become provider-facing messages.
        """
        messages: List[Dict[str, str]] = [{"role": "system", "content": self.role_prompt}]

        if self._context_enabled and self._history:
            if self._history_window is None:
                prior = self._history
            elif self._history_window == 0:
                prior = []
            else:
                prior = self._history[-self._history_window:]

            for turn in prior:
                messages.extend(self.render_turn(turn))

        user_msg = {"role": "user", "content": prompt}
        messages.append(user_msg)

        return messages

    def render_turn(self, turn: AgentTurn) -> List[Dict[str, str]]:
        """Render one stored AgentTurn into an LLM-facing user/assistant message pair.

        The assistant content is selected from either `turn.raw_response` or
        `turn.final_response` according to `assistant_response_source`. The optional
        `response_preview_limit` is applied only to the rendered text; stored turn values
        are never mutated.
        """
        if not isinstance(turn, AgentTurn):
            raise AgentInvocationError(
                f"render_turn expected AgentTurn, got {type(turn)!r}"
            )

        response = (
            turn.raw_response
            if self._assistant_response_source == "raw"
            else turn.final_response
        )
        response_text = str(response)

        if (
            self._response_preview_limit is not None
            and len(response_text) > self._response_preview_limit
        ):
            response_text = response_text[:self._response_preview_limit] + "..."

        return [
            {"role": "user", "content": turn.prompt},
            {"role": "assistant", "content": response_text},
        ]

    def _make_turn(
        self,
        *,
        prompt: str,
        raw_response: Any,
        final_response: Any,
        **metadata: Any,
    ) -> AgentTurn:
        """Construct the canonical stored turn for one completed invocation.

        Base Agent turns accept no extra metadata. Subclasses that return metadata from
        `_invoke(...)` or `_ainvoke(...)` should override this method and consume that
        metadata explicitly.
        """
        if metadata:
            raise AgentInvocationError(
                f"{type(self).__name__}._make_turn received unexpected metadata: "
                f"{sorted(metadata.keys())!r}"
            )
        return AgentTurn(
            prompt=prompt,
            raw_response=raw_response,
            final_response=final_response,
        )

    async def _ainvoke(self, messages: List[Dict[str, str]]) -> Tuple[Any, Mapping[str, Any]]:
        """Async internal call path used by `async_invoke`.

        Default implementation delegates to the engine's async interface and returns the
        raw engine response plus a metadata mapping for `_make_turn(...)`.

        Subclasses may override this method to implement more complex async behavior, but
        must not mutate `self._history` directly. Memory is committed by `async_invoke`
        after post-processing has produced the final response.
        """
        try:
            logger.debug(f"[Agent - {self.name}]._ainvoke: Invoking LLM asynchronously")
            text = await self._llm_engine.async_invoke({"messages": messages})
        except Exception as e:  # pragma: no cover - engine-specific failures
            raise AgentInvocationError(f"engine async invocation failed: {e}") from e

        if not isinstance(text, str):
            raise AgentInvocationError(
                f"engine returned non-string (type={type(text)!r}); a string is required"
            )

        return text, {}

    def _invoke(self, messages: List[Dict[str, str]]) -> Tuple[Any, Mapping[str, Any]]:
        """Internal call path used by :meth:`invoke`.

        This base implementation:
        - Invokes the configured LLM engine with the provided ``messages``.
        - Returns the raw engine response plus turn metadata.

        Subclasses may override this method to implement more complex behavior,
        but **must not** mutate ``self._history`` directly. Instead, they should
        return:
        - a raw response
        - a metadata mapping consumed by :meth:`_make_turn`
        """
        # 1) Call engine (attachments are managed by the engine itself)
        try:
            logger.debug(f"[Agent - {self.name}]._invoke: Invoking LLM")
            text = self._llm_engine.invoke({"messages": messages})
        except Exception as e:  # pragma: no cover - engine-specific failures
            raise AgentInvocationError(f"engine invocation failed: {e}") from e

        # 2) Engine contract: base Agent expects a string
        if not isinstance(text, str):
            raise AgentInvocationError(
                f"engine returned non-string (type={type(text)!r}); a string is required"
            )

        return text, {}

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def attach(self, path: str) -> Mapping[str, Any]:
        """
        Attach a file to this Agent via the underlying LLM engine.

        This method delegates to the engine's attachment system, which validates
        paths, extracts metadata, and prepares provider-specific structures.
        Each engine has its own supported formats, policies, and size limits.

        Parameters
        ----------
        path : str
            Local filesystem path to the file. Must be a non-empty string.

        Returns
        -------
        Mapping[str, Any]
            Provider-specific attachment metadata. The structure depends on the
            engine; see the engine class documentation for details.

        Raises
        ------
        LLMEngineError
            If the path is invalid, the engine does not support the file format,
            or the file is too large for the provider.

        Notes
        -----
        - Not all engines support file attachments. Check your engine's
          documentation (e.g., ``OpenAIEngine``, ``AnthropicEngine``).
        - Some engines may have format or size restrictions.
        - Multiple calls with the same path are idempotent if the file hasn't changed.
        """
        return self._llm_engine.attach(path)

    def detach(self, path: str) -> bool:
        """
        Detach a previously attached file from this Agent.

        Delegates to the underlying engine's detach logic, which performs
        provider-specific cleanup if needed.

        Parameters
        ----------
        path : str
            The local filesystem path to detach.

        Returns
        -------
        bool
            ``True`` if the path was attached and has been removed;
            ``False`` if the path was not in the attachments.
        """
        return self._llm_engine.detach(path)

    def clear_attachments(self) -> None:
        """
        Remove all currently attached files from this Agent.

        Delegates to the underlying engine to detach all paths and perform
        any necessary provider-specific cleanup.
        """
        return self.llm_engine.clear_attachments()

    def clear_memory(self) -> None:
        """Clear the stored turn history."""
        self._history.clear()

    async def async_invoke(self, inputs: Mapping[str, Any]) -> Any:
        """Async analog of `Agent.invoke`.

        This version awaits async-capable pre/post tools and the engine instead of pushing
        the entire sync invoke path into a worker thread. It follows the same memory
        pipeline as `invoke`: build messages, get a raw response plus metadata, run
        post-processing, and commit a canonical turn if `context_enabled=True`.

        Concurrent calls to the same stateful agent instance may interleave unless the
        caller serializes them externally or the class is later configured with an async
        invoke lock.
        """
        logger.info(f"[Async {self.full_name} started]")

        inputs = self.filter_inputs(inputs)

        try:
            logger.debug(f"Agent.{self.name}.pre_invoke preprocessing inputs asynchronously")
            prompt = await self._pre_invoke.async_invoke(inputs)
        except ToolInvocationError:
            raise
        except Exception as e:  # pragma: no cover
            raise AgentInvocationError(f"pre_invoke Tool failed: {e}") from e

        if not isinstance(prompt, str):
            raise AgentInvocationError(
                f"pre_invoke returned non-string (type={type(prompt)!r}); a prompt string is required"
            )

        logger.debug(f"Agent.{self.name} building messages for class '{type(self).__name__}'")
        messages = self.build_messages(prompt)

        logger.debug(f"Agent.{self.name} performing async logic for class '{type(self).__name__}'")
        raw_result, turn_metadata = await self._ainvoke(messages=messages)

        if not isinstance(turn_metadata, Mapping):
            raise AgentInvocationError(
                f"_ainvoke returned non-mapping metadata (type={type(turn_metadata)!r})"
            )

        try:
            logger.debug(f"Agent.{self.name}.post_invoke postprocessing result asynchronously")
            result = await self._post_invoke.async_invoke({self._post_param_name: raw_result})
        except ToolInvocationError:
            raise
        except Exception as e:  # pragma: no cover
            raise AgentInvocationError(f"post_invoke Tool failed: {e}") from e

        if self._context_enabled:
            logger.debug(f"Agent.{self.name} updating history")
            turn = self._make_turn(
                prompt=prompt,
                raw_response=raw_result,
                final_response=result,
                **dict(turn_metadata),
            )
            self._history.append(turn)

        logger.info(f"[Async {self.full_name} finished]")

        return result

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
           returns the raw result plus turn metadata for this run.
        5) Run ``post_invoke`` on the raw result to obtain the final output.
        6) If ``context_enabled`` is True, construct and commit the newest turn
           into persistent history.
        7) Return the final output.

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
            logger.info(f"[{self.full_name} started]")
            # Filter inputs
            inputs = self.filter_inputs(inputs)
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

            # Build messages
            logger.debug(f"Agent.{self.name} building messages for class '{type(self).__name__}'")
            messages = self.build_messages(prompt)

            # Invoke core logic
            logger.debug(f"Agent.{self.name} performing logic for class '{type(self).__name__}'")
            raw_result, turn_metadata = self._invoke(messages=messages)

            if not isinstance(turn_metadata, Mapping):
                raise AgentInvocationError(
                    f"_invoke returned non-mapping metadata (type={type(turn_metadata)!r})"
                )

            # Postprocess raw result
            try:
                logger.debug(f"Agent.{self.name}.post_invoke postprocessing result")
                result = self._post_invoke.invoke({self._post_param_name: raw_result})
            except ToolInvocationError:
                raise
            except Exception as e:  # pragma: no cover
                raise AgentInvocationError(f"post_invoke Tool failed: {e}") from e

            # Update history if enabled
            if self._context_enabled:
                logger.debug(f"Agent.{self.name} updating history")
                turn = self._make_turn(
                    prompt=prompt,
                    raw_response=raw_result,
                    final_response=result,
                    **dict(turn_metadata),
                )
                self._history.append(turn)

            # Final logging and return
            logger.info(f"[{self.full_name} finished]")
            return result

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        """Return a minimal diagnostic snapshot of this agent.

        The `history` field is a rendered compatibility view. The `turn_history` field is
        the canonical stored turn representation.
        """
        rendered_history: List[Dict[str, str]] = []
        for turn in self._history:
            rendered_history.extend(self.render_turn(turn))

        d = super().to_dict()
        d.update({
            "role_prompt": self.role_prompt,
            "pre_invoke": self.pre_invoke.to_dict(),
            "post_invoke": self.post_invoke.to_dict(),
            "llm": self._llm_engine.to_dict(),
            "context_enabled": self.context_enabled,
            "history_window": self.history_window,
            "response_preview_limit": self.response_preview_limit,
            "assistant_response_source": self.assistant_response_source,
            "history": rendered_history,
            "turn_history": [turn.to_dict() for turn in self._history],
        })
        return d
