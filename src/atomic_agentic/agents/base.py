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
    5) Assemble post-invoke inputs from the raw response and configured passthrough inputs, then pass them through `post_invoke`.
    6) Record a canonical turn if `context_enabled`; return the final output.

    Inputs and schema
    -----------------
    - Inputs are always a mapping (`Mapping[str, Any]`).
    - The schema is derived from the `pre_invoke` Tool plus selected
      `post_invoke` passthrough parameters.
    - The `pre_invoke` Tool validates and normalizes the pre-invoke subset,
      then converts it into a prompt string.

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
        Tool that converts the raw result from `_invoke` and configured passthrough
        inputs into the final return value.
    post_result_key : Optional[str], default None
        Name of the post_invoke parameter that receives the raw `_invoke` result.
        Defaults to the first declared post_invoke parameter.
    passthrough_inputs : Optional[list[str]], default None
        Post-invoke parameter names to expose as Agent inputs and pass through by name.
        Names must refer to post_invoke parameters and must not include post_result_key.
        Post-only passthrough parameters are grafted into the Agent schema as keyword-only parameters.
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
    pre_invoke : Tool (read-only)
    post_invoke : Tool (read-only)
    post_result_key : str (read-only)
    passthrough_inputs : List[str] (read-only)
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
        post_result_key: Optional[str] = None,
        passthrough_inputs: Optional[list[str]] = None,
        history_window: Optional[int] = None,
        response_preview_limit: Optional[int] = None,
        assistant_response_source: Literal["raw", "final"] = "raw",
    ) -> None:

        # Prepare pre_invoke Tool.
        if not pre_invoke:
            pre_tool = identity_pre_tool
        else:
            pre_tool = toolify(pre_invoke,
                            name="pre_invoke",
                            namespace=name,
                            description=f"The tool that preprocesses inputs into a string for Agent {name}")
        if pre_tool.return_type.lower() not in {"any", "str"}:
            raise AgentError("Agent.pre_invoke must return a type 'str'|'any' after updating pre_invoke")

        # Prepare post_invoke Tool, passthrough config, and composed Agent schema.
        post_tool, resolved_post_result_key, resolved_passthrough_inputs, agent_parameters = (
            self._prepare_agent_lifecycle_config(
                candidate=post_invoke,
                agent_name=name,
                pre_parameters=pre_tool.parameters,
                post_result_key=post_result_key,
                passthrough_inputs=passthrough_inputs,
            )
        )

        # Store lifecycle components and post-processing configuration.
        self._pre_invoke = pre_tool
        self._post_invoke = post_tool
        self._post_result_key = resolved_post_result_key
        self._passthrough_inputs = resolved_passthrough_inputs

        # Store Agent runtime configuration.
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

        # Store history-rendering controls.
        self.response_preview_limit = response_preview_limit
        self.assistant_response_source = assistant_response_source

        filter = filter_extraneous_inputs if filter_extraneous_inputs is not None else pre_tool.filter_extraneous_inputs

        # Delegate to parent with the composed Agent schema.
        super().__init__(name=name,
                        description=description,
                        parameters=agent_parameters,
                        return_type=self._post_invoke.return_type,
                        filter_extraneous_inputs=filter,)

    @classmethod
    def _prepare_agent_lifecycle_config(
        cls,
        *,
        candidate: Optional[Union[Callable, AtomicInvokable]],
        agent_name: str,
        pre_parameters: list[ParamSpec],
        post_result_key: Optional[str],
        passthrough_inputs: Optional[list[str]],
    ) -> tuple[Tool, str, tuple[str, ...], list[ParamSpec], str]:
        """
        Create and validate post-invoke configuration and composed Agent parameters.

        post_invoke must have at least one parameter. Required post_invoke
        parameters must be satisfiable by post_result_key, passthrough input
        names, or callable defaults.

        Returns
        -------
        tuple[Tool, str, tuple[str, ...], list[ParamSpec], str]
            - post Tool
            - resolved post_result_key
            - normalized passthrough input names
            - composed Agent parameters
            - legacy post parameter name retained for compatibility
        """
        # Build post_invoke Tool.
        if candidate is None:
            post_tool = identity_post_tool
        else:
            post_tool = toolify(candidate,
                            name="post_invoke",
                            namespace=agent_name,
                            description=f"The tool that postprocesses outputs of Agent {agent_name}")

        post_params = post_tool.parameters
        if len(post_params) == 0:
            raise AgentError("Agent.post_invoke must expect at least 1 argument")

        # Normalize passthrough input names.
        if passthrough_inputs is None:
            resolved_passthrough_inputs: tuple[str, ...] = ()
        elif isinstance(passthrough_inputs, list):
            cleaned_passthrough_inputs: list[str] = []
            for index, name in enumerate(passthrough_inputs):
                if not isinstance(name, str) or not name.strip():
                    raise AgentError(
                        f"passthrough_inputs[{index}] must be a non-empty string."
                    )
                cleaned_passthrough_inputs.append(name.strip())

            duplicate_passthrough_inputs = sorted({
                name
                for name in cleaned_passthrough_inputs
                if cleaned_passthrough_inputs.count(name) > 1
            })
            if duplicate_passthrough_inputs:
                raise AgentError(
                    "passthrough_inputs must not contain duplicate names; "
                    f"got {duplicate_passthrough_inputs!r}."
                )

            resolved_passthrough_inputs = tuple(cleaned_passthrough_inputs)
        else:
            raise AgentError("passthrough_inputs must be a list of strings or None.")

        # Build parameter lookup maps.
        pre_param_map = {param.name: param for param in pre_parameters}
        post_param_map = {param.name: param for param in post_params}
        declared_post_param_names = set(post_param_map)

        # Resolve and validate post_result_key.
        resolved_post_result_key = (
            post_result_key.strip()
            if isinstance(post_result_key, str) and post_result_key.strip()
            else post_params[0].name
        )

        if post_result_key is not None and (
            not isinstance(post_result_key, str) or not post_result_key.strip()
        ):
            raise AgentError("post_result_key must be None or a non-empty string.")

        if resolved_post_result_key not in declared_post_param_names:
            raise AgentError(
                "post_result_key must name one of post_invoke's declared parameters; "
                f"got {resolved_post_result_key!r}."
            )

        if resolved_post_result_key in resolved_passthrough_inputs:
            raise AgentError(
                "post_result_key must not be one of the passthrough input names."
            )

        # Validate passthrough names against post_invoke parameters.
        unknown_passthrough_inputs = set(resolved_passthrough_inputs) - declared_post_param_names
        if unknown_passthrough_inputs:
            raise AgentError(
                "passthrough_inputs must name post_invoke parameters; "
                f"got unknown passthrough input(s): {sorted(unknown_passthrough_inputs)!r}."
            )

        # Validate overlap, variadic compatibility, and type compatibility.
        variadic_kinds = {ParamSpec.VAR_POSITIONAL, ParamSpec.VAR_KEYWORD}
        any_types = {"any", "typing.any"}

        for name in resolved_passthrough_inputs:
            post_param = post_param_map[name]
            pre_param = pre_param_map.get(name)
            post_is_variadic = post_param.kind in variadic_kinds

            # Raise an error if a passthrough input is post-only and variadic.
            if pre_param is None:
                if post_is_variadic:
                    raise AgentError(
                        "Post-only passthrough inputs must be non-variadic; "
                        f"got {name!r} with kind {post_param.kind!r}."
                    )
                continue

            # Raise an error if overlapping fields have incompatible variadic status.
            pre_is_variadic = pre_param.kind in variadic_kinds
            if pre_is_variadic != post_is_variadic:
                raise AgentError(
                    "Overlapping passthrough inputs must both be non-variadic or "
                    "both be the same variadic kind; "
                    f"got {name!r} as {pre_param.kind!r} and {post_param.kind!r}."
                )
            if pre_is_variadic and pre_param.kind != post_param.kind:
                raise AgentError(
                    "Overlapping variadic passthrough inputs must have the same kind; "
                    f"got {name!r} as {pre_param.kind!r} and {post_param.kind!r}."
                )

            # Raise a type compatibility error if both sides declare different
            # concrete types and neither side is generic.
            pre_type = (pre_param.type or "").strip()
            post_type = (post_param.type or "").strip()
            if (
                pre_type
                and post_type
                and pre_type != post_type
                and pre_type.lower() not in any_types
                and post_type.lower() not in any_types
            ):
                raise AgentError(
                    "Overlapping passthrough input type mismatch for "
                    f"{name!r}: pre_invoke has {pre_type!r}, post_invoke has {post_type!r}."
                )

        # Validate required post_invoke parameters are reachable.
        # Variadic parameters are open collectors, not required named inputs.
        provided_post_keys = {resolved_post_result_key} | set(resolved_passthrough_inputs)
        required_post_keys = {
            param.name
            for param in post_params
            if param.kind not in variadic_kinds
            and param.default is NO_VAL
        }
        missing_required = required_post_keys - provided_post_keys
        if missing_required:
            raise AgentError(
                "Agent.post_invoke required parameter(s) are not satisfied by "
                "post_result_key, passthrough_inputs, or defaults: "
                f"{sorted(missing_required)!r}"
            )

        # Compose Agent parameters with keyword-only passthrough grafts.
        composed_parameters = list(pre_parameters)
        grafted_parameters = [
            ParamSpec(
                name=post_param_map[name].name,
                index=0,
                kind=ParamSpec.KEYWORD_ONLY,
                type=post_param_map[name].type,
                default=post_param_map[name].default,
            )
            for name in resolved_passthrough_inputs
            if name not in pre_param_map
        ]

        if grafted_parameters:
            varkw_index = next(
                (
                    index
                    for index, param in enumerate(composed_parameters)
                    if param.kind == ParamSpec.VAR_KEYWORD
                ),
                None,
            )
            if varkw_index is None:
                composed_parameters.extend(grafted_parameters)
            else:
                composed_parameters = (
                    composed_parameters[:varkw_index]
                    + grafted_parameters
                    + composed_parameters[varkw_index:]
                )

        composed_parameters = [
            ParamSpec(
                name=param.name,
                index=index,
                kind=param.kind,
                type=param.type,
                default=param.default,
            )
            for index, param in enumerate(composed_parameters)
        ]

        return (
            post_tool,
            resolved_post_result_key,
            resolved_passthrough_inputs,
            composed_parameters,
        )

    # ------------------------------------------------------------------ #
    # Agent Properties
    # ------------------------------------------------------------------ #
    @property
    def post_result_key(self) -> str:
        """
        Name of the post_invoke parameter that receives the raw _invoke result.
        """
        return self._post_result_key


    @property
    def passthrough_inputs(self) -> list[str]:
        """
        Post-invoke parameter names accepted as Agent inputs and passed through by name.

        A shallow copy is returned to prevent external mutation of Agent state.
        """
        return list(self._passthrough_inputs)

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

        This is configured at construction time. A plain callable provided to the
        constructor is wrapped in a Tool.
        """
        return self._pre_invoke

    @property
    def post_invoke(self) -> Tool:
        """
        Tool that converts the raw result from :meth:`_invoke` into the final
        return value for :meth:`invoke`.

        This Tool is configured with a post_result_key naming the post-invoke
        parameter that receives the raw result. Configured passthrough inputs are
        copied from the filtered Agent input mapping into post_invoke.
        """
        return self._post_invoke

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

    def _split_inputs(
        self,
        inputs: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Split already-filtered Agent inputs into pre_invoke inputs and post_invoke
        passthrough inputs.

        This method assumes `filter_inputs(...)` has already run. It materializes
        defaults from the composed Agent schema before splitting so overlapping
        pre/post passthrough parameters use the Agent-visible default.
        """
        inputs = dict(inputs)

        for param in self.parameters:
            if param.kind in {ParamSpec.VAR_POSITIONAL, ParamSpec.VAR_KEYWORD}:
                continue
            if param.name not in inputs and param.default is not NO_VAL:
                inputs[param.name] = param.default

        pre_param_names = {param.name for param in self._pre_invoke.parameters}
        pre_inputs = {
            key: value
            for key, value in inputs.items()
            if key in pre_param_names
        }

        passthrough_inputs = {
            name: inputs[name]
            for name in self._passthrough_inputs
            if name in inputs
        }

        return pre_inputs, passthrough_inputs

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
        pre_inputs, post_inputs = self._split_inputs(inputs)

        try:
            logger.debug(f"Agent.{self.name}.pre_invoke preprocessing inputs asynchronously")
            prompt = await self._pre_invoke.async_invoke(pre_inputs)
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
            post_inputs[self._post_result_key] = raw_result
            result = await self._post_invoke.async_invoke(post_inputs)
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
        5) Run ``post_invoke`` on the raw result and configured passthrough inputs to obtain the final output.
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
            ``post_invoke`` Tool applied to the raw LLM output and configured
            passthrough inputs.

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
            pre_inputs, post_inputs = self._split_inputs(inputs)

            # Preprocess inputs to prompt string
            try:
                logger.debug(f"Agent.{self.name}.pre_invoke preprocessing inputs")
                prompt = self._pre_invoke.invoke(pre_inputs)
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
                post_inputs[self._post_result_key] = raw_result
                result = self._post_invoke.invoke(post_inputs)
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
            "post_result_key": self.post_result_key,
            "passthrough_inputs": self.passthrough_inputs,
            "llm": self._llm_engine.to_dict(),
            "context_enabled": self.context_enabled,
            "history_window": self.history_window,
            "response_preview_limit": self.response_preview_limit,
            "assistant_response_source": self.assistant_response_source,
            "history": rendered_history,
            "turn_history": [turn.to_dict() for turn in self._history],
        })
        return d
