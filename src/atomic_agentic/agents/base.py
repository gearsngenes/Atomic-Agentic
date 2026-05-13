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

    An Agent is a stateful software unit that points to an LLM engine and carries
    a persona/system role prompt. It accepts a single input mapping and uses a
    pre-invoke Tool to convert a subset of that mapping into a prompt string
    before invoking the engine.

    Core behavior
    -------------
    ``invoke(inputs: Mapping[str, Any]) -> Any`` follows this lifecycle:

    1) Filter the caller-provided input mapping through the Agent's composed
       ``AtomicInvokable`` input contract.
    2) Split the filtered mapping into:
       - ``pre_invoke`` inputs, used to build the prompt.
       - post-invoke passthrough inputs, copied by configured name.
    3) ``pre_invoke.invoke(pre_inputs) -> str``.
    4) Build messages from role prompt, rendered prior turns, and current prompt.
    5) Delegate to ``_invoke(messages)`` to obtain a raw response plus turn metadata.
    6) Assemble post-invoke inputs from:
       - the raw response under ``post_result_key``.
       - configured passthrough inputs.
    7) ``post_invoke.invoke(post_inputs) -> final output``.
    8) Record a canonical turn if ``context_enabled``; return the final output.

    Inputs and schema
    -----------------
    Inputs are always mapping-shaped. The Agent-facing parameter schema is
    composed from:

    - all ``pre_invoke`` parameters; plus
    - post-only passthrough parameters explicitly named in ``passthrough_inputs``.

    If a passthrough name exists in both ``pre_invoke`` and ``post_invoke``, the
    ``pre_invoke`` parameter owns the Agent-facing schema/default. This keeps the
    prompt-building view and post-processing view of the same input value
    consistent.

    Type annotations are descriptive metadata for introspection. The Agent
    validates routing names and structural parameter shape, but it does not try
    to enforce semantic type compatibility between pre/post annotations. Runtime
    value validation belongs to the Tools/callables that consume those values.

    History and context
    -------------------
    - The Agent keeps an in-memory history of ``AgentTurn`` objects.
    - ``history_window`` controls how many turns from the tail of stored history
      are rendered into future LLM-facing messages.
    - Stored history is append-only; no trimming or summarization is performed by
      default.
    - ``history`` is currently a deprecated rendered compatibility view.
    - ``turn_history`` is the canonical stored turn view.

    Parameters
    ----------
    name : str
        Logical name for this agent.
    description : str
        Short, human-readable description.
    llm_engine : LLMEngine
        Engine used to perform the model call. Must be an instance of
        ``LLMEngine``.
    filter_extraneous_inputs : Optional[bool], default None
        Agent-level filtering policy. If None, inherits from ``pre_invoke``.
    role_prompt : Optional[str], default None
        Optional system persona. If None or empty, a default assistant persona is
        used.
    context_enabled : bool, default True
        If True, the agent renders prior turns and records new turns. If False,
        no prior turns are sent and no new turns are recorded.
    pre_invoke : Optional[Tool or Callable], default None
        Tool that converts pre-invoke inputs into a prompt string. If None, a
        strict identity Tool is used that accepts ``{"prompt": str}``.
    post_invoke : Optional[Tool or Callable], default None
        Tool that converts the raw result from ``_invoke`` plus configured
        passthrough inputs into the final return value.
    post_result_key : Optional[str], default None
        Name of the post-invoke parameter that receives the raw ``_invoke``
        result. If None, defaults to the first declared post-invoke parameter.
        The result key may name any declared post-invoke parameter, including a
        variadic parameter; post-invoke binding owns shape validation.
    passthrough_inputs : Optional[list[str]], default None
        Post-invoke parameter names to expose as Agent inputs and pass through by
        name. Names must refer to post-invoke parameters and must not include
        ``post_result_key``. Post-only passthrough parameters are grafted into
        the Agent schema as keyword-only parameters.
    history_window : Optional[int], default None
        Send-window measured in turns. None sends all stored turns; 0 sends no
        prior turns. Stored history is never trimmed.
    response_preview_limit : Optional[int], default None
        Optional character limit applied only when rendering stored assistant
        responses into future LLM-facing message history. Stored turn values are
        not mutated.
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
    history_window : Optional[int] (read-write)
    history : List[Dict[str, str]] (deprecated rendered message view)
    turn_history : List[AgentTurn] (read-only canonical turn view)
    attachments : Dict[str, Dict[str, Any]] (read-only view)
    pre_invoke : Tool (read-only lifecycle reference)
    post_invoke : Tool (read-only lifecycle reference)
    post_result_key : str (read-only)
    passthrough_inputs : List[str] (read-only copy)
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
        if pre_invoke is None:
            pre_tool = identity_pre_tool
        else:
            pre_tool = toolify(
                pre_invoke,
                name="pre_invoke",
                namespace=name,
                description=(
                    f"The tool that preprocesses inputs into a string for Agent {name}"
                ),
            )

        if pre_tool.return_type.lower() not in {"any", "str"}:
            raise AgentError(
                "Agent.pre_invoke must return a type 'str'|'any' after updating pre_invoke"
            )

        # Prepare post_invoke Tool, passthrough config, and composed Agent schema.
        post_tool, resolved_post_result_key, resolved_passthrough_inputs, agent_parameters = (
            self._prepare_agent_lifecycle_config(
                post_invoke=post_invoke,
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

        # history_window: strict int semantics (>= 0). None means send-all mode.
        if history_window is not None and (not type(history_window) is int or history_window < 0):
            raise AgentError("history_window must be an int >= 0 or be 'None'.")
        self._history_window: Optional[int] = history_window

        # Stored turn history.
        # We never trim storage; we only limit what we send to the engine.
        self._history: List[AgentTurn] = []

        # Store history-rendering controls.
        self.response_preview_limit = response_preview_limit
        self.assistant_response_source = assistant_response_source

        resolved_filter_extraneous_inputs = (
            filter_extraneous_inputs
            if filter_extraneous_inputs is not None
            else pre_tool.filter_extraneous_inputs
        )

        # Delegate to parent with the composed Agent schema.
        super().__init__(
            name=name,
            description=description,
            parameters=agent_parameters,
            return_type=self._post_invoke.return_type,
            filter_extraneous_inputs=resolved_filter_extraneous_inputs,)

    # ------------------------------------------------------------------ #
    # agent lifecycle configuration and validation
    # ------------------------------------------------------------------ #
    @classmethod
    def _prepare_post_invoke_tool(
        cls,
        *,
        candidate: Optional[Union[Callable, AtomicInvokable]],
        agent_name: str,
    ) -> Tool:
        """
        Normalize the configured post-invoke component into a Tool.

        This helper owns only post-invoke Tool preparation:

        - If ``candidate`` is None, the shared identity post-invoke Tool is used.
        - Otherwise, ``candidate`` is normalized through ``toolify(...)``.
        - The resulting Tool must expose at least one parameter, because one
          declared parameter must receive the raw result from ``_invoke(...)``.

        It intentionally does not resolve ``post_result_key`` or validate
        passthrough routing. Those concerns are handled by the routing helpers so
        construction-time lifecycle preparation can be read as a sequence of
        small, explicit steps.

        Parameters
        ----------
        candidate : Optional[Union[Callable, AtomicInvokable]]
            User-provided post-invoke component, or None for the default identity
            post-invoke Tool.
        agent_name : str
            Agent name used as the namespace when wrapping a plain callable or
            invokable into a Tool.

        Returns
        -------
        Tool
            Prepared post-invoke Tool.

        Raises
        ------
        AgentError
            If the prepared post-invoke Tool has no declared parameters.
        """
        if candidate is None:
            post_tool = identity_post_tool
        else:
            post_tool = toolify(
                candidate,
                name="post_invoke",
                namespace=agent_name,
                description=f"The tool that postprocesses outputs of Agent {agent_name}",
            )

        if len(post_tool.parameters) == 0:
            raise AgentError("Agent.post_invoke must expect at least 1 argument")

        return post_tool

    @staticmethod
    def _normalize_passthrough_inputs(
        passthrough_inputs: Optional[list[str]],
    ) -> tuple[str, ...]:
        """
        Normalize configured post-invoke passthrough input names.

        ``passthrough_inputs`` is an explicit list of post-invoke parameter names
        that should also be exposed on the Agent's input contract and copied from
        the filtered Agent input mapping into ``post_invoke``.

        This helper performs only list/name normalization:

        - None becomes an empty tuple.
        - The value must otherwise be a list of strings.
        - Names are stripped.
        - Empty names are rejected.
        - Duplicate names after stripping are rejected.

        It intentionally does not check whether the names exist on
        ``post_invoke``. That belongs to routing validation, after the post Tool
        and its parameter map are known.

        Parameters
        ----------
        passthrough_inputs : Optional[list[str]]
            User-provided passthrough input names.

        Returns
        -------
        tuple[str, ...]
            Normalized passthrough names.

        Raises
        ------
        AgentError
            If the value is not None or list[str], if any name is empty, or if
            duplicate normalized names are present.
        """
        if passthrough_inputs is None:
            return ()

        if not isinstance(passthrough_inputs, list):
            raise AgentError("passthrough_inputs must be a list of strings or None.")

        normalized: list[str] = []
        seen: set[str] = set()
        duplicates: set[str] = set()

        for index, name in enumerate(passthrough_inputs):
            if not isinstance(name, str) or not name.strip():
                raise AgentError(
                    f"passthrough_inputs[{index}] must be a non-empty string."
                )

            cleaned_name = name.strip()
            if cleaned_name in seen:
                duplicates.add(cleaned_name)
            else:
                seen.add(cleaned_name)

            normalized.append(cleaned_name)

        if duplicates:
            raise AgentError(
                "passthrough_inputs must not contain duplicate names; "
                f"got {sorted(duplicates)!r}."
            )

        return tuple(normalized)

    @staticmethod
    def _resolve_post_result_key(
        *,
        post_result_key: Optional[str],
        post_params: list[ParamSpec],
    ) -> str:
        """
        Resolve the post-invoke parameter that receives the raw ``_invoke`` result.

        If ``post_result_key`` is None, the first declared post-invoke parameter
        is used. If provided, the key must be a non-empty string after stripping.

        This helper only resolves and normalizes the key. It does not check
        whether the resolved name exists in ``post_params``; that is handled by
        ``_validate_post_routing_contract``.

        A resolved result key may name any declared post-invoke parameter,
        including a variadic parameter. The Agent's responsibility is only to
        route the raw result under the configured key; the post-invoke Tool owns
        binding and shape validation.

        Parameters
        ----------
        post_result_key : Optional[str]
            User-provided result parameter name, or None to use the first
            post-invoke parameter.
        post_params : list[ParamSpec]
            Declared post-invoke parameters.

        Returns
        -------
        str
            Resolved result key.

        Raises
        ------
        AgentError
            If ``post_params`` is empty, or if an explicit ``post_result_key`` is
            not a non-empty string.
        """
        if not post_params:
            raise AgentError("Agent.post_invoke must expect at least 1 argument")

        if post_result_key is None:
            return post_params[0].name

        if not isinstance(post_result_key, str) or not post_result_key.strip():
            raise AgentError("post_result_key must be None or a non-empty string.")

        return post_result_key.strip()

    @staticmethod
    def _validate_post_routing_contract(
        *,
        post_result_key: str,
        passthrough_inputs: tuple[str, ...],
        post_params: list[ParamSpec],
    ) -> None:
        """
        Validate the name-level routing contract into ``post_invoke``.

        This helper validates whether the Agent can route values into the
        configured post-invoke Tool in principle:

        - ``post_result_key`` must name a declared post-invoke parameter.
        - ``post_result_key`` must not also be configured as a passthrough input.
        - Every passthrough input must name a declared post-invoke parameter.
        - Every required, non-variadic post-invoke parameter must be reachable
          from either ``post_result_key`` or ``passthrough_inputs``.

        This helper intentionally does not validate whether a required
        passthrough value will actually be present in a particular invocation.
        Runtime value validation remains the responsibility of the post-invoke
        Tool and the callable it wraps.

        Parameters
        ----------
        post_result_key : str
            Resolved post-invoke parameter name that receives the raw
            ``_invoke`` result.
        passthrough_inputs : tuple[str, ...]
            Normalized passthrough input names.
        post_params : list[ParamSpec]
            Declared post-invoke parameters.

        Raises
        ------
        AgentError
            If routing names are unknown, ambiguous, or insufficient to satisfy
            required post-invoke parameters.
        """
        post_param_map = {param.name: param for param in post_params}
        declared_post_param_names = set(post_param_map)

        if post_result_key not in declared_post_param_names:
            raise AgentError(
                "post_result_key must name one of post_invoke's declared parameters; "
                f"got {post_result_key!r}."
            )

        if post_result_key in passthrough_inputs:
            raise AgentError(
                "post_result_key must not be one of the passthrough input names."
            )

        unknown_passthrough_inputs = set(passthrough_inputs) - declared_post_param_names
        if unknown_passthrough_inputs:
            raise AgentError(
                "passthrough_inputs must name post_invoke parameters; "
                f"got unknown passthrough input(s): {sorted(unknown_passthrough_inputs)!r}."
            )

        variadic_kinds = {ParamSpec.VAR_POSITIONAL, ParamSpec.VAR_KEYWORD}
        provided_post_keys = {post_result_key} | set(passthrough_inputs)
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

    @staticmethod
    def _validate_passthrough_parameter_shapes(
        *,
        passthrough_inputs: tuple[str, ...],
        pre_parameters: list[ParamSpec],
        post_params: list[ParamSpec],
    ) -> None:
        """
        Validate structural compatibility for configured passthrough parameters.

        Passthrough validation is intentionally shape-focused rather than
        type-sensitive. The Agent only needs to know whether a passthrough name
        can be routed deterministically between the Agent input mapping and
        ``post_invoke``. It does not enforce semantic compatibility between
        annotation strings.

        Rules
        -----
        - A post-only passthrough parameter must be non-variadic, because the
          Agent grafts post-only passthroughs into its schema as named
          keyword-only inputs.
        - If a passthrough name exists in both ``pre_invoke`` and
          ``post_invoke``, both sides must either be non-variadic or must be the
          same variadic kind.
        - Type strings are descriptive metadata and are not compared here.

        Parameters
        ----------
        passthrough_inputs : tuple[str, ...]
            Normalized passthrough input names.
        pre_parameters : list[ParamSpec]
            Declared pre-invoke parameters.
        post_params : list[ParamSpec]
            Declared post-invoke parameters.

        Raises
        ------
        AgentError
            If a passthrough parameter has an unsupported variadic shape.
        """
        pre_param_map = {param.name: param for param in pre_parameters}
        post_param_map = {param.name: param for param in post_params}
        variadic_kinds = {ParamSpec.VAR_POSITIONAL, ParamSpec.VAR_KEYWORD}

        for name in passthrough_inputs:
            post_param = post_param_map.get(name)
            if post_param is None:
                raise AgentError(
                    "passthrough_inputs must name post_invoke parameters; "
                    f"got unknown passthrough input {name!r}."
                )

            pre_param = pre_param_map.get(name)
            post_is_variadic = post_param.kind in variadic_kinds

            if pre_param is None:
                if post_is_variadic:
                    raise AgentError(
                        "Post-only passthrough inputs must be non-variadic; "
                        f"got {name!r} with kind {post_param.kind!r}."
                    )
                continue

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

    @staticmethod
    def _compose_agent_parameters(
        *,
        pre_parameters: list[ParamSpec],
        post_params: list[ParamSpec],
        passthrough_inputs: tuple[str, ...],
    ) -> list[ParamSpec]:
        """
        Compose the Agent-facing parameter schema from pre and post lifecycle inputs.

        The composed Agent schema is pre-invoke-owned by default:

        - All pre-invoke parameters are retained.
        - Passthrough names that already exist in pre-invoke are not replaced.
          This preserves the pre-invoke ``ParamSpec`` as the Agent-facing
          contract, including its default.
        - Passthrough names that exist only in post-invoke are grafted into the
          Agent schema as keyword-only parameters.
        - Grafted keyword-only passthroughs are inserted before an existing
          ``**kwargs`` parameter if pre-invoke declares one.
        - The final list is reindexed so it satisfies the ``AtomicInvokable``
          parameter contract.

        This helper assumes routing and passthrough shape validation have already
        run. It still raises a clear ``AgentError`` if a passthrough name cannot
        be found in ``post_params`` so misuse fails explicitly.

        Parameters
        ----------
        pre_parameters : list[ParamSpec]
            Declared pre-invoke parameters.
        post_params : list[ParamSpec]
            Declared post-invoke parameters.
        passthrough_inputs : tuple[str, ...]
            Normalized passthrough input names.

        Returns
        -------
        list[ParamSpec]
            Reindexed Agent-facing parameter list.

        Raises
        ------
        AgentError
            If a passthrough name does not exist in post-invoke parameters.
        """
        pre_param_map = {param.name: param for param in pre_parameters}
        post_param_map = {param.name: param for param in post_params}

        composed_parameters = list(pre_parameters)
        grafted_parameters: list[ParamSpec] = []

        for name in passthrough_inputs:
            if name in pre_param_map:
                continue

            post_param = post_param_map.get(name)
            if post_param is None:
                raise AgentError(
                    "passthrough_inputs must name post_invoke parameters; "
                    f"got unknown passthrough input {name!r}."
                )

            grafted_parameters.append(
                ParamSpec(
                    name=post_param.name,
                    index=0,
                    kind=ParamSpec.KEYWORD_ONLY,
                    type=post_param.type,
                    default=post_param.default,
                )
            )

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

        return [
            ParamSpec(
                name=param.name,
                index=index,
                kind=param.kind,
                type=param.type,
                default=param.default,
            )
            for index, param in enumerate(composed_parameters)
        ]

    @classmethod
    def _prepare_agent_lifecycle_config(
        cls,
        *,
        post_invoke: Optional[Union[Callable, AtomicInvokable]],
        agent_name: str,
        pre_parameters: list[ParamSpec],
        post_result_key: Optional[str],
        passthrough_inputs: Optional[list[str]],
    ) -> tuple[Tool, str, tuple[str, ...], list[ParamSpec]]:
        """
        Create and validate post-invoke routing and the composed Agent schema.

        This helper prepares the construction-time Agent lifecycle contract:

        1) Normalize ``post_invoke`` into a Tool.
        2) Normalize configured passthrough input names.
        3) Resolve the post-invoke result key.
        4) Validate the name-level post-routing contract.
        5) Validate passthrough parameter shape.
        6) Compose the Agent-facing parameter list from pre-invoke parameters
           plus post-only passthrough grafts.

        The Agent validates routing names and structural shape, not semantic type
        compatibility. Type annotations remain descriptive metadata; the
        underlying Tools/callables own runtime value validation.

        Returns
        -------
        tuple[Tool, str, tuple[str, ...], list[ParamSpec]]
            - prepared post-invoke Tool
            - resolved post_result_key
            - normalized passthrough input names
            - composed Agent-facing parameters
        """
        post_tool = cls._prepare_post_invoke_tool(
            candidate=post_invoke,
            agent_name=agent_name,
        )
        post_params = post_tool.parameters

        resolved_passthrough_inputs = cls._normalize_passthrough_inputs(
            passthrough_inputs
        )
        resolved_post_result_key = cls._resolve_post_result_key(
            post_result_key=post_result_key,
            post_params=post_params,
        )

        cls._validate_post_routing_contract(
            post_result_key=resolved_post_result_key,
            passthrough_inputs=resolved_passthrough_inputs,
            post_params=post_params,
        )
        cls._validate_passthrough_parameter_shapes(
            passthrough_inputs=resolved_passthrough_inputs,
            pre_parameters=pre_parameters,
            post_params=post_params,
        )

        agent_parameters = cls._compose_agent_parameters(
            pre_parameters=pre_parameters,
            post_params=post_params,
            passthrough_inputs=resolved_passthrough_inputs,
        )

        return (
            post_tool,
            resolved_post_result_key,
            resolved_passthrough_inputs,
            agent_parameters,
        )

    # ------------------------------------------------------------------ #
    # Agent Properties
    # ------------------------------------------------------------------ #
    @property
    def post_result_key(self) -> str:
        """
        Post-invoke parameter name that receives the raw ``_invoke`` result.

        If no explicit key was provided at construction time, this is the first
        declared post-invoke parameter. The key may name any declared
        post-invoke parameter, including a variadic parameter. The Agent only
        routes the raw result under this key; the post-invoke Tool owns binding
        and shape validation.
        """
        return self._post_result_key

    @property
    def passthrough_inputs(self) -> list[str]:
        """
        Post-invoke parameter names accepted as Agent inputs and passed through.

        These names are copied from the filtered Agent input mapping into
        ``post_invoke``. If a passthrough name also exists in ``pre_invoke``,
        the pre-invoke ``ParamSpec`` owns the Agent-facing schema/default.

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
        Tool that converts the raw ``_invoke`` result into the final Agent output.

        At runtime, the Agent calls this Tool with:

        - the raw result under ``post_result_key``; and
        - configured passthrough inputs copied from the filtered Agent input
          mapping.

        This lifecycle reference is configured at construction time and is not
        replaceable through the Agent API.
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
        Split already-filtered Agent inputs into pre and post-passthrough inputs.

        This method assumes ``filter_inputs(...)`` has already run. It
        materializes defaults from the composed Agent schema before splitting.

        This matters for overlapping passthrough names: if the same name exists
        in both ``pre_invoke`` and ``post_invoke``, the pre-invoke ``ParamSpec``
        owns the Agent-facing default, so both pre-processing and
        post-processing receive the same resolved Agent-visible value.

        Returns
        -------
        tuple[dict[str, Any], dict[str, Any]]
            - inputs passed to ``pre_invoke``
            - passthrough inputs later augmented with the raw result and passed
              to ``post_invoke``
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
