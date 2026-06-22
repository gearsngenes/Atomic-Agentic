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
)
from dataclasses import replace
from datetime import datetime, timezone
import logging

from ..exceptions import (
    AgentError,
    AgentInvocationError,
    ToolInvocationError,
)
from ..core.Invokable import AtomicInvokable
from ..models.parameters import ParamSpec
from ..constants.core import NO_VAL
from ..engines.LLMEngines import LLMEngine
from ..models.results import AgentResult, LLMModelData
from ..tools import Tool, toolify
from ..models.agents.records import AgentRecord, LLMRecord

logger = logging.getLogger(__name__)

from .tools import identity_pre_tool, identity_post_tool

# ───────────────────────────────────────────────────────────────────────────────
# Agent
# ───────────────────────────────────────────────────────────────────────────────
class Agent(AtomicInvokable):
    """
    Schema-driven LLM Agent.

    An Agent is a stateful software unit that points to an LLM engine and carries
    a system/role prompt. It accepts a single input mapping and uses a
    pre-invoke Tool to convert a subset of that mapping into the current prompt
    string for the protected invocation path.

    Core behavior
    -------------
    ``invoke(inputs: Mapping[str, Any]) -> AgentResult`` follows this lifecycle:

    1) Filter the caller-provided input mapping through the Agent's composed
       ``AtomicInvokable`` input contract.
    2) Split the filtered mapping into:
       - ``pre_invoke`` inputs, used to produce the current prompt.
       - post-invoke passthrough inputs, copied by configured name.
    3) ``pre_invoke.invoke(pre_inputs) -> str``.
    4) Select prior ``AgentRecord`` records according to ``context_enabled`` and
       ``records_window``.
    5) Delegate ``turns`` and ``prompt`` to ``_invoke(...)`` or ``_ainvoke(...)``
       to obtain a 2-tuple of a draft ``AgentRecord`` (``final_result`` is
       ``None`` at this point) and a metadata dict carrying LLM accounting.
    6) The protected invocation path owns any provider-facing message rendering
       it needs, usually through ``build_messages(...)``.
    7) Assemble post-invoke inputs from:
       - the raw response under ``post_result_key``.
       - configured passthrough inputs.
    8) ``post_invoke.invoke(post_inputs) -> final output``.
    9) Construct the ``AgentResult`` via ``make_result(**metadata)``; complete
       the draft record (``final_result`` filled in) and store it if
       ``context_enabled``; return the ``AgentResult``.

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
    - The Agent keeps an in-memory history of ``AgentRecord`` objects.
    - ``records_window`` controls how many turns from the tail of stored history
      are selected for the protected invocation path.
    - Selected turns are rendered into provider-facing messages only when an
      invocation path calls ``build_messages(...)`` or equivalent rendering logic.
    - Stored history is append-only; no trimming or summarization is performed by
      default.
    - ``records`` is the canonical stored turn view.

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
        Optional system prompt. If None or empty, a default assistant prompt is
        used.
    context_enabled : bool, default True
        If True, prior turns are selected for the protected invocation path and
        completed invocations are stored as canonical turns. If False, no prior
        turns are selected and no new turns are recorded.
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
    records_window : Optional[int], default None
        Turn-selection window. None selects all stored turns; 0 selects no prior
        turns. Stored history is never trimmed.
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
    records_window : Optional[int] (read-write)
    records : List[AgentRecord] (read-only canonical turn view)
    attachments : Dict[str, Dict[str, Any]] (read-only view)
    pre_invoke : Tool (read-only lifecycle reference)
    post_invoke : Tool (read-only lifecycle reference)
    post_result_key : str (read-only)
    passthrough_inputs : List[str] (read-only copy)
    """

    DEFAULT_ROLE_PROMPT = "You are a helpful AI assistant"

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
        namespace: str = "default",
        pre_invoke: Optional[AtomicInvokable | Callable] = None,
        post_invoke: Optional[AtomicInvokable | Callable] = None,
        post_result_key: Optional[str] = None,
        passthrough_inputs: Optional[list[str]] = None,
        records_window: Optional[int] = None,
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
        self._role_prompt: str = Agent.DEFAULT_ROLE_PROMPT
        if role_prompt is not None:
            if not isinstance(role_prompt, str):
                raise TypeError(
                    f"role_prompt must be of type 'str' or 'None', but got {type(role_prompt).__name__}"
                )
            cleaned_role_prompt = role_prompt.strip()
            if cleaned_role_prompt:
                self._role_prompt = cleaned_role_prompt
        self._context_enabled: bool = context_enabled

        # records_window: strict int semantics (>= 0). None means select all stored turns.
        if records_window is not None and (not type(records_window) is int or records_window < 0):
            raise AgentError("records_window must be an int >= 0 or be 'None'.")
        self._records_window: Optional[int] = records_window

        # Stored turn history.
        # We never trim storage; we only limit which turns are selected per invocation.
        self._records: List[AgentRecord] = []

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
            namespace=namespace,
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

        # Reserve continue_from as a framework-level input parameter.
        existing_names = {p.name for p in composed_parameters}
        if "continue_from" in existing_names:
            raise AgentError(
                "'continue_from' is a reserved Agent input parameter; "
                "pre_invoke may not declare a parameter with that name."
            )
        _varkw_idx = next(
            (i for i, p in enumerate(composed_parameters) if p.kind == ParamSpec.VAR_KEYWORD),
            None,
        )
        _continue_from_param = ParamSpec(
            name="continue_from",
            index=0,
            kind=ParamSpec.KEYWORD_ONLY,
            type="str | None",
            default=None,
        )
        if _varkw_idx is None:
            composed_parameters.append(_continue_from_param)
        else:
            composed_parameters.insert(_varkw_idx, _continue_from_param)

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
        """Base system prompt supplied when provider-facing messages are rendered."""
        return self._role_prompt

    @role_prompt.setter
    def role_prompt(self, value: Optional[str]) -> None:
        if value is None:
            self._role_prompt = Agent.DEFAULT_ROLE_PROMPT
            return
        if not isinstance(value, str):
            raise TypeError(
                f"role_prompt must be of type 'str' or 'None', but got {type(value).__name__}"
            )
        cleaned = value.strip()
        self._role_prompt = cleaned or Agent.DEFAULT_ROLE_PROMPT

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
        Whether the agent uses turn memory.

        If True, prior turns are selected for the protected invocation path and
        completed invocations are stored as canonical turns. If False, no prior
        turns are selected and no new turns are recorded.
        """
        return self._context_enabled

    @context_enabled.setter
    def context_enabled(self, value: bool) -> None:
        if type(value) is not bool:
            raise ValueError("context_enabled must be a bool.")
        self._context_enabled = value

    @property
    def records_window(self) -> Optional[int]:
        """
        Number of stored turns to select from the tail of turn history.

        None selects all stored turns. 0 selects no prior turns. Stored history is
        never trimmed by this setting. Selected turns may later be rendered into
        provider-facing messages by ``_invoke(...)``, ``_ainvoke(...)``, or
        subclass-specific logic.
        """
        return self._records_window

    @records_window.setter
    def records_window(self, value: Optional[int]) -> None:
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError("records_window must be an int >= 0 or be 'None'.")
        self._records_window = value

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
    def records(self) -> List[AgentRecord]:
        """Return a shallow copy of the stored turn history (never trimmed)."""
        return list(self._records)

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
    def build_messages(self, system_prompt: str, turns: List[AgentRecord], prompt: str) -> List[Dict[str, str]]:
        """Render provider-facing message dicts from canonical turn inputs.

        This method is the default rendering boundary between Agent-native
        memory and LLM-engine-facing chat messages. It does not select turns and
        does not mutate internal history; callers provide the exact turn window
        to render.

        Each supplied turn is rendered through ``render_turn(...)``, allowing
        subclasses to preserve richer canonical turn records while customizing
        their provider-facing representation. The current prompt is appended as
        the final user message.

        Subclasses may call this method multiple times with different system
        prompts, turn windows, or current prompts when a more complex invocation
        requires multiple model calls.
        """
        messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]

        if turns:
            for turn in turns:
                messages.extend(self.render_turn(turn))

        user_msg = {"role": "user", "content": prompt}
        messages.append(user_msg)

        return messages

    def render_turn(self, turn: AgentRecord) -> List[Dict[str, str]]:
        """Render one canonical AgentRecord into LLM-facing messages.

        The assistant content is selected from either ``turn.generated_response``
        or ``turn.final_result.result`` according to ``assistant_response_source``.
        The optional ``response_preview_limit`` is applied only to the rendered
        text; stored turn values are never mutated.

        Subclasses can override this method to preserve richer canonical turn
        records while controlling their provider-facing representation.
        """
        if not isinstance(turn, AgentRecord):
            raise AgentInvocationError(
                f"render_turn expected AgentRecord, got {type(turn)!r}"
            )

        response = (
            turn.generated_response
            if self._assistant_response_source == "raw"
            else turn.final_result.result
        )
        response_text = str(response)

        if (
            self._response_preview_limit is not None
            and len(response_text) > self._response_preview_limit
        ):
            response_text = response_text[:self._response_preview_limit] + "..."

        return [
            {"role": "user", "content": turn.user_prompt},
            {"role": "assistant", "content": response_text},
        ]

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

    async def _ainvoke(self, turns: List[AgentRecord], prompt: str) -> tuple[AgentRecord, dict]:
        """Async internal call path used by ``async_invoke``.

        The protected async contract receives the selected canonical turns plus
        the current prompt string. This base implementation renders those values
        into provider-facing messages with ``build_messages(...)``, delegates to
        the engine's async interface, and returns a 2-tuple of a **draft**
        ``AgentRecord`` and a metadata dict for this invocation. The draft's
        ``final_result`` is ``None`` because post-processing has not run yet.

        The metadata dict carries ``llm_records`` and ``llm_model_data``, which
        ``async_invoke`` passes to ``make_result`` to construct the AgentResult.
        ``async_invoke`` later completes the draft via
        ``dataclasses.replace(draft, final_result=agent_result)``.

        Subclasses may override this method to implement more complex async
        behavior, including delayed rendering, multiple model calls, or alternate
        system prompts. They must not mutate ``self._records`` directly; memory
        is committed by ``async_invoke`` after post-processing has produced the
        final response.
        """
        try:
            logger.debug(f"[Agent - {self.name}]._ainvoke: Invoking LLM asynchronously")
            messages = self.build_messages(self.role_prompt, turns, prompt)
            engine_result = await self._llm_engine.async_invoke({"messages": messages})
            text = engine_result.result
        except Exception as e:  # pragma: no cover - engine-specific failures
            raise AgentInvocationError(f"engine async invocation failed: {e}") from e

        if not isinstance(text, str):
            raise AgentInvocationError(
                f"engine returned non-string (type={type(text)!r}); a string is required"
            )

        # Capture the full LLM envelope so the eventual AgentResult can report
        # model identity and per-generation records without re-deriving them later.
        llm_record = LLMRecord(user_prompt=prompt, llm_result=engine_result)
        draft = AgentRecord(
            user_prompt=prompt,
            generated_response=text,
        )
        metadata: dict = {
            "llm_records": (llm_record,),
            "llm_model_data": engine_result.model_data,
        }
        return draft, metadata

    def _invoke(self, turns: List[AgentRecord], prompt: str) -> tuple[AgentRecord, dict]:
        """Internal call path used by ``invoke``.

        The protected sync contract receives the selected canonical turns plus
        the current prompt string. This base implementation renders those values
        into provider-facing messages with ``build_messages(...)``, delegates to
        the configured LLM engine, and returns a 2-tuple of a **draft**
        ``AgentRecord`` and a metadata dict for this invocation. The draft's
        ``final_result`` is ``None`` because post-processing has not run yet.

        The metadata dict carries ``llm_records`` and ``llm_model_data``, which
        ``invoke`` passes to ``make_result`` to construct the AgentResult.
        ``invoke`` later completes the draft via
        ``dataclasses.replace(draft, final_result=agent_result)``.

        Subclasses may override this method to implement more complex behavior,
        including delayed rendering, multiple model calls, or alternate system
        prompts. They must not mutate ``self._records`` directly; memory is
        committed by ``invoke`` after post-processing has produced the final
        response.
        """
        # 1) Call engine (attachments are managed by the engine itself)
        try:
            logger.debug(f"[Agent - {self.name}]._invoke: Invoking LLM")
            messages = self.build_messages(self.role_prompt, turns, prompt)
            engine_result = self._llm_engine.invoke({"messages": messages})
            text = engine_result.result
        except Exception as e:  # pragma: no cover - engine-specific failures
            raise AgentInvocationError(f"engine invocation failed: {e}") from e

        # 2) Engine contract: base Agent expects a string.
        if not isinstance(text, str):
            raise AgentInvocationError(
                f"engine returned non-string (type={type(text)!r}); a string is required"
            )

        # 3) Capture the full LLM envelope so the eventual AgentResult can report
        #    model identity and per-generation records without re-deriving them later.
        llm_record = LLMRecord(user_prompt=prompt, llm_result=engine_result)

        # 4) Return the draft AgentRecord (final_result=None until make_result runs)
        #    alongside the metadata dict that invoke passes to make_result.
        draft = AgentRecord(
            user_prompt=prompt,
            generated_response=text,
        )
        metadata: dict = {
            "llm_records": (llm_record,),
            "llm_model_data": engine_result.model_data,
        }
        return draft, metadata

    def make_result(
        self,
        result: Any,
        started_at: datetime,
        ended_at: datetime,
        **result_kwargs: Any,
    ) -> AgentResult:
        """
        Construct this Agent's ``AgentResult`` envelope.

        ``result`` is the caller-facing post-processed payload. LLM accounting
        and model context gathered during the invocation are passed via
        ``result_kwargs`` from the metadata dict returned by ``_invoke``.
        UUID minting is handled by ``AtomicResult.__post_init__`` automatically
        when ``run_id`` is not provided.
        """
        unexpected = set(result_kwargs) - {"llm_records", "llm_model_data"}
        if unexpected:
            raise AgentInvocationError(
                f"make_result: unexpected result kwarg(s): {sorted(unexpected)!r}."
            )

        llm_records = result_kwargs.get("llm_records")
        llm_model_data = result_kwargs.get("llm_model_data")

        if (
            not isinstance(llm_records, tuple)
            or not llm_records
            or not all(isinstance(r, LLMRecord) for r in llm_records)
        ):
            raise AgentInvocationError(
                "Agent.make_result: llm_records must be a non-empty tuple of LLMRecord instances."
            )

        if not isinstance(llm_model_data, LLMModelData):
            raise AgentInvocationError(
                "Agent.make_result: llm_model_data must be an LLMModelData instance."
            )

        # Derive per-call token usage from the validated records.
        # Entries where token_usage is None (provider did not report) are omitted.
        llm_token_usage = tuple(
            r.llm_result.token_usage
            for r in llm_records
            if r.llm_result.token_usage is not None
        )

        return self._make_result(
            result=result,
            started_at=started_at,
            ended_at=ended_at,
            result_cls=AgentResult,
            llm_token_usage=llm_token_usage,
            llm_model_data=llm_model_data,
        )

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
        self._records.clear()

    def get_conversation(
        self,
        run_id: str | None = None,
        turns: int | None = None,
    ) -> list[AgentRecord]:
        """Walk the ``prev`` chain backward from a target record and return it oldest-first.

        This is the canonical entry point for branch-aware turn selection. Linear
        history (no branching) degenerates to a reversed tail-slice. Branched
        history returns only the records on the chain leading to the target record,
        not siblings committed via a different ``continue_from`` invocation.

        Parameters
        ----------
        run_id:
            ``run_id`` of the target record to start the walk from. If ``None``,
            the walk starts from the most recently committed record. If provided
            and not found in history, ``AgentInvocationError`` is raised.
        turns:
            Maximum number of records to return. ``None`` means return the full
            chain from the target record back to the root. ``0`` is not valid and
            always raises ``ValueError`` before any history check.

        Returns
        -------
        list[AgentRecord]
            Records in oldest-first order (i.e. ``list[0]`` is the chain root or
            earliest record within the requested window).

        Raises
        ------
        ValueError
            If ``turns == 0``.
        AgentInvocationError
            If ``run_id`` is provided and no record with that ``run_id`` exists
            in history.
        """
        # 1. turns=0 is always invalid.
        if turns == 0:
            raise ValueError(
                "get_conversation: turns must be a positive integer or None; "
                "0 is not valid (the method always returns at least the target record)."
            )

        # 2. Empty history with no specific run_id → nothing to walk.
        if not self._records:
            return []

        # 3. Resolve the starting record.
        if run_id is None:
            start = self._records[-1]
        else:
            start = next(
                (r for r in self._records if r.final_result.run_id == run_id),
                None,
            )
            if start is None:
                raise AgentInvocationError(
                    f"get_conversation: no record with run_id {run_id!r} "
                    "found in agent history."
                )

        # 4. Walk the prev chain backward, collecting up to `turns` records.
        chain: list[AgentRecord] = []
        current: AgentRecord | None = start
        while current is not None:
            chain.append(current)
            if turns is not None and len(chain) >= turns:
                break
            current = current.prev

        # 5. Reverse to oldest-first order and return.
        chain.reverse()
        return chain

    async def async_invoke(self, inputs: Mapping[str, Any]) -> AgentResult:
        """Async analog of ``Agent.invoke``.

        This version awaits async-capable pre/post tools and the engine instead
        of pushing the entire sync invoke path into a worker thread. It follows
        the same lifecycle as ``invoke``:

        1) Filter and split inputs.
        2) Run ``pre_invoke`` to produce the current prompt string.
        3) Select the appropriate turn window according to ``records_window`` and
           ``context_enabled``.
        4) Delegate ``turns`` and ``prompt`` to ``_ainvoke(...)``, which owns any
           provider-facing rendering and async generation work, and returns a
           2-tuple of a draft ``AgentRecord`` (``final_result`` still ``None``)
           and a metadata dict.
        5) Run ``post_invoke`` on the draft's generated response and configured
           passthrough inputs to obtain the final response.
        6) Construct the ``AgentResult`` via ``make_result(**metadata)``.
        7) Complete the draft via ``dataclasses.replace(draft, final_result=agent_result)``
           and commit it to history if ``context_enabled=True``.

        Concurrent calls to the same stateful agent instance may interleave unless the
        caller serializes them externally or the class is later configured with an async
        invoke lock.
        """
        logger.info(f"[Async {self.full_name} started]")

        # Capture the invocation span up front so AgentResult.elapsed_s reflects
        # the full pre -> LLM -> post pipeline, mirroring LLMEngine.invoke.
        started_at = datetime.now(timezone.utc)

        inputs = self.filter_inputs(inputs)
        # Site A: pop continue_from before _split_inputs so it is not forwarded
        # to pre_invoke or post_invoke.
        continue_from = inputs.pop("continue_from", None)
        pre_inputs, post_inputs = self._split_inputs(inputs)

        try:
            logger.debug(f"Agent.{self.name}.pre_invoke preprocessing inputs asynchronously")
            pre_result = await self._pre_invoke.async_invoke(pre_inputs)
            prompt = pre_result.result
        except ToolInvocationError:
            raise
        except Exception as e:  # pragma: no cover
            raise AgentInvocationError(f"pre_invoke Tool failed: {e}") from e

        if not isinstance(prompt, str):
            raise AgentInvocationError(
                f"pre_invoke returned non-string (type={type(prompt)!r}); a prompt string is required"
            )

        # Site B: chain-aware turn selection replacing the inline tail-slice.
        logger.debug(f"Agent.{self.name} selecting turns for class '{type(self).__name__}'")
        turns: list[AgentRecord] = []
        if self._context_enabled and continue_from != "new":
            if self._records_window != 0:
                turns = self.get_conversation(
                    run_id=continue_from,
                    turns=self._records_window,
                )

        # Delegate selected turns and current prompt to the protected core logic.
        # Returns a 2-tuple: draft AgentRecord (final_result=None) + metadata dict.
        logger.debug(f"Agent.{self.name} performing async logic for class '{type(self).__name__}'")
        draft, metadata = await self._ainvoke(turns=turns, prompt=prompt)

        if not isinstance(draft, AgentRecord):
            raise AgentInvocationError(
                f"_ainvoke returned non-AgentRecord draft (type={type(draft)!r})"
            )

        try:
            logger.debug(f"Agent.{self.name}.post_invoke postprocessing result asynchronously")
            post_inputs[self._post_result_key] = draft.generated_response
            post_result = await self._post_invoke.async_invoke(post_inputs)
        except ToolInvocationError:
            raise
        except Exception as e:  # pragma: no cover
            raise AgentInvocationError(f"post_invoke Tool failed: {e}") from e

        final_response = post_result.result

        # The invocation is complete once post-processing has produced the final
        # response; this is the natural "ended_at" boundary for the AgentResult.
        ended_at = datetime.now(timezone.utc)

        agent_result = self.make_result(
            result=final_response,
            started_at=started_at,
            ended_at=ended_at,
            **metadata,
        )

        # Complete and store the canonical record only when memory is kept.
        # Site C: stamp prev with the most recent turn used as context.
        if self._context_enabled:
            logger.debug(f"Agent.{self.name} updating history")
            record = replace(
                draft,
                final_result=agent_result,
                llm_records=metadata["llm_records"],
                prev=turns[-1] if turns else None,
            )
            self._records.append(record)

        logger.info(f"[Async {self.full_name} finished]")

        return agent_result

    def invoke(self, inputs: Mapping[str, Any]) -> AgentResult:
        """Invoke the Agent with a single input mapping.

        Steps
        -----
        1) Filter the caller-provided mapping through the Agent's input contract.
        2) Split filtered inputs into ``pre_invoke`` inputs and post-invoke
           passthrough inputs.
        3) Run ``pre_invoke(pre_inputs)`` to produce the current prompt string.
           If the Tool raises ``ToolInvocationError``, it propagates unchanged.
           Other exceptions are wrapped as ``AgentInvocationError``.
        4) Select the appropriate turn window according to ``records_window`` and
           ``context_enabled``.
        5) Delegate ``turns`` and ``prompt`` to ``_invoke(...)``, which owns any
           provider-facing rendering and sync generation work, and returns a
           2-tuple of a draft ``AgentRecord`` (``final_result`` still ``None``)
           and a metadata dict.
        6) Run ``post_invoke`` on the draft's generated response and configured
           passthrough inputs to obtain the final response.
        7) Construct the ``AgentResult`` via ``make_result(**metadata)``.
        8) Complete the draft via ``dataclasses.replace(draft, final_result=agent_result)``
           and commit it to history if ``context_enabled`` is True.

        Parameters
        ----------
        inputs : Mapping[str, Any]
            Input mapping to be adapted to a prompt string via ``pre_invoke``.

        Returns
        -------
        AgentResult
            This invocation's successful-invocation envelope: the post-processed
            output plus run identity, timing, and the LLM activity gathered while
            producing it.

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

            # Capture the invocation span up front so AgentResult.elapsed_s
            # reflects the full pre -> LLM -> post pipeline, mirroring
            # LLMEngine.invoke.
            started_at = datetime.now(timezone.utc)

            # Filter inputs.
            inputs = self.filter_inputs(inputs)
            # Site A: pop continue_from before _split_inputs so it is not
            # forwarded to pre_invoke or post_invoke.
            continue_from = inputs.pop("continue_from", None)
            pre_inputs, post_inputs = self._split_inputs(inputs)

            # Preprocess inputs to prompt string.
            try:
                logger.debug(f"Agent.{self.name}.pre_invoke preprocessing inputs")
                pre_result = self._pre_invoke.invoke(pre_inputs)
                prompt = pre_result.result
            except ToolInvocationError:
                raise
            except Exception as e:  # pragma: no cover
                raise AgentInvocationError(f"pre_invoke Tool failed: {e}") from e

            if not isinstance(prompt, str):
                raise AgentInvocationError(
                    f"pre_invoke returned non-string (type={type(prompt)!r}); a prompt string is required"
                )

            # Site B: chain-aware turn selection replacing the inline tail-slice.
            logger.debug(f"Agent.{self.name} selecting turns for class '{type(self).__name__}'")
            turns: list[AgentRecord] = []
            if self._context_enabled and continue_from != "new":
                if self._records_window != 0:
                    turns = self.get_conversation(
                        run_id=continue_from,
                        turns=self._records_window,
                    )

            # Delegate selected turns and current prompt to the protected core
            # logic. Returns a 2-tuple: draft AgentRecord (final_result=None)
            # + metadata dict that make_result unpacks.
            logger.debug(f"Agent.{self.name} performing logic for class '{type(self).__name__}'")
            draft, metadata = self._invoke(turns=turns, prompt=prompt)

            if not isinstance(draft, AgentRecord):
                raise AgentInvocationError(
                    f"_invoke returned non-AgentRecord draft (type={type(draft)!r})"
                )

            # Postprocess raw result.
            try:
                logger.debug(f"Agent.{self.name}.post_invoke postprocessing result")
                post_inputs[self._post_result_key] = draft.generated_response
                post_result = self._post_invoke.invoke(post_inputs)
            except ToolInvocationError:
                raise
            except Exception as e:  # pragma: no cover
                raise AgentInvocationError(f"post_invoke Tool failed: {e}") from e

            final_response = post_result.result

            # The invocation is complete once post-processing has produced the
            # final response; this is the natural "ended_at" boundary for the
            # AgentResult.
            ended_at = datetime.now(timezone.utc)

            agent_result = self.make_result(
                result=final_response,
                started_at=started_at,
                ended_at=ended_at,
                **metadata,
            )

            # Complete and store the canonical record only when memory is kept.
            # Site C: stamp prev with the most recent turn used as context.
            if self._context_enabled:
                logger.debug(f"Agent.{self.name} updating history")
                record = replace(
                    draft,
                    final_result=agent_result,
                    llm_records=metadata["llm_records"],
                    prev=turns[-1] if turns else None,
                )
                self._records.append(record)

            # Final logging.
            logger.info(f"[{self.full_name} finished]")

            return agent_result

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        """Return a minimal diagnostic snapshot of this agent."""
        d = super().to_dict()
        d.update({
            "role_prompt": self.role_prompt,
            "pre_invoke": self.pre_invoke.to_dict(),
            "post_invoke": self.post_invoke.to_dict(),
            "post_result_key": self.post_result_key,
            "passthrough_inputs": self.passthrough_inputs,
            "llm": self._llm_engine.to_dict(),
            "context_enabled": self.context_enabled,
            "records_window": self.records_window,
            "response_preview_limit": self.response_preview_limit,
            "assistant_response_source": self.assistant_response_source,
            "records": [turn.to_dict() for turn in self._records],
        })
        return d
