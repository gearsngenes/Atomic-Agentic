from __future__ import annotations

from abc import ABC, abstractmethod
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
import warnings

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
from ..models.agents.prompts import PromptConfig

logger = logging.getLogger(__name__)

from .tools import identity_pre_tool, identity_post_tool
from ..constants.agents import RUN_ID_PARAM

# ───────────────────────────────────────────────────────────────────────────────
# Agent
# ───────────────────────────────────────────────────────────────────────────────
class Agent(AtomicInvokable, ABC):
    """
    Abstract base for schema-driven LLM agents.

    ``Agent`` owns the full invocation lifecycle shell — input filtering,
    context extraction, turn selection, pre/post-invoke dispatch, record
    management — but delegates the actual LLM work to the ``_invoke`` /
    ``_ainvoke`` abstract methods that concrete subclasses implement.

    Lifecycle (four-tier input model)
    ----------------------------------
    ``invoke(inputs)`` follows this sequence:

    1. ``filter_inputs`` collects declared keys and injects defaults.
    2. Framework-reserved args (``run_id``) are popped.
    3. ``_build_context`` extracts declared context keys (and any
       instance-state values added by a subclass override) into ``context``.
    4. Remaining inputs are sliced into ``pre_inputs`` and ``post_inputs``.
    5. Turns are selected from history if ``context_enabled`` is True.
    6. ``pre_invoke`` converts ``pre_inputs`` to a prompt string.
    7. ``_invoke(turns, prompt, context)`` performs LLM work.
    8. ``post_invoke`` transforms the raw response to the final result.
    9. A completed ``AgentRecord`` is always appended to ``_records``.

    Schema composition
    ------------------
    The agent's parameter schema is composed at construction time from:

    - All ``pre_invoke`` parameters.
    - Post-only non-result non-variadic parameters, grafted as KEYWORD_ONLY.
    - ``context_keys`` parameters, grafted as KEYWORD_ONLY.
    - ``run_id`` (KEYWORD_ONLY, default None).

    ``context_enabled``
    -------------------
    ``True``:  ``get_conversation`` selects prior turns for each invocation.
    ``False``: turns are always ``[]``; ``run_id`` is ignored.
    Records are appended unconditionally regardless of this setting.
    """

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        name: str,
        namespace: str,
        description: str,
        llm_engine: LLMEngine,
        filter_extraneous_inputs: Optional[bool] = None,
        context_enabled: bool = True,
        *,
        pre_invoke: Optional[AtomicInvokable | Callable] = None,
        post_invoke: Optional[AtomicInvokable | Callable] = None,
        post_result_key: Optional[str] = None,
        context_keys: list[str] | list[ParamSpec] | None = None,
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
                description=f"The tool that preprocesses inputs into a string for Agent {name}",
            )

        if pre_tool.return_type.lower() not in {"any", "str"}:
            raise AgentError(
                "Agent.pre_invoke must return a type 'str'|'any' after updating pre_invoke"
            )

        # Prepare post_invoke Tool, context_keys, and composed Agent schema.
        post_tool, resolved_post_result_key, context_key_params, agent_parameters = (
            self._prepare_agent_lifecycle_config(
                post_invoke=post_invoke,
                agent_name=name,
                pre_parameters=pre_tool.parameters,
                post_result_key=post_result_key,
                context_keys=context_keys,
            )
        )

        # Store lifecycle components.
        self._pre_invoke = pre_tool
        self._post_invoke = post_tool
        self._post_result_key = resolved_post_result_key
        self._context_key_names: frozenset[str] = frozenset(
            p.name for p in context_key_params
        )

        # System prompt registry — subclasses populate after super().__init__.
        self._system_prompts: dict[str, PromptConfig] = {}

        # Store Agent runtime configuration.
        self._llm_engine: LLMEngine = llm_engine
        self._context_enabled: bool = context_enabled

        if records_window is not None and (not type(records_window) is int or records_window < 0):
            raise AgentError("records_window must be an int >= 0 or be 'None'.")
        self._records_window: Optional[int] = records_window

        self._records: List[AgentRecord] = []

        if response_preview_limit is None:
            self._response_preview_limit = None
        elif type(response_preview_limit) is not int or response_preview_limit <= 0:
            raise AgentError("response_preview_limit must be None or a positive integer > 0.")
        else:
            self._response_preview_limit = response_preview_limit

        if not isinstance(assistant_response_source, str) or assistant_response_source not in {"raw", "final"}:
            raise AgentError("assistant_response_source must be either 'raw' or 'final'.")
        self._assistant_response_source = assistant_response_source

        resolved_filter_extraneous_inputs = (
            filter_extraneous_inputs
            if filter_extraneous_inputs is not None
            else pre_tool.filter_extraneous_inputs
        )

        super().__init__(
            name=name,
            namespace=namespace,
            description=description,
            parameters=agent_parameters,
            return_type=self._post_invoke.return_type,
            filter_extraneous_inputs=resolved_filter_extraneous_inputs,
        )

    # ------------------------------------------------------------------ #
    # Agent lifecycle configuration and validation
    # ------------------------------------------------------------------ #
    @classmethod
    def _prepare_post_invoke_tool(
        cls,
        *,
        candidate: Optional[Union[Callable, AtomicInvokable]],
        agent_name: str,
    ) -> Tool:
        """Normalize the configured post-invoke component into a Tool."""
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
    def _resolve_post_result_key(
        *,
        post_result_key: Optional[str],
        post_params: list[ParamSpec],
    ) -> str:
        """Resolve the post-invoke parameter that receives the raw ``_invoke`` result."""
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
        post_params: list[ParamSpec],
    ) -> None:
        """Validate that ``post_result_key`` names a declared post-invoke parameter."""
        post_param_names = {p.name for p in post_params}
        if post_result_key not in post_param_names:
            raise AgentError(
                "post_result_key must name one of post_invoke's declared parameters; "
                f"got {post_result_key!r}."
            )

    @staticmethod
    def _normalize_context_keys(
        context_keys: list[str] | list[ParamSpec] | None,
    ) -> list[ParamSpec]:
        """Normalise ``context_keys`` to a list of KEYWORD_ONLY ParamSpecs.

        ``list[str]``      → KEYWORD_ONLY ParamSpecs with ``default=NO_VAL``.
        ``list[ParamSpec]`` → coerced to KEYWORD_ONLY (variadic items rejected).
        ``None``            → ``[]``.
        Duplicate names and empty strings are rejected.
        """
        if context_keys is None:
            return []
        if not isinstance(context_keys, list):
            raise AgentError("context_keys must be a list of str, a list of ParamSpec, or None.")
        variadic_kinds = {ParamSpec.VAR_POSITIONAL, ParamSpec.VAR_KEYWORD}
        result: list[ParamSpec] = []
        seen: set[str] = set()
        for i, item in enumerate(context_keys):
            if isinstance(item, str):
                name = item.strip()
                if not name:
                    raise AgentError(f"context_keys[{i}] must be a non-empty string.")
                param = ParamSpec(
                    name=name, index=i, kind=ParamSpec.KEYWORD_ONLY,
                    type="Any", default=NO_VAL,
                )
            elif isinstance(item, ParamSpec):
                if item.kind in variadic_kinds:
                    raise AgentError(
                        f"context_keys[{i}] ({item.name!r}) must not be variadic; "
                        f"got kind {item.kind!r}."
                    )
                param = ParamSpec(
                    name=item.name, index=i, kind=ParamSpec.KEYWORD_ONLY,
                    type=item.type, default=item.default,
                )
            else:
                raise AgentError(
                    f"context_keys items must be str or ParamSpec; "
                    f"got {type(item).__name__} at index {i}."
                )
            if param.name in seen:
                raise AgentError(f"context_keys contains duplicate name {param.name!r}.")
            seen.add(param.name)
            result.append(param)
        return result

    @staticmethod
    def _validate_pre_post_overlap_shapes(
        pre_params: list[ParamSpec],
        post_params: list[ParamSpec],
    ) -> None:
        """Validate that overlapping pre/post parameter names have compatible variadic shapes.

        Both must be non-variadic, or both must be the same variadic kind.
        A mismatch raises ``AgentError``.
        """
        pre_map = {p.name: p for p in pre_params}
        post_map = {p.name: p for p in post_params}
        variadic_kinds = {ParamSpec.VAR_POSITIONAL, ParamSpec.VAR_KEYWORD}
        for name in pre_map.keys() & post_map.keys():
            pre = pre_map[name]
            post = post_map[name]
            pre_v = pre.kind in variadic_kinds
            post_v = post.kind in variadic_kinds
            if pre_v != post_v or (pre_v and pre.kind != post.kind):
                raise AgentError(
                    f"Overlapping pre/post parameter {name!r}: both must be non-variadic "
                    f"or both must be the same variadic kind; "
                    f"got {pre.kind!r} (pre_invoke) and {post.kind!r} (post_invoke)."
                )

    @staticmethod
    def _warn_reserved_name_collisions(
        pre_params: list[ParamSpec],
        post_params: list[ParamSpec],
        context_key_names: frozenset[str],
    ) -> None:
        """Warn or raise when a pre/post param name collides with a context key or reserved arg.

        Context-key collisions always warn: the param will never receive a value.

        Reserved-arg collisions are tiered on ``run_id``:
        - Semantically identical to ``RUN_ID_PARAM`` (name/kind/type/default match): warn
          (redundant declaration; the framework grafts it automatically).
        - Semantically different: raise ``AgentError`` (true collision — the caller's
          param would silently shadow the framework's meaning).
        """
        reserved_agent_args = frozenset({"run_id"})
        all_reserved = context_key_names | reserved_agent_args
        checked: set[str] = set()
        all_params = (
            [(p, "pre_invoke") for p in pre_params]
            + [(p, "post_invoke") for p in post_params]
        )
        for param, source in all_params:
            if param.name not in all_reserved or param.name in checked:
                continue
            checked.add(param.name)
            if param.name in context_key_names:
                warnings.warn(
                    f"{param.name!r} declared in {source} will be popped from inputs "
                    "before reaching it (a context key); it will never receive a "
                    "caller-supplied value.",
                    UserWarning,
                    stacklevel=4,
                )
            elif param.name == "run_id":
                semantically_equal = (
                    param.kind == RUN_ID_PARAM.kind
                    and param.type == RUN_ID_PARAM.type
                    and param.default == RUN_ID_PARAM.default
                    and param.description == RUN_ID_PARAM.description
                )
                if semantically_equal:
                    warnings.warn(
                        f"'run_id' declared in {source} is redundant; "
                        "the framework grafts it automatically.",
                        UserWarning,
                        stacklevel=4,
                    )
                else:
                    raise AgentError(
                        f"'run_id' declared in {source} conflicts with the "
                        "framework-reserved 'run_id' parameter "
                        "(kind, type, or default mismatch)."
                    )
            else:
                warnings.warn(
                    f"{param.name!r} declared in {source} will be popped from inputs "
                    "before reaching it (a reserved agent argument); it will never "
                    "receive a caller-supplied value.",
                    UserWarning,
                    stacklevel=4,
                )

    @staticmethod
    def _compose_agent_parameters(
        *,
        pre_params: list[ParamSpec],
        post_params: list[ParamSpec],
        result_key: str,
        context_key_params: list[ParamSpec],
    ) -> list[ParamSpec]:
        """Compose the agent-facing parameter schema from the four-tier model.

        Graft A: post-only non-result non-variadic params (as KEYWORD_ONLY).
        Graft B: context_key_params (as KEYWORD_ONLY).
        Graft C: ``run_id`` (KEYWORD_ONLY, default=None).

        All grafts are inserted before an existing ``**kwargs`` parameter.
        Pre's ``ParamSpec`` wins on name overlaps with post params.
        """
        composed = list(pre_params)
        pre_names = {p.name for p in pre_params}
        variadic_kinds = {ParamSpec.VAR_POSITIONAL, ParamSpec.VAR_KEYWORD}

        def _insert_before_varkw(
            lst: list[ParamSpec], items: list[ParamSpec]
        ) -> list[ParamSpec]:
            if not items:
                return lst
            idx = next(
                (i for i, p in enumerate(lst) if p.kind == ParamSpec.VAR_KEYWORD),
                None,
            )
            return lst[:idx] + items + lst[idx:] if idx is not None else lst + items

        # Graft A: post-only non-result non-variadic params
        grafts: list[ParamSpec] = []
        for param in post_params:
            if param.name == result_key or param.name in pre_names or param.kind in variadic_kinds:
                continue
            grafts.append(
                ParamSpec(
                    name=param.name, index=0, kind=ParamSpec.KEYWORD_ONLY,
                    type=param.type, default=param.default,
                )
            )
        composed = _insert_before_varkw(composed, grafts)

        # Graft B: context key params (skip any whose name already appears in composed)
        existing_names = {p.name for p in composed}
        ck_grafts = [p for p in context_key_params if p.name not in existing_names]
        composed = _insert_before_varkw(composed, ck_grafts)

        # Graft C: run_id — always the canonical framework version.
        # Pop any collision (already warned/raised above) then reinsert.
        composed = [p for p in composed if p.name != "run_id"]
        composed = _insert_before_varkw(composed, [RUN_ID_PARAM])

        return [
            ParamSpec(name=p.name, index=i, kind=p.kind, type=p.type, default=p.default, description=p.description)
            for i, p in enumerate(composed)
        ]

    @classmethod
    def _prepare_agent_lifecycle_config(
        cls,
        *,
        post_invoke: Optional[Union[Callable, AtomicInvokable]],
        agent_name: str,
        pre_parameters: list[ParamSpec],
        post_result_key: Optional[str],
        context_keys: list[str] | list[ParamSpec] | None,
    ) -> tuple[Tool, str, list[ParamSpec], list[ParamSpec]]:
        """Prepare and validate the full construction-time agent lifecycle contract.

        Steps:
        1. Normalise post_invoke to a Tool.
        2. Resolve post_result_key.
        3. Validate result_key exists in post params.
        4. Normalise context_keys to ParamSpecs.
        5. Validate pre/post overlap shapes.
        6. Warn on reserved-name collisions.
        7. Compose agent parameters.

        Returns (post_tool, result_key, context_key_params, agent_parameters).
        """
        post_tool = cls._prepare_post_invoke_tool(candidate=post_invoke, agent_name=agent_name)
        result_key = cls._resolve_post_result_key(
            post_result_key=post_result_key, post_params=post_tool.parameters
        )
        cls._validate_post_routing_contract(
            post_result_key=result_key, post_params=post_tool.parameters
        )
        context_key_params = cls._normalize_context_keys(context_keys)
        context_key_names = frozenset(p.name for p in context_key_params)
        cls._validate_pre_post_overlap_shapes(pre_parameters, post_tool.parameters)
        cls._warn_reserved_name_collisions(
            pre_parameters, post_tool.parameters, context_key_names
        )
        agent_parameters = cls._compose_agent_parameters(
            pre_params=pre_parameters,
            post_params=post_tool.parameters,
            result_key=result_key,
            context_key_params=context_key_params,
        )
        return post_tool, result_key, context_key_params, agent_parameters

    # ------------------------------------------------------------------ #
    # Agent Properties
    # ------------------------------------------------------------------ #
    @property
    def post_result_key(self) -> str:
        """Post-invoke parameter name that receives the raw ``_invoke`` result."""
        return self._post_result_key

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
        """Whether the agent feeds prior turns into each invocation.

        When ``False``, turns are always ``[]`` (fresh conversation) but records
        are still appended for observability.
        """
        return self._context_enabled

    @context_enabled.setter
    def context_enabled(self, value: bool) -> None:
        if type(value) is not bool:
            raise ValueError("context_enabled must be a bool.")
        self._context_enabled = value

    @property
    def records_window(self) -> Optional[int]:
        """Number of stored turns to select per invocation. ``None`` means all."""
        return self._records_window

    @records_window.setter
    def records_window(self, value: Optional[int]) -> None:
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError("records_window must be an int >= 0 or be 'None'.")
        self._records_window = value

    @property
    def response_preview_limit(self) -> Optional[int]:
        """Character limit for rendered assistant responses. ``None`` means no truncation."""
        return self._response_preview_limit

    @property
    def assistant_response_source(self) -> Literal["raw", "final"]:
        """Whether rendered assistant history uses raw or final turn responses."""
        return self._assistant_response_source

    @property
    def records(self) -> List[AgentRecord]:
        """Shallow copy of the stored turn history."""
        return list(self._records)

    @property
    def pre_invoke(self) -> Tool:
        """Tool that converts the input mapping into a prompt string."""
        return self._pre_invoke

    @property
    def post_invoke(self) -> Tool:
        """Tool that converts the raw ``_invoke`` result into the final agent output."""
        return self._post_invoke

    @property
    def system_prompts(self) -> dict[str, PromptConfig]:
        """Shallow copy of the system prompt registry."""
        return dict(self._system_prompts)

    def update_prompt(self, key: str, config: PromptConfig) -> None:
        """Register or replace a system prompt by key."""
        if not isinstance(key, str) or not key.strip():
            raise AgentError("update_prompt: key must be a non-empty string.")
        if not isinstance(config, PromptConfig):
            raise AgentError("update_prompt: config must be a PromptConfig instance.")
        self._system_prompts[key.strip()] = config

    # ------------------------------------------------------------------ #
    # Agent Helpers
    # ------------------------------------------------------------------ #
    def build_messages(
        self,
        system_prompt: str,
        turns: List[AgentRecord],
        prompt: str,
    ) -> List[Dict[str, str]]:
        """Render provider-facing message dicts from canonical turn inputs.

        Each supplied turn is rendered through ``render_turn``. The current
        prompt is appended as the final user message.
        """
        messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
        if turns:
            for turn in turns:
                messages.extend(self.render_turn(turn))
        messages.append({"role": "user", "content": prompt})
        return messages

    def render_turn(self, turn: AgentRecord) -> List[Dict[str, str]]:
        """Render one canonical ``AgentRecord`` into LLM-facing messages.

        The assistant content is selected from either ``turn.generated_response``
        or ``turn.final_result.result`` according to ``assistant_response_source``.
        ``response_preview_limit`` is applied only to the rendered text.
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
            response_text = response_text[: self._response_preview_limit] + "..."

        return [
            {"role": "user", "content": turn.user_prompt},
            {"role": "assistant", "content": response_text},
        ]

    def _build_context(self, inputs: dict) -> tuple[dict, dict]:
        """Extract caller-supplied context values from the filtered input dict.

        Pops any key named in ``_context_key_names`` from ``inputs`` into the
        returned context dict. Subclasses may override to also inject instance-
        state values that do not appear in the agent schema.

        Returns ``(context, remaining)`` where ``remaining`` is the input dict
        with context keys removed.
        """
        context: dict = {}
        remaining = dict(inputs)
        for name in self._context_key_names:
            if name in remaining:
                context[name] = remaining.pop(name)
        return context, remaining

    # ------------------------------------------------------------------ #
    # Abstract core LLM work
    # ------------------------------------------------------------------ #
    @abstractmethod
    def _invoke(
        self,
        turns: list[AgentRecord],
        prompt: str,
        context: dict,
    ) -> tuple[AgentRecord, dict]:
        """Sync core LLM call path.

        Receives the selected conversation turns, the current prompt string,
        and the assembled context dict. Returns a 2-tuple of a draft
        ``AgentRecord`` (``final_result=None``) and a metadata dict carrying
        ``"llm_records"`` and ``"llm_model_data"``.
        """
        ...

    @abstractmethod
    async def _ainvoke(
        self,
        turns: list[AgentRecord],
        prompt: str,
        context: dict,
    ) -> tuple[AgentRecord, dict]:
        """Async core LLM call path. Mirror of ``_invoke``."""
        ...

    # ------------------------------------------------------------------ #
    # Result construction
    # ------------------------------------------------------------------ #
    def make_result(
        self,
        result: Any,
        started_at: datetime,
        ended_at: datetime,
        **result_kwargs: Any,
    ) -> AgentResult:
        """Construct this Agent's ``AgentResult`` envelope.

        ``result`` is the caller-facing post-processed payload. LLM accounting
        is passed via ``result_kwargs`` from ``_invoke``'s metadata dict.
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

        llm_token_usage = tuple(r.llm_result.token_usage for r in llm_records)

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
    def clear_memory(self) -> None:
        """Clear the stored turn history."""
        self._records.clear()

    def get_conversation(
        self,
        run_id: str | None = None,
        turns: int | None = None,
    ) -> list[AgentRecord]:
        """Walk the ``prev`` chain from a target record and return it oldest-first.

        This is the canonical entry point for branch-aware turn selection.

        Parameters
        ----------
        run_id:
            ``run_id`` of the target record. ``None`` starts from the most
            recently committed record. An unknown string raises
            ``AgentInvocationError``.
        turns:
            Maximum chain length to return. ``None`` means the full chain.
            ``0`` always raises ``ValueError``.
        """
        if turns == 0:
            raise ValueError(
                "get_conversation: turns must be a positive integer or None; "
                "0 is not valid (the method always returns at least the target record)."
            )

        if not self._records:
            return []

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

        chain: list[AgentRecord] = []
        current: AgentRecord | None = start
        while current is not None:
            chain.append(current)
            if turns is not None and len(chain) >= turns:
                break
            current = current.prev

        chain.reverse()
        return chain

    async def async_invoke(self, inputs: Mapping[str, Any]) -> AgentResult:
        """Async analog of ``invoke``.

        Lifecycle steps mirror ``invoke`` with ``await`` at pre/post and ``_ainvoke``.
        """
        logger.info(f"[Async {self.full_name} started]")
        started_at = datetime.now(timezone.utc)

        # ① Filter inputs
        inputs = self.filter_inputs(inputs)

        # ② Agent args
        run_id = inputs.pop("run_id", None)

        # ③ Context extraction
        context, remaining = self._build_context(inputs)

        # ④ Pre-slice / ⑤ Post-slice
        pre_param_names = {p.name for p in self._pre_invoke.parameters}
        pre_inputs = {k: v for k, v in remaining.items() if k in pre_param_names}
        post_param_names = {
            p.name for p in self._post_invoke.parameters
            if p.name != self._post_result_key
        }
        post_inputs = {k: v for k, v in remaining.items() if k in post_param_names}

        # ⑦ Task prompt
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

        # ⑥ History
        logger.debug(f"Agent.{self.name} selecting turns")
        turns: list[AgentRecord] = []
        if self._context_enabled:
            if self._records_window != 0:
                turns = self.get_conversation(run_id=run_id, turns=self._records_window)

        # ⑧ Core LLM work
        logger.debug(f"Agent.{self.name} performing async logic")
        draft, metadata = await self._ainvoke(turns=turns, prompt=prompt, context=context)

        if not isinstance(draft, AgentRecord):
            raise AgentInvocationError(
                f"_ainvoke returned non-AgentRecord draft (type={type(draft)!r})"
            )

        # ⑨ Output transformation
        try:
            logger.debug(f"Agent.{self.name}.post_invoke postprocessing result asynchronously")
            post_inputs[self._post_result_key] = draft.generated_response
            post_result = await self._post_invoke.async_invoke(post_inputs)
        except ToolInvocationError:
            raise
        except Exception as e:  # pragma: no cover
            raise AgentInvocationError(f"post_invoke Tool failed: {e}") from e

        final_response = post_result.result
        ended_at = datetime.now(timezone.utc)

        # ⑩ Result
        agent_result = self.make_result(
            result=final_response,
            started_at=started_at,
            ended_at=ended_at,
            **metadata,
        )

        # ⑪ Record — always appended
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
        """Invoke the agent with a single input mapping.

        Steps
        -----
        ① filter_inputs — collect declared keys; inject defaults.
        ② Pop framework-reserved arg ``run_id``.
        ③ _build_context — extract context keys into ``context``; remainder stays.
        ④ Slice ``pre_inputs`` from remaining.
        ⑤ Slice ``post_inputs`` from remaining (excludes result_key).
        ⑥ Select conversation turns according to ``context_enabled``.
        ⑦ pre_invoke → prompt string.
        ⑧ _invoke(turns, prompt, context) → draft + metadata.
        ⑨ post_invoke → final result.
        ⑩ Construct AgentResult.
        ⑪ Commit AgentRecord unconditionally.
        """
        with self._invoke_lock:
            logger.info(f"[{self.full_name} started]")
            started_at = datetime.now(timezone.utc)

            # ① Filter inputs
            inputs = self.filter_inputs(inputs)

            # ② Agent args
            run_id = inputs.pop("run_id", None)

            # ③ Context extraction
            context, remaining = self._build_context(inputs)

            # ④ Pre-slice / ⑤ Post-slice
            pre_param_names = {p.name for p in self._pre_invoke.parameters}
            pre_inputs = {k: v for k, v in remaining.items() if k in pre_param_names}
            post_param_names = {
                p.name for p in self._post_invoke.parameters
                if p.name != self._post_result_key
            }
            post_inputs = {k: v for k, v in remaining.items() if k in post_param_names}

            # ⑦ Task prompt
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

            # ⑥ History
            logger.debug(f"Agent.{self.name} selecting turns")
            turns: list[AgentRecord] = []
            if self._context_enabled:
                if self._records_window != 0:
                    turns = self.get_conversation(
                        run_id=run_id, turns=self._records_window
                    )

            # ⑧ Core LLM work
            logger.debug(f"Agent.{self.name} performing logic")
            draft, metadata = self._invoke(turns=turns, prompt=prompt, context=context)

            if not isinstance(draft, AgentRecord):
                raise AgentInvocationError(
                    f"_invoke returned non-AgentRecord draft (type={type(draft)!r})"
                )

            # ⑨ Output transformation
            try:
                logger.debug(f"Agent.{self.name}.post_invoke postprocessing result")
                post_inputs[self._post_result_key] = draft.generated_response
                post_result = self._post_invoke.invoke(post_inputs)
            except ToolInvocationError:
                raise
            except Exception as e:  # pragma: no cover
                raise AgentInvocationError(f"post_invoke Tool failed: {e}") from e

            final_response = post_result.result
            ended_at = datetime.now(timezone.utc)

            # ⑩ Result
            agent_result = self.make_result(
                result=final_response,
                started_at=started_at,
                ended_at=ended_at,
                **metadata,
            )

            # ⑪ Record — always appended
            record = replace(
                draft,
                final_result=agent_result,
                llm_records=metadata["llm_records"],
                prev=turns[-1] if turns else None,
            )
            self._records.append(record)

            logger.info(f"[{self.full_name} finished]")
            return agent_result

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        """Return a minimal diagnostic snapshot of this agent."""
        d = super().to_dict()
        d.update({
            "system_prompts": {
                key: {"template": cfg.template, "description": cfg.description}
                for key, cfg in self._system_prompts.items()
            },
            "pre_invoke": self.pre_invoke.to_dict(),
            "post_invoke": self.post_invoke.to_dict(),
            "post_result_key": self.post_result_key,
            "llm": self._llm_engine.to_dict(),
            "context_enabled": self.context_enabled,
            "records_window": self.records_window,
            "response_preview_limit": self.response_preview_limit,
            "assistant_response_source": self.assistant_response_source,
            "records": [turn.to_dict() for turn in self._records],
        })
        return d
