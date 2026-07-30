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
from ..llm.base import LLMEngine
from ..models.results import AgentResult
from ..tools import toolify
from ..models.agents.records import AgentRecord, LLMRecord
from ..models.agents.prompts import PromptConfig
from ..models.agents.tasks import AgentTask
from ..models.agents.thought_models2 import AgentThought2

logger = logging.getLogger(__name__)

from .tools import identity_pre_tool, identity_post_tool
from .prompts import THINKING_PROMPT
from ..constants.agents import (
    RUN_ID_PARAM,
    STOP_THINKING_SENTINEL,
    THINKING_CONTENT_FIELD,
    THOUGHTS_PER_ROUND_FIELD,
    THINKING_ADDITIONAL_INSTRUCTIONS_HEADER,
    THINKING_ADDITIONAL_INSTRUCTIONS_FOOTER,
)
from ..utils.agents import parse_thoughts, normalize_thinking_instructions
from ..utils.parameters import (
    semantically_compatible,
    semantically_identical,
    parameter_overlap,
    parameter_collisions,
    variadic_compatible,
    insert_by_category,
    to_paramspec_list,
)


# ───────────────────────────────────────────────────────────────────────────────
# Agent
# ───────────────────────────────────────────────────────────────────────────────
class Agent2(AtomicInvokable, ABC):
    """
    Abstract base for schema-driven LLM agents.

    ``Agent`` owns the full invocation lifecycle shell — input filtering,
    turn selection, pre/post-invoke dispatch, record management — but
    delegates the actual LLM work to the ``_initialize_task``/``think``/
    ``prepare``/``execute`` abstract-or-overridable hooks that concrete
    subclasses implement.

    Lifecycle
    ---------
    ``invoke(inputs)`` follows this sequence:

    1. ``filter_inputs`` collects declared keys and injects defaults.
    2. Reserved names (``run_id`` by default; see ``get_reserved_parameters``)
       are read non-destructively — ``inputs`` itself is never mutated.
    3. Remaining inputs are sliced into ``pre_inputs`` / ``post_inputs`` by
       name membership against ``pre_invoke`` / ``post_invoke``'s own
       declared parameters, excluding reserved names.
    4. ``pre_invoke`` converts ``pre_inputs`` to a prompt string.
    5. Turns are selected from history if ``context_enabled`` is True and
       ``records_window != 0`` — ``records_window=0`` forces ``turns=[]``
       even when ``context_enabled`` is True, since ``get_conversation``
       itself always raises on ``turns=0``.
    6. ``_initialize_task(turns, prompt, inputs)`` builds a task from the
       full filtered ``inputs`` dict untouched — not a slice, not popped of
       anything; then, every round, unconditionally: ``think(task)``,
       ``prepare(task)``, ``execute(task)`` — until ``task.complete``.
    7. ``post_invoke`` transforms ``task.generated_response`` into
       ``task.final_response``.
    8. ``_commit_emit`` assembles the completed ``AgentRecord`` (with
       ``inputs`` set to the exact same full dict passed to
       ``_initialize_task``) and the final ``AgentResult`` together, and
       appends the record to ``_records`` unconditionally.

    Schema composition
    -------------------
    The agent's parameter schema is composed at construction time from four
    flat sources, reconciled in order:

    - All ``pre_invoke`` parameters.
    - Post-only non-result parameters, grafted while preserving their
      original declared kind (no forced ``KEYWORD_ONLY`` coercion).
    - ``extra_parameters`` — a flat, subclass-computed source, populated by
      whatever concrete ``Agent2`` subclass constructs it (no such
      subclass has been built yet).
    - This (sub)class's reserved parameters (``get_reserved_parameters()``;
      ``run_id`` by default), grafted last.

    Name collisions between sources are resolved via
    ``semantically_compatible`` / ``semantically_identical``
    (``utils/parameters.py``): an incompatible collision raises
    ``AgentError``; a compatible-but-not-identical overlap warns (the
    earlier-reconciled source wins); an identical overlap is silent.
    Grafting uses ``insert_by_category`` so every new parameter lands at the
    position that preserves a valid Python-style signature ordering,
    regardless of its declared kind.

    ``context_enabled``
    -------------------
    ``True``:  ``get_conversation`` selects prior turns for each invocation.
    ``False``: turns are always ``[]``; ``run_id`` is ignored.
    Records are appended unconditionally regardless of this setting.
    """

    # Fixed per-class trait, never a constructor param. A subclass that
    # needs unbounded thinking permitted (e.g. a future ReActAgent2, whose
    # action phase doesn't depend on thinking finishing and already has
    # its own hard stop via tool_calls_limit) overrides this via plain
    # class-body reassignment — never inside __init__, so it is already
    # correct via MRO attribute lookup before Agent2.__init__ ever runs,
    # avoiding a super().__init__() ordering hazard.
    _permits_unbounded_thinking: bool = False

    # Framework-fixed "think" system prompt, assembled once at prompts.py's
    # import time — not per-instance, no compose method. A subclass whose
    # thinking phase needs more context overrides this via plain
    # class-body reassignment (e.g. a future ToolAgent2 pointing at
    # prompts.TOOL_THINKING_PROMPT instead), mirroring
    # _permits_unbounded_thinking's own override convention above.
    _THINK_PROMPT: PromptConfig = THINKING_PROMPT

    # ------------------------------------------------------------------ #
    # Class-level configuration
    # ------------------------------------------------------------------ #
    @classmethod
    def get_reserved_parameters(cls) -> list[ParamSpec]:
        """Return this Agent (sub)class's framework-reserved parameters.

        Base ``Agent`` reserves only ``run_id``. A subclass with its own
        fixed/reserved parameter overrides this as
        ``return super().get_reserved_parameters() + [MY_RESERVED_PARAM]``,
        so the same collision/overlap/graft machinery in ``__init__`` covers
        its reserved name too, without base ``Agent`` needing advance
        knowledge of it.
        """
        return [RUN_ID_PARAM]

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
        pre_invoke: Optional[AtomicInvokable | Callable] = None,
        post_invoke: Optional[AtomicInvokable | Callable] = None,
        post_result_key: Optional[str] = None,
        extra_parameters: list[str] | list[ParamSpec] | None = None,
        records_window: Optional[int] = None,
        response_preview_limit: Optional[int] = None,
        assistant_response_source: Literal["raw", "final"] = "raw",
        thinking_instructions: str | PromptConfig | None = None,
        thinking_rounds: int | None = 0,
        thoughts_per_round: int = 1,
        thoughts_window: int | None = 0,
    ) -> None:

        # ── Thinking knobs ──────────────────────────────────────────────────
        # Validated and fail-fast checked before any of the rest of
        # construction runs -- in particular before the reserved/
        # pre-post-extra reconciliation pipeline below, which is comparatively
        # expensive and shouldn't run at all for a config that's already known
        # to be invalid.
        if thinking_rounds is not None and (
            type(thinking_rounds) is not int or thinking_rounds < 0
        ):
            raise AgentError("thinking_rounds must be None or a non-negative int.")
        if thinking_rounds is None and not self._permits_unbounded_thinking:
            raise AgentError(
                f"{type(self).__name__} does not permit unbounded thinking "
                "(thinking_rounds=None); pass a non-negative int instead."
            )
        if type(thoughts_per_round) is not int or thoughts_per_round < 1:
            raise AgentError("thoughts_per_round must be a positive int (>= 1).")
        if thoughts_window is not None and (
            type(thoughts_window) is not int or thoughts_window < 0
        ):
            raise AgentError("thoughts_window must be None or a non-negative int.")

        thinking_instructions_config = normalize_thinking_instructions(thinking_instructions)

        # ── Pre-invoke ───────────────────────────────────────────────────────
        if pre_invoke is None:
            pre_tool = identity_pre_tool
        elif isinstance(pre_invoke, AtomicInvokable):
            pre_tool = pre_invoke
        else:
            pre_tool = toolify(
                pre_invoke,
                name="pre_invoke",
                namespace=name,
                description=f"The tool that preprocesses inputs into a string for Agent {name}",
            )
        if pre_tool.return_type.lower() not in {"any", "str"}:
            raise AgentError(
                "Agent.pre_invoke must return a type 'str'|'any'"
            )

        # ── Post-invoke ──────────────────────────────────────────────────────
        if post_invoke is None:
            post_tool = identity_post_tool
        elif isinstance(post_invoke, AtomicInvokable):
            post_tool = post_invoke
        else:
            post_tool = toolify(
                post_invoke,
                name="post_invoke",
                namespace=name,
                description=f"The tool that postprocesses outputs of Agent {name}",
            )
        if len(post_tool.parameters) == 0:
            raise AgentError("Agent.post_invoke must expect at least 1 argument")

        # 2. Reserved parameters for this (sub)class.
        reserved_params = self.get_reserved_parameters()

        # 3. Normalize extra_parameters; reject variadic entries.
        extra_params = to_paramspec_list(extra_parameters)
        variadic_kinds = {ParamSpec.VAR_POSITIONAL, ParamSpec.VAR_KEYWORD}
        for p in extra_params:
            if p.kind in variadic_kinds:
                raise AgentError(
                    f"extra_parameters entry {p.name!r} must not be variadic; "
                    f"got kind {p.kind!r}."
                )

        # 4-5. Reserved-name reconciliation — three independent passes, each
        # popped of its own reserved-name entries once reconciled.
        pre_params = self._reconcile_reserved_names(
            list(pre_tool.parameters), reserved_params, "pre_invoke"
        )
        post_params_full = self._reconcile_reserved_names(
            list(post_tool.parameters), reserved_params, "post_invoke"
        )
        extra_params = self._reconcile_reserved_names(
            extra_params, reserved_params, "extra_parameters"
        )

        # 6. Resolve post_result_key against the reserved-popped post pool.
        if post_result_key is None:
            if not post_params_full:
                raise AgentError("Agent.post_invoke must expect at least 1 argument")
            resolved_post_result_key = post_params_full[0].name
        else:
            if not isinstance(post_result_key, str) or not post_result_key.strip():
                raise AgentError("post_result_key must be None or a non-empty string.")
            resolved_post_result_key = post_result_key.strip()
            if resolved_post_result_key not in {p.name for p in post_params_full}:
                raise AgentError(
                    "post_result_key must name one of post_invoke's declared parameters; "
                    f"got {resolved_post_result_key!r}."
                )
        if resolved_post_result_key in {p.name for p in pre_params} or (
            resolved_post_result_key in {p.name for p in extra_params}
        ):
            raise AgentError(
                f"post_result_key {resolved_post_result_key!r} collides with a "
                "pre_invoke or extra_parameters name; a name cannot mean both a "
                "caller-supplied input and the internal result-routing slot."
            )

        # 7. Pre-vs-post reconciliation.
        post_only = [p for p in post_params_full if p.name != resolved_post_result_key]
        pre_post_collisions = parameter_collisions(pre_params, post_only)
        if pre_post_collisions:
            raise AgentError(
                f"pre_invoke/post_invoke parameter collision(s): {pre_post_collisions!r} "
                "(same name, incompatible type/kind)."
            )
        pre_post_overlap = parameter_overlap(pre_params, post_only)
        if not variadic_compatible(pre_params, post_only, set(pre_post_overlap)):
            raise AgentError(
                "pre_invoke and post_invoke each declare an independent variadic "
                "parameter of the same kind under different names; rename one."
            )
        pre_by_name = {p.name: p for p in pre_params}
        post_by_name = {p.name: p for p in post_only}
        for overlap_name in pre_post_overlap:
            if not semantically_identical(pre_by_name[overlap_name], post_by_name[overlap_name]):
                warnings.warn(
                    f"Parameter {overlap_name!r} is declared by both pre_invoke and "
                    "post_invoke and is compatible but not identical; pre_invoke's "
                    "declaration wins.",
                    UserWarning,
                    stacklevel=3,
                )
        post_only_remainder = [p for p in post_only if p.name not in pre_post_overlap]
        combined = insert_by_category(pre_params, post_only_remainder)

        # 8. Combined-vs-extra reconciliation.
        combined_extra_collisions = parameter_collisions(combined, extra_params)
        if combined_extra_collisions:
            raise AgentError(
                "pre_invoke/post_invoke schema vs. extra_parameters collision(s): "
                f"{combined_extra_collisions!r} (same name, incompatible type/kind)."
            )
        combined_extra_overlap = parameter_overlap(combined, extra_params)
        combined_by_name = {p.name: p for p in combined}
        extra_by_name = {p.name: p for p in extra_params}
        for overlap_name in combined_extra_overlap:
            if not semantically_identical(combined_by_name[overlap_name], extra_by_name[overlap_name]):
                warnings.warn(
                    f"extra_parameters entry {overlap_name!r} is compatible with an "
                    "existing pre_invoke/post_invoke parameter but not identical; "
                    "the pre_invoke/post_invoke declaration wins.",
                    UserWarning,
                    stacklevel=3,
                )
        extra_remainder = [p for p in extra_params if p.name not in combined_extra_overlap]
        combined = insert_by_category(combined, extra_remainder)

        # 9. Final reserved-parameter graft — produces the schema directly.
        agent_parameters = insert_by_category(combined, reserved_params)

        # Store lifecycle components.
        self._pre_invoke = pre_tool
        self._post_invoke = post_tool
        self._post_result_key = resolved_post_result_key

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

        # ── thinking_instructions reconciliation ────────────────────────────
        # thinking_instructions never becomes a new extra_parameters source --
        # unlike role_prompt, its placeholders must already be real, and only
        # need to be *compatible*, not identical (looser than the
        # pre/post/extra reconciliation above). One containment check covers
        # both failure modes at once: a name parameter_overlap doesn't return
        # is either wholly unknown to self.parameters, or present but
        # incompatible (a true collision) -- either way it doesn't belong in
        # thinking_instructions.
        thinking_overlap = set(
            parameter_overlap(list(thinking_instructions_config.parameters), self.parameters)
        )
        thinking_names = {p.name for p in thinking_instructions_config.parameters}
        if not thinking_names <= thinking_overlap:
            raise AgentError(
                "thinking_instructions references parameter(s) "
                f"{sorted(thinking_names - thinking_overlap)!r} that are not "
                "among this agent's real declared parameters (pre_invoke, "
                "post_invoke, extra_parameters, or reserved), or that "
                "collide incompatibly with one of them."
            )

        # ── Thinking system prompts ──────────────────────────────────────────
        # Both registered unconditionally -- present in every agent class
        # regardless of whether thinking_rounds=0 or thinking_instructions
        # was omitted. "think" is framework-fixed (the same shared
        # PromptConfig object across every instance of a class, via
        # self._THINK_PROMPT); "user-think" is the caller's own (possibly
        # empty) sub-prompt.
        self._system_prompts["think"] = self._THINK_PROMPT
        self._system_prompts["user-think"] = thinking_instructions_config

        self._thinking_rounds: int | None = thinking_rounds
        self._thoughts_per_round: int = thoughts_per_round
        self._thoughts_window: int | None = thoughts_window
        self._thoughts: List[List[AgentThought2]] = []

    # ------------------------------------------------------------------ #
    # Agent lifecycle configuration and validation
    # ------------------------------------------------------------------ #
    @staticmethod
    def _reconcile_reserved_names(
        params: list[ParamSpec],
        reserved_params: list[ParamSpec],
        source_label: str,
    ) -> list[ParamSpec]:
        """Warn or raise on reserved-name collisions, then pop reserved names.

        For each param whose name matches a ``reserved_params`` entry:
        ``semantically_identical`` → warn (redundant declaration);
        ``semantically_compatible`` (but not identical) → warn (distinct
        message); otherwise → raise ``AgentError`` (true collision). Returns
        ``params`` with every reserved-name entry removed — only the
        warned/compatible ones can still be present, since true collisions
        already raised.
        """
        reserved_by_name = {p.name: p for p in reserved_params}
        for param in params:
            reserved = reserved_by_name.get(param.name)
            if reserved is None:
                continue
            if semantically_identical(param, reserved):
                warnings.warn(
                    f"{param.name!r} declared in {source_label} is redundant; "
                    "the framework grafts it automatically.",
                    UserWarning,
                    stacklevel=4,
                )
            elif semantically_compatible(param, reserved):
                warnings.warn(
                    f"{param.name!r} declared in {source_label} is compatible with "
                    "the framework-reserved parameter of the same name but not "
                    "identical; the framework's version will be used.",
                    UserWarning,
                    stacklevel=4,
                )
            else:
                raise AgentError(
                    f"{param.name!r} declared in {source_label} conflicts with the "
                    "framework-reserved parameter of the same name "
                    "(kind, type, or default mismatch)."
                )
        return [p for p in params if p.name not in reserved_by_name]

    # ------------------------------------------------------------------ #
    # Agent Properties
    # ------------------------------------------------------------------ #
    @property
    def post_result_key(self) -> str:
        """Post-invoke parameter name that receives the task's raw generated response."""
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
    def thinking_rounds(self) -> int | None:
        """Thinking-round hard cap. ``0`` = off, ``None`` = unbounded (only
        on classes with ``_permits_unbounded_thinking = True``), a positive
        int = capped. Mutable -- same posture as ``thoughts_window``."""
        return self._thinking_rounds

    @thinking_rounds.setter
    def thinking_rounds(self, value: int | None) -> None:
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError("thinking_rounds must be None or a non-negative int.")
        if value is None and not self._permits_unbounded_thinking:
            raise ValueError(
                f"{type(self).__name__} does not permit unbounded thinking "
                "(thinking_rounds=None); pass a non-negative int instead."
            )
        self._thinking_rounds = value

    @property
    def thoughts_per_round(self) -> int:
        """Max thoughts kept per thinking round; extras are silently dropped."""
        return self._thoughts_per_round

    @thoughts_per_round.setter
    def thoughts_per_round(self, value: int) -> None:
        if type(value) is not int or value < 1:
            raise ValueError("thoughts_per_round must be a positive int (>= 1).")
        self._thoughts_per_round = value

    @property
    def thoughts_window(self) -> int | None:
        """Trailing-record window governing how many past records' thoughts
        get spliced into replay by ``_render_history_messages``. ``0``
        means none, ``None`` means every replayed record, a positive int
        ``k`` means only the last ``k`` records."""
        return self._thoughts_window

    @thoughts_window.setter
    def thoughts_window(self, value: int | None) -> None:
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError("thoughts_window must be None or a non-negative int.")
        self._thoughts_window = value

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
    def pre_invoke(self) -> AtomicInvokable:
        """Invokable that converts the input mapping into a prompt string."""
        return self._pre_invoke

    @property
    def post_invoke(self) -> AtomicInvokable:
        """Invokable that converts the task's raw generated response into the final agent output."""
        return self._post_invoke

    @property
    def system_prompts(self) -> dict[str, PromptConfig]:
        """Shallow copy of the system prompt registry."""
        return dict(self._system_prompts)

    # ------------------------------------------------------------------ #
    # Render pipeline
    # ------------------------------------------------------------------ #
    def render_task(
        self,
        task: AgentTask,
        *,
        additional_messages: list[dict[str, str]] | None = None,
    ) -> list[dict[str, str]]:
        """Build the exact send-payload message list for one LLM call.

        Fixed, unconditional pipeline shared by every phase and every
        family: system message, then historic turns, then this phase's
        own task messages. Family/phase-specific behavior lives entirely
        in ``_render_system_message``/``_render_task_messages`` — this
        method never branches on ``task.system_prompt_name`` itself.

        ``_render_history_messages`` is computed first, ahead of the
        system message, so its lazy build-once cache
        (``task.historic_messages``) is already populated by the time
        either of the other two steps run — neither currently reads it,
        but nothing downstream has to worry about ordering if a future
        override ever does. Computation order is independent of the final
        concatenation order below, which is fixed by message-list shape
        (system first, then history, then task messages), not by which
        step ran first.

        ``additional_messages`` (``None`` treated as ``[]``) is appended
        directly onto whatever ``_render_task_messages`` returns — which
        is ``task.task_messages`` itself, so the extension persists there
        too, across generation retries within one phase.

        Pure computation, no I/O — the same method serves both the sync
        and async lifecycle paths; there is no ``_async_render_task``.
        """
        history = self._render_history_messages(task)
        system = self._render_system_message(task)
        task_messages = self._render_task_messages(task)
        task_messages.extend(additional_messages or [])
        return system + history + task_messages

    def _render_system_message(self, task: AgentTask) -> list[dict[str, str]]:
        """Render ``task``'s active system message.

        Concrete, but meant to be overridden wholesale — base ``Agent2``
        only knows how to render the "think" phase, unconditionally (no
        internal dispatch on ``task.system_prompt_name`` here). A subclass
        with its own named system prompt(s) overrides this method, checks
        its own name(s) first, and falls back to
        ``super()._render_system_message(task)`` when the active name is
        "think".

        Builds the "think"/"user-think" two-stage splice: ``"user-think"``
        (the caller's own ``thinking_instructions``, possibly empty)
        renders against the full ``task.inputs``; its resolved text is
        wrapped in a labeled section only when non-empty, so the whole
        "additional instructions" block is invisible in the rendered
        prompt when the caller supplied none.
        """
        user_text = self._system_prompts["user-think"].render(task.inputs)
        think_context = {
            THINKING_CONTENT_FIELD: (
                THINKING_ADDITIONAL_INSTRUCTIONS_HEADER
                + user_text
                + THINKING_ADDITIONAL_INSTRUCTIONS_FOOTER
                if user_text
                else ""
            ),
            THOUGHTS_PER_ROUND_FIELD: self._thoughts_per_round,
        }
        rendered = self._render_system_prompt(task, think_context)
        return [{"role": "system", "content": rendered}]

    def _render_system_prompt(
        self,
        task: AgentTask,
        render_context: dict[str, Any],
    ) -> str:
        """Render ``task``'s active system prompt template against a context.

        Low-level primitive: looks up
        ``self._system_prompts[task.system_prompt_name]`` and renders it
        against ``render_context``, returning the raw string only — no
        message-list wrapping (that's the caller's job). An unregistered
        or unset name raises ``KeyError`` naturally; not guarded against,
        since every real call path sets a valid name before rendering.

        Also an override point: a family whose active template needs extra
        context beyond what its caller already assembled (e.g. a future
        ``ToolAgent2``'s think prompt needing
        ``TOOLS``/``CONSTANTS``/``TOOL_CALLS_LIMIT``) overrides this
        method, augments ``render_context``, and calls
        ``super()._render_system_prompt(task, augmented_context)``.
        """
        return self._system_prompts[task.system_prompt_name].render(render_context)

    def _render_history_messages(self, task: AgentTask) -> list[dict[str, str]]:
        """Lazily render ``task.turns`` into ``task.historic_messages``.

        Built once per invoke and reused thereafter. If ``task.turns`` is
        empty, ``task.historic_messages`` correctly stays empty — later
        calls re-checking an empty ``turns`` are a harmless no-op, not an
        ambiguous state.

        Applies ``thoughts_window`` positionally — only the trailing
        window of replayed turns gets ``include_thoughts=True``; earlier
        turns render plain. ``render_turn`` itself stays position-blind
        (see its docstring); this is the one place with visibility into
        where a turn sits among all turns being replayed this call.
        ``records_window``'s own cap falls out for free here:
        ``task.turns`` is already ``records_window``-limited by the time
        this runs, so ``window`` never exceeds however many turns were
        actually selected.
        """
        if task.historic_messages:
            return task.historic_messages
        if not task.turns:
            return task.historic_messages

        n = len(task.turns)
        window = n if self._thoughts_window is None else min(self._thoughts_window, n)
        cutoff = n - window

        rendered: list[dict[str, str]] = []
        for i, turn in enumerate(task.turns):
            rendered.extend(self.render_turn(turn, include_thoughts=i >= cutoff))
        task.historic_messages = rendered
        return task.historic_messages

    def render_turn(
        self,
        turn: AgentRecord,
        *,
        include_thoughts: bool = False,
    ) -> List[Dict[str, str]]:
        """Render one canonical ``AgentRecord`` into LLM-facing messages.

        The assistant content is selected from either ``turn.generated_response``
        or ``turn.final_result.result`` according to ``assistant_response_source``.
        ``response_preview_limit`` is applied only to the rendered text.
        ``turn.user_prompt`` is used verbatim as the user message content — it
        is already a fully-resolved string, not a template.

        ``include_thoughts``: when ``True``, splices a thought-trail block
        into the assistant message's content (never a separate message, to
        respect strict user/assistant alternation), scoped to this record's
        own ``thoughts_start``/``thoughts_end`` span. Position-blind by
        design — this method has no way to know whether ``turn`` falls
        inside a trailing window; that decision lives entirely in
        ``_render_history_messages``, the only place with visibility into a
        turn's position among all turns being replayed for one call.
        """
        if not isinstance(turn, AgentRecord):
            raise AgentInvocationError(
                f"render_turn expected AgentRecord, got {type(turn)!r}"
            )

        if self._assistant_response_source == "final" and turn.final_result is None:
            raise AgentInvocationError(
                "render_turn: assistant_response_source='final' but this record's "
                "final_result is None (record is a draft and has not been committed)."
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

        messages = [
            {"role": "user", "content": turn.user_prompt},
            {"role": "assistant", "content": response_text},
        ]

        if not include_thoughts:
            return messages

        start, end = turn.thoughts_start, turn.thoughts_end
        if start is None or end is None or start == end:
            return messages

        snapshot = self._format_thoughts(self._thoughts[start:end])
        assistant_content = f"THOUGHTS:\n\n{snapshot}\n\nRESPONSE:\n{response_text}"
        return [messages[0], {"role": "assistant", "content": assistant_content}]

    @abstractmethod
    def _render_task_messages(self, task: AgentTask) -> list[dict[str, str]]:
        """Family-specific task-message construction, every phase included.

        Replaces the old abstract ``_render_action_messages`` — that
        method never saw the "think" phase at all (the old ``render_task``
        filtered it out externally); this one owns the full
        ``task.system_prompt_name`` branching itself, "think" included, so
        every concrete family is responsible for both cases.

        Every implementation must follow this contract:

        1. If ``task.task_messages`` is already non-empty, this phase's
           messages are already built for this round — return it as-is
           (``render_task`` extends it with ``additional_messages`` next).
        2. Otherwise, branch on ``task.system_prompt_name``:
           - ``"think"``: build ``task.task_messages`` from scratch with
             the shared CURRENT TASK banner / thoughts-so-far snapshot
             (via ``self._render_task_thoughts(task)``) / continue-thinking
             instruction shape.
           - anything else: build this family's own phase-specific content
             from scratch.
        3. Return ``task.task_messages`` — the same list object now stored
           on the task, not a copy.

        No body exists yet — every current caller of ``render_task`` in
        this pass only ever produces ``"think"``-phase tasks; concrete
        family bodies land with ``BasicAgent2``/``ToolAgent2``.
        """
        ...

    def _render_task_thoughts(self, task: AgentTask) -> str:
        """Format ``task.thoughts`` (this run's thoughts so far) as text.

        Shared, concrete primitive — callable from any family's own
        ``_render_task_messages`` implementation, in either branch, not
        just the "think" case.
        """
        return f"# CURRENT THOUGHTS\n{self._format_thoughts(task.thoughts)}"

    @staticmethod
    def _format_thoughts(rounds: List[List[AgentThought2]]) -> str:
        """Plain-text thought-trail formatter.

        Deliberately not ``pprint.pformat`` — verified that it wraps long
        string values across multiple lines via adjacent string-literal
        splitting and doesn't escape embedded newlines, both of which break
        visual structure for free-text ``content``. Flat, one thought per
        line, no index labeling — nothing downstream needs to address a
        thought by position.
        """
        lines = [f"[{t.category}] {t.content}" for round_thoughts in rounds for t in round_thoughts]
        return "\n".join(lines) if lines else "No thoughts yet."

    # ------------------------------------------------------------------ #
    # Thinking
    # ------------------------------------------------------------------ #
    def think(self, task: AgentTask) -> AgentTask:
        """Advance ``task`` by exactly one thinking round.

        No-ops (returns ``task`` unchanged) if ``not task.keep_thinking`` —
        covers both "thinking disabled" and "already finished" uniformly.
        Otherwise renders, calls the engine, parses the (possibly
        ``|STOP_THINKING|``-truncated) output into categorized thoughts,
        truncates to ``thoughts_per_round``, records one new round, and
        advances ``keep_thinking``. No retries — an empty raw LLM response is
        a hard failure, and the lax category-marker format degrades to a
        single ``OTHER`` thought when unmarked text is present. A round whose
        parsed thoughts end up empty regardless (e.g. a bare or
        whitespace-only ``|STOP_THINKING|`` sentinel with nothing preceding
        it) is also a hard failure — never silently recorded as a no-op
        round.
        """
        if not task.keep_thinking:
            return task

        task.system_prompt_name = "think"
        task.task_messages = []
        messages = self.render_task(task)

        engine_result = self._llm_engine.invoke({"messages": messages})
        raw = engine_result.result
        if not raw:
            raise AgentInvocationError(
                f"{self.full_name}: thinking round produced empty output."
            )

        stop_seen = STOP_THINKING_SENTINEL in raw
        prefix = raw.split(STOP_THINKING_SENTINEL, 1)[0] if stop_seen else raw
        parsed = parse_thoughts(prefix)[: self._thoughts_per_round]
        if not parsed:
            raise AgentInvocationError(
                f"{self.full_name}: thinking round produced no parsable "
                "thoughts (stop sentinel or empty content with no thought "
                "text)."
            )
        task.thoughts.append(parsed)

        task.llm_records.append(LLMRecord(
            messages=list(task.task_messages),
            llm_result=engine_result,
            system_prompt_name="think",
        ))

        if stop_seen or (
            self._thinking_rounds is not None and len(task.thoughts) >= self._thinking_rounds
        ):
            task.keep_thinking = False

        return task

    async def async_think(self, task: AgentTask) -> AgentTask:
        """Async mirror of ``think`` — always a genuine independent
        implementation (real LLM I/O every call), never a thread-offload
        default, matching this family's existing convention for hooks that
        perform real generation."""
        if not task.keep_thinking:
            return task

        task.system_prompt_name = "think"
        task.task_messages = []
        messages = self.render_task(task)

        engine_result = await self._llm_engine.async_invoke({"messages": messages})
        raw = engine_result.result
        if not raw:
            raise AgentInvocationError(
                f"{self.full_name}: thinking round produced empty output."
            )

        stop_seen = STOP_THINKING_SENTINEL in raw
        prefix = raw.split(STOP_THINKING_SENTINEL, 1)[0] if stop_seen else raw
        parsed = parse_thoughts(prefix)[: self._thoughts_per_round]
        if not parsed:
            raise AgentInvocationError(
                f"{self.full_name}: thinking round produced no parsable "
                "thoughts (stop sentinel or empty content with no thought "
                "text)."
            )
        task.thoughts.append(parsed)

        task.llm_records.append(LLMRecord(
            messages=list(task.task_messages),
            llm_result=engine_result,
            system_prompt_name="think",
        ))

        if stop_seen or (
            self._thinking_rounds is not None and len(task.thoughts) >= self._thinking_rounds
        ):
            task.keep_thinking = False

        return task

    # ------------------------------------------------------------------ #
    # Task-lifecycle hooks
    # ------------------------------------------------------------------ #
    # _initialize_task is concrete here: wrapping turns/prompt/inputs into a
    # bare AgentTask is subclass-agnostic base-contract work. A future
    # Agent2 subclass needing richer per-task bookkeeping would override
    # this and return its own AgentTask subclass instead -- no such
    # subclass has been designed yet (the pre-Agent2 ToolAgent family
    # layers ToolAgentTask/PlanActTask/ReActTask on top of the shared
    # AgentTask this way, but that's a different class hierarchy, not a
    # precedent Agent2 subclasses are bound to follow). think/prepare/
    # execute stay @abstractmethod (prepare/execute) or
    # concrete-but-family-agnostic (think) here — every subclass's
    # advance-by-one-round logic genuinely differs.
    def _initialize_task(
        self,
        *,
        turns: list[AgentRecord],
        prompt: str,
        inputs: dict,
    ) -> AgentTask:
        """Build and return this invocation's AgentTask.

        Receives the selected conversation turns, the current prompt
        string, and the full filtered ``inputs`` dict — untouched. Concrete
        base implementation just wraps the three into a bare ``AgentTask``;
        subclasses that need a richer ``AgentTask`` subclass (with
        additional bookkeeping fields populated) override this.

        ``system_prompt_name`` is seeded to ``"think"`` here — the only
        phase base ``Agent2`` itself has any concept of (its own
        ``_render_system_message`` only knows how to render "think"). A
        subclass with additional phases (e.g. ``BasicAgent2``'s ``"role"``)
        transitions it away from "think" during its own ``prepare()``/
        ``execute()``, once thinking is actually done — never here, since
        this hook runs before any thinking round has had a chance to run.

        ``task.keep_thinking`` is seeded here from ``self._thinking_rounds``
        (``!= 0``) — a subclass whose own ``_initialize_task`` builds a
        richer task type from scratch must remember to seed it the same
        way; not handled automatically for those subclasses.
        """
        task = AgentTask(turns=turns, inputs=inputs, user_prompt=prompt, system_prompt_name="think")
        task.keep_thinking = self._thinking_rounds != 0
        return task

    @abstractmethod
    def prepare(self, task: AgentTask) -> AgentTask:
        """Advance ``task``'s preparation phase by one round.

        Hard-abstract, no base default — every concrete family implements
        this. What "preparation" means varies materially by family (e.g. a
        no-op for a single-turn reply agent vs. an LLM-driven planning step
        for a tool-calling agent); base ``Agent2`` has no generic behavior
        to offer here. Exact per-family bodies are out of scope for this
        pass.
        """
        ...

    @abstractmethod
    async def async_prepare(self, task: AgentTask) -> AgentTask:
        """Async mirror of ``prepare``.

        Hard-abstract, no soft default. Mirrors ``think``/``async_think``'s
        precedent: a family's preparation phase is expected to perform
        genuine async I/O of its own (an LLM call, a tool's own
        ``async_invoke``) often enough that a soft default would just mask a
        missing real implementation.
        """
        ...

    @abstractmethod
    def execute(self, task: AgentTask) -> AgentTask:
        """Advance ``task``'s execution phase by one round.

        Hard-abstract, no base default. Implementations set
        ``task.generated_response`` and ``task.complete = True`` on the
        round that finishes the invocation, per family-specific rules.
        Exact per-family bodies are out of scope for this pass.
        """
        ...

    @abstractmethod
    async def async_execute(self, task: AgentTask) -> AgentTask:
        """Async mirror of ``execute``, same rationale as ``async_prepare``."""
        ...

    def _build_record_from_task(self, task: AgentTask) -> AgentRecord:
        """Assemble a complete AgentRecord from a finished AgentTask.

        The not-yet-built next concrete ``Agent2`` subclass may use this
        base implementation as-is, or override it to return a richer
        ``AgentRecord`` subclass with its own bookkeeping folded in — no
        such subclass has been designed yet. ``final_result`` is
        deliberately left at its dataclass default (``None``) — it is not
        knowable until ``build_result_from_record`` runs afterward; the
        caller attaches it via ``dataclasses.replace(...)``.

        Persists ``task.thoughts`` (this run's rounds) into the agent-level
        ``self._thoughts`` here — the only point in the lifecycle where a
        run's thoughts move from task-local to agent-level, mirroring how
        ``llm_records`` is read exactly once at this same point.

        Uses ``task.turns`` (not a separate parameter) — it's the same list
        ``_initialize_task`` was given; no caller ever has a different one.
        """
        prev = task.turns[-1] if task.turns else None
        thoughts_start = len(self._thoughts)
        self._thoughts.extend(task.thoughts)
        thoughts_end = len(self._thoughts)
        return AgentRecord(
            user_prompt=task.user_prompt,
            generated_response=task.generated_response,
            inputs=task.inputs,
            llm_records=tuple(task.llm_records),
            prev=prev,
            thoughts_start=thoughts_start,
            thoughts_end=thoughts_end,
        )

    # ------------------------------------------------------------------ #
    # Result construction
    # ------------------------------------------------------------------ #
    def build_result_from_record(
        self,
        record: AgentRecord,
        *,
        result: Any,
        started_at: datetime,
        ended_at: datetime,
    ) -> AgentResult:
        """Construct this Agent's ``AgentResult`` envelope directly from a
        completed ``AgentRecord``, wrapping the low-level ``_make_result``
        primitive.

        Public — external subclasses may legitimately override this to
        choose a more specific result class or add subclass-specific result
        fields.

        ``result`` (the post-``post_invoke`` final caller-facing value) and
        ``started_at``/``ended_at`` (the outer invocation's timing envelope)
        are not derivable from the record, so both stay explicit parameters —
        distinct from ``record.generated_response``, which is the
        pre-``post_invoke`` raw value.

        ``llm_model_data`` comes from ``self._llm_engine._get_model_data()``
        directly (engine configuration, not a provider response) rather than
        ``record.llm_records[-1]`` — the record's own ``llm_records`` can
        legitimately be empty (e.g. ``thinking_rounds=0`` and an ``execute``
        phase that makes no LLM call of its own), and model identity doesn't
        depend on any call having happened.
        """
        llm_token_usage = tuple(r.llm_result.token_usage for r in record.llm_records)
        llm_model_data = self._llm_engine._get_model_data()

        return self._make_result(
            result=result,
            started_at=started_at,
            ended_at=ended_at,
            result_cls=AgentResult,
            llm_token_usage=llm_token_usage,
            llm_model_data=llm_model_data,
            thoughts_start=record.thoughts_start,
            thoughts_end=record.thoughts_end,
        )

    def _commit_emit(
        self,
        task: AgentTask,
        started_at: datetime,
        ended_at: datetime,
    ) -> AgentResult:
        """Assemble, commit, and return this invocation's final ``AgentResult``.

        Shared tail for both ``invoke``/``async_invoke``, called once
        ``task.final_response`` has already been stamped by ``post_invoke``.
        Pure computation, no I/O — one method serves both lifecycle paths.
        Takes ``task`` alone — ``task.turns`` already holds the same list
        ``_initialize_task`` was given, so no separate ``turns`` parameter
        is threaded through.

        1. ``record = self._build_record_from_task(task)``, validated as an
           ``AgentRecord``. Must happen before step 2 below —
           ``thoughts_start``/``thoughts_end`` (and the ``llm_token_usage``/
           ``llm_model_data`` derived from ``record.llm_records``) only
           exist once this has run, since it's the one place ``task.thoughts``
           merges into agent-level ``self._thoughts`` and the before/after
           offsets get computed. There is no task-only shortcut to them.
        2. ``result = self.build_result_from_record(record,
           result=task.final_response, started_at=started_at,
           ended_at=ended_at)``.
        3. ``record = replace(record, final_result=result)``.
        4. ``self._records.append(record)`` — committed unconditionally.
        5. Return ``result``.
        """
        record = self._build_record_from_task(task)
        if not isinstance(record, AgentRecord):
            raise AgentInvocationError(
                f"_build_record_from_task returned non-AgentRecord (type={type(record)!r})"
            )

        result = self.build_result_from_record(
            record,
            result=task.final_response,
            started_at=started_at,
            ended_at=ended_at,
        )

        record = replace(record, final_result=result)
        self._records.append(record)
        return result

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def clear_memory(self) -> None:
        """Clear the stored turn history."""
        self._records.clear()
        self._thoughts.clear()

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

    def get_thoughts(self, run_id: str | None = None) -> list[list[AgentThought2]]:
        """Return the thoughts produced during one specific completed run.

        Mirrors ``get_conversation``'s ``run_id`` resolution: ``None``
        resolves to the most recently committed record; an unresolvable
        ``run_id`` raises ``AgentInvocationError``. Returns a plain slice of
        the persisted ``self._thoughts`` list (rounds, not individual
        thoughts), scoped to that record's own ``thoughts_start``/
        ``thoughts_end`` span — never a copy of the full persisted list.
        """
        if not self._records:
            return []
        if run_id is None:
            record = self._records[-1]
        else:
            record = next(
                (r for r in self._records if r.final_result.run_id == run_id),
                None,
            )
            if record is None:
                raise AgentInvocationError(
                    f"get_thoughts: no record with run_id {run_id!r} found in agent history."
                )
        return self._thoughts[record.thoughts_start:record.thoughts_end]

    async def async_invoke(self, inputs: Mapping[str, Any]) -> AgentResult:
        """Async analog of ``invoke``.

        Lifecycle steps mirror ``invoke`` with ``await`` at pre/post and the
        ``async_think``/``async_prepare``/``async_execute`` task-lifecycle
        hooks. Task construction itself (``_initialize_task``) is plain
        object construction with no I/O, so it's called directly here, not
        threaded through an async wrapper. No ``self._invoke_lock``
        (unchanged from current implementation, which does not lock
        async_invoke either).
        """
        logger.info(f"[Async {self.full_name} started]")
        started_at = datetime.now(timezone.utc)

        # 1. Filter inputs.
        inputs = self.filter_inputs(inputs)

        # 2. Reserved names + non-destructive run_id read.
        reserved_names = {p.name for p in self.get_reserved_parameters()}
        run_id = inputs.get("run_id")

        # 3. Slice pre/post inputs, excluding reserved names.
        pre_names = {p.name for p in self._pre_invoke.parameters} - reserved_names
        pre_inputs = {k: v for k, v in inputs.items() if k in pre_names}

        post_names = (
            {p.name for p in self._post_invoke.parameters}
            - {self._post_result_key}
            - reserved_names
        )
        post_inputs = {k: v for k, v in inputs.items() if k in post_names}

        # 4. Task prompt.
        try:
            logger.debug(f"Agent.{self.name}.pre_invoke preprocessing inputs asynchronously")
            pre_result = await self._pre_invoke.async_invoke(pre_inputs)
            raw_prompt = pre_result.result
        except ToolInvocationError:
            raise
        except Exception as e:  # pragma: no cover
            raise AgentInvocationError(f"pre_invoke Tool failed: {e}") from e

        if not isinstance(raw_prompt, str):
            raise AgentInvocationError(
                f"pre_invoke returned a non-string result "
                f"(type={type(raw_prompt)!r}); a prompt string is required"
            )
        prompt = raw_prompt

        # 5. History.
        logger.debug(f"Agent.{self.name} selecting turns")
        turns: list[AgentRecord] = []
        if self._context_enabled and self._records_window != 0:
            turns = self.get_conversation(run_id=run_id, turns=self._records_window)

        # 6. Task lifecycle: initialize, think -> prepare -> execute until
        # complete.
        logger.debug(f"Agent.{self.name} performing async logic")
        task = self._initialize_task(turns=turns, prompt=prompt, inputs=inputs)
        while not task.complete:
            task = await self.async_think(task)
            task = await self.async_prepare(task)
            task = await self.async_execute(task)

        # 7. Output transformation.
        try:
            logger.debug(f"Agent.{self.name}.post_invoke postprocessing result asynchronously")
            post_inputs[self._post_result_key] = task.generated_response
            post_result = await self._post_invoke.async_invoke(post_inputs)
        except ToolInvocationError:
            raise
        except Exception as e:  # pragma: no cover
            raise AgentInvocationError(f"post_invoke Tool failed: {e}") from e
        task.final_response = post_result.result

        ended_at = datetime.now(timezone.utc)

        # 8-9. Record + result construction, committed to history.
        agent_result = self._commit_emit(task, started_at, ended_at)

        logger.info(f"[Async {self.full_name} finished]")
        return agent_result

    def invoke(self, inputs: Mapping[str, Any]) -> AgentResult:
        """Invoke the agent with a single input mapping.

        Runs under ``self._invoke_lock``.

        Steps
        -----
        1. ``filter_inputs`` — collect declared keys; inject defaults.
        2. Read reserved names (``run_id`` by default) non-destructively via
           ``inputs.get(...)`` — ``inputs`` is never mutated.
        3. Slice ``pre_inputs`` / ``post_inputs`` from ``inputs`` by name
           membership against ``pre_invoke`` / ``post_invoke``'s own
           declared parameters, excluding reserved names.
        4. ``pre_invoke`` → prompt string (validated).
        5. Select conversation turns according to ``context_enabled``.
        6. ``_initialize_task(turns, prompt, inputs)`` → task; loop
           ``task = think(task); task = prepare(task); task = execute(task)``
           until ``task.complete``.
        7. ``post_invoke`` transforms ``task.generated_response`` into
           ``task.final_response``.
        8-9. ``_commit_emit(task, started_at, ended_at)`` builds the
           completed record and ``AgentResult`` together, commits the
           record unconditionally, and returns the result.
        """
        with self._invoke_lock:
            logger.info(f"[{self.full_name} started]")
            started_at = datetime.now(timezone.utc)

            # 1. Filter inputs.
            inputs = self.filter_inputs(inputs)

            # 2. Reserved names + non-destructive run_id read.
            reserved_names = {p.name for p in self.get_reserved_parameters()}
            run_id = inputs.get("run_id")

            # 3. Slice pre/post inputs, excluding reserved names.
            pre_names = {p.name for p in self._pre_invoke.parameters} - reserved_names
            pre_inputs = {k: v for k, v in inputs.items() if k in pre_names}

            post_names = (
                {p.name for p in self._post_invoke.parameters}
                - {self._post_result_key}
                - reserved_names
            )
            post_inputs = {k: v for k, v in inputs.items() if k in post_names}

            # 4. Task prompt.
            try:
                logger.debug(f"Agent.{self.name}.pre_invoke preprocessing inputs")
                pre_result = self._pre_invoke.invoke(pre_inputs)
                raw_prompt = pre_result.result
            except ToolInvocationError:
                raise
            except Exception as e:  # pragma: no cover
                raise AgentInvocationError(f"pre_invoke Tool failed: {e}") from e

            if not isinstance(raw_prompt, str):
                raise AgentInvocationError(
                    f"pre_invoke returned a non-string result "
                    f"(type={type(raw_prompt)!r}); a prompt string is required"
                )
            prompt = raw_prompt

            # 5. History.
            logger.debug(f"Agent.{self.name} selecting turns")
            turns: list[AgentRecord] = []
            if self._context_enabled and self._records_window != 0:
                turns = self.get_conversation(run_id=run_id, turns=self._records_window)

            # 6. Task lifecycle: initialize, think -> prepare -> execute
            # until complete.
            logger.debug(f"Agent.{self.name} performing logic")
            task = self._initialize_task(turns=turns, prompt=prompt, inputs=inputs)
            while not task.complete:
                task = self.think(task)
                task = self.prepare(task)
                task = self.execute(task)

            # 7. Output transformation.
            try:
                logger.debug(f"Agent.{self.name}.post_invoke postprocessing result")
                post_inputs[self._post_result_key] = task.generated_response
                post_result = self._post_invoke.invoke(post_inputs)
            except ToolInvocationError:
                raise
            except Exception as e:  # pragma: no cover
                raise AgentInvocationError(f"post_invoke Tool failed: {e}") from e
            task.final_response = post_result.result

            ended_at = datetime.now(timezone.utc)

            # 8-9. Record + result construction, committed to history.
            agent_result = self._commit_emit(task, started_at, ended_at)

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
            "thinking_rounds": self.thinking_rounds,
            "thoughts_per_round": self.thoughts_per_round,
            "thoughts_window": self.thoughts_window,
            "thoughts": [
                [thought.to_dict() for thought in round_]
                for round_ in self._thoughts
            ],
        })
        return d
