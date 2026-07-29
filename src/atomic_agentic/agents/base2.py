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
import asyncio
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
    delegates the actual LLM work to the ``_initialize_task``/``_progress``
    abstract-or-overridable hooks that concrete subclasses implement.

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
    5. Turns are selected from history if ``context_enabled`` is True.
    6. ``_initialize_task(turns, prompt, inputs)`` builds a task from the
       full filtered ``inputs`` dict untouched — not a slice, not popped of
       anything; then ``_progress`` runs until ``task.complete``;
       ``_build_record_from_task`` assembles the completed record.
    7. ``post_invoke`` transforms the raw response into the final result.
    8. A completed ``AgentRecord`` is always appended to ``_records``, with
       ``inputs`` set to the exact same full dict passed to
       ``_initialize_task``.

    Schema composition
    -------------------
    The agent's parameter schema is composed at construction time from four
    flat sources, reconciled in order:

    - All ``pre_invoke`` parameters.
    - Post-only non-result parameters, grafted while preserving their
      original declared kind (no forced ``KEYWORD_ONLY`` coercion).
    - ``extra_parameters`` — a flat, subclass-computed source (e.g.
      ``BasicAgent`` passes its role-prompt placeholders as this agent's
      sole ``extra_parameters`` source).
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
    # class-body reassignment -- never inside __init__, so it is already
    # correct via MRO attribute lookup before Agent2.__init__ ever runs,
    # avoiding a super().__init__() ordering hazard.
    _permits_unbounded_thinking: bool = False

    # Framework-fixed "think" system prompt, assembled once at prompts.py's
    # import time -- not per-instance, no compose method. A subclass whose
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
        get spliced into replay by ``_render_historic_messages``. ``0``
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
        ``_render_historic_messages``, the only place with visibility into a
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

    # ------------------------------------------------------------------ #
    # Thinking
    # ------------------------------------------------------------------ #
    @staticmethod
    def _render_user_instructions_block(text: str) -> str:
        """Wrap resolved ``thinking_instructions`` text in its own labeled
        section, or return ``""`` unchanged if there's nothing to show.

        Keeps the whole "additional instructions" section invisible in the
        rendered ``"think"`` prompt when the caller supplied none, rather
        than leaving an empty header — the header/markers live here, at
        render time, not in ``THINKING_BASE_TEMPLATE`` itself, since a
        static template can't know in advance whether this will be empty.
        """
        if not text:
            return ""
        return THINKING_ADDITIONAL_INSTRUCTIONS_HEADER + text + THINKING_ADDITIONAL_INSTRUCTIONS_FOOTER

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

    def _render_think_messages(
        self,
        task: AgentTask,
        *,
        additional_messages: list[dict[str, str]] | None = None,
    ) -> list[dict[str, str]]:
        """Shared/concrete rendering for the thinking phase.

        Every family reuses this unchanged: the ``"think"``/``"user-think"``
        two-stage splice for the system message, the ordinary historic
        messages, and a fixed current-task/thoughts-so-far/continue-thinking
        task-message shape (built lazily, once per round — ``think()`` resets
        ``task.task_messages`` before each round so this always rebuilds).
        """
        user_text = self._system_prompts["user-think"].render(task.inputs)
        think_context = {
            THINKING_CONTENT_FIELD: self._render_user_instructions_block(user_text),
            THOUGHTS_PER_ROUND_FIELD: self._thoughts_per_round,
        }
        system = self._render_system_message(task, think_context)
        historic = self._render_historic_messages(task)

        if not task.task_messages:
            task.task_messages = [
                {
                    "role": "user",
                    "content": (
                        f"===== CURRENT TASK =====\n{task.user_prompt}\n===== END TASK ====="
                    ),
                },
                {
                    "role": "assistant",
                    "content": f"THOUGHTS SO FAR:\n\n{self._format_thoughts(task.thoughts)}",
                },
                {
                    "role": "user",
                    "content": (
                        "Given the current task and your thoughts so far, continue thinking. "
                        "Produce your next thought(s), or include |STOP_THINKING| if you are done."
                    ),
                },
            ]
        task.task_messages.extend(additional_messages or [])
        return system + historic + task.task_messages

    def think(self, task: AgentTask) -> AgentTask:
        """Advance ``task`` by exactly one thinking round.

        No-ops (returns ``task`` unchanged) if ``not task.keep_thinking`` —
        covers both "thinking disabled" and "already finished" uniformly.
        Otherwise renders, calls the engine, parses the (possibly
        ``|STOP_THINKING|``-truncated) output into categorized thoughts,
        truncates to ``thoughts_per_round``, records one new round, and
        advances ``keep_thinking``. No retries — an empty raw response is a
        hard failure, and the lax category-marker format degrades to a
        single ``OTHER`` thought rather than needing correction.
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
    # bare AgentTask is subclass-agnostic base-contract work. BasicAgent
    # inherits this unchanged. ToolAgent re-declares it @abstractmethod,
    # since only PlanActAgent/ReActAgent know how to build their own richer
    # ToolAgentTask subclass; the bare AgentTask this base method returns
    # isn't sufficient for them. _progress stays @abstractmethod here —
    # every subclass's advance-by-one-round logic genuinely differs.
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

        ``system_prompt_name`` is set to ``None`` here — this base hook has
        no domain knowledge of which (if any) system prompt applies;
        subclasses that need one override this and set it explicitly
        (``BasicAgent`` calls ``super()._initialize_task(...)`` then stamps
        ``"role"`` over this ``None``).

        ``task.keep_thinking`` is seeded here from ``self._thinking_rounds``
        (``!= 0``) — a subclass whose own ``_initialize_task`` builds a
        richer task type from scratch (mirroring today's ``ToolAgent``
        re-declaring this abstract) must remember to seed it the same way;
        not handled automatically for those subclasses.
        """
        task = AgentTask(turns=turns, inputs=inputs, user_prompt=prompt, system_prompt_name=None)
        task.keep_thinking = self._thinking_rounds != 0
        return task

    @abstractmethod
    def _progress(self, task: AgentTask) -> AgentTask:
        """Advance the task by exactly one round; may set ``task.complete``.

        The base lifecycle loop in ``invoke()``/``async_invoke()`` is
        ``while not task.complete: task = self._progress(task)``.
        Implementations must set ``task.generated_response`` and
        ``task.complete = True`` on the round that finishes the
        invocation.
        """
        ...

    def render_task(
        self,
        task: AgentTask,
        *,
        additional_messages: list[dict[str, str]] | None = None,
    ) -> list[dict[str, str]]:
        """Build the exact send-payload message list for one LLM call.

        No longer ``@abstractmethod`` — concrete dispatcher on
        ``task.system_prompt_name``: ``"think"`` delegates to the shared
        ``_render_think_messages`` (identical for every family); anything
        else delegates to the new abstract ``_render_action_messages``,
        which is where each concrete subclass's own non-thinking content
        (a reply, a plan, a step) still fully reimplements rather than
        composing via ``super()`` — same posture ``render_task`` itself
        used to have, just relocated one level down.

        ``additional_messages`` (``None`` treated as ``[]``) is appended
        onto ``task.task_messages`` and **persists** there — generation
        retries within one phase accumulate across multiple ``render_task``
        calls rather than resetting each time.

        Pure computation, no I/O — the same method serves both the sync and
        async lifecycle paths; there is no ``_async_render_task``.
        """
        if task.system_prompt_name == "think":
            return self._render_think_messages(task, additional_messages=additional_messages)
        return self._render_action_messages(task, additional_messages=additional_messages)

    @abstractmethod
    def _render_action_messages(
        self,
        task: AgentTask,
        *,
        additional_messages: list[dict[str, str]] | None = None,
    ) -> list[dict[str, str]]:
        """Family-specific, non-thinking action-phase message construction.

        Replaces what used to be ``render_task`` itself for this purpose —
        each concrete subclass (``BasicAgent2``'s reply, a future
        ``PlanActAgent2``'s plan messages, a future ``ReActAgent2``'s step
        messages) fully reimplements this, calling
        ``_render_system_message``/``_render_historic_messages`` explicitly
        for the two pieces that render identically everywhere, then lazily
        building its own ``task.task_messages`` when empty. No bodies exist
        yet — every current caller of ``render_task`` in this pass only
        ever produces ``"think"``-phase tasks.
        """
        ...

    def _render_system_message(
        self,
        task: AgentTask,
        render_context: dict[str, Any],
    ) -> list[dict[str, str]]:
        """Render ``task``'s active system prompt, if any.

        Returns ``[]`` when ``task.system_prompt_name`` is ``None``.
        Otherwise renders ``self._system_prompts[task.system_prompt_name]``
        against ``render_context`` and returns a single-element system
        message list. An unregistered name raises ``KeyError`` naturally —
        an internal-contract violation, not guarded against.
        """
        if task.system_prompt_name is None:
            return []
        rendered = self._system_prompts[task.system_prompt_name].render(render_context)
        return [{"role": "system", "content": rendered}]

    def _render_historic_messages(self, task: AgentTask) -> list[dict[str, str]]:
        """Lazily render ``task.turns`` into ``task.historic_messages``.

        Built once per invoke and reused thereafter. If ``task.turns`` is
        empty, ``task.historic_messages`` correctly stays empty — later
        calls re-checking an empty ``turns`` are a harmless no-op, not an
        ambiguous state.

        Applies ``thoughts_window`` positionally — only the trailing window
        of replayed turns gets ``include_thoughts=True``; earlier turns
        render plain. ``render_turn`` itself stays position-blind (see its
        docstring); this is the one place with visibility into where a turn
        sits among all turns being replayed this call. ``records_window``'s
        own cap falls out for free here: ``task.turns`` is already
        ``records_window``-limited by the time this runs, so ``window``
        never exceeds however many turns were actually selected.
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

    async def _async_initialize_task(
        self,
        *,
        turns: list[AgentRecord],
        prompt: str,
        inputs: dict,
    ) -> AgentTask:
        """Async mirror of ``_initialize_task``; default offloads to a worker thread."""
        return await asyncio.to_thread(
            self._initialize_task, turns=turns, prompt=prompt, inputs=inputs
        )

    async def _async_progress(self, task: AgentTask) -> AgentTask:
        """Async mirror of ``_progress``; default offloads to a worker thread."""
        return await asyncio.to_thread(self._progress, task)

    def _build_record_from_task(
        self,
        task: AgentTask,
        turns: list[AgentRecord],
    ) -> AgentRecord:
        """Assemble a complete AgentRecord from a finished AgentTask.

        ``BasicAgent`` uses this base implementation as-is; ``ToolAgent``
        overrides it to return a ``ToolAgentRecord`` with blackboard
        bookkeeping folded in. ``final_result`` is deliberately left at its
        dataclass default (``None``) — it is not knowable until
        ``build_result_from_record`` runs afterward; the caller attaches it
        via ``dataclasses.replace(...)``.

        Persists ``task.thoughts`` (this run's rounds) into the agent-level
        ``self._thoughts`` here — the only point in the lifecycle where a
        run's thoughts move from task-local to agent-level, mirroring how
        ``llm_records`` is read exactly once at this same point.
        """
        prev = turns[-1] if turns else None
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
        """
        llm_token_usage = tuple(r.llm_result.token_usage for r in record.llm_records)
        llm_model_data = record.llm_records[-1].llm_result.model_data

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

        Lifecycle steps mirror ``invoke`` with ``await`` at pre/post and
        the ``_async_initialize_task``/``_async_progress`` task-lifecycle
        hooks. No ``self._invoke_lock`` (unchanged from current
        implementation, which does not lock async_invoke either).
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

        # 6. Task lifecycle: initialize, progress until complete, build record.
        logger.debug(f"Agent.{self.name} performing async logic")
        task = await self._async_initialize_task(turns=turns, prompt=prompt, inputs=inputs)
        while not task.complete:
            task = await self._async_progress(task)
        record = self._build_record_from_task(task, turns)

        if not isinstance(record, AgentRecord):
            raise AgentInvocationError(
                f"_build_record_from_task returned non-AgentRecord (type={type(record)!r})"
            )

        # 7. Output transformation.
        try:
            logger.debug(f"Agent.{self.name}.post_invoke postprocessing result asynchronously")
            post_inputs[self._post_result_key] = record.generated_response
            post_result = await self._post_invoke.async_invoke(post_inputs)
        except ToolInvocationError:
            raise
        except Exception as e:  # pragma: no cover
            raise AgentInvocationError(f"post_invoke Tool failed: {e}") from e

        final_response = post_result.result
        ended_at = datetime.now(timezone.utc)

        # 8. Result.
        agent_result = self.build_result_from_record(
            record,
            result=final_response,
            started_at=started_at,
            ended_at=ended_at,
        )

        # 9. Record — always appended.
        record = replace(record, final_result=agent_result)
        self._records.append(record)

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
           ``task = self._progress(task)`` until ``task.complete``; then
           ``_build_record_from_task(task, turns)`` → record.
        7. ``post_invoke`` → final result.
        8. ``build_result_from_record(record, ...)`` → ``AgentResult``.
        9. Commit the completed ``AgentRecord`` unconditionally.
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

            # 6. Task lifecycle: initialize, progress until complete, build record.
            logger.debug(f"Agent.{self.name} performing logic")
            task = self._initialize_task(turns=turns, prompt=prompt, inputs=inputs)
            while not task.complete:
                task = self._progress(task)
            record = self._build_record_from_task(task, turns)

            if not isinstance(record, AgentRecord):
                raise AgentInvocationError(
                    f"_build_record_from_task returned non-AgentRecord (type={type(record)!r})"
                )

            # 7. Output transformation.
            try:
                logger.debug(f"Agent.{self.name}.post_invoke postprocessing result")
                post_inputs[self._post_result_key] = record.generated_response
                post_result = self._post_invoke.invoke(post_inputs)
            except ToolInvocationError:
                raise
            except Exception as e:  # pragma: no cover
                raise AgentInvocationError(f"post_invoke Tool failed: {e}") from e

            final_response = post_result.result
            ended_at = datetime.now(timezone.utc)

            # 8. Result.
            agent_result = self.build_result_from_record(
                record,
                result=final_response,
                started_at=started_at,
                ended_at=ended_at,
            )

            # 9. Record — always appended.
            record = replace(record, final_result=agent_result)
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
