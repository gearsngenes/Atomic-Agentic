from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
)
from datetime import datetime, timezone
from uuid import uuid4
import asyncio
import logging

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
from ..models.agents.records import AgentRecord
from ..models.agents.prompts import PromptConfig
from ..models.agents.tasks import AgentTask
from ..models.agents.conversations import Conversation
from ..models.agents.context import ContextVariable

logger = logging.getLogger(__name__)

from .tools import identity_pre_tool, identity_post_tool
from ..constants.agents import (
    DEFAULT_CONVERSATION_NAME,
    CONVERSATION_NAME_PATTERN,
    TRAILING_FORK_INDEX_PATTERN,
)
from ..utils.parameters import (
    semantically_identical,
    insert_by_category,
    to_paramspec_list,
    build_parameter_reports,
    apply_parameter_reports,
)
from ..utils.agents import stringify_result

# ───────────────────────────────────────────────────────────────────────────────
# Agent
# ───────────────────────────────────────────────────────────────────────────────
class Agent(AtomicInvokable, ABC):
    """
    Abstract base for schema-driven LLM agents.

    ``Agent`` owns the full invocation lifecycle shell — input filtering,
    turn selection, pre/post-invoke dispatch, record management — but
    delegates the actual LLM work to the ``_initialize_task``/``think``/
    ``prepare``/``act`` abstract-or-overridable hooks that concrete
    subclasses implement.

    Lifecycle
    ---------
    ``invoke(inputs)`` follows this sequence:

    1. ``filter_inputs`` collects declared keys and injects defaults.
    2. Reserved names (see ``get_reserved_parameters``; none by default on
       base ``Agent``) are read non-destructively — ``inputs`` itself is
       never mutated.
    3. Remaining inputs are sliced into ``pre_inputs`` / ``post_inputs`` by
       name membership against ``pre_invoke`` / ``post_invoke``'s own
       declared parameters, excluding reserved names.
    4. ``pre_invoke`` converts ``pre_inputs`` to a prompt string.
    5. Turns are selected from the active conversation's own tail via
       ``_resolve_context`` if ``records_window != 0``.
    6. ``_initialize_task(turns, prompt, inputs)`` builds a task from the
       full filtered ``inputs`` dict untouched — not a slice, not popped of
       anything; then, every round, unconditionally: ``task = think(task);
       task = prepare(task); task = act(task)`` — until ``task.complete``.
    7. ``post_invoke`` transforms the raw response into the final result.
    8. ``_commit_emit`` assembles the completed ``AgentRecord`` (with
       ``inputs`` set to the exact same full dict passed to
       ``_initialize_task``) and the final ``AgentResult`` together, then
       either appends the record in place or forks a new conversation --
       decided by a fresh children check at commit time -- unless context/
       window is disabled, in which case nothing is stored at all.

    Schema composition
    -------------------
    The agent's parameter schema is composed at construction time from four
    flat sources:

    - All ``pre_invoke`` parameters.
    - Post-only non-result parameters, preserving their original declared
      kind (no forced ``KEYWORD_ONLY`` coercion).
    - ``extra_parameters`` — a flat, subclass-computed source (e.g.
      ``BasicAgent`` passes its role-prompt placeholders as this agent's
      sole ``extra_parameters`` source).
    - This (sub)class's reserved parameters (``get_reserved_parameters()``;
      empty by default on base ``Agent``), grafted last.

    Reserved-name overlaps in pre_invoke/post_invoke/extra_parameters are
    resolved first, independently per source (never against each other): a
    declaration matching a reserved name exactly (``semantically_identical``)
    is silently stripped — it is never forwarded to that source at invocation
    time regardless of the match. Any other overlap raises ``AgentError``;
    there is no warning tier on this axis, since a reserved declaration is
    never negotiable. ``VAR_POSITIONAL``/``VAR_KEYWORD`` parameters are exempt
    from this check entirely — a catch-all's bucket name colliding with a
    reserved name is a naming coincidence, not the same slot.

    The remaining pre_invoke/post-only/extra_parameters sources are then
    reconciled together via one N-way compatibility report
    (``utils.parameters.build_parameter_reports``) rather than a pairwise
    fold: a name with no shared compatible type, or an incompatible kind,
    across the sources that declare it raises ``AgentError``; a
    compatible-but-not-identical overlap warns once per construction call,
    grouped across every such name — not one warning per name. The
    highest-priority declaring source (``pre_invoke`` > ``post_invoke`` >
    ``extra_parameters``) supplies ``kind``/``default``/``description``
    together; the constructed ``type`` is the full witness set of compatible
    type tokens, not just the winning source's own declared type. An
    identical overlap is silent. Grafting uses ``insert_by_category`` so every
    new parameter lands at the position that preserves a valid Python-style
    signature ordering, regardless of its declared kind.

    ``record_enabled`` / ``records_window``
    ----------------------------------------
    Two fully independent axes. ``record_enabled`` (write): ``True``
    persists the completed turn; ``False`` stores nothing, regardless of
    ``records_window``. ``records_window`` (read): ``0`` means no prior
    turns are read this call; ``N`` means the last ``N``; ``None`` means
    full history -- independent of ``record_enabled``.

        record_enabled=True,  records_window=None  -- read all, persist (default)
        record_enabled=False, records_window=0     -- read nothing, persist nothing
        record_enabled=False, records_window=None  -- read all, persist nothing (judge/probe)
        record_enabled=True,  records_window=0     -- read nothing, persist anyway

    Conversation storage
    --------------------
    History is stored per-conversation in ``self._conversations: dict[str,
    Conversation]``, seeded with a single empty ``"default"`` conversation
    at construction. Exactly one conversation is "active" at a time
    (``self._active_conversation``); ``invoke``/``async_invoke`` always read
    from and (when they commit) write to whichever conversation is active at
    call time -- always its own current tail, never an arbitrary earlier
    record. Retargeting to an earlier point requires an explicit
    ``clone_conversation``/``switch_conversation`` call before invoking
    (``create_conversation``/``get_record``/``delete_conversation`` round
    out the rest of the explicit conversation-management API). The one
    remaining *implicit* behavior is a concurrency-safety detail, not a
    caller-facing feature: two invocations racing to extend the same tail
    resolve safely via a fresh children check taken under
    ``self._invoke_lock`` at commit time -- whichever commits second
    automatically forks instead of colliding.
    """

    # ------------------------------------------------------------------ #
    # Class-level configuration
    # ------------------------------------------------------------------ #
    @classmethod
    def get_reserved_parameters(cls) -> list[ParamSpec]:
        """Return this Agent (sub)class's framework-reserved parameters.

        Base ``Agent`` reserves nothing by default. A subclass with its own
        fixed/reserved parameter overrides this as
        ``return [MY_RESERVED_PARAM] + super().get_reserved_parameters()``
        -- prepended, not appended -- so the same collision/overlap/graft
        machinery in ``__init__`` covers its reserved name too, without base
        ``Agent`` needing advance knowledge of it, and so that each
        subclass level's own reservation keeps sorting ahead of any
        reservation declared further up its ancestry, no matter how many
        levels deep the prepending goes.
        """
        return []

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        name: str,
        namespace: str,
        description: str,
        llm_engine: LLMEngine,
        record_enabled: bool = True,
        *,
        pre_invoke: Optional[AtomicInvokable | Callable] = None,
        post_invoke: Optional[AtomicInvokable | Callable] = None,
        post_result_key: Optional[str] = None,
        extra_parameters: list[str] | list[ParamSpec] | None = None,
        records_window: Optional[int] = None,
        response_preview_limit: Optional[int] = None,
    ) -> None:

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

        # 3. Reserved parameters for this (sub)class, computed once.
        reserved = self.get_reserved_parameters()
        reserved_names = {p.name for p in reserved}

        # 4. Normalize extra_parameters; reject variadic entries.
        extra_raw = to_paramspec_list(extra_parameters)
        variadic_kinds = {ParamSpec.VAR_POSITIONAL, ParamSpec.VAR_KEYWORD}
        for p in extra_raw:
            if p.kind in variadic_kinds:
                raise AgentError(
                    f"extra_parameters entry {p.name!r} must not be variadic; "
                    f"got kind {p.kind!r}."
                )

        # 5. Resolve post_result_key against post_tool.parameters directly
        # (unstripped of any reserved-name overlaps).
        if post_result_key is None:
            resolved_post_result_key = post_tool.parameters[0].name
        else:
            if not isinstance(post_result_key, str) or not post_result_key.strip():
                raise AgentError("post_result_key must be None or a non-empty string.")
            resolved_post_result_key = post_result_key.strip()
            if resolved_post_result_key not in {p.name for p in post_tool.parameters}:
                raise AgentError(
                    "post_result_key must name one of post_invoke's declared parameters; "
                    f"got {resolved_post_result_key!r}."
                )

        # 6. The result-routing slot can never itself be a reserved name.
        if resolved_post_result_key in reserved_names:
            raise AgentError(
                f"post_result_key {resolved_post_result_key!r} is a framework-reserved "
                "parameter name; it cannot also serve as the internal "
                "result-routing slot."
            )

        # 7. Post-only parameters: post_tool's declared params minus the result key.
        post_only_raw = [
            p for p in post_tool.parameters if p.name != resolved_post_result_key
        ]

        # 8. Reserved-overlap pass, fully independent per source (never
        # compared against each other for a reserved name).
        pre_params = self._strip_reserved_overlaps(
            list(pre_tool.parameters), reserved, "pre_invoke"
        )
        post_only = self._strip_reserved_overlaps(post_only_raw, reserved, "post_invoke")
        extra_params = self._strip_reserved_overlaps(extra_raw, reserved, "extra_parameters")

        # 9. N-way peer reconciliation: pre_invoke > post_invoke > extra_parameters.
        reports = build_parameter_reports([pre_params, post_only, extra_params])

        # 10. Apply reports: raises on the first genuine conflict; emits at
        # most one grouped UserWarning for compatible-but-not-identical names.
        constructed = apply_parameter_reports(
            reports, ("pre_invoke", "post_invoke", "extra_parameters"), error_cls=AgentError, stacklevel=4
        )

        # 11. Category-normalize the reconciled peer schema.
        combined = insert_by_category([], constructed)

        # 12. Final reserved-parameter graft — produces the schema directly.
        agent_parameters = insert_by_category(combined, reserved)

        # Store lifecycle components.
        self._pre_invoke = pre_tool
        self._post_invoke = post_tool
        self._post_result_key = resolved_post_result_key

        # System prompt registry — subclasses populate after super().__init__.
        self._system_prompts: dict[str, PromptConfig] = {}

        # Store Agent runtime configuration.
        self._llm_engine: LLMEngine = llm_engine
        self._record_enabled: bool = record_enabled

        if records_window is not None and (not type(records_window) is int or records_window < 0):
            raise AgentError("records_window must be an int >= 0 or be 'None'.")
        self._records_window: Optional[int] = records_window

        self._conversations: dict[str, Conversation] = {
            DEFAULT_CONVERSATION_NAME: Conversation(
                name=DEFAULT_CONVERSATION_NAME, agent_type=type(self), records=[]
            )
        }
        self._active_conversation: str = DEFAULT_CONVERSATION_NAME
        self._branch_counter: int = 0

        if response_preview_limit is None:
            self._response_preview_limit = None
        elif type(response_preview_limit) is not int or response_preview_limit <= 0:
            raise AgentError("response_preview_limit must be None or a positive integer > 0.")
        else:
            self._response_preview_limit = response_preview_limit

        super().__init__(
            name=name,
            namespace=namespace,
            description=description,
            parameters=agent_parameters,
            return_type=self._post_invoke.return_type,
        )

    # ------------------------------------------------------------------ #
    # Agent lifecycle configuration and validation
    # ------------------------------------------------------------------ #
    @staticmethod
    def _strip_reserved_overlaps(
        params: list[ParamSpec],
        reserved: list[ParamSpec],
        source_label: str,
    ) -> list[ParamSpec]:
        """Strip reserved-name overlaps from one source, independently.

        Never compares this source's declarations to any other source's --
        each of ``pre_invoke``/``post_invoke``/``extra_parameters`` is
        checked only against the fixed reserved spec. ``VAR_POSITIONAL``/
        ``VAR_KEYWORD`` params are exempt entirely, regardless of name -- a
        catch-all's bucket name colliding with a reserved name is a naming
        coincidence, not the same slot. For every other param whose name
        matches a reserved name: ``semantically_identical`` -> stripped
        silently (redundant, harmless declaration -- no warning); otherwise
        -> raises ``AgentError`` immediately. There is no warning tier on
        this axis -- a reserved declaration is never negotiable, it must
        match exactly or be omitted.
        """
        reserved_by_name = {p.name: p for p in reserved}
        variadic_kinds = {ParamSpec.VAR_POSITIONAL, ParamSpec.VAR_KEYWORD}
        stripped_names: set[str] = set()

        for param in params:
            if param.kind in variadic_kinds:
                continue
            reserved_spec = reserved_by_name.get(param.name)
            if reserved_spec is None:
                continue
            if semantically_identical(param, reserved_spec):
                stripped_names.add(param.name)
            else:
                raise AgentError(
                    f"{source_label} declares {param.name!r} "
                    f"(type={param.type!r}, kind={param.kind!r}, "
                    f"default={param.default!r}, description={param.description!r}), "
                    "which is not identical to the framework-reserved parameter of "
                    f"the same name (type={reserved_spec.type!r}, "
                    f"kind={reserved_spec.kind!r}, default={reserved_spec.default!r}, "
                    f"description={reserved_spec.description!r}); reserved declarations "
                    f"must match exactly or be omitted -- they are never forwarded to "
                    f"{source_label} regardless of compatibility."
                )

        return [p for p in params if p.name not in stripped_names]

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
    def record_enabled(self) -> bool:
        """Whether this agent persists each invocation's completed turn.

        Write-only axis, independent of ``records_window`` (the read
        axis). ``True``: the completed turn is appended (or auto-forked,
        per the existing commit-time children check) into the active
        conversation. ``False``: a normal ``AgentResult`` is still
        returned, but nothing is stored in any conversation for that
        invocation -- regardless of ``records_window``.
        """
        return self._record_enabled

    @record_enabled.setter
    def record_enabled(self, value: bool) -> None:
        if type(value) is not bool:
            raise ValueError("record_enabled must be a bool.")
        self._record_enabled = value

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
    def active_conversation(self) -> str:
        """Key of the currently active conversation."""
        return self._active_conversation

    @property
    def conversation_names(self) -> list[str]:
        """Snapshot list of every existing conversation key."""
        return list(self._conversations.keys())

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
            for i, turn in enumerate(turns):
                messages.extend(self.render_turn(turn, index=i))
        messages.append({"role": "user", "content": prompt})
        return messages

    def render_turn(self, turn: AgentRecord, **kwargs: Any) -> List[Dict[str, str]]:
        """Render one canonical ``AgentRecord`` into LLM-facing messages.

        The assistant content is always ``turn.generated_response`` — the
        LLM's own raw output, never ``post_invoke``'s transformed result
        (not retained on the record at all). ``response_preview_limit`` is
        applied only to the rendered text. ``turn.user_prompt`` is used
        verbatim as the user message content — it is already a
        fully-resolved string, not a template.

        ``**kwargs`` is accepted and ignored at this level — base rendering
        has no family-specific use for it. A subclass override consumes
        whatever keys it needs (e.g. ``ToolAgent``'s required ``index``)
        and forwards ``**kwargs`` through its own ``super().render_turn``
        call if it still wants this base behavior too.
        """
        if not isinstance(turn, AgentRecord):
            raise AgentInvocationError(
                f"render_turn expected AgentRecord, got {type(turn)!r}"
            )

        response_text = stringify_result(turn.generated_response)

        if (
            self._response_preview_limit is not None
            and len(response_text) > self._response_preview_limit
        ):
            response_text = response_text[: self._response_preview_limit] + "..."

        return [
            {"role": "user", "content": turn.user_prompt},
            {"role": "assistant", "content": response_text},
        ]

    # ------------------------------------------------------------------ #
    # Task-lifecycle hooks
    # ------------------------------------------------------------------ #
    # _initialize_task is concrete here: wrapping turns/prompt/inputs into a
    # bare AgentTask is subclass-agnostic base-contract work. BasicAgent
    # inherits this unchanged. ToolAgent re-declares it @abstractmethod,
    # since only PlanActAgent/ReActAgent/ScriptActAgent know how to build
    # their own richer AgentTask subclass; the bare AgentTask this base method returns
    # isn't sufficient for them. act stays @abstractmethod here — every
    # subclass's advance-by-one-round execution logic genuinely differs.
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
        """
        return AgentTask(turns=turns, inputs=inputs, user_prompt=prompt, system_prompt_name=None)

    def think(self, task: AgentTask) -> AgentTask:
        """Advance ``task``'s reasoning phase by one round.

        Concrete no-op passthrough — base ``Agent`` has no shared reasoning
        behavior. A family whose phase genuinely reasons before acting
        overrides it; every other family inherits this unchanged.
        """
        return task

    async def async_think(self, task: AgentTask) -> AgentTask:
        """Async mirror of ``think``; default offloads to a worker thread."""
        return await asyncio.to_thread(self.think, task)

    def prepare(self, task: AgentTask) -> AgentTask:
        """Advance ``task``'s preparation phase by one round.

        Concrete no-op passthrough, same rationale as ``think``.
        """
        return task

    async def async_prepare(self, task: AgentTask) -> AgentTask:
        """Async mirror of ``prepare``; default offloads to a worker thread."""
        return await asyncio.to_thread(self.prepare, task)

    @abstractmethod
    def act(self, task: AgentTask) -> AgentTask:
        """Advance ``task``'s execution phase by one round; may set
        ``task.complete``.

        The base lifecycle loop in ``invoke()``/``async_invoke()`` is
        ``task = think(task); task = prepare(task); task = act(task)``,
        repeated until ``task.complete``. Implementations must set
        ``task.generated_response`` and ``task.complete = True`` on the
        round that finishes the invocation. Hard-abstract — every concrete
        family performs genuine work here.
        """
        ...

    @abstractmethod
    async def async_act(self, task: AgentTask) -> AgentTask:
        """Async mirror of ``act``.

        Hard-abstract, no thread-offload default — every family that
        reaches this hook performs real work of its own, same rationale as
        ``act`` itself.
        """
        ...

    def render_task(
        self,
        task: AgentTask,
        *,
        additional_messages: list[dict[str, str]] | None = None,
    ) -> list[dict[str, str]]:
        """Build the exact send-payload message list for one LLM call.

        Concrete, class-independent pipeline shared by every phase and
        every family: system message, then historic turns, then this
        phase's own task messages. Family/phase-specific behavior lives
        entirely in ``_render_task_messages`` (and, for a family with a
        richer system-prompt context, in what it passes to
        ``_render_system_message``) — this method never branches on
        ``task.system_prompt_name`` itself.

        ``additional_messages`` (``None`` treated as ``[]``) is appended
        directly onto whatever ``_render_task_messages`` returns — which is
        ``task.task_messages`` itself, so the extension persists there too,
        across generation retries within one phase.

        Pure computation, no I/O — the same method serves both the sync and
        async lifecycle paths; there is no ``_async_render_task``.
        """
        system = self._render_system_message(task)
        history = self._render_historic_messages(task)
        task_messages = self._render_task_messages(task)
        task_messages.extend(additional_messages or [])
        return system + history + task_messages

    def _render_system_message(self, task: AgentTask) -> list[dict[str, str]]:
        """Render ``task``'s active system prompt, if any.

        Returns ``[]`` when ``task.system_prompt_name`` is ``None``.
        Otherwise renders ``self._system_prompts[task.system_prompt_name]``
        against ``task.inputs`` and returns a single-element system message
        list. An unregistered name raises ``KeyError`` naturally — an
        internal-contract violation, not guarded against.

        A family whose active template needs a richer context than
        ``task.inputs`` alone (e.g. tool/constants data) overrides this
        method wholesale, builds its own context, and renders directly —
        there is no parameterized "extra context" hook here by design.
        """
        if task.system_prompt_name is None:
            return []
        rendered = self._system_prompts[task.system_prompt_name].render(task.inputs)
        return [{"role": "system", "content": rendered}]

    def _render_historic_messages(self, task: AgentTask) -> list[dict[str, str]]:
        """Lazily render ``task.turns`` into ``task.historic_messages``.

        Built once per invoke and reused thereafter. If ``task.turns`` is
        empty, ``task.historic_messages`` correctly stays empty — later
        calls re-checking an empty ``turns`` are a harmless no-op, not an
        ambiguous state.

        Each turn is rendered with its 0-based position within *this
        invocation's* ``task.turns`` passed as ``index`` — window-relative
        (after any ``records_window`` truncation), not the turn's position
        in the full stored conversation. Base rendering ignores it;
        ``ToolAgent``'s override requires it.
        """
        if task.historic_messages:
            return task.historic_messages
        if not task.turns:
            return task.historic_messages
        rendered: list[dict[str, str]] = []
        for i, turn in enumerate(task.turns):
            rendered.extend(self.render_turn(turn, index=i))
        task.historic_messages = rendered
        return task.historic_messages

    @abstractmethod
    def _render_task_messages(self, task: AgentTask) -> list[dict[str, str]]:
        """Build this round's phase-specific task messages.

        Every implementation must follow this contract:

        1. If ``task.task_messages`` is already non-empty, this phase's
           messages are already built for this round — return it as-is
           (``render_task`` extends it with ``additional_messages`` next).
        2. Otherwise, build this phase's content from scratch and store it
           onto ``task.task_messages``.
        3. Return ``task.task_messages`` — the same list object now stored
           on the task, not a copy.

        Hard-abstract, no shared content — base ``Agent`` has no phase of
        its own to render generically.
        """
        ...

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

    def _build_record_from_task(
        self,
        task: AgentTask,
        turns: list[AgentRecord],
        *,
        run_id: str,
        started_at: datetime,
        ended_at: datetime,
    ) -> AgentRecord:
        """Assemble a complete AgentRecord from a finished AgentTask.

        ``BasicAgent`` uses this base implementation as-is; ``PlanActAgent``/
        ``ReActAgent``/``ScriptActAgent`` each override it directly (not via
        a shared ``ToolAgent`` implementation) to return their own richer
        ``ToolAgentRecord``-family (or, for ``ScriptActAgent``,
        ``ScriptActAgentRecord``) subclass. ``run_id``/``started_at``/
        ``ended_at`` are minted once by ``_commit_emit`` and threaded
        through here and into ``build_result_from_record`` — no draft
        phase, the record is complete the moment it's constructed.
        """
        prev = turns[-1] if turns else None
        return AgentRecord(
            user_prompt=task.user_prompt,
            generated_response=task.generated_response,
            run_id=run_id,
            started_at=started_at,
            ended_at=ended_at,
            inputs=task.inputs,
            llm_records=tuple(task.llm_records),
            prev=prev,
        )

    # ------------------------------------------------------------------ #
    # Result construction
    # ------------------------------------------------------------------ #
    def build_result_from_record(
        self,
        record: AgentRecord,
        *,
        result: Any,
        run_id: str,
        started_at: datetime,
        ended_at: datetime,
    ) -> AgentResult:
        """Construct this Agent's ``AgentResult`` envelope directly from a
        completed ``AgentRecord``, wrapping the low-level ``_make_result``
        primitive.

        Public — external subclasses may legitimately override this to
        choose a more specific result class or add subclass-specific result
        fields.

        ``result`` (the post-``post_invoke`` final caller-facing value),
        ``run_id``, and ``started_at``/``ended_at`` (the outer invocation's
        identity/timing envelope) are not derivable from the record, so all
        stay explicit parameters — distinct from ``record.generated_response``,
        which is the pre-``post_invoke`` raw value.
        """
        llm_token_usage = tuple(r.llm_result.token_usage for r in record.llm_records)
        llm_model_data = record.llm_records[-1].llm_result.model_data

        return self._make_result(
            result=result,
            started_at=started_at,
            ended_at=ended_at,
            result_cls=AgentResult,
            run_id=run_id,
            llm_token_usage=llm_token_usage,
            llm_model_data=llm_model_data,
        )

    def _commit_emit(
        self,
        task: AgentTask,
        turns: list[AgentRecord],
        result: Any,
        started_at: datetime,
        ended_at: datetime,
        *,
        active_key: str,
    ) -> AgentResult:
        """Assemble, commit, and return this invocation's final ``AgentResult``.

        Shared tail for both ``invoke``/``async_invoke``, called once the
        task-lifecycle loop has completed and ``post_invoke`` has produced
        ``result``. Pure computation, no I/O — one method serves both
        lifecycle paths. ``active_key`` is the active-conversation snapshot
        taken by ``_resolve_context`` before the task-lifecycle loop ran; no
        separate ``target`` parameter is needed here since ``record.prev``
        already equals the resolved target (or ``None``) once step 1
        completes -- ``turns`` was constructed by ``_resolve_context`` to
        always end exactly at that target.

        1. ``run_id = uuid4().hex``, minted once, up front -- before either
           the record or the result exists.
        2. ``record = self._build_record_from_task(task, turns,
           run_id=run_id, started_at=started_at, ended_at=ended_at)``,
           validated as an ``AgentRecord``. The record is already complete
           (no draft phase) the moment this returns.
        3. ``agent_result = self.build_result_from_record(record,
           result=result, run_id=run_id, started_at=started_at,
           ended_at=ended_at)``.
        4. Under ``self._invoke_lock``: if ``record_enabled`` is False,
           return ``agent_result`` with nothing stored. Otherwise, a fresh
           children check on ``record.prev`` decides append-vs-fork --
           delegating fork construction to the same ``clone_conversation``
           primitive exposed publicly -- and the record is appended to the
           resolved conversation's ``.records``, with ``target.children``
           updated to match.
        5. Return ``agent_result``.
        """
        run_id = uuid4().hex
        record = self._build_record_from_task(
            task, turns, run_id=run_id, started_at=started_at, ended_at=ended_at
        )
        if not isinstance(record, AgentRecord):
            raise AgentInvocationError(
                f"_build_record_from_task returned non-AgentRecord (type={type(record)!r})"
            )

        agent_result = self.build_result_from_record(
            record,
            result=result,
            run_id=run_id,
            started_at=started_at,
            ended_at=ended_at,
        )

        with self._invoke_lock:
            if not self._record_enabled:
                return agent_result

            # Fresh (not stale) children check, taken under the same lock as
            # the mutation below -- this is what makes two concurrent
            # commits targeting the same childless point resolve safely:
            # whichever commits second observes the first's already-appended
            # child and forks instead of colliding.
            target = record.prev
            if target is not None and target.children:
                dest_conv = self.clone_conversation(
                    source=active_key,
                    end_id=target.run_id,
                )
            else:
                dest_conv = self._conversations[active_key]

            dest_conv.records.append(record)
            if target is not None:
                target.children.append(record)

        return agent_result

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def clear_memory(self) -> None:
        """Hard-reset all conversation storage to the just-constructed state.

        Every conversation is discarded; ``self._conversations`` becomes
        ``{"default": []}`` and the active conversation resets to
        ``"default"``. ``self._branch_counter`` is deliberately NOT reset --
        an auto-generated conversation name must never be reissued, even
        across a full clear. Takes ``self._invoke_lock`` like every other
        conversation-mutating method -- without it, an in-flight
        ``invoke``/``async_invoke`` whose ``_resolve_context`` snapshot
        predates this call could have its later ``_commit_emit`` reference a
        conversation key this reset just removed. Every conversation's
        context is discarded with it.
        """
        with self._invoke_lock:
            self._conversations = {
                DEFAULT_CONVERSATION_NAME: Conversation(
                    name=DEFAULT_CONVERSATION_NAME, agent_type=type(self), records=[]
                )
            }
            self._active_conversation = DEFAULT_CONVERSATION_NAME

    def _conversation_key(self, conversation: str | None, caller: str) -> str:
        key = self._active_conversation if conversation is None else conversation
        if key not in self._conversations:
            raise AgentInvocationError(f"{caller}: unknown conversation {key!r}.")
        return key

    def remember(
        self,
        key: str,
        value: Any,
        description: str | None = None,
        *,
        readonly: bool = False,
        conversation: str | None = None,
    ) -> None:
        """``Conversation.remember`` on a conversation (``None`` = the active one)."""
        with self._invoke_lock:
            self._conversations[
                self._conversation_key(conversation, "remember")
            ].remember(key, value, description, readonly=readonly)

    def describe(
        self, key: str, description: str | None, *, conversation: str | None = None
    ) -> None:
        """``Conversation.describe`` on a conversation (``None`` = the active one)."""
        with self._invoke_lock:
            self._conversations[
                self._conversation_key(conversation, "describe")
            ].describe(key, description)

    def forget(self, key: str, *, conversation: str | None = None) -> None:
        """``Conversation.forget`` on a conversation (``None`` = the active one)."""
        with self._invoke_lock:
            self._conversations[
                self._conversation_key(conversation, "forget")
            ].forget(key)

    def remember_many(
        self, variables: Iterable[ContextVariable], *, conversation: str | None = None
    ) -> None:
        """``Conversation.remember_many`` on a conversation (``None`` = the active one)."""
        with self._invoke_lock:
            self._conversations[
                self._conversation_key(conversation, "remember_many")
            ].remember_many(variables)

    def forget_many(
        self, keys: Iterable[str], *, conversation: str | None = None
    ) -> None:
        """``Conversation.forget_many`` on a conversation (``None`` = the active one)."""
        with self._invoke_lock:
            self._conversations[
                self._conversation_key(conversation, "forget_many")
            ].forget_many(keys)

    def get_conversation(self, conversation_id: str | None = None) -> Conversation:
        """Return a snapshot of a conversation.

        A distinct ``Conversation`` holding a copy of the record list and of
        the context, tagged with the same ``agent_type``; mutating it (its
        list, or calling its mutators) never affects the agent -- the
        ``Agent`` wrappers are the only locked write path. ``conversation_id``
        is a dict key, not a record identity: once forking exists, a single
        ``AgentRecord`` can legitimately belong to more than one
        conversation's list.

        ``conversation_id=None`` resolves to the active conversation. An
        unknown key raises ``AgentInvocationError``.
        """
        with self._invoke_lock:
            conv = self._conversations[
                self._conversation_key(conversation_id, "get_conversation")
            ]
            return Conversation(
                name=conv.name,
                agent_type=conv.agent_type,
                records=list(conv.records),
                context=conv.context,
            )

    def _find_in_conversation(
        self, conversation_id: str, run_id: str
    ) -> AgentRecord | None:
        """Pure query for a record within a given conversation; never raises.

        Resolves ``run_id`` to the matching record, or ``None`` if no
        record in that conversation has it. Takes ``conversation_id``
        explicitly rather than reading ``self._active_conversation``
        internally, matching every other conversation-keyed helper on this
        class. Its only remaining caller is ``get_record``'s ``run_id``
        mode, which scans every conversation via this method -- always
        with a real, caller-supplied ``run_id`` (never ``None``; there is
        no "most recent record" shorthand anymore, since nothing else
        needs one now that ``_resolve_context`` no longer resolves an
        arbitrary target).
        """
        history = self._conversations[conversation_id].records
        return next(
            (r for r in history if r.run_id == run_id), None
        )

    def _resolve_context(self) -> tuple[str, list[AgentRecord]]:
        """Resolve this invocation's active-conversation snapshot and context turns.

        Called unlocked, early in ``invoke``/``async_invoke``, before the
        think/prepare/act loop -- the active-conversation key is snapshotted
        here so it travels into ``_commit_emit`` unchanged even if another
        thread calls ``switch_conversation`` before this invocation
        commits.

        1. Snapshot ``active_key = self._active_conversation``.
        2. If ``records_window == 0``, return ``(active_key, [])`` -- the
           read axis's only gate, independent of ``record_enabled``.
        3. Otherwise read ``active_key``'s full history and window it
           directly from the end -- every invocation always continues from
           the active conversation's own tail; retargeting to an earlier
           point is only ever done explicitly, via ``clone_conversation``/
           ``switch_conversation``, before calling here.
        """
        active_key = self._active_conversation
        if self._records_window == 0:
            return active_key, []

        history = self._conversations[active_key].records
        window = self._records_window
        turns = history if window is None else history[-window:]
        return active_key, turns

    def create_conversation(self, name: str) -> Conversation:
        """Create a new, empty conversation.

        ``name`` must be alphanumeric-plus-underscore and must not end in
        ``_<digits>`` (that suffix shape is reserved for auto-generated fork
        names) -- an invalid shape raises ``ValueError``. A name that already
        exists raises ``AgentInvocationError`` -- matches ``register()``'s
        collision behavior elsewhere in AA; never auto-activates either way.
        """
        with self._invoke_lock:
            if not CONVERSATION_NAME_PATTERN.fullmatch(name):
                raise ValueError(
                    f"create_conversation: invalid name {name!r} (must be "
                    "alphanumeric/underscore, must not end in _<digits>)."
                )
            if name in self._conversations:
                raise AgentInvocationError(
                    f"create_conversation: name {name!r} already exists."
                )
            conv = Conversation(name=name, agent_type=type(self), records=[])
            self._conversations[name] = conv
            return conv

    def get_record(
        self,
        *,
        run_id: str | None = None,
        conversation: str | None = None,
        index: int | None = None,
    ) -> AgentRecord:
        """Single-record lookup by exactly one of two addressing modes.

        Direct addressing only, not a search/query-by-criteria API: either
        ``run_id`` alone, or ``conversation`` + ``index`` together. Any other
        combination (one of the pair alone, both modes together, all
        ``None``) raises ``ValueError``.

        ``run_id`` mode scans every conversation (no global index backs
        this -- see the conversation-model-rebuild proposal's rationale for
        why); an unresolvable ``run_id`` raises ``ValueError``, matching the
        existing run_id-not-found convention used elsewhere (distinct from
        the ``AgentInvocationError`` convention for unknown *conversation
        keys*). ``conversation`` + ``index`` mode resolves via direct list
        indexing; an unknown ``conversation`` raises ``AgentInvocationError``,
        an out-of-range ``index`` raises the natural ``IndexError``.
        """
        run_id_mode = run_id is not None
        pair_mode = conversation is not None and index is not None
        if run_id_mode and (conversation is not None or index is not None):
            raise ValueError(
                "get_record: run_id may not be combined with conversation/index."
            )
        if not run_id_mode and not pair_mode:
            raise ValueError(
                "get_record: must supply either run_id alone, or both "
                "conversation and index."
            )

        if run_id_mode:
            for key in self._conversations:
                found = self._find_in_conversation(key, run_id)
                if found is not None:
                    return found
            raise ValueError(f"get_record: no record with run_id {run_id!r} found.")

        if conversation not in self._conversations:
            raise AgentInvocationError(
                f"get_record: unknown conversation {conversation!r}."
            )
        return self._conversations[conversation].records[index]

    @staticmethod
    def _resolve_range(
        records: list[AgentRecord],
        name: str,
        start: int | None,
        end: int | None,
        start_id: str | None,
        end_id: str | None,
    ) -> list[AgentRecord]:
        """Resolve an ``[start, end]`` range of ``records`` to a new list.

        Shared by ``import_conversation``. Index pair (``start``/``end``) xor
        run_id pair (``start_id``/``end_id``); mixing raises ``ValueError``.
        All four ``None`` returns every record (including none). Out-of-range
        bounds raise ``IndexError``; a resolved start after end, or a run_id
        not found in ``records``, raises ``ValueError``.
        """
        if start is None and end is None and start_id is None and end_id is None:
            return list(records)

        index_pair_given = start is not None or end is not None
        id_pair_given = start_id is not None or end_id is not None
        if index_pair_given and id_pair_given:
            raise ValueError(
                "import_conversation: start/end and start_id/end_id may not be mixed."
            )

        def index_of(run_id: str) -> int:
            found = next(
                (i for i, r in enumerate(records) if r.run_id == run_id), None
            )
            if found is None:
                raise ValueError(
                    f"import_conversation: no record with run_id {run_id!r} "
                    f"found in conversation {name!r}."
                )
            return found

        if id_pair_given:
            start_idx = 0 if start_id is None else index_of(start_id)
            end_idx = len(records) - 1 if end_id is None else index_of(end_id)
        else:
            start_idx = 0 if start is None else (
                start if start >= 0 else len(records) + start
            )
            end_idx = len(records) - 1 if end is None else (
                end if end >= 0 else len(records) + end
            )

        if not (0 <= start_idx <= len(records) - 1) or not (
            0 <= end_idx <= len(records) - 1
        ):
            raise IndexError(
                f"import_conversation: resolved range [{start_idx}, {end_idx}] "
                f"out of bounds for conversation {name!r} of length {len(records)}."
            )
        if start_idx > end_idx:
            raise ValueError(
                f"import_conversation: resolved start {start_idx} is after "
                f"resolved end {end_idx}."
            )
        return list(records[start_idx : end_idx + 1])

    def import_conversation(
        self,
        conversation: Conversation,
        *,
        start: int | None = None,
        end: int | None = None,
        start_id: str | None = None,
        end_id: str | None = None,
        name: str | None = None,
    ) -> Conversation:
        """Import an ``[start, end]`` range of ``conversation`` as a new conversation.

        ``conversation`` is typically a ``get_conversation`` snapshot, so
        cross-agent use is ``b.import_conversation(a.get_conversation("x"))``.
        Its ``agent_type`` must be exactly ``type(self)`` (``TypeError``
        otherwise, before anything is created). Always creates a new
        conversation, never merges.

        The range is addressed either by plain list index (``start``/``end``,
        negative indices resolved the normal Python way) or by run_id
        (``start_id``/``end_id``). Exactly one pair may have any non-None
        member -- mixing raises ``ValueError``. All four ``None`` imports the
        whole conversation (including an empty one). Out-of-range bounds
        raise ``IndexError``; a resolved ``start`` after ``end`` raises
        ``ValueError`` -- neither is silently clamped/emptied.

        ``name``, if given, is validated/collision-checked exactly like
        ``create_conversation``'s ``name`` (either failure raises
        ``ValueError``); if omitted, a name is auto-derived from
        ``conversation.name``.

        Records are reference-copied. Context is copied into a distinct
        container (``ContextVariable``s shared by identity, values by
        reference). Context is conversation-level, not per-record: a range
        import gets the conversation's **latest** context, not its state at
        that record. Reference-copied records share ``record.children`` lists
        across agents with different locks.
        """
        if not isinstance(conversation, Conversation):
            raise TypeError(
                "import_conversation: expected a Conversation, "
                f"got {type(conversation)!r}."
            )
        if conversation.agent_type is not type(self):
            raise TypeError(
                f"import_conversation: conversation was created for "
                f"{conversation.agent_type.__name__}, not {type(self).__name__}."
            )

        with self._invoke_lock:
            new_records = self._resolve_range(
                conversation.records, conversation.name, start, end, start_id, end_id
            )

            if name is not None:
                if not CONVERSATION_NAME_PATTERN.fullmatch(name):
                    raise ValueError(f"import_conversation: invalid name {name!r}.")
                if name in self._conversations:
                    raise ValueError(
                        f"import_conversation: name {name!r} already exists."
                    )
                resolved_name = name
            else:
                base = TRAILING_FORK_INDEX_PATTERN.sub("", conversation.name)
                candidate = f"{base}_{self._branch_counter}"
                self._branch_counter += 1
                while candidate in self._conversations:
                    candidate = f"{base}_{self._branch_counter}"
                    self._branch_counter += 1
                resolved_name = candidate

            new_conv = Conversation(
                name=resolved_name,
                agent_type=type(self),
                records=new_records,
                context=conversation.context,
            )
            self._conversations[resolved_name] = new_conv
            return new_conv

    def clone_conversation(
        self,
        source: str,
        *,
        start: int | None = None,
        end: int | None = None,
        start_id: str | None = None,
        end_id: str | None = None,
        name: str | None = None,
    ) -> Conversation:
        """Clone an arbitrary ``[start, end]`` range of an existing conversation.

        A thin wrapper over ``import_conversation`` with this agent's own
        ``source`` conversation: same range addressing, naming and errors, and
        the clone carries the source's context. An unknown ``source`` raises
        ``AgentInvocationError``.
        """
        with self._invoke_lock:
            snapshot = self.get_conversation(
                self._conversation_key(source, "clone_conversation")
            )
            return self.import_conversation(
                snapshot,
                start=start,
                end=end,
                start_id=start_id,
                end_id=end_id,
                name=name,
            )

    def switch_conversation(self, key: str) -> None:
        """Switch the active conversation. Unknown key raises ``AgentInvocationError``."""
        with self._invoke_lock:
            if key not in self._conversations:
                raise AgentInvocationError(
                    f"switch_conversation: unknown conversation {key!r}."
                )
            self._active_conversation = key

    def delete_conversation(self, key: str) -> Conversation:
        """Remove and return a conversation.

        ``"default"`` is never actually removed -- regardless of whether
        it is currently active, deleting it resets it to an empty
        ``Conversation`` (empty records and context) in place and returns the
        prior one; the key itself always stays present. Any other conversation is refused while it is
        the active conversation (switch active first); otherwise it is
        popped and returned.
        """
        with self._invoke_lock:
            if key not in self._conversations:
                raise AgentInvocationError(
                    f"delete_conversation: unknown conversation {key!r}."
                )
            if key == DEFAULT_CONVERSATION_NAME:
                old = self._conversations[key]
                self._conversations[key] = Conversation(
                    name=DEFAULT_CONVERSATION_NAME, agent_type=type(self), records=[]
                )
                return old
            if key == self._active_conversation:
                raise AgentInvocationError(
                    f"delete_conversation: cannot delete the active "
                    f"conversation {key!r}; switch active first."
                )
            return self._conversations.pop(key)

    async def async_invoke(self, inputs: Mapping[str, Any]) -> AgentResult:
        """Async analog of ``invoke``.

        Lifecycle steps mirror ``invoke`` with ``await`` at pre/post and
        the ``_async_initialize_task``/``async_think``/``async_prepare``/
        ``async_act`` task-lifecycle hooks. Same lock scope as ``invoke``:
        only ``_commit_emit`` runs under ``self._invoke_lock`` (a
        ``threading.RLock`` — safe to hold briefly here since
        ``_commit_emit`` performs no I/O and never awaits), letting
        concurrent ``async_invoke`` calls against the same instance
        genuinely overlap everywhere else. See ``invoke``'s own docstring
        for the full reasoning.
        """
        logger.info(f"[Async {self.full_name} started]")
        started_at = datetime.now(timezone.utc)

        # 1. Filter inputs.
        inputs = self.filter_inputs(inputs)

        # 2. Reserved names (exclusion only).
        reserved_names = {p.name for p in self.get_reserved_parameters()}

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

        # 5. History -- the active conversation's own tail.
        logger.debug(f"Agent.{self.name} selecting turns")
        active_key, turns = self._resolve_context()

        # 6. Task lifecycle: initialize, think -> prepare -> act until complete.
        logger.debug(f"Agent.{self.name} performing async logic")
        task = await self._async_initialize_task(turns=turns, prompt=prompt, inputs=inputs)
        while not task.complete:
            task = await self.async_think(task)
            task = await self.async_prepare(task)
            task = await self.async_act(task)

        # 7. Output transformation.
        try:
            logger.debug(f"Agent.{self.name}.post_invoke postprocessing result asynchronously")
            post_inputs[self._post_result_key] = task.generated_response
            post_result = await self._post_invoke.async_invoke(post_inputs)
        except ToolInvocationError:
            raise
        except Exception as e:  # pragma: no cover
            raise AgentInvocationError(f"post_invoke Tool failed: {e}") from e

        final_response = post_result.result
        ended_at = datetime.now(timezone.utc)

        # 8-9. Record + result construction, committed to the active conversation.
        # _commit_emit acquires self._invoke_lock itself for the whole of
        # its own work -- no outer lock needed here.
        agent_result = self._commit_emit(
            task, turns, final_response, started_at, ended_at, active_key=active_key
        )

        logger.info(f"[Async {self.full_name} finished]")
        return agent_result

    def invoke(self, inputs: Mapping[str, Any]) -> AgentResult:
        """Invoke the agent with a single input mapping.

        Only the final commit runs under ``self._invoke_lock`` — everything
        before it (pre_invoke, turn selection, the full task lifecycle,
        post_invoke) is unlocked. The only shared, mutable state any of that
        touches (``self._conversations``) is only ever *mutated* inside
        ``_commit_emit``; reads elsewhere are stale-safe by construction
        (append-only, immutable-once-persisted, aside from the active-key
        snapshot ``_resolve_context`` takes up front). Narrowing the lock
        this way lets true concurrent execution of the actual work happen
        for both sync and async callers sharing one instance. A deliberate,
        accepted consequence: two concurrent calls targeting the same
        childless point may race, since nothing serializes the
        read-tail-then-commit span as a whole anymore -- resolved not by
        preventing the race but by ``_commit_emit``'s fresh children check
        at commit time, which automatically upgrades whichever commits
        second into a real fork instead of colliding.

        Steps
        -----
        1. ``filter_inputs`` — collect declared keys; inject defaults.
        2. Collect reserved names (see ``get_reserved_parameters``; none by
           default) for exclusion purposes only — ``inputs`` is never
           mutated.
        3. Slice ``pre_inputs`` / ``post_inputs`` from ``inputs`` by name
           membership against ``pre_invoke`` / ``post_invoke``'s own
           declared parameters, excluding reserved names.
        4. ``pre_invoke`` → prompt string (validated).
        5. ``_resolve_context()`` selects conversation turns from the
           active conversation's own tail, snapshotting the active-conversation key.
        6. ``_initialize_task(turns, prompt, inputs)`` → task; loop
           ``task = think(task); task = prepare(task); task = act(task)``
           until ``task.complete``.
        7. ``post_invoke`` transforms ``task.generated_response`` into the
           final result.
        8-9. ``_commit_emit(task, turns, result, started_at, ended_at,
           active_key=active_key)`` builds the completed record and
           ``AgentResult`` together, then commits (append-or-fork) or skips
           storage entirely per the context/window gate --
           ``_commit_emit`` acquires ``self._invoke_lock`` itself for the
           whole of this step; no outer lock is taken at this call site.
        """
        logger.info(f"[{self.full_name} started]")
        started_at = datetime.now(timezone.utc)

        # 1. Filter inputs.
        inputs = self.filter_inputs(inputs)

        # 2. Reserved names (exclusion only).
        reserved_names = {p.name for p in self.get_reserved_parameters()}

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

        # 5. History -- the active conversation's own tail.
        logger.debug(f"Agent.{self.name} selecting turns")
        active_key, turns = self._resolve_context()

        # 6. Task lifecycle: initialize, think -> prepare -> act until complete.
        logger.debug(f"Agent.{self.name} performing logic")
        task = self._initialize_task(turns=turns, prompt=prompt, inputs=inputs)
        while not task.complete:
            task = self.think(task)
            task = self.prepare(task)
            task = self.act(task)

        # 7. Output transformation.
        try:
            logger.debug(f"Agent.{self.name}.post_invoke postprocessing result")
            post_inputs[self._post_result_key] = task.generated_response
            post_result = self._post_invoke.invoke(post_inputs)
        except ToolInvocationError:
            raise
        except Exception as e:  # pragma: no cover
            raise AgentInvocationError(f"post_invoke Tool failed: {e}") from e

        final_response = post_result.result
        ended_at = datetime.now(timezone.utc)

        # 8-9. Record + result construction, committed to the active conversation.
        # _commit_emit acquires self._invoke_lock itself for the whole of
        # its own work -- no outer lock needed here.
        agent_result = self._commit_emit(
            task, turns, final_response, started_at, ended_at, active_key=active_key
        )

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
            "record_enabled": self.record_enabled,
            "records_window": self.records_window,
            "response_preview_limit": self.response_preview_limit,
            "conversations": {
                key: [r.to_dict() for r in conv.records]
                for key, conv in self._conversations.items()},
        })
        return d
