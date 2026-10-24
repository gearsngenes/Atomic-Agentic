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

    ``Agent`` owns the full invocation lifecycle shell as a sequence of
    named, public steps (``create_task``, ``summarize``, ``pre_process``,
    ``post_process``, ``record``) so a caller can assemble their own
    lifecycle by hand, but delegates the actual LLM work to the ``think``/
    ``prepare``/``act`` abstract-or-overridable hooks that concrete
    subclasses implement. A family varies ``create_task`` only through the
    ``_TASK_CLS``/``_INITIAL_PROMPT_NAME`` class attributes and the
    ``_seed_task`` hook -- ``create_task`` itself is never overridden.

    Lifecycle
    ---------
    ``invoke(inputs)`` follows this sequence (``async_invoke`` mirrors it):

    1. ``filter_inputs`` collects declared keys and injects defaults.
    2. ``create_task`` binds the active conversation, pins the windowed
       turns, mints ``run_id``/``started_at``, slices the filtered ``inputs``
       into the ``pre_inputs``/``post_inputs``/``agentic_inputs`` views
       (reserved names are excluded from the pre/post views), and builds
       this family's task.
    3. ``summarize`` renders ``task.turns`` into ``task.historic_messages``.
    4. ``pre_process`` converts ``task.pre_inputs`` to the prompt string.
    5. Every round, unconditionally: ``task = think(task); task =
       prepare(task); task = act(task)`` — until ``task.complete``.
    6. ``post_process`` transforms the raw response through ``post_invoke``
       and builds the final ``AgentResult`` from the task.
    7. ``record`` builds the ``AgentRecord`` from the task and appends it to
       the conversation the task is bound to, under the lock, unless
       ``record_enabled`` is false, in which case nothing is stored at all.
    8. The step-6 result is returned.

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
    out the rest of the explicit conversation-management API). There is no
    implicit forking: two invocations racing to extend the same tail both
    append to their bound conversation in commit order (``_invoke_lock``
    serializes the commits), each with ``prev`` set to the record it read.
    """

    # ------------------------------------------------------------------ #
    # Class-level configuration
    # ------------------------------------------------------------------ #
    # The only per-family variation ``create_task`` consults: which task
    # class to build and which system prompt it starts on. Further per-family
    # seeding goes through ``_seed_task``.
    _TASK_CLS: type[AgentTask] = AgentTask
    _INITIAL_PROMPT_NAME: str | None = None

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
        self._extra_names: frozenset[str] = frozenset(p.name for p in extra_params)

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
        axis). ``True``: the completed turn is appended to the conversation
        the call was bound to. ``False``: a normal ``AgentResult`` is still
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
    # Task-lifecycle steps
    # ------------------------------------------------------------------ #
    # Every step of invoke()/async_invoke() is a public method that takes the
    # task (or, for create_task, the filtered inputs), so a caller can
    # assemble their own lifecycle: filter_inputs -> create_task -> summarize
    # -> pre_process -> (think -> prepare -> act)* -> post_process -> record.
    # create_task is concrete and never overridden; a family varies it only
    # through _TASK_CLS / _INITIAL_PROMPT_NAME and the _seed_task hook.
    # think/prepare ship as no-ops; act stays hard-abstract -- every
    # subclass's advance-by-one-round execution logic genuinely differs.
    def create_task(self, inputs: Mapping[str, Any]) -> AgentTask:
        """Build this invocation's task from the filtered ``inputs``.

        Reads: the active conversation (name and records), ``records_window``,
        the pre/post/agentic parameter names, ``inputs``. Writes (onto the new
        task): ``conversation`` (the active name, snapshotted here so a later
        ``switch_conversation`` cannot retarget this call), ``turns`` (a
        pinned copy of the window: ``records_window == 0`` -> ``[]``,
        ``None`` -> all, ``N`` -> the last ``N``), ``inputs`` (the same full
        mapping, untouched), ``context`` (a shallow copy of the bound
        conversation's context taken here -- the seed point; a later
        ``remember()`` is not visible to the task), the ``pre_inputs``/
        ``post_inputs``/``agentic_inputs`` views, ``run_id`` and
        ``started_at``. The base loop never writes ``task.context``; a
        subclass writer must check a variable's ``readonly`` flag itself.

        The three views are slices of ``inputs`` by name -- views, not a
        partition: a name consumed by several buckets goes into each. Reserved
        names are excluded from pre/post; ``post_result_key`` is excluded from
        post (``post_process`` supplies it). ``agentic_inputs`` holds the
        reserved parameters plus the ``extra_parameters``-sourced names
        (not every system-prompt placeholder -- a ``ToolAgent``'s ``{TOOLS}``
        etc. are render context, not agent parameters).

        The task class and starting system prompt come from ``_TASK_CLS`` and
        ``_INITIAL_PROMPT_NAME``; ``_seed_task`` then does any family-specific
        seeding. Not overridden by subclasses. Pure computation, no I/O.
        """
        active_key = self._active_conversation
        conversation = self._conversations[active_key]
        history = conversation.records
        window = self._records_window
        if window == 0:
            turns: list[AgentRecord] = []
        elif window is None:
            turns = list(history)
        else:
            turns = history[-window:]

        reserved_names = {p.name for p in self.get_reserved_parameters()}
        pre_names = {p.name for p in self._pre_invoke.parameters} - reserved_names
        post_names = (
            {p.name for p in self._post_invoke.parameters}
            - {self._post_result_key}
            - reserved_names
        )
        agentic_names = reserved_names | self._extra_names

        task = self._TASK_CLS(
            turns=turns,
            inputs=inputs,
            system_prompt_name=self._INITIAL_PROMPT_NAME,
            conversation=active_key,
            context=dict(conversation.context),
            pre_inputs={k: v for k, v in inputs.items() if k in pre_names},
            post_inputs={k: v for k, v in inputs.items() if k in post_names},
            agentic_inputs={k: v for k, v in inputs.items() if k in agentic_names},
            run_id=uuid4().hex,
            started_at=datetime.now(timezone.utc),
        )
        self._seed_task(task)
        return task

    def _seed_task(self, task: AgentTask) -> None:
        """Family-specific seeding of a freshly built task, in place.

        Called once by ``create_task`` after construction. Default: nothing
        to seed. ``ToolAgent`` seeds its cache and constants; ``ThinkingAgent``
        validates and sets its round budget.
        """

    def summarize(self, task: AgentTask) -> AgentTask:
        """Render ``task.turns`` into ``task.historic_messages``.

        Reads ``task.turns``; writes ``task.historic_messages``; returns
        ``task``. Each turn is rendered with its 0-based position within
        *this invocation's* ``task.turns`` passed as ``index`` --
        window-relative (after any ``records_window`` truncation), not the
        turn's position in the full stored conversation. Base rendering
        ignores it; ``ToolAgent``'s ``render_turn`` requires it. Empty
        ``turns`` leaves ``historic_messages`` empty. Renders only -- there
        is no compaction yet.
        """
        rendered: list[dict[str, str]] = []
        for i, turn in enumerate(task.turns):
            rendered.extend(self.render_turn(turn, index=i))
        task.historic_messages = rendered
        return task

    async def async_summarize(self, task: AgentTask) -> AgentTask:
        """Async mirror of ``summarize``; default offloads to a worker thread."""
        return await asyncio.to_thread(self.summarize, task)

    def pre_process(self, task: AgentTask) -> AgentTask:
        """Run ``pre_invoke`` over ``task.pre_inputs`` to produce the prompt.

        Reads ``task.pre_inputs``; always overwrites ``task.user_prompt``;
        returns ``task``. Errors raised by ``pre_invoke`` propagate
        unchanged. A non-``str`` result raises ``AgentInvocationError`` --
        a prompt string is required.
        """
        logger.debug(f"Agent.{self.name}.pre_invoke preprocessing inputs")
        raw_prompt = self._pre_invoke.invoke(task.pre_inputs).result
        task.user_prompt = self._require_prompt_str(raw_prompt)
        return task

    async def async_pre_process(self, task: AgentTask) -> AgentTask:
        """Async mirror of ``pre_process``."""
        logger.debug(f"Agent.{self.name}.pre_invoke preprocessing inputs asynchronously")
        raw_prompt = (await self._pre_invoke.async_invoke(task.pre_inputs)).result
        task.user_prompt = self._require_prompt_str(raw_prompt)
        return task

    @staticmethod
    def _require_prompt_str(raw_prompt: Any) -> str:
        if not isinstance(raw_prompt, str):
            raise AgentInvocationError(
                f"pre_invoke returned a non-string result "
                f"(type={type(raw_prompt)!r}); a prompt string is required"
            )
        return raw_prompt

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
        task_messages = self._render_task_messages(task)
        task_messages.extend(additional_messages or [])
        return system + task.historic_messages + task_messages

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

    def post_process(self, task: AgentTask) -> AgentResult:
        """Run ``post_invoke`` over the raw response and build the result.

        Reads ``task.post_inputs`` and ``task.generated_response``; writes
        ``task.ended_at``; returns the final ``AgentResult``, built directly
        from the task (the record is built independently by ``record``).
        ``task.post_inputs`` itself is not mutated -- a copy gains the
        ``post_result_key`` entry. Errors raised by ``post_invoke`` propagate
        unchanged.
        """
        logger.debug(f"Agent.{self.name}.post_invoke postprocessing result")
        post_inputs = {**task.post_inputs, self._post_result_key: task.generated_response}
        post_result = self._post_invoke.invoke(post_inputs)
        task.ended_at = datetime.now(timezone.utc)
        return self.build_result_from_task(task, post_result.result)

    async def async_post_process(self, task: AgentTask) -> AgentResult:
        """Async mirror of ``post_process``."""
        logger.debug(f"Agent.{self.name}.post_invoke postprocessing result asynchronously")
        post_inputs = {**task.post_inputs, self._post_result_key: task.generated_response}
        post_result = await self._post_invoke.async_invoke(post_inputs)
        task.ended_at = datetime.now(timezone.utc)
        return self.build_result_from_task(task, post_result.result)

    def _build_record_from_task(self, task: AgentTask) -> AgentRecord:
        """Assemble a complete AgentRecord from a finished AgentTask.

        ``BasicAgent`` uses this base implementation as-is; ``PlanActAgent``/
        ``ReActAgent``/``ScriptActAgent`` each override it directly (not via
        a shared ``ToolAgent`` implementation) to return their own richer
        ``ToolAgentRecord``-family (or, for ``ScriptActAgent``,
        ``ScriptActAgentRecord``) subclass. Identity and timing come from the
        task; ``prev`` is the last turn this call read (``task.turns[-1]``),
        ``None`` when it read none.
        """
        prev = task.turns[-1] if task.turns else None
        return AgentRecord(
            user_prompt=task.user_prompt,
            generated_response=task.generated_response,
            run_id=task.run_id,
            started_at=task.started_at,
            ended_at=task.ended_at,
            inputs=task.inputs,
            llm_records=tuple(task.llm_records),
            prev=prev,
        )

    # ------------------------------------------------------------------ #
    # Result construction
    # ------------------------------------------------------------------ #
    def build_result_from_task(self, task: AgentTask, result: Any) -> AgentResult:
        """Construct this Agent's ``AgentResult`` envelope directly from a
        finished task, wrapping the low-level ``_make_result`` primitive.

        Public -- external subclasses may legitimately override this to
        choose a more specific result class or add subclass-specific result
        fields. Built independently of the record: both read the task.

        ``result`` (the post-``post_invoke`` final caller-facing value) is not
        on the task, so it stays an explicit parameter -- distinct from
        ``task.generated_response``, the pre-``post_invoke`` raw value.
        ``run_id`` and ``started_at``/``ended_at`` come from the task.
        """
        llm_token_usage = tuple(r.llm_result.token_usage for r in task.llm_records)
        llm_model_data = task.llm_records[-1].llm_result.model_data

        return self._make_result(
            result=result,
            started_at=task.started_at,
            ended_at=task.ended_at,
            result_cls=AgentResult,
            run_id=task.run_id,
            llm_token_usage=llm_token_usage,
            llm_model_data=llm_model_data,
        )

    def record(self, task: AgentTask) -> None:
        """Build this invocation's ``AgentRecord`` from the task and commit it.

        Reads the task's record fields, ``turns``, ``conversation``,
        ``run_id`` and timing; returns nothing. Shared by ``invoke`` and
        ``async_invoke`` -- pure computation plus a brief lock, no I/O.

        1. The record is built from the task (independent of the result) and
           validated as an ``AgentRecord``.
        2. Under ``self._invoke_lock``: if ``record_enabled`` is False,
           nothing is stored. Otherwise the record is appended to
           ``task.conversation``'s ``.records`` (the conversation bound at
           ``create_task``, even if the active one has since changed), and
           appended to ``record.prev.children`` when ``prev`` is set. There
           is no implicit forking: branching happens only through explicit
           ``clone_conversation``/``import_conversation``.
        3. Context write-back, same lock, same conversation: every ``task.context`` variable whose name is in the
           destination's context and whose destination variable is not
           ``readonly`` is written with one ``remember_many``. Scratch keys
           and keys ``forget``-ed mid-call are dropped; readonly ones are
           skipped silently. Every eligible key is written, changed or not.

        Context limitations: last-writer-wins for keys this call started
        with (a mid-call ``remember()`` or another call's commit to such a
        key is overwritten); stored values are shared by reference. A
        variable a writer marked ``readonly=True`` persists as readonly when
        the destination's was not.

        ``prev`` is "the last turn this call read" (``task.turns[-1]``) and is
        provenance only: turn selection slices the conversation's record list
        and never uses ``prev``. ``children`` is likewise provenance, shared
        by every conversation that contains a record, so a record continued
        in two conversations (or by two racing calls) simply has two
        children. Records land in commit order; with ``records_window == 0``
        (nothing read) the record is a chain root (``prev=None``).
        """
        record = self._build_record_from_task(task)
        if not isinstance(record, AgentRecord):
            raise AgentInvocationError(
                f"_build_record_from_task returned non-AgentRecord (type={type(record)!r})"
            )

        with self._invoke_lock:
            if not self._record_enabled:
                return

            dest_conv = self._conversations[task.conversation]
            dest_conv.records.append(record)
            if record.prev is not None:
                record.prev.children.append(record)

            # Context write-back. The destination's flags are read fresh under
            # the lock, so they are authoritative even if a writer ignored
            # `readonly`. A task key absent from the destination (call-local
            # scratch, or forgotten mid-call) is dropped; a readonly
            # destination variable is skipped silently.
            dest_context = dest_conv.context
            dest_conv.remember_many(
                variable
                for variable in task.context.values()
                if variable.name in dest_context
                and not dest_context[variable.name].readonly
            )

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
        ``invoke``/``async_invoke`` whose ``create_task`` snapshot
        predates this call could have its later ``record`` reference a
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
        is a dict key, not a record identity: once cloning exists, a single
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
        needs one now that ``create_task`` no longer resolves an
        arbitrary target).
        """
        history = self._conversations[conversation_id].records
        return next(
            (r for r in history if r.run_id == run_id), None
        )

    def create_conversation(self, name: str) -> Conversation:
        """Create a new, empty conversation.

        ``name`` must be alphanumeric-plus-underscore and must not end in
        ``_<digits>`` (that suffix shape is reserved for auto-generated clone
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

        The same lifecycle with ``await`` at ``summarize``/``pre_process``/
        ``post_process`` and the ``async_think``/``async_prepare``/
        ``async_act`` hooks. ``create_task`` and ``record`` are sync (pure
        computation, no I/O) and serve both paths. Same lock scope as
        ``invoke``: only ``record`` runs under ``self._invoke_lock`` (a
        ``threading.RLock`` -- safe to hold briefly here since ``record``
        never awaits), letting concurrent ``async_invoke`` calls against the
        same instance genuinely overlap everywhere else. See ``invoke``'s own
        docstring for the full reasoning.
        """
        logger.info(f"[Async {self.full_name} started]")

        inputs = self.filter_inputs(inputs)
        task = self.create_task(inputs)
        task = await self.async_summarize(task)
        task = await self.async_pre_process(task)

        logger.debug(f"Agent.{self.name} performing async logic")
        while not task.complete:
            task = await self.async_think(task)
            task = await self.async_prepare(task)
            task = await self.async_act(task)

        result = await self.async_post_process(task)
        self.record(task)

        logger.info(f"[Async {self.full_name} finished]")
        return result

    def invoke(self, inputs: Mapping[str, Any]) -> AgentResult:
        """Invoke the agent with a single input mapping.

        Only ``record`` runs under ``self._invoke_lock`` -- everything before
        it (turn selection, pre_invoke, the full task lifecycle, post_invoke)
        is unlocked. The only shared, mutable state any of that touches
        (``self._conversations``) is only ever *mutated* inside ``record``;
        reads elsewhere are stale-safe by construction (append-only,
        immutable-once-persisted, aside from the active-key snapshot and the
        pinned turn copy ``create_task`` takes up front). Narrowing the lock
        this way lets true concurrent execution of the actual work happen
        for both sync and async callers sharing one instance. A deliberate,
        accepted consequence: two concurrent calls reading the same tail may
        race, since nothing serializes the read-tail-then-commit span as a
        whole -- both simply append in commit order, each with ``prev`` set
        to the record it read (so that record has two children).

        The lifecycle, spelled out -- each step is public, so this is also the
        recipe for assembling your own::

            inputs = self.filter_inputs(inputs)
            task = self.create_task(inputs)
            task = self.summarize(task)
            task = self.pre_process(task)
            while not task.complete:
                task = self.think(task)
                task = self.prepare(task)
                task = self.act(task)
            result = self.post_process(task)
            self.record(task)
            return result
        """
        logger.info(f"[{self.full_name} started]")

        inputs = self.filter_inputs(inputs)
        task = self.create_task(inputs)
        task = self.summarize(task)
        task = self.pre_process(task)

        logger.debug(f"Agent.{self.name} performing logic")
        while not task.complete:
            task = self.think(task)
            task = self.prepare(task)
            task = self.act(task)

        result = self.post_process(task)
        self.record(task)

        logger.info(f"[{self.full_name} finished]")
        return result

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
