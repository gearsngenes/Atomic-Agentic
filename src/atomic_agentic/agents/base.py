from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Set as AbstractSet
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
)
from dataclasses import replace
from datetime import datetime, timezone
from uuid import uuid4
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

from ..constants.core import IDENTIFIER_PATTERN
from ..constants.agents import (
    DEFAULT_CONVERSATION_NAME,
    CONVERSATION_NAME_PATTERN,
    TASK_RESULT_PREFIX,
    PROMPT_PARAM,
)
from ..utils.parameters import (
    semantically_identical,
    insert_by_category,
    to_paramspec_list,
    build_parameter_reports,
    apply_parameter_reports,
)
from ..utils.rendering import (
    render_context_entries,
    render_task_block,
    stringify_value,
)

logger = logging.getLogger(__name__)

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
    ``invoke(inputs)`` runs the public steps spelled out in its docstring
    (``async_invoke`` mirrors it); each can also be called by hand to assemble
    a custom lifecycle. ``record`` is the only step that runs under the lock,
    and nothing is stored at all when ``record_enabled`` is false.

    Schema composition
    -------------------
    The agent's parameter schema is composed at construction time from four
    flat sources:

    - All ``pre_invoke`` parameters.
    - Post-only parameters not filled by a ``bind_response_to`` destination
      (those leave the public signature unless pre_invoke/extra_parameters
      also declares them), preserving their original declared kind (no
      forced ``KEYWORD_ONLY`` coercion).
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

    ``bind_response_to``
    --------------------
    Says which pieces of the generated response get which names: a ``str``
    (whole response), a collection of ``str`` (those keys of a mapping
    response), or a ``{selector: name}`` dict. A bound name equal to a
    ``post_invoke`` parameter fills it; nothing is passed to the post stage
    by default. ``post_invoke=None`` means no post stage. ``**kwargs`` is
    rejected in ``pre_invoke``/``post_invoke`` (``AgentError``).

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
    ``import_conversation`` (of a ``Conversation.slice``) or
    ``switch_conversation`` call before invoking (``create_conversation``/
    ``get_record``/``remove_conversation`` round out the rest of the explicit
    conversation-management API). Context mutation goes through the
    conversation itself (``agent.get_conversation(name).remember(...)``).
    There is no implicit forking: two invocations racing to extend the same
    tail both append to their bound conversation in commit order
    (``Conversation.commit`` serializes the commits), each with ``prev`` set
    to the record it read.

    A conversation may be shared: ``import_conversation`` registers the same
    object under a key, so several agents (of any type) can read and extend
    one history and context. Each reader renders the records with its own
    ``render_turn``. The registry itself (which keys exist, which is active)
    stays per-agent under ``_invoke_lock``.

    Context and recording
    ---------------------
    Context changes made during a call (``bind_response_to`` values) are
    written back to the conversation by ``record`` -- the same ``commit`` that
    appends the turn. So with ``record_enabled`` False neither happens: such an
    agent (a judge or probe) never changes a conversation, and its
    ``bind_response_to`` values reach only the result and the post stage.
    """

    # ------------------------------------------------------------------ #
    # Class-level configuration
    # ------------------------------------------------------------------ #
    # The only per-family variation ``create_task`` consults: which task
    # class to build and which system prompt it starts on. Further per-family
    # seeding goes through ``_seed_task``.
    _TASK_CLS: type[AgentTask] = AgentTask
    _INITIAL_PROMPT_NAME: str | None = None

    # Context-block placement and naming, read by _render_task_block. A family
    # overrides these as class attributes; nothing else about the block varies.
    _CONTEXT_NAME_PREFIX: str = ""   # "$" for the sigil families (Pass 14b)
    _CONTEXT_FIRST: bool = True      # context block above the task block

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
        bind_response_to: Optional[str | Iterable[str] | Mapping[Any, str]] = None,
        extra_parameters: list[str] | list[ParamSpec] | None = None,
        records_window: Optional[int] = None,
    ) -> None:

        # ── Pre-invoke ───────────────────────────────────────────────────────
        if pre_invoke is None:
            pre_tool = None  # no pre stage: the caller's prompt passes straight through
        elif isinstance(pre_invoke, AtomicInvokable):
            pre_tool = pre_invoke
        else:
            pre_tool = toolify(
                pre_invoke,
                name="pre_invoke",
                namespace=name,
                description=f"The tool that preprocesses inputs into a string for Agent {name}",
            )
        if pre_tool is not None and pre_tool.return_type.lower() not in {"any", "str"}:
            raise AgentError(
                "Agent.pre_invoke must return a type 'str'|'any'"
            )

        # ── Post-invoke ──────────────────────────────────────────────────────
        if post_invoke is None:
            post_tool = None  # no post stage: the response is returned as-is
        elif isinstance(post_invoke, AtomicInvokable):
            post_tool = post_invoke
        else:
            post_tool = toolify(
                post_invoke,
                name="post_invoke",
                namespace=name,
                description=f"The tool that postprocesses outputs of Agent {name}",
            )

        for tool, label in ((pre_tool, "pre_invoke"), (post_tool, "post_invoke")):
            for p in () if tool is None else tool.parameters:
                if p.kind == ParamSpec.VAR_KEYWORD:
                    raise AgentError(
                        f"Agent.{label} must not declare a **kwargs parameter "
                        f"({p.name!r}); every name it consumes must be enumerable at "
                        "construction. Wrap the callable to drop it."
                    )

        # ── Reserved and extra parameters ────────────────────────────────────
        reserved = self.get_reserved_parameters()
        reserved_names = {p.name for p in reserved}

        extra_raw = to_paramspec_list(extra_parameters)
        for p in extra_raw:
            if p.kind in (ParamSpec.VAR_POSITIONAL, ParamSpec.VAR_KEYWORD):
                raise AgentError(
                    f"extra_parameters entry {p.name!r} must not be variadic; "
                    f"got kind {p.kind!r}."
                )

        # ── Response bindings ────────────────────────────────────────────────
        # A bound post parameter that no other source declares is fed by the
        # response alone, so it leaves the public signature. One that
        # pre_invoke/extra_parameters also declares stays in the reconciliation
        # below: when the response lacks its key, the caller's value is what the
        # post receives, so the declarations must be compatible.
        bindings = self._normalize_bind_response_to(bind_response_to, reserved_names)
        bound_names = (
            set() if bindings is None
            else {bindings} if isinstance(bindings, str)
            else set(bindings.values())
        )
        pre_declared = list(pre_tool.parameters) if pre_tool is not None else [PROMPT_PARAM]
        declared_elsewhere = {p.name for p in pre_declared} | {
            p.name for p in extra_raw
        }
        response_filled = bound_names - declared_elsewhere

        # ── Schema composition ───────────────────────────────────────────────
        # Reserved overlaps are resolved per source, independently; the three
        # sources are then reconciled N-way (pre_invoke > post_invoke >
        # extra_parameters), raising on the first genuine conflict and warning
        # once for compatible-but-not-identical names.
        post_params = list(post_tool.parameters) if post_tool is not None else []
        pre_params = self._strip_reserved_overlaps(
            pre_declared, reserved, "pre_invoke"
        )
        post_only = self._strip_reserved_overlaps(
            [p for p in post_params if p.name not in response_filled],
            reserved,
            "post_invoke",
        )
        extra_params = self._strip_reserved_overlaps(extra_raw, reserved, "extra_parameters")
        reports = build_parameter_reports([pre_params, post_only, extra_params])
        constructed = apply_parameter_reports(
            reports, ("pre_invoke", "post_invoke", "extra_parameters"), error_cls=AgentError, stacklevel=4
        )
        agent_parameters = insert_by_category(insert_by_category([], constructed), reserved)

        # Lifecycle components and the per-consumer name sets create_task slices by.
        self._pre_invoke: Optional[AtomicInvokable] = pre_tool
        self._post_invoke: Optional[AtomicInvokable] = post_tool
        self._bind_response_to = bindings  # None | str | dict[Any, str]
        self._pre_names = frozenset(p.name for p in pre_declared) - reserved_names
        self._post_names = frozenset(p.name for p in post_params) - reserved_names
        self._agentic_names = frozenset(reserved_names | {p.name for p in extra_params})

        # System prompt registry — subclasses populate after super().__init__.
        self._system_prompts: dict[str, PromptConfig] = {}

        # Store Agent runtime configuration.
        self._llm_engine: LLMEngine = llm_engine
        self._record_enabled: bool = record_enabled

        if records_window is not None and (not type(records_window) is int or records_window < 0):
            raise AgentError("records_window must be an int >= 0 or be 'None'.")
        self._records_window: Optional[int] = records_window

        self._conversations: dict[str, Conversation] = {
            DEFAULT_CONVERSATION_NAME: self._new_conversation()
        }
        self._active_conversation: str = DEFAULT_CONVERSATION_NAME

        super().__init__(
            name=name,
            namespace=namespace,
            description=description,
            parameters=agent_parameters,
            return_type=(
                self._post_invoke.return_type if self._post_invoke is not None else "any"
            ),
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

    @staticmethod
    def _normalize_bind_response_to(
        value: Any, reserved_names: set[str]
    ) -> Optional[str | dict[Any, str]]:
        """Validate ``bind_response_to`` and return its normalized form.

        1. ``None`` -> ``None``; ``str`` -> the str itself (whole response);
           ``Mapping`` -> ``dict(value)``; any other non-bytes iterable of
           names (list, tuple, set, dict view, generator...) -> ``{d: d}``,
           sorted when unordered; anything else raises.
        2. An empty collection/mapping normalizes to ``None``.
        3. Destinations are stripped of surrounding whitespace (as
           ``ContextVariable``/``ParamSpec`` names are), then each must be an
           identifier string, must not start with
           ``TASK_RESULT_PREFIX``, and must not be a reserved parameter name.
        4. Duplicate destinations raise (checked before collapsing to a dict).
        """
        if value is None:
            return None

        if isinstance(value, str):
            selectors, names = None, [value]
        elif isinstance(value, Mapping):
            selectors, names = list(value), list(value.values())
        elif isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
            selectors, names = None, list(value)
            if isinstance(value, AbstractSet) and all(isinstance(n, str) for n in names):
                names.sort()  # unordered: store deterministically
        else:
            raise AgentError(
                "bind_response_to must be None, a str, a collection of str, "
                f"or a mapping of selector -> str; got {type(value).__name__}."
            )

        if not names:
            return None

        names = [n.strip() if isinstance(n, str) else n for n in names]
        for n in names:
            if not (isinstance(n, str) and IDENTIFIER_PATTERN.fullmatch(n)):
                raise AgentError(
                    f"bind_response_to destination {n!r} must be a valid identifier string."
                )
            if n.startswith(TASK_RESULT_PREFIX) or n in reserved_names:
                raise AgentError(
                    f"bind_response_to destination {n!r} is reserved (a framework-reserved "
                    f"parameter name or a {TASK_RESULT_PREFIX!r} name)."
                )
        if len(set(names)) != len(names):
            raise AgentError("bind_response_to has duplicate destinations.")

        if isinstance(value, str):
            return names[0]
        return dict(zip(names if selectors is None else selectors, names))

    # ------------------------------------------------------------------ #
    # Agent Properties
    # ------------------------------------------------------------------ #
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
        invocation -- regardless of ``records_window``. Context write-back
        (``bind_response_to`` values) is part of the same commit, so it is
        skipped too.
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
    def active_conversation(self) -> str:
        """Key of the currently active conversation."""
        return self._active_conversation

    @property
    def conversation_names(self) -> list[str]:
        """Snapshot list of every existing conversation key."""
        return list(self._conversations.keys())

    @property
    def pre_invoke(self) -> Optional[AtomicInvokable]:
        """Invokable that converts the input mapping into a prompt string, or
        ``None`` when there is no pre stage (the caller's ``prompt`` is used as-is)."""
        return self._pre_invoke

    @property
    def post_invoke(self) -> Optional[AtomicInvokable]:
        """Invokable that converts the response into the final output, or ``None``
        when there is no post stage (the response is returned as-is)."""
        return self._post_invoke

    @property
    def system_prompts(self) -> dict[str, PromptConfig]:
        """Shallow copy of the system prompt registry."""
        return dict(self._system_prompts)

    # ------------------------------------------------------------------ #
    # Agent Helpers
    # ------------------------------------------------------------------ #
    def render_turn(
        self,
        turn: AgentRecord,
        *,
        response_preview_limit: Optional[int] = None,
        prompt_preview_limit: Optional[int] = None,
        render_mode: str = "s",
        **kwargs: Any,
    ) -> List[Dict[str, str]]:
        """Render one canonical ``AgentRecord`` into LLM-facing messages.

        The assistant content is always ``turn.generated_response`` — the
        LLM's own raw output, never ``post_invoke``'s transformed result
        (not retained on the record at all). ``turn.user_prompt`` is the user
        message content — already a fully-resolved string, not a template.
        Each is cut by its limit (``None`` truncates nothing, ``0`` hides the
        text); ``summarize`` passes the bound conversation's limits and
        ``render_mode`` (how the response is formatted; the prompt is always a
        plain string), so called directly with no arguments nothing is
        truncated and the response is rendered with ``str``.

        ``**kwargs`` is accepted and ignored at this level — base rendering
        has no family-specific use for it. A subclass override consumes
        whatever keys it needs (e.g. ``ToolAgent``'s required ``index``)
        and forwards ``**kwargs`` (limits included) through its own
        ``super().render_turn`` call if it still wants this base behavior too.
        """
        if not isinstance(turn, AgentRecord):
            raise AgentInvocationError(
                f"render_turn expected AgentRecord, got {type(turn)!r}"
            )

        return [
            {
                "role": "user",
                "content": stringify_value(
                    turn.user_prompt, mode="s", limit=prompt_preview_limit
                ),
            },
            {
                "role": "assistant",
                "content": stringify_value(
                    turn.generated_response,
                    mode=render_mode,
                    limit=response_preview_limit,
                ),
            },
        ]

    # ------------------------------------------------------------------ #
    # Task-lifecycle steps
    # ------------------------------------------------------------------ #
    # Every step of invoke()/async_invoke() is a public method (see invoke's
    # docstring for the sequence). create_task is concrete and never
    # overridden; a family varies it only through _TASK_CLS /
    # _INITIAL_PROMPT_NAME and the _seed_task hook. think/prepare ship as
    # no-ops; act stays hard-abstract -- every subclass's advance-by-one-round
    # execution logic genuinely differs.
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
        ``started_at``. ``post_process`` writes bound response values into
        ``task.context`` (replace-only); any other writer must check a
        variable's ``readonly`` flag itself.

        The three views are slices of ``inputs`` by name -- views, not a
        partition: a name consumed by several buckets goes into each. Reserved
        names are excluded from pre/post. Post receives the caller inputs for
        its declared parameters, with bound values overlaid by ``post_process``.
        ``agentic_inputs`` holds the reserved parameters plus the
        ``extra_parameters``-sourced names
        (not every system-prompt placeholder -- a ``ToolAgent``'s ``{TOOLS}``
        etc. are render context, not agent parameters).

        The task class and starting system prompt come from ``_TASK_CLS`` and
        ``_INITIAL_PROMPT_NAME``; ``_seed_task`` then does any family-specific
        seeding. Not overridden by subclasses. Pure computation, no I/O.
        """
        with self._invoke_lock:
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

        task = self._TASK_CLS(
            turns=turns,
            inputs=inputs,
            system_prompt_name=self._INITIAL_PROMPT_NAME,
            conversation=active_key,
            context=dict(conversation.context),
            pre_inputs={k: v for k, v in inputs.items() if k in self._pre_names},
            post_inputs={k: v for k, v in inputs.items() if k in self._post_names},
            agentic_inputs={k: v for k, v in inputs.items() if k in self._agentic_names},
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
        is no compaction yet. Each turn is limited by the bound conversation's
        ``prompt_preview_limit`` / ``response_preview_limit`` / ``render_mode``,
        read here when the messages are built.
        """
        conv = self._conversations[task.conversation]
        rendered: list[dict[str, str]] = []
        for i, turn in enumerate(task.turns):
            rendered.extend(
                self.render_turn(
                    turn,
                    index=i,
                    response_preview_limit=conv.response_preview_limit,
                    prompt_preview_limit=conv.prompt_preview_limit,
                    render_mode=conv.render_mode,
                )
            )
        task.historic_messages = rendered
        return task

    async def async_summarize(self, task: AgentTask) -> AgentTask:
        """Async mirror of ``summarize``; pure computation, so it runs inline."""
        return self.summarize(task)

    def pre_process(self, task: AgentTask) -> AgentTask:
        """Run ``pre_invoke`` over ``task.pre_inputs`` to produce the prompt.

        Reads ``task.pre_inputs``; always overwrites ``task.user_prompt``;
        returns ``task``. With no ``pre_invoke`` the prompt is
        ``task.pre_inputs["prompt"]``. Errors raised by ``pre_invoke``
        propagate unchanged. A non-``str`` prompt raises
        ``AgentInvocationError`` -- a prompt string is required.
        """
        logger.debug(f"Agent.{self.name}.pre_invoke preprocessing inputs")
        if self._pre_invoke is None:
            raw_prompt, subject = task.pre_inputs["prompt"], "the prompt is not a string"
        else:
            raw_prompt = self._pre_invoke.invoke(task.pre_inputs).result
            subject = "pre_invoke returned a non-string result"
        task.user_prompt = self._require_prompt_str(raw_prompt, subject)
        return task

    async def async_pre_process(self, task: AgentTask) -> AgentTask:
        """Async mirror of ``pre_process``."""
        logger.debug(f"Agent.{self.name}.pre_invoke preprocessing inputs asynchronously")
        if self._pre_invoke is None:
            raw_prompt, subject = task.pre_inputs["prompt"], "the prompt is not a string"
        else:
            raw_prompt = (await self._pre_invoke.async_invoke(task.pre_inputs)).result
            subject = "pre_invoke returned a non-string result"
        task.user_prompt = self._require_prompt_str(raw_prompt, subject)
        return task

    @staticmethod
    def _require_prompt_str(raw_prompt: Any, subject: str) -> str:
        if not isinstance(raw_prompt, str):
            raise AgentInvocationError(
                f"{subject} (type={type(raw_prompt)!r}); a prompt string is required"
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
        """Async mirror of ``think``; the default is a no-op, so it runs inline.
        A family that overrides ``think`` must override this too."""
        return task

    def prepare(self, task: AgentTask) -> AgentTask:
        """Advance ``task``'s preparation phase by one round.

        Concrete no-op passthrough, same rationale as ``think``.
        """
        return task

    async def async_prepare(self, task: AgentTask) -> AgentTask:
        """Async mirror of ``prepare``; the default is a no-op, so it runs inline.
        A family that overrides ``prepare`` must override this too."""
        return task

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
        its own to render generically. A family whose task message carries
        the prompt and context builds its text with ``_render_task_block``.
        """
        ...

    def _render_task_block(self, task: AgentTask) -> str:
        """Build the single user-message text for a context-aware task.

        The wrapped current prompt, preceded (by default) by a context block.
        Private and overridable; ``BasicAgent`` calls it from
        ``_render_task_messages``, and later passes reuse it for the other
        families.

        1. ``limit`` is the bound conversation's ``context_preview_limit``,
           read now (not cached), so an edit made before this render is seen.
           A removed conversation raises ``KeyError``, as in ``summarize``
           (known hazard).
        2. ``render_context_entries`` over ``task.context`` (the call's working
           copy, never a first-round snapshot) with this family's name prefix.
        3. ``render_task_block`` wraps ``task.user_prompt`` and places the
           context block per ``_CONTEXT_FIRST``.

        Never modifies ``task.user_prompt`` or ``task.context``. The prompt is
        always wrapped; the context block appears only when ``task.context`` is
        non-empty (a variable with a hidden value still counts); the prompt is
        never truncated. A pure function of the task and the conversation's
        limit.
        """
        limit = self._conversations[task.conversation].context_preview_limit
        entries = render_context_entries(
            task.context, limit=limit, name_prefix=self._CONTEXT_NAME_PREFIX
        )
        return render_task_block(
            task.user_prompt, entries, context_first=self._CONTEXT_FIRST
        )

    def _bind_response(self, task: AgentTask) -> dict[str, Any]:
        """Slice the generated response per ``bind_response_to``.

        1. ``None`` binding -> ``{}``. A ``str`` binds the whole response; a
           mapping indexes each selector out of it (a missing key/index is
           skipped, a present ``None`` is bound, other errors propagate; a
           ``str``/``bytes`` response raises ``AgentInvocationError`` since it
           would index characters).
        2. Bound values named after post parameters overwrite those entries of
           ``task.post_inputs``; other bound names are not added. With no post
           stage nothing matches.

        Returns the resolved values (a skipped selector has no entry), which
        ``post_process`` hands to the context sink.
        """
        spec = self._bind_response_to
        if spec is None:
            return {}
        response = task.generated_response
        if isinstance(spec, str):
            bound = {spec: response}
        else:
            if isinstance(response, (str, bytes)):
                raise AgentInvocationError(
                    "bind_response_to selectors cannot index a str/bytes response; "
                    "use the str form to bind the whole response."
                )
            bound = {}
            for selector, destination in spec.items():
                try:
                    bound[destination] = response[selector]
                except (KeyError, IndexError):
                    continue
        task.post_inputs.update({k: v for k, v in bound.items() if k in self._post_names})
        return bound

    @staticmethod
    def _write_context(task: AgentTask, bound: Mapping[str, Any]) -> None:
        """Context sink: replace the value of each bound name that already
        exists, writable, in ``task.context``.

        A readonly variable is skipped and an absent name is ignored (nothing
        is created); description and flags are kept. Persisting beyond the call
        is ``record``'s existing write-back. Per-call state, so no lock.
        """
        for name, value in bound.items():
            variable = task.context.get(name)
            if variable is not None and not variable.readonly:
                task.context[name] = replace(variable, value=value)

    def post_process(self, task: AgentTask) -> AgentResult:
        """Resolve response bindings, run ``post_invoke``, write the context
        sink, and build the result.

        1. Slice ``task.generated_response`` per ``bind_response_to`` and, with a
           post stage, overlay the pieces named after post parameters onto
           ``task.post_inputs`` (can raise before anything runs).
        2. With no post stage the response is the result; otherwise ``post_invoke``
           runs over ``task.post_inputs``.
        3. Context sink: bound values replace existing writable ``task.context``
           variables (``_write_context``).
        4. Stamp ``task.ended_at``; build the ``AgentResult`` from the task.
        Errors from steps 1-2 propagate unchanged, before the sink and before
        ``record``, so nothing is written and the turn is not recorded.
        """
        logger.debug(f"Agent.{self.name}.post_invoke postprocessing result")
        bound = self._bind_response(task)
        if self._post_invoke is None:
            result = task.generated_response
        else:
            result = self._post_invoke.invoke(task.post_inputs).result
        self._write_context(task, bound)
        task.ended_at = datetime.now(timezone.utc)
        return self.build_result_from_task(task, result)

    async def async_post_process(self, task: AgentTask) -> AgentResult:
        """Async mirror of ``post_process``."""
        logger.debug(f"Agent.{self.name}.post_invoke postprocessing result asynchronously")
        bound = self._bind_response(task)
        if self._post_invoke is None:
            result = task.generated_response
        else:
            result = (await self._post_invoke.async_invoke(task.post_inputs)).result
        self._write_context(task, bound)
        task.ended_at = datetime.now(timezone.utc)
        return self.build_result_from_task(task, result)

    def _record_base_fields(self, task: AgentTask) -> dict[str, Any]:
        """Common ``AgentRecord`` kwargs for a finished task.

        Identity and timing come from the task; ``prev`` is the last turn this
        call read (``task.turns[-1]``), ``None`` when it read none;
        ``agent_id`` names this agent as the writer. Family overrides of
        ``_build_record_from_task`` splat this and add their own fields.
        """
        return {
            "user_prompt": task.user_prompt,
            "generated_response": task.generated_response,
            "run_id": task.run_id,
            "started_at": task.started_at,
            "ended_at": task.ended_at,
            "inputs": task.inputs,
            "llm_records": tuple(task.llm_records),
            "prev": task.turns[-1] if task.turns else None,
            "agent_id": f"{self.full_name};{self.instance_id}",
        }

    def _build_record_from_task(self, task: AgentTask) -> AgentRecord:
        """Assemble a complete AgentRecord from a finished AgentTask.

        ``BasicAgent`` uses this base implementation as-is; ``PlanActAgent``/
        ``ReActAgent``/``ScriptActAgent``/``ThinkingAgent`` each override it
        to return their own richer record subclass, built from
        ``_record_base_fields`` plus their own fields.
        """
        return AgentRecord(**self._record_base_fields(task))

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

        ``llm_model_data`` comes from the engine's configuration, so a task
        that finished without any LLM call still builds a result.
        """
        llm_token_usage = tuple(r.llm_result.token_usage for r in task.llm_records)
        llm_model_data = self._llm_engine.model_data

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

        Reads the task's record fields and bound conversation name; returns
        nothing. Shared by ``invoke`` and ``async_invoke``.

        The record is built first (independent of the result). With
        ``record_enabled`` False nothing is stored. Otherwise the conversation
        bound at ``create_task`` is looked up under ``self._invoke_lock``
        (even if the active one has since changed) and ``Conversation.commit``
        runs outside it, so the agent and conversation locks are never held
        together; see ``commit`` for the append, ``children`` link and context
        write-back rules. There is no implicit forking: branching happens only
        through explicit ``import_conversation``.

        ``prev`` is "the last turn this call read" (``task.turns[-1]``) and is
        provenance only: turn selection never uses it. ``children`` is
        likewise provenance, shared by every conversation that contains a
        record, so a record continued by two racing calls or two agents simply
        has two children. With ``records_window == 0`` the record is a chain
        root (``prev=None``).
        """
        if not self._record_enabled:
            return
        record = self._build_record_from_task(task)

        with self._invoke_lock:
            conv = self._conversations[task.conversation]
        conv.commit(record, task.context)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def clear_memory(self) -> None:
        """Factory-reset all conversation storage to the just-constructed state.

        Every conversation is unregistered from this agent, including its
        context and preview limits (a conversation shared with another agent
        stays intact there); ``self._conversations`` becomes a single fresh
        ``"default"`` built by ``_new_conversation()`` and the active
        conversation resets to ``"default"``.
        Takes ``self._invoke_lock`` like every other registry-mutating method
        -- without it, an in-flight ``invoke``/``async_invoke`` whose
        ``create_task`` snapshot predates this call could have its later
        ``record`` reference a conversation key this reset just removed.
        """
        with self._invoke_lock:
            self._conversations = {
                DEFAULT_CONVERSATION_NAME: self._new_conversation()
            }
            self._active_conversation = DEFAULT_CONVERSATION_NAME

    def _new_conversation(self) -> Conversation:
        """Build one fresh, empty default conversation for this agent.

        1. return ``Conversation()`` -- all default limits and render_mode
           ``"s"``.
        2. pure: no I/O, no lock, and it must not read ``AtomicInvokable``
           state or anything a subclass sets after ``Agent.__init__`` (it runs
           inside ``Agent.__init__`` before the invokable parts are
           initialized).

        The single source of every default conversation this agent builds: the
        initial ``"default"``, the one rebuilt by ``clear_memory()``, the one
        rebuilt by ``remove_conversation("default")``, and the starting point
        ``create_conversation`` overlays its arguments onto. A subclass
        overrides it to give its conversations different defaults
        (``BasicAgent``: render_mode ``"j"`` when it has a response_schema).
        """
        return Conversation()

    def _check_new_conversation_name(self, name: str, caller: str) -> None:
        """Raise ``ValueError`` if ``name`` is badly shaped or already taken.

        The caller holds ``self._invoke_lock``.
        """
        if not CONVERSATION_NAME_PATTERN.fullmatch(name):
            raise ValueError(
                f"{caller}: invalid conversation name {name!r} (must be "
                "alphanumeric/underscore)."
            )
        if name in self._conversations:
            raise ValueError(f"{caller}: conversation {name!r} already exists.")

    def get_conversation(self, conversation_id: str | None = None) -> Conversation:
        """Return the stored conversation itself (not a snapshot).

        Context is changed through the returned object
        (``agent.get_conversation().remember(...)``), whose mutators are
        atomic under the conversation's own lock; its preview limits are
        assigned directly
        (``agent.get_conversation().response_preview_limit = 500``).
        ``conversation_id`` is a registry key, not a record identity: a
        single ``AgentRecord`` can belong to more than one conversation's
        list, and one conversation object can be registered in several agents.

        ``conversation_id=None`` resolves to the active conversation. An
        unknown key raises ``AgentInvocationError``.
        """
        with self._invoke_lock:
            key = self._active_conversation if conversation_id is None else conversation_id
            if key not in self._conversations:
                raise AgentInvocationError(f"get_conversation: unknown conversation {key!r}.")
            return self._conversations[key]

    def create_conversation(
        self,
        name: str,
        *,
        response_preview_limit: int | None = None,
        prompt_preview_limit: int | None = None,
        context_preview_limit: int | None = None,
        render_mode: str | None = None,
    ) -> Conversation:
        """Create a new, empty conversation and return the stored object.

        ``name`` must be alphanumeric-plus-underscore; a bad shape or a name
        that already exists raises ``ValueError`` with a specific message.

        1. under ``self._invoke_lock``: check the name.
        2. start from ``self._new_conversation()``.
        3. assign each argument that is not ``None`` onto it. The three
           preview limits (``None`` unlimited, ``0`` hidden, else
           ``int > 0``) and ``render_mode`` (how prior responses are formatted
           in history; omitted -> the agent's default) are validated by
           ``Conversation`` (``ValueError``); a failure leaves nothing
           registered.
        4. register and return it. Never auto-activates.

        ``None`` means "not given" for every argument, so a limit cannot be
        set back to ``None`` over a default the agent's hook already set.
        """
        with self._invoke_lock:
            self._check_new_conversation_name(name, "create_conversation")
            conv = self._new_conversation()
            given = {
                "response_preview_limit": response_preview_limit,
                "prompt_preview_limit": prompt_preview_limit,
                "context_preview_limit": context_preview_limit,
                "render_mode": render_mode,
            }
            for field, value in given.items():
                if value is not None:
                    setattr(conv, field, value)
            self._conversations[name] = conv
            return conv

    def get_record(
        self,
        *,
        run_id: str | None = None,
        conversation: str | None = None,
        index: int | None = None,
    ) -> AgentRecord:
        """Single-record lookup. Valid combinations:

        - ``run_id`` alone: first match scanning every conversation.
        - ``run_id`` + ``conversation``: first match in that conversation.
        - ``conversation`` + ``index``: ``records[index]``.
        - ``index`` alone: index into the active conversation.

        Anything else (``run_id`` + ``index``, ``conversation`` alone,
        nothing) raises ``ValueError``. An unknown ``conversation`` raises
        ``AgentInvocationError``; no record with the ``run_id`` raises
        ``ValueError``; an out-of-range ``index`` raises the natural
        ``IndexError``.
        """
        if run_id is not None:
            if index is not None:
                raise ValueError("get_record: run_id may not be combined with index.")
            if conversation is None:
                with self._invoke_lock:
                    convs = list(self._conversations.values())
            else:
                convs = [self.get_conversation(conversation)]
            for conv in convs:
                found = next((r for r in conv.records if r.run_id == run_id), None)
                if found is not None:
                    return found
            raise ValueError(f"get_record: no record with run_id {run_id!r} found.")

        if index is None:
            raise ValueError(
                "get_record: must supply run_id, conversation + index, or index."
            )
        return self.get_conversation(conversation).records[index]

    def import_conversation(self, conversation: Conversation, *, name: str) -> None:
        """Register ``conversation`` (the SAME object, shared) under ``name``.

        ``conversation`` is typically another agent's ``get_conversation``
        result, so two agents (of any type) can share one history and
        context. Never copies; to copy a range:
        ``b.import_conversation(a.get_conversation("x").slice(2, 5), name="y")``.
        Never activates; returns ``None``.

        Not a ``Conversation`` raises ``TypeError`` (before anything is
        registered). ``name`` is validated/collision-checked exactly like
        ``create_conversation``'s (``ValueError``); the same object already
        registered under another key of this agent also raises ``ValueError``.
        """
        if not isinstance(conversation, Conversation):
            raise TypeError(
                "import_conversation: expected a Conversation, "
                f"got {type(conversation)!r}."
            )

        with self._invoke_lock:
            self._check_new_conversation_name(name, "import_conversation")
            if any(c is conversation for c in self._conversations.values()):
                raise ValueError(
                    "import_conversation: conversation is already registered here."
                )
            self._conversations[name] = conversation

    def switch_conversation(self, key: str) -> None:
        """Switch the active conversation. Unknown key raises ``AgentInvocationError``."""
        with self._invoke_lock:
            if key not in self._conversations:
                raise AgentInvocationError(
                    f"switch_conversation: unknown conversation {key!r}."
                )
            self._active_conversation = key

    def remove_conversation(self, key: str) -> Conversation:
        """Unregister ``key`` and return the popped conversation.

        Other agents that share the conversation keep it. ``"default"`` is
        never actually removed -- regardless of whether it is currently
        active, removing it stores a fresh empty conversation built by
        ``_new_conversation()`` under that key, leaves the active key
        unchanged and returns the prior one. Any other key is refused while it is the active
        conversation (switch first); otherwise it is popped and returned. An
        unknown key raises ``AgentInvocationError``.
        """
        with self._invoke_lock:
            if key not in self._conversations:
                raise AgentInvocationError(
                    f"remove_conversation: unknown conversation {key!r}."
                )
            if key == DEFAULT_CONVERSATION_NAME:
                old = self._conversations[key]
                self._conversations[key] = self._new_conversation()
                return old
            if key == self._active_conversation:
                raise AgentInvocationError(
                    f"remove_conversation: cannot remove the active "
                    f"conversation {key!r}; switch first."
                )
            return self._conversations.pop(key)

    async def async_invoke(self, inputs: Mapping[str, Any]) -> AgentResult:
        """Async analog of ``invoke``.

        The same lifecycle with ``await`` at ``pre_process``/``post_process``
        and the ``async_*`` hooks. ``create_task``, ``summarize`` and
        ``record`` are sync (pure computation, no I/O) and serve both paths.
        Same lock scope as ``invoke``; see its docstring.
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

        Only ``record`` takes locks (``self._invoke_lock`` for the registry
        lookup, the conversation's own lock for the commit); everything
        before it is unlocked, so concurrent calls on one instance genuinely
        overlap. Reads are stale-safe (records are append-only and
        ``create_task`` pins its turn copy). Accepted consequence: two calls
        reading the same tail both append in commit order, each with ``prev``
        set to the record it read (so that record has two children).

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
        with self._invoke_lock:
            conversations = dict(self._conversations)
        d = super().to_dict()
        d.update({
            "system_prompts": {
                key: {"template": cfg.template, "description": cfg.description}
                for key, cfg in self._system_prompts.items()
            },
            "pre_invoke": self.pre_invoke.to_dict() if self.pre_invoke is not None else None,
            "post_invoke": self.post_invoke.to_dict() if self.post_invoke is not None else None,
            "bind_response_to": (
                dict(self._bind_response_to)
                if isinstance(self._bind_response_to, dict)
                else self._bind_response_to
            ),
            "llm": self._llm_engine.to_dict(),
            "record_enabled": self.record_enabled,
            "records_window": self.records_window,
            "conversations": {
                key: [r.to_dict() for r in conv.records]
                for key, conv in conversations.items()},
        })
        return d
