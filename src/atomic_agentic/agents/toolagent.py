"""
ToolAgent: shared ancestor for every tool-registering Agent family.

Owns the surface that ``PlanActAgent``/``ReActAgent`` and ``ScriptActAgent``
share: tool registration (alias-based, ``register_tool``/``register_tools``),
the optional user-defined ``instructions``, the execution knobs common to every
grammar (``tool_calls_limit``, ``regeneration_limit``,
``tool_concurrency_limit``), and the context-native task frame. There are no
registered constants: a value the model may reference lives in
``task.context`` (conversation variables, runtime and input-bound values, and
the ``task_result_i`` entries ``_seed_task`` adds), shown in the runtime
context block like any other variable.

The task-message frame is concrete here (``_render_task_messages``): the
wrapped task block, then -- only when the family has steps to show -- an
assistant message framed as ``STEPS``, then a closing user directive. The
family supplies only the two string hooks (``_render_steps``,
``_render_directive``); the ``think``/``prepare``/``act`` (+ async mirror)
hooks stay ``@abstractmethod``, since each family's generation, resolution and
dispatch logic genuinely differs.

``fail_fast`` is deliberately NOT here: ``PlanActAgent``'s version governs
whole-batch cascade-vs-abort, ``ReActAgent``'s governs single-call
raise-vs-tolerate -- different granularity, coincidentally the same name and
type, never one shared feature. Each declares its own independently.
``ScriptActAgent`` has no concept of it at all.
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import (
    Any,
    Callable,
    ClassVar,
    Literal,
    Mapping,
    Optional,
)
from collections.abc import Iterable

from .base import Agent
from ..models.agents.records import AgentRecord
from ..models.agents.context import ContextVariable
from ..models.agents.tasks import AgentTask, ToolAgentTask
from ..exceptions import (
    ToolAgentError,
    ToolRegistrationError,
)
from ..constants.agents import INSTRUCTIONS_PRECEDENCE, TASK_RESULT_PREFIX, TOOLS_PREAMBLE
from ..constants.core import IDENTIFIER_PATTERN
from ..core.Invokable import AtomicInvokable
from ..llm.base import LLMEngine
from ..tools import toolify
from ..mcp import MCPClientHub
from ..a2a import A2AClientHub, PyA2AtomicClient
from ..models.agents.prompts import PromptConfig
from ..utils.agents import normalize_prompt_config
from ..utils.rendering import frame_content


class ToolAgent(Agent, ABC):
    """
    Abstract base shared by every tool-registering Agent family: tool
    registry, optional user-defined instructions, the context-native task
    frame, and execution knobs. See module docstring for the full rationale.

    What this class owns:

    - Tool registration (``register_tool``/``register_tools``, alias-based,
      never the dotted ``full_name``) and the read accessors
      (``list_tools``/``has_tool``/``get_tool``/``remove_tool``/
      ``clear_tools``/``actions_context``), plus a ``_validate_effective_tool_id``
      hook (default: reserved-name check only) a subclass with its own
      additional naming constraints (e.g. ``ScriptActAgent``'s parser
      sentinels/real-builtin collision) extends via ``super()``.
    - ``instructions`` (construction-time only, read-only property): standing
      user guidance rendered after the tools frame inside a
      ``USER-DEFINED INSTRUCTIONS`` frame, only when set.
    - Execution knobs: ``tool_calls_limit``, ``regeneration_limit``,
      ``tool_concurrency_limit``.
    - ``_seed_task``: one read-only ``task_result_i`` context entry per
      visible prior turn. ``render_turn`` labels the history with the same
      names.
    - The concrete ``_render_task_messages`` frame and the shared
      ``_render_system_message``: framed ``TOOL-CALLING INSTRUCTIONS``
      (the static family prompt), framed ``AVAILABLE TOOLS`` (``TOOLS_PREAMBLE``,
      the tool list, then ``_BUILTINS_PREAMBLE``), then the framed user instructions.
    - ``to_dict()`` diagnostics for the above.

    What it does not own: any lifecycle hook body, any reference-resolution
    grammar, any retry loop, or record/result construction -- every concrete
    family owns all of that itself.

    Phase contract (what each hook may do):

    - ``think``: every model call (including validation/regeneration
      retries); ends with staged work for ``prepare``; no tool effects.
    - ``prepare``: resolution only -- bind the staged statements' arguments
      against ``task.context[name].value``; no model call, no tool I/O.
      A family that dispatches exactly one call per round and can resolve it
      at generation time (``ReActAgent``) resolves inside ``think``'s
      regeneration loop instead, so a resolution failure is regeneration
      feedback; its ``prepare`` is a documented no-op.
    - ``act``: effects -- dispatch tools, bind results into ``task.context``
      via ``Agent._write_context(task, {...}, create=True)``, and set
      ``task.generated_response`` and ``task.complete``.
    - Every new ``result_name``/assignment binding is written to
      ``task.context``; nothing is stored twice.
    - Build-once: a phase's messages are built on its first ``render_task``
      call and reused by its retries; the family clears ``task.task_messages``
      at the end of each round (a forgotten clear shows a stale frame).
    """

    INSTRUCTIONS_KEY: ClassVar[str] = "instructions"

    #: Closing text of the ``AVAILABLE TOOLS`` frame, after the tool list;
    #: empty by default (``ScriptActAgent`` adds the permitted Python builtins).
    _BUILTINS_PREAMBLE: ClassVar[str] = ""

    _TOOL_COLLISION_POLICIES: ClassVar[tuple[str, ...]] = ("raise", "skip", "replace")

    #: Effective tool ids no subclass instance may ever register, remove, or
    #: replace through the public API -- seeded only via ``_seed_reserved_tool``,
    #: bypassing ``register_tool`` entirely. Empty by default; each subclass
    #: that needs reserved tools overrides this.
    _RESERVED_TOOL_NAMES: ClassVar[frozenset[str]] = frozenset()

    def __init__(
        self,
        name: str,
        namespace: str,
        description: str,
        llm_engine: LLMEngine,
        record_enabled: bool = False,
        *,
        tool_calls_limit: Optional[int] = None,
        regeneration_limit: int = 5,
        tool_concurrency_limit: Optional[int] = None,
        pre_invoke: Optional[AtomicInvokable | Callable[..., Any]] = None,
        post_invoke: Optional[AtomicInvokable | Callable[..., Any]] = None,
        inputs_to_context: Optional[Iterable[str] | Mapping[str, str]] = None,
        response_to_context: Optional[str | Iterable[str] | Mapping[Any, str]] = None,
        response_to_post: Optional[str | Iterable[str] | Mapping[Any, str]] = None,
        use_runtime_context: bool = False,
        records_window: Optional[int] = None,
        tools: Optional[list[AtomicInvokable | Callable | MCPClientHub | A2AClientHub | PyA2AtomicClient]] = None,
        instructions: Optional[str | PromptConfig] = None,
    ) -> None:
        """
        Parameters
        ----------
        name, namespace, description, llm_engine, record_enabled
            Forwarded to ``Agent.__init__`` unchanged.
        tool_calls_limit : int | None
            Maximum number of non-return action calls per invoke run.
            ``None`` means unlimited. Must be ``>= 0`` if set.
        regeneration_limit : int
            Bounded-attempts ceiling for regenerating a single round's
            malformed/invalid draft before raising. Always a plain ``int``,
            never ``None``. Defaults to ``5``.
        tool_concurrency_limit : int | None
            Max dispatched calls allowed in a single concurrently-executed
            batch. ``None`` means unlimited (default).
        pre_invoke, post_invoke, inputs_to_context, response_to_context,
            response_to_post, use_runtime_context, records_window
            Forwarded to ``Agent.__init__`` unchanged.
        tools : list | None
            Construction-time convenience: registered via ``register_tools()``
            if given.
        instructions : str | PromptConfig | None
            Optional standing user guidance (e.g. "always call the outliner
            first"), rendered after the family prompt inside a
            ``USER-DEFINED INSTRUCTIONS`` frame, only when set. A plain
            ``str`` is free-form prose (empty/whitespace-only normalizes to
            "not provided"); a ``PromptConfig`` is accepted directly for a
            caller who wants template fields of their own. A discovered
            ``{name}`` field becomes a declared parameter of this agent --
            validated/defaulted by the normal ``invoke(inputs)`` pipeline --
            and is rendered against ``task.inputs``. Frozen at construction --
            read-only via the ``instructions`` property.

        ``fail_fast`` is intentionally not a parameter here -- each concrete
        subclass that needs it declares and validates its own. A removed
        parameter (``constants``, ``constant_aliases``,
        ``constant_descriptions``, ``tool_instructions``) is rejected by
        Python with ``TypeError``.
        """
        # Normalized before super().__init__() so a {name} field discovered
        # inside the template can be declared as a real extra_parameters entry.
        normalized_instructions = normalize_prompt_config(
            instructions,
            default_template=None,
            provided_description="Instructions",
            error_label="instructions",
        )

        super().__init__(
            name=name,
            namespace=namespace,
            description=description,
            llm_engine=llm_engine,
            record_enabled=record_enabled,
            pre_invoke=pre_invoke,
            post_invoke=post_invoke,
            inputs_to_context=inputs_to_context,
            response_to_context=response_to_context,
            response_to_post=response_to_post,
            use_runtime_context=use_runtime_context,
            records_window=records_window,
            extra_parameters=(
                list(normalized_instructions.parameters)
                if normalized_instructions is not None
                else []
            ),
        )

        self._toolbox: dict[str, AtomicInvokable] = {}

        if type(regeneration_limit) is not int or regeneration_limit < 0:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: regeneration_limit must "
                f"be an int >= 0; got {regeneration_limit!r}."
            )
        self._regeneration_limit = regeneration_limit

        self.tool_concurrency_limit = tool_concurrency_limit

        self._tool_calls_limit: Optional[int] = None
        self.tool_calls_limit = tool_calls_limit

        if tools is not None:
            self.register_tools(tools)

        if normalized_instructions is not None:
            self._system_prompts[self.INSTRUCTIONS_KEY] = normalized_instructions

    # ------------------------------------------------------------------ #
    # ToolAgent Properties
    # ------------------------------------------------------------------ #
    @property
    def tool_calls_limit(self) -> Optional[int]:
        """Max allowed non-return tool calls per invoke() run. None means unlimited."""
        return self._tool_calls_limit

    @tool_calls_limit.setter
    def tool_calls_limit(self, value: Optional[int]) -> None:
        if value is None:
            self._tool_calls_limit = None
            return
        if type(value) is not int or value < 0:
            raise ToolAgentError("tool_calls_limit must be None or an int >= 0.")
        self._tool_calls_limit = value

    @property
    def regeneration_limit(self) -> int:
        """Bounded-attempts ceiling for regenerating a single round's
        malformed/invalid draft before raising -- always a plain ``int``,
        never ``None``. Read-only."""
        return self._regeneration_limit

    @property
    def tool_concurrency_limit(self) -> Optional[int]:
        """Max dispatched (real tool) calls allowed in a single
        concurrently-executed batch. ``None`` means unlimited."""
        return self._tool_concurrency_limit

    @tool_concurrency_limit.setter
    def tool_concurrency_limit(self, value: Optional[int]) -> None:
        if value is not None and (type(value) is not int or value < 1):
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: tool_concurrency_limit "
                f"must be None or an int >= 1; got {value!r}."
            )
        self._tool_concurrency_limit = value

    @property
    def instructions(self) -> Optional[str]:
        """Template of the registered instructions, or None. Read-only."""
        config = self._system_prompts.get(self.INSTRUCTIONS_KEY)
        return config.template if config is not None else None

    # ------------------------------------------------------------------ #
    # Shared helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalize_collision_policy(name_collision_policy: str, allowed: tuple[str, ...]) -> str:
        """Takes an explicit ``allowed`` set; tool call sites pass
        ``self._TOOL_COLLISION_POLICIES``."""
        policy = name_collision_policy.lower().strip()
        if policy not in allowed:
            raise ToolRegistrationError(
                f"name_collision_policy must be one of: "
                f"{', '.join(repr(p) for p in allowed)}."
            )
        return policy

    @staticmethod
    def _validate_tool_alias(alias: Optional[str]) -> None:
        """Shape-only check on an explicit ``alias`` -- ``None`` (no alias
        given) always passes; a given alias must be a Python-identifier-
        legal string."""
        if alias is not None and (
            not isinstance(alias, str) or not IDENTIFIER_PATTERN.fullmatch(alias)
        ):
            raise ToolRegistrationError(
                f"alias must be None or a Python-identifier-legal string; got {alias!r}."
            )

    def _validate_effective_tool_id(self, effective_id: str) -> None:
        """
        Validate a tool's final effective id (an explicit alias, or the
        bare ``name``/``invokable.name`` fallback when none is given) --
        called from ``register_tool``/``register_tools`` for every
        candidate, before it's ever added to the toolbox.

        Default body: reject only a real reserved-tool-id collision
        (``self._RESERVED_TOOL_NAMES``). A subclass with additional naming
        constraints (e.g. parser sentinels or real-builtin-name collisions)
        overrides this, calling ``super()._validate_effective_tool_id(...)``
        first and adding its own checks after.
        """
        if effective_id in self._RESERVED_TOOL_NAMES:
            raise ToolRegistrationError(
                f"{type(self).__name__}.{self.name}: {effective_id!r} is "
                "reserved and cannot be registered through this API."
            )

    # ------------------------------------------------------------------ #
    # Tool registration
    # ------------------------------------------------------------------ #
    def register_tool(
        self,
        tool: AtomicInvokable | Callable,
        alias: Optional[str] = None,
        description: Optional[str] = None,
        *,
        name_collision_policy: Literal["raise", "skip", "replace"] = "raise",
    ) -> bool:
        """
        Register one invokable under an alias (sole effective identity) or,
        absent one, its own bare ``name`` — never the dotted ``full_name``,
        which only confuses an LLM asked to copy a tool id verbatim.
        ``AtomicInvokable`` inputs are stored as-is (never re-``toolify``'d);
        ``description`` is a documented no-op in that case — ``alias`` alone
        covers the "cover name" use case. Callables are normalized via
        ``toolify(namespace=self.name)``.

        A bare ``name`` collision across tools from different namespaces is
        not auto-disambiguated — it's caught by the same
        ``name_collision_policy`` as any other duplicate effective id; give
        one of them an explicit ``alias`` to resolve it.

        Returns whether the tool was newly registered (``False`` only when
        ``name_collision_policy="skip"`` hits an existing effective id).
        """
        policy = self._normalize_collision_policy(name_collision_policy, self._TOOL_COLLISION_POLICIES)

        if isinstance(tool, AtomicInvokable):
            invokable = tool
        elif callable(tool):
            try:
                invokable = toolify(
                    component=tool,
                    name=tool.__name__,
                    description=description or tool.__doc__,
                    namespace=self.name,
                )
            except Exception as exc:
                raise ToolRegistrationError(
                    f"{type(self).__name__}.{self.name}: failed to toolify component: {exc}"
                ) from exc
        else:
            raise ToolRegistrationError(
                f"{type(self).__name__}.{self.name}: unsupported component type "
                f"{type(tool).__name__!r}. Expected AtomicInvokable or Callable."
            )

        self._validate_tool_alias(alias)
        effective_id = alias if alias is not None else invokable.name
        self._validate_effective_tool_id(effective_id)

        if effective_id in self._toolbox:
            if policy == "raise":
                raise ToolRegistrationError(
                    f"{type(self).__name__}.{self.name}: tool already registered: {effective_id!r}."
                )
            if policy == "skip":
                return False

        self._toolbox[effective_id] = invokable
        return True

    def register_tools(
        self,
        tools: list[AtomicInvokable | Callable | MCPClientHub | A2AClientHub | PyA2AtomicClient],
        aliases: Optional[list[Optional[str]]] = None,
        *,
        name_collision_policy: Literal["raise", "skip", "replace"] = "raise",
    ) -> bool:
        """
        Register a mixed batch of plain invokables/callables and
        hub/client objects (``MCPClientHub``/``A2AClientHub``/
        ``PyA2AtomicClient``) in one pass. A hub/client entry expands into
        every tool it exposes, each under its own intrinsic bare ``name`` —
        never the dotted ``full_name`` — and no aliasing possible for
        hub-expanded entries. Every candidate's effective id is validated
        via ``_validate_effective_tool_id`` as soon as it's computed, during
        this same expansion pass -- not a second, separate pass afterward.
        Whole-batch validation happens before any toolbox mutation;
        intra-batch duplicate effective ids always raise regardless of
        ``name_collision_policy``.

        Returns ``True`` iff every item was newly registered (only ``"skip"``
        can make this ``False``; ``"raise"``/``"replace"`` are unconditionally
        ``True`` once the call doesn't raise).
        """
        policy = self._normalize_collision_policy(name_collision_policy, self._TOOL_COLLISION_POLICIES)

        if aliases is not None and len(aliases) != len(tools):
            raise ValueError(
                f"{type(self).__name__}.{self.name}: aliases must be the same "
                f"length as tools; got {len(aliases)} vs {len(tools)}."
            )

        # Expand every entry into (effective_id, invokable) candidates
        # without mutating self._toolbox yet.
        candidates: list[tuple[str, AtomicInvokable]] = []
        for index, item in enumerate(tools):
            item_alias = aliases[index] if aliases is not None else None

            if isinstance(item, (MCPClientHub, A2AClientHub, PyA2AtomicClient)):
                if item_alias is not None:
                    raise ValueError(
                        f"{type(self).__name__}.{self.name}: a hub/client entry at "
                        f"index {index} cannot take an alias (it expands into "
                        "multiple tools)."
                    )
                if isinstance(item, MCPClientHub):
                    remote_names = item.list_tools()
                elif isinstance(item, A2AClientHub):
                    remote_names = list(item.get_atomic_skills())
                else:
                    remote_names = item.list_invokables()

                for remote_name in remote_names:
                    try:
                        proxy = toolify(
                            component=item,
                            namespace=self.name,
                            remote_name=remote_name,
                        )
                    except Exception as exc:
                        raise ToolRegistrationError(
                            f"{type(self).__name__}.{self.name}: failed to toolify "
                            f"remote {remote_name!r}: {exc}"
                        ) from exc
                    self._validate_effective_tool_id(proxy.name)
                    candidates.append((proxy.name, proxy))

                if isinstance(item, A2AClientHub):
                    try:
                        generic_proxy = toolify(component=item, namespace=self.name)
                    except Exception as exc:
                        raise ToolRegistrationError(
                            f"{type(self).__name__}.{self.name}: failed to toolify "
                            f"generic A2A tool: {exc}"
                        ) from exc
                    self._validate_effective_tool_id(generic_proxy.name)
                    candidates.append((generic_proxy.name, generic_proxy))

            elif isinstance(item, AtomicInvokable):
                self._validate_tool_alias(item_alias)
                effective_id = item_alias if item_alias is not None else item.name
                self._validate_effective_tool_id(effective_id)
                candidates.append((effective_id, item))

            elif callable(item):
                try:
                    invokable = toolify(
                        component=item,
                        name=item.__name__,
                        description=item.__doc__,
                        namespace=self.name,
                    )
                except Exception as exc:
                    raise ToolRegistrationError(
                        f"{type(self).__name__}.{self.name}: failed to toolify "
                        f"{item!r}: {exc}"
                    ) from exc
                self._validate_tool_alias(item_alias)
                effective_id = item_alias if item_alias is not None else invokable.name
                self._validate_effective_tool_id(effective_id)
                candidates.append((effective_id, invokable))

            else:
                raise ToolRegistrationError(
                    f"{type(self).__name__}.{self.name}: unsupported item type "
                    f"{type(item).__name__!r} at index {index}."
                )

        # Intra-batch dedup — always raises regardless of name_collision_policy.
        seen: set[str] = set()
        for effective_id, _ in candidates:
            if effective_id in seen:
                raise ToolRegistrationError(
                    f"{type(self).__name__}.{self.name}: duplicate effective id in "
                    f"incoming batch: {effective_id!r}."
                )
            seen.add(effective_id)

        # Cross-check against the existing toolbox per name_collision_policy.
        any_skipped = False
        vetted: list[tuple[str, AtomicInvokable]] = []
        for effective_id, invokable in candidates:
            if effective_id in self._toolbox:
                if policy == "raise":
                    raise ToolRegistrationError(
                        f"{type(self).__name__}.{self.name}: tool already registered: "
                        f"{effective_id!r}."
                    )
                if policy == "skip":
                    any_skipped = True
                    continue
            vetted.append((effective_id, invokable))

        for effective_id, invokable in vetted:
            self._toolbox[effective_id] = invokable

        return not any_skipped

    # ------------------------------------------------------------------ #
    # Tool accessors
    # ------------------------------------------------------------------ #
    def actions_context(self) -> str:
        """
        Render every registered tool for prompt injection, one block per
        tool (signature line + a 4-space-indented triple-quoted docstring
        description, via ``_render_docstring_block``), joined by
        ``"\\n---\\n"``. Every rendered id is bare, never dotted: when
        ``tid`` is the tool's own ``name`` this is just ``tool.signature``
        verbatim; when ``tid`` is a distinct alias, only the leading
        identity token of ``tool.signature`` is swapped for the alias — the
        ``(args) -> ReturnType`` portion renders unchanged.
        """
        blocks: list[str] = []
        for tid, tool in self._toolbox.items():
            if tid == tool.name:
                rendered_signature = tool.signature
            else:
                rendered_signature = tid + tool.signature[len(tool.name):]

            blocks.append(f"{rendered_signature}\n{self._render_docstring_block(tool.description)}")

        return "\n---\n".join(blocks)

    def list_tools(self) -> dict[str, AtomicInvokable]:
        """Shallow copy of the toolbox, keyed by effective id (alias or bare name)."""
        return dict(self._toolbox)

    def has_tool(self, tool_id: str) -> bool:
        """Return ``True`` if ``tool_id`` (an alias or the tool's own bare ``name``) is registered."""
        return tool_id in self._toolbox

    def get_tool(self, tool_id: str) -> AtomicInvokable:
        """Return the registered invokable for ``tool_id``.

        Raises
        ------
        ToolAgentError
            If ``tool_id`` is not registered.
        """
        tool = self._toolbox.get(tool_id)
        if tool is None:
            raise ToolAgentError(f"{type(self).__name__}.{self.name}: unknown tool {tool_id!r}.")
        return tool

    def remove_tool(self, tool_id: str) -> bool:
        """Remove the tool stored under ``tool_id``.

        Raises
        ------
        ToolRegistrationError
            If ``tool_id`` is reserved (see ``_RESERVED_TOOL_NAMES``) —
            reserved tools can never be removed through this API, regardless
            of whether they're actually present.

        Returns
        -------
        bool
            ``True`` if the tool was present and removed; ``False`` if it was
            not registered.
        """
        if tool_id in self._RESERVED_TOOL_NAMES:
            raise ToolRegistrationError(
                f"{type(self).__name__}.{self.name}: {tool_id!r} is reserved "
                "and cannot be removed."
            )
        return self._toolbox.pop(tool_id, None) is not None

    def clear_tools(self) -> None:
        """Remove every registered tool except reserved ones (see
        ``_RESERVED_TOOL_NAMES``) — reserved tools are never cleared,
        regardless of how they were seeded."""
        self._toolbox = {
            tid: tool for tid, tool in self._toolbox.items()
            if tid in self._RESERVED_TOOL_NAMES
        }

    def _seed_reserved_tool(
        self, tool: AtomicInvokable | Callable, name: str, description: Optional[str] = None,
    ) -> None:
        """
        Directly insert ``tool`` into the toolbox under ``name``, bypassing
        ``register_tool`` entirely — the one sanctioned way to populate a
        reserved effective id (``register_tool``/``register_tools`` reject
        any id in ``self._RESERVED_TOOL_NAMES`` outright). Callables are
        toolified first, exactly like ``register_tool``'s own callable
        branch; ``AtomicInvokable`` instances are stored as-is. Intended to
        be called from a subclass's own ``__init__``, after
        ``super().__init__()``, once per reserved name that subclass
        declares.
        """
        if isinstance(tool, AtomicInvokable):
            invokable = tool
        else:
            invokable = toolify(
                component=tool,
                name=name,
                description=description or tool.__doc__,
                namespace=self.name,
            )
        self._toolbox[name] = invokable

    # ------------------------------------------------------------------ #
    # Docstring rendering
    # ------------------------------------------------------------------ #
    @staticmethod
    def _render_docstring_block(description: str) -> str:
        """
        Render ``description`` as a 4-space-indented triple-quoted docstring
        block, used by ``actions_context``. A
        single-line description closes on the same line
        (``    \"\"\"text\"\"\"``); a multi-line description continues indented
        (blank lines left bare, matching ordinary docstring convention) with
        the closing triple-quote on its own indented line.
        """
        lines = description.splitlines() or [""]
        if len(lines) == 1:
            return f'    """{lines[0]}"""'

        continuation = "\n".join(
            f"    {line}" if line.strip() else line for line in lines[1:]
        )
        return f'    """{lines[0]}\n{continuation}\n    """'

    def render_turn(self, turn: AgentRecord, **kwargs: Any) -> list[dict[str, str]]:
        """
        Labels a historic turn with its ``task_result_i`` address so a model
        can reference it by name -- base ``Agent.render_turn`` renders the raw
        value with no such label.

        1. ``index = kwargs["index"]`` (plain ``KeyError`` if missing: this is
           framework-internal plumbing from ``summarize``, not a validated
           boundary). It is the turn's 0-based position within *this
           invocation's* ``task.turns``, matching ``_seed_task``.
        2. ``messages = super().render_turn(turn, **kwargs)``
        3. Prepend ``{_CONTEXT_NAME_PREFIX}task_result_{index}: <type> = `` to
           the last message's content.
        """
        index = kwargs["index"]
        messages = super().render_turn(turn, **kwargs)
        label = (
            f"{self._CONTEXT_NAME_PREFIX}{TASK_RESULT_PREFIX}{index}: "
            f"{type(turn.generated_response).__name__} = "
        )
        messages[-1]["content"] = label + messages[-1]["content"]
        return messages

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #
    def _render_system_message(self, task: AgentTask) -> list[dict[str, str]]:
        """
        Render the system message as up to three framed sections.

        1. ``task.system_prompt_name is None`` -> ``[]``.
        2. ``TOOL-CALLING INSTRUCTIONS``: the active family prompt, rendered
           with no fields (it is static).
        3. ``AVAILABLE TOOLS``: ``TOOLS_PREAMBLE``, ``actions_context()`` and,
           when the family sets one, ``_BUILTINS_PREAMBLE``.
        4. With instructions registered: ``INSTRUCTIONS_PRECEDENCE``, then the
           instructions rendered against ``task.inputs`` in their
           ``USER-DEFINED INSTRUCTIONS`` frame.

        Sections are joined by a blank line. Carries no budget text: per-call
        facts belong in the directive.
        """
        if task.system_prompt_name is None:
            return []
        contract = self._system_prompts[task.system_prompt_name].render({})
        parts = (TOOLS_PREAMBLE, self.actions_context(), self._BUILTINS_PREAMBLE)
        tools_body = "\n\n".join(part for part in parts if part)
        sections = [
            frame_content(contract, "TOOL-CALLING INSTRUCTIONS"),
            frame_content(tools_body, "AVAILABLE TOOLS"),
        ]

        config = self._system_prompts.get(self.INSTRUCTIONS_KEY)
        if config is not None:
            text = config.render(dict(task.inputs))
            sections.append(
                f"{INSTRUCTIONS_PRECEDENCE}\n"
                f"{frame_content(text, 'USER-DEFINED INSTRUCTIONS')}"
            )

        return [{"role": "system", "content": "\n\n".join(sections)}]

    def _render_task_messages(self, task: AgentTask) -> list[dict[str, str]]:
        """
        Build-once per phase (the family clears ``task.task_messages`` at
        round end).

        1. If ``task.task_messages`` is set, return it as-is.
        2. ``block = self._render_task_block(task)``; ``steps`` and
           ``directive`` from the two family hooks.
        3. No steps: one user message ``f"{block}\\n\\n{directive}"``. Steps:
           user ``block``, assistant ``frame_content(steps, "STEPS")``, user
           ``directive``.
        4. Store on ``task.task_messages`` and return that same list.

        No guard for an empty directive: it must be a non-empty string (with
        steps an empty one would end the list on an assistant message).
        """
        if task.task_messages:
            return task.task_messages
        block = self._render_task_block(task)
        steps = self._render_steps(task)
        directive = self._render_directive(task)
        if not steps:
            messages = [{"role": "user", "content": f"{block}\n\n{directive}"}]
        else:
            messages = [
                {"role": "user", "content": block},
                {"role": "assistant", "content": frame_content(steps, "STEPS")},
                {"role": "user", "content": directive},
            ]
        task.task_messages = messages
        return messages

    @abstractmethod
    def _render_steps(self, task: AgentTask) -> str:
        """Interior of the steps block: what has run so far this call
        (completed statements, failure notes). ``""`` means nothing to show,
        so no block. Pure function of the task."""
        ...

    @abstractmethod
    def _render_directive(self, task: AgentTask) -> str:
        """Closing instruction for this phase, including per-call facts
        (budget line, failure/repair wording). Non-empty ``str``. Pure
        function of the task."""
        ...

    # ------------------------------------------------------------------ #
    # Task seeding and lifecycle hooks (abstract, no shared body)
    # ------------------------------------------------------------------ #
    def _seed_task(self, task: ToolAgentTask) -> None:
        """
        Add one read-only ``task_result_i`` entry per visible prior turn.

        Runs after ``_seed_context``; ``task.context`` already holds the
        conversation's variables, runtime and input-bound values.

        1. For ``i, turn in enumerate(task.turns)`` (the enumeration
           ``summarize`` uses, so ``i`` matches the history labels): the
           turn's ``generated_response``, deep-copied (an uncopyable result is
           used as-is, never an error), becomes a ``readonly`` variable named
           ``task_result_{i}`` with ``preview_limit=0`` and no description.
        2. Written straight into ``task.context``, not through
           ``_write_context``: the entries stay out of ``task.changed_context``
           and are never persisted.
        """
        for i, turn in enumerate(task.turns):
            value = turn.generated_response
            try:
                value = copy.deepcopy(value)
            except Exception:
                pass
            name = f"{TASK_RESULT_PREFIX}{i}"
            task.context[name] = ContextVariable(
                name=name, value=value, readonly=True, preview_limit=0
            )

    @abstractmethod
    def think(self, task: AgentTask) -> AgentTask:
        """Every model call of the round, including validation/regeneration
        retries; ends with staged work for ``prepare``; no tool effects.
        Hard-abstract -- every family has genuine generation work here."""
        ...

    @abstractmethod
    async def async_think(self, task: AgentTask) -> AgentTask:
        """Async mirror of ``think``. No default — real I/O of its own."""
        ...

    @abstractmethod
    def prepare(self, task: AgentTask) -> AgentTask:
        """Resolution only: bind the staged statements' arguments against
        ``task.context[name].value``; no model call, no tool I/O.
        Hard-abstract -- each family's resolve/cascade logic differs enough
        that no shared body fits. ``ReActAgent`` resolves inside ``think``
        and implements this as a no-op."""
        ...

    @abstractmethod
    async def async_prepare(self, task: AgentTask) -> AgentTask:
        """Async mirror of ``prepare``."""
        ...

    @abstractmethod
    def act(self, task: AgentTask) -> AgentTask:
        """Effects: dispatch tools, bind results into ``task.context`` via
        ``Agent._write_context(task, {...}, create=True)``, and set
        ``task.generated_response`` and ``task.complete``. Hard-abstract --
        each family's dispatch shape genuinely differs."""
        ...

    @abstractmethod
    async def async_act(self, task: AgentTask) -> AgentTask:
        """Async mirror of ``act``."""
        ...

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        """Return a diagnostic snapshot of this ToolAgent.

        Extends the base Agent snapshot with tool/knob diagnostics.
        No ``fail_fast`` key here — that lives on each concrete subclass's
        own ``to_dict()`` override instead.
        """
        d = super().to_dict()
        d.update({
            "tool_calls_limit": self.tool_calls_limit,
            "regeneration_limit": self.regeneration_limit,
            "tool_concurrency_limit": self.tool_concurrency_limit,
            "tools": {
                name: tool.to_dict()
                for name, tool in self._toolbox.items()
            },
        })
        return d
