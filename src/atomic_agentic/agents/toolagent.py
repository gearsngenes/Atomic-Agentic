"""
ToolAgent: shared ancestor for every tool-registering, constant-registering
Agent family.

Owns the surface that ``PlanActAgent``/``ReActAgent`` (formerly unified
under the now-removed ``JsonToolAgent``) and ``ScriptActAgent`` already
shared byte-for-byte or near-identically: tool registration (alias-based,
``register_tool``/``register_tools``), constant registration (alias-named,
construction-time only), shared rendering
(``actions_context``/``constants_context``/``render_turn``), and the
execution knobs common to both grammars (``tool_calls_limit``,
``regeneration_limit``, ``tool_concurrency_limit``).

Deliberately owns no execution engine and no lifecycle body of any kind --
every ``_initialize_task``/``think``/``prepare``/``act`` (+ async mirror)
hook stays ``@abstractmethod`` here, purely for base-``Agent``-compatible
type-narrowing/documentation (no new enforcement beyond what ``Agent``'s own
abstracts already require). Each concrete family's generation/resolution/
dispatch logic stays fully its own, duplicated rather than shared, per this
branch's established Stage A/B duplication posture.

``fail_fast`` is deliberately NOT here: ``PlanActAgent``'s version governs
whole-batch cascade-vs-abort, ``ReActAgent``'s governs single-call
raise-vs-tolerate -- different granularity, coincidentally the same name and
type, never one shared feature. Each declares its own independently.
``ScriptActAgent`` has no concept of it at all.
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import replace
from typing import (
    Any,
    Callable,
    ClassVar,
    Literal,
    Optional,
)

from .base import Agent
from ..models.agents.records import AgentRecord
from ..models.agents.blackboard_models import ConstantSpec
from ..models.agents.tasks import AgentTask
from ..exceptions import (
    ToolAgentError,
    ToolRegistrationError,
)
from ..constants.core import IDENTIFIER_PATTERN
from ..core.Invokable import AtomicInvokable
from ..llm.base import LLMEngine
from ..tools import toolify
from ..mcp import MCPClientHub
from ..a2a import A2AClientHub, PyA2AtomicClient


class ToolAgent(Agent, ABC):
    """
    Abstract base shared by every tool-registering, constant-registering
    Agent family: tool registry, constant registry, shared rendering, and
    execution knobs. See module docstring for the full rationale.

    What this class owns:

    - Tool registration (``register_tool``/``register_tools``, alias-based,
      never the dotted ``full_name``) and the read accessors
      (``list_tools``/``has_tool``/``get_tool``/``remove_tool``/
      ``clear_tools``/``actions_context``), plus a ``_validate_effective_tool_id``
      hook (default: reserved-name check only) a subclass with its own
      additional naming constraints (e.g. ``ScriptActAgent``'s parser
      sentinels/real-builtin collision) extends via ``super()``.
    - Constant registration (``register_constant``/``register_constants``,
      construction-time only, alias-named wire names) and its own
      accessors.
    - Execution knobs: ``tool_calls_limit``, ``regeneration_limit``,
      ``tool_concurrency_limit``.
    - Shared rendering: ``actions_context``/``constants_context``/
      ``_render_docstring_block``, ``_turn_position``/``render_turn``,
      ``_copy_for_task_namespace``, and ``_render_system_message`` (built
      around a ``_extra_system_context()`` hook, default ``{}``).
    - ``to_dict()`` diagnostics for the above.

    What it does not own: any lifecycle hook body, any placeholder/
    reference-resolution grammar, any retry loop, or record/result
    construction -- every concrete family owns all of that itself.
    """

    TOOLS_FIELD: ClassVar[str] = "TOOLS"
    CONSTANTS_FIELD: ClassVar[str] = "CONSTANTS"

    _TOOL_COLLISION_POLICIES: ClassVar[tuple[str, ...]] = ("raise", "skip", "replace")
    _CONSTANT_COLLISION_POLICIES: ClassVar[tuple[str, ...]] = ("raise", "skip", "replace", "suffix")

    #: Effective tool ids no subclass instance may ever register, remove, or
    #: replace through the public API -- seeded only via ``_seed_reserved_tool``,
    #: bypassing ``register_tool`` entirely. Empty by default; each subclass
    #: that needs reserved tools overrides this.
    _RESERVED_TOOL_NAMES: ClassVar[frozenset[str]] = frozenset()

    #: Shared by ``_copy_for_task_namespace`` -- known atomic-immutable types
    #: that never need a defensive deep copy.
    _ATOMIC_IMMUTABLE_TYPES: ClassVar[tuple[type, ...]] = (
        str, int, float, bool, complex, bytes, type(None),
    )

    def __init__(
        self,
        name: str,
        namespace: str,
        description: str,
        llm_engine: LLMEngine,
        context_enabled: bool = False,
        *,
        tool_calls_limit: Optional[int] = None,
        regeneration_limit: int = 5,
        tool_concurrency_limit: Optional[int] = None,
        response_preview_limit: Optional[int] = None,
        pre_invoke: Optional[AtomicInvokable | Callable[..., Any]] = None,
        post_invoke: Optional[AtomicInvokable | Callable[..., Any]] = None,
        post_result_key: Optional[str] = None,
        records_window: Optional[int] = None,
        tools: Optional[list[AtomicInvokable | Callable | MCPClientHub | A2AClientHub | PyA2AtomicClient]] = None,
        constants: Optional[list[Any]] = None,
        constant_aliases: Optional[list[Optional[str]]] = None,
        constant_descriptions: Optional[list[Optional[str]]] = None,
    ) -> None:
        """
        Parameters
        ----------
        name, namespace, description, llm_engine, context_enabled
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
        response_preview_limit, pre_invoke, post_invoke, post_result_key,
        records_window
            Forwarded to ``Agent.__init__`` unchanged.
        tools : list | None
            Construction-time convenience: registered via ``register_tools()``
            if given.
        constants : list[Any] | None
            Construction-time convenience: registered via
            ``register_constants()`` if given, alongside ``constant_aliases``/
            ``constant_descriptions``.
        constant_aliases, constant_descriptions
            Positionally aligned with ``constants``. See ``register_constant``.

        ``fail_fast`` is intentionally not a parameter here -- each concrete
        subclass that needs it declares and validates its own.
        """
        super().__init__(
            name=name,
            namespace=namespace,
            description=description,
            llm_engine=llm_engine,
            context_enabled=context_enabled,
            pre_invoke=pre_invoke,
            post_invoke=post_invoke,
            post_result_key=post_result_key,
            records_window=records_window,
            response_preview_limit=response_preview_limit,
        )

        self._toolbox: dict[str, AtomicInvokable] = {}
        self._constants: dict[str, ConstantSpec] = {}
        self._constant_counter: int = 0

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
        if constants is not None:
            self.register_constants(
                constants, aliases=constant_aliases, descriptions=constant_descriptions
            )

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

    # ------------------------------------------------------------------ #
    # Shared helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalize_collision_policy(name_collision_policy: str, allowed: tuple[str, ...]) -> str:
        """Takes an explicit ``allowed`` set -- tool call sites pass
        ``self._TOOL_COLLISION_POLICIES``, constant call sites pass
        ``self._CONSTANT_COLLISION_POLICIES`` (constants alone support the
        extra ``"suffix"`` mode)."""
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
    # Constants Helpers
    # ------------------------------------------------------------------ #
    @property
    def constants(self) -> dict[str, ConstantSpec]:
        """Shallow copy of registered constants, keyed by the bare uppercase alias."""
        return dict(self._constants)

    def register_constant(
        self,
        constant: Any,
        alias: Optional[str] = None,
        description: Optional[str] = None,
        *,
        name_collision_policy: Literal["raise", "skip", "replace", "suffix"] = "raise",
    ) -> bool:
        """
        Register one named runtime constant. ``alias`` is optional: given,
        stored under ``alias.upper()`` (wire-facing ``ConstantSpec.name`` is
        ``f"K_{alias.upper()}"``); omitted, auto-named from
        ``self._constant_counter`` (``f"K_{counter}"`` -- both the dict
        key and ``ConstantSpec.name`` are this same string, then the counter
        increments). The counter never decrements, so a retired auto-name is
        never reissued.

        ``description``, when omitted or blank/whitespace-only, stores the
        literal string ``"No details"`` -- never ``None``.

        ``name_collision_policy`` gains a fourth value, ``"suffix"``: on
        collision, retries ``f"{key}_0"``, ``f"{key}_1"``, ... until an
        unused key is found, and registers under that key instead of
        raising/skipping.

        Returns whether the constant was newly registered under its
        originally intended key (``False`` only when
        ``name_collision_policy="skip"`` hits an existing key; ``"suffix"``
        always returns ``True``, since it renames instead of skipping).
        """
        policy = self._normalize_collision_policy(name_collision_policy, self._CONSTANT_COLLISION_POLICIES)

        if alias is None:
            key = f"K_{self._constant_counter}"
            self._constant_counter += 1
            name = key
        else:
            if not isinstance(alias, str) or not alias.strip():
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: alias must be None or a "
                    f"non-empty string; got {alias!r}."
                )
            key = alias.upper()
            name = f"K_{key}"

        normalized_description = description.strip() if isinstance(description, str) else None
        effective_description = normalized_description if normalized_description else "No details"

        if key in self._constants:
            if policy == "raise":
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: constant already registered: "
                    f"{key!r}."
                )
            if policy == "skip":
                return False
            if policy == "suffix":
                i = 0
                candidate = f"{key}_{i}"
                while candidate in self._constants:
                    i += 1
                    candidate = f"{key}_{i}"
                key = candidate
                name = f"K_{key}"
            # "replace" falls through to the unconditional overwrite below.

        self._constants[key] = ConstantSpec(name=name, value=constant, description=effective_description)
        return True

    def register_constants(
        self,
        constants: list[Any],
        aliases: Optional[list[Optional[str]]] = None,
        descriptions: Optional[list[Optional[str]]] = None,
        *,
        name_collision_policy: Literal["raise", "skip", "replace", "suffix"] = "raise",
    ) -> bool:
        """
        Register a batch of constants. ``aliases``/``descriptions`` are
        optional and positionally aligned with ``constants`` -- an omitted
        list, or an individual ``None`` entry, means "auto-name"/"default
        description" for that position, exactly mirroring
        ``register_constant``'s single-item behavior. Whole-batch length
        validation before any mutation.

        Under ``name_collision_policy="suffix"``, the existing "intra-batch
        duplicates always raise" rule is bypassed: a colliding key (whether
        against another entry in this same batch or against the existing
        registry) is resolved by the same ``f"{key}_i"`` retry used by
        ``register_constant``, rather than raising. The other three policies
        keep the unconditional intra-batch raise.

        Returns ``True`` iff every item was newly registered under its
        originally intended key (only ``"skip"`` can make this ``False``).
        """
        policy = self._normalize_collision_policy(name_collision_policy, self._CONSTANT_COLLISION_POLICIES)

        if aliases is not None and len(aliases) != len(constants):
            raise ValueError(
                f"{type(self).__name__}.{self.name}: aliases must be the same "
                f"length as constants; got {len(aliases)} vs {len(constants)}."
            )
        if descriptions is not None and len(descriptions) != len(constants):
            raise ValueError(
                f"{type(self).__name__}.{self.name}: descriptions must be the same "
                f"length as constants; got {len(descriptions)} vs {len(constants)}."
            )

        candidates: dict[str, ConstantSpec] = {}
        for index, value in enumerate(constants):
            alias = aliases[index] if aliases is not None else None
            raw_description = descriptions[index] if descriptions is not None else None
            normalized_description = raw_description.strip() if isinstance(raw_description, str) else None
            effective_description = normalized_description if normalized_description else "No details"

            if alias is None:
                key = f"K_{self._constant_counter}"
                self._constant_counter += 1
                name = key
            else:
                if not isinstance(alias, str) or not alias.strip():
                    raise ToolAgentError(
                        f"{type(self).__name__}.{self.name}: alias must be None or "
                        f"a non-empty string; got {alias!r}."
                    )
                key = alias.upper()
                name = f"K_{key}"

            # Intra-batch dedup: "suffix" resolves it below instead of
            # raising; the other three policies keep the unconditional raise.
            if key in candidates and policy != "suffix":
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: duplicate constant name in "
                    f"batch: {key!r}."
                )

            # "suffix" resolves either collision source (intra-batch or
            # pre-existing registry) uniformly, right here -- this entry is
            # guaranteed collision-free before it ever reaches `candidates`.
            if policy == "suffix" and (key in candidates or key in self._constants):
                i = 0
                candidate_key = f"{key}_{i}"
                while candidate_key in candidates or candidate_key in self._constants:
                    i += 1
                    candidate_key = f"{key}_{i}"
                key = candidate_key
                name = f"K_{key}"

            candidates[key] = ConstantSpec(name=name, value=value, description=effective_description)

        # Cross-check against the existing registry per policy. "suffix"
        # entries never trigger anything here -- already resolved above.
        any_skipped = False
        vetted: dict[str, ConstantSpec] = {}
        for key, spec in candidates.items():
            if key in self._constants and policy != "suffix":
                if policy == "raise":
                    raise ToolAgentError(
                        f"{type(self).__name__}.{self.name}: constant already "
                        f"registered: {key!r}."
                    )
                if policy == "skip":
                    any_skipped = True
                    continue
                # "replace" falls through to the unconditional overwrite below.
            vetted[key] = spec

        self._constants.update(vetted)
        return not any_skipped

    def has_constant(self, alias: str) -> bool:
        """Return whether a constant is registered under ``alias`` (case-insensitive)."""
        return alias.upper() in self._constants

    def get_constant(self, alias: str) -> ConstantSpec:
        """Return the registered constant for ``alias`` (case-insensitive).

        Raises ``ToolAgentError`` if no constant is registered under it.
        """
        spec = self._constants.get(alias.upper())
        if spec is None:
            raise ToolAgentError(f"{type(self).__name__}.{self.name}: unknown constant {alias!r}.")
        return spec

    def remove_constant(self, alias: str) -> bool:
        """Remove the constant registered under ``alias`` (case-insensitive).
        Returns whether it was present."""
        return self._constants.pop(alias.upper(), None) is not None

    def update_constant_description(self, alias: str, description: str) -> None:
        """
        Replace only the ``description`` of an already-registered constant
        -- ``.name``/``.value`` are never touched. Looked up the same way
        ``get_constant`` resolves ``alias`` (``alias.upper()``; works for a
        user-given alias or a literal auto-generated ``K_i`` key).
        ``description`` must be a real, non-empty string once stripped --
        there is no reset-to-default (``None``) path.

        Raises ``ToolAgentError`` if no constant is registered under
        ``alias``, or if ``description`` is not a non-empty string.
        """
        key = alias.upper()
        spec = self._constants.get(key)
        if spec is None:
            raise ToolAgentError(f"{type(self).__name__}.{self.name}: unknown constant {alias!r}.")
        if not isinstance(description, str) or not description.strip():
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: description must be a "
                f"non-empty string; got {description!r}."
            )
        self._constants[key] = replace(spec, description=description.strip())

    def clear_constants(self) -> None:
        """Remove every registered constant."""
        self._constants.clear()

    @staticmethod
    def _render_docstring_block(description: str) -> str:
        """
        Render ``description`` as a 4-space-indented triple-quoted docstring
        block, shared by ``actions_context``/``constants_context`` so a
        tool's and a constant's description render identically. A
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

    def constants_context(self) -> str:
        """
        Render every registered constant for prompt injection, one block
        per constant (a ``NAME: type`` annotation line + a 4-space-indented
        triple-quoted docstring description, via ``_render_docstring_block``),
        matching ``actions_context``'s own docstring-style rendering. Names
        print exactly as stored (``K_<ALIAS>`` when aliased, or ``K_i`` when
        auto-named) — no prefix synthesized here. Empty
        registry renders a "no constants" message.
        """
        if not self._constants:
            return "No constants registered."

        rendered: list[str] = []
        for spec in self._constants.values():
            description = (
                spec.description if spec.description is not None else "No description provided."
            )
            rendered.append(f"{spec.name}: {spec.type}\n{self._render_docstring_block(description)}")

        return "\n\n".join(rendered)

    def _copy_for_task_namespace(self, value: Any) -> Any:
        """
        Return ``value`` unchanged if it's a known atomic-immutable type
        (nothing callable on these ever mutates in place); otherwise return
        a deep copy. Used to seed both ``task.cache``'s ``task_result_i``
        entries and ``task.constant_values`` exactly once per invocation,
        so a mutating attribute/method call on either can never reach the
        real, shared registered constant or a prior invocation's stored
        result -- covers aliasing and nested-attribute mutation uniformly,
        since nothing shared is ever exposed by direct reference in the
        first place.
        """
        if isinstance(value, self._ATOMIC_IMMUTABLE_TYPES):
            return value
        return copy.deepcopy(value)

    # ------------------------------------------------------------------ #
    # Cross-invocation result addressing
    # ------------------------------------------------------------------ #
    def _turn_position(self, turn: AgentRecord) -> int:
        """
        Walk ``turn.prev`` backward to the conversation root, counting
        hops. The root itself is position 0, its child is 1, etc. --
        matches the turn's actual index in ``get_conversation()``'s full
        list, computed fresh from existing structure (correct even when
        ``task.turns`` is a ``records_window``-truncated tail, since this
        always walks all the way to the true root regardless of window).
        """
        position = 0
        node = turn
        while node.prev is not None:
            node = node.prev
            position += 1
        return position

    def render_turn(self, turn: AgentRecord) -> list[dict[str, str]]:
        """
        Labels a historic turn with its ``task_result_i`` address so a
        model can reference it by name in a later plan -- base
        ``Agent.render_turn`` renders the raw value with no such label.
        """
        messages = super().render_turn(turn)
        i = self._turn_position(turn)
        label = f"task_result_{i}: {type(turn.generated_response).__name__} = "
        messages[-1]["content"] = label + messages[-1]["content"]
        return messages

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #
    def _extra_system_context(self) -> dict[str, str]:
        """
        Extra render-context entries a subclass's active system prompt
        needs beyond ``{TOOLS}``/``{CONSTANTS}``. Default: none.
        ``ScriptActAgent`` overrides this to add ``EXCLUDED_PY_BUILTINS``.
        """
        return {}

    def _render_system_message(self, task: AgentTask) -> list[dict[str, str]]:
        """
        Render this agent's active system prompt with tool/constant context
        injected. Returns ``[]`` when ``task.system_prompt_name`` is
        ``None``. Deliberately carries no budget content — ``tool_calls_limit``
        is a per-invocation fact, not a standing instruction, so each family
        surfaces it in its own task-message banner instead.
        """
        if task.system_prompt_name is None:
            return []
        render_context = {
            self.TOOLS_FIELD: self.actions_context(),
            self.CONSTANTS_FIELD: self.constants_context(),
            **self._extra_system_context(),
        }
        rendered = self._system_prompts[task.system_prompt_name].render(render_context)
        return [{"role": "system", "content": rendered}]

    # ------------------------------------------------------------------ #
    # Task-lifecycle hooks -- stay fully abstract, no shared body
    # ------------------------------------------------------------------ #
    @abstractmethod
    def _initialize_task(
        self,
        *,
        turns: list[AgentRecord],
        prompt: str,
        inputs: dict,
    ) -> AgentTask:
        """
        Build and return this invocation's task. A concrete override is
        responsible for seeding whatever cross-invocation context this
        family's own task model needs directly from ``turns``.
        """
        ...

    @abstractmethod
    def think(self, task: AgentTask) -> AgentTask:
        """Make this round's real decision via the LLM. Hard-abstract —
        every family has genuine generation work here."""
        ...

    @abstractmethod
    async def async_think(self, task: AgentTask) -> AgentTask:
        """Async mirror of ``think``. No default — real I/O of its own."""
        ...

    @abstractmethod
    def prepare(self, task: AgentTask) -> AgentTask:
        """Turn this round's decision into something ``act`` can run, with
        no further LLM calls. Hard-abstract — each family's resolve/cascade
        logic differs enough that no shared body fits."""
        ...

    @abstractmethod
    async def async_prepare(self, task: AgentTask) -> AgentTask:
        """Async mirror of ``prepare``."""
        ...

    @abstractmethod
    def act(self, task: AgentTask) -> AgentTask:
        """Execute this round's prepared work and apply results, advancing
        ``task``. Hard-abstract — each family's dispatch shape genuinely
        differs."""
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

        Extends the base Agent snapshot with tool/constant/knob diagnostics.
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
