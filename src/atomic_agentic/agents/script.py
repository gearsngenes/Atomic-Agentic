from __future__ import annotations

import ast
import asyncio
import builtins
import copy
from dataclasses import replace
from typing import Any, Callable, ClassVar, Literal, Optional

from ..mcp.MCPClientHub import MCPClientHub
from ..a2a.A2AClientHub import A2AClientHub
from ..a2a.PyA2AtomicClient import PyA2AtomicClient

from .base import Agent
from .prompts import ONESHOT_PLANNER_PROMPT
from .tools import attr_call_tool, builtin_call_tool
from ..core.Invokable import AtomicInvokable
from ..llm.base import LLMEngine
from ..tools.Toolify import toolify
from ..models.agents.blackboard_models import CodeStatement, ConstantSpec
from ..models.agents.records import AgentRecord, LLMRecord, ScriptAgentRecord
from ..models.agents.tasks import ScriptAgentTask
from ..constants.core import IDENTIFIER_PATTERN, NO_VAL
from ..constants.agents import (
    ATTR_CALL_ALIAS,
    EXCLUDED_PY_BUILTINS,
    FINAL_ROUND_WARNING,
    PY_BUILTIN_ALIAS,
    RETURN_ALIAS,
    RHS_ASSIGN_ALIAS,
)
from ..exceptions import BlackboardParseError, ToolAgentError, ToolRegistrationError
from ..utils.core import run_coro_sync
from ..utils.script import (
    compile_batches,
    is_dispatched_slot,
    parse_generation,
    render_cache_snapshot,
    render_completed_as_python,
    resolve_slot_args,
    rewrite_builtin_calls,
    validate_references,
)


def _render_docstring_block(description: str) -> str:
    """
    Render ``description`` as a 4-space-indented triple-quoted docstring
    block, shared by ``ScriptAgent.actions_context``/``constants_context``
    so a tool's and a constant's description render identically. A
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


class ScriptAgent(Agent):
    """
    Adaptive, one-shot-planning tool-invoking agent (sibling family to
    ``ToolAgent``, not a subclass). Writes native-grammar, Python-style
    statements toward a task from a single generated plan. There is no
    separate decomposition, orchestration, or synthesis call, and no
    construction-time mode knob -- adaptivity is meant to be emergent from
    how many ``# PAUSE`` markers end up in one continuous plan, not picked
    up front.

    Checkpoint-triggered reactive continuation (Pass 2.3): a generation
    always terminates at the first of a ``return``, an explicit
    ``# PAUSE`` (a bare, complete sentinel -- no trailing note), or a
    defensively-pruned ``if`` statement (the grammar still forbids
    conditionals outright; a model that writes one anyway is handled by
    silent truncation, not rejection, unless nothing real precedes it). A
    resolution failure in ``prepare()`` or a real tool-execution failure in
    ``act()`` are handled the same way -- none of these four cases raise
    anymore. Whichever one occurs (other than ``return``) sets
    ``task.continue_planning``, and ``think()`` re-invokes the same planner
    for a fresh continuation, seeing a Python-source snapshot of the work
    already done this invoke plus, only for a resolution/execution failure,
    the real failure text (an explicit ``# PAUSE``/if-cutoff carries no
    reason of its own to show) -- no separate judge/critic model. Also told
    in-band when it has reached its final allowed planning round: writing
    another ``# PAUSE`` there is rejected as a regen-repair issue rather
    than granted as a continuation. `tool_calls_limit` is an optional,
    fully independent budget on total dispatched calls (tools and builtins
    counted identically) across every generation round in one invoke --
    `None` (the default) means no such cap, relying on
    `planning_rounds_limit` alone. `planning_rounds_limit` separately
    bounds how many times the agent is permitted to plan in total --
    the first generation counts as one, not a free attempt before the
    budget starts -- defaulting to 25 (safe-by-default); an explicit
    `None` opts into unbounded rounds. `regeneration_limit` (always a
    plain `int`, never `None`, default 5) independently bounds how many
    times a single round's malformed/invalid draft may be regenerated
    before raising -- distinct from `planning_rounds_limit`, which governs
    genuine incremental progress across rounds ("how many times may it
    plan, starting from scratch"), not within-round mistake recovery
    ("how many second chances does one attempt get").

    Cross-invocation result addressing is implemented: a prior turn's
    result is seeded into `task.cache` and labeled in rendered history as
    `task_result_i`, a fixed, read-only reference a later plan can use by
    name.
    """

    _TOOL_COLLISION_POLICIES: ClassVar[tuple[str, ...]] = ("raise", "skip", "replace")
    _CONSTANT_COLLISION_POLICIES: ClassVar[tuple[str, ...]] = ("raise", "skip", "replace", "suffix")

    def __init__(
        self,
        name: str,
        namespace: str,
        description: str,
        llm_engine: LLMEngine,
        context_enabled: bool = False,
        *,
        regeneration_limit: int = 5,
        tool_calls_limit: Optional[int] = None,
        planning_rounds_limit: Optional[int] = 25,
        tool_concurrency_limit: Optional[int] = None,
        response_preview_limit: Optional[int] = None,
        pre_invoke: Optional[AtomicInvokable | Callable[..., Any]] = None,
        post_invoke: Optional[AtomicInvokable | Callable[..., Any]] = None,
        post_result_key: Optional[str] = None,
        records_window: Optional[int] = None,
        assistant_response_source: Literal["raw", "final"] = "raw",
        tools: Optional[list[AtomicInvokable | Callable | MCPClientHub | A2AClientHub | PyA2AtomicClient]] = None,
        constants: Optional[list[Any]] = None,
        constant_aliases: Optional[list[Optional[str]]] = None,
        constant_descriptions: Optional[list[Optional[str]]] = None,
    ) -> None:
        """
        Validate/store ``regeneration_limit`` (must be an ``int >= 0`` -- no
        ``None``/unlimited option), then assign ``tool_calls_limit``/
        ``planning_rounds_limit``/``tool_concurrency_limit`` through their
        own independent setters (no cross-field validation between any of
        them -- orthogonal concerns: concurrency shape, total-call budget,
        and round budget), then initialize empty toolbox/constants storage
        and register any construction-time ``tools``/``constants`` by
        delegating to ``register_tools``/``register_constants`` -- no
        validation duplicated here. ``extra_parameters`` is never forwarded
        to ``super().__init__`` — matches ``ToolAgent.__init__``'s own
        precedent.
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
            assistant_response_source=assistant_response_source,
        )

        if type(regeneration_limit) is not int or regeneration_limit < 0:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: regeneration_limit must "
                f"be an int >= 0; got {regeneration_limit!r}."
            )
        self._regeneration_limit = regeneration_limit

        self.tool_calls_limit = tool_calls_limit
        self.planning_rounds_limit = planning_rounds_limit
        self.tool_concurrency_limit = tool_concurrency_limit

        self._toolbox: dict[str, AtomicInvokable] = {}
        self._constants: dict[str, ConstantSpec] = {}
        self._constant_counter: int = 0

        self._system_prompts["planner"] = ONESHOT_PLANNER_PROMPT

        if tools is not None:
            self.register_tools(tools)
        if constants is not None:
            self.register_constants(
                constants, aliases=constant_aliases, descriptions=constant_descriptions
            )

    # ------------------------------------------------------------------ #
    # Construction-time / mutable knobs
    # ------------------------------------------------------------------ #
    @property
    def regeneration_limit(self) -> int:
        """Bounded-attempts ceiling for regenerating a single round's
        malformed/invalid draft (structural parse/reference/budget
        failures) before raising -- always a plain ``int``, never ``None``
        (unlike ``planning_rounds_limit``, mistake-recovery within one
        round is never legitimately unbounded). Read-only."""
        return self._regeneration_limit

    @property
    def tool_calls_limit(self) -> Optional[int]:
        """Max real tool calls allowed per ``invoke()`` run, shared across
        every generation round in that invoke (not reset per round).
        ``None`` means unlimited."""
        return self._tool_calls_limit

    @tool_calls_limit.setter
    def tool_calls_limit(self, value: Optional[int]) -> None:
        if value is not None and (type(value) is not int or value < 0):
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: tool_calls_limit must be "
                f"None or an int >= 0; got {value!r}."
            )
        self._tool_calls_limit = value

    @property
    def planning_rounds_limit(self) -> Optional[int]:
        """Max total planning generations (the first generation plus every
        pause/if-cutoff/failure-triggered re-generation) permitted per
        ``invoke()`` run -- how many times the agent is allowed to plan,
        starting from scratch counts as one, not a count of "extra chances"
        beyond a free first attempt. Defaults to ``25`` (safe-by-default,
        matching common industry convention for this kind of round/
        iteration cap). ``None`` means unlimited -- a deliberate,
        non-default opt-in, not the default posture."""
        return self._planning_rounds_limit

    @planning_rounds_limit.setter
    def planning_rounds_limit(self, value: Optional[int]) -> None:
        if value is not None and (type(value) is not int or value < 0):
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: planning_rounds_limit "
                f"must be None or an int >= 0; got {value!r}."
            )
        self._planning_rounds_limit = value

    @property
    def tool_concurrency_limit(self) -> Optional[int]:
        """Max dispatched (real tool/builtin) calls allowed in a single
        concurrently-executed batch -- a bound on the existing
        dependency-inferred batching, not a model-facing signal (zero
        prompt footprint). ``None`` means unlimited (today's status quo)."""
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
        legal string. Reserved-sentinel/builtin-collision checks live on
        the *resolved* effective id instead (``_validate_effective_tool_id``
        below), since a bare tool ``name`` falling back from no alias needs
        the same protection an explicit alias does."""
        if alias is not None and (
            not isinstance(alias, str) or not IDENTIFIER_PATTERN.fullmatch(alias)
        ):
            raise ToolRegistrationError(
                f"alias must be None or a Python-identifier-legal string; got {alias!r}."
            )

    @staticmethod
    def _validate_effective_tool_id(effective_id: str) -> None:
        """Validate a tool's final effective id (an explicit alias, or the
        bare ``name``/``invokable.name`` fallback when none is given) --
        called after resolution at every registration call site, so both
        paths get identical protection.

        Never a reserved parser sentinel (``RHS_ASSIGN_ALIAS``/
        ``RETURN_ALIAS``/``PY_BUILTIN_ALIAS``/``ATTR_CALL_ALIAS``), and
        never a real, non-excluded Python builtin name -- a builtin always
        resolves first (see ``utils/script.py``'s ``rewrite_builtin_calls``),
        so a tool registered under a colliding name would be permanently,
        silently unreachable rather than raising here.
        """
        if effective_id in (RHS_ASSIGN_ALIAS, RETURN_ALIAS, PY_BUILTIN_ALIAS, ATTR_CALL_ALIAS):
            raise ToolRegistrationError(
                f"effective id {effective_id!r} is reserved for the parser's "
                "own sentinel tool names and cannot be used as a registered "
                "tool alias or name."
            )
        if hasattr(builtins, effective_id) and effective_id not in EXCLUDED_PY_BUILTINS:
            raise ToolRegistrationError(
                f"effective id {effective_id!r} collides with a real Python "
                "builtin of the same name -- registering a tool under this "
                "name would make it permanently unreachable (the builtin "
                "always resolves first). Choose a different alias."
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
        hub-expanded entries. A bare-name collision among expanded entries
        (or against an already-registered tool) is caught by the usual
        intra-batch/``name_collision_policy`` checks below, same as any other
        duplicate effective id. Whole-batch validation happens before any
        toolbox mutation; intra-batch duplicate effective ids always raise
        regardless of ``name_collision_policy``.

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
                    candidates.append((proxy.name, proxy))

                if isinstance(item, A2AClientHub):
                    try:
                        generic_proxy = toolify(component=item, namespace=self.name)
                    except Exception as exc:
                        raise ToolRegistrationError(
                            f"{type(self).__name__}.{self.name}: failed to toolify "
                            f"generic A2A tool: {exc}"
                        ) from exc
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
    def list_tools(self) -> dict[str, AtomicInvokable]:
        """Shallow copy of the toolbox, keyed by effective id."""
        return dict(self._toolbox)

    def has_tool(self, tool_id: str) -> bool:
        """Return ``True`` if ``tool_id`` (an alias or the tool's own bare ``name``) is registered."""
        return tool_id in self._toolbox

    def get_tool(self, tool_id: str) -> AtomicInvokable:
        """Return the registered invokable for ``tool_id``.

        Raises ``ToolAgentError`` if ``tool_id`` is not registered.
        """
        tool = self._toolbox.get(tool_id)
        if tool is None:
            raise ToolAgentError(f"{type(self).__name__}.{self.name}: unknown tool {tool_id!r}.")
        return tool

    def remove_tool(self, tool_id: str) -> bool:
        """Remove the tool stored under ``tool_id``. Returns whether it was present."""
        return self._toolbox.pop(tool_id, None) is not None

    def clear_tools(self) -> None:
        """Remove every registered tool. No re-registration afterward — there
        is no ``return_tool`` to restore."""
        self._toolbox.clear()

    # ------------------------------------------------------------------ #
    # Constant registration
    # ------------------------------------------------------------------ #
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
        ``f"K_{alias.upper()}"``), exactly as before; omitted, auto-named
        from ``self._constant_counter`` (``f"K_{counter}"`` -- both the dict
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

    # ------------------------------------------------------------------ #
    # Constant accessors
    # ------------------------------------------------------------------ #
    @property
    def constants(self) -> dict[str, ConstantSpec]:
        """Shallow copy of registered constants, keyed by the bare uppercase alias."""
        return dict(self._constants)

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

    # ------------------------------------------------------------------ #
    # Prompt-context rendering
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

            blocks.append(f"{rendered_signature}\n{_render_docstring_block(tool.description)}")

        return "\n---\n".join(blocks)

    def constants_context(self) -> str:
        """
        Render every registered constant for prompt injection, one block
        per constant (a ``K_NAME: type`` annotation line + a 4-space-indented
        triple-quoted docstring description, via ``_render_docstring_block``),
        matching ``actions_context``'s own docstring-style rendering. Names
        print exactly as stored (``K_``-prefixed) — no prefix synthesized
        here. Empty registry renders a "no constants" message.
        """
        if not self._constants:
            return "No constants registered."

        rendered: list[str] = []
        for spec in self._constants.values():
            description = (
                spec.description if spec.description is not None else "No description provided."
            )
            rendered.append(f"{spec.name}: {spec.type}\n{_render_docstring_block(description)}")

        return "\n\n".join(rendered)

    # ------------------------------------------------------------------ #
    # Record construction
    # ------------------------------------------------------------------ #
    def _build_record_from_task(
        self,
        task: ScriptAgentTask,
        turns: list[AgentRecord],
    ) -> ScriptAgentRecord:
        """
        Assemble a completed ``ScriptAgentRecord`` from a finished
        ``ScriptAgentTask``. No agent-level global blackboard to persist
        into (unlike v1 ``ToolAgent``'s span-tracking
        ``update_blackboard`` append) -- each record owns its own slots
        outright, so this is a direct field copy.
        """
        prev = turns[-1] if turns else None
        return ScriptAgentRecord(
            user_prompt=task.user_prompt,
            generated_response=task.generated_response,
            inputs=task.inputs,
            llm_records=tuple(task.llm_records),
            prev=prev,
            statements=tuple(task.completed),
            failed_statements=tuple(task.failed_statements),
        )

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
    # Task-lifecycle hooks
    # ------------------------------------------------------------------ #
    _ATOMIC_IMMUTABLE_TYPES: ClassVar[tuple[type, ...]] = (
        str, int, float, bool, complex, bytes, type(None),
    )

    @classmethod
    def _copy_for_task_namespace(cls, value: Any) -> Any:
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
        if isinstance(value, cls._ATOMIC_IMMUTABLE_TYPES):
            return value
        return copy.deepcopy(value)

    def _initialize_task(
        self,
        *,
        turns: list[AgentRecord],
        prompt: str,
        inputs: dict,
    ) -> ScriptAgentTask:
        """Return a ``ScriptAgentTask`` with the planner system prompt
        active, its ``cache`` pre-seeded with every visible prior turn's
        result under ``task_result_{i}`` -- unconditional over whatever
        ``turns`` contains (empty when there's nothing to seed; already
        gated upstream by conversation-resolution/``context_enabled``/
        ``records_window``) -- and its ``constant_values`` pre-seeded from
        every registered constant. Both are copied via
        ``_copy_for_task_namespace`` exactly once here, not re-derived
        later, so a mutation is visible for the rest of this invocation's
        own rounds but never reaches the real constant or a future
        invocation. No other field needs seeding -- completed/pending/
        resolved_args/continue_planning/planning_rounds_used/
        tool_calls_used/continuation_note/failed_statements all start at
        their dataclass defaults."""
        task = ScriptAgentTask(
            turns=turns, inputs=inputs, user_prompt=prompt, system_prompt_name="planner",
        )
        for turn in turns:
            task.cache[f"task_result_{self._turn_position(turn)}"] = self._copy_for_task_namespace(
                turn.generated_response
            )
        for spec in self._constants.values():
            task.constant_values[spec.name] = self._copy_for_task_namespace(spec.value)
        return task

    def _render_system_message(self, task: ScriptAgentTask) -> list[dict[str, str]]:
        """Renders the active system prompt against tool/constant context.
        Mirrors ``ToolAgent._render_system_message``'s established shape
        exactly: a fresh, framework-controlled context dict, never merged
        with ``task.inputs`` (neither prompt uses an input-derived
        placeholder). No budget content is rendered here -- `tool_calls_limit`
        is a silent, structural-only backstop, never shown to the model
        (see ``_process_generation_output``/``utils.script.validate_references``)."""
        context = {
            "TOOLS": self.actions_context(),
            "CONSTANTS": self.constants_context(),
            "EXCLUDED_PY_BUILTINS": ", ".join(sorted(EXCLUDED_PY_BUILTINS)),
        }
        rendered = self._system_prompts[task.system_prompt_name].render(context)
        return [{"role": "system", "content": rendered}]

    def _render_current_task_message(self, task: ScriptAgentTask) -> dict[str, str]:
        """Bare "what is the task" user message -- reused verbatim for
        round 1 and every continuation round's opening message. Mirrors
        ``ToolAgent._render_task_banner``'s role (dedup a repeated banner
        across every round) scoped to this family's own established
        wording (no ``===== ... =====`` markers -- that's ToolAgent-family
        styling, this family never used it). No "translate this into a
        plan" framing -- ``ONESHOT_PLANNER_PROMPT``'s OBJECTIVE section
        already states that once; repeating it every round would be
        redundant."""
        return {"role": "user", "content": f"CURRENT TASK:\n{task.user_prompt}"}

    def _render_task_messages(self, task: ScriptAgentTask) -> list[dict[str, str]]:
        """Build-once contract per base ``Agent``'s documented pattern.
        Mirrors ``ReActAgent._render_task_messages``'s own 3-part
        organization (banner / assistant-authored state snapshot / user
        instruction) instead of cramming everything into one user message.

        Branches on ``task.continue_planning`` rather than
        ``task.completed`` -- ``continue_planning`` is the one flag
        reliably ``True`` for every real continuation, including the edge
        case where a resolution/execution failure hits on the very first
        batch of round 1 (``prepare()``/``_apply_batch_results`` set it
        directly, before anything ever lands in ``completed``); checking
        ``completed`` alone would silently drop that failure's reason and
        misrender it as a fresh round-1 call.

        Round 1 (``continue_planning`` still ``False``): the banner plus a
        trailing imperative ("Write a plan to accomplish this task now.")
        -- mirrors the continuation branch's own closing instruction, so
        round 1 isn't the one case with no explicit "go" signal right
        before generation -- with the final-round warning appended inline
        when ``self._is_final_round(task)`` (covers ``planning_rounds_limit
        == 1``, where round 1 is immediately the only round permitted)
        after that.

        A continuation round: banner, then an assistant-role state message
        (reconstructed code -- flat, no batch grouping, see
        ``render_completed_as_python`` -- plus ``render_cache_snapshot``'s
        block; directive-free, reads as state not instruction), then a
        user instruction. ``task.continuation_note`` (framework-authored
        only -- never a model-authored pause note, which no longer exists)
        now carries a multi-line block: the failed batch's own rendered
        source plus its labeled issue/failure list (see ``prepare()``/
        ``_apply_batch_results()``). When present, it is prepended as its
        own paragraph before the fixed instruction sentence, rather than
        folded into one inline sentence -- it no longer fits a "previous
        attempt failed: <reason>" framing now that it can span several
        lines. Consulted and cleared back to ``None`` in the same read, so
        a stale, already-addressed note can never leak into a later round.
        The final-round warning is appended last when applicable.
        """
        if task.task_messages:
            return task.task_messages

        banner = self._render_current_task_message(task)

        if not task.continue_planning:
            content = (
                banner["content"]
                + "\n\nWrite a plan to accomplish this task now."
            )
            if self._is_final_round(task):
                content += " " + FINAL_ROUND_WARNING
            task.task_messages = [{"role": "user", "content": content}]
            return task.task_messages

        snapshot = render_completed_as_python(task.completed) or "(nothing completed yet)"
        cache_snapshot = render_cache_snapshot(
            task.completed, task.cache, self._response_preview_limit
        )
        cache_section = f"\n\n{cache_snapshot}" if cache_snapshot else ""
        state_message = {
            "role": "assistant",
            "content": f"# WORK COMPLETED SO FAR:\n{snapshot}{cache_section}",
        }

        note = task.continuation_note
        task.continuation_note = None
        fixed_instruction = (
            "Continue planning the rest of this task. Use the existing "
            "work done to guide you on what the next steps should be."
        )
        instruction = f"{note}\n\n{fixed_instruction}" if note else fixed_instruction
        if self._is_final_round(task):
            instruction += " " + FINAL_ROUND_WARNING

        task.task_messages = [banner, state_message, {"role": "user", "content": instruction}]
        return task.task_messages

    # ------------------------------------------------------------------ #
    # Generation (think())
    # ------------------------------------------------------------------ #
    def _process_generation_output(
        self, raw_text: str, task: ScriptAgentTask,
    ) -> tuple[list[list[CodeStatement]], bool] | str:
        """
        Pure-computation validate callback for the planning retry loop:
        parse, validate references + remaining tool-call budget + the
        final-round pause prohibition, and compile into batches. Returns
        the compiled result on success, or a feedback string describing
        every problem found on failure -- a ``BlackboardParseError`` from
        parsing is converted here, not propagated, so the retry loop can
        inject it as corrective feedback.
        """
        try:
            flat_slots, continue_planning = parse_generation(raw_text)
        except BlackboardParseError as e:
            return str(e)

        known_tools = frozenset(self._toolbox.keys())
        # Rewrite eligible builtin calls before whole-sequence validation,
        # so a rewritten slot is validated as PY_BUILTIN_ALIAS, not as an
        # unregistered tool; excluded-but-real builtin names are reported
        # here with a specific message instead of validate_references'
        # generic "unregistered tool" one.
        builtin_issues = rewrite_builtin_calls(flat_slots)
        # Derived from each ConstantSpec's own .name (already correctly
        # K_-prefixed exactly once for both the auto-named and aliased
        # registration paths -- see register_constant/register_constants),
        # never reconstructed by re-prefixing the internal dict key itself:
        # an auto-named key is already "K_0"-shaped, so blindly prepending
        # "K_" again would double-prefix it ("K_K_0"), silently mismatching
        # the name actually shown to the model via constants_context().
        known_constants = frozenset(spec.name for spec in self._constants.values())
        # Every key already in task.cache is safe to reference by name here:
        # cross-invocation task_result_i entries (seeded in
        # _initialize_task) AND, on a continuation round, every identifier
        # bound by an earlier round's own completed slots this same
        # invoke -- the exact names the "work completed so far" snapshot
        # (_render_task_messages) shows the model. Not scoped to the
        # TASK_RESULT_PREFIX any more; that was only ever correct back when
        # every generation was a fresh, single-round invocation.
        known_history = frozenset(task.cache.keys())
        remaining_budget = (
            None if self._tool_calls_limit is None
            else self._tool_calls_limit - task.tool_calls_used
        )
        issues = builtin_issues + validate_references(
            flat_slots, known_tools, known_constants, known_history, remaining_budget
        )

        if self._is_final_round(task) and continue_planning:
            issues.append(
                "this was your final planning round -- you may not pause "
                "again; produce a complete plan with no trailing # PAUSE, "
                "ending in return."
            )

        if issues:
            issues_msg = "\n".join(f"{i + 1}. {m}" for i, m in enumerate(issues))
            return issues_msg

        pending = compile_batches(
            flat_slots,
            max_concurrency=self._tool_concurrency_limit,
            start_batch_index=task.batch_counter,
        )
        task.batch_counter += len(pending)
        return pending, continue_planning

    def _run_planning_retry_loop(
        self, *, task: ScriptAgentTask,
    ) -> tuple[list[list[CodeStatement]], bool]:
        """
        Render, call the engine, record the attempt, validate/compile via
        ``_process_generation_output``, and retry with injected feedback on
        failure until success or the regeneration budget
        (``self._regeneration_limit``, tracked via
        ``task.regenerations_used``) is exhausted. ``regeneration_limit``
        is always a plain ``int`` (never ``None``), so this check is a
        direct comparison -- no ``None``-guard needed, unlike
        ``planning_rounds_limit``/``tool_calls_limit`` elsewhere in this
        class.
        """
        additional_messages: list[dict[str, str]] = []

        while True:
            messages = self.render_task(task, additional_messages=additional_messages)
            engine_result = self._llm_engine.invoke({"messages": messages})
            raw_output: str = engine_result.result

            task.llm_records.append(LLMRecord(
                messages=list(task.task_messages),
                llm_result=engine_result,
                system_prompt_name=task.system_prompt_name,
            ))

            result = self._process_generation_output(raw_output, task)
            if isinstance(result, str):
                if task.regenerations_used >= self._regeneration_limit:
                    raise ToolAgentError(
                        f"{type(self).__name__}.{self.name}: regeneration budget "
                        f"exhausted after {task.regenerations_used + 1} attempt(s). "
                        f"Last feedback: {result}"
                    )
                additional_messages = [
                    {"role": "assistant", "content": raw_output},
                    {"role": "user", "content": (
                        f"Your plan could not be used:\n\n{result}\n\n"
                        "Produce a corrected plan."
                    )},
                ]
                task.regenerations_used += 1
                continue

            return result

    async def _arun_planning_retry_loop(
        self, *, task: ScriptAgentTask,
    ) -> tuple[list[list[CodeStatement]], bool]:
        """Async mirror of ``_run_planning_retry_loop``: uses
        ``async_invoke`` for the engine call, otherwise identical."""
        additional_messages: list[dict[str, str]] = []

        while True:
            messages = self.render_task(task, additional_messages=additional_messages)
            engine_result = await self._llm_engine.async_invoke({"messages": messages})
            raw_output: str = engine_result.result

            task.llm_records.append(LLMRecord(
                messages=list(task.task_messages),
                llm_result=engine_result,
                system_prompt_name=task.system_prompt_name,
            ))

            result = self._process_generation_output(raw_output, task)
            if isinstance(result, str):
                if task.regenerations_used >= self._regeneration_limit:
                    raise ToolAgentError(
                        f"{type(self).__name__}.{self.name}: regeneration budget "
                        f"exhausted after {task.regenerations_used + 1} attempt(s). "
                        f"Last feedback: {result}"
                    )
                additional_messages = [
                    {"role": "assistant", "content": raw_output},
                    {"role": "user", "content": (
                        f"Your plan could not be used:\n\n{result}\n\n"
                        "Produce a corrected plan."
                    )},
                ]
                task.regenerations_used += 1
                continue

            return result

    def _is_final_round(self, task: ScriptAgentTask) -> bool:
        """
        True iff the round currently being generated is the last one
        ``planning_rounds_limit`` permits -- computed on demand from
        ``task.planning_rounds_used`` vs. ``self._planning_rounds_limit``,
        never stored, so the render and validation sites that both consult
        it can never see it drift out of sync. ``None`` limit means never
        final (unlimited rounds). ``planning_rounds_used`` is already
        incremented (by ``think()``/``async_think()``, unconditionally,
        including for round 1) before this is ever consulted, so the
        comparison alone correctly covers round 1 too when the limit is
        ``1`` -- there's no separate zero-based special case to reason
        about.

        A final round can no longer successfully re-pause: if it still
        writes ``# PAUSE`` anyway, ``_process_generation_output`` rejects
        that as a regen-repair issue instead of granting a continuation --
        there is deliberately no dedicated "planning round budget
        exhausted" raise anymore (unlike ``tool_calls_limit``): a round
        beyond the limit is never reachable, since a final round either
        completes or exhausts ``regeneration_limit`` first.
        """
        return (
            self._planning_rounds_limit is not None
            and task.planning_rounds_used >= self._planning_rounds_limit
        )

    def _finalize_without_continuation(self, task: ScriptAgentTask) -> ScriptAgentTask:
        """A drain with no ``# PAUSE`` and no error -- the absence of a
        ``return`` is NOT an invitation to keep planning, only an explicit
        ``# PAUSE`` is. Infers ``None`` if nothing was ever returned
        and marks the task complete. Called from wherever ``task.pending``
        actually reaches empty with ``continue_planning`` still ``False``:
        ``prepare`` (an already-empty round, or a same-round empty
        generation) and ``_apply_batch_results`` (the last batch of a
        multi-round plan draining) -- never from ``think``, which only ever
        reads ``continue_planning``, never decides based on it."""
        if task.generated_response is NO_VAL:
            task.generated_response = None
        task.complete = True
        return task

    def think(self, task: ScriptAgentTask) -> ScriptAgentTask:
        """
        Generate, validate, and compile the next segment of the plan --
        either the unconditional first generation, or (once a prior round
        set ``continue_planning``) a fresh continuation. No-op whenever
        there is still pending work to drain, or the task is already fully
        complete -- a pause/if-cutoff/failure-triggered continuation is
        requested by re-entering this same hook, not a separate mechanism.

        Never called for a drained, uninvited round: whichever of
        ``prepare``/``_apply_batch_results`` actually empties
        ``task.pending`` with ``continue_planning`` still ``False`` marks
        ``task.complete`` there and then, so this hook's own top guard
        already short-circuits before ever reaching the regeneration call
        below -- no separate check needed here.

        ``task.planning_rounds_used`` increments unconditionally on every
        real call to this hook, including the first -- ``planning_rounds_limit``
        bounds the total number of planning generations permitted for this
        invoke, not a count of continuations beyond a free first one. This
        also means a ``prepare()`` resolution failure on the very first
        batch of round 1 (which sets ``continue_planning`` before anything
        has ever completed) is correctly counted the same as any other
        round -- there is no separate signal to consult here, just a plain
        increment every time this hook actually runs.
        """
        if task.pending or task.complete:
            return task

        task.planning_rounds_used += 1

        pending, continue_planning = self._run_planning_retry_loop(task=task)
        task.pending = pending
        task.continue_planning = continue_planning
        task.task_messages.clear()
        return task

    async def async_think(self, task: ScriptAgentTask) -> ScriptAgentTask:
        """Async mirror of ``think``, using ``_arun_planning_retry_loop``."""
        if task.pending or task.complete:
            return task

        task.planning_rounds_used += 1

        pending, continue_planning = await self._arun_planning_retry_loop(task=task)
        task.pending = pending
        task.continue_planning = continue_planning
        task.task_messages.clear()
        return task

    def _resolve_dispatch_tool(self, tool_id: str) -> AtomicInvokable:
        """Resolve a slot's ``tool`` id to the actual invokable to dispatch
        -- the shared ``PY_BUILTIN_ALIAS``/``ATTR_CALL_ALIAS``-vs-registered-
        tool selection used by ``prepare()`` and ``_gather_batch_results``."""
        if tool_id == PY_BUILTIN_ALIAS:
            return builtin_call_tool
        if tool_id == ATTR_CALL_ALIAS:
            return attr_call_tool
        return self.get_tool(tool_id)

    # ------------------------------------------------------------------ #
    # Prepare next batch
    # ------------------------------------------------------------------ #
    def prepare(self, task: ScriptAgentTask) -> ScriptAgentTask:
        """
        Resolve the next pending batch's args, or short-circuit completion
        (or a needed continuation) if nothing remains.

        If ``task.pending`` is empty: reset ``task.resolved_args``, then
        either leave the round as-is if ``task.continue_planning`` is
        already set (nothing to prepare -- ``think()`` will regenerate next
        round) or, otherwise, infer an implicit ``return None`` if no
        executed ``return`` slot already set ``task.generated_response``
        and mark the task complete -- covers both a genuinely empty
        generation and the natural end-of-plan drain, so ``act()`` needs
        only a bare no-op guard, not a second check.

        Otherwise: resolves every slot's args in ``task.pending[0]``,
        collecting every failure (not stopping at the first). Any collected
        issue no longer raises -- it abandons this batch and every batch
        still queued after it (they may depend on bindings this batch was
        supposed to produce), requires a continuation round, and surfaces
        the real issue text as ``continuation_note`` for the next
        generation to read. Nothing here was ever dispatched, so this does
        not consume ``tool_calls_used``.
        """
        if not task.pending:
            task.resolved_args = []
            if task.continue_planning:
                return task
            return self._finalize_without_continuation(task)

        batch = task.pending[0]
        resolved: list[dict[str, Any]] = []
        issues: list[str] = []
        # Constants are validated as known references (validate_references'
        # known_constants) and rendered to the model (constants_context()),
        # but their actual runtime values live in task.constant_values --
        # a per-invocation copy seeded once by _initialize_task, never
        # re-derived from self._constants here (that would hand out the
        # live, shared constant object fresh every batch, defeating the
        # "a mutation stays visible for the rest of this invocation, never
        # leaks to another" guarantee). Merged in here, once per batch, so
        # a correct K_NAME reference actually resolves instead of raising
        # NameError; task.cache last so a real bound/history name would win
        # on the (never expected) collision.
        resolution_namespace = {**task.constant_values, **task.cache}
        for slot in batch:
            label = slot.identifier if slot.identifier is not None else "(unassigned)"

            try:
                positional, keyword = resolve_slot_args(slot, resolution_namespace)
            except Exception as e:
                issues.append(f"{label}: could not resolve argument value(s): {e!r}")
                continue

            if slot.tool in (RHS_ASSIGN_ALIAS, RETURN_ALIAS):
                resolved.append({"val": keyword["val"]})
                continue

            try:
                tool = self._resolve_dispatch_tool(slot.tool)
                resolved.append(tool._args_kwargs_to_dict(*positional, **keyword))
            except Exception as e:
                issues.append(
                    f"{label}: argument(s) do not match {slot.tool!r}'s "
                    f"parameter contract: {e!r}"
                )

        if issues:
            issues_msg = "\n".join(f"{i + 1}. {m}" for i, m in enumerate(issues))
            task.continuation_note = (
                "The following batch could not be resolved:\n"
                f"{render_completed_as_python(batch)}"
                f"\n\nIssues:\n{issues_msg}"
            )
            task.pending.clear()
            task.resolved_args = []
            task.continue_planning = True
            return task

        task.resolved_args = resolved
        return task

    async def async_prepare(self, task: ScriptAgentTask) -> ScriptAgentTask:
        """Direct passthrough to ``prepare`` -- no I/O of its own."""
        return self.prepare(task)

    # ------------------------------------------------------------------ #
    # Execute prepared batch
    # ------------------------------------------------------------------ #
    async def _gather_batch_results(
        self, batch: list[CodeStatement], resolved: list[dict[str, Any]],
    ) -> list[Any]:
        """
        Dispatch every dispatched-call slot in ``batch`` (registered tool
        or approved builtin, via ``is_dispatched_slot``) concurrently;
        ``rhs_assign``/``return`` slots need no dispatch, their result is
        already the resolved ``"val"`` value. Shared by ``act``/
        ``async_act`` -- both differ only in how the resulting coroutine is
        driven.
        """
        coros: list[Any] = []
        dispatch_map: dict[int, int] = {}
        for i, slot in enumerate(batch):
            if is_dispatched_slot(slot):
                dispatch_map[i] = len(coros)
                tool = self._resolve_dispatch_tool(slot.tool)
                coros.append(tool.async_invoke(resolved[i]))

        gathered = await asyncio.gather(*coros, return_exceptions=True) if coros else []
        return [
            gathered[dispatch_map[i]] if i in dispatch_map else resolved[i]["val"]
            for i in range(len(batch))
        ]

    def _apply_batch_results(
        self,
        task: ScriptAgentTask,
        batch: list[CodeStatement],
        resolved: list[dict[str, Any]],
        raw_results: list[Any],
    ) -> ScriptAgentTask:
        """
        Shared post-gather bookkeeping for ``act``/``async_act``: apply
        results, update completed/cache, handle a terminal ``return`` slot,
        pop the consumed batch -- or, if any real call in this batch
        failed, record whichever succeeded, abandon the rest of this round,
        and require a continuation instead of raising.

        Every dispatched call in ``batch`` (registered tool or approved
        builtin, via ``is_dispatched_slot``) was actually dispatched via
        ``asyncio.gather`` regardless of whether any of them failed, so all
        of them count against ``tool_calls_used`` unconditionally, before
        checking for failures.
        """
        real_call_count = sum(1 for slot in batch if is_dispatched_slot(slot))
        task.tool_calls_used += real_call_count

        def _failure_label(slot: CodeStatement) -> str:
            # A py_builtin slot's real, model-written name lives in
            # args[0], not slot.tool; an attr_call slot's lives in
            # "obj.method" (args[0]/args[1]) -- report either instead of
            # leaking the internal sentinel into model-facing failure
            # feedback.
            if slot.tool == PY_BUILTIN_ALIAS:
                return repr(slot.args[0])
            if slot.tool == ATTR_CALL_ALIAS:
                return f"{ast.unparse(slot.args[0])}.{slot.args[1]}"
            return repr(slot.tool)

        failures = [
            f"{_failure_label(batch[idx])} (identifier={batch[idx].identifier!r}): {raw!r}"
            for idx, raw in enumerate(raw_results)
            if isinstance(raw, BaseException)
        ]

        for slot, value in zip(batch, raw_results):
            if isinstance(value, BaseException):
                slot.exception = value
                task.failed_statements.append(slot)
                continue
            task.completed.append(slot)
            # A dispatched tool call's raw_results entry is a full
            # AtomicResult envelope (slot.result stores it verbatim,
            # matching v1's board[idx].result precedent); rhs_assign/return
            # slots were never dispatched, so their value is already the
            # plain resolved Python value -- no envelope to unwrap.
            if slot.tool not in (RHS_ASSIGN_ALIAS, RETURN_ALIAS):
                slot.result = value
                unwrapped = value.result
            else:
                unwrapped = value
            if slot.identifier is not None:
                task.cache[slot.identifier] = unwrapped

        if failures:
            failures_msg = "\n".join(f"{i + 1}. {m}" for i, m in enumerate(failures))
            task.continuation_note = (
                "The following batch encountered execution failures:\n"
                f"{render_completed_as_python(batch)}"
                f"\n\nFailures:\n{failures_msg}"
            )
            task.pending.clear()
            task.resolved_args = []
            task.continue_planning = True
            return task

        for slot, kwargs in zip(batch, resolved):
            if slot.tool == RETURN_ALIAS:
                task.generated_response = kwargs["val"]
                task.complete = True

        task.pending.pop(0)
        task.resolved_args = []

        # This batch just drained the plan. If nothing asked for a
        # continuation (no `# PAUSE`) and nothing already completed it
        # (no `return` above), finalize right here -- the natural point
        # `task.pending` actually reaches empty -- instead of leaving it for
        # a future `prepare()` call that `think()` would otherwise reach
        # first on the next loop iteration and regenerate an uninvited round.
        if not task.pending and not task.complete and not task.continue_planning:
            return self._finalize_without_continuation(task)

        return task

    def act(self, task: ScriptAgentTask) -> ScriptAgentTask:
        """
        Execute the currently resolved batch, or no-op if ``prepare``
        produced nothing to run this round (covers a short-circuited round
        from ``prepare``'s empty-``pending`` guard).
        """
        if not task.resolved_args:
            return task

        batch = task.pending[0]
        resolved = task.resolved_args
        raw_results = run_coro_sync(self._gather_batch_results(batch, resolved))
        return self._apply_batch_results(task, batch, resolved, raw_results)

    async def async_act(self, task: ScriptAgentTask) -> ScriptAgentTask:
        """Async mirror of ``act``; awaits ``_gather_batch_results``
        directly rather than ``run_coro_sync``-wrapping it."""
        if not task.resolved_args:
            return task

        batch = task.pending[0]
        resolved = task.resolved_args
        raw_results = await self._gather_batch_results(batch, resolved)
        return self._apply_batch_results(task, batch, resolved, raw_results)
