from __future__ import annotations

import asyncio
from typing import Any, Callable, Literal, Optional

from ..mcp.MCPClientHub import MCPClientHub
from ..a2a.A2AClientHub import A2AClientHub
from ..a2a.PyA2AtomicClient import PyA2AtomicClient

from .base import Agent
from .prompts import ONESHOT_PLANNER_PROMPT
from ..core.Invokable import AtomicInvokable
from ..llm.base import LLMEngine
from ..tools.Toolify import toolify
from ..models.agents.blackboard_models import ConstantSpec
from ..models.agents.records import AgentRecord, LLMRecord, ToolAgentRecordV2
from ..models.agents.tasks import ToolAgentTaskV2
from ..models.agents.toolagent2_models import BlackboardSlotV2
from ..constants.core import IDENTIFIER_PATTERN, NO_VAL
from ..constants.toolagent2 import RETURN_ALIAS, RHS_ASSIGN_ALIAS
from ..exceptions import BlackboardParseError, ToolAgentError, ToolRegistrationError
from ..utils.core import run_coro_sync
from ..utils.toolagent2 import (
    compile_batches,
    parse_generation,
    render_completed_as_python,
    resolve_slot_args,
    validate_references,
)

_HubClient = MCPClientHub | A2AClientHub | PyA2AtomicClient
_COLLISION_POLICIES = ("raise", "skip", "replace")


class ToolAgent2(Agent):
    """
    Adaptive, one-shot-planning tool-invoking agent (sibling family to
    ``ToolAgent``, not a subclass). Writes native-grammar, Python-style
    statements toward a task from a single generated plan. There is no
    separate decomposition, orchestration, or synthesis call, and no
    construction-time mode knob -- adaptivity is meant to be emergent from
    how many ``# CHECKPOINT`` markers end up in one continuous plan, not
    picked up front.

    Checkpoint-triggered reactive continuation (Pass 2.3): a generation
    always terminates at the first of a ``return``, an explicit
    ``# CHECKPOINT`` (with an optional trailing triple-quoted note), or a
    defensively-pruned ``if`` statement (the grammar still forbids
    conditionals outright; a model that writes one anyway is handled by
    silent truncation, not rejection, unless nothing real precedes it). A
    resolution failure in ``prepare()`` or a real tool-execution failure in
    ``act()`` are handled the same way -- none of these four cases raise
    anymore. Whichever one occurs (other than ``return``) sets
    ``task.continue_planning``, and ``think()`` re-invokes the same planner
    for a fresh continuation, seeing a Python-source snapshot of the work
    already done this invoke plus a reason (the model's own note, a
    hardcoded default, or the real failure text, depending on which of the
    four cases applies) -- no separate judge/critic model. `tool_calls_limit`
    is a budget shared across every generation round in one invoke, not a
    fresh allowance per round; `planning_rounds_limit` separately bounds how
    many continuation rounds are allowed, independent of
    `generation_retries` (which governs only per-generation malformed-draft
    regen-repair). Exactly one of `tool_calls_limit`/`planning_rounds_limit`
    may be `None` -- never both.

    Cross-invocation result addressing is implemented: a prior turn's
    result is seeded into `task.cache` and labeled in rendered history as
    `task_result_i`, a fixed, read-only reference a later plan can use by
    name.
    """

    def __init__(
        self,
        name: str,
        namespace: str,
        description: str,
        llm_engine: LLMEngine,
        context_enabled: bool = False,
        *,
        generation_retries: Optional[int] = 0,
        tool_calls_limit: Optional[int] = None,
        planning_rounds_limit: Optional[int] = None,
        response_preview_limit: Optional[int] = None,
        pre_invoke: Optional[AtomicInvokable | Callable[..., Any]] = None,
        post_invoke: Optional[AtomicInvokable | Callable[..., Any]] = None,
        post_result_key: Optional[str] = None,
        records_window: Optional[int] = None,
        assistant_response_source: Literal["raw", "final"] = "raw",
    ) -> None:
        """
        Validate/store ``generation_retries``, then ``tool_calls_limit``/
        ``planning_rounds_limit`` together (at most one may be ``None``),
        then initialize empty toolbox/constants storage. ``extra_parameters``
        is never forwarded to ``super().__init__`` — matches
        ``ToolAgent.__init__``'s own precedent.
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

        if generation_retries is not None and (
            type(generation_retries) is not int or generation_retries < 0
        ):
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: generation_retries must be "
                f"None or an int >= 0; got {generation_retries!r}."
            )
        self._generation_retries = generation_retries

        # Cross-field validation (tool_calls_limit/planning_rounds_limit may
        # not both be None) needs each field's other value already present
        # before either setter's own cross-check runs below -- pre-seed the
        # raw attribute directly (bypassing its own validation transiently)
        # so the first setter call sees the real final value, not a
        # placeholder. Each field still gets its own full validation via its
        # own setter call.
        self._planning_rounds_limit: Optional[int] = planning_rounds_limit
        self._tool_calls_limit: Optional[int] = None
        self.tool_calls_limit = tool_calls_limit
        self.planning_rounds_limit = planning_rounds_limit

        self._toolbox: dict[str, AtomicInvokable] = {}
        self._constants: dict[str, ConstantSpec] = {}

        self._system_prompts["planner"] = ONESHOT_PLANNER_PROMPT

    # ------------------------------------------------------------------ #
    # Construction-time / mutable knobs
    # ------------------------------------------------------------------ #
    @property
    def generation_retries(self) -> Optional[int]:
        """Bounded-attempts ceiling shared by structural generation retries and
        (later pass) failure-triggered checklist regeneration. ``None`` means
        unlimited. Read-only."""
        return self._generation_retries

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
        if value is None and self._planning_rounds_limit is None:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: tool_calls_limit and "
                "planning_rounds_limit cannot both be None -- that would "
                "leave both the tool-call budget and the replanning-round "
                "budget unbounded at once."
            )
        self._tool_calls_limit = value

    @property
    def planning_rounds_limit(self) -> Optional[int]:
        """Max continuation rounds (checkpoint/if-cutoff/failure-triggered
        re-generations) allowed per ``invoke()`` run -- the unconditional
        first generation never counts against this. ``None`` means
        unlimited."""
        return self._planning_rounds_limit

    @planning_rounds_limit.setter
    def planning_rounds_limit(self, value: Optional[int]) -> None:
        if value is not None and (type(value) is not int or value < 0):
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: planning_rounds_limit "
                f"must be None or an int >= 0; got {value!r}."
            )
        if value is None and self._tool_calls_limit is None:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: tool_calls_limit and "
                "planning_rounds_limit cannot both be None -- that would "
                "leave both the tool-call budget and the replanning-round "
                "budget unbounded at once."
            )
        self._planning_rounds_limit = value

    # ------------------------------------------------------------------ #
    # Shared helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalize_collision_policy(name_collision_policy: str) -> str:
        policy = name_collision_policy.lower().strip()
        if policy not in _COLLISION_POLICIES:
            raise ToolRegistrationError(
                "name_collision_policy must be one of: 'raise', 'skip', 'replace'."
            )
        return policy

    @staticmethod
    def _validate_tool_alias(alias: Optional[str]) -> None:
        if alias is not None and (
            not isinstance(alias, str) or not IDENTIFIER_PATTERN.fullmatch(alias)
        ):
            raise ToolRegistrationError(
                f"alias must be None or a Python-identifier-legal string; got {alias!r}."
            )
        if alias in (RHS_ASSIGN_ALIAS, RETURN_ALIAS):
            raise ToolRegistrationError(
                f"alias {alias!r} is reserved for the parser's own sentinel tool "
                "names and cannot be used as a registered tool alias."
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
        absent one, its own ``full_name``. ``AtomicInvokable`` inputs are
        stored as-is (never re-``toolify``'d); ``description`` is a
        documented no-op in that case — ``alias`` alone covers the
        "cover name" use case. Callables are normalized via
        ``toolify(namespace=self.name)``.

        Returns whether the tool was newly registered (``False`` only when
        ``name_collision_policy="skip"`` hits an existing effective id).
        """
        policy = self._normalize_collision_policy(name_collision_policy)

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
        effective_id = alias if alias is not None else invokable.full_name

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
        tools: list[AtomicInvokable | Callable | _HubClient],
        aliases: Optional[list[Optional[str]]] = None,
        *,
        name_collision_policy: Literal["raise", "skip", "replace"] = "raise",
    ) -> bool:
        """
        Register a mixed batch of plain invokables/callables and
        hub/client objects (``MCPClientHub``/``A2AClientHub``/
        ``PyA2AtomicClient``) in one pass. A hub/client entry expands into
        every tool it exposes, each under its own intrinsic ``full_name`` —
        no aliasing possible for hub-expanded entries. Whole-batch validation
        happens before any toolbox mutation; intra-batch duplicate effective
        ids always raise regardless of ``name_collision_policy``.

        Returns ``True`` iff every item was newly registered (only ``"skip"``
        can make this ``False``; ``"raise"``/``"replace"`` are unconditionally
        ``True`` once the call doesn't raise).
        """
        policy = self._normalize_collision_policy(name_collision_policy)

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
                    candidates.append((proxy.full_name, proxy))

                if isinstance(item, A2AClientHub):
                    try:
                        generic_proxy = toolify(component=item, namespace=self.name)
                    except Exception as exc:
                        raise ToolRegistrationError(
                            f"{type(self).__name__}.{self.name}: failed to toolify "
                            f"generic A2A tool: {exc}"
                        ) from exc
                    candidates.append((generic_proxy.full_name, generic_proxy))

            elif isinstance(item, AtomicInvokable):
                self._validate_tool_alias(item_alias)
                effective_id = item_alias if item_alias is not None else item.full_name
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
                effective_id = item_alias if item_alias is not None else invokable.full_name
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
        """Return ``True`` if ``tool_id`` (an alias or ``full_name``) is registered."""
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
        alias: str,
        description: Optional[str] = None,
        *,
        name_collision_policy: Literal["raise", "skip", "replace"] = "raise",
    ) -> bool:
        """
        Register one named runtime constant. ``alias`` is required (a raw
        value has no intrinsic identity to fall back to); it is stored under
        ``alias.upper()`` (no ``K_`` prefix — the dict key stays search-
        friendly), while the wire-facing ``ConstantSpec.name`` is
        unconditionally ``f"K_{alias.upper()}"``.

        Returns whether the constant was newly registered (``False`` only
        when ``name_collision_policy="skip"`` hits an existing key).
        """
        policy = self._normalize_collision_policy(name_collision_policy)

        if not isinstance(alias, str) or not alias.strip():
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: alias must be a non-empty string; "
                f"got {alias!r}."
            )

        key = alias.upper()
        spec = ConstantSpec(name=f"K_{key}", value=constant, description=description)

        if key in self._constants:
            if policy == "raise":
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: constant already registered: "
                    f"{key!r}."
                )
            if policy == "skip":
                return False
            # "replace" falls through to the unconditional overwrite below.

        self._constants[key] = spec
        return True

    def register_constants(
        self,
        constants: list[Any],
        aliases: list[str],
        *,
        name_collision_policy: Literal["raise", "skip", "replace"] = "raise",
    ) -> bool:
        """
        Register a batch of constants; ``aliases`` is required and must match
        ``constants`` in length. Whole-batch validation before any mutation;
        intra-batch duplicate ``alias.upper()`` keys always raise regardless
        of ``name_collision_policy``.

        Returns ``True`` iff every item was newly registered.
        """
        policy = self._normalize_collision_policy(name_collision_policy)

        if len(aliases) != len(constants):
            raise ValueError(
                f"{type(self).__name__}.{self.name}: aliases must be the same "
                f"length as constants; got {len(aliases)} vs {len(constants)}."
            )

        candidates: dict[str, ConstantSpec] = {}
        for value, alias in zip(constants, aliases):
            if not isinstance(alias, str) or not alias.strip():
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: alias must be a non-empty "
                    f"string; got {alias!r}."
                )

            key = alias.upper()
            if key in candidates:
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: duplicate constant name in "
                    f"batch: {key!r}."
                )
            candidates[key] = ConstantSpec(name=f"K_{key}", value=value)

        any_skipped = False
        vetted: dict[str, ConstantSpec] = {}
        for key, spec in candidates.items():
            if key in self._constants:
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

    def clear_constants(self) -> None:
        """Remove every registered constant."""
        self._constants.clear()

    # ------------------------------------------------------------------ #
    # Prompt-context rendering
    # ------------------------------------------------------------------ #
    def actions_context(self) -> str:
        """
        Render every registered tool for prompt injection, one block per
        tool (signature line + indented description), joined by ``"\\n---\\n"``.
        When ``tid`` is an alias (not the tool's own ``full_name``), only the
        leading identity token of ``tool.signature`` is swapped for the alias
        — the ``(args) -> ReturnType`` portion renders unchanged.
        """
        blocks: list[str] = []
        for tid, tool in self._toolbox.items():
            if tid == tool.full_name:
                rendered_signature = tool.signature
            else:
                rendered_signature = tid + tool.signature[len(tool.full_name):]

            indented_lines = [
                line if not line.strip() else f"  {line}"
                for line in tool.description.splitlines()
            ]
            blocks.append(f"{rendered_signature}\n" + "\n".join(indented_lines))

        return "\n---\n".join(blocks)

    def constants_context(self) -> str:
        """
        Render every registered constant for prompt injection. Names print
        exactly as stored (``K_``-prefixed) — no prefix synthesized here.
        Empty registry renders a "no constants" message.
        """
        if not self._constants:
            return "No constants registered."

        rendered: list[str] = []
        for spec in self._constants.values():
            description = (
                spec.description if spec.description is not None else "No description provided."
            )
            rendered.append(
                f"- {spec.name}\n"
                f"  Type: {spec.type}\n"
                f"  Description: {description}"
            )

        return "\n\n".join(rendered)

    # ------------------------------------------------------------------ #
    # Record construction
    # ------------------------------------------------------------------ #
    def _build_record_from_task(
        self,
        task: ToolAgentTaskV2,
        turns: list[AgentRecord],
    ) -> ToolAgentRecordV2:
        """
        Assemble a completed ``ToolAgentRecordV2`` from a finished
        ``ToolAgentTaskV2``. No agent-level global blackboard to persist
        into (unlike v1 ``ToolAgent``'s span-tracking
        ``update_blackboard`` append) -- each record owns its own slots
        outright, so this is a direct field copy.
        """
        prev = turns[-1] if turns else None
        return ToolAgentRecordV2(
            user_prompt=task.user_prompt,
            generated_response=task.generated_response,
            inputs=task.inputs,
            llm_records=tuple(task.llm_records),
            prev=prev,
            blackboard=tuple(task.completed),
            annotations=tuple(task.annotations),
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
    def _initialize_task(
        self,
        *,
        turns: list[AgentRecord],
        prompt: str,
        inputs: dict,
    ) -> ToolAgentTaskV2:
        """Return a ``ToolAgentTaskV2`` with the planner system prompt
        active, its ``cache`` pre-seeded with every visible prior turn's
        result under ``task_result_{i}`` -- unconditional over whatever
        ``turns`` contains (empty when there's nothing to seed; already
        gated upstream by conversation-resolution/``context_enabled``/
        ``records_window``). No other field needs seeding -- completed/
        pending/annotations/resolved_args/continue_planning/
        planning_rounds_used/tool_calls_used/continuation_note all start at
        their dataclass defaults."""
        task = ToolAgentTaskV2(
            turns=turns, inputs=inputs, user_prompt=prompt, system_prompt_name="planner",
        )
        for turn in turns:
            task.cache[f"task_result_{self._turn_position(turn)}"] = turn.generated_response
        return task

    def _render_system_message(self, task: ToolAgentTaskV2) -> list[dict[str, str]]:
        """Renders the active system prompt against tool/constant/budget
        context. Mirrors ``ToolAgent._render_system_message``'s established
        shape exactly: a fresh, framework-controlled context dict, never
        merged with ``task.inputs`` (neither prompt uses an input-derived
        placeholder). ``{TOOL_CALLS_LIMIT}`` renders the REMAINING budget
        (the configured limit minus every real call dispatched so far this
        invoke, across every generation round) -- on the first generation
        this is numerically identical to the configured limit, since
        nothing has been dispatched yet."""
        if self._tool_calls_limit is None:
            limit_text = "unlimited"
        else:
            limit_text = str(self._tool_calls_limit - task.tool_calls_used)
        context = {
            "TOOLS": self.actions_context(),
            "TOOL_CALLS_LIMIT": limit_text,
            "CONSTANTS": self.constants_context(),
        }
        rendered = self._system_prompts[task.system_prompt_name].render(context)
        return [{"role": "system", "content": rendered}]

    def _render_task_messages(self, task: ToolAgentTaskV2) -> list[dict[str, str]]:
        """Build-once contract per base ``Agent``'s documented pattern: a
        single user message carrying the task's own prompt, unchanged
        across generation retries within this phase.

        ``task.completed`` empty means this is the unconditional first
        generation -- rendered exactly as before. Non-empty means this is a
        checkpoint/if-cutoff/failure-triggered continuation (this method is
        only ever called when ``task.pending`` is already empty, so a
        non-empty ``completed`` here can only mean a genuine prior round,
        never a same-round partial drain) -- rendered instead as a snapshot
        of the work already done plus whatever reason/note applies, asking
        for only the remaining plan.
        """
        if task.task_messages:
            return task.task_messages

        if not task.completed:
            current_task_framed_prompt = (
                "Given the below latest message, translate the request into "
                "a python-formatted plan:\n"
                f"\nCURRENT TASK:\n{task.user_prompt}"
            )
        else:
            snapshot = render_completed_as_python(task.completed, self._response_preview_limit)
            # `continuation_note` is `None` for a silent if-cutoff -- omit
            # that section entirely rather than interpolating the literal
            # text "None" into the rendered message, which would violate
            # the if-cutoff's own "purely defensive, zero visibility" intent.
            note_section = f"{task.continuation_note}\n\n" if task.continuation_note else ""
            current_task_framed_prompt = (
                f"WORK COMPLETED SO FAR:\n{snapshot}\n\n"
                f"{note_section}"
                "Write only the remaining plan, in one shot, from this point forward."
            )

        task.task_messages = [{"role": "user", "content": current_task_framed_prompt}]
        return task.task_messages

    # ------------------------------------------------------------------ #
    # Generation (think())
    # ------------------------------------------------------------------ #
    def _process_generation_output(
        self, raw_text: str, task: ToolAgentTaskV2,
    ) -> tuple[list[list[BlackboardSlotV2]], list[str], bool, Optional[str]] | str:
        """
        Pure-computation validate callback for the planning retry loop:
        parse, validate references + remaining tool-call budget, and
        compile into batches. Returns the compiled result on success, or a
        feedback string describing every problem found on failure -- a
        ``BlackboardParseError`` from parsing is converted here, not
        propagated, so the retry loop can inject it as corrective feedback.
        """
        print("[DEBUG] Raw generation output:\n", raw_text)
        try:
            flat_slots, annotations, continue_planning, continuation_note = parse_generation(raw_text)
        except BlackboardParseError as e:
            return str(e)

        known_tools = frozenset(self._toolbox.keys())
        known_constants = frozenset(f"K_{key}" for key in self._constants.keys())
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
        issues = validate_references(
            flat_slots, known_tools, known_constants, known_history, remaining_budget
        )
        if issues:
            return "\n".join(f"{i + 1}. {m}" for i, m in enumerate(issues))

        pending = compile_batches(flat_slots)
        return pending, annotations, continue_planning, continuation_note

    def _run_planning_retry_loop(
        self, *, task: ToolAgentTaskV2,
    ) -> tuple[list[list[BlackboardSlotV2]], list[str], bool, Optional[str]]:
        """
        Render, call the engine, record the attempt, validate/compile via
        ``_process_generation_output``, and retry with injected feedback on
        failure until success or the retry budget (``self._generation_retries``,
        tracked via ``task.retries_used``) is exhausted. ``None`` means
        unlimited -- unlike v1's shared retry loop, which assumes ``int``,
        the budget check here explicitly guards against ``None`` rather
        than comparing directly (which would raise ``TypeError``).
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
                if self._generation_retries is not None and task.retries_used >= self._generation_retries:
                    raise ToolAgentError(
                        f"{type(self).__name__}.{self.name}: generation retry budget "
                        f"exhausted after {task.retries_used + 1} attempt(s). "
                        f"Last feedback: {result}"
                    )
                additional_messages = [
                    {"role": "assistant", "content": raw_output},
                    {"role": "user", "content": (
                        f"Your plan could not be used:\n\n{result}\n\n"
                        "Produce a corrected plan."
                    )},
                ]
                task.retries_used += 1
                continue

            return result

    async def _arun_planning_retry_loop(
        self, *, task: ToolAgentTaskV2,
    ) -> tuple[list[list[BlackboardSlotV2]], list[str], bool, Optional[str]]:
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
                if self._generation_retries is not None and task.retries_used >= self._generation_retries:
                    raise ToolAgentError(
                        f"{type(self).__name__}.{self.name}: generation retry budget "
                        f"exhausted after {task.retries_used + 1} attempt(s). "
                        f"Last feedback: {result}"
                    )
                additional_messages = [
                    {"role": "assistant", "content": raw_output},
                    {"role": "user", "content": (
                        f"Your plan could not be used:\n\n{result}\n\n"
                        "Produce a corrected plan."
                    )},
                ]
                task.retries_used += 1
                continue

            return result

    def _check_planning_rounds_budget(self, task: ToolAgentTaskV2) -> None:
        """Shared by ``think``/``async_think``: raises once the continuation-
        round budget is exhausted, else increments it. Only ever called for
        a genuine continuation (``task.completed`` non-empty) -- the
        unconditional first generation never consumes this budget."""
        if (
            self._planning_rounds_limit is not None
            and task.planning_rounds_used >= self._planning_rounds_limit
        ):
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: planning round budget "
                f"exhausted after {task.planning_rounds_used} continuation "
                "round(s)."
            )
        task.planning_rounds_used += 1

    def think(self, task: ToolAgentTaskV2) -> ToolAgentTaskV2:
        """
        Generate, validate, and compile the next segment of the plan --
        either the unconditional first generation, or (once a prior round
        set ``continue_planning``) a fresh continuation. No-op whenever
        there is still pending work to drain, or the task is already fully
        complete -- a checkpoint/if-cutoff/failure-triggered continuation is
        requested by re-entering this same hook, not a separate mechanism.
        """
        if task.pending or task.complete:
            return task

        if task.completed:
            self._check_planning_rounds_budget(task)

        pending, annotations, continue_planning, continuation_note = self._run_planning_retry_loop(task=task)
        task.pending = pending
        task.annotations.extend(annotations)
        task.continue_planning = continue_planning
        task.continuation_note = continuation_note
        task.task_messages.clear()
        return task

    async def async_think(self, task: ToolAgentTaskV2) -> ToolAgentTaskV2:
        """Async mirror of ``think``, using ``_arun_planning_retry_loop``."""
        if task.pending or task.complete:
            return task

        if task.completed:
            self._check_planning_rounds_budget(task)

        pending, annotations, continue_planning, continuation_note = await self._arun_planning_retry_loop(task=task)
        task.pending = pending
        task.annotations.extend(annotations)
        task.continue_planning = continue_planning
        task.continuation_note = continuation_note
        task.task_messages.clear()
        return task

    # ------------------------------------------------------------------ #
    # Prepare next batch
    # ------------------------------------------------------------------ #
    def prepare(self, task: ToolAgentTaskV2) -> ToolAgentTaskV2:
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
            if task.generated_response is NO_VAL:
                task.generated_response = None
            task.complete = True
            return task

        batch = task.pending[0]
        resolved: list[dict[str, Any]] = []
        issues: list[str] = []
        for slot in batch:
            try:
                resolved.append(resolve_slot_args(slot.args, task.cache))
            except Exception as e:
                label = slot.identifier if slot.identifier is not None else "(unassigned)"
                issues.append(f"{label}: {e!r}")

        if issues:
            task.pending.clear()
            task.resolved_args = []
            task.continue_planning = True
            task.continuation_note = "\n".join(f"{i + 1}. {m}" for i, m in enumerate(issues))
            return task

        task.resolved_args = resolved
        return task

    async def async_prepare(self, task: ToolAgentTaskV2) -> ToolAgentTaskV2:
        """Direct passthrough to ``prepare`` -- no I/O of its own."""
        return self.prepare(task)

    # ------------------------------------------------------------------ #
    # Execute prepared batch
    # ------------------------------------------------------------------ #
    async def _gather_batch_results(
        self, batch: list[BlackboardSlotV2], resolved: list[dict[str, Any]],
    ) -> list[Any]:
        """
        Dispatch every real tool-call slot in ``batch`` concurrently;
        ``rhs_assign``/``return`` slots need no dispatch, their result is
        already the resolved ``"val"`` value. Shared by ``act``/
        ``async_act`` -- both differ only in how the resulting coroutine is
        driven.
        """
        coros: list[Any] = []
        dispatch_map: dict[int, int] = {}
        for i, slot in enumerate(batch):
            if slot.tool not in (RHS_ASSIGN_ALIAS, RETURN_ALIAS):
                dispatch_map[i] = len(coros)
                coros.append(self.get_tool(slot.tool).async_invoke(resolved[i]))

        gathered = await asyncio.gather(*coros, return_exceptions=True) if coros else []
        return [
            gathered[dispatch_map[i]] if i in dispatch_map else resolved[i]["val"]
            for i in range(len(batch))
        ]

    def _apply_batch_results(
        self,
        task: ToolAgentTaskV2,
        batch: list[BlackboardSlotV2],
        resolved: list[dict[str, Any]],
        raw_results: list[Any],
    ) -> ToolAgentTaskV2:
        """
        Shared post-gather bookkeeping for ``act``/``async_act``: apply
        results, update completed/cache, handle a terminal ``return`` slot,
        pop the consumed batch -- or, if any real call in this batch
        failed, record whichever succeeded, abandon the rest of this round,
        and require a continuation instead of raising.

        Every real call in ``batch`` was actually dispatched via
        ``asyncio.gather`` regardless of whether any of them failed, so all
        of them count against ``tool_calls_used`` unconditionally, before
        checking for failures.
        """
        real_call_count = sum(
            1 for slot in batch if slot.tool not in (RHS_ASSIGN_ALIAS, RETURN_ALIAS)
        )
        task.tool_calls_used += real_call_count

        failures = [
            f"{batch[idx].tool!r} (identifier={batch[idx].identifier!r}): {raw!r}"
            for idx, raw in enumerate(raw_results)
            if isinstance(raw, BaseException)
        ]

        for slot, value in zip(batch, raw_results):
            if isinstance(value, BaseException):
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
            task.pending.clear()
            task.resolved_args = []
            task.continue_planning = True
            task.continuation_note = "\n".join(f"{i + 1}. {m}" for i, m in enumerate(failures))
            return task

        for slot, kwargs in zip(batch, resolved):
            if slot.tool == RETURN_ALIAS:
                task.generated_response = kwargs["val"]
                task.complete = True

        task.pending.pop(0)
        task.resolved_args = []

        return task

    def act(self, task: ToolAgentTaskV2) -> ToolAgentTaskV2:
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

    async def async_act(self, task: ToolAgentTaskV2) -> ToolAgentTaskV2:
        """Async mirror of ``act``; awaits ``_gather_batch_results``
        directly rather than ``run_coro_sync``-wrapping it."""
        if not task.resolved_args:
            return task

        batch = task.pending[0]
        resolved = task.resolved_args
        raw_results = await self._gather_batch_results(batch, resolved)
        return self._apply_batch_results(task, batch, resolved, raw_results)
