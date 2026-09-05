from __future__ import annotations

from typing import Any, Callable, Literal, Optional

from ..mcp.MCPClientHub import MCPClientHub
from ..a2a.A2AClientHub import A2AClientHub
from ..a2a.PyA2AtomicClient import PyA2AtomicClient

from .base import Agent
from ..core.Invokable import AtomicInvokable
from ..llm.base import LLMEngine
from ..tools.Toolify import toolify
from ..models.agents.blackboard_models import ConstantSpec
from ..constants.core import IDENTIFIER_PATTERN
from ..exceptions import ToolAgentError, ToolRegistrationError

_HubClient = MCPClientHub | A2AClientHub | PyA2AtomicClient
_COLLISION_POLICIES = ("raise", "skip", "replace")


class ToolAgent2(Agent):
    """
    Decomposition-first tool-invoking agent (sibling family to ``ToolAgent``,
    not a subclass).

    This slice implements only the constructor plus tool/constant
    registration and prompt-context rendering. Tools and constants live in
    two independent keyspaces: the toolbox (``dict[str, AtomicInvokable]``,
    keyed by an alias when supplied, else the tool's own ``full_name``) and
    the constants list (``list[ConstantSpec]``, each stored under a
    mandatory ``f"K_{alias}"`` name). No blackboard, checklist/subtask
    model, or ``think``/``prepare``/``act`` override exists yet — those are
    later passes (`.claude/context/04-current-task.md` §0).
    """

    def __init__(
        self,
        name: str,
        namespace: str,
        description: str,
        llm_engine: LLMEngine,
        context_enabled: bool = False,
        *,
        mode: Literal["one_shot", "reactive"] = "one_shot",
        generation_retries: Optional[int] = 0,
        tool_calls_limit: Optional[int] = None,
        response_preview_limit: Optional[int] = None,
        pre_invoke: Optional[AtomicInvokable | Callable[..., Any]] = None,
        post_invoke: Optional[AtomicInvokable | Callable[..., Any]] = None,
        post_result_key: Optional[str] = None,
        records_window: Optional[int] = None,
        assistant_response_source: Literal["raw", "final"] = "raw",
    ) -> None:
        """
        Validate/store ``mode``, ``generation_retries``, ``tool_calls_limit``,
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

        if mode not in ("one_shot", "reactive"):
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: mode must be 'one_shot' or "
                f"'reactive'; got {mode!r}."
            )
        self._mode = mode

        if generation_retries is not None and (
            type(generation_retries) is not int or generation_retries < 0
        ):
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: generation_retries must be "
                f"None or an int >= 0; got {generation_retries!r}."
            )
        self._generation_retries = generation_retries

        self._tool_calls_limit: Optional[int] = None
        self.tool_calls_limit = tool_calls_limit

        self._toolbox: dict[str, AtomicInvokable] = {}
        self._constants: dict[str, ConstantSpec] = {}

    # ------------------------------------------------------------------ #
    # Construction-time / mutable knobs
    # ------------------------------------------------------------------ #
    @property
    def mode(self) -> Literal["one_shot", "reactive"]:
        """Construction-time decomposition cadence. Fixed-topology; read-only."""
        return self._mode

    @property
    def generation_retries(self) -> Optional[int]:
        """Bounded-attempts ceiling shared by structural generation retries and
        (later pass) failure-triggered checklist regeneration. ``None`` means
        unlimited. Read-only."""
        return self._generation_retries

    @property
    def tool_calls_limit(self) -> Optional[int]:
        """Max allowed tool calls per ``invoke()`` run. ``None`` means unlimited."""
        return self._tool_calls_limit

    @tool_calls_limit.setter
    def tool_calls_limit(self, value: Optional[int]) -> None:
        if value is None:
            self._tool_calls_limit = None
            return
        if type(value) is not int or value < 0:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: tool_calls_limit must be "
                f"None or an int >= 0; got {value!r}."
            )
        self._tool_calls_limit = value

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
    def actions_context(self, tools_subset: Optional[list[str]]) -> str:
        """
        Render the given toolbox subset for prompt injection, one block per
        tool (signature line + indented description), joined by ``"\\n---\\n"``.
        When ``tid`` is an alias (not the tool's own ``full_name``), only the
        leading identity token of ``tool.signature`` is swapped for the alias
        — the ``(args) -> ReturnType`` portion renders unchanged.
        """
        blocks: list[str] = []
        resolved_subset = tools_subset if tools_subset is not None else list(self._toolbox.keys())
        for tid in resolved_subset:
            tool = self.get_tool(tid)
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

    def constants_context(self, constants_subset: Optional[list[str]]) -> str:
        """
        Render the given constants subset for prompt injection. Names print
        exactly as stored (``K_``-prefixed) — no prefix synthesized here.
        Empty subset renders the same "no constants" message as an empty
        registry.
        """
        resolved_subset = constants_subset if constants_subset is not None else list(self._constants.keys())
        constants = [self.get_constant(name) for name in resolved_subset]
        if not constants:
            return "No constants registered."

        rendered: list[str] = []
        for spec in constants:
            description = (
                spec.description if spec.description is not None else "No description provided."
            )
            rendered.append(
                f"- {spec.name}\n"
                f"  Type: {spec.type}\n"
                f"  Description: {description}"
            )

        return "\n\n".join(rendered)
