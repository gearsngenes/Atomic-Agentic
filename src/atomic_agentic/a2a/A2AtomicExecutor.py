from __future__ import annotations

from typing import Any, Mapping, Sequence

from a2a.helpers import get_data_parts, new_data_part, new_task_from_user_message, new_text_part
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import AgentCapabilities, AgentCard, AgentExtension, AgentInterface, AgentSkill

from ..constants.a2a_sdk import ATOMIC_RESULT_KEY, PARAM_SCHEMA_EXT_URI, SKILL_ROUTING_KEY, VALID_TRANSPORT_MODES
from ..constants.core import IDENTIFIER_PATTERN
from ..core.Invokable import AtomicInvokable
from ..models.a2a_sdk import A2AtomicSkillMetadata

__all__ = ["A2AtomicExecutor"]


class A2AtomicExecutor(AgentExecutor):
    """
    Server-side a2a-sdk AgentExecutor dispatching to a fixed registry of
    registered AtomicInvokables, routed via Message.metadata[SKILL_ROUTING_KEY].

    Fixed at construction -- no register()/remove(). A running server's own
    blocking execution model leaves no live call site to mutate the registry
    from once serving starts, so post-construction mutation would only add
    complexity (agent-card invalidation, concurrent-access guards) with no
    reachable caller ever benefiting from it.

    Deliberately routed-calls-only: a message arriving without a valid
    SKILL_ROUTING_KEY entry fails the task the same way an unknown skill id
    does. There is no natural-language/generic fallback here -- reachability
    for a foreign, non-AA-aware caller is a registration-time choice (wrap
    the capability in an Agent/ToolAgent instead of a bare Tool), not new
    executor logic.
    """

    def __init__(self, invokables: list[AtomicInvokable] | Mapping[str, AtomicInvokable]) -> None:
        self._invokables: dict[str, AtomicInvokable] = self._normalize_invokables(invokables)
        # Single source of truth for both to_agent_card()'s AgentSkill
        # fields and its extension payload -- built once, since the
        # registry itself never changes post-construction.
        self._skill_metadata: dict[str, A2AtomicSkillMetadata] = {
            remote_name: self._build_skill_metadata(remote_name, invokable)
            for remote_name, invokable in self._invokables.items()
        }

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalize_invokables(
        invokables: list[AtomicInvokable] | Mapping[str, AtomicInvokable],
    ) -> dict[str, AtomicInvokable]:
        """
        Direct mirror of PyA2AtomicHost._normalize_invokables, minus any
        reserved-remote-name check -- no reserved names exist for this
        executor (skill discovery happens via the AgentCard/extension, not a
        callable-dispatch namespace the way python_a2a's list_invokables/
        get_invokable_metadata function names are).
        """
        registry: dict[str, AtomicInvokable] = {}

        if isinstance(invokables, Mapping):
            for remote_name, invokable in invokables.items():
                if not isinstance(invokable, AtomicInvokable):
                    raise TypeError(
                        f"invokables[{remote_name!r}] must be an AtomicInvokable, "
                        f"got {type(invokable).__name__!r}."
                    )
                if not isinstance(remote_name, str) or not IDENTIFIER_PATTERN.fullmatch(remote_name):
                    raise ValueError(f"remote_name {remote_name!r} is not a valid identifier.")
                registry[remote_name] = invokable
            return registry

        if isinstance(invokables, list):
            for index, invokable in enumerate(invokables):
                if not isinstance(invokable, AtomicInvokable):
                    raise TypeError(
                        f"invokables[{index}] must be an AtomicInvokable, "
                        f"got {type(invokable).__name__!r}."
                    )
                key = invokable.name
                if key in registry:
                    raise ValueError(f"Duplicate invokable name {key!r} is not allowed.")
                registry[key] = invokable
            return registry

        raise TypeError("invokables must be a list[AtomicInvokable] or Mapping[str, AtomicInvokable].")

    @staticmethod
    def _build_skill_metadata(remote_name: str, invokable: AtomicInvokable) -> A2AtomicSkillMetadata:
        """
        The one place AtomicInvokable-awareness meets A2AtomicSkillMetadata --
        raw `_description` (not the rendered getter) and `_extra_description()`
        called directly, matching PyA2AtomicHost's own metadata-payload
        precedent for avoiding bullet-duplication on the receiving end.
        """
        return A2AtomicSkillMetadata(
            remote_name=remote_name,
            description=invokable._description,
            extra_description=invokable._extra_description(),
            params=tuple(invokable.parameters),
            return_type=invokable.return_type,
        )

    # ------------------------------------------------------------------ #
    # Card production
    # ------------------------------------------------------------------ #
    def to_agent_card(
        self,
        base_url: str,
        *,
        name: str,
        description: str,
        version: str = "1.0.0",
        transport_modes: Sequence[str] = VALID_TRANSPORT_MODES,
    ) -> AgentCard:
        """
        Builds a fresh AgentCard on every call -- name/description/version/
        transport_modes are call-time, not fixed executor identity, so
        caching by base_url alone would risk serving a stale card. Building
        one is pure local protobuf construction (no I/O; self._skill_metadata
        is already resolved), so there's nothing expensive to cache anyway.
        """
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("base_url must be a non-empty string.")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be a non-empty string.")
        if not isinstance(description, str) or not description.strip():
            raise ValueError("description must be a non-empty string.")
        if not isinstance(version, str) or not version.strip():
            raise ValueError("version must be a non-empty string.")
        if not transport_modes or any(mode not in VALID_TRANSPORT_MODES for mode in transport_modes):
            raise ValueError(
                f"transport_modes must be a non-empty sequence drawn from {VALID_TRANSPORT_MODES!r}; "
                f"got {transport_modes!r}."
            )

        skills = [
            AgentSkill(
                id=remote_name,
                name=remote_name,
                description=metadata.description or "",
            )
            for remote_name, metadata in self._skill_metadata.items()
        ]

        interfaces = [
            AgentInterface(url=base_url.strip(), protocol_binding=mode)
            for mode in transport_modes
        ]

        # required=False so a non-AA client that doesn't understand this
        # extension can still interact with the card's other advertised
        # surface, degrading gracefully rather than being refused outright.
        extension = AgentExtension(
            uri=PARAM_SCHEMA_EXT_URI,
            description="Atomic-Agentic per-skill parameter schema.",
            required=False,
            params={remote_name: metadata.to_dict() for remote_name, metadata in self._skill_metadata.items()},
        )

        return AgentCard(
            name=name.strip(),
            description=description.strip(),
            version=version.strip(),
            supported_interfaces=interfaces,
            capabilities=AgentCapabilities(extensions=[extension]),
            skills=skills,
        )

    # ------------------------------------------------------------------ #
    # AgentExecutor interface
    # ------------------------------------------------------------------ #
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        # A Task must exist before any TaskStatusUpdateEvent/
        # TaskArtifactUpdateEvent can be published -- confirmed against the
        # a2a-sdk's own EventConsumer._handle_task_modification_event, which
        # raises InvalidAgentResponseError for a TaskStatusUpdateEvent
        # arriving before any Task was created. Every response path below
        # goes through TaskUpdater (even the earliest failure checks), so
        # this has to run first, unconditionally.
        task = new_task_from_user_message(context.message)
        await event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue, task.id, task.context_id)

        remote_name = context.metadata.get(SKILL_ROUTING_KEY)
        if not isinstance(remote_name, str) or not remote_name or remote_name not in self._invokables:
            await updater.failed(
                updater.new_agent_message(
                    [new_text_part(f"Unknown or missing {SKILL_ROUTING_KEY!r} metadata: {remote_name!r}.")]
                )
            )
            return

        data_parts = get_data_parts(context.message.parts)
        if not data_parts:
            await updater.failed(
                updater.new_agent_message([new_text_part("No DataPart inputs supplied.")])
            )
            return

        inputs = data_parts[0]
        invokable = self._invokables[remote_name]

        # Must be async_invoke, not a sync call -- that's what gives
        # cancel() below a real await point to interrupt.
        try:
            result = (await invokable.async_invoke(inputs)).result
        except Exception as exc:
            await updater.failed(
                updater.new_agent_message([new_text_part(f"{type(exc).__name__}: {exc}")])
            )
            return

        response_part = new_data_part({ATOMIC_RESULT_KEY: result})
        await updater.add_artifact([response_part])
        await updater.complete()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """
        Publishes TASK_STATE_CANCELED. The actual interrupt already happened
        via the framework's own asyncio.Task.cancel() on the producer task
        running execute() -- this method's only job is the status
        publication the framework doesn't emit automatically. Effectiveness
        of that underlying interrupt depends on whether the invoked engine
        call is natively async (real abort) or still asyncio.to_thread-backed
        (abandons the wait only -- the background thread runs to completion
        regardless).
        """
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        await updater.cancel()

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "invokable_names": list(self._invokables.keys()),
            "invokable_count": len(self._invokables),
        }
