from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Callable, Mapping, Optional


from ..constants.agents import DEFAULT_INSTRUCTIONS
from ..core.atomic_invokable import AtomicInvokable
from ..exceptions import AgentError
from ..llm.base import LLMEngine
from ..models.agents.records import LLMRecord
from ..models.agents.tasks import AgentTask
from ..tools import to_invokable
from .base import Agent


class BasicAgent(Agent):
    """
    Concrete single-turn LLM agent with an instructions slot.

    ``BasicAgent`` is the simplest ``Agent`` subclass: one instructions slot,
    one LLM call per invocation. The slot is a string, a callable or any
    ``AtomicInvokable`` (an agent included), coerced through ``to_invokable``;
    a string becomes a ``PromptTemplate`` and its ``{placeholder}``s are
    auto-discovered. The slot's parameters join the agent's flat parameter
    schema through the parameter-source hook -- callers supply them as
    ordinary top-level ``inputs`` keys, exactly like ``pre_invoke``/
    ``post_invoke`` parameters. Recursion (an agent as the slot) is allowed and
    unguarded.

    The slot is rendered once per invoke, as the last step of ``pre_process``,
    into ``task.instructions`` (it is invoked with only its own parameters);
    the system message is that text, unframed. The slot is fixed at
    construction -- there is no mutation API for it, matching the
    fixed-topology invariant every other ``AtomicInvokable`` in this codebase
    follows.

    The single user message each call sends is a task block: the current
    prompt wrapped in the task markers, preceded by a runtime-context block
    when the call's context has any variable (see ``_render_task_block``).
    History stays raw. When a ``response_schema`` is set, the agent's default
    conversations use render mode ``"j"`` so structured replies replay in
    history as JSON, not as a Python repr.
    """

    _TASK_CLS = AgentTask

    def __init__(
        self,
        name: str,
        description: str,
        llm_engine: LLMEngine,
        instructions: str | Callable[..., Any] | AtomicInvokable = DEFAULT_INSTRUCTIONS,
        pre_invoke: Optional[Callable | Any] = None,
        post_invoke: Optional[Callable | Any] = None,
        inputs_to_context: Iterable[str] | Mapping[str, str] = (),
        response_to_context: str | Iterable[str] | Mapping[Any, str] = (),
        response_to_post: str | Iterable[str] | Mapping[Any, str] = (),
        use_runtime_context: bool = False,
        record_enabled: bool = True,
        records_window: Optional[int] = None,
        *,
        response_schema: dict[str, Any] | None = None,
    ) -> None:
        # 1. Coerce the instructions slot. Set before the base init because
        # _extra_parameter_sources reads it there.
        self._instructions = to_invokable(
            instructions,
            name=f"{name}_instructions",
            description=f"Instructions of agent {name}",
        )

        # 2. Validate and store response_schema -- no deeper validation
        # (no JSON-Schema key checks), matching LLMEngine's own
        # output_structure contract. Stored before the base init because
        # Agent.__init__ builds the default conversation through
        # _new_conversation(), which reads it.
        if response_schema is not None and not isinstance(response_schema, Mapping):
            raise AgentError(
                f"{type(self).__name__}.response_schema must be a dict/Mapping "
                f"or None, got {type(response_schema).__name__}."
            )
        self._response_schema = response_schema

        # 3. Delegate to Agent base.
        super().__init__(
            name=name,
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
        )

    # ------------------------------------------------------------------ #
    # Instructions slot API
    # ------------------------------------------------------------------ #
    @property
    def instructions(self) -> AtomicInvokable:
        """The instructions slot invokable (a ``PromptTemplate`` for a ``str``)."""
        return self._instructions

    @property
    def response_schema(self) -> dict[str, Any] | None:
        """Structured-output schema applied to this agent's replies;
        ``None`` requests plain text. Frozen at construction -- no setter."""
        return self._response_schema

    def _extra_parameter_sources(self) -> list[AtomicInvokable]:
        """The instructions slot joins the agent's signature."""
        return [*super()._extra_parameter_sources(), self._instructions]

    # ------------------------------------------------------------------ #
    # Conversation defaults
    # ------------------------------------------------------------------ #
    def _new_conversation(self):
        """Build one fresh default conversation for this agent.

        1. no ``response_schema``: the base default.
        2. otherwise ``Conversation(render_mode="j")``, so structured
           (dict/list) replies replay in history as JSON, not as a Python
           repr.

        Applies to every default conversation this agent builds, including
        those after ``clear_memory()`` and ``remove_conversation("default")``.
        A conversation built by hand and imported keeps its own settings.
        """
        new_convo = super()._new_conversation()
        if self._response_schema is not None:
            new_convo.render_mode = "j"
        return new_convo

    # ------------------------------------------------------------------ #
    # Task-lifecycle hooks
    # ------------------------------------------------------------------ #
    def _render_instructions(self, task: AgentTask) -> None:
        """Render the instructions slot into ``task.instructions``.

        1. Take only the slot's own parameter names from ``task.agentic_inputs``.
        2. Invoke the slot once; errors from it propagate unwrapped.
        3. A non-``str`` result raises ``TypeError`` naming the slot.
        """
        slot = self._instructions
        inputs = {
            p.name: task.agentic_inputs[p.name]
            for p in slot.parameters if p.name in task.agentic_inputs
        }
        text = slot.invoke(inputs).result
        if not isinstance(text, str):
            raise TypeError(
                f"{self.full_name}: instructions ({slot.name!r}) returned "
                f"{type(text).__name__}; instructions must render to a str."
            )
        task.instructions = text

    async def _async_render_instructions(self, task: AgentTask) -> None:
        """Async twin of ``_render_instructions``; awaits ``slot.async_invoke``
        so an agent used as the slot never has its sync ``invoke`` called."""
        slot = self._instructions
        inputs = {
            p.name: task.agentic_inputs[p.name]
            for p in slot.parameters if p.name in task.agentic_inputs
        }
        text = (await slot.async_invoke(inputs)).result
        if not isinstance(text, str):
            raise TypeError(
                f"{self.full_name}: instructions ({slot.name!r}) returned "
                f"{type(text).__name__}; instructions must render to a str."
            )
        task.instructions = text

    def _render_system_message(self, task: AgentTask) -> list[dict[str, str]]:
        """The rendered instructions as the single, unframed system message.
        A synchronous read of ``task.instructions``; valid after ``pre_process``."""
        return [{"role": "system", "content": task.instructions}]

    def _render_task_messages(self, task: AgentTask) -> list[dict[str, str]]:
        """Build this invocation's single task message.

        1. if ``task.task_messages`` is non-empty, return it (build-once
           contract; ``BasicAgent`` never retries, so this holds once set).
        2. otherwise store one user message holding ``_render_task_block``.
        3. return ``task.task_messages``.

        The message is the wrapped prompt, preceded by the context block when
        ``task.context`` is non-empty. ``task.user_prompt`` is not modified;
        the sent message is what the ``LLMRecord`` stores.
        """
        if not task.task_messages:
            task.task_messages = [
                {"role": "user", "content": self._render_task_block(task)}
            ]
        return task.task_messages

    def act(self, task: AgentTask) -> AgentTask:
        """Advance (and complete) a single-turn BasicAgent invocation.

        ``BasicAgent`` always finishes in exactly one ``act`` call: render
        the task, call the engine, validate the response, and mark the
        task complete.
        """
        messages = self.render_task(task)
        engine_result = self._llm_engine.invoke({
            "messages": messages,
            "output_structure": self._response_schema,
        })
        text = engine_result.result
        llm_record = LLMRecord(
            messages=list(task.task_messages),
            llm_result=engine_result,
        )
        task.llm_records.append(llm_record)
        task.generated_response = text
        task.complete = True
        return task

    async def async_act(self, task: AgentTask) -> AgentTask:
        """Async mirror of ``act``, awaiting the engine's native async path."""
        messages = self.render_task(task)
        engine_result = await self._llm_engine.async_invoke({
            "messages": messages,
            "output_structure": self._response_schema,
        })
        text = engine_result.result
        llm_record = LLMRecord(
            messages=list(task.task_messages),
            llm_result=engine_result,
        )
        task.llm_records.append(llm_record)
        task.generated_response = text
        task.complete = True
        return task

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict:
        """Return a diagnostic snapshot including the instructions slot."""
        d = super().to_dict()
        d["instructions"] = self._instructions.to_dict()
        d["response_schema"] = self.response_schema
        return d
