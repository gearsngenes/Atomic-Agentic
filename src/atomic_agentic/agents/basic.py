from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Callable, Mapping, Optional

import logging

from ..exceptions import AgentError
from ..llm.base import LLMEngine
from ..models.agents.records import LLMRecord
from ..models.agents.prompts import PromptConfig
from ..models.agents.tasks import AgentTask
from ..models.parameters import ParamSpec
from ..utils.agents import normalize_prompt_config
from .base import Agent

logger = logging.getLogger(__name__)


class BasicAgent(Agent):
    """
    Concrete single-turn LLM agent with a role prompt.

    ``BasicAgent`` is the simplest ``Agent`` subclass: one role prompt, one
    LLM call per invocation. The ``role_prompt`` may contain ``{placeholder}``
    slots whose names are auto-discovered and wired into the agent's flat
    parameter schema as ``extra_parameters`` -- callers supply them as
    ordinary top-level ``inputs`` keys, exactly like ``pre_invoke``/
    ``post_invoke`` parameters.

    The role prompt is fixed at construction. There is no supported way to
    change its text or placeholders afterward -- ``BasicAgent`` declares no
    mutation API for it, matching the fixed-topology invariant every other
    ``AtomicInvokable`` in this codebase already follows (parameters are
    resolved once at construction and are immutable thereafter).

    The single user message each call sends is a task block: the current
    prompt wrapped in the task markers, preceded by a runtime-context block
    when the call's context has any variable (see ``_render_task_block``).
    History stays raw. When a ``response_schema`` is set, the agent's default
    conversations use render mode ``"j"`` so structured replies replay in
    history as JSON, not as a Python repr.
    """

    DEFAULT_ROLE_PROMPT = "You are a helpful AI assistant"

    _TASK_CLS = AgentTask
    _INITIAL_PROMPT_NAME = "role"

    def __init__(
        self,
        name: str,
        namespace: str,
        description: str,
        llm_engine: LLMEngine,
        role_prompt: str | PromptConfig | None = None,
        record_enabled: bool = True,
        *,
        pre_invoke: Optional[Callable | Any] = None,
        post_invoke: Optional[Callable | Any] = None,
        inputs_to_context: Optional[Iterable[str] | Mapping[str, str]] = None,
        response_to_context: Optional[str | Iterable[str] | Mapping[Any, str]] = None,
        response_to_post: Optional[str | Iterable[str] | Mapping[Any, str]] = None,
        records_window: Optional[int] = None,

        response_schema: dict[str, Any] | None = None,
    ) -> None:
        # 1. Normalize the role prompt and discover its placeholders.
        config = normalize_prompt_config(
            role_prompt,
            default_template=self.DEFAULT_ROLE_PROMPT,
            provided_description="Role prompt",
            default_description="Default assistant role prompt",
            error_label="role_prompt",
        )
        role_params = self._merge_prompt_parameters(list(config.parameters))

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

        # 3. Delegate to Agent base; role placeholders are the sole
        # extra_parameters source.
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
            extra_parameters=role_params,
            records_window=records_window,

        )

        # 4. Register the role prompt.
        self._system_prompts["role"] = config

    # ------------------------------------------------------------------ #
    # Role prompt API
    # ------------------------------------------------------------------ #
    @property
    def role_prompt(self) -> str:
        """Role prompt template string."""
        return self._system_prompts["role"].template

    @property
    def response_schema(self) -> dict[str, Any] | None:
        """Structured-output schema applied to this agent's replies;
        ``None`` requests plain text. Frozen at construction -- no setter."""
        return self._response_schema

    def _merge_prompt_parameters(self, role_params: list[ParamSpec]) -> list[ParamSpec]:
        """Seam for a subclass that registers more prompt templates than the
        role prompt and wants their placeholders in the agent's signature.

        1. return ``role_params`` unchanged (``BasicAgent``: identity)

        An override returns the full list of prompt-derived parameters, the
        role prompt's included and first so it keeps priority. It runs inside
        ``__init__`` before ``Agent.__init__``, so it may read only attributes
        the subclass set before calling ``super().__init__``.
        """
        return role_params

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
        those after ``clear_memory()`` and ``remove_conversation("default")``,
        and to ``create_conversation()`` calls that omit ``render_mode``.
        Conversations imported from elsewhere keep their own settings.
        """
        new_convo = super()._new_conversation()
        if self._response_schema is not None:
            new_convo.render_mode = "j"
        return new_convo

    # ------------------------------------------------------------------ #
    # Task-lifecycle hooks
    # ------------------------------------------------------------------ #
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
            system_prompt_name=task.system_prompt_name,
        )
        task.llm_records.append(llm_record)
        task.generated_response = text
        task.complete = True
        return task

    async def async_act(self, task: AgentTask) -> AgentTask:
        """Async mirror of ``act``.

        Awaits the engine's native async path directly rather than relying
        on ``Agent``'s inherited ``asyncio.to_thread``-wrapping default.
        """
        messages = self.render_task(task)
        engine_result = await self._llm_engine.async_invoke({
            "messages": messages,
            "output_structure": self._response_schema,
        })
        text = engine_result.result
        llm_record = LLMRecord(
            messages=list(task.task_messages),
            llm_result=engine_result,
            system_prompt_name=task.system_prompt_name,
        )
        task.llm_records.append(llm_record)
        task.generated_response = text
        task.complete = True
        return task

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict:
        """Return a diagnostic snapshot including the role prompt."""
        d = super().to_dict()
        d["role_prompt"] = self.role_prompt
        d["response_schema"] = self.response_schema
        return d
