from __future__ import annotations

from typing import Any, Callable, Literal, Optional

import logging

from ..exceptions import AgentInvocationError
from ..llm.base import LLMEngine
from ..models.agents.records import LLMRecord
from ..models.agents.prompts import PromptConfig
from ..models.agents.tasks import AgentTask
from ..utils.agents import normalize_role_prompt
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
    """

    DEFAULT_ROLE_PROMPT = "You are a helpful AI assistant"

    def __init__(
        self,
        name: str,
        namespace: str,
        description: str,
        llm_engine: LLMEngine,
        role_prompt: str | PromptConfig | None = None,
        filter_extraneous_inputs: Optional[bool] = None,
        context_enabled: bool = True,
        *,
        pre_invoke: Optional[Callable | Any] = None,
        post_invoke: Optional[Callable | Any] = None,
        post_result_key: Optional[str] = None,
        records_window: Optional[int] = None,
        response_preview_limit: Optional[int] = None,
        assistant_response_source: Literal["raw", "final"] = "raw",
    ) -> None:
        # 1. Normalize the role prompt and discover its placeholders.
        config = normalize_role_prompt(role_prompt, self.DEFAULT_ROLE_PROMPT)
        role_params = list(config.parameters)

        # 2. Delegate to Agent base; role placeholders are the sole
        # extra_parameters source.
        super().__init__(
            name=name,
            namespace=namespace,
            description=description,
            llm_engine=llm_engine,
            filter_extraneous_inputs=filter_extraneous_inputs,
            context_enabled=context_enabled,
            pre_invoke=pre_invoke,
            post_invoke=post_invoke,
            post_result_key=post_result_key,
            extra_parameters=role_params,
            records_window=records_window,
            response_preview_limit=response_preview_limit,
            assistant_response_source=assistant_response_source,
        )

        # 3. Register the role prompt.
        self._system_prompts["role"] = config

    # ------------------------------------------------------------------ #
    # Role prompt API
    # ------------------------------------------------------------------ #
    @property
    def role_prompt(self) -> str:
        """Role prompt template string."""
        return self._system_prompts["role"].template

    # ------------------------------------------------------------------ #
    # Task-lifecycle hooks
    # ------------------------------------------------------------------ #
    # No _initialize_task/_async_initialize_task override: Agent's concrete
    # base implementation already builds exactly the bare AgentTask
    # BasicAgent needs.
    def _progress(self, task: AgentTask) -> AgentTask:
        """Advance (and complete) a single-turn BasicAgent invocation.

        ``BasicAgent`` always finishes in exactly one ``_progress`` call:
        render the role prompt, build messages, call the engine, validate
        the response, and mark the task complete.
        """
        system = self._system_prompts["role"].render(task.inputs)
        messages = self.build_messages(system, task.turns, task.user_prompt)
        engine_result = self._llm_engine.invoke({"messages": messages})
        text = engine_result.result
        if not isinstance(text, str):
            raise AgentInvocationError(
                f"LLM engine returned non-string result (type={type(text).__name__})."
            )
        llm_record = LLMRecord(
            messages=(messages[-1],),
            llm_result=engine_result,
            system_prompt_name="role",
        )
        task.llm_records.append(llm_record)
        task.generated_response = text
        task.complete = True
        return task

    async def _async_progress(self, task: AgentTask) -> AgentTask:
        """Async mirror of ``_progress``.

        Awaits the engine's native async path directly rather than relying
        on ``Agent``'s inherited ``asyncio.to_thread``-wrapping default.
        """
        system = self._system_prompts["role"].render(task.inputs)
        messages = self.build_messages(system, task.turns, task.user_prompt)
        engine_result = await self._llm_engine.async_invoke({"messages": messages})
        text = engine_result.result
        if not isinstance(text, str):
            raise AgentInvocationError(
                f"LLM engine returned non-string result (type={type(text).__name__})."
            )
        llm_record = LLMRecord(
            messages=(messages[-1],),
            llm_result=engine_result,
            system_prompt_name="role",
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
        return d
