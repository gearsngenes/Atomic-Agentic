from __future__ import annotations

from typing import Any, Callable, Dict, Literal, Mapping, Optional, Union

import logging

from ..exceptions import AgentError, AgentInvocationError
from ..engines.LLMEngines import LLMEngine
from ..models.agents.records import AgentRecord, LLMRecord
from ..models.agents.prompts import PromptConfig
from .base import Agent

logger = logging.getLogger(__name__)


def _normalize_role_prompt(
    value: str | PromptConfig | None,
    default_template: str,
) -> PromptConfig:
    """Coerce ``role_prompt`` to a ``PromptConfig``."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return PromptConfig(
            template=default_template,
            description="Default assistant role prompt",
        )
    if isinstance(value, str):
        return PromptConfig(template=value.strip(), description="Role prompt")
    if isinstance(value, PromptConfig):
        return value
    raise TypeError(
        f"role_prompt must be str, PromptConfig, or None; got {type(value).__name__}."
    )


class BasicAgent(Agent):
    """
    Concrete single-turn LLM agent with a role prompt and optional context placeholders.

    ``BasicAgent`` is the simplest ``Agent`` subclass: one role prompt, one LLM
    call per invocation. The ``role_prompt`` may contain ``{placeholder}`` slots
    whose names are auto-discovered and wired into the agent schema as
    KEYWORD_ONLY ``context_properties`` — callers supply them via a single
    ``context: dict`` param that is forwarded to ``PromptConfig.render`` at invocation time.

    ``role_prompt`` is immutable after construction. Use ``update_prompt`` to
    manage any additional prompts; writes to the ``"role"`` key are rejected.
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
        config = _normalize_role_prompt(role_prompt, self.DEFAULT_ROLE_PROMPT)
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
            context_properties=list(config.parameters),
            records_window=records_window,
            response_preview_limit=response_preview_limit,
            assistant_response_source=assistant_response_source,
        )
        self._system_prompts["role"] = config

    # ------------------------------------------------------------------ #
    # Role prompt API
    # ------------------------------------------------------------------ #
    @property
    def role_prompt(self) -> str:
        """Role prompt template string. Read-only; construct a new BasicAgent to change it."""
        return self._system_prompts["role"].template

    def update_prompt(self, key: str, config: PromptConfig) -> None:
        """Register or replace a system prompt. Raises AgentError if ``key == "role"``."""
        if isinstance(key, str) and key.strip() == "role":
            raise AgentError(
                "role_prompt is immutable on BasicAgent; "
                "construct a new BasicAgent to change the role prompt."
            )
        super().update_prompt(key, config)

    # ------------------------------------------------------------------ #
    # Core LLM work
    # ------------------------------------------------------------------ #
    def _invoke(
        self,
        turns: list[AgentRecord],
        prompt: str,
        context: dict,
    ) -> tuple[AgentRecord, dict]:
        """Sync single-LLM-call implementation.

        Renders the role prompt with ``context``, builds the message list, calls
        the engine, and returns a draft ``AgentRecord`` plus accounting metadata.
        """
        system = self._system_prompts["role"].render(context)
        messages = self.build_messages(system, turns, prompt)
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
        draft = AgentRecord(
            user_prompt=PromptConfig(template=prompt, description=""),
            generated_response=text,
        )
        return draft, {
            "llm_records": (llm_record,),
            "llm_model_data": engine_result.model_data,
        }

    async def _ainvoke(
        self,
        turns: list[AgentRecord],
        prompt: str,
        context: dict,
    ) -> tuple[AgentRecord, dict]:
        """Async mirror of ``_invoke``."""
        system = self._system_prompts["role"].render(context)
        messages = self.build_messages(system, turns, prompt)
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
        draft = AgentRecord(
            user_prompt=PromptConfig(template=prompt, description=""),
            generated_response=text,
        )
        return draft, {
            "llm_records": (llm_record,),
            "llm_model_data": engine_result.model_data,
        }

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict:
        """Return a diagnostic snapshot including the role_prompt convenience key."""
        d = super().to_dict()
        d["role_prompt"] = self.role_prompt
        return d
