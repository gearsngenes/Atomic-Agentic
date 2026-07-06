from __future__ import annotations

from typing import Any, Callable, Dict, Literal, Mapping, Optional, Union

import logging

from ..exceptions import AgentError, AgentInvocationError
from ..engines.LLMEngines import LLMEngine
from ..models.parameters import ParamSpec
from ..models.agents.records import AgentRecord, LLMRecord
from ..models.agents.prompts import PromptConfig
from ..utils.agents import normalize_context_properties, normalize_role_prompt
from .base import Agent

logger = logging.getLogger(__name__)


class BasicAgent(Agent):
    """
    Concrete single-turn LLM agent with a role prompt and optional context placeholders.

    ``BasicAgent`` is the simplest ``Agent`` subclass: one role prompt, one LLM
    call per invocation. The ``role_prompt`` may contain ``{placeholder}`` slots
    whose names are auto-discovered and wired into the agent schema as
    KEYWORD_ONLY ``context_properties`` — callers supply them via a single
    ``context: dict`` param that is forwarded to ``PromptConfig.render`` at invocation time.

    An optional ``extra_context_properties`` constructor param declares additional
    context dict entries beyond what the role prompt auto-discovers. These are held
    separately and unioned with role-derived properties at construction and on every
    mutation. Role-derived and extra names must not overlap.

    ``role_prompt`` may be updated via ``update_prompt("role", config)`` or the
    ``role_prompt`` setter. On update, ``_context_properties`` is rebuilt from the
    new placeholders unioned with ``_extra_context_properties``, and the ``context``
    schema-param description is refreshed. Any key other than ``"role"`` is
    rejected — BasicAgent has no other prompts.
    """

    DEFAULT_ROLE_PROMPT = "You are a helpful AI assistant"

    def __init__(
        self,
        name: str,
        namespace: str,
        description: str,
        llm_engine: LLMEngine,
        role_prompt: str | PromptConfig | None = None,
        extra_context_properties: list[str] | list[ParamSpec] | None = None,
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
        # ① Normalize the role prompt and discover its placeholders.
        config = normalize_role_prompt(role_prompt, self.DEFAULT_ROLE_PROMPT)
        role_params = list(config.parameters)

        # ② Normalize extra context properties; check no name overlap with role.
        extra_params = normalize_context_properties(extra_context_properties)
        role_names = {p.name for p in role_params}
        for p in extra_params:
            if p.name in role_names:
                raise AgentError(
                    f"extra_context_properties name {p.name!r} overlaps with a "
                    "role-prompt placeholder; keep role-derived and extra properties distinct."
                )

        # ③ Delegate to Agent base with the unioned context_properties.
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
            context_properties=role_params + extra_params,
            records_window=records_window,
            response_preview_limit=response_preview_limit,
            assistant_response_source=assistant_response_source,
        )

        # ④ Register the role prompt and store extra props separately.
        self._system_prompts["role"] = config
        self._extra_context_properties: tuple[ParamSpec, ...] = tuple(extra_params)

    # ------------------------------------------------------------------ #
    # Role prompt API
    # ------------------------------------------------------------------ #
    @property
    def role_prompt(self) -> str:
        """Role prompt template string."""
        return self._system_prompts["role"].template

    @role_prompt.setter
    def role_prompt(self, value: str | PromptConfig | None) -> None:
        """Normalize ``value`` and delegate to ``update_prompt('role', ...)``."""
        config = normalize_role_prompt(value, self.DEFAULT_ROLE_PROMPT)
        self.update_prompt("role", config)

    def set_context_properties(
        self,
        properties: list[str] | list[ParamSpec] | None,
    ) -> None:
        """Raise — ``BasicAgent`` context properties have two sources.

        Use ``set_extra_context_properties(...)`` to replace the extra portion,
        or ``update_prompt('role', config)`` / ``role_prompt`` setter to swap
        the role-derived portion.
        """
        raise AgentError(
            "Cannot set context properties directly on BasicAgent: "
            "_context_properties is always the union of role-derived and extra params. "
            "Use set_extra_context_properties() or update_prompt('role', ...) instead."
        )

    def update_prompt(self, key: str, config: PromptConfig) -> None:
        """Allow updates to the ``'role'`` prompt only.

        Lifecycle:
        1. Validate key — must be a non-empty string equal to ``'role'``; any
           other key raises ``AgentError`` (BasicAgent has no other prompts).
        2. Validate config type (must be ``PromptConfig``).
        3. Extract new role params from ``config.parameters``; raise if any name
           overlaps with a current ``_extra_context_properties`` name.
        4. Delegate to ``super().update_prompt`` to store the config.
        5. Re-index the union of new role params and unchanged
           ``_extra_context_properties`` via ``_set_context_properties``,
           which stores and refreshes the schema description atomically.
        """
        # ① Key validation.
        if not isinstance(key, str) or not key.strip():
            raise AgentError("update_prompt: key must be a non-empty string.")
        if key.strip() != "role":
            raise AgentError(
                f"BasicAgent has only a 'role' prompt; "
                f"cannot update key {key.strip()!r}."
            )
        # ② Config type validation.
        if not isinstance(config, PromptConfig):
            raise AgentError("update_prompt: config must be a PromptConfig instance.")
        # ③ Overlap check: new role placeholders vs current extra props.
        new_role_params = list(config.parameters)
        extra_names = {p.name for p in self._extra_context_properties}
        for p in new_role_params:
            if p.name in extra_names:
                raise AgentError(
                    f"new role-prompt placeholder {p.name!r} overlaps with an existing "
                    "extra_context_properties name."
                )
        # ④ Store via base (validates config type, writes _system_prompts["role"]).
        super().update_prompt(key, config)
        # ⑤ Re-index union, store, and refresh description.
        self._set_context_properties(new_role_params + list(self._extra_context_properties))

    def set_extra_context_properties(
        self,
        properties: list[str] | list[ParamSpec] | None,
    ) -> None:
        """Replace extra context properties and refresh the schema description.

        Lifecycle:
        1. Normalize ``properties`` via ``normalize_context_properties``.
        2. Raise ``AgentError`` if any new name overlaps with a current
           role-prompt placeholder name.
        3. Store the new ``_extra_context_properties``; re-index the union of
           current role params and the new extras via ``_set_context_properties``,
           which stores and refreshes the schema description atomically.
        """
        # ① Normalize.
        new_extra = normalize_context_properties(properties)
        # ② Overlap check: new extras vs current role placeholders.
        role_names = {p.name for p in self._system_prompts["role"].parameters}
        for p in new_extra:
            if p.name in role_names:
                raise AgentError(
                    f"extra_context_properties name {p.name!r} overlaps with a "
                    "role-prompt placeholder; use non-overlapping names."
                )
        # ③ Store new extras; re-index union, store, and refresh.
        self._extra_context_properties = tuple(new_extra)
        role_params = list(self._system_prompts["role"].parameters)
        self._set_context_properties(role_params + new_extra)

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
        """Return a diagnostic snapshot including role and extra context properties."""
        d = super().to_dict()
        d["role_prompt"] = self.role_prompt
        d["extra_context_properties"] = [p.to_dict() for p in self._extra_context_properties]
        return d
