"""
ThinkingAgent: Self-Questioning BasicAgent

This module provides ``ThinkingAgent``, a concrete ``BasicAgent`` subclass
overriding ``think()``/``act()`` alone to add a self-questioning phase
before the reply. Each thinking round is one LLM call producing one
free-form thought (a plain string, or a structured value when
``thinking_schema`` is set); the agent always runs exactly
``thinking_rounds`` rounds -- a reserved per-invocation runtime parameter,
not a construction-time knob -- before replying. There is no early exit.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Callable, Mapping, Optional

from .base import Agent
from .basic import BasicAgent
from ..constants.agents import THINKING_ROUNDS_PARAM
from ..exceptions import AgentError, AgentInvocationError
from ..llm.base import LLMEngine
from ..models.agents.prompts import PromptConfig
from ..models.agents.records import AgentRecord, LLMRecord, ThinkingAgentRecord
from ..models.agents.tasks import ThinkingTask
from ..models.parameters import ParamSpec
from ..models.results.agents import ThinkingAgentResult
from ..utils.agents import normalize_role_prompt, normalize_thinking_instructions
from ..utils.parameters import (
    apply_parameter_reports,
    build_parameter_reports,
    insert_by_category,
)

_THINKING_FRAMING = (
    "Think about this task and produce insights, questions, and "
    "reasoning that clarify or enhance your understanding of it -- this "
    "is preparation for a final response, not the response itself."
)
_THINKING_CONTINUATION_NUDGE = (
    "Continue thinking about this task. Produce additional insights, "
    "questions, or reasoning that further clarify or enhance your "
    "understanding of it."
)


class ThinkingAgent(BasicAgent):
    """
    ``BasicAgent`` subclass adding a fixed-round self-questioning phase.

    Overrides ``think()``/``act()``; ``prepare()`` stays ``BasicAgent``'s
    inherited no-op -- there is no deterministic bookkeeping step between
    a thinking round's decision and either the next round or the reply,
    since the phase switch is knowledge ``think()`` already has the moment
    it happens.

    Exactly two system prompts exist for any instance: ``"role"`` (the
    caller's own role prompt, reply phase only) and ``"thinking"`` (built
    directly from ``thinking_instructions`` each render, thinking phase
    only -- no fixed scaffold, no wrapper).

    Bypasses ``BasicAgent.__init__`` and calls ``Agent.__init__`` directly
    -- ``thinking_instructions`` needs to contribute its own placeholders
    to the schema alongside ``role_prompt``'s, which ``BasicAgent.__init__``'s
    narrower signature has no hook for. ``role_prompt``'s params take
    priority on any compatible-but-not-identical overlap with
    ``thinking_instructions``'s; a true collision raises.
    """

    THINKING_PROMPT_NAME = "thinking"

    DEFAULT_THINKING_PROMPT: str = (
        "You are a thinking assistant that produces various thoughts, "
        "questions, and task-enhancement clarifications to describe or "
        "enhance a user task prompt"
    )

    @classmethod
    def get_reserved_parameters(cls) -> list[ParamSpec]:
        """Prepend ``thinking_rounds`` onto the base ``run_id`` reservation,
        per ``Agent.get_reserved_parameters``'s own documented subclass-
        extension convention -- keeps ``run_id`` sorting last among
        reserved keyword-only parameters in the declared schema."""
        return [THINKING_ROUNDS_PARAM] + super().get_reserved_parameters()

    def __init__(
        self,
        name: str,
        namespace: str,
        description: str,
        llm_engine: LLMEngine,
        role_prompt: str | PromptConfig | None = None,
        thinking_instructions: str | PromptConfig | None = None,
        context_enabled: bool = True,
        *,
        thinking_llm_engine: LLMEngine | None = None,
        pre_invoke: Optional[Callable | Any] = None,
        post_invoke: Optional[Callable | Any] = None,
        post_result_key: Optional[str] = None,
        records_window: Optional[int] = None,
        response_preview_limit: Optional[int] = None,
        response_schema: dict[str, Any] | None = None,
        thinking_schema: dict[str, Any] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        role_prompt : str | PromptConfig | None
            Reply-phase persona prompt. Same contract as ``BasicAgent``'s.
        thinking_instructions : str | PromptConfig | None
            Free-form instructions rendered directly as the thinking
            phase's entire system message -- no imposed structure, no
            wrapper. Defaults to ``DEFAULT_THINKING_PROMPT`` when omitted.
            May declare its own ``{placeholder}``s, reconciled against
            ``role_prompt``'s (role_prompt wins on compatible-but-not-
            identical overlap; true collisions raise).
        thinking_llm_engine : LLMEngine | None
            Optional secondary engine used for thinking-round calls only
            (``think``/``async_think``). ``None`` (default) falls back to
            ``llm_engine`` -- every thinking round then uses the same
            engine as the reply phase, identical to this class's behavior
            before this parameter existed. The reply phase
            (``act``/``async_act``) never consults this value under any
            setting.
        response_schema : dict[str, Any] | None
            Structured-output schema applied to the reply phase only --
            the thinking phase stays free-form regardless. Same contract
            as ``BasicAgent``'s.
        thinking_schema : dict[str, Any] | None
            Structured-output schema applied to thinking-round calls only
            (``think``/``async_think``) -- the reply phase never consults
            this value. ``None`` (default) keeps thinking fully free-form.
            Fully independent of ``response_schema``; setting one has no
            effect on the other. Same validation contract as
            ``response_schema`` (``Mapping``-or-raise, no deeper
            JSON-Schema key checking). When set, a thinking round's stored
            value may be any JSON-decodable type, not just ``str``
            (mirrors ``ThinkingTask.thoughts``'s already-widened type).
            Field-ordering guidance (place reasoning-shaped fields before
            label/decision fields -- production data shows a measurable
            quality hit otherwise) is documentation only, never
            runtime-enforced.

        The number of thinking rounds run per invocation is not a
        construction-time parameter -- it is the reserved runtime
        parameter ``thinking_rounds`` (default ``1``, no ceiling here),
        validated in ``_initialize_task``.
        """
        role_config = normalize_role_prompt(role_prompt, self.DEFAULT_ROLE_PROMPT)
        role_params = list(role_config.parameters)

        thinking_config = normalize_thinking_instructions(thinking_instructions, self.DEFAULT_THINKING_PROMPT)
        thinking_params = list(thinking_config.parameters)

        # Reconcile role_prompt vs thinking_instructions BEFORE combining --
        # role_prompt is the priority source (source index 0), same idiom
        # Agent.__init__ uses for its own pre/post/extra reconciliation.
        reports = build_parameter_reports([role_params, thinking_params])
        constructed = apply_parameter_reports(
            reports, ("role_prompt", "thinking_instructions"), error_cls=AgentError, stacklevel=3
        )
        combined_extra_params = insert_by_category([], constructed)

        Agent.__init__(
            self,
            name=name,
            namespace=namespace,
            description=description,
            llm_engine=llm_engine,
            context_enabled=context_enabled,
            pre_invoke=pre_invoke,
            post_invoke=post_invoke,
            post_result_key=post_result_key,
            extra_parameters=combined_extra_params,
            records_window=records_window,
            response_preview_limit=response_preview_limit,
        )

        self._system_prompts["role"] = role_config
        self._thinking_instructions_config = thinking_config

        # response_schema is not part of Agent.__init__'s param set --
        # BasicAgent.__init__ is bypassed above, so this class stores it
        # itself, matching BasicAgent's own validation exactly.
        if response_schema is not None and not isinstance(response_schema, Mapping):
            raise AgentError(
                f"{type(self).__name__}.response_schema must be a dict/Mapping "
                f"or None, got {type(response_schema).__name__}."
            )
        self._response_schema = response_schema

        if thinking_llm_engine is not None and not isinstance(thinking_llm_engine, LLMEngine):
            raise AgentError(
                f"{type(self).__name__}.thinking_llm_engine must be an LLMEngine "
                f"instance or None, got {type(thinking_llm_engine).__name__}."
            )
        self._thinking_llm_engine = thinking_llm_engine

        if thinking_schema is not None and not isinstance(thinking_schema, Mapping):
            raise AgentError(
                f"{type(self).__name__}.thinking_schema must be a dict/Mapping "
                f"or None, got {type(thinking_schema).__name__}."
            )
        self._thinking_schema = thinking_schema

    # ------------------------------------------------------------------ #
    # Secondary thinking engine
    # ------------------------------------------------------------------ #
    @property
    def thinking_llm_engine(self) -> LLMEngine | None:
        """The secondary engine used for thinking rounds, or ``None`` if
        unset. Returned exactly as stored -- ``None`` means "no override
        configured," distinct from "explicitly set to the same object as
        ``llm_engine``." ``think()``/``async_think()`` resolve the engine
        actually used via ``self._thinking_llm_engine or self._llm_engine``
        internally; this property is the raw, unresolved value."""
        return self._thinking_llm_engine

    @thinking_llm_engine.setter
    def thinking_llm_engine(self, engine: LLMEngine | None) -> None:
        if engine is not None and not isinstance(engine, LLMEngine):
            raise TypeError("thinking_llm_engine must be an LLMEngine instance or None.")
        self._thinking_llm_engine = engine

    @property
    def thinking_schema(self) -> dict[str, Any] | None:
        """Structured-output schema applied to thinking rounds only;
        ``None`` requests free-form text. Frozen at construction -- no
        setter. Mirrors ``BasicAgent.response_schema``'s exact shape."""
        return self._thinking_schema

    # ------------------------------------------------------------------ #
    # Task-lifecycle hooks
    # ------------------------------------------------------------------ #
    def _initialize_task(
        self,
        *,
        turns: list[AgentRecord],
        prompt: str,
        inputs: dict,
    ) -> ThinkingTask:
        """
        Validate the reserved ``thinking_rounds`` runtime parameter and
        seed the task with it -- no LLM call here. ``inputs["thinking_rounds"]``
        is always present (``filter_inputs`` already injected
        ``THINKING_ROUNDS_PARAM``'s default of ``1`` if the caller omitted
        it, same guarantee ``run_id`` relies on). An invalid value raises
        here, before any round runs, rather than deeper inside ``think()``.

        The task always starts in the thinking phase; ``thinking_rounds == 0``
        means ``think()``'s own round-budget check
        (``len(task.thoughts) >= task.thinking_rounds``, true even before
        any round runs) switches straight to the reply phase on the very
        first call, without ever invoking the engine.
        """
        thinking_rounds = inputs["thinking_rounds"]
        if type(thinking_rounds) is not int or thinking_rounds < 0:
            raise AgentInvocationError(
                f"{self.full_name}: thinking_rounds must be a concrete int "
                f">= 0, got {thinking_rounds!r}."
            )

        return ThinkingTask(
            turns=turns,
            inputs=inputs,
            user_prompt=prompt,
            system_prompt_name=self.THINKING_PROMPT_NAME,
            thinking_rounds=thinking_rounds,
        )

    def think(self, task: ThinkingTask) -> ThinkingTask:
        """
        Advance ``task`` by one thinking round, or none at all if the round
        budget is already exhausted.

        No-ops (returns ``task`` unchanged) once ``task.system_prompt_name
        == "role"``. Otherwise: if the round budget is already exhausted
        (covers ``thinking_rounds=0``), switches straight to the reply
        phase without calling the engine. Otherwise renders, calls the
        engine, and stores the stripped output as this round's raw thought
        value.

        A round's stored value is the engine's raw returned output verbatim
        -- stripped if it's a ``str``, used as-is for any other
        JSON-decodable type (only possible when ``thinking_schema`` is
        set). There is no "empty round" concept: whatever the engine
        returns becomes this round's thought unconditionally, including a
        falsy value like ``False``, ``0``, or an empty string/collection --
        AA imposes no judgment on provider output here, matching how
        ``response_schema``/``LLMEngine`` already treat every
        JSON-decodable value as legitimate.
        """
        if task.system_prompt_name == "role":
            return task

        if len(task.thoughts) >= task.thinking_rounds:
            task.system_prompt_name = "role"
            task.task_messages = []
            return task

        task.task_messages = []
        messages = self.render_task(task)
        engine_result = (self._thinking_llm_engine or self._llm_engine).invoke({
            "messages": messages,
            "output_structure": self._thinking_schema,
        })
        raw = engine_result.result

        task.llm_records.append(LLMRecord(
            messages=list(task.task_messages),
            llm_result=engine_result,
            system_prompt_name=task.system_prompt_name,
        ))

        content = raw.strip() if isinstance(raw, str) else raw
        task.thoughts.append(content)

        if len(task.thoughts) >= task.thinking_rounds:
            task.system_prompt_name = "role"
            task.task_messages = []

        return task

    async def async_think(self, task: ThinkingTask) -> ThinkingTask:
        """Async mirror of ``think`` -- always a genuine independent
        implementation (real LLM I/O every call), matching this family's
        convention for hooks that perform real generation."""
        if task.system_prompt_name == "role":
            return task

        if len(task.thoughts) >= task.thinking_rounds:
            task.system_prompt_name = "role"
            task.task_messages = []
            return task

        task.task_messages = []
        messages = self.render_task(task)
        engine_result = await (self._thinking_llm_engine or self._llm_engine).async_invoke({
            "messages": messages,
            "output_structure": self._thinking_schema,
        })
        raw = engine_result.result

        task.llm_records.append(LLMRecord(
            messages=list(task.task_messages),
            llm_result=engine_result,
            system_prompt_name=task.system_prompt_name,
        ))

        content = raw.strip() if isinstance(raw, str) else raw
        task.thoughts.append(content)

        if len(task.thoughts) >= task.thinking_rounds:
            task.system_prompt_name = "role"
            task.task_messages = []

        return task

    def act(self, task: ThinkingTask) -> ThinkingTask:
        """No-ops while still thinking (``system_prompt_name != "role"``);
        otherwise ``BasicAgent.act``'s body verbatim."""
        if task.system_prompt_name != "role":
            return task
        return super().act(task)

    async def async_act(self, task: ThinkingTask) -> ThinkingTask:
        """Async mirror of ``act``, same gate."""
        if task.system_prompt_name != "role":
            return task
        return await super().async_act(task)

    # ------------------------------------------------------------------ #
    # Render pipeline
    # ------------------------------------------------------------------ #
    def _render_system_message(self, task: ThinkingTask) -> list[dict[str, str]]:
        """Dispatch ``thinking`` locally by rendering
        ``thinking_instructions`` directly, with no additional wrapper or
        fixed scaffold; delegate ``role`` to ``BasicAgent``'s own
        implementation."""
        if task.system_prompt_name != self.THINKING_PROMPT_NAME:
            return super()._render_system_message(task)

        rendered = self._thinking_instructions_config.render(task.inputs)
        return [{"role": "system", "content": rendered}]

    def _render_task_messages(self, task: ThinkingTask) -> list[dict[str, str]]:
        """Build this phase's task messages.

        Build-once contract as with every other family. The banner gains a
        fixed framing line -- signaling this is a preparatory thinking step,
        not the final reply -- only while ``task.system_prompt_name`` is
        the thinking phase; the role phase keeps the bare banner.

        The two phases diverge in shape, not just instruction text, past
        the banner: the thinking phase replays each completed thought as
        its own assistant turn (raw content, no round-number header),
        immediately followed by a "continue thinking" user turn -- so the
        model is never shown a self-labeling convention of its own to
        imitate or miscount. The reply phase keeps the single combined,
        headered summary block -- there is no imitation risk once thinking
        has concluded and the model is doing one-shot synthesis, not
        continuing its own act.
        """
        if task.task_messages:
            return task.task_messages

        banner = self._render_task_banner_text(task)

        if task.system_prompt_name == self.THINKING_PROMPT_NAME:
            banner = banner + "\n---\n" + _THINKING_FRAMING
            messages = [{"role": "user", "content": banner}]
            for thought in task.thoughts:
                messages.append({"role": "assistant", "content": self._stringify_thought(thought)})
                messages.append({"role": "user", "content": _THINKING_CONTINUATION_NUDGE})
        else:
            messages = [{"role": "user", "content": banner}]
            if task.thoughts:
                messages.append(
                    {"role": "assistant", "content": self._format_thoughts(task.thoughts)}
                )
                messages.append({"role": "user", "content": (
                    "Given the current task and the thoughts above, respond "
                    "to the current task."
                )})

        task.task_messages = messages
        return task.task_messages

    def _render_task_banner_text(self, task: ThinkingTask) -> str:
        """``===== CURRENT TASK =====`` banner text, matching the
        ``ToolAgent``/``_render_task_banner`` convention (``BasicAgent``
        itself has no banner helper to inherit -- its single-message reply
        never needed one)."""
        return f"===== CURRENT TASK =====\n{task.user_prompt}\n===== END TASK ====="

    @staticmethod
    def _stringify_thought(value: str | int | float | bool | list | dict | None) -> str:
        """Render one raw thought value as display text -- a ``str`` value
        used as-is, any other JSON-decodable value ``json.dumps``-rendered.
        Shared by the thinking-phase per-thought rendering above and
        ``_format_thoughts`` below, so the two can't drift on how a non-str
        thought gets stringified."""
        return value if isinstance(value, str) else json.dumps(value)

    @staticmethod
    def _format_thoughts(
        rounds: list[str | int | float | bool | list | dict | None],
        *,
        start_round: int = 0,
    ) -> str:
        """Render rounds of raw thought values as ``## Round N`` blocks --
        reply-phase use only (see ``_render_task_messages``); the thinking
        phase never shows this labeled shape to the model."""
        blocks = [
            f"## Round {i}\n{ThinkingAgent._stringify_thought(value)}"
            for i, value in enumerate(rounds, start=start_round)
        ]
        return "\n\n".join(blocks)

    # ------------------------------------------------------------------ #
    # Finalization
    # ------------------------------------------------------------------ #
    def _build_record_from_task(
        self,
        task: ThinkingTask,
        turns: list[AgentRecord],
    ) -> ThinkingAgentRecord:
        """Persist ``task.thoughts`` directly onto the completed record --
        no agent-level accumulator, no span bookkeeping."""
        prev = turns[-1] if turns else None
        return ThinkingAgentRecord(
            user_prompt=task.user_prompt,
            generated_response=task.generated_response,
            inputs=task.inputs,
            llm_records=tuple(task.llm_records),
            prev=prev,
            thoughts=tuple(task.thoughts),
        )

    def build_result_from_record(
        self,
        record: ThinkingAgentRecord,
        *,
        result: Any,
        started_at: datetime,
        ended_at: datetime,
    ) -> ThinkingAgentResult:
        """Construct this agent's ``ThinkingAgentResult`` directly from a
        completed ``ThinkingAgentRecord``."""
        llm_token_usage = tuple(r.llm_result.token_usage for r in record.llm_records)
        llm_model_data = record.llm_records[-1].llm_result.model_data

        return self._make_result(
            result=result,
            started_at=started_at,
            ended_at=ended_at,
            result_cls=ThinkingAgentResult,
            llm_token_usage=llm_token_usage,
            llm_model_data=llm_model_data,
            thinking_rounds_used=len(record.thoughts),
        )

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict:
        """Return a diagnostic snapshot including this agent's own
        construction knobs."""
        d = super().to_dict()
        d["thinking_instructions"] = self._thinking_instructions_config.template
        d["thinking_schema"] = self._thinking_schema
        if self._thinking_llm_engine is not None and self._thinking_llm_engine is not self._llm_engine:
            d["thinking_llm"] = self._thinking_llm_engine.to_dict()
        return d
