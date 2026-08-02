"""
SelfAskAgent: Adaptive Self-Questioning BasicAgent

This module provides ``SelfAskAgent``, a concrete ``BasicAgent`` subclass
overriding ``think()``/``act()`` alone to add an adaptive, free-flowing
self-questioning phase before the reply. Each thinking round is one LLM
call producing categorized ``[CATEGORY] content`` lines (no JSON schema);
thinking continues until the model emits ``|STOP_THINKING|`` or
``max_thinking_rounds`` is hit -- a hard safety valve that forces a reply
using whatever thoughts exist, never a raise.
"""

from __future__ import annotations

import warnings
from datetime import datetime
from typing import Any, Callable, Literal, Optional

from .base import Agent
from .basic import BasicAgent
from .prompts import SELF_ASK_PROMPT
from ..constants.agents import (
    STOP_THINKING_SENTINEL,
    THINKING_ADDITIONAL_INSTRUCTIONS_FOOTER,
    THINKING_ADDITIONAL_INSTRUCTIONS_HEADER,
)
from ..exceptions import AgentError, AgentInvocationError, ThinkingAgentError
from ..llm.base import LLMEngine
from ..models.agents.prompts import PromptConfig
from ..models.agents.records import AgentRecord, LLMRecord, ThinkingAgentRecord
from ..models.agents.tasks import ThinkingTask
from ..models.agents.thought_models import AgentThought
from ..models.results.agents import ThinkingAgentResult
from ..utils.agents import normalize_role_prompt, normalize_thinking_instructions
from ..utils.agents import parse_thoughts
from ..utils.parameters import (
    insert_by_category,
    parameter_collisions,
    parameter_overlap,
    semantically_identical,
)


class SelfAskAgent(BasicAgent):
    """
    ``BasicAgent`` subclass adding an adaptive self-questioning phase.

    Overrides ``think()``/``act()``; ``prepare()`` stays ``BasicAgent``'s
    inherited no-op -- there is no deterministic bookkeeping step between
    a thinking round's decision and either the next round or the reply,
    since the phase switch is knowledge ``think()`` already has the moment
    it happens.

    Exactly two system prompts exist for any instance: ``"role"`` (the
    caller's own role prompt, reply phase only) and ``"self_ask"`` (this
    class's own fixed prompt, thinking phase only, with
    ``thinking_instructions`` spliced into its own reserved slot).

    Bypasses ``BasicAgent.__init__`` and calls ``Agent.__init__`` directly
    -- ``thinking_instructions`` needs to contribute its own placeholders
    to the schema alongside ``role_prompt``'s, which ``BasicAgent.__init__``'s
    narrower signature has no hook for. ``role_prompt``'s params take
    priority on any compatible-but-not-identical overlap with
    ``thinking_instructions``'s; a true collision raises.
    """

    SELF_ASK_PROMPT_NAME = "self_ask"

    def __init__(
        self,
        name: str,
        namespace: str,
        description: str,
        llm_engine: LLMEngine,
        role_prompt: str | PromptConfig | None = None,
        thinking_instructions: str | PromptConfig | None = None,
        filter_extraneous_inputs: Optional[bool] = None,
        context_enabled: bool = True,
        *,
        max_thinking_rounds: int,
        thoughts_per_round: int = 1,
        pre_invoke: Optional[Callable | Any] = None,
        post_invoke: Optional[Callable | Any] = None,
        post_result_key: Optional[str] = None,
        records_window: Optional[int] = None,
        response_preview_limit: Optional[int] = None,
        assistant_response_source: Literal["raw", "final"] = "raw",
    ) -> None:
        """
        Parameters
        ----------
        role_prompt : str | PromptConfig | None
            Reply-phase persona prompt. Same contract as ``BasicAgent``'s.
        thinking_instructions : str | PromptConfig | None
            Optional additional instructions spliced into the thinking
            phase's own prompt. May declare its own ``{placeholder}``s,
            reconciled against ``role_prompt``'s (role_prompt wins on
            compatible-but-not-identical overlap; true collisions raise).
        max_thinking_rounds : int
            Required. Hard cap on thinking rounds -- the only backstop
            guaranteeing the self-ask loop terminates. Must be a concrete
            int ``>= 0``; ``0`` means "skip thinking, reply immediately"
            (``think()``'s own round-budget check is already satisfied
            before any round runs). ``None`` (unbounded) is not permitted.
        thoughts_per_round : int
            Max thoughts kept per round; excess parsed thoughts are
            silently truncated. Must be a positive int (``>= 1``).
        """
        if max_thinking_rounds is None or type(max_thinking_rounds) is not int or max_thinking_rounds < 0:
            raise AgentError(
                f"{type(self).__name__} requires max_thinking_rounds to be a "
                "concrete int >= 0 -- it is the only backstop guaranteeing "
                "the self-ask loop terminates. None (unbounded) is not permitted."
            )
        if type(thoughts_per_round) is not int or thoughts_per_round < 1:
            raise AgentError("thoughts_per_round must be a positive int (>= 1).")

        role_config = normalize_role_prompt(role_prompt, self.DEFAULT_ROLE_PROMPT)
        role_params = list(role_config.parameters)

        thinking_config = normalize_thinking_instructions(thinking_instructions)
        thinking_params = list(thinking_config.parameters)

        # Reconcile role_prompt vs thinking_instructions BEFORE combining --
        # role_prompt is the priority source, same idiom Agent.__init__ uses
        # for its own pre/post/extra reconciliation.
        collisions = parameter_collisions(role_params, thinking_params)
        if collisions:
            raise AgentError(
                f"role_prompt/thinking_instructions parameter collision(s): "
                f"{collisions!r} (same name, incompatible type/kind)."
            )
        overlap = parameter_overlap(role_params, thinking_params)
        role_by_name = {p.name: p for p in role_params}
        thinking_by_name = {p.name: p for p in thinking_params}
        for overlap_name in overlap:
            if not semantically_identical(role_by_name[overlap_name], thinking_by_name[overlap_name]):
                warnings.warn(
                    f"Parameter {overlap_name!r} is declared by both role_prompt and "
                    "thinking_instructions and is compatible but not identical; "
                    "role_prompt's declaration wins.",
                    UserWarning,
                    stacklevel=3,
                )
        thinking_remainder = [p for p in thinking_params if p.name not in overlap]
        combined_extra_params = insert_by_category(role_params, thinking_remainder)

        Agent.__init__(
            self,
            name=name,
            namespace=namespace,
            description=description,
            llm_engine=llm_engine,
            filter_extraneous_inputs=filter_extraneous_inputs,
            context_enabled=context_enabled,
            pre_invoke=pre_invoke,
            post_invoke=post_invoke,
            post_result_key=post_result_key,
            extra_parameters=combined_extra_params,
            records_window=records_window,
            response_preview_limit=response_preview_limit,
            assistant_response_source=assistant_response_source,
        )

        self._system_prompts["role"] = role_config
        self._system_prompts[self.SELF_ASK_PROMPT_NAME] = SELF_ASK_PROMPT
        self._thinking_instructions_config = thinking_config

        self._max_thinking_rounds = max_thinking_rounds
        self._thoughts_per_round = thoughts_per_round
        self._thoughts: list[list[AgentThought]] = []

    # ------------------------------------------------------------------ #
    # Memory management
    # ------------------------------------------------------------------ #
    def clear_memory(self) -> None:
        """Clear the stored turn history and the persisted thoughts."""
        super().clear_memory()
        self._thoughts.clear()

    def get_thoughts(self, run_id: str | None = None) -> list[list[AgentThought]]:
        """Return the thought rounds produced by one invocation.

        Mirrors ``get_conversation``'s ``run_id`` resolution: ``None``
        resolves to the most recently committed record; an unknown
        ``run_id`` raises ``AgentInvocationError``. Every record in
        ``self._records`` for this agent is always a
        ``ThinkingAgentRecord`` (built exclusively by this class's own
        ``_build_record_from_task``), so ``thoughts_start``/``thoughts_end``
        are always present -- not re-checked here.

        Returns a shallow copy of the relevant slice of
        ``self._thoughts`` (one inner list per round).
        """
        if run_id is None:
            if not self._records:
                return []
            record = self._records[-1]
        else:
            record = next(
                (r for r in self._records if r.final_result.run_id == run_id),
                None,
            )
            if record is None:
                raise AgentInvocationError(
                    f"get_thoughts: no record with run_id {run_id!r} found in agent history."
                )
        return list(self._thoughts[record.thoughts_start:record.thoughts_end])

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
        Bare seed -- no LLM call. The task always starts in the self-ask
        phase; ``max_thinking_rounds == 0`` means ``think()``'s own
        round-budget check (``len(task.thoughts) >= self._max_thinking_rounds``,
        true even before any round runs) switches straight to the reply
        phase on the very first call, without ever invoking the engine.
        """
        return ThinkingTask(
            turns=turns,
            inputs=inputs,
            user_prompt=prompt,
            system_prompt_name=self.SELF_ASK_PROMPT_NAME,
        )

    def think(self, task: ThinkingTask) -> ThinkingTask:
        """
        Advance ``task`` by one thinking round, or none at all if the round
        budget is already exhausted.

        No-ops (returns ``task`` unchanged) once ``task.system_prompt_name
        == "role"``. Otherwise: if the round budget is already exhausted
        (covers ``max_thinking_rounds=0``), switches straight to the reply
        phase without calling the engine. Otherwise renders, calls the
        engine, and parses the (possibly ``|STOP_THINKING|``-truncated)
        output into categorized thoughts via ``parse_thoughts``.

        No retries: an empty raw LLM response is a hard failure
        (``ThinkingAgentError``), and the lax category-marker format
        degrades unmarked text to a single ``OTHER`` thought -- so the
        only other hard failure is a round whose parsed thoughts end up
        empty regardless (e.g. a bare/whitespace-only ``|STOP_THINKING|``
        with nothing before it). Never silently recorded as a no-op round.
        """
        if task.system_prompt_name == "role":
            return task

        if len(task.thoughts) >= self._max_thinking_rounds:
            task.system_prompt_name = "role"
            task.task_messages = []
            return task

        task.task_messages = []
        messages = self.render_task(task)
        engine_result = self._llm_engine.invoke({"messages": messages})
        raw = engine_result.result
        if not raw:
            raise ThinkingAgentError(
                f"{self.full_name}: thinking round produced empty output."
            )

        task.llm_records.append(LLMRecord(
            messages=list(task.task_messages),
            llm_result=engine_result,
            system_prompt_name=task.system_prompt_name,
        ))

        stop_seen = STOP_THINKING_SENTINEL in raw
        prefix = raw.split(STOP_THINKING_SENTINEL, 1)[0] if stop_seen else raw
        parsed = parse_thoughts(prefix)[: self._thoughts_per_round]
        if not parsed:
            raise ThinkingAgentError(
                f"{self.full_name}: thinking round produced no parsable "
                "thoughts (stop sentinel or empty content with no thought text)."
            )
        task.thoughts.append(parsed)

        if stop_seen or len(task.thoughts) >= self._max_thinking_rounds:
            task.system_prompt_name = "role"
            task.task_messages = []

        return task

    async def async_think(self, task: ThinkingTask) -> ThinkingTask:
        """Async mirror of ``think`` -- always a genuine independent
        implementation (real LLM I/O every call), matching this family's
        convention for hooks that perform real generation."""
        if task.system_prompt_name == "role":
            return task

        if len(task.thoughts) >= self._max_thinking_rounds:
            task.system_prompt_name = "role"
            task.task_messages = []
            return task

        task.task_messages = []
        messages = self.render_task(task)
        engine_result = await self._llm_engine.async_invoke({"messages": messages})
        raw = engine_result.result
        if not raw:
            raise ThinkingAgentError(
                f"{self.full_name}: thinking round produced empty output."
            )

        task.llm_records.append(LLMRecord(
            messages=list(task.task_messages),
            llm_result=engine_result,
            system_prompt_name=task.system_prompt_name,
        ))

        stop_seen = STOP_THINKING_SENTINEL in raw
        prefix = raw.split(STOP_THINKING_SENTINEL, 1)[0] if stop_seen else raw
        parsed = parse_thoughts(prefix)[: self._thoughts_per_round]
        if not parsed:
            raise ThinkingAgentError(
                f"{self.full_name}: thinking round produced no parsable "
                "thoughts (stop sentinel or empty content with no thought text)."
            )
        task.thoughts.append(parsed)

        if stop_seen or len(task.thoughts) >= self._max_thinking_rounds:
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
        """Dispatch ``self_ask`` locally; delegate ``role`` to
        ``BasicAgent``'s own implementation.

        The self-ask render context resolves ``thinking_instructions``
        against ``task.inputs`` first, wraps the result in a labeled
        section only when non-empty, then plugs it into
        ``SELF_ASK_PROMPT``'s own ``{user_thinking_instructions}`` slot
        alongside ``{thoughts_per_round}``/``{max_thinking_rounds}``.
        """
        if task.system_prompt_name != self.SELF_ASK_PROMPT_NAME:
            return super()._render_system_message(task)

        user_text = self._thinking_instructions_config.render(task.inputs)
        round_limit_text = (
            f"You may think across AT MOST {self._max_thinking_rounds} round(s) "
            "total for this task."
        )
        self_ask_context = {
            "thoughts_per_round": self._thoughts_per_round,
            "max_thinking_rounds": round_limit_text,
            "user_thinking_instructions": (
                THINKING_ADDITIONAL_INSTRUCTIONS_HEADER
                + user_text
                + THINKING_ADDITIONAL_INSTRUCTIONS_FOOTER
                if user_text
                else ""
            ),
        }
        rendered = self._system_prompts[self.SELF_ASK_PROMPT_NAME].render(self_ask_context)
        return [{"role": "system", "content": rendered}]

    def _render_task_messages(self, task: ThinkingTask) -> list[dict[str, str]]:
        """Build this phase's task messages, self_ask and role alike.

        Build-once contract as with every other family. self_ask: banner +
        thoughts-so-far snapshot (only when ``task.thoughts`` is
        non-empty) + fixed per-round instruction. role: banner + thoughts
        snapshot (if any) + reply instruction -- rendered regardless of
        *why* thinking concluded (sentinel or round budget), same content
        either way.
        """
        if task.task_messages:
            return task.task_messages

        banner = self._render_task_banner_text(task)

        if task.system_prompt_name == self.SELF_ASK_PROMPT_NAME:
            if not task.thoughts:
                task.task_messages = [{"role": "user", "content": banner}]
            else:
                task.task_messages = [
                    {"role": "user", "content": banner},
                    {"role": "assistant", "content": self._format_thoughts(task.thoughts)},
                    {
                        "role": "user",
                        "content": (
                            "Produce the next round of thoughts, one per line in "
                            "[CATEGORY] content format."
                        ),
                    },
                ]
            return task.task_messages

        if not task.thoughts:
            task.task_messages = [{"role": "user", "content": banner}]
        else:
            task.task_messages = [
                {"role": "user", "content": banner},
                {"role": "assistant", "content": self._format_thoughts(task.thoughts)},
                {
                    "role": "user",
                    "content": (
                        "Given the current task and the thoughts above, respond "
                        "to the current task."
                    ),
                },
            ]
        return task.task_messages

    def _render_task_banner_text(self, task: ThinkingTask) -> str:
        """``===== CURRENT TASK =====`` banner text, matching the
        ``ToolAgent``/``_render_task_banner`` convention (``BasicAgent``
        itself has no banner helper to inherit -- its single-message reply
        never needed one)."""
        return f"===== CURRENT TASK =====\n{task.user_prompt}\n===== END TASK ====="

    @staticmethod
    def _format_thoughts(rounds: list[list[AgentThought]], *, start_round: int = 0) -> str:
        """Render rounds of thoughts as ``## Round N`` grouped blocks, each
        thought as ``[CATEGORY] content``."""
        blocks: list[str] = []
        for i, round_thoughts in enumerate(rounds, start=start_round):
            lines = "\n".join(f"[{t.category}] {t.content}" for t in round_thoughts)
            blocks.append(f"## Round {i}\n{lines}")
        return "\n\n".join(blocks)

    # ------------------------------------------------------------------ #
    # Finalization
    # ------------------------------------------------------------------ #
    def _build_record_from_task(
        self,
        task: ThinkingTask,
        turns: list[AgentRecord],
    ) -> ThinkingAgentRecord:
        """Persist ``task.thoughts`` into ``self._thoughts`` and capture
        the span, mirroring ``ToolAgent.update_blackboard``'s pattern."""
        prev = turns[-1] if turns else None
        thoughts_start = len(self._thoughts)
        self._thoughts.extend(task.thoughts)
        thoughts_end = len(self._thoughts)
        return ThinkingAgentRecord(
            user_prompt=task.user_prompt,
            generated_response=task.generated_response,
            inputs=task.inputs,
            llm_records=tuple(task.llm_records),
            prev=prev,
            thoughts_start=thoughts_start,
            thoughts_end=thoughts_end,
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
            thoughts_start=record.thoughts_start,
            thoughts_end=record.thoughts_end,
        )

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict:
        """Return a diagnostic snapshot including persisted thoughts."""
        d = super().to_dict()
        d["thoughts"] = [[t.to_dict() for t in round_] for round_ in self._thoughts]
        return d
