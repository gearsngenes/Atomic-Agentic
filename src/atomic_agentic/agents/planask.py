"""
PlanAskAgent: Batch Self-Questioning ThinkingAgent

This module provides ``PlanAskAgent``, a concrete ``ThinkingAgent`` subclass
implementing the batch, ask-all-upfront strategy: one fused JSON LLM call
produces the entire list of already-answered thoughts at once -- no per-item
continuation signal, since batch completion is structural (decided in one
shot, never iterative). Range-validated for length, then the shared base
``ThinkingAgent._reply`` produces the outward response.

Contrast
--------
For the shared thinking/reply lifecycle, thought persistence, and
``role_description`` mechanics, see ``agents/thinking.py`` (``ThinkingAgent``).
For the adaptive, one-question-at-a-time strategy see ``agents/selfask.py``
(``SelfAskAgent``).
"""

from __future__ import annotations

import json
from typing import Any, Callable, Literal, Mapping, Optional

from .thinking import ThinkingAgent
from .prompts import PLAN_ASK_PROMPT
from ..constants.agents import (
    ANSWER_FIELD,
    OBSERVATION_FIELD,
    PLANNED_QUESTION_FIELDS,
    QUESTION_FIELD,
)
from ..exceptions import ThinkingAgentError
from ..llm.base import LLMEngine
from ..models.agents.prompts import PromptConfig
from ..models.agents.records import AgentRecord, LLMRecord
from ..models.agents.tasks import ThinkingTask
from ..models.agents.thought_models import AgentThought
from ..utils.agents import extract_json_object


class PlanAskAgent(ThinkingAgent):
    """
    Batch ``ThinkingAgent``: one fused JSON LLM call produces the entire
    self-ask thought list at once -- ``{observation, question, answer}`` per
    item, no ``keep_thinking`` (batch completion is structural, not
    self-declared). List length must be ``>= 1``; upper-bounded by
    ``max_thinking_rounds`` only when it is not ``None`` (``None`` permits an
    unbounded batch, mirroring ``ToolAgent.tool_calls_limit``'s own
    ``None``-means-unlimited precedent). Malformed output retries up to
    ``generation_retries`` times with feedback injection; exhausting the
    budget raises ``ThinkingAgentError``.

    Exactly two system prompts exist for any instance: ``"role"`` (built by
    the base class, reply phase only) and ``"plan_ask"`` (this class's own
    fixed, non-configurable prompt, thinking phase only, used exactly once
    per run). No constructor parameter exposes the plan-ask prompt for
    customization -- it is identical across every instance.
    """

    PLAN_ASK_PROMPT_NAME = "plan_ask"

    def __init__(
        self,
        name: str,
        namespace: str,
        description: str,
        llm_engine: LLMEngine,
        role_prompt: str | PromptConfig | None = None,
        role_description: str | None = None,
        filter_extraneous_inputs: Optional[bool] = None,
        context_enabled: bool = False,
        *,
        max_thinking_rounds: int | None,
        thoughts_window: int | None = 0,
        generation_retries: int = 0,
        pre_invoke: Optional[Callable | Any] = None,
        post_invoke: Optional[Callable | Any] = None,
        post_result_key: Optional[str] = None,
        records_window: Optional[int] = None,
        response_preview_limit: Optional[int] = None,
        assistant_response_source: Literal["raw", "final"] = "raw",
    ) -> None:
        """
        Pure passthrough to ``ThinkingAgent.__init__`` -- no new parameters,
        no guard on ``max_thinking_rounds`` (``None`` is permitted here: no
        upper bound on the produced batch list, unlike ``SelfAskAgent``).
        Registers the fixed ``plan_ask`` system prompt after construction.
        """
        super().__init__(
            name=name,
            namespace=namespace,
            description=description,
            llm_engine=llm_engine,
            role_prompt=role_prompt,
            role_description=role_description,
            filter_extraneous_inputs=filter_extraneous_inputs,
            context_enabled=context_enabled,
            max_thinking_rounds=max_thinking_rounds,
            thoughts_window=thoughts_window,
            generation_retries=generation_retries,
            pre_invoke=pre_invoke,
            post_invoke=post_invoke,
            post_result_key=post_result_key,
            records_window=records_window,
            response_preview_limit=response_preview_limit,
            assistant_response_source=assistant_response_source,
        )
        self._system_prompts[self.PLAN_ASK_PROMPT_NAME] = PLAN_ASK_PROMPT

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
        Bare seed -- no LLM call. The one batch call happens in ``_think``
        on the first ``_progress()`` call.
        """
        return ThinkingTask(
            turns=turns,
            inputs=inputs,
            user_prompt=prompt,
            system_prompt_name=self.PLAN_ASK_PROMPT_NAME,
        )

    def _think(self, task: ThinkingTask) -> ThinkingTask:
        """
        The one fused batch round this task will ever run, with a bounded
        JSON-decode/spec-validate retry loop mirroring
        ``SelfAskAgent._think``'s two-stage shape.
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

            # JSON extraction
            try:
                parsed = extract_json_object(raw_output, source_label=f"{type(self).__name__}.{self.name}")
            except json.JSONDecodeError as exc:
                feedback = (
                    f"Your output could not be parsed as valid JSON.\n\n"
                    f"Decoder error: {exc}\n\n"
                    f"The response you produced was:\n\n{raw_output}\n\n"
                    "Produce a correctly formatted JSON array."
                )
                if task.retries_used >= self._generation_retries:
                    raise ThinkingAgentError(
                        f"{type(self).__name__}.{self.name}: generation retry budget exhausted "
                        f"after {task.retries_used + 1} attempt(s). Last error is a JSONDecodeError: {exc}"
                    )
                additional_messages = [
                    {"role": "assistant", "content": raw_output},
                    {"role": "user", "content": feedback},
                ]
                task.retries_used += 1
                continue

            # Spec validation
            result = self._validate_planned_thoughts(parsed)
            if isinstance(result, str):
                feedback = result
                if task.retries_used >= self._generation_retries:
                    raise ThinkingAgentError(
                        f"{type(self).__name__}.{self.name}: generation retry budget exhausted "
                        f"after {task.retries_used + 1} attempt(s). Last error: {feedback}"
                    )
                batch_repr = json.dumps(parsed, indent=2)
                additional_messages = [
                    {"role": "assistant", "content": batch_repr},
                    {"role": "user", "content": (
                        f"The batch you produced contains an error:\n\n"
                        f"{batch_repr}\n\n"
                        f"Error: {feedback}\n\n"
                        "Reflect on this and produce a corrected batch."
                    )},
                ]
                task.retries_used += 1
                continue

            thoughts = result
            break

        task.thoughts.extend(thoughts)
        # Count of thoughts actually produced, not "+= 1" -- this phase
        # never runs a second time this task.
        task.rounds_used = len(thoughts)
        task.task_messages = []
        task.system_prompt_name = "role"
        return task

    async def _async_think(self, task: ThinkingTask) -> ThinkingTask:
        """Full independent async mirror of ``_think`` -- uses ``async_invoke``
        for each attempt. Same retry/budget/clearing logic throughout."""
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

            try:
                parsed = extract_json_object(raw_output, source_label=f"{type(self).__name__}.{self.name}")
            except json.JSONDecodeError as exc:
                feedback = (
                    f"Your output could not be parsed as valid JSON.\n\n"
                    f"Decoder error: {exc}\n\n"
                    f"The response you produced was:\n\n{raw_output}\n\n"
                    "Produce a correctly formatted JSON array."
                )
                if task.retries_used >= self._generation_retries:
                    raise ThinkingAgentError(
                        f"{type(self).__name__}.{self.name}: generation retry budget exhausted "
                        f"after {task.retries_used + 1} attempt(s). Last error is a JSONDecodeError: {exc}"
                    )
                additional_messages = [
                    {"role": "assistant", "content": raw_output},
                    {"role": "user", "content": feedback},
                ]
                task.retries_used += 1
                continue

            result = self._validate_planned_thoughts(parsed)
            if isinstance(result, str):
                feedback = result
                if task.retries_used >= self._generation_retries:
                    raise ThinkingAgentError(
                        f"{type(self).__name__}.{self.name}: generation retry budget exhausted "
                        f"after {task.retries_used + 1} attempt(s). Last error: {feedback}"
                    )
                batch_repr = json.dumps(parsed, indent=2)
                additional_messages = [
                    {"role": "assistant", "content": batch_repr},
                    {"role": "user", "content": (
                        f"The batch you produced contains an error:\n\n"
                        f"{batch_repr}\n\n"
                        f"Error: {feedback}\n\n"
                        "Reflect on this and produce a corrected batch."
                    )},
                ]
                task.retries_used += 1
                continue

            thoughts = result
            break

        task.thoughts.extend(thoughts)
        task.rounds_used = len(thoughts)
        task.task_messages = []
        task.system_prompt_name = "role"
        return task

    def _render_thinking_messages(
        self,
        task: ThinkingTask,
        *,
        additional_messages: list[dict[str, str]] | None = None,
    ) -> list[dict[str, str]]:
        """
        Render the plan-ask system prompt (with ``role_description`` and
        ``thought_count_range`` filled via an internally-computed render
        context, never ``task.inputs``) plus a fresh 2-message task thread:
        current-task banner and a fixed batch-output instruction. No
        thoughts-so-far snapshot -- ``task.thoughts`` is always empty going
        into this single round.
        """
        render_context = {
            "role_description": self.role_description,
            "thought_count_range": self._render_thought_count_range(),
        }
        system = self._render_system_message(task, render_context)
        historic = self._render_historic_messages(task)

        if not task.task_messages:
            task.task_messages = [
                {
                    "role": "user",
                    "content": (
                        f"===== CURRENT TASK =====\n{task.user_prompt}\n===== END TASK ====="
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Produce the full set of self-ask thoughts needed to answer this "
                        f"task well: a JSON array of {self._render_thought_count_range()} "
                        "objects, each {observation, question, answer}."
                    ),
                },
            ]
        task.task_messages.extend(additional_messages or [])
        return system + historic + task.task_messages

    def _render_thought_count_range(self) -> str:
        """
        Phrase the batch's length constraint for both the system prompt's
        ``{thought_count_range}`` placeholder and the fixed per-call
        instruction. Mirrors ``ReActAgent``'s own ``"unlimited" if X is None
        else str(X)`` idiom, generalized to a full phrase since this is
        prose, not a bare number substitution.
        """
        if self._max_thinking_rounds is None:
            return "as many as needed (no upper limit, minimum 1)"
        return f"between 1 and {self._max_thinking_rounds}"

    # ------------------------------------------------------------------ #
    # Private validation
    # ------------------------------------------------------------------ #
    def _validate_planned_thoughts(self, parsed: Any) -> list[AgentThought] | str:
        """
        Validate one parsed batch of planned thoughts.

        Returns ``list[AgentThought]`` on success, or a plain feedback
        string on the first failure (no class/name prefix, matching
        ``SelfAskAgent._validate_self_ask_round``'s convention).
        """
        if not isinstance(parsed, list):
            return f"batch output must be a JSON array; got {type(parsed).__name__!r}."

        if len(parsed) < 1:
            return "at least one thought is required in the batch."

        if self._max_thinking_rounds is not None and len(parsed) > self._max_thinking_rounds:
            return (
                f"batch contains {len(parsed)} thoughts, exceeding the max of "
                f"{self._max_thinking_rounds}."
            )

        thoughts: list[AgentThought] = []
        for i, item in enumerate(parsed):
            if not isinstance(item, Mapping):
                return f"item at index {i} must be a JSON object; got {type(item).__name__!r}."

            raw = dict(item)

            extra = set(raw) - PLANNED_QUESTION_FIELDS
            if extra:
                return f"item at index {i} contains unsupported keys: {sorted(extra)!r}."

            missing = PLANNED_QUESTION_FIELDS - set(raw)
            if missing:
                return f"item at index {i} missing required keys: {sorted(missing)!r}."

            observation = raw[OBSERVATION_FIELD]
            if observation is not None and not isinstance(observation, str):
                return (
                    f"item at index {i}: {OBSERVATION_FIELD!r} must be a string or null; "
                    f"got {type(observation).__name__!r}."
                )

            question = raw[QUESTION_FIELD]
            if not isinstance(question, str) or not question.strip():
                return f"item at index {i}: {QUESTION_FIELD!r} must be a non-empty string."

            answer = raw[ANSWER_FIELD]
            if not isinstance(answer, str) or not answer.strip():
                return f"item at index {i}: {ANSWER_FIELD!r} must be a non-empty string."

            thoughts.append(AgentThought(
                observation=observation,
                question=question.strip(),
                answer=answer.strip(),
            ))

        return thoughts
