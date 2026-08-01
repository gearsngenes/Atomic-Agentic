"""
SelfAskAgent: Adaptive Self-Questioning ThinkingAgent

This module provides ``SelfAskAgent``, a concrete ``ThinkingAgent`` subclass
implementing the Self-Ask pattern: one self-asked follow-up question per
round, fused (observe + ask + answer + continue-signal) into a single JSON
LLM call. Thinking continues until the model self-declares it's done
(``keep_thinking=false``) or ``max_thinking_rounds`` is hit -- a hard safety
valve that forces a reply using whatever thoughts exist, never a raise.

Contrast
--------
For the shared thinking/reply lifecycle, thought persistence, and
``role_description`` mechanics, see ``agents/thinking.py`` (``ThinkingAgent``).
For the batch, ask-all-upfront strategy see ``agents/planask.py``
(``PlanAskAgent``, a later pass).
"""

from __future__ import annotations

import json
from typing import Any, Callable, Literal, Mapping, Optional

from .prompts import SELF_ASK_PROMPT
from ..constants.agents import (
    ANSWER_FIELD,
    KEEP_THINKING_FIELD,
    OBSERVATION_FIELD,
    QUESTION_FIELD,
    THOUGHT_FIELDS,
)
from ..exceptions import ThinkingAgentError
from ..llm.base import LLMEngine
from ..models.agents.prompts import PromptConfig
from ..models.agents.records import AgentRecord, LLMRecord
from ..models.agents.tasks import ThinkingTask
from ..models.agents.thought_models import AgentThought
from ..utils.agents import extract_json_object


class SelfAskAgent(ThinkingAgent):
    """
    Adaptive ``ThinkingAgent``: one self-asked follow-up question per round,
    fused into a single JSON LLM call producing
    ``{observation, question, answer, keep_thinking}``. Thinking continues
    until the model sets ``keep_thinking=false`` or ``max_thinking_rounds``
    is reached (safety valve -- replies with whatever thoughts exist so far,
    never a raise). Malformed output retries up to ``generation_retries``
    times with feedback injection; exhausting the budget raises
    ``ThinkingAgentError``.

    Exactly two system prompts exist for any instance: ``"role"`` (built by
    the base class from the caller's ``role_prompt``, reply phase only) and
    ``"self_ask"`` (this class's own fixed, non-configurable prompt,
    thinking phase only). No constructor parameter exposes the self-ask
    prompt for customization -- it is identical across every instance.
    """

    SELF_ASK_PROMPT_NAME = "self_ask"

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
        max_thinking_rounds: int,
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
        Pure passthrough to ``ThinkingAgent.__init__`` -- no new parameters
        except the ``max_thinking_rounds`` guard below. Registers the fixed
        ``self_ask`` system prompt after construction.
        """
        if max_thinking_rounds is None:
            raise TypeError(
                f"{type(self).__name__} requires a concrete positive int "
                "max_thinking_rounds -- it is the only backstop guaranteeing "
                "the self-ask loop terminates. None (unlimited) is not "
                "permitted."
            )
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
        self._system_prompts[self.SELF_ASK_PROMPT_NAME] = SELF_ASK_PROMPT

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
        Bare seed -- no LLM call. Every invocation performs at least one
        self-ask round before a reply is possible.
        """
        return ThinkingTask(
            turns=turns,
            inputs=inputs,
            user_prompt=prompt,
            system_prompt_name=self.SELF_ASK_PROMPT_NAME,
        )

    def _think(self, task: ThinkingTask) -> ThinkingTask:
        """
        One fused self-ask round, with a bounded JSON-decode/spec-validate
        retry loop mirroring ``ReActAgent._generate_next_step``'s shape.
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
                    "Produce a correctly formatted JSON object."
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
            result = self._validate_self_ask_round(parsed)
            if isinstance(result, str):
                feedback = result
                if task.retries_used >= self._generation_retries:
                    raise ThinkingAgentError(
                        f"{type(self).__name__}.{self.name}: generation retry budget exhausted "
                        f"after {task.retries_used + 1} attempt(s). Last error: {feedback}"
                    )
                round_repr = json.dumps(parsed, indent=2)
                additional_messages = [
                    {"role": "assistant", "content": round_repr},
                    {"role": "user", "content": (
                        f"The self-ask round you produced contains an error:\n\n"
                        f"{round_repr}\n\n"
                        f"Error: {feedback}\n\n"
                        "Reflect on this and produce a corrected round."
                    )},
                ]
                task.retries_used += 1
                continue

            thought, keep_thinking = result
            break

        task.thoughts.append(thought)
        task.rounds_used += 1
        # Clear now -- the next round's render_task (or the reply phase's)
        # must rebuild fresh, reflecting the just-appended thought.
        task.task_messages = []
        if not keep_thinking or task.rounds_used >= self._max_thinking_rounds:
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
                    "Produce a correctly formatted JSON object."
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

            result = self._validate_self_ask_round(parsed)
            if isinstance(result, str):
                feedback = result
                if task.retries_used >= self._generation_retries:
                    raise ThinkingAgentError(
                        f"{type(self).__name__}.{self.name}: generation retry budget exhausted "
                        f"after {task.retries_used + 1} attempt(s). Last error: {feedback}"
                    )
                round_repr = json.dumps(parsed, indent=2)
                additional_messages = [
                    {"role": "assistant", "content": round_repr},
                    {"role": "user", "content": (
                        f"The self-ask round you produced contains an error:\n\n"
                        f"{round_repr}\n\n"
                        f"Error: {feedback}\n\n"
                        "Reflect on this and produce a corrected round."
                    )},
                ]
                task.retries_used += 1
                continue

            thought, keep_thinking = result
            break

        task.thoughts.append(thought)
        task.rounds_used += 1
        task.task_messages = []
        if not keep_thinking or task.rounds_used >= self._max_thinking_rounds:
            task.system_prompt_name = "role"
        return task

    def _render_thinking_messages(
        self,
        task: ThinkingTask,
        *,
        additional_messages: list[dict[str, str]] | None = None,
    ) -> list[dict[str, str]]:
        """
        Render the self-ask system prompt (with ``role_description`` filled
        via an internally-computed render context, never ``task.inputs``)
        plus a fresh 3-message task thread: current-task banner, the
        thoughts-so-far snapshot, and a fixed per-round instruction.
        """
        system = self._render_system_message(task, {"role_description": self.role_description})
        historic = self._render_historic_messages(task)

        if not task.task_messages:
            task.task_messages = [
                {
                    "role": "user",
                    "content": (
                        f"===== CURRENT TASK =====\n{task.user_prompt}\n===== END TASK ====="
                    ),
                },
                {"role": "assistant", "content": self._render_thoughts_snapshot(task)},
                {
                    "role": "user",
                    "content": (
                        "Produce the next self-ask round: one JSON object with keys "
                        "{observation, question, answer, keep_thinking}."
                    ),
                },
            ]
        task.task_messages.extend(additional_messages or [])
        return system + historic + task.task_messages

    # ------------------------------------------------------------------ #
    # Private validation
    # ------------------------------------------------------------------ #
    def _validate_self_ask_round(self, parsed: Any) -> tuple[AgentThought, bool] | str:
        """
        Validate one parsed self-ask round.

        Returns ``(AgentThought, keep_thinking)`` on success, or a plain
        feedback string on failure (no class/name prefix, matching
        ``ReActAgent``'s validator convention).
        """
        if not isinstance(parsed, Mapping):
            return f"self-ask round output must be a JSON object; got {type(parsed).__name__!r}."

        raw = dict(parsed)

        extra = set(raw) - THOUGHT_FIELDS
        if extra:
            return f"self-ask round contains unsupported keys: {sorted(extra)!r}."

        missing = THOUGHT_FIELDS - set(raw)
        if missing:
            return f"self-ask round missing required keys: {sorted(missing)!r}."

        observation = raw[OBSERVATION_FIELD]
        if observation is not None and not isinstance(observation, str):
            return f"{OBSERVATION_FIELD!r} must be a string or null; got {type(observation).__name__!r}."

        question = raw[QUESTION_FIELD]
        if not isinstance(question, str) or not question.strip():
            return f"{QUESTION_FIELD!r} must be a non-empty string."

        answer = raw[ANSWER_FIELD]
        if not isinstance(answer, str) or not answer.strip():
            return f"{ANSWER_FIELD!r} must be a non-empty string."

        keep_thinking = raw[KEEP_THINKING_FIELD]
        if type(keep_thinking) is not bool:
            return f"{KEEP_THINKING_FIELD!r} must be a bool; got {type(keep_thinking).__name__!r}."

        thought = AgentThought(
            observation=observation,
            question=question.strip(),
            answer=answer.strip(),
        )
        return thought, keep_thinking
