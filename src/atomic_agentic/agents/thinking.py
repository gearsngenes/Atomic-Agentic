from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Callable, Literal, Optional

import pprint

from ..constants.agents import (
    ANSWER_FIELD,
    OBSERVATION_FIELD,
    QUESTION_FIELD,
    ROLE_DESCRIPTION_FIELD,
)
from ..exceptions import AgentError, AgentInvocationError
from ..llm.base import LLMEngine
from ..models.agents.prompts import PromptConfig
from ..models.agents.records import AgentRecord, LLMRecord, ThinkingAgentRecord
from ..models.agents.tasks import ThinkingTask
from ..models.agents.thought_models import AgentThought
from ..models.results.agents import ThinkingAgentResult
from ..utils.agents import normalize_role_prompt
from .base import Agent


class ThinkingAgent(Agent, ABC):
    """
    Abstract direct-``Agent`` sibling to ``ToolAgent`` — an internal
    self-questioning "thinking" phase precedes the outward response.

    Not a ``ToolAgent`` subclass: no tool registry/dispatch/blackboard
    vocabulary applies here. Concrete subclasses (``SelfAskAgent``,
    ``PlanAskAgent``) differ only in how thoughts get produced each round;
    everything about accumulating them, persisting them, and replying once
    thinking is done is shared here.

    The system-prompt name ``"role"`` is reserved for the reply/response
    phase across this whole family — subclasses must not reuse it for a
    thinking-phase prompt name. ``_progress``/``render_task`` both key off
    this exact string to decide which phase is active, so
    ``ThinkingTask`` needs no separate ``phase`` field: whatever
    ``system_prompt_name`` a round is stamped with already says what to do.
    """

    DEFAULT_ROLE_PROMPT: str = "You are a helpful AI assistant"

    _RESPONSE_INSTRUCTIONS_PREFIX = "=== RESPONSE INSTRUCTIONS START ===\n"
    _RESPONSE_INSTRUCTIONS_SUFFIX = (
        "\n=== RESPONSE INSTRUCTIONS END ===\n\n"
        "You will see the current task and the thoughts you formed while "
        "working through it. Use them to inform your response without "
        "restating them verbatim."
    )

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
        render_thoughts_in_history: bool = False,
        generation_retries: int = 0,
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
        max_thinking_rounds : int
            Hard cap on the number of thinking rounds this agent will run
            before forcing the transition to the reply phase. Required, no
            default — every concrete subclass enforces it differently
            (``SelfAskAgent``: a live safety backstop overriding any
            self-declared continuation signal; ``PlanAskAgent``: a
            validated upper bound on the upfront-planned question count).
        role_prompt : str | PromptConfig | None
            The caller's persona/response instructions. Wrapped at
            construction time inside a fixed fence plus a framework-authored
            note that thoughts precede the response in conversation — see
            class-level ``_RESPONSE_INSTRUCTIONS_PREFIX``/``_SUFFIX``.
        role_description : str | None
            A narrow, mutable pointer to what the role prompt is going for
            -- available to a subclass's thinking-phase prompts via the
            reserved ``{role_description}`` template field, without ever
            injecting the role prompt's full text. Resolved via fallback if
            not given: the resolved role prompt's own ``description``, else
            this agent's own ``description``. Freely settable afterward via
            the ``role_description`` property (a behavioral/tuning knob, not
            identity/shape).
        render_thoughts_in_history : bool
            Frozen at construction. Governs whether ``render_turn`` splices
            this agent's past thoughts into replayed history for later
            invocations — never required for the current run's own
            function, purely a memory/replay concern.
        generation_retries : int
            Number of additional LLM generation attempts subclasses may
            make when thinking-phase output fails validation. Same
            convention as ``ToolAgent.generation_retries``.

        Notes
        -----
        No ``extra_thinking_params``-style constructor knob exists.
        ``role_prompt``'s own discovered placeholders are the *only*
        ``extra_parameters`` source, mirroring ``ToolAgent`` (which has no
        legitimate source and drops the param outright, a22). A subclass's
        own thinking-phase prompt(s) (e.g. ``SelfAskAgent``'s ``"thinking"``)
        are rendered with an internally-computed render context (which may
        include ``{role_description}``), never through the caller-facing
        ``extra_parameters``/``inputs`` pipeline -- any such prompt must not
        declare placeholders that need a caller-supplied value.
        """
        if type(max_thinking_rounds) is not int or max_thinking_rounds <= 0:
            raise TypeError("max_thinking_rounds must be a positive int.")
        if type(generation_retries) is not int or generation_retries < 0:
            raise TypeError("generation_retries must be a non-negative int.")
        if type(render_thoughts_in_history) is not bool:
            raise TypeError("render_thoughts_in_history must be a bool.")
        if role_description is not None:
            if not isinstance(role_description, str):
                raise TypeError("role_description must be a str.")
            if not role_description.strip():
                raise ValueError("role_description must be a non-empty string.")

        # Wrap the caller's role prompt in the fixed RESPONSE INSTRUCTIONS
        # fence via plain concatenation (not .format()/re-templating) so an
        # already-normalized template's escaped literal braces pass through
        # untouched; the combined string is then normalized exactly once,
        # by this single PromptConfig construction.
        base_config = normalize_role_prompt(role_prompt, self.DEFAULT_ROLE_PROMPT)
        wrapped_template = (
            self._RESPONSE_INSTRUCTIONS_PREFIX
            + base_config.template
            + self._RESPONSE_INSTRUCTIONS_SUFFIX
        )
        role_config = PromptConfig(template=wrapped_template, description="Response role prompt")

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
            extra_parameters=list(role_config.parameters),
            records_window=records_window,
            response_preview_limit=response_preview_limit,
            assistant_response_source=assistant_response_source,
        )

        # One-time reserved-name check, now that pre_invoke/post_invoke/
        # role_prompt/run_id have all been merged into self.parameters --
        # catches a collision from any source in one pass, since by this
        # point everything is already unioned into one list.
        if any(p.name == ROLE_DESCRIPTION_FIELD for p in self.parameters):
            raise AgentError(
                "role_description is a framework-reserved name; no pre_invoke, "
                "post_invoke, or role_prompt parameter may be named "
                "'role_description'."
            )

        self._system_prompts["role"] = role_config
        self._max_thinking_rounds: int = max_thinking_rounds
        self._render_thoughts_in_history: bool = render_thoughts_in_history
        self._generation_retries: int = generation_retries
        self._role_description: str = (
            role_description
            or base_config.description.strip()
            or self._description
        )
        if not self._role_description:
            raise AgentError(
                "ThinkingAgent could not resolve a non-empty role_description "
                "from any source (explicit arg, role prompt description, or "
                "agent description)."
            )
        self._thoughts: list[AgentThought] = []

    @property
    def role_description(self) -> str:
        """
        Narrow, mutable pointer to what the role prompt is going for --
        available to a subclass's thinking-phase prompts via the reserved
        ``{role_description}`` template field, without ever injecting the
        role prompt's full text. A subclass renders it by merging
        ``role_description=self.role_description`` directly into its own
        ``.render(inputs)`` call wherever a thinking-phase prompt needs it.
        """
        return self._role_description

    @role_description.setter
    def role_description(self, value: str) -> None:
        if not isinstance(value, str):
            raise TypeError("role_description must be a str.")
        if not value.strip():
            raise ValueError("role_description must be a non-empty string.")
        self._role_description = value

    # ------------------------------------------------------------------ #
    # Thought inspection
    # ------------------------------------------------------------------ #
    def get_thoughts(self, run_id: str | None = None) -> list[AgentThought]:
        """
        Return the thoughts produced during one specific completed run.

        Mirrors ``Agent.get_conversation``'s ``run_id`` resolution: ``None``
        resolves to the most recently committed record; an unresolvable
        ``run_id`` raises ``AgentInvocationError``. Returns a plain slice of
        the persisted ``self._thoughts`` list, scoped to that record's own
        ``thoughts_start``/``thoughts_end`` span -- never a copy of the full
        persisted list.
        """
        if not self._records:
            return []
        if run_id is None:
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
        return self._thoughts[record.thoughts_start:record.thoughts_end]

    # ------------------------------------------------------------------ #
    # Task-lifecycle hooks
    # ------------------------------------------------------------------ #
    @abstractmethod
    def _initialize_task(
        self,
        *,
        turns: list[AgentRecord],
        prompt: str,
        inputs: dict,
    ) -> ThinkingTask:
        """
        Build and return this invocation's ``ThinkingTask`` (or subclass).

        Re-declared abstract over base ``Agent``'s concrete version —
        mirrors ``ToolAgent`` re-declaring it abstract, since only
        ``SelfAskAgent``/``PlanAskAgent`` know how to seed their own richer
        task. ``SelfAskAgent``'s is expected to be a bare seed
        (``system_prompt_name`` set to its ``"thinking"`` prompt name, no
        LLM call). ``PlanAskAgent``'s is expected to perform the upfront
        ``ask_questions`` LLM call and retry loop before returning.

        Async note: ``_async_initialize_task`` is intentionally not
        re-declared abstract here, matching ``ToolAgent``'s own precedent —
        it stays base ``Agent``'s inherited ``asyncio.to_thread``-wrapped
        default. A subclass whose ``_initialize_task`` performs real LLM
        I/O should override ``_async_initialize_task`` as a fully
        independent implementation (as ``PlanActAgent`` does); a subclass
        whose ``_initialize_task`` is I/O-free can rely on the default (as
        ``ReActAgent`` does).
        """
        ...

    def _progress(self, task: ThinkingTask) -> ThinkingTask:
        """Advance the task by exactly one round.

        Dispatches on ``task.system_prompt_name``: ``"role"`` means the
        reply phase is active, anything else means a thinking round.
        """
        if task.system_prompt_name == "role":
            return self._reply(task)
        return self._think(task)

    async def _async_progress(self, task: ThinkingTask) -> ThinkingTask:
        """Async mirror of ``_progress``.

        Overrides base ``Agent``'s inherited ``asyncio.to_thread``-wrapped
        default -- mirrors ``ToolAgent._async_progress``, since every round
        this dispatches to performs real LLM I/O.
        """
        if task.system_prompt_name == "role":
            return await self._async_reply(task)
        return await self._async_think(task)

    @abstractmethod
    def _think(self, task: ThinkingTask) -> ThinkingTask:
        """
        One full thinking round.

        Render (via ``render_task`` -> ``_render_thinking_messages``), call
        the engine, validate/retry as the subclass's schema requires,
        append a completed ``AgentThought`` to ``task.thoughts``, and
        increment ``task.rounds_used``. Before returning, decide whether
        thinking continues: if not, set ``task.system_prompt_name =
        "role"`` so the next ``_progress`` call routes to ``_reply``.
        Subclasses must also force this transition when
        ``task.rounds_used >= self._max_thinking_rounds`` regardless of any
        self-declared continuation signal (hard safety cap).
        """
        ...

    @abstractmethod
    async def _async_think(self, task: ThinkingTask) -> ThinkingTask:
        """
        Async mirror of ``_think``.

        Always a genuine independent implementation, never a thread-offload
        default -- every subclass's thinking round performs real LLM I/O,
        so this always needs ``await self._llm_engine.async_invoke(...)``.
        Mirrors ``PlanActAgent``/``ReActAgent``'s precedent of writing sync
        and async generation-retry loops as fully independent
        implementations rather than sharing one via a helper.
        """
        ...

    def _reply(self, task: ThinkingTask) -> ThinkingTask:
        """
        Final round: produce the outward response.

        Plain-string output only -- no retry, matching
        ``BasicAgent._progress``'s posture exactly (there is nothing
        structural to malform once thinking is over).
        """
        messages = self.render_task(task)
        engine_result = self._llm_engine.invoke({"messages": messages})
        text = engine_result.result
        if not isinstance(text, str):
            raise AgentInvocationError(
                f"LLM engine returned non-string result (type={type(text).__name__})."
            )
        llm_record = LLMRecord(
            messages=list(task.task_messages),
            llm_result=engine_result,
            system_prompt_name=task.system_prompt_name,
        )
        task.llm_records.append(llm_record)
        task.generated_response = text
        task.complete = True
        return task

    async def _async_reply(self, task: ThinkingTask) -> ThinkingTask:
        """
        Async mirror of ``_reply``.

        Reuses the plain sync ``render_task`` -- pure message construction,
        no I/O, no async twin needed -- but awaits the engine's native
        async path directly instead of thread-offloading (mirrors
        ``BasicAgent._async_progress``).
        """
        messages = self.render_task(task)
        engine_result = await self._llm_engine.async_invoke({"messages": messages})
        text = engine_result.result
        if not isinstance(text, str):
            raise AgentInvocationError(
                f"LLM engine returned non-string result (type={type(text).__name__})."
            )
        llm_record = LLMRecord(
            messages=list(task.task_messages),
            llm_result=engine_result,
            system_prompt_name=task.system_prompt_name,
        )
        task.llm_records.append(llm_record)
        task.generated_response = text
        task.complete = True
        return task

    def render_task(
        self,
        task: ThinkingTask,
        *,
        additional_messages: list[dict[str, str]] | None = None,
    ) -> list[dict[str, str]]:
        """
        Concrete dispatcher (deviates from ``ToolAgent``'s "every concrete
        subclass fully reimplements ``render_task``" convention -- justified
        here since the reply-phase shape below is byte-identical across
        both subclasses).

        For ``"role"``: builds the shared 3-message reply thread (mirrors
        ``ReActAgent.render_task``'s convention) -- a ``CURRENT TASK``
        banner, a self-labeling thoughts snapshot, and a fixed instruction.
        Otherwise delegates to the subclass's ``_render_thinking_messages``.
        """
        if task.system_prompt_name == "role":
            system = self._render_system_message(task, task.inputs)
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
                            "Given the current task and the thoughts above, respond to "
                            "the current task."
                        ),
                    },
                ]
            task.task_messages.extend(additional_messages or [])
            return system + historic + task.task_messages
        return self._render_thinking_messages(task, additional_messages=additional_messages)

    def _render_thoughts_snapshot(self, task: ThinkingTask) -> str:
        """
        Build the self-labeling thoughts-so-far snapshot text.

        Mirrors ``ReActAgent.render_task``'s running-plan-snapshot
        convention, including its empty-case fallback string.
        """
        if not task.thoughts:
            return "THOUGHTS SO FAR:\nNo thoughts yet."
        records = [
            {
                OBSERVATION_FIELD: t.observation,
                QUESTION_FIELD: t.question,
                ANSWER_FIELD: t.answer,
            }
            for t in task.thoughts
        ]
        return (
            f"THOUGHTS 0-{len(task.thoughts) - 1} SO FAR:\n\n"
            + pprint.pformat(records, indent=2, width=160, sort_dicts=False)
        )

    @abstractmethod
    def _render_thinking_messages(
        self,
        task: ThinkingTask,
        *,
        additional_messages: list[dict[str, str]] | None = None,
    ) -> list[dict[str, str]]:
        """
        Phase-specific message construction for one thinking round,
        consumed by ``_think``/``_async_think``. Fully subclass-owned --
        ``SelfAskAgent``'s fused-call shape and ``PlanAskAgent``'s
        per-round answer-shape share nothing beyond the system/historic
        message helpers already on base ``Agent``.
        """
        ...

    # ------------------------------------------------------------------ #
    # History rendering
    # ------------------------------------------------------------------ #
    def render_turn(self, turn: AgentRecord) -> list[dict[str, str]]:
        """
        Render one stored ``ThinkingAgentRecord`` into LLM-facing messages.

        Gated by ``render_thoughts_in_history`` -- when ``False`` (default)
        this is identical to base ``Agent.render_turn``. When ``True``,
        splices a thought-trail block into the single assistant message's
        content (never a separate message, to respect strict
        user/assistant alternation), scoped to this record's own
        ``thoughts_start``/``thoughts_end`` span. Thoughts are listed first,
        with the final response appended last -- reading order matches the
        order they actually happened in. Thought indices are not labeled;
        nothing downstream needs to address a thought by position.
        """
        if not isinstance(turn, ThinkingAgentRecord):
            raise AgentInvocationError(
                f"render_turn expected ThinkingAgentRecord, got {type(turn)!r}"
            )

        messages = super().render_turn(turn)
        if not self._render_thoughts_in_history:
            return messages

        start = turn.thoughts_start
        end = turn.thoughts_end
        if start is None or end is None or start == end:
            return messages

        assistant_response = messages[1]["content"]
        records = [
            {
                OBSERVATION_FIELD: t.observation,
                QUESTION_FIELD: t.question,
                ANSWER_FIELD: t.answer,
            }
            for t in self._thoughts[start:end]
        ]
        dump = pprint.pformat(records, indent=2, width=160, sort_dicts=False)
        assistant_content = (
            f"THOUGHTS:\n\n{dump}\n\n"
            f"RESPONSE:\n{assistant_response}"
        )
        return [messages[0], {"role": "assistant", "content": assistant_content}]

    # ------------------------------------------------------------------ #
    # Result construction
    # ------------------------------------------------------------------ #
    def _persist_thoughts(self, task: ThinkingTask) -> ThinkingTask:
        """
        Merge this run's task-local thoughts into the agent-level
        persisted list.

        Much simpler than ``ToolAgent.update_blackboard`` -- thoughts carry
        no placeholder-reference syntax and no in-progress status to
        reconcile, so no rewriting or tail-trimming is needed.
        """
        self._thoughts.extend(task.thoughts)
        return task

    def _build_record_from_task(
        self,
        task: ThinkingTask,
        turns: list[AgentRecord],
    ) -> ThinkingAgentRecord:
        """
        Assemble a completed ``ThinkingAgentRecord`` from a finished
        ``ThinkingTask``. Mirrors ``ToolAgent._build_record_from_task``,
        substituting thoughts for blackboard slots.
        """
        prev = turns[-1] if turns else None
        thoughts_start = len(self._thoughts)
        task = self._persist_thoughts(task)
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
        """
        Construct this ThinkingAgent's ``ThinkingAgentResult`` envelope
        directly from a completed ``ThinkingAgentRecord``.

        Extends ``Agent.build_result_from_record``: carries the record's own
        ``thoughts_start``/``thoughts_end`` span straight through -- indices
        only, not thought content (mirrors ``ToolAgent``'s pattern of
        deriving the result from the record rather than task-local state,
        though here there's nothing left to derive; the record already has
        exactly what the result needs).
        """
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
