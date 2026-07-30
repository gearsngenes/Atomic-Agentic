from __future__ import annotations

from typing import Any, Callable, Literal, Optional

from ..llm.base import LLMEngine
from ..models.agents.records import LLMRecord
from ..models.agents.prompts import PromptConfig
from ..models.agents.tasks import AgentTask
from ..utils.agents import normalize_role_prompt
from .base2 import Agent2


class BasicAgent2(Agent2):
    """
    Concrete single-role-prompt LLM agent, with thinking as an optional
    lifecycle feature (the ``thinking_rounds`` knob) rather than a distinct
    class.

    Direct functional successor to ``BasicAgent`` (``thinking_rounds=0``
    reproduces its exact single-call behavior) and to
    ``SelfAskAgent``/``PlanAskAgent`` (``thinking_rounds>0`` reasons across
    multiple internal rounds before replying, using ``Agent2``'s shared
    ``think``/``prepare``/``execute`` lifecycle instead of a bespoke one).
    """

    DEFAULT_ROLE_PROMPT = "You are a helpful AI assistant"

    # _permits_unbounded_thinking inherited from Agent2 (False, not
    # overridden) -- thinking_rounds=None is rejected at construction,
    # bounded thinking only.

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
        thinking_instructions: str | PromptConfig | None = None,
        thinking_rounds: int | None = 0,
        thoughts_per_round: int = 1,
        thoughts_window: int | None = 0,
    ) -> None:
        # 1-2. Normalize the role prompt and discover its placeholders.
        config = normalize_role_prompt(role_prompt, self.DEFAULT_ROLE_PROMPT)
        role_params = list(config.parameters)

        # 3. Delegate to Agent2; role placeholders are the sole
        # extra_parameters source, thinking knobs pass straight through.
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
            thinking_instructions=thinking_instructions,
            thinking_rounds=thinking_rounds,
            thoughts_per_round=thoughts_per_round,
            thoughts_window=thoughts_window,
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

    # ------------------------------------------------------------------ #
    # Task-lifecycle hooks
    # ------------------------------------------------------------------ #
    def prepare(self, task: AgentTask) -> AgentTask:
        """Advance ``task`` into the reply phase, once thinking is done.

        No-ops while ``task.keep_thinking`` is still ``True`` — thinking
        hasn't finished this round, so there's nothing to prepare yet.
        Otherwise stamps the reply phase's system prompt name and clears
        ``task.task_messages`` so ``_render_task_messages`` rebuilds fresh
        content instead of reusing leftover "think"-phase messages.
        """
        if task.keep_thinking:
            return task

        task.system_prompt_name = "role"
        task.task_messages = []
        return task

    async def async_prepare(self, task: AgentTask) -> AgentTask:
        """Async mirror of ``prepare``. No I/O either way, so the sync
        implementation is reused directly rather than duplicated."""
        return self.prepare(task)

    def execute(self, task: AgentTask) -> AgentTask:
        """Generate and record the reply, completing ``task``.

        No-ops while ``task.keep_thinking`` is still ``True`` (mirrors
        ``prepare``'s own gate). Otherwise renders the reply-phase
        messages, calls the engine, and marks the task complete. No
        response-type validation — ``LLMEngine.invoke()``'s own
        ``_extract_text`` contract already guarantees a ``str`` result.
        """
        if task.keep_thinking:
            return task

        messages = self.render_task(task)
        engine_result = self._llm_engine.invoke({"messages": messages})

        task.llm_records.append(LLMRecord(
            messages=list(task.task_messages),
            llm_result=engine_result,
            system_prompt_name=task.system_prompt_name,
        ))
        task.generated_response = engine_result.result
        task.complete = True
        return task

    async def async_execute(self, task: AgentTask) -> AgentTask:
        """Async mirror of ``execute``, awaiting the engine's native async
        path directly rather than a threaded default."""
        if task.keep_thinking:
            return task

        messages = self.render_task(task)
        engine_result = await self._llm_engine.async_invoke({"messages": messages})

        task.llm_records.append(LLMRecord(
            messages=list(task.task_messages),
            llm_result=engine_result,
            system_prompt_name=task.system_prompt_name,
        ))
        task.generated_response = engine_result.result
        task.complete = True
        return task

    # ------------------------------------------------------------------ #
    # Render pipeline
    # ------------------------------------------------------------------ #
    def _render_system_message(self, task: AgentTask) -> list[dict[str, str]]:
        """Render ``task``'s active system message.

        Delegates to ``Agent2``'s own "think"-phase rendering unchanged;
        overrides only to add the "role" phase, rendering
        ``system_prompts["role"]`` against the full ``task.inputs`` — same
        context ``BasicAgent``'s role-prompt placeholders resolve against
        today.
        """
        if task.system_prompt_name != "role":
            return super()._render_system_message(task)

        rendered = self._render_system_prompt(task, task.inputs)
        return [{"role": "system", "content": rendered}]

    def _render_task_messages(self, task: AgentTask) -> list[dict[str, str]]:
        """Build this phase's task messages, "think" and "role" alike.

        Mirrors the old ``ThinkingAgent``/``ReActAgent`` message
        convention: a ``CURRENT TASK`` banner, a self-labeled thoughts
        snapshot as an assistant turn, and a fixed instruction — used
        whenever ``task.thoughts`` is non-empty. When there's nothing to
        show, collapses to a single banner-wrapped message instead. The
        banner is always present, in every branch — a deliberate departure
        from old ``BasicAgent``'s bare, unwrapped reply message, so the
        active task stays visually anchored regardless of what else sits in
        the rendered message list (e.g. thought-laden history turns ahead
        of it via ``include_thoughts``/``thoughts_window``).
        """
        if task.task_messages:
            return task.task_messages

        banner = f"===== CURRENT TASK =====\n{task.user_prompt}\n===== END TASK ====="

        if task.system_prompt_name == "think":
            if not task.thoughts:
                task.task_messages = [{
                    "role": "user",
                    "content": (
                        f"{banner}\n\n"
                        "Begin thinking about this task, following the "
                        "OUTPUT FORMAT and STOP CONDITION rules in your "
                        "instructions."
                    ),
                }]
            else:
                task.task_messages = [
                    {"role": "user", "content": banner},
                    {"role": "assistant", "content": self._render_task_thoughts(task)},
                    {
                        "role": "user",
                        "content": (
                            "Continue thinking about this task, following "
                            "the OUTPUT FORMAT and STOP CONDITION rules in "
                            "your instructions."
                        ),
                    },
                ]
        else:
            if not task.thoughts:
                task.task_messages = [{"role": "user", "content": banner}]
            else:
                task.task_messages = [
                    {"role": "user", "content": banner},
                    {"role": "assistant", "content": self._render_task_thoughts(task)},
                    {
                        "role": "user",
                        "content": (
                            "Given the current task and the thoughts above, "
                            "respond to the current task."
                        ),
                    },
                ]

        return task.task_messages

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict:
        """Return a diagnostic snapshot including the role prompt."""
        d = super().to_dict()
        d["role_prompt"] = self.role_prompt
        return d
