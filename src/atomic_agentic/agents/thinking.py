"""
ThinkingAgent: abstract think-then-reply base.

``ThinkingAgent`` is the shared mechanism for agents that run a thinking phase
before replying: the phase gate, the task-message rendering frame and three
strategy hooks. It owns no engine call, thinking prompt, schema, round count or
ceiling; those are strategy decisions made inside the hooks.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

from .basic import BasicAgent
from ..constants.agents import REPLY_DIRECTIVE, THINKING_DIRECTIVE
from ..models.agents.records import ThinkingAgentRecord
from ..models.agents.tasks import ThinkingTask
from ..models.results.agents import ThinkingAgentResult
from ..utils.rendering import frame_content, stringify_value


class ThinkingAgent(BasicAgent):
    """
    Abstract ``BasicAgent`` subclass: think, then reply.

    Construction is ``BasicAgent``'s, unchanged (no ``__init__`` override, no
    thinking-specific parameter). A concrete subclass implements the three
    abstract hooks and supplies its thinking-phase system message by
    overriding ``_render_system_message``; while thinking, this base renders
    none. The reply phase uses the role prompt, rendered by ``BasicAgent``. A
    thinking prompt a subclass renders itself sees ``task.inputs`` only: a
    ``{placeholder}`` in it is not wired into the agent's parameters by this base
    (only role-prompt placeholders are); a subclass that wants them in the
    signature does so through ``_merge_prompt_parameters``.

    Phase: ``task.thinking_complete``. False means thinking, True means reply (the
    "reply phase" below). ``think`` is its only writer.

    Lifecycle (one ``invoke``): ``create_task``, then the base loop. Each pass
    runs ``think`` (set the flag from ``_thinking_done``, then generate if it is
    still false), ``prepare`` (no-op) and ``act`` (no-op until the flag is true).
    The flag therefore turns true on the pass after the last thought is
    generated, and ``act`` replies in that same pass. ``_thinking_done`` is only
    ever called on a complete task, after ``summarize`` and ``pre_process``.

    Rendering (every call, ``_render_task_messages``), after the system message
    and history. The task block is the context block (when ``task.context`` is
    non-empty) then the wrapped prompt.
      * thoughts present: user task block, one assistant message holding the
        thoughts framed by the thoughts markers, user directive (``REPLY_DIRECTIVE``
        in the reply phase, else ``THINKING_DIRECTIVE``).
      * no thoughts, reply phase: the task block alone as one user message,
        identical to ``BasicAgent``'s (no directive).
      * no thoughts, thinking phase: the task block and ``THINKING_DIRECTIVE`` in
        one user message.
    Absent ``additional_messages``, the list never holds two consecutive user
    messages and ends on a user message (several providers reject a request that
    ends on an assistant message).

    The result is ``ThinkingAgentResult`` (``thoughts_produced``), built by
    ``build_result_from_task``.

    Hook contracts (the strategy author's side):
      * ``_thinking_done`` -- deterministic and side-effect free; the only
        termination mechanism, so every implementation must guarantee it
        eventually returns True (the base counts nothing).
      * ``_generate_thoughts`` -- see its docstring.
    """

    _TASK_CLS = ThinkingTask

    # ------------------------------------------------------------------ #
    # Strategy hooks
    # ------------------------------------------------------------------ #
    @abstractmethod
    def _thinking_done(self, task: ThinkingTask) -> bool:
        """Whether thinking is finished for ``task``.

        Evaluated at the start of every ``think`` call (so after
        ``pre_process``: ``task.user_prompt`` and ``task.historic_messages`` are
        populated) and must be deterministic and free of side effects. Must
        eventually return True: a strategy whose ``_thinking_done`` never becomes
        true hangs ``invoke``, since the base has no ceiling by design.
        """
        ...

    @abstractmethod
    def _generate_thoughts(self, task: ThinkingTask) -> None:
        """Run one thinking step, in place.

        1. build the messages with ``self.render_task(task)`` (rebuilt each
           call, so it reflects ``task.thoughts`` so far); a strategy that
           needs a different system message for this step overrides
           ``_render_system_message`` and reads its own task state
        2. make the engine call(s) the strategy needs with ``self._llm_engine``
        3. for every call append
           ``LLMRecord(messages=list(task.task_messages), llm_result=...)`` to
           ``task.llm_records``
        4. append the produced thought(s), any type, to ``task.thoughts``
        5. leave the task in a state where ``_thinking_done`` will eventually
           be true
        """
        ...

    @abstractmethod
    async def _agenerate_thoughts(self, task: ThinkingTask) -> None:
        """Async twin of ``_generate_thoughts``: same steps, awaiting the
        engine's ``async_invoke``. Hard-abstract (no thread-offload default),
        like ``act``/``async_act``."""
        ...

    def _render_thoughts(self, task: ThinkingTask) -> str:
        """Return the text between the thoughts markers (default format;
        override for another, e.g. a tree or transcript).

        Only called when ``task.thoughts`` is non-empty. Must return ``str``.

        1. each thought as its string (a ``str`` as-is, anything else as JSON)
        2. joined by ``---`` rules, unnumbered

        The base adds the markers and elides forged marker strings.
        """
        return "\n\n---\n\n".join(
            t if isinstance(t, str) else stringify_value(t, mode="j")
            for t in task.thoughts
        )

    # ------------------------------------------------------------------ #
    # Lifecycle (concrete)
    # ------------------------------------------------------------------ #
    def think(self, task: ThinkingTask) -> ThinkingTask:
        """
        1. ``task.thinking_complete = self._thinking_done(task)``
        2. if not complete: ``self._generate_thoughts(task)``
        3. return ``task``
        """
        task.thinking_complete = self._thinking_done(task)
        if not task.thinking_complete:
            self._generate_thoughts(task)
        return task

    async def async_think(self, task: ThinkingTask) -> ThinkingTask:
        """Async mirror of ``think``: step 2 is ``await
        self._agenerate_thoughts(task)``."""
        task.thinking_complete = self._thinking_done(task)
        if not task.thinking_complete:
            await self._agenerate_thoughts(task)
        return task

    def act(self, task: ThinkingTask) -> ThinkingTask:
        """
        1. if not ``task.thinking_complete``: return ``task``
        2. return ``super().act(task)`` (``BasicAgent``: one engine call with
           ``response_schema``, ``generated_response`` set, task complete)
        """
        if not task.thinking_complete:
            return task
        return super().act(task)

    async def async_act(self, task: ThinkingTask) -> ThinkingTask:
        """Async mirror of ``act`` (``await super().async_act(task)``)."""
        if not task.thinking_complete:
            return task
        return await super().async_act(task)

    # ------------------------------------------------------------------ #
    # Rendering (concrete)
    # ------------------------------------------------------------------ #
    def _render_system_message(self, task: ThinkingTask) -> list[dict[str, str]]:
        """
        1. while thinking (``not task.thinking_complete``): ``[]``; a concrete
           strategy overrides this to render its thinking message
        2. otherwise the role message (``BasicAgent``'s)
        """
        if not task.thinking_complete:
            return []
        return super()._render_system_message(task)

    def _render_task_messages(self, task: ThinkingTask) -> list[dict[str, str]]:
        """
        Rebuilt on every call (no build-once guard; this family never retries).
        A reply with no thoughts is exactly ``BasicAgent``'s message.
        ``additional_messages`` passed to ``render_task`` extend the returned list
        for that call only; they do not persist to later calls.

        1. ``block = self._render_task_block(task)``
        2. thoughts present: user ``block``, one assistant message holding
           ``frame_content(self._render_thoughts(task), "THOUGHTS")``, user directive
        3. no thoughts, reply phase: user ``block`` alone
        4. no thoughts, thinking phase: user ``block``, a blank line, directive
        5. ``task.task_messages = <the list>``; return that same list object
        """
        block = self._render_task_block(task)
        replying = task.thinking_complete
        if task.thoughts:
            body = frame_content(self._render_thoughts(task), "THOUGHTS")
            messages = [
                {"role": "user", "content": block},
                {"role": "assistant", "content": body},
                {"role": "user", "content": REPLY_DIRECTIVE if replying else THINKING_DIRECTIVE},
            ]
        elif replying:
            messages = [{"role": "user", "content": block}]
        else:
            messages = [{"role": "user", "content": f"{block}\n\n{THINKING_DIRECTIVE}"}]
        task.task_messages = messages
        return messages

    # ------------------------------------------------------------------ #
    # Finalization (concrete)
    # ------------------------------------------------------------------ #
    def _build_record_from_task(self, task: ThinkingTask) -> ThinkingAgentRecord:
        """
        1. return ``ThinkingAgentRecord`` built from ``_record_base_fields`` plus
           ``thoughts=tuple(task.thoughts)``
        """
        return ThinkingAgentRecord(
            **self._record_base_fields(task),
            thoughts=tuple(task.thoughts),
        )

    def build_result_from_task(self, task: ThinkingTask, result: Any) -> ThinkingAgentResult:
        """
        ``Agent.build_result_from_task`` with the family result class.

        1. ``llm_token_usage`` from ``task.llm_records``; ``llm_model_data`` from
           the engine's configuration
        2. return ``ThinkingAgentResult`` with ``thoughts_produced=len(task.thoughts)``
        """
        llm_token_usage = tuple(r.llm_result.token_usage for r in task.llm_records)
        llm_model_data = self._llm_engine.model_data

        return self._make_result(
            result=result,
            started_at=task.started_at,
            ended_at=task.ended_at,
            result_cls=ThinkingAgentResult,
            run_id=task.run_id,
            llm_token_usage=llm_token_usage,
            llm_model_data=llm_model_data,
            thoughts_produced=len(task.thoughts),
        )
