"""
ThinkingAgent: abstract think-then-reply base.

``ThinkingAgent`` is the shared mechanism for agents that run a thinking phase
before replying: the phase gate, the task-message rendering frame and four
strategy hooks. It owns no engine call, thinking prompt, schema, round count or
ceiling; those are strategy decisions made inside the hooks.
"""

from __future__ import annotations

from abc import abstractmethod

from .basic import BasicAgent
from ..models.agents.records import ThinkingAgentRecord
from ..models.agents.tasks import ThinkingTask
from ..utils.rendering import render_thoughts_block


class ThinkingAgent(BasicAgent):
    """
    Abstract ``BasicAgent`` subclass: think, then reply.

    Construction is ``BasicAgent``'s, unchanged (no ``__init__`` override, no
    thinking-specific parameter). A concrete subclass sets
    ``_INITIAL_PROMPT_NAME`` to its (first) thinking prompt name, registers its
    thinking prompt(s) in ``self._system_prompts``, and implements the four
    abstract hooks. The reply phase always uses the role prompt, registered by
    ``BasicAgent`` under ``"role"``. A thinking prompt registered this way is
    rendered against ``task.inputs`` only: a ``{placeholder}`` in it is not wired
    into the agent's parameters (only role-prompt placeholders are), so a
    subclass using placeholders must declare them itself or avoid them.

    Phase: ``task.system_prompt_name`` is the indicator. ``"role"`` means the
    reply phase; any other value means thinking. ``_thinking_done`` is
    evaluated at seed and after each ``_generate_thoughts`` call; its only
    effect is to switch the name to ``"role"``.

    Lifecycle (one ``invoke``):
      ``create_task`` -> ``_seed_task`` (reply immediately if done) -> loop
      {``think`` (no-op in the reply phase), ``prepare`` (no-op), ``act`` (no-op
      until the reply phase)} until the task completes. The pass that flips the
      phase runs ``think`` and then ``act``.

    Rendering (every call, ``_render_task_messages``), after the system message
    and history, is one of four shapes. The task block is the context block (when
    ``task.context`` is non-empty) then the wrapped prompt.
      * reply, no thoughts: the task block alone, as one user message,
        identical to ``BasicAgent``'s (no directive).
      * thinking, no thoughts: the task block and the directive in one user
        message.
      * reply, thoughts present: user task block, one assistant thoughts
        message framed by the thoughts markers, user directive.
      * thinking, thoughts present: the same three messages as the previous
        shape (only the directive differs).
    Absent ``additional_messages``, the list never holds two consecutive user
    messages and ends on a user message.

    Hook contracts (the strategy author's side):
      * ``_thinking_done`` -- deterministic, side-effect free, safe on an empty
        task; the only termination mechanism, so every implementation must
        guarantee it eventually returns True (the base counts nothing).
      * ``_generate_thoughts`` -- see its docstring.
    """

    _TASK_CLS = ThinkingTask
    # Deliberately overrides BasicAgent's "role" with None. Unset until a
    # concrete subclass sets its thinking prompt name; an unset subclass still
    # thinks (the hooks run) but with no system message, instead of silently
    # skipping thinking.
    _INITIAL_PROMPT_NAME: str | None = None

    # ------------------------------------------------------------------ #
    # Strategy hooks (abstract)
    # ------------------------------------------------------------------ #
    @abstractmethod
    def _thinking_done(self, task: ThinkingTask) -> bool:
        """Whether thinking is finished for ``task``.

        Called at seed (``task.thoughts`` empty) and after every
        ``_generate_thoughts`` call. Must be deterministic, free of side
        effects and tolerate an empty task. Must eventually return True: a
        strategy whose ``_thinking_done`` never becomes true hangs ``invoke``,
        since the base has no ceiling by design. The seed call runs before
        ``summarize``/``pre_process``, so ``task.user_prompt`` is ``""`` and
        ``task.historic_messages`` is ``[]`` then.
        """
        ...

    @abstractmethod
    def _generate_thoughts(self, task: ThinkingTask) -> None:
        """Run one thinking step, in place.

        1. build the messages with ``self.render_task(task)`` (rebuilt each
           call, so it reflects ``task.thoughts`` so far); a strategy that
           needs a different system prompt for this step sets
           ``task.system_prompt_name`` to one of its registered thinking
           names first (never to ``"role"``: the base cannot tell such a write
           from its own flip, so assigning it ends thinking at once; the base
           switches to the reply phase through ``_thinking_done``)
        2. make the engine call(s) the strategy needs with ``self._llm_engine``
        3. for every call append
           ``LLMRecord(messages=list(task.task_messages), llm_result=...,
           system_prompt_name=task.system_prompt_name)`` to ``task.llm_records``
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

    @abstractmethod
    def _render_thoughts(self, task: ThinkingTask) -> str:
        """Return the text between the thoughts markers.

        Only called when ``task.thoughts`` is non-empty. Must return ``str``
        (a non-str fails later in ``elide_markers``). The format is the
        strategy's (chain, tree, transcript, ...); the base adds the markers
        and elides forged marker strings.
        """
        ...

    # ------------------------------------------------------------------ #
    # Lifecycle (concrete)
    # ------------------------------------------------------------------ #
    def _seed_task(self, task: ThinkingTask) -> None:
        """
        1. if ``self._thinking_done(task)``: set ``task.system_prompt_name =
           "role"`` (no thinking call will be made)

        A subclass overriding this seeds its own state first and calls
        ``super()._seed_task(task)`` last, so the check sees the seeded state.
        """
        if self._thinking_done(task):
            task.system_prompt_name = "role"

    def think(self, task: ThinkingTask) -> ThinkingTask:
        """
        1. if ``task.system_prompt_name == "role"``: return ``task``
        2. ``self._generate_thoughts(task)``
        3. if ``self._thinking_done(task)``: set ``task.system_prompt_name =
           "role"``
        4. return ``task``
        """
        if task.system_prompt_name == "role":
            return task
        self._generate_thoughts(task)
        if self._thinking_done(task):
            task.system_prompt_name = "role"
        return task

    async def async_think(self, task: ThinkingTask) -> ThinkingTask:
        """Async mirror of ``think``: step 2 is ``await
        self._agenerate_thoughts(task)``."""
        if task.system_prompt_name == "role":
            return task
        await self._agenerate_thoughts(task)
        if self._thinking_done(task):
            task.system_prompt_name = "role"
        return task

    def act(self, task: ThinkingTask) -> ThinkingTask:
        """
        1. if ``task.system_prompt_name != "role"``: return ``task``
        2. return ``super().act(task)`` (``BasicAgent``: one engine call with
           ``response_schema``, ``generated_response`` set, task complete)
        """
        if task.system_prompt_name != "role":
            return task
        return super().act(task)

    async def async_act(self, task: ThinkingTask) -> ThinkingTask:
        """Async mirror of ``act`` (``await super().async_act(task)``)."""
        if task.system_prompt_name != "role":
            return task
        return await super().async_act(task)

    # ------------------------------------------------------------------ #
    # Rendering (concrete)
    # ------------------------------------------------------------------ #
    def _render_task_messages(self, task: ThinkingTask) -> list[dict[str, str]]:
        """
        Rebuilt on every call (no build-once guard; this family never retries).
        The only call with no directive is a reply with no thoughts, which is
        exactly ``BasicAgent``'s message. ``additional_messages`` passed to
        ``render_task`` extend the returned list for that call only and do not
        persist to later calls (the list is rebuilt each call), unlike the
        build-once families.

        1. ``block = self._render_task_block(task)``
        2. ``thoughts = self._render_thoughts_message(task)``
        3. thoughts present: user ``block``, the thoughts message, user directive
        4. no thoughts, reply phase: user ``block`` alone
        5. no thoughts, thinking phase: user ``block``, a blank line, directive
        6. ``task.task_messages = <the list>``; return that same list object
        """
        block = self._render_task_block(task)
        thoughts = self._render_thoughts_message(task)
        if thoughts:
            messages = [
                {"role": "user", "content": block},
                *thoughts,
                {"role": "user", "content": self._render_directive(task)},
            ]
        elif task.system_prompt_name == "role":
            messages = [{"role": "user", "content": block}]
        else:
            directive = self._render_directive(task)
            messages = [{"role": "user", "content": f"{block}\n\n{directive}"}]
        task.task_messages = messages
        return messages

    def _render_thoughts_message(self, task: ThinkingTask) -> list[dict[str, str]]:
        """
        1. if ``task.thoughts`` is empty: return ``[]``
        2. return one assistant message holding
           ``render_thoughts_block(self._render_thoughts(task))``
        """
        if not task.thoughts:
            return []
        body = self._render_thoughts(task)
        return [{"role": "assistant", "content": render_thoughts_block(body)}]

    def _render_directive(self, task: ThinkingTask) -> str:
        """Generic closing instruction; override to vary per step.

        The text depends on the phase and on what the model can see: the
        runtime context (``task.context``) and the thoughts so far. The default
        rendering calls it for every shape except a reply with no thoughts, so
        the two reply branches without thoughts only serve overrides.

        1. ``seen`` names what precedes the directive: "RUNTIME CONTEXT" when
           ``task.context`` is non-empty, "THOUGHTS" when ``task.thoughts`` is
           non-empty (joined with "and")
        2. reply phase (``task.system_prompt_name == "role"``): "Given the
           above <seen>, complete the USER TASK PROMPT."; with nothing seen,
           "Complete the USER TASK PROMPT."
        3. thinking phase: "think about" the USER TASK PROMPT as preparation
           for the final reply, not the reply itself ("continue thinking
           about" once thoughts exist), led by "Given the above <seen>," when
           anything is seen
        """
        seen = " and ".join(
            label
            for label, present in (("RUNTIME CONTEXT", task.context), ("THOUGHTS", task.thoughts))
            if present
        )
        if task.system_prompt_name == "role":
            if not seen:
                return "Complete the USER TASK PROMPT."
            return f"Given the above {seen}, complete the USER TASK PROMPT."
        verb = "continue thinking about" if task.thoughts else "think about"
        tail = "the USER TASK PROMPT as preparation for the final reply, not the reply itself."
        if not seen:
            return f"{verb.capitalize()} {tail}"
        return f"Given the above {seen}, {verb} {tail}"

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
