"""
ScratchpadAgent: user-authored, fixed-count thinking strategy.

``ScratchpadAgent`` is the one concrete ``ThinkingAgent`` shipped in this
release. Each thinking round is one LLM call producing one free-text thought;
after the rounds the agent replies as a ``BasicAgent`` would. With ``stop_early`` the model may end
thinking before the cap by returning ``more_thinking_required: false``.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable
from typing import Any, Callable, Mapping, Optional

from .thinking import ThinkingAgent
from ..constants.agents import (
    DEFAULT_INSTRUCTIONS,
    DEFAULT_PRE_INVOKE,
    DEFAULT_THINKING_INSTRUCTIONS,
    STOP_EARLY_PROMPT_SECTION,
    STOP_EARLY_SCHEMA_TEMPLATE,
)
from ..core.atomic_invokable import AtomicInvokable
from ..exceptions import AgentError, AgentInvocationError
from ..llm.base import LLMEngine
from ..models.agents.records import LLMRecord
from ..models.agents.tasks import ScratchpadTask
from ..tools import to_invokable
from ..utils.rendering import frame_content


class ScratchpadAgent(ThinkingAgent):
    """
    N rounds of notes, then the reply.

    Two phases: thinking (the author's ``thinking_instructions`` as the system
    message) then reply (the ``instructions``). The thinker does not
    see the instructions; an author who wants the persona there writes it into
    ``thinking_instructions``, ideally through a placeholder shared with
    ``instructions``. Both slots accept a string, a callable or any
    ``AtomicInvokable`` and are coerced through ``to_invokable``; their
    placeholders join the agent's signature through the parameter sources
    (``[instructions, thinking_instructions]``: the instructions slot wins on a
    compatible-but-not-identical overlap, a true collision raises).

    Each slot is rendered once per invoke, at the end of ``pre_process``, into
    ``task.instructions`` and ``task.thinking_instructions``; the system message
    reads those. The thinking slot is skipped when ``thinking_rounds`` is 0 at
    pre-process time, so raising ``thinking_rounds`` from 0 during that same run
    leaves the thinking message empty.

    ``thinking_rounds`` is the exact number of thinking calls, or the cap when
    ``stop_early`` is on. ``stop_early`` makes every thinking call return
    ``{thought, more_thinking_required}`` (a string thought and a boolean);
    thinking ends at the first ``more_thinking_required: false`` or at the cap,
    and round 1 always runs when ``thinking_rounds >= 1``.

    ``stop_early`` is fixed at construction. ``thinking_rounds`` is a behavioral
    knob and may be reassigned (validated like the constructor argument); a call
    already running reads the new value at its next ``_thinking_done`` check.
    """

    _TASK_CLS = ScratchpadTask

    def __init__(
        self,
        name: str,
        description: str,
        llm_engine: LLMEngine,
        instructions: str | Callable[..., Any] | AtomicInvokable = DEFAULT_INSTRUCTIONS,
        thinking_instructions: str | Callable[..., Any] | AtomicInvokable = DEFAULT_THINKING_INSTRUCTIONS,
        pre_invoke: str | Callable[..., Any] | AtomicInvokable = DEFAULT_PRE_INVOKE,
        post_invoke: Optional[str | Callable[..., Any] | AtomicInvokable] = None,
        inputs_to_context: Iterable[str] | Mapping[str, str] = (),
        response_to_context: str | Iterable[str] | Mapping[Any, str] = (),
        response_to_post: str | Iterable[str] | Mapping[Any, str] = (),
        use_runtime_context: bool = False,
        record_enabled: bool = True,
        records_window: Optional[int] = None,
        *,
        response_schema: dict[str, Any] | None = None,
        thinking_rounds: int,
        stop_early: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        thinking_instructions : str | Callable | AtomicInvokable
            The thinking phase's entire system message (framed by markers and
            followed by a format section when ``stop_early`` is on). Defaults
            to ``DEFAULT_THINKING_INSTRUCTIONS``; ``None`` raises ``TypeError``
            and a blank string ``ValueError``. Its parameters (a string's
            ``{placeholder}``s) become agent inputs.
        thinking_rounds : int
            Required. Exact thinking-call count, or the cap under
            ``stop_early``; ``0`` skips thinking.
        stop_early : bool
            Let the model end thinking before the cap.

        1. validate ``stop_early`` and set ``thinking_rounds``.
        2. coerce the thinking slot, set before ``super().__init__`` because
           ``_extra_parameter_sources``, called inside it, reads it.
        3. ``super().__init__`` forwards ``instructions`` (``BasicAgent`` builds
           that slot).
        """
        # 1. Validate construction knobs (thinking_rounds through its setter).
        if type(stop_early) is not bool:
            raise AgentError(
                f"{type(self).__name__}.stop_early must be a bool, got "
                f"{type(stop_early).__name__}."
            )
        self.thinking_rounds = thinking_rounds

        # 2. Store the thinking state the parameter-source hook and hooks read.
        self._thinking_instructions = to_invokable(
            thinking_instructions,
            name=f"{name}_thinking_instructions",
            description=f"Thinking instructions of agent {name}",
        )
        self._stop_early = stop_early
        # Built once; the wrapper is the only structured thinking output.
        self._thinking_output = (
            copy.deepcopy(STOP_EARLY_SCHEMA_TEMPLATE) if stop_early else None
        )

        # 3. Construct through BasicAgent.
        super().__init__(
            name=name,
            description=description,
            llm_engine=llm_engine,
            instructions=instructions,
            record_enabled=record_enabled,
            pre_invoke=pre_invoke,
            post_invoke=post_invoke,
            inputs_to_context=inputs_to_context,
            response_to_context=response_to_context,
            response_to_post=response_to_post,
            use_runtime_context=use_runtime_context,
            records_window=records_window,
            response_schema=response_schema,
        )

    # ------------------------------------------------------------------ #
    # Properties (read-only)
    # ------------------------------------------------------------------ #
    @property
    def thinking_rounds(self) -> int:
        """Exact thinking-call count, or the cap under ``stop_early``. Mutable."""
        return self._thinking_rounds

    @thinking_rounds.setter
    def thinking_rounds(self, value: int) -> None:
        if type(value) is not int or value < 0:
            raise AgentError(
                f"{type(self).__name__}.thinking_rounds must be a concrete int "
                f">= 0, got {value!r}."
            )
        self._thinking_rounds = value

    @property
    def stop_early(self) -> bool:
        """Whether the model may end thinking before ``thinking_rounds``."""
        return self._stop_early

    @property
    def thinking_instructions(self) -> AtomicInvokable:
        """The thinking instructions slot invokable."""
        return self._thinking_instructions

    # ------------------------------------------------------------------ #
    # Parameter sources
    # ------------------------------------------------------------------ #
    def _extra_parameter_sources(self) -> list[AtomicInvokable]:
        """``[instructions, thinking_instructions]``: the instructions slot
        wins an overlap."""
        return [*super()._extra_parameter_sources(), self._thinking_instructions]

    # ------------------------------------------------------------------ #
    # Strategy hooks
    # ------------------------------------------------------------------ #
    def _thinking_done(self, task: ScratchpadTask) -> bool:
        """The round count is reached, or the model has ended thinking."""
        return len(task.thoughts) >= self._thinking_rounds or not task.more_thinking_required

    def _generate_thoughts(self, task: ScratchpadTask) -> None:
        """
        One thinking round.

        1. render, call the engine, record the call; the output structure is
           ``None`` (free text), or the ``stop_early`` wrapper
        2. under ``stop_early``, check the reply is ``{thought: str,
           more_thinking_required: bool}`` (``AgentInvocationError`` if not),
           unwrap it and set ``task.more_thinking_required``
        3. append the stripped thought
        """
        messages = self.render_task(task)
        engine_result = self._llm_engine.invoke({
            "messages": messages,
            "output_structure": self._thinking_output,
        })
        task.llm_records.append(LLMRecord(
            messages=list(task.task_messages),
            llm_result=engine_result,
        ))

        thought = engine_result.result
        if self._stop_early:
            if not (
                isinstance(thought, Mapping)
                and isinstance(thought.get("thought"), str)
                and isinstance(thought.get("more_thinking_required"), bool)
            ):
                raise AgentInvocationError(
                    f"{self.full_name}: a stop_early thinking reply must be an object "
                    "with a string 'thought' and a boolean 'more_thinking_required', "
                    f"got {thought!r}."
                )
            task.more_thinking_required = thought["more_thinking_required"]
            thought = thought["thought"]
        task.thoughts.append(thought.strip())

    async def _agenerate_thoughts(self, task: ScratchpadTask) -> None:
        """Async twin of ``_generate_thoughts`` (awaits ``async_invoke``)."""
        messages = self.render_task(task)
        engine_result = await self._llm_engine.async_invoke({
            "messages": messages,
            "output_structure": self._thinking_output,
        })
        task.llm_records.append(LLMRecord(
            messages=list(task.task_messages),
            llm_result=engine_result,
        ))

        thought = engine_result.result
        if self._stop_early:
            if not (
                isinstance(thought, Mapping)
                and isinstance(thought.get("thought"), str)
                and isinstance(thought.get("more_thinking_required"), bool)
            ):
                raise AgentInvocationError(
                    f"{self.full_name}: a stop_early thinking reply must be an object "
                    "with a string 'thought' and a boolean 'more_thinking_required', "
                    f"got {thought!r}."
                )
            task.more_thinking_required = thought["more_thinking_required"]
            thought = thought["thought"]
        task.thoughts.append(thought.strip())

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #
    def _render_instructions(self, task: ScratchpadTask) -> None:
        """Render both slots into the task.

        1. ``super()._render_instructions(task)`` (the main instructions).
        2. ``thinking_rounds == 0``: return (thinking never runs).
        3. Invoke the thinking slot once with only its own parameter names;
           a non-``str`` result raises ``TypeError``; store the text in
           ``task.thinking_instructions``.
        """
        super()._render_instructions(task)
        if self._thinking_rounds == 0:
            return
        slot = self._thinking_instructions
        inputs = {
            p.name: task.agentic_inputs[p.name]
            for p in slot.parameters if p.name in task.agentic_inputs
        }
        text = slot.invoke(inputs).result
        if not isinstance(text, str):
            raise TypeError(
                f"{self.full_name}: thinking instructions ({slot.name!r}) returned "
                f"{type(text).__name__}; thinking instructions must render to a str."
            )
        task.thinking_instructions = text

    async def _async_render_instructions(self, task: ScratchpadTask) -> None:
        """Async twin of ``_render_instructions`` (awaits ``async_invoke``)."""
        await super()._async_render_instructions(task)
        if self._thinking_rounds == 0:
            return
        slot = self._thinking_instructions
        inputs = {
            p.name: task.agentic_inputs[p.name]
            for p in slot.parameters if p.name in task.agentic_inputs
        }
        text = (await slot.async_invoke(inputs)).result
        if not isinstance(text, str):
            raise TypeError(
                f"{self.full_name}: thinking instructions ({slot.name!r}) returned "
                f"{type(text).__name__}; thinking instructions must render to a str."
            )
        task.thinking_instructions = text

    def _render_system_message(self, task: ScratchpadTask) -> list[dict[str, str]]:
        """
        1. thinking finished: the inherited (instructions) message
        2. otherwise ``task.thinking_instructions``
        3. with ``stop_early``: those between the instruction markers, followed
           by the early-stop format section

        ``stop_early`` is fixed at construction, so the agent attribute is read
        directly; nothing is snapshotted on the task.
        """
        if task.thinking_complete:
            return super()._render_system_message(task)
        content = task.thinking_instructions
        if self._stop_early:
            content = (
                frame_content(content, "USER-DEFINED THINKING INSTRUCTIONS")
                + "\n\n"
                + STOP_EARLY_PROMPT_SECTION
            )
        return [{"role": "system", "content": content}]

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict:
        """Diagnostic snapshot including this agent's thinking knobs."""
        d = super().to_dict()
        d["thinking_instructions"] = self._thinking_instructions.to_dict()
        d["thinking_rounds"] = self._thinking_rounds
        d["stop_early"] = self._stop_early
        return d
