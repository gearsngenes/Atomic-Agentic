"""
ScratchpadAgent: user-authored, fixed-count thinking strategy.

``ScratchpadAgent`` is the one concrete ``ThinkingAgent`` shipped in this
release. Each thinking round is one LLM call producing one thought (free text,
or a structured value when ``thinking_schema`` is set); after the rounds the
agent replies as a ``BasicAgent`` would. With ``stop_early`` the model may end
thinking before the cap by returning ``more_thinking_required: false``.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Callable, Mapping, Optional

from .thinking import ThinkingAgent
from ..constants.agents import (
    STOP_EARLY_PROMPT_SECTION,
    STOP_EARLY_UNSAFE_SCHEMA_KEYS,
    THINKING_INSTRUCTIONS_END,
    THINKING_INSTRUCTIONS_START,
)
from ..exceptions import AgentError
from ..llm.base import LLMEngine
from ..models.agents.prompts import PromptConfig
from ..models.agents.records import LLMRecord
from ..models.agents.tasks import ScratchpadTask
from ..models.parameters import ParamSpec
from ..utils.agents import build_thinking_step_schema, normalize_prompt_config
from ..utils.parameters import (
    apply_parameter_reports,
    build_parameter_reports,
    insert_by_category,
)
from ..utils.rendering import stringify_value


class ScratchpadAgent(ThinkingAgent):
    """
    N rounds of notes, then the reply.

    Two phases: thinking (system prompt ``"thinking"``, the author's
    ``thinking_instructions``) then reply (``"role"``). The thinker does not
    see the role prompt; an author who wants the persona there writes it into
    ``thinking_instructions``, ideally through a placeholder shared with
    ``role_prompt``. ``thinking_instructions`` placeholders join the agent's
    signature, reconciled with ``role_prompt``'s (role wins on a
    compatible-but-not-identical overlap; a true collision raises).

    ``thinking_rounds`` is the exact number of thinking calls, or the cap when
    ``stop_early`` is on. ``stop_early`` makes every thinking call return
    ``{thought, more_thinking_required}``; thinking ends at the first
    ``more_thinking_required: false`` or at the cap, and round 1 always runs
    when ``thinking_rounds >= 1``. Both are fixed at construction.

    Limit with ``stop_early``: ``thinking_schema`` must not use ``$defs``,
    ``definitions`` or ``$ref`` at its root (the schema is nested under
    ``thought``, which would break their pointers). A nested ref that points
    back into the schema is not detected; the provider reports it.
    """

    THINKING_PROMPT_NAME = "thinking"

    _TASK_CLS = ScratchpadTask
    _INITIAL_PROMPT_NAME = THINKING_PROMPT_NAME

    DEFAULT_THINKING_PROMPT: str = (
        "You are a thinking assistant that produces various thoughts, "
        "questions, and task-enhancement clarifications to describe or "
        "enhance a user task prompt"
    )

    def __init__(
        self,
        name: str,
        namespace: str,
        description: str,
        llm_engine: LLMEngine,
        role_prompt: str | PromptConfig | None = None,
        thinking_instructions: str | PromptConfig | None = None,
        record_enabled: bool = True,
        *,
        pre_invoke: Optional[Callable | Any] = None,
        post_invoke: Optional[Callable | Any] = None,
        bind_response_to: Optional[str | Iterable[str] | Mapping[Any, str]] = None,
        records_window: Optional[int] = None,
        response_schema: dict[str, Any] | None = None,
        thinking_schema: dict[str, Any] | None = None,
        thinking_rounds: int,
        stop_early: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        thinking_instructions : str | PromptConfig | None
            The thinking phase's entire system message (framed by markers and
            followed by a format section when ``stop_early`` is on). Defaults
            to ``DEFAULT_THINKING_PROMPT`` when omitted or blank. Its
            ``{placeholder}``s become agent inputs.
        thinking_schema : dict[str, Any] | None
            Output structure of each thought; ``None`` keeps thinking free
            text (or a string ``thought`` under ``stop_early``). Independent of
            ``response_schema``, which applies to the reply only.
        thinking_rounds : int
            Required. Exact thinking-call count, or the cap under
            ``stop_early``; ``0`` skips thinking.
        stop_early : bool
            Let the model end thinking before the cap.

        Everything before ``super().__init__`` runs first because
        ``_merge_prompt_parameters`` (called inside it) reads
        ``self._thinking_config``.
        """
        # 1. Validate construction knobs.
        if thinking_schema is not None and not isinstance(thinking_schema, Mapping):
            raise AgentError(
                f"{type(self).__name__}.thinking_schema must be a dict/Mapping "
                f"or None, got {type(thinking_schema).__name__}."
            )
        if type(thinking_rounds) is not int or thinking_rounds < 0:
            raise AgentError(
                f"{type(self).__name__}.thinking_rounds must be a concrete int "
                f">= 0, got {thinking_rounds!r}."
            )
        if type(stop_early) is not bool:
            raise AgentError(
                f"{type(self).__name__}.stop_early must be a bool, got "
                f"{type(stop_early).__name__}."
            )
        if stop_early and thinking_schema is not None:
            unsafe = [key for key in STOP_EARLY_UNSAFE_SCHEMA_KEYS if key in thinking_schema]
            if unsafe:
                raise AgentError(
                    f"{type(self).__name__}.thinking_schema must not use {unsafe} "
                    "at its root when stop_early is on (it is nested under "
                    "'thought', which would break those references)."
                )

        # 2. Store the thinking state the merge hook and hooks read.
        self._thinking_config = normalize_prompt_config(
            thinking_instructions,
            default_template=self.DEFAULT_THINKING_PROMPT,
            provided_description="Thinking instructions",
            default_description="Default thinking instructions.",
            error_label="thinking_instructions",
        )
        self._thinking_schema = thinking_schema
        self._thinking_rounds = thinking_rounds
        self._stop_early = stop_early

        # 3. Construct through BasicAgent, then register the thinking prompt.
        super().__init__(
            name=name,
            namespace=namespace,
            description=description,
            llm_engine=llm_engine,
            role_prompt=role_prompt,
            record_enabled=record_enabled,
            pre_invoke=pre_invoke,
            post_invoke=post_invoke,
            bind_response_to=bind_response_to,
            records_window=records_window,
            response_schema=response_schema,
        )
        self._system_prompts[self.THINKING_PROMPT_NAME] = self._thinking_config

    # ------------------------------------------------------------------ #
    # Properties (read-only)
    # ------------------------------------------------------------------ #
    @property
    def thinking_schema(self) -> dict[str, Any] | None:
        """The author's thinking schema, unwrapped (not the ``stop_early``
        wrapper). Frozen at construction."""
        return self._thinking_schema

    @property
    def thinking_rounds(self) -> int:
        """Exact thinking-call count, or the cap under ``stop_early``."""
        return self._thinking_rounds

    @property
    def stop_early(self) -> bool:
        """Whether the model may end thinking before ``thinking_rounds``."""
        return self._stop_early

    # ------------------------------------------------------------------ #
    # Signature seam
    # ------------------------------------------------------------------ #
    def _merge_prompt_parameters(self, role_params: list[ParamSpec]) -> list[ParamSpec]:
        """
        Reconcile the role prompt's placeholders with the thinking prompt's.

        1. build reports with the role prompt as the priority source
        2. raise on a true collision, warn on a compatible-but-not-identical
           overlap (role wins)
        3. return the reconciled parameters in signature order
        """
        reports = build_parameter_reports(
            [role_params, list(self._thinking_config.parameters)]
        )
        # stacklevel 5: apply_parameter_reports <- here <- BasicAgent.__init__
        # <- ScratchpadAgent.__init__ <- the code constructing the agent.
        constructed = apply_parameter_reports(
            reports,
            ("role_prompt", "thinking_instructions"),
            error_cls=AgentError,
            stacklevel=5,
        )
        return insert_by_category([], constructed)

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
           the author's ``thinking_schema``, or its ``stop_early`` wrapper
        2. unwrap the result under ``stop_early`` (setting
           ``task.more_thinking_required``)
        3. append the thought (a ``str`` stripped, anything else as-is)
        """
        messages = self.render_task(task)
        engine_result = self._llm_engine.invoke({
            "messages": messages,
            "output_structure": (
                build_thinking_step_schema(self._thinking_schema)
                if self._stop_early else self._thinking_schema
            ),
        })
        task.llm_records.append(LLMRecord(
            messages=list(task.task_messages),
            llm_result=engine_result,
            system_prompt_name=task.system_prompt_name,
        ))

        raw = engine_result.result
        if self._stop_early:
            thought = raw["thought"]
            task.more_thinking_required = raw["more_thinking_required"]
        else:
            thought = raw
        task.thoughts.append(thought.strip() if isinstance(thought, str) else thought)

    async def _agenerate_thoughts(self, task: ScratchpadTask) -> None:
        """Async twin of ``_generate_thoughts`` (awaits ``async_invoke``)."""
        messages = self.render_task(task)
        engine_result = await self._llm_engine.async_invoke({
            "messages": messages,
            "output_structure": (
                build_thinking_step_schema(self._thinking_schema)
                if self._stop_early else self._thinking_schema
            ),
        })
        task.llm_records.append(LLMRecord(
            messages=list(task.task_messages),
            llm_result=engine_result,
            system_prompt_name=task.system_prompt_name,
        ))

        raw = engine_result.result
        if self._stop_early:
            thought = raw["thought"]
            task.more_thinking_required = raw["more_thinking_required"]
        else:
            thought = raw
        task.thoughts.append(thought.strip() if isinstance(thought, str) else thought)

    def _render_thoughts(self, task: ScratchpadTask) -> str:
        """Each thought as its string (a ``str`` as-is, anything else as JSON),
        unnumbered, separated by ``---``."""
        return "\n\n---\n\n".join(
            t if isinstance(t, str) else stringify_value(t, mode="j")
            for t in task.thoughts
        )

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #
    def _render_system_message(self, task: ScratchpadTask) -> list[dict[str, str]]:
        """
        1. ``stop_early`` off, or the reply phase: the inherited message
        2. otherwise the rendered instructions between the instruction markers,
           followed by the early-stop format section

        ``stop_early`` is fixed at construction, so the agent attribute is read
        directly; nothing is snapshotted on the task.
        """
        messages = super()._render_system_message(task)
        if not self._stop_early or task.system_prompt_name == "role":
            return messages
        body = messages[0]["content"]
        content = (
            f"{THINKING_INSTRUCTIONS_START}\n{body}\n{THINKING_INSTRUCTIONS_END}"
            f"\n\n{STOP_EARLY_PROMPT_SECTION}"
        )
        return [{"role": "system", "content": content}]

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict:
        """Diagnostic snapshot including this agent's thinking knobs."""
        d = super().to_dict()
        d["thinking_instructions"] = self._thinking_config.template
        d["thinking_schema"] = self._thinking_schema
        d["thinking_rounds"] = self._thinking_rounds
        d["stop_early"] = self._stop_early
        return d
