"""
ReActAgent: Iterative LLM Actor with Per-Step Reactive Planning

Generates, resolves, and dispatches exactly one registered-tool call per
round via provider-native structured output (``REACT_OUTPUT_SCHEMA``),
observes the result, and repeats until the model calls the registered
``return`` tool. Design lineage: mirrors ``PlanActAgent``'s own
``$name``-sigil machinery (``utils/dag.py``, reused directly) collapsed to
a single call per round instead of a whole batched plan -- the per-step
sibling to ``PlanActAgent``'s one-shot multi-call plan. See
``agents/planact.py``'s own module docstring for the shared design lineage
this mirrors.

Distinguishing features relative to ``PlanActAgent``/``DagAgent``:
- ``return`` is an ordinary, really-dispatched tool call (``return_tool``,
  registered under its own bare name) -- never a synthesized,
  non-dispatched ``RETURN_ALIAS`` sentinel, and ``act()``/``async_act()``
  never gate *dispatch* on ``is_dispatched_call`` (every call, including
  ``return``, is always actually invoked). That claim is scoped to dispatch
  specifically: ``RETURN_TOOL_NAME`` and ``RETURN_ALIAS`` are the identical
  literal string ``"return"`` (`constants/agents.py`), so anywhere else
  ``is_dispatched_call`` is consulted for *accounting* purposes (see
  ``build_result_from_record`` below), a real, successful ``return`` call
  is still indistinguishable from the sentinel and gets excluded --
  correctly, per ``ToolUsageRecord.call_count``'s own "non-return
  executions" contract, but via the same string coincidence, not a
  deliberate discriminator.
- ``think()`` owns validation AND argument resolution in one retry loop;
  ``prepare()`` is a documented no-op -- this family's ``think()`` and
  ``prepare()`` are logically inseparable (one call, resolved immediately).
- No batching, no gathering, no cascade machinery -- exactly one call is
  dispatched per round, and a tolerated failure (``fail_fast=False``) is
  simply information the model sees and redirects around next round.
"""

from __future__ import annotations

import copy
import json
from datetime import datetime
from typing import Any, Callable, ClassVar, Optional

from .json_tool_agent import JsonToolAgent
from .prompts import REACT_PROMPT
from .tools import make_dict, make_sequence, return_tool
from ..constants.agents import RETURN_TOOL_NAME
from ..core import AtomicInvokable
from ..llm.base import LLMEngine
from ..exceptions import ToolAgentError, ToolInvocationError
from ..mcp import MCPClientHub
from ..a2a import A2AClientHub, PyA2AtomicClient
from ..models.agents.blackboard_models import DagToolCall
from ..models.agents.tasks import ReActTask
from ..models.agents.records import AgentRecord, JsonToolAgentRecord, LLMRecord
from ..models.results.agents import JsonToolAgentResult, ToolUsageRecord
from ..utils.dag import (
    build_react_schema,
    is_dispatched_call,
    parse_react_call,
    render_cache_snapshot,
    render_completed_as_json,
    render_failed_as_json,
    resolve_call_args,
    validate_calls,
)


# --------------------------------------------------------------------------- #
# ReAct Agent
# --------------------------------------------------------------------------- #
class ReActAgent(JsonToolAgent):
    """
    Iterative agent with reactive step-by-step planning (ReAct-style
    architecture): one call generated, resolved, and dispatched per round,
    observed before the next round's decision.

    Advantages
    ~~~~~~~~~~
    - Fully adaptive: each step can react to prior tool results, including
      failures -- ``fail_fast=False`` (this family's own default) makes a
      failure simply a redirect signal, not an abort.
    - Schema-guaranteed shape: ``output_structure`` strict mode makes a
      malformed-shape generation structurally impossible.

    Limitations
    ~~~~~~~~~~~
    - Higher latency: one LLM call per step, no concurrency.
    - Step quality depends on the model's ability to select the next best
      action from the rendered history alone.
    """

    _ATOMIC_IMMUTABLE_TYPES: ClassVar[tuple[type, ...]] = (
        str, int, float, bool, complex, bytes, type(None),
    )

    # A class attribute (active before __init__ runs) — return_tool plus the
    # two composite-value-building utility tools this family's schema text
    # tells the model to use. Reserved so no caller can register a different
    # tool under any of these ids, or remove/replace them once seeded. No
    # get_item entry -- dropped (2026-09-26, user call): too niche relative
    # to make_sequence/make_dict/return, and its presence in AVAILABLE TOOLS
    # was observed distracting live generations in agent-orchestrating-agent
    # examples toward container-shaped detours a flat call sequence didn't
    # need.
    _RESERVED_TOOL_NAMES: ClassVar[frozenset[str]] = frozenset(
        {RETURN_TOOL_NAME, "make_sequence", "make_dict"}
    )

    def __init__(
        self,
        name: str,
        namespace: str,
        description: str,
        llm_engine: LLMEngine,
        context_enabled: bool = False,
        *,
        tool_calls_limit: Optional[int] = 25,
        regeneration_limit: int = 5,
        tool_concurrency_limit: Optional[int] = None,
        fail_fast: bool = False,
        response_preview_limit: Optional[int] = None,
        pre_invoke: Optional[AtomicInvokable | Callable[..., Any]] = None,
        post_invoke: Optional[AtomicInvokable | Callable[..., Any]] = None,
        post_result_key: Optional[str] = None,
        records_window: Optional[int] = None,
        tools: Optional[list[AtomicInvokable | Callable | MCPClientHub | A2AClientHub | PyA2AtomicClient]] = None,
        constants: Optional[list[Any]] = None,
        constant_aliases: Optional[list[Optional[str]]] = None,
        constant_descriptions: Optional[list[Optional[str]]] = None,
    ) -> None:
        """
        Every parameter forwards verbatim to ``JsonToolAgent.__init__`` --
        no ``extra_parameters`` keyword, matching that base class's own
        signature exactly (it accepts none). Two defaults deliberately
        diverge from ``JsonToolAgent``'s own: ``tool_calls_limit`` defaults
        to ``25`` (not ``None`` -- this family has no second round-ceiling
        knob the way ``DagAgent`` has ``planning_rounds_limit``, so an
        unbounded default would have zero structural backstop), and
        ``fail_fast`` defaults to ``False`` (not ``True`` -- redirecting on
        failure is this family's entire reason for existing; a caller who
        genuinely wants one failure to be fatal can still pass
        ``fail_fast=True`` explicitly).
        """
        super().__init__(
            name=name,
            namespace=namespace,
            description=description,
            llm_engine=llm_engine,
            context_enabled=context_enabled,
            tool_calls_limit=tool_calls_limit,
            regeneration_limit=regeneration_limit,
            tool_concurrency_limit=tool_concurrency_limit,
            fail_fast=fail_fast,
            response_preview_limit=response_preview_limit,
            pre_invoke=pre_invoke,
            post_invoke=post_invoke,
            post_result_key=post_result_key,
            records_window=records_window,
            tools=tools,
            constants=constants,
            constant_aliases=constant_aliases,
            constant_descriptions=constant_descriptions,
        )
        # Seeded directly, bypassing register_tool -- all three are reserved
        # (self._RESERVED_TOOL_NAMES), and register_tool now rejects any
        # attempt to register something under a reserved id, including this
        # one. return_tool's own bare name already equals RETURN_TOOL_NAME;
        # naming it explicitly here decouples seeding from that incidental
        # fact.
        self._seed_reserved_tool(return_tool, RETURN_TOOL_NAME)
        self._seed_reserved_tool(make_sequence, "make_sequence")
        self._seed_reserved_tool(make_dict, "make_dict")
        self._system_prompts["reason_then_act"] = REACT_PROMPT

    # ------------------------------------------------------------------ #
    # Shared per-invocation helpers
    # ------------------------------------------------------------------ #
    @classmethod
    def _copy_for_task_namespace(cls, value: Any) -> Any:
        """
        Own private copy, not shared on ``JsonToolAgent`` (matches this
        release's Stage A/B duplication posture; the lift is Pass 7's job).
        Returns ``value`` unchanged if it's a known atomic-immutable type;
        otherwise a deep copy, so a mutating call on a cached/constant
        value can never reach the real, shared object.
        """
        if isinstance(value, cls._ATOMIC_IMMUTABLE_TYPES):
            return value
        return copy.deepcopy(value)

    def _initialize_task(
        self,
        *,
        turns: list[AgentRecord],
        prompt: str,
        inputs: dict,
    ) -> ReActTask:
        """
        Build a bare ``ReActTask``, seed its ``cache`` with every visible
        prior turn's result under ``task_result_{i}``, and seed
        ``constant_values`` from every registered constant. Mirrors
        ``PlanActAgent._initialize_task`` exactly.
        """
        task = ReActTask(
            turns=turns, inputs=inputs, user_prompt=prompt, system_prompt_name="reason_then_act",
        )
        for turn in turns:
            task.cache[f"task_result_{self._turn_position(turn)}"] = self._copy_for_task_namespace(
                turn.generated_response
            )
        for spec in self._constants.values():
            task.constant_values[spec.name] = self._copy_for_task_namespace(spec.value)
        return task

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #
    def _render_current_task_message(self, task: ReActTask) -> dict[str, str]:
        """
        The "what is the task" user message. Unlike ``DagAgent`` (whose
        budget is a silent backstop, never shown), this family surfaces a
        live *remaining* count here -- recomputed fresh every round, since
        this method is re-invoked each time ``task.task_messages`` is
        rebuilt (every round). Mirrors ``PlanActAgent``'s own
        ``_render_current_task_message`` (which shows a static *total*,
        since it plans once) adapted for a figure that actually depletes
        round over round. Enforcement itself is still structural (the
        schema enum narrows to ``return``-only at the boundary, see
        ``_run_step_retry_loop``) -- this text is purely informational, so
        the model can see it coming rather than getting funneled into
        ``return`` with no warning.
        """
        content = f"CURRENT TASK:\n{task.user_prompt}"
        if self.tool_calls_limit is not None:
            remaining = self.tool_calls_limit - task.tool_calls_used
            content += f"\n\nTool calls remaining: {remaining} of {self.tool_calls_limit}."
        return {"role": "user", "content": content}

    def _render_task_messages(self, task: ReActTask) -> list[dict[str, str]]:
        """
        Build-once contract: returns ``task.task_messages`` as-is if
        already non-empty (cleared by ``think()`` at the end of the prior
        round). One synthesized snapshot per round -- banner / assistant
        "state so far" (completed calls, cached values, and any persistent
        failure history) / user instruction, framed differently after a
        tolerated failure.
        """
        if task.task_messages:
            return task.task_messages

        banner = self._render_current_task_message(task)

        completed_section = render_completed_as_json(task.completed) or "(nothing completed yet)"

        cache_snapshot = render_cache_snapshot(
            task.completed, task.cache, self._response_preview_limit
        )
        cache_section = f"\n\n{cache_snapshot}" if cache_snapshot else ""

        # Only the single most recent failure -- never the accumulated
        # history. Showing every past failure alongside a since-succeeded
        # retry would leave the model unable to tell what it's actually
        # reacting to right now; task.last_call_failed already answers
        # "is there anything to react to" precisely (False the instant a
        # later call succeeds), so the render must be gated on it, not on
        # whether failed_statements is merely non-empty.
        failed_section = ""
        if task.last_call_failed:
            failed_section = (
                f"\n\nYOUR LAST CALL FAILED:\n"
                f"{render_failed_as_json([task.failed_statements[-1]])}"
            )

        state_message = {
            "role": "assistant",
            "content": f"# STEPS COMPLETED SO FAR:\n{completed_section}{cache_section}{failed_section}",
        }

        base_instruction = (
            "Produce the NEXT BEST single tool call for the current task, "
            "or call the return tool if the task is complete."
        )
        instruction = (
            "Your last call failed -- see YOUR LAST CALL FAILED above for "
            f"why. Adjust your approach. {base_instruction}"
            if task.last_call_failed
            else base_instruction
        )

        task.task_messages = [banner, state_message, {"role": "user", "content": instruction}]
        return task.task_messages

    # ------------------------------------------------------------------ #
    # Generation (think())
    # ------------------------------------------------------------------ #
    def _process_generation_output(
        self, raw_output: dict[str, Any], task: ReActTask,
    ) -> tuple[DagToolCall, dict[str, Any]] | str:
        """
        Pure-computation validate-and-resolve callback for the step retry
        loop: parse, validate semantics + remaining tool-call budget, then
        attempt argument resolution -- the one failure category
        ``validate_calls`` structurally can't catch (a resolved value's
        real type mismatching the target tool's parameter contract).
        Returns ``(call, resolved_kwargs)`` on success, or a feedback
        string describing the problem on failure.
        """
        call = parse_react_call(raw_output)

        remaining_budget = (
            None if self._tool_calls_limit is None
            else self._tool_calls_limit - task.tool_calls_used
        )
        known_names = frozenset(task.cache) | frozenset(task.constant_values)
        issues = validate_calls([call], None, remaining_budget, known_names)

        if issues:
            issues_msg = "\n".join(f"{i + 1}. {m}" for i, m in enumerate(issues))
            # TODO(smoke-test aid, mirrors PlanActAgent's/DagAgent's own): remove once
            # cross-provider output_structure reliability is confirmed.
            print(f"[ReActAgent DEBUG] step rejected, issues:\n{issues_msg}")
            return issues_msg

        try:
            positional, keyword = resolve_call_args(call, {**task.cache, **task.constant_values})
            tool = self.get_tool(call.tool)
            resolved = tool._args_kwargs_to_dict(*positional, **keyword)
        except Exception as e:
            label = call.identifier if call.identifier is not None else "(unassigned)"
            return f"{label}: could not resolve or bind argument(s): {e!r}"

        return call, resolved

    def _run_step_retry_loop(self, *, task: ReActTask) -> tuple[DagToolCall, dict[str, Any]]:
        """
        Render, compute the schema's tool set, call the engine (with
        ``output_structure``), record the attempt, validate+resolve via
        ``_process_generation_output``, and retry with injected feedback on
        failure until success or the regeneration budget
        (``self._regeneration_limit``, tracked via
        ``task.regenerations_used``) is exhausted. Mirrors ``PlanActAgent``'s
        own smoke-test-aid debug print of every raw generated step (see
        ``_process_generation_output`` for the matching rejected-issues
        print).
        """
        additional_messages: list[dict[str, str]] = []

        while True:
            messages = self.render_task(task, additional_messages=additional_messages)

            tool_names = (
                [RETURN_TOOL_NAME]
                if (
                    self._tool_calls_limit is not None
                    and task.tool_calls_used >= self._tool_calls_limit
                )
                else self._toolbox.keys()
            )
            schema = build_react_schema(tool_names)

            engine_result = self._llm_engine.invoke(
                {"messages": messages, "output_structure": schema}
            )
            raw_output: dict[str, Any] = engine_result.result
            # TODO(smoke-test aid, mirrors PlanActAgent's/DagAgent's own): remove once
            # cross-provider output_structure reliability is confirmed.
            print(f"[ReActAgent DEBUG] generated step:\n{json.dumps(raw_output, indent=2)}")

            task.llm_records.append(LLMRecord(
                messages=list(task.task_messages),
                llm_result=engine_result,
                system_prompt_name=task.system_prompt_name,
            ))

            result = self._process_generation_output(raw_output, task)
            if isinstance(result, str):
                if task.regenerations_used >= self._regeneration_limit:
                    raise ToolAgentError(
                        f"{type(self).__name__}.{self.name}: regeneration budget "
                        f"exhausted after {task.regenerations_used + 1} attempt(s). "
                        f"Last feedback: {result}"
                    )
                additional_messages = [
                    {"role": "assistant", "content": json.dumps(raw_output)},
                    {"role": "user", "content": (
                        f"Your step could not be used:\n\n{result}\n\n"
                        "Produce a corrected step."
                    )},
                ]
                task.regenerations_used += 1
                continue

            return result

    async def _arun_step_retry_loop(self, *, task: ReActTask) -> tuple[DagToolCall, dict[str, Any]]:
        """Async mirror of ``_run_step_retry_loop``, using
        ``async_invoke`` for the engine call. Same regeneration-budget
        check, feedback-message construction, and smoke-test-aid debug
        print of every raw generated step."""
        additional_messages: list[dict[str, str]] = []

        while True:
            messages = self.render_task(task, additional_messages=additional_messages)

            tool_names = (
                [RETURN_TOOL_NAME]
                if (
                    self._tool_calls_limit is not None
                    and task.tool_calls_used >= self._tool_calls_limit
                )
                else self._toolbox.keys()
            )
            schema = build_react_schema(tool_names)

            engine_result = await self._llm_engine.async_invoke(
                {"messages": messages, "output_structure": schema}
            )
            raw_output: dict[str, Any] = engine_result.result
            # TODO(smoke-test aid, mirrors PlanActAgent's/DagAgent's own): remove once
            # cross-provider output_structure reliability is confirmed.
            print(f"[ReActAgent DEBUG] generated step:\n{json.dumps(raw_output, indent=2)}")

            task.llm_records.append(LLMRecord(
                messages=list(task.task_messages),
                llm_result=engine_result,
                system_prompt_name=task.system_prompt_name,
            ))

            result = self._process_generation_output(raw_output, task)
            if isinstance(result, str):
                if task.regenerations_used >= self._regeneration_limit:
                    raise ToolAgentError(
                        f"{type(self).__name__}.{self.name}: regeneration budget "
                        f"exhausted after {task.regenerations_used + 1} attempt(s). "
                        f"Last feedback: {result}"
                    )
                additional_messages = [
                    {"role": "assistant", "content": json.dumps(raw_output)},
                    {"role": "user", "content": (
                        f"Your step could not be used:\n\n{result}\n\n"
                        "Produce a corrected step."
                    )},
                ]
                task.regenerations_used += 1
                continue

            return result

    def think(self, task: ReActTask) -> ReActTask:
        """
        Generate and validate this round's single call, without dispatching
        it. Stashes the resolved decision onto ``task.generated_step``/
        ``task.resolved_args`` for ``act()`` to run.
        """
        call, resolved = self._run_step_retry_loop(task=task)
        task.generated_step = call
        task.resolved_args = resolved
        task.task_messages.clear()
        return task

    async def async_think(self, task: ReActTask) -> ReActTask:
        """Async mirror of ``think``, using ``_arun_step_retry_loop``."""
        call, resolved = await self._arun_step_retry_loop(task=task)
        task.generated_step = call
        task.resolved_args = resolved
        task.task_messages.clear()
        return task

    # ------------------------------------------------------------------ #
    # Prepare (no-op for this family)
    # ------------------------------------------------------------------ #
    def prepare(self, task: ReActTask) -> ReActTask:
        """
        Documented no-op -- all validation/resolution already happened
        inside ``think()``'s own retry loop; nothing is deferred to this
        hook for this family. Unlike ``DagAgent``/``PlanActAgent``, whose
        ``prepare()`` resolves a batch compiled by an earlier ``think()``
        call, this family's ``think()`` and ``prepare()`` are logically
        inseparable -- one call, resolved immediately.
        """
        return task

    async def async_prepare(self, task: ReActTask) -> ReActTask:
        """Direct passthrough -- ``prepare`` has no I/O of its own to
        justify a thread offload."""
        return self.prepare(task)

    # ------------------------------------------------------------------ #
    # Execute (act())
    # ------------------------------------------------------------------ #
    def _apply_call_result(
        self, task: ReActTask, call: DagToolCall, raw_result: Any,
    ) -> ReActTask:
        """
        Shared post-dispatch bookkeeping for ``act``/``async_act``.

        A failure with ``fail_fast=True`` raises immediately (preserving
        ``ToolInvocationError`` as-is, wrapping anything else in
        ``ToolAgentError``); with ``fail_fast=False`` it is recorded into
        ``task.failed_statements`` and ``task.last_call_failed`` is set, so
        the model sees it next round and redirects -- no cascade machinery
        of any kind. Every successful call, including a call to
        ``return_tool``, is unwrapped and recorded identically -- there is
        no synthesized, never-dispatched sentinel in this family to skip.
        """
        if isinstance(raw_result, BaseException):
            if self._fail_fast:
                if isinstance(raw_result, ToolInvocationError):
                    raise raw_result
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: tool call failed "
                    f"for {call.tool!r} (identifier={call.identifier!r}): {raw_result}"
                ) from raw_result

            call.exception = raw_result
            task.failed_statements.append(call)
            task.last_call_failed = True
        else:
            call.result = raw_result
            unwrapped = raw_result.result

            task.completed.append(call)
            if call.identifier is not None:
                task.cache[call.identifier] = unwrapped
            task.last_call_failed = False

            if call.tool == RETURN_TOOL_NAME:
                task.generated_response = unwrapped
                task.complete = True

        task.generated_step = None
        task.resolved_args = None
        return task

    def act(self, task: ReActTask) -> ReActTask:
        """
        Dispatch this round's one resolved call for real -- no gathering,
        no ``asyncio``/``run_coro_sync`` of any kind, since exactly one
        call is ever dispatched per round.
        """
        call = task.generated_step
        resolved = task.resolved_args
        tool = self.get_tool(call.tool)
        try:
            raw_result = tool.invoke(resolved)
        except BaseException as e:
            raw_result = e
        return self._apply_call_result(task, call, raw_result)

    async def async_act(self, task: ReActTask) -> ReActTask:
        """Async mirror of ``act``."""
        call = task.generated_step
        resolved = task.resolved_args
        tool = self.get_tool(call.tool)
        try:
            raw_result = await tool.async_invoke(resolved)
        except BaseException as e:
            raw_result = e
        return self._apply_call_result(task, call, raw_result)

    # ------------------------------------------------------------------ #
    # Record / result construction
    # ------------------------------------------------------------------ #
    def _build_record_from_task(
        self,
        task: ReActTask,
        turns: list[AgentRecord],
    ) -> JsonToolAgentRecord:
        """
        Assemble a completed ``JsonToolAgentRecord`` from a finished
        ``ReActTask``. Mirrors ``PlanActAgent._build_record_from_task``
        exactly -- no agent-level global blackboard to persist into.
        """
        prev = turns[-1] if turns else None
        return JsonToolAgentRecord(
            user_prompt=task.user_prompt,
            generated_response=task.generated_response,
            inputs=task.inputs,
            llm_records=tuple(task.llm_records),
            prev=prev,
            statements=tuple(task.completed),
            failed_statements=tuple(task.failed_statements),
            regenerations_used=task.regenerations_used,
        )

    def build_result_from_record(
        self,
        record: JsonToolAgentRecord,
        *,
        result: Any,
        started_at: datetime,
        ended_at: datetime,
    ) -> JsonToolAgentResult:
        """
        Construct this agent's ``JsonToolAgentResult`` envelope directly
        from a completed ``JsonToolAgentRecord``. Mirrors
        ``PlanActAgent.build_result_from_record`` exactly.
        """
        llm_token_usage = tuple(r.llm_result.token_usage for r in record.llm_records)
        llm_model_data = record.llm_records[-1].llm_result.model_data

        # is_dispatched_call excludes a real, successful `return` call here
        # too -- correct per ToolUsageRecord.call_count's own "non-return
        # executions" contract (return isn't meant to be counted as tool
        # usage for any family), but note this family's return call is a
        # genuine dispatch, excluded only because RETURN_TOOL_NAME and
        # RETURN_ALIAS are the identical string "return" -- the same
        # collision ReActTask.tool_calls_used's own docstring already
        # documents, not a second, independent mechanism.
        counts: dict[str, int] = {}
        for call in record.statements:
            if is_dispatched_call(call):
                counts[call.tool] = counts.get(call.tool, 0) + 1
        tool_usage = tuple(
            ToolUsageRecord(tool_name=name, call_count=count)
            for name, count in counts.items()
        )

        return self._make_result(
            result=result,
            started_at=started_at,
            ended_at=ended_at,
            result_cls=JsonToolAgentResult,
            llm_token_usage=llm_token_usage,
            llm_model_data=llm_model_data,
            tool_usage=tool_usage,
            failed_call_count=len(record.failed_statements),
            regenerations_used=record.regenerations_used,
        )
