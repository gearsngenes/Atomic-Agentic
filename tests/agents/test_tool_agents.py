from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping
import asyncio

import pytest

from atomic_agentic.agents.tool_agents import (
    _CACHE_TOKEN,
    _STEP_TOKEN,
    BlackboardSlot,
    ToolAgent,
    ToolAgentRunState,
    extract_dependencies,
    return_tool,
    PlanActAgent,
    PlannedStep,
    ReActAgent,
)
from atomic_agentic.core.Exceptions import (
    AgentError,
    ToolAgentError,
    ToolInvocationError,
    ToolRegistrationError,
)
from atomic_agentic.core.sentinels import NO_VAL
from atomic_agentic.agents.data_classes import AgentTurn, ToolAgentTurn
from atomic_agentic.engines.LLMEngines import LLMEngine
from atomic_agentic.tools import Tool


ROLE_TEMPLATE = "Tools:\n{TOOLS}\nLimit: {TOOL_CALLS_LIMIT}"


class EchoLLMEngine(LLMEngine):
    """Minimal deterministic LLMEngine used only to satisfy Agent construction."""

    def __init__(self, *, response: str = "{}", **kwargs: Any) -> None:
        super().__init__(
            name="echo_llm_engine",
            description="Echo LLM engine for ToolAgent tests.",
            **kwargs,
        )
        self.response = response
        self.calls: list[list[dict[str, str]]] = []

    def _build_provider_payload(
        self,
        messages: list[dict[str, str]],
        attachments: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Any]:
        self.calls.append([dict(message) for message in messages])
        return {"messages": messages}

    def _call_provider(self, payload: Any) -> str:
        return self.response

    def _extract_text(self, response: Any) -> str:
        return str(response)

    def _prepare_attachment(self, path: str) -> Mapping[str, Any]:
        return {"path": path}

    def _on_detach(self, meta: Mapping[str, Any]) -> None:
        return None

class ScriptedLLMEngine(LLMEngine):
    """Deterministic LLMEngine that returns one scripted text response per call."""

    def __init__(self, responses: list[str], **kwargs: Any) -> None:
        super().__init__(
            name="scripted_llm_engine",
            description="Scripted LLM engine for ToolAgent subclass tests.",
            **kwargs,
        )
        self.responses = list(responses)
        self.calls: list[list[dict[str, str]]] = []

    def _build_provider_payload(
        self,
        messages: list[dict[str, str]],
        attachments: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Any]:
        copied_messages = [dict(message) for message in messages]
        self.calls.append(copied_messages)
        return {"messages": copied_messages}

    def _call_provider(self, payload: Any) -> str:
        if not self.responses:
            raise RuntimeError("No scripted LLM responses remain.")
        return self.responses.pop(0)

    def _extract_text(self, response: Any) -> str:
        return str(response)

    def _prepare_attachment(self, path: str) -> Mapping[str, Any]:
        return {"path": path}

    def _on_detach(self, meta: Mapping[str, Any]) -> None:
        return None


class BadRepr:
    def __repr__(self) -> str:
        raise RuntimeError("repr failed")

    def __str__(self) -> str:
        return "fallback string value that is long"


def make_planact_agent(
    responses: list[str],
    *,
    context_enabled: bool = False,
    tool_calls_limit: int | None = None,
    peek_at_cache: bool = False,
    preview_limit: int | None = None,
    response_preview_limit: int | None = None,
    blackboard_preview_limit: int | None = None,
) -> PlanActAgent:
    agent = PlanActAgent(
        name="planact_agent",
        description="PlanAct agent under test.",
        llm_engine=ScriptedLLMEngine(responses),
        context_enabled=context_enabled,
        tool_calls_limit=tool_calls_limit,
        peek_at_cache=peek_at_cache,
        preview_limit=preview_limit,
        response_preview_limit=response_preview_limit,
        blackboard_preview_limit=blackboard_preview_limit,
    )
    register_math_tools(agent)  # type: ignore[arg-type]
    return agent


def make_react_agent(
    responses: list[str],
    *,
    context_enabled: bool = False,
    tool_calls_limit: int = 3,
    peek_at_cache: bool = False,
    preview_limit: int | None = None,
    response_preview_limit: int | None = None,
    blackboard_preview_limit: int | None = None,
) -> ReActAgent:
    agent = ReActAgent(
        name="react_agent",
        description="ReAct agent under test.",
        llm_engine=ScriptedLLMEngine(responses),
        context_enabled=context_enabled,
        tool_calls_limit=tool_calls_limit,
        peek_at_cache=peek_at_cache,
        preview_limit=preview_limit,
        response_preview_limit=response_preview_limit,
        blackboard_preview_limit=blackboard_preview_limit,
    )
    register_math_tools(agent)  # type: ignore[arg-type]
    return agent


def add(x: int, y: int) -> int:
    return x + y


def multiply(x: int, y: int) -> int:
    return x * y


def join_text(prefix: str, value: Any) -> str:
    return f"{prefix}:{value}"


def fail_tool() -> str:
    raise RuntimeError("intentional failure")


@dataclass(slots=True)
class ScriptedRunState(ToolAgentRunState):
    batches: list[list[dict[str, Any]]] = field(default_factory=list)
    batch_index: int = 0
    next_step_index: int = 0


class ScriptedToolAgent(ToolAgent[ScriptedRunState]):
    """Deterministic ToolAgent subclass for testing the base ToolAgent loop."""

    def __init__(
        self,
        *,
        script: list[list[dict[str, Any]]] | None = None,
        context_enabled: bool = False,
        tool_calls_limit: int | None = None,
        peek_at_cache: bool = False,
        preview_limit: int | None = None,
        response_preview_limit: int | None = None,
        blackboard_preview_limit: int | None = None,
    ) -> None:
        super().__init__(
            name="scripted_agent",
            description="Scripted ToolAgent for unit tests.",
            llm_engine=EchoLLMEngine(),
            role_prompt=ROLE_TEMPLATE,
            context_enabled=context_enabled,
            tool_calls_limit=tool_calls_limit,
            peek_at_cache=peek_at_cache,
            preview_limit=preview_limit,
            response_preview_limit=response_preview_limit,
            blackboard_preview_limit=blackboard_preview_limit,
        )
        self.script = script or []

    def set_script(self, script: list[list[dict[str, Any]]]) -> None:
        self.script = script

    def _initialize_run_state(
        self,
        *,
        messages: list[dict[str, str]],
    ) -> ScriptedRunState:
        total_steps = sum(len(batch) for batch in self.script)
        running_blackboard = [BlackboardSlot(step=index) for index in range(total_steps)]

        return ScriptedRunState(
            messages=[dict(message) for message in messages],
            cache_blackboard=list(self._blackboard),
            running_blackboard=running_blackboard,
            executed_steps=set(),
            prepared_steps=[],
            tool_calls_used=0,
            is_done=False,
            return_value=NO_VAL,
            batches=[[dict(call) for call in batch] for batch in self.script],
            batch_index=0,
            next_step_index=0,
        )

    def _prepare_next_batch(self, state: ScriptedRunState) -> ScriptedRunState:
        if state.batch_index >= len(state.batches):
            raise ToolAgentError("No scripted batches remain.")

        batch = state.batches[state.batch_index]
        prepared_steps: list[int] = []

        for call in batch:
            step = state.next_step_index
            if step >= len(state.running_blackboard):
                raise ToolAgentError("Scripted batch exceeded running blackboard size.")

            tool_name = call["tool"]
            args = call.get("args", {})

            self.get_tool(tool_name)

            slot = state.running_blackboard[step]
            slot.tool = tool_name
            slot.args = args
            slot.resolved_args = self._resolve_placeholders(args, state=state)
            slot.result = NO_VAL
            slot.error = NO_VAL

            prepared_steps.append(step)
            state.next_step_index += 1

        state.prepared_steps = prepared_steps
        state.batch_index += 1
        return state


class BadInitializeToolAgent(ScriptedToolAgent):
    def _initialize_run_state(self, *, messages: list[dict[str, str]]) -> Any:
        return {"bad": "state"}


class PendingPreparedToolAgent(ScriptedToolAgent):
    def _initialize_run_state(
        self,
        *,
        messages: list[dict[str, str]],
    ) -> ScriptedRunState:
        state = super()._initialize_run_state(messages=messages)
        state.prepared_steps = [0]
        return state


def make_agent(
    *,
    context_enabled: bool = False,
    tool_calls_limit: int | None = None,
    peek_at_cache: bool = False,
    preview_limit: int | None = None,
    response_preview_limit: int | None = None,
    blackboard_preview_limit: int | None = None,
) -> ScriptedToolAgent:
    return ScriptedToolAgent(
        context_enabled=context_enabled,
        tool_calls_limit=tool_calls_limit,
        peek_at_cache=peek_at_cache,
        preview_limit=preview_limit,
        response_preview_limit=response_preview_limit,
        blackboard_preview_limit=blackboard_preview_limit,
    )


def register_math_tools(agent: ScriptedToolAgent) -> dict[str, str]:
    return {
        "add": agent.register(add, namespace="tests"),
        "multiply": agent.register(multiply, namespace="tests"),
        "join_text": agent.register(join_text, namespace="tests"),
        "fail_tool": agent.register(fail_tool, namespace="tests"),
    }


def prepared_slot(step: int, tool: str, args: Mapping[str, Any]) -> BlackboardSlot:
    slot = BlackboardSlot(step=step)
    slot.tool = tool
    slot.args = dict(args)
    slot.resolved_args = dict(args)
    return slot


def executed_slot(step: int, result: Any, *, tool: str = "Tool.tests.add") -> BlackboardSlot:
    slot = BlackboardSlot(step=step)
    slot.tool = tool
    slot.args = {}
    slot.resolved_args = {}
    slot.result = result
    return slot


def make_state(
    *,
    running: list[BlackboardSlot] | None = None,
    cache: list[BlackboardSlot] | None = None,
    prepared_steps: list[int] | None = None,
    tool_calls_used: int = 0,
) -> ScriptedRunState:
    return ScriptedRunState(
        messages=[{"role": "user", "content": "run"}],
        cache_blackboard=cache or [],
        running_blackboard=running or [],
        executed_steps=set(),
        prepared_steps=prepared_steps or [],
        tool_calls_used=tool_calls_used,
        is_done=False,
        return_value=NO_VAL,
        batches=[],
        batch_index=0,
        next_step_index=0,
    )


class TestBlackboardSlot:
    def test_empty_slot_state(self) -> None:
        slot = BlackboardSlot(step=0)

        assert slot.is_empty() is True
        assert slot.is_prepared() is False
        assert slot.is_executed() is False

    def test_prepared_slot_state(self) -> None:
        slot = prepared_slot(0, "Tool.tests.add", {"x": 1, "y": 2})

        assert slot.is_empty() is False
        assert slot.is_prepared() is True
        assert slot.is_executed() is False

    def test_executed_slot_state(self) -> None:
        slot = executed_slot(0, 3)

        assert slot.is_empty() is False
        assert slot.is_prepared() is False
        assert slot.is_executed() is True

    def test_error_slot_is_not_executed_without_result(self) -> None:
        slot = prepared_slot(0, "Tool.tests.fail_tool", {})
        slot.error = RuntimeError("boom")

        assert slot.is_executed() is False

    def test_to_dict_shape(self) -> None:
        slot = executed_slot(0, 3)
        data = slot.to_dict()

        assert data == {
            "step": 0,
            "tool": "Tool.tests.add",
            "args": {},
            "resolved_args": {},
            "result": 3,
            "error": NO_VAL,
            "completed": True,
        }


class TestToolAgentConstruction:
    def test_valid_construction_auto_registers_return_tool(self) -> None:
        agent = make_agent()

        assert agent.has_tool(return_tool.full_name)
        assert agent.get_tool(return_tool.full_name) is return_tool

    def test_role_prompt_renders_tools_and_limit(self) -> None:
        agent = make_agent(tool_calls_limit=3)
        keys = register_math_tools(agent)

        rendered = agent.role_prompt

        assert "Limit: 3" in rendered
        assert keys["add"] in rendered
        assert "{TOOLS}" not in rendered
        assert "{TOOL_CALLS_LIMIT}" not in rendered

    def test_role_prompt_missing_tools_placeholder_raises(self) -> None:
        with pytest.raises(ToolAgentError, match="TOOLS"):
            ScriptedToolAgent._validate_role_prompt_template("Limit: {TOOL_CALLS_LIMIT}")

    def test_role_prompt_missing_tool_calls_limit_placeholder_raises(self) -> None:
        with pytest.raises(ToolAgentError, match="TOOL_CALLS_LIMIT"):
            ScriptedToolAgent._validate_role_prompt_template("Tools: {TOOLS}")

    def test_role_prompt_extra_placeholder_raises(self) -> None:
        with pytest.raises(ToolAgentError, match="unsupported placeholder"):
            ScriptedToolAgent._validate_role_prompt_template(
                "Tools: {TOOLS} Limit: {TOOL_CALLS_LIMIT} Extra: {EXTRA}"
            )

    def test_role_prompt_positional_placeholder_raises(self) -> None:
        with pytest.raises(ToolAgentError, match="positional fields"):
            ScriptedToolAgent._validate_role_prompt_template(
                "Tools: {TOOLS} Limit: {TOOL_CALLS_LIMIT} {}"
            )

    def test_role_prompt_field_expression_raises(self) -> None:
        with pytest.raises(ToolAgentError, match="unsupported field expression"):
            ScriptedToolAgent._validate_role_prompt_template(
                "Tools: {TOOLS.name} Limit: {TOOL_CALLS_LIMIT}"
            )

    @pytest.mark.parametrize("value", [None, 0, 1, 5])
    def test_tool_calls_limit_accepts_none_and_non_negative_int(
        self,
        value: int | None,
    ) -> None:
        agent = make_agent(tool_calls_limit=value)

        assert agent.tool_calls_limit == value

    @pytest.mark.parametrize("value", [-1, "1", 1.5, True])
    def test_tool_calls_limit_rejects_negative_or_non_int(self, value: Any) -> None:
        with pytest.raises(ToolAgentError, match="tool_calls_limit"):
            make_agent(tool_calls_limit=value)  # type: ignore[arg-type]

    @pytest.mark.parametrize("value", [True, False])
    def test_peek_at_cache_accepts_bool(self, value: bool) -> None:
        agent = make_agent()

        agent.peek_at_cache = value

        assert agent.peek_at_cache is value

    @pytest.mark.parametrize("value", ["yes", 1, None])
    def test_peek_at_cache_rejects_non_bool(self, value: Any) -> None:
        agent = make_agent()

        with pytest.raises(ToolAgentError, match="peek_at_cache"):
            agent.peek_at_cache = value  # type: ignore[assignment]

    @pytest.mark.parametrize("value", [None, 1, 20])
    def test_preview_limit_alias_sets_response_preview_limit(
        self,
        value: int | None,
    ) -> None:
        agent = make_agent()

        agent.preview_limit = value

        assert agent.preview_limit == value
        assert agent.response_preview_limit == value

    @pytest.mark.parametrize("value", [0, -1, "10"])
    def test_preview_limit_alias_rejects_via_response_preview_limit(self, value: Any) -> None:
        agent = make_agent()

        with pytest.raises(AgentError, match="response_preview_limit"):
            agent.preview_limit = value  # type: ignore[assignment]

    def test_constructor_rejects_preview_limit_and_response_preview_limit_together(self) -> None:
        with pytest.raises(ToolAgentError, match="preview_limit and response_preview_limit"):
            make_agent(preview_limit=10, response_preview_limit=20)

    @pytest.mark.parametrize("value", [None, 1, 20])
    def test_blackboard_preview_limit_accepts_none_or_positive_int(
        self,
        value: int | None,
    ) -> None:
        agent = make_agent()

        agent.blackboard_preview_limit = value

        assert agent.blackboard_preview_limit == value

    @pytest.mark.parametrize("value", [0, -1, "10"])
    def test_blackboard_preview_limit_rejects_zero_negative_or_non_int(self, value: Any) -> None:
        agent = make_agent()

        with pytest.raises(ToolAgentError, match="blackboard_preview_limit"):
            agent.blackboard_preview_limit = value  # type: ignore[assignment]


class TestToolRegistration:
    def test_register_callable_adds_tool(self) -> None:
        agent = make_agent()

        key = agent.register(add, namespace="tests")

        assert key == "Tool.tests.add"
        assert agent.has_tool(key)
        assert agent.get_tool(key).invoke({"x": 1, "y": 2}) == 3

    def test_register_tool_instance_adds_tool(self) -> None:
        agent = make_agent()
        tool = Tool(
            function=add,
            name="adder",
            namespace="tests",
            description="Add values.",
        )

        key = agent.register(tool)

        assert key == "Tool.tests.adder"
        assert agent.get_tool(key) is tool

    def test_register_duplicate_raises_by_default(self) -> None:
        agent = make_agent()
        agent.register(add, namespace="tests")

        with pytest.raises(ToolRegistrationError, match="already registered"):
            agent.register(add, namespace="tests")

    def test_register_duplicate_skip_returns_existing_key(self) -> None:
        agent = make_agent()
        first = agent.register(add, namespace="tests")
        second = agent.register(add, namespace="tests", name_collision_mode="skip")

        assert second == first
        assert agent.get_tool(first).function is add

    def test_register_duplicate_replace_replaces_tool(self) -> None:
        agent = make_agent()
        first = agent.register(add, name="calc", namespace="tests")
        second = agent.register(
            multiply,
            name="calc",
            namespace="tests",
            name_collision_mode="replace",
        )

        assert second == first
        assert agent.get_tool(first).invoke({"x": 3, "y": 4}) == 12

    def test_register_invalid_collision_mode_raises(self) -> None:
        agent = make_agent()

        with pytest.raises(ToolRegistrationError, match="name_collision_mode"):
            agent.register(add, namespace="tests", name_collision_mode="bad")

    def test_get_tool_unknown_raises(self) -> None:
        agent = make_agent()

        with pytest.raises(ToolAgentError, match="unknown tool"):
            agent.get_tool("Tool.tests.missing")

    def test_remove_tool_returns_true_then_false(self) -> None:
        agent = make_agent()
        key = agent.register(add, namespace="tests")

        assert agent.remove_tool(key) is True
        assert agent.remove_tool(key) is False

    def test_clear_tools_removes_all_tools(self) -> None:
        agent = make_agent()
        register_math_tools(agent)

        agent.clear_tools()

        assert agent.list_tools() == {}

    def test_batch_register_callables(self) -> None:
        agent = make_agent()

        keys = agent.batch_register([add, multiply], batch_namespace="tests")

        assert keys == ["Tool.tests.add", "Tool.tests.multiply"]
        assert agent.has_tool("Tool.tests.add")
        assert agent.has_tool("Tool.tests.multiply")

    def test_batch_register_empty_sources_raises(self) -> None:
        agent = make_agent()

        with pytest.raises(ValueError, match="non-empty"):
            agent.batch_register([])

    def test_actions_context_lists_registered_tools(self) -> None:
        agent = make_agent()
        key = agent.register(add, namespace="tests")

        context = agent.actions_context()

        assert key in context


class TestPlaceholderResolution:
    def test_full_step_placeholder_preserves_type(self) -> None:
        agent = make_agent()
        state = make_state(running=[executed_slot(0, [1, 2, 3])])

        assert agent._resolve_placeholders("<<__s0__>>", state=state) == [1, 2, 3]

    def test_full_cache_placeholder_preserves_type(self) -> None:
        agent = make_agent()
        state = make_state(cache=[executed_slot(0, {"cached": 10})])

        assert agent._resolve_placeholders("<<__c0__>>", state=state) == {"cached": 10}

    def test_inline_step_placeholder_uses_repr(self) -> None:
        agent = make_agent()
        state = make_state(running=[executed_slot(0, ["a", "b"])])

        assert agent._resolve_placeholders("result=<<__s0__>>", state=state) == "result=['a', 'b']"

    def test_inline_cache_placeholder_uses_repr(self) -> None:
        agent = make_agent()
        state = make_state(cache=[executed_slot(0, {"cached": 10})])

        assert agent._resolve_placeholders("cache=<<__c0__>>", state=state) == "cache={'cached': 10}"

    def test_nested_dict_list_tuple_set_resolution(self) -> None:
        agent = make_agent()
        state = make_state(
            cache=[executed_slot(0, "cached")],
            running=[
                executed_slot(0, 5),
                executed_slot(1, ("a", "b")),
            ],
        )

        resolved = agent._resolve_placeholders(
            {
                "a": "<<__s0__>>",
                "b": ["<<__c0__>>", "inline <<__s1__>>"],
                "c": ("<<__s0__>>",),
                "d": {"<<__c0__>>"},
            },
            state=state,
        )

        assert resolved == {
            "a": 5,
            "b": ["cached", "inline ('a', 'b')"],
            "c": (5,),
            "d": {"cached"},
        }

    def test_placeholder_in_dict_key_resolves(self) -> None:
        agent = make_agent()
        state = make_state(running=[executed_slot(0, "dynamic_key")])

        resolved = agent._resolve_placeholders({"<<__s0__>>": "value"}, state=state)

        assert resolved == {"dynamic_key": "value"}

    def test_multiple_placeholders_in_one_string(self) -> None:
        agent = make_agent()
        state = make_state(
            running=[
                executed_slot(0, 1),
                executed_slot(1, 2),
            ]
        )

        assert agent._resolve_placeholders("<<__s0__>> + <<__s1__>>", state=state) == "1 + 2"

    def test_out_of_range_step_placeholder_raises(self) -> None:
        agent = make_agent()
        state = make_state(running=[])

        with pytest.raises(ToolAgentError, match="Step reference 0 out of range"):
            agent._resolve_placeholders("<<__s0__>>", state=state)

    def test_out_of_range_cache_placeholder_raises(self) -> None:
        agent = make_agent()
        state = make_state(cache=[])

        with pytest.raises(ToolAgentError, match="Cache reference 0 out of range"):
            agent._resolve_placeholders("<<__c0__>>", state=state)

    def test_unexecuted_step_placeholder_raises(self) -> None:
        agent = make_agent()
        state = make_state(running=[BlackboardSlot(step=0)])

        with pytest.raises(ToolAgentError, match="Referenced step 0 is not executed"):
            agent._resolve_placeholders("<<__s0__>>", state=state)

    def test_unexecuted_cache_placeholder_raises(self) -> None:
        agent = make_agent()
        state = make_state(cache=[BlackboardSlot(step=0)])

        with pytest.raises(ToolAgentError, match="Referenced cache 0 is not executed"):
            agent._resolve_placeholders("<<__c0__>>", state=state)

    def test_non_string_scalar_is_returned_unchanged(self) -> None:
        agent = make_agent()
        state = make_state()

        assert agent._resolve_placeholders(123, state=state) == 123
        assert agent._resolve_placeholders(None, state=state) is None

    def test_extract_dependencies_finds_nested_placeholders(self) -> None:
        obj = {"a": "<<__s0__>>", "b": ["prefix <<__s1__>>"], "<<__s2__>>": "key"}

        assert extract_dependencies(obj, _STEP_TOKEN) == {0, 1, 2}
        assert extract_dependencies(obj, _CACHE_TOKEN) == set()


class TestExecutePreparedBatch:
    def test_executes_single_non_return_tool_and_stores_result(self) -> None:
        agent = make_agent()
        keys = register_math_tools(agent)

        slot = prepared_slot(0, keys["add"], {"x": 2, "y": 3})
        state = make_state(running=[slot], prepared_steps=[0])

        updated = agent._execute_prepared_batch(state)

        assert updated.running_blackboard[0].result == 5
        assert updated.executed_steps == {0}
        assert updated.tool_calls_used == 1
        assert updated.prepared_steps == []

    def test_executes_multiple_non_return_tools(self) -> None:
        agent = make_agent()
        keys = register_math_tools(agent)

        state = make_state(
            running=[
                prepared_slot(0, keys["add"], {"x": 2, "y": 3}),
                prepared_slot(1, keys["multiply"], {"x": 4, "y": 5}),
            ],
            prepared_steps=[0, 1],
        )

        updated = agent._execute_prepared_batch(state)

        assert updated.running_blackboard[0].result == 5
        assert updated.running_blackboard[1].result == 20
        assert updated.tool_calls_used == 2

    def test_executes_return_tool_sets_done_and_return_value(self) -> None:
        agent = make_agent()

        state = make_state(
            running=[prepared_slot(0, return_tool.full_name, {"val": 123})],
            prepared_steps=[0],
        )

        updated = agent._execute_prepared_batch(state)

        assert updated.is_done is True
        assert updated.return_value == 123
        assert updated.tool_calls_used == 0

    def test_return_tool_does_not_increment_tool_calls_used(self) -> None:
        agent = make_agent()

        state = make_state(
            running=[prepared_slot(0, return_tool.full_name, {"val": "done"})],
            prepared_steps=[0],
            tool_calls_used=3,
        )

        updated = agent._execute_prepared_batch(state)

        assert updated.tool_calls_used == 3

    def test_empty_prepared_steps_raises(self) -> None:
        agent = make_agent()
        state = make_state(running=[])

        with pytest.raises(ToolAgentError, match="no prepared steps"):
            agent._execute_prepared_batch(state)

    def test_duplicate_prepared_steps_raises(self) -> None:
        agent = make_agent()
        keys = register_math_tools(agent)
        state = make_state(
            running=[prepared_slot(0, keys["add"], {"x": 1, "y": 2})],
            prepared_steps=[0, 0],
        )

        with pytest.raises(ToolAgentError, match="duplicates"):
            agent._execute_prepared_batch(state)

    def test_non_int_prepared_step_raises(self) -> None:
        agent = make_agent()
        state = make_state(
            running=[BlackboardSlot(step=0)],
            prepared_steps=["0"],  # type: ignore[list-item]
        )

        with pytest.raises(ToolAgentError, match="must be int"):
            agent._execute_prepared_batch(state)

    def test_out_of_range_prepared_step_raises(self) -> None:
        agent = make_agent()
        state = make_state(running=[], prepared_steps=[0])

        with pytest.raises(ToolAgentError, match="out of range"):
            agent._execute_prepared_batch(state)

    def test_step_mismatch_raises(self) -> None:
        agent = make_agent()
        keys = register_math_tools(agent)
        slot = prepared_slot(99, keys["add"], {"x": 1, "y": 2})
        state = make_state(running=[slot], prepared_steps=[0])

        with pytest.raises(ToolAgentError, match="step mismatch"):
            agent._execute_prepared_batch(state)

    def test_already_executed_step_raises(self) -> None:
        agent = make_agent()
        state = make_state(running=[executed_slot(0, 3)], prepared_steps=[0])

        with pytest.raises(ToolAgentError, match="already executed"):
            agent._execute_prepared_batch(state)

    def test_unprepared_slot_raises(self) -> None:
        agent = make_agent()
        state = make_state(running=[BlackboardSlot(step=0)], prepared_steps=[0])

        with pytest.raises(ToolAgentError, match="not prepared"):
            agent._execute_prepared_batch(state)

    def test_invalid_tool_name_raises(self) -> None:
        agent = make_agent()
        slot = prepared_slot(0, "", {})
        state = make_state(running=[slot], prepared_steps=[0])

        with pytest.raises(ToolAgentError, match="invalid tool name"):
            agent._execute_prepared_batch(state)

    def test_unknown_tool_raises(self) -> None:
        agent = make_agent()
        slot = prepared_slot(0, "Tool.tests.missing", {})
        state = make_state(running=[slot], prepared_steps=[0])

        with pytest.raises(ToolAgentError, match="unknown tool"):
            agent._execute_prepared_batch(state)

    def test_multiple_return_tools_in_same_batch_raises(self) -> None:
        agent = make_agent()
        state = make_state(
            running=[
                prepared_slot(0, return_tool.full_name, {"val": 1}),
                prepared_slot(1, return_tool.full_name, {"val": 2}),
            ],
            prepared_steps=[0, 1],
        )

        with pytest.raises(ToolAgentError, match="multiple return"):
            agent._execute_prepared_batch(state)

    def test_tool_calls_limit_exceeded_raises(self) -> None:
        agent = make_agent(tool_calls_limit=0)
        keys = register_math_tools(agent)
        state = make_state(
            running=[prepared_slot(0, keys["add"], {"x": 1, "y": 2})],
            prepared_steps=[0],
        )

        with pytest.raises(ToolAgentError, match="tool_calls_limit exceeded"):
            agent._execute_prepared_batch(state)

    def test_tool_failure_records_error_and_raises(self) -> None:
        agent = make_agent()
        keys = register_math_tools(agent)
        slot = prepared_slot(0, keys["fail_tool"], {})
        state = make_state(running=[slot], prepared_steps=[0])

        with pytest.raises((ToolInvocationError, ToolAgentError)):
            agent._execute_prepared_batch(state)

        assert state.running_blackboard[0].error is not NO_VAL

    def test_prepared_steps_cleared_after_success(self) -> None:
        agent = make_agent()
        keys = register_math_tools(agent)
        state = make_state(
            running=[prepared_slot(0, keys["add"], {"x": 1, "y": 2})],
            prepared_steps=[0],
        )

        updated = agent._execute_prepared_batch(state)

        assert updated.prepared_steps == []


class TestScriptedInvokeLoop:
    def test_scripted_invoke_runs_tools_placeholders_and_return(self) -> None:
        agent = make_agent()
        keys = register_math_tools(agent)
        agent.set_script(
            [
                [{"tool": keys["add"], "args": {"x": 2, "y": 3}}],
                [{"tool": keys["multiply"], "args": {"x": "<<__s0__>>", "y": 10}}],
                [{"tool": return_tool.full_name, "args": {"val": "<<__s1__>>"}}],
            ]
        )

        result = agent.invoke({"prompt": "run"})

        assert result == 50

    def test_context_disabled_does_not_persist_blackboard(self) -> None:
        agent = make_agent(context_enabled=False)
        keys = register_math_tools(agent)
        agent.set_script(
            [
                [{"tool": keys["add"], "args": {"x": 2, "y": 3}}],
                [{"tool": return_tool.full_name, "args": {"val": "<<__s0__>>"}}],
            ]
        )

        assert agent.invoke({"prompt": "run"}) == 5
        assert agent.blackboard == []
        assert agent.turn_history == []

    def test_context_enabled_stores_tool_agent_turn_with_blackboard_span(self) -> None:
        agent = make_agent(context_enabled=True)
        keys = register_math_tools(agent)
        agent.set_script(
            [
                [{"tool": keys["add"], "args": {"x": 2, "y": 3}}],
                [{"tool": return_tool.full_name, "args": {"val": "<<__s0__>>"}}],
            ]
        )

        assert agent.invoke({"prompt": "run"}) == 5

        assert len(agent.turn_history) == 1
        turn = agent.turn_history[0]
        assert isinstance(turn, ToolAgentTurn)
        assert turn.prompt == "run"
        assert turn.raw_response == 5
        assert turn.final_response == 5
        assert turn.blackboard_start == 0
        assert turn.blackboard_end == len(agent.blackboard)

    def test_context_enabled_persists_executed_blackboard(self) -> None:
        agent = make_agent(context_enabled=True)
        keys = register_math_tools(agent)
        agent.set_script(
            [
                [{"tool": keys["add"], "args": {"x": 2, "y": 3}}],
                [{"tool": return_tool.full_name, "args": {"val": "<<__s0__>>"}}],
            ]
        )

        assert agent.invoke({"prompt": "run"}) == 5

        board = agent.blackboard
        assert len(board) == 2
        assert board[0].tool == keys["add"]
        assert board[0].result == 5
        assert board[1].tool == return_tool.full_name
        assert board[1].result == 5

    def test_context_enabled_rewrites_step_placeholders_to_cache_placeholders(self) -> None:
        agent = make_agent(context_enabled=True)
        keys = register_math_tools(agent)
        agent.set_script(
            [
                [{"tool": keys["add"], "args": {"x": 2, "y": 3}}],
                [{"tool": keys["multiply"], "args": {"x": "<<__s0__>>", "y": 10}}],
                [{"tool": return_tool.full_name, "args": {"val": "<<__s1__>>"}}],
            ]
        )

        assert agent.invoke({"prompt": "run"}) == 50

        board = agent.blackboard
        assert board[1].args == {"x": "<<__c0__>>", "y": 10}
        assert board[2].args == {"val": "<<__c1__>>"}

    def test_clear_memory_clears_agent_history_and_blackboard(self) -> None:
        agent = make_agent(context_enabled=True)
        keys = register_math_tools(agent)
        agent.set_script(
            [
                [{"tool": keys["add"], "args": {"x": 2, "y": 3}}],
                [{"tool": return_tool.full_name, "args": {"val": "<<__s0__>>"}}],
            ]
        )

        agent.invoke({"prompt": "run"})

        assert agent.blackboard
        assert agent.turn_history

        agent.clear_memory()

        assert agent.blackboard == []
        assert agent.turn_history == []

        with pytest.warns(DeprecationWarning):
            rendered_history = agent.history

        assert rendered_history == []

    def test_prepare_empty_batch_raises(self) -> None:
        agent = make_agent()
        agent.set_script([[]])

        with pytest.raises(ToolAgentError, match="empty batch"):
            agent.invoke({"prompt": "run"})

    def test_initialize_run_state_wrong_type_raises(self) -> None:
        agent = BadInitializeToolAgent(script=[])

        with pytest.raises(ToolAgentError, match="must return a ToolAgentRunState"):
            agent.invoke({"prompt": "run"})

    def test_pending_prepared_steps_before_prepare_raises(self) -> None:
        agent = PendingPreparedToolAgent(
            script=[[{"tool": return_tool.full_name, "args": {"val": 1}}]]
        )

        with pytest.raises(ToolAgentError, match="prepared_(steps|indices) is non-empty"):
            agent.invoke({"prompt": "run"})

    def test_cached_placeholder_can_be_used_on_later_invoke_when_context_enabled(self) -> None:
        agent = make_agent(context_enabled=True)
        keys = register_math_tools(agent)

        agent.set_script(
            [
                [{"tool": keys["add"], "args": {"x": 2, "y": 3}}],
                [{"tool": return_tool.full_name, "args": {"val": "<<__s0__>>"}}],
            ]
        )
        assert agent.invoke({"prompt": "first"}) == 5

        agent.set_script(
            [
                [{"tool": return_tool.full_name, "args": {"val": "<<__c0__>>"}}],
            ]
        )
        assert agent.invoke({"prompt": "second"}) == 5


class TestBlackboardPersistenceAndDisplay:
    def test_blackboard_property_returns_copied_slots(self) -> None:
        agent = make_agent(context_enabled=True)
        keys = register_math_tools(agent)
        agent.set_script(
            [
                [{"tool": keys["add"], "args": {"x": 1, "y": 2}}],
                [{"tool": return_tool.full_name, "args": {"val": "<<__s0__>>"}}],
            ]
        )
        agent.invoke({"prompt": "run"})

        snapshot = agent.blackboard
        snapshot[0].result = 999

        assert agent.blackboard[0].result == 3

    def test_blackboard_serialized_returns_dicts(self) -> None:
        agent = make_agent(context_enabled=True)
        keys = register_math_tools(agent)
        agent.set_script(
            [
                [{"tool": keys["add"], "args": {"x": 1, "y": 2}}],
                [{"tool": return_tool.full_name, "args": {"val": "<<__s0__>>"}}],
            ]
        )
        agent.invoke({"prompt": "run"})

        serialized = agent.blackboard_serialized

        assert isinstance(serialized, list)
        assert serialized[0]["result"] == 3
        assert serialized[0]["completed"] is True

    def test_blackboard_dumps_without_peek_hides_results(self) -> None:
        agent = make_agent(context_enabled=True)
        keys = register_math_tools(agent)
        agent.set_script(
            [
                [{"tool": keys["add"], "args": {"x": 1, "y": 2}}],
                [{"tool": return_tool.full_name, "args": {"val": "<<__s0__>>"}}],
            ]
        )
        agent.invoke({"prompt": "run"})

        dump = agent.blackboard_dumps(peek=False)

        assert "Tool.tests.add" in dump
        assert "'result'" not in dump

    def test_blackboard_dumps_with_peek_includes_results_but_hides_resolved_args(self) -> None:
        agent = make_agent(context_enabled=True)
        keys = register_math_tools(agent)
        agent.set_script(
            [
                [{"tool": keys["add"], "args": {"x": 1, "y": 2}}],
                [{"tool": return_tool.full_name, "args": {"val": "<<__s0__>>"}}],
            ]
        )
        agent.invoke({"prompt": "run"})

        dump = agent.blackboard_dumps(peek=True)

        assert "val" in dump
        assert "'result'" in dump
        assert "resolved_args" not in dump
        assert "<<__c0__>>" in dump

    def test_rendered_history_with_peek_at_cache_includes_cached_step_results(self) -> None:
        agent = make_agent(context_enabled=True, peek_at_cache=True, blackboard_preview_limit=10)
        keys = register_math_tools(agent)
        agent.set_script(
            [
                [
                    {
                        "tool": keys["join_text"],
                        "args": {
                            "prefix": "long",
                            "value": "abcdefghijklmnopqrstuvwxyz",
                        },
                    }
                ],
                [{"tool": return_tool.full_name, "args": {"val": "<<__s0__>>"}}],
            ]
        )

        result = agent.invoke({"prompt": "run"})

        assert result == "long:abcdefghijklmnopqrstuvwxyz"
        with pytest.warns(DeprecationWarning):
            history = agent.history

        content = history[-1]["content"]
        assert "CACHED STEPS" in content
        assert "result" in content
        assert "long:abcd" in content
        assert "long:abcdefghijklmnopqrstuvwxyz" in content

    def test_rendered_history_without_peek_at_cache_hides_cached_step_results(self) -> None:
        agent = make_agent(context_enabled=True, peek_at_cache=False)
        keys = register_math_tools(agent)
        agent.set_script(
            [
                [{"tool": keys["add"], "args": {"x": 1, "y": 2}}],
                [{"tool": return_tool.full_name, "args": {"val": "<<__s0__>>"}}],
            ]
        )

        assert agent.invoke({"prompt": "run"}) == 3

        with pytest.warns(DeprecationWarning):
            history = agent.history

        content = history[-1]["content"]
        assert "CACHED STEPS" in content
        assert "'args'" in content
        assert "'result'" not in content

    def test_rendered_history_blackboard_preview_limit_truncates_results_only(self) -> None:
        agent = make_agent(
            context_enabled=True,
            peek_at_cache=True,
            blackboard_preview_limit=10,
        )
        keys = register_math_tools(agent)
        agent.set_script(
            [
                [
                    {
                        "tool": keys["join_text"],
                        "args": {
                            "prefix": "long",
                            "value": "abcdefghijklmnopqrstuvwxyz",
                        },
                    }
                ],
                [{"tool": return_tool.full_name, "args": {"val": "<<__s0__>>"}}],
            ]
        )

        assert agent.invoke({"prompt": "run"}) == "long:abcdefghijklmnopqrstuvwxyz"

        with pytest.warns(DeprecationWarning):
            history = agent.history

        content = history[-1]["content"]
        assert "abcdefghijklmnopqrstuvwxyz" in content
        assert "'long:abcd..." in content

    def test_response_preview_limit_truncates_response_not_cached_args(self) -> None:
        agent = make_agent(
            context_enabled=True,
            peek_at_cache=True,
            response_preview_limit=10,
            blackboard_preview_limit=None,
        )
        keys = register_math_tools(agent)
        agent.set_script(
            [
                [
                    {
                        "tool": keys["join_text"],
                        "args": {
                            "prefix": "long",
                            "value": "abcdefghijklmnopqrstuvwxyz",
                        },
                    }
                ],
                [{"tool": return_tool.full_name, "args": {"val": "<<__s0__>>"}}],
            ]
        )

        assert agent.invoke({"prompt": "run"}) == "long:abcdefghijklmnopqrstuvwxyz"

        with pytest.warns(DeprecationWarning):
            history = agent.history

        content = history[-1]["content"]
        response_section = content.split("CACHED STEPS", maxsplit=1)[0]
        cached_section = content.split("CACHED STEPS", maxsplit=1)[1]
        assert "RESPONSE:\nlong:abcde..." in response_section
        assert "abcdefghijklmnopqrstuvwxyz" in cached_section
        assert "'long:abcdefghijklmnopqrstuvwxyz'" in cached_section


class TestToolAgentTurnRendering:
    def test_render_turn_returns_single_user_assistant_pair(self) -> None:
        agent = make_agent(context_enabled=True, peek_at_cache=True)
        keys = register_math_tools(agent)
        agent.set_script(
            [
                [{"tool": keys["add"], "args": {"x": 1, "y": 2}}],
                [{"tool": return_tool.full_name, "args": {"val": "<<__s0__>>"}}],
            ]
        )

        assert agent.invoke({"prompt": "run"}) == 3

        rendered = agent.render_turn(agent.turn_history[0])

        assert len(rendered) == 2
        assert [message["role"] for message in rendered] == ["user", "assistant"]
        assert rendered[0]["content"] == "run"
        assert rendered[1]["content"].startswith("RESPONSE:")
        assert "CACHED STEPS #0-1 PRODUCED" in rendered[1]["content"]

    def test_render_turn_uses_blackboard_span_only_for_that_turn(self) -> None:
        agent = make_agent(context_enabled=True, peek_at_cache=True)
        keys = register_math_tools(agent)

        agent.set_script(
            [
                [{"tool": keys["add"], "args": {"x": 1, "y": 2}}],
                [{"tool": return_tool.full_name, "args": {"val": "<<__s0__>>"}}],
            ]
        )
        assert agent.invoke({"prompt": "first"}) == 3

        agent.set_script(
            [
                [{"tool": keys["multiply"], "args": {"x": 4, "y": 5}}],
                [{"tool": return_tool.full_name, "args": {"val": "<<__s0__>>"}}],
            ]
        )
        assert agent.invoke({"prompt": "second"}) == 20

        first_rendered = agent.render_turn(agent.turn_history[0])[1]["content"]
        second_rendered = agent.render_turn(agent.turn_history[1])[1]["content"]

        assert "CACHED STEPS #0-1 PRODUCED" in first_rendered
        assert "CACHED STEPS #2-3 PRODUCED" in second_rendered
        assert "Tool.tests.add" in first_rendered
        assert "Tool.tests.multiply" not in first_rendered
        assert "Tool.tests.multiply" in second_rendered
        assert "Tool.tests.add" not in second_rendered

    def test_render_turn_raises_for_non_tool_agent_turn(self) -> None:
        agent = make_agent()
        turn = AgentTurn(prompt="run", raw_response="raw", final_response="final")

        with pytest.raises(ToolAgentError, match="ToolAgentTurn"):
            agent.render_turn(turn)



class TestParsingHelpers:
    def test_str_to_steps_extracts_json_array_from_plain_text(self) -> None:
        agent = make_agent()

        steps = agent._str_to_steps(
            'Plan:\n[{"tool": "Tool.tests.add", "args": {"x": 1, "y": 2}}]\nDone.'
        )

        assert steps == [{"tool": "Tool.tests.add", "args": {"x": 1, "y": 2}}]

    def test_str_to_steps_extracts_json_array_from_markdown_fence(self) -> None:
        agent = make_agent()

        steps = agent._str_to_steps(
            '```json\n[{"tool": "Tool.tests.add", "args": {"x": 1, "y": 2}}]\n```'
        )

        assert steps == [{"tool": "Tool.tests.add", "args": {"x": 1, "y": 2}}]

    @pytest.mark.parametrize("raw", ["", "   ", "not json", "[]", "{}"])
    def test_str_to_steps_rejects_invalid_text(self, raw: str) -> None:
        agent = make_agent()

        with pytest.raises(ToolAgentError):
            agent._str_to_steps(raw)

    def test_str_to_dict_extracts_json_object_from_plain_text(self) -> None:
        agent = make_agent()

        data = agent._str_to_dict(
            'Before {"step": 0, "tool": "Tool.tests.add", "args": {}} after'
        )

        assert data == {"step": 0, "tool": "Tool.tests.add", "args": {}}

    def test_str_to_dict_extracts_json_object_from_markdown_fence(self) -> None:
        agent = make_agent()

        data = agent._str_to_dict(
            '```json\n{"step": 0, "tool": "Tool.tests.add", "args": {}}\n```'
        )

        assert data == {"step": 0, "tool": "Tool.tests.add", "args": {}}

    @pytest.mark.parametrize("raw", ["", "   ", "not json", "[]"])
    def test_str_to_dict_rejects_invalid_text(self, raw: str) -> None:
        agent = make_agent()

        with pytest.raises(ToolAgentError):
            agent._str_to_dict(raw)


class TestPlannedStep:
    def test_from_dict_extracts_placeholder_dependencies(self) -> None:
        step = PlannedStep.from_dict(
            {
                "tool": "Tool.tests.multiply",
                "args": {"x": "<<__s0__>>", "y": "inline <<__s2__>>"},
            }
        )

        assert step.tool == "Tool.tests.multiply"
        assert step.args == {"x": "<<__s0__>>", "y": "inline <<__s2__>>"}
        assert step.deps == frozenset({0, 2})
        assert step.is_return is False

    def test_from_dict_adds_await_dependency(self) -> None:
        step = PlannedStep.from_dict(
            {
                "tool": "Tool.tests.multiply",
                "args": {"x": "<<__s0__>>", "y": 10},
                "await": 1,
            }
        )

        assert step.deps == frozenset({0, 1})

    def test_from_dict_ignores_await_for_return_tool(self) -> None:
        step = PlannedStep.from_dict(
            {
                "tool": return_tool.full_name,
                "args": {"val": "<<__s0__>>"},
                "await": 99,
            }
        )

        assert step.is_return is True
        assert step.deps == frozenset({0})

    def test_from_dict_rejects_non_mapping(self) -> None:
        with pytest.raises(ToolAgentError, match="requires a mapping"):
            PlannedStep.from_dict(["bad"])  # type: ignore[arg-type]

    def test_from_dict_rejects_extra_keys(self) -> None:
        with pytest.raises(ToolAgentError, match="unsupported keys"):
            PlannedStep.from_dict(
                {
                    "tool": "Tool.tests.add",
                    "args": {},
                    "extra": True,
                }
            )

    def test_from_dict_rejects_missing_tool(self) -> None:
        with pytest.raises(ToolAgentError, match="missing required key: 'tool'"):
            PlannedStep.from_dict({"args": {}})

    def test_from_dict_rejects_missing_args(self) -> None:
        with pytest.raises(ToolAgentError, match="missing required key: 'args'"):
            PlannedStep.from_dict({"tool": "Tool.tests.add"})

    def test_from_dict_rejects_invalid_tool(self) -> None:
        with pytest.raises(ToolAgentError, match="'tool' must be"):
            PlannedStep.from_dict({"tool": "", "args": {}})

    @pytest.mark.parametrize("await_value", [-1, "0", 1.5])
    def test_from_dict_rejects_invalid_await(self, await_value: Any) -> None:
        with pytest.raises(ToolAgentError, match="'await'"):
            PlannedStep.from_dict(
                {
                    "tool": "Tool.tests.add",
                    "args": {},
                    "await": await_value,
                }
            )

    def test_to_dict_preserves_tool_and_args_only(self) -> None:
        step = PlannedStep.from_dict(
            {
                "tool": "Tool.tests.add",
                "args": {"x": 1, "y": 2},
                "await": 0,
                "step": 99,
            }
        )

        assert step.to_dict() == {
            "tool": "Tool.tests.add",
            "args": {"x": 1, "y": 2},
        }


class TestPlanActAgent:
    def test_invokes_planned_tools_and_returns_value(self) -> None:
        agent = make_planact_agent(
            [
                f"""
                [
                  {{"tool": "Tool.tests.add", "args": {{"x": 2, "y": 3}}}},
                  {{"tool": "Tool.tests.multiply", "args": {{"x": "<<__s0__>>", "y": 10}}}},
                  {{"tool": "{return_tool.full_name}", "args": {{"val": "<<__s1__>>"}}}}
                ]
                """
            ]
        )

        result = agent.invoke({"prompt": "run plan"})

        assert result == 50

    def test_auto_appends_return_none_when_plan_has_no_return(self) -> None:
        agent = make_planact_agent(
            [
                """
                [
                  {"tool": "Tool.tests.add", "args": {"x": 2, "y": 3}}
                ]
                """
            ]
        )

        result = agent.invoke({"prompt": "run plan"})

        assert result is None

    def test_moves_return_step_to_end(self) -> None:
        agent = make_planact_agent(
            [
                f"""
                [
                  {{"tool": "{return_tool.full_name}", "args": {{"val": "<<__s0__>>"}}}},
                  {{"tool": "Tool.tests.add", "args": {{"x": 1, "y": 2}}}}
                ]
                """
            ]
        )

        result = agent.invoke({"prompt": "run plan"})

        assert result == 3

    def test_executes_independent_steps_in_same_batch(self) -> None:
        agent = make_planact_agent(
            [
                f"""
                [
                  {{"tool": "Tool.tests.add", "args": {{"x": 1, "y": 2}}}},
                  {{"tool": "Tool.tests.multiply", "args": {{"x": 3, "y": 4}}}},
                  {{"tool": "{return_tool.full_name}", "args": {{"val": ["<<__s0__>>", "<<__s1__>>"]}}}}
                ]
                """
            ]
        )

        state = agent._initialize_run_state(
            messages=[{"role": "user", "content": "plan"}]
        )

        assert state.batches == [[0, 1], [2]]

    def test_rejects_multiple_return_steps(self) -> None:
        agent = make_planact_agent(
            [
                f"""
                [
                  {{"tool": "{return_tool.full_name}", "args": {{"val": 1}}}},
                  {{"tool": "{return_tool.full_name}", "args": {{"val": 2}}}}
                ]
                """
            ]
        )

        with pytest.raises(ToolAgentError, match="multiple return"):
            agent.invoke({"prompt": "run plan"})

    def test_rejects_unknown_tool_in_plan(self) -> None:
        agent = make_planact_agent(
            [
                """
                [
                  {"tool": "Tool.tests.missing", "args": {}}
                ]
                """
            ]
        )

        with pytest.raises(ToolAgentError, match="unknown tool"):
            agent.invoke({"prompt": "run plan"})

    def test_rejects_plan_exceeding_tool_calls_limit(self) -> None:
        agent = make_planact_agent(
            [
                f"""
                [
                  {{"tool": "Tool.tests.add", "args": {{"x": 1, "y": 2}}}},
                  {{"tool": "Tool.tests.multiply", "args": {{"x": 3, "y": 4}}}},
                  {{"tool": "{return_tool.full_name}", "args": {{"val": "<<__s1__>>"}}}}
                ]
                """
            ],
            tool_calls_limit=1,
        )

        with pytest.raises(ToolAgentError, match="tool_calls_limit"):
            agent.invoke({"prompt": "run plan"})

    def test_rejects_out_of_range_cache_reference(self) -> None:
        agent = make_planact_agent(
            [
                f"""
                [
                  {{"tool": "{return_tool.full_name}", "args": {{"val": "<<__c0__>>"}}}}
                ]
                """
            ]
        )

        with pytest.raises(ToolAgentError, match="out-of-range cache"):
            agent.invoke({"prompt": "run plan"})

    def test_rejects_future_step_dependency(self) -> None:
        agent = make_planact_agent(
            [
                f"""
                [
                  {{"tool": "Tool.tests.add", "args": {{"x": "<<__s1__>>", "y": 2}}}},
                  {{"tool": "Tool.tests.multiply", "args": {{"x": 3, "y": 4}}}},
                  {{"tool": "{return_tool.full_name}", "args": {{"val": "<<__s1__>>"}}}}
                ]
                """
            ]
        )

        with pytest.raises(ToolAgentError, match="illegal deps"):
            agent.invoke({"prompt": "run plan"})

    def test_context_enabled_can_reference_cached_result_on_next_invoke(self) -> None:
        agent = make_planact_agent(
            [
                f"""
                [
                  {{"tool": "Tool.tests.add", "args": {{"x": 2, "y": 3}}}},
                  {{"tool": "{return_tool.full_name}", "args": {{"val": "<<__s0__>>"}}}}
                ]
                """,
                f"""
                [
                  {{"tool": "{return_tool.full_name}", "args": {{"val": "<<__c0__>>"}}}}
                ]
                """,
            ],
            context_enabled=True,
        )

        assert agent.invoke({"prompt": "first"}) == 5
        assert agent.invoke({"prompt": "second"}) == 5

    def test_compile_batches_isolates_return_step(self) -> None:
        agent = make_planact_agent(["[]"])
        planned = [
            PlannedStep.from_dict(
                {"tool": "Tool.tests.add", "args": {"x": 1, "y": 2}}
            ),
            PlannedStep.from_dict(
                {"tool": "Tool.tests.multiply", "args": {"x": 3, "y": 4}}
            ),
            PlannedStep.from_dict(
                {"tool": return_tool.full_name, "args": {"val": "<<__s1__>>"}}
            ),
        ]

        batches = agent._compile_batches_from_deps(
            planned_steps=planned,
            return_idx=2,
        )

        assert batches == [[0, 1], [2]]

    def test_async_invoke_executes_plan_and_returns_value(self) -> None:
        agent = make_planact_agent(
            [
                f"""
                [
                  {{"tool": "Tool.tests.add", "args": {{"x": 2, "y": 3}}}},
                  {{"tool": "Tool.tests.multiply", "args": {{"x": "<<__s0__>>", "y": 10}}}},
                  {{"tool": "{return_tool.full_name}", "args": {{"val": "<<__s1__>>"}}}}
                ]
                """
            ]
        )

        result = asyncio.run(agent.async_invoke({"prompt": "run plan"}))

        assert result == 50


class TestReActAgent:
    def test_requires_concrete_non_negative_tool_calls_limit(self) -> None:
        with pytest.raises(ToolAgentError, match="tool_calls_limit"):
            ReActAgent(
                name="bad_react",
                description="Bad ReAct agent.",
                llm_engine=ScriptedLLMEngine([]),
                tool_calls_limit=-1,
            )

    def test_invokes_step_by_step_until_return(self) -> None:
        agent = make_react_agent(
            [
                '{"step": 0, "tool": "Tool.tests.add", "args": {"x": 2, "y": 3}}',
                '{"step": 1, "tool": "Tool.tests.multiply", "args": {"x": "<<__s0__>>", "y": 10}}',
                f'{{"step": 2, "tool": "{return_tool.full_name}", "args": {{"val": "<<__s1__>>"}}}}',
            ],
            tool_calls_limit=2,
        )

        result = agent.invoke({"prompt": "run react"})

        assert result == 50

    def test_injects_observation_after_first_step(self) -> None:
        agent = make_react_agent(
            [
                '{"step": 0, "tool": "Tool.tests.add", "args": {"x": 2, "y": 3}}',
                f'{{"step": 1, "tool": "{return_tool.full_name}", "args": {{"val": "<<__s0__>>"}}}}',
            ],
            tool_calls_limit=1,
        )

        result = agent.invoke({"prompt": "run react"})

        assert result == 5
        engine = agent.llm_engine
        assert isinstance(engine, ScriptedLLMEngine)
        assert len(engine.calls) == 2
        second_call_text = "\n".join(message["content"] for message in engine.calls[1])
        assert "Most recently executed steps and results" in second_call_text
        assert "Tool.tests.add" in second_call_text
        assert "5" in second_call_text

    @pytest.mark.parametrize(
        "raw_response, match",
        [
            ('{"tool": "Tool.tests.add", "args": {}}', "missing required keys"),
            ('{"step": 0, "args": {}}', "missing required keys"),
            ('{"step": 0, "tool": "Tool.tests.add"}', "missing required keys"),
        ],
    )
    def test_rejects_missing_required_step_keys(
        self,
        raw_response: str,
        match: str,
    ) -> None:
        agent = make_react_agent([raw_response], tool_calls_limit=1)

        with pytest.raises(ToolAgentError, match=match):
            agent.invoke({"prompt": "run react"})

    def test_rejects_extra_step_keys(self) -> None:
        agent = make_react_agent(
            [
                '{"step": 0, "tool": "Tool.tests.add", "args": {}, "extra": true}',
            ],
            tool_calls_limit=1,
        )

        with pytest.raises(ToolAgentError, match="unsupported keys"):
            agent.invoke({"prompt": "run react"})

    def test_rejects_non_dict_args(self) -> None:
        agent = make_react_agent(
            [
                '{"step": 0, "tool": "Tool.tests.add", "args": []}',
            ],
            tool_calls_limit=1,
        )

        with pytest.raises(ToolAgentError, match="'args' must be a dict"):
            agent.invoke({"prompt": "run react"})

    def test_rejects_illegal_step_index(self) -> None:
        agent = make_react_agent(
            [
                '{"step": 1, "tool": "Tool.tests.add", "args": {"x": 1, "y": 2}}',
            ],
            tool_calls_limit=1,
        )

        with pytest.raises(ToolAgentError, match="illegal step index"):
            agent.invoke({"prompt": "run react"})

    def test_rejects_future_step_dependency(self) -> None:
        agent = make_react_agent(
            [
                '{"step": 0, "tool": "Tool.tests.add", "args": {"x": "<<__s0__>>", "y": 2}}',
            ],
            tool_calls_limit=1,
        )

        with pytest.raises(ToolAgentError, match="illegal dependency"):
            agent.invoke({"prompt": "run react"})

    def test_rejects_unknown_tool(self) -> None:
        agent = make_react_agent(
            [
                '{"step": 0, "tool": "Tool.tests.missing", "args": {}}',
            ],
            tool_calls_limit=1,
        )

        with pytest.raises(ToolAgentError, match="unknown tool"):
            agent.invoke({"prompt": "run react"})

    def test_rejects_when_next_step_exceeds_capacity(self) -> None:
        agent = make_react_agent(
            [
                '{"step": 0, "tool": "Tool.tests.add", "args": {"x": 1, "y": 2}}',
                '{"step": 1, "tool": "Tool.tests.multiply", "args": {"x": "<<__s0__>>", "y": 3}}',
            ],
            tool_calls_limit=1,
        )

        with pytest.raises(ToolAgentError, match="tool_calls_limit exceeded"):
            agent.invoke({"prompt": "run react"})

    def test_async_invoke_executes_step_by_step_until_return(self) -> None:
        agent = make_react_agent(
            [
                '{"step": 0, "tool": "Tool.tests.add", "args": {"x": 2, "y": 3}}',
                '{"step": 1, "tool": "Tool.tests.multiply", "args": {"x": "<<__s0__>>", "y": 10}}',
                f'{{"step": 2, "tool": "{return_tool.full_name}", "args": {{"val": "<<__s1__>>"}}}}',
            ],
            tool_calls_limit=2,
        )

        result = asyncio.run(agent.async_invoke({"prompt": "run react"}))

        assert result == 50


class TestToolAgentAsyncBaseLoop:
    def test_async_scripted_invoke_runs_tools_placeholders_and_return(self) -> None:
        agent = make_agent()
        keys = register_math_tools(agent)
        agent.set_script(
            [
                [{"tool": keys["add"], "args": {"x": 2, "y": 3}}],
                [{"tool": keys["multiply"], "args": {"x": "<<__s0__>>", "y": 10}}],
                [{"tool": return_tool.full_name, "args": {"val": "<<__s1__>>"}}],
            ]
        )

        result = asyncio.run(agent.async_invoke({"prompt": "run"}))

        assert result == 50

    def test_async_context_enabled_persists_blackboard(self) -> None:
        agent = make_agent(context_enabled=True)
        keys = register_math_tools(agent)
        agent.set_script(
            [
                [{"tool": keys["add"], "args": {"x": 2, "y": 3}}],
                [{"tool": return_tool.full_name, "args": {"val": "<<__s0__>>"}}],
            ]
        )

        result = asyncio.run(agent.async_invoke({"prompt": "run"}))

        assert result == 5
        assert len(agent.blackboard) == 2
        assert agent.blackboard[0].result == 5
        assert agent.blackboard[1].result == 5

    def test_async_execute_prepared_batch_records_tool_error(self) -> None:
        agent = make_agent()
        keys = register_math_tools(agent)
        slot = prepared_slot(0, keys["fail_tool"], {})
        state = make_state(running=[slot], prepared_steps=[0])

        with pytest.raises((ToolInvocationError, ToolAgentError)):
            asyncio.run(agent._async_execute_prepared_batch(state))

        assert state.running_blackboard[0].error is not NO_VAL

    def test_async_initialize_run_state_wrong_type_raises(self) -> None:
        agent = BadInitializeToolAgent(script=[])

        with pytest.raises(ToolAgentError, match="must return a ToolAgentRunState"):
            asyncio.run(agent.async_invoke({"prompt": "run"}))

    def test_async_prepare_empty_batch_raises(self) -> None:
        agent = make_agent()
        agent.set_script([[]])

        with pytest.raises(ToolAgentError, match="empty batch"):
            asyncio.run(agent.async_invoke({"prompt": "run"}))

class TestToolAgentTurnMetadataContract:
    def test_make_turn_accepts_valid_blackboard_span(self) -> None:
        agent = make_agent()

        turn = agent._make_turn(
            prompt="run",
            raw_response=3,
            final_response=3,
            blackboard_start=0,
            blackboard_end=2,
        )

        assert isinstance(turn, ToolAgentTurn)
        assert turn.prompt == "run"
        assert turn.raw_response == 3
        assert turn.final_response == 3
        assert turn.blackboard_start == 0
        assert turn.blackboard_end == 2

    def test_make_turn_accepts_none_blackboard_span(self) -> None:
        agent = make_agent()

        turn = agent._make_turn(
            prompt="run",
            raw_response="raw",
            final_response="final",
            blackboard_start=None,
            blackboard_end=None,
        )

        assert isinstance(turn, ToolAgentTurn)
        assert turn.blackboard_start is None
        assert turn.blackboard_end is None

    def test_make_turn_rejects_partial_none_blackboard_span(self) -> None:
        agent = make_agent()

        with pytest.raises(ToolAgentError, match="both be None or both be integers"):
            agent._make_turn(
                prompt="run",
                raw_response="raw",
                final_response="final",
                blackboard_start=0,
                blackboard_end=None,
            )

        with pytest.raises(ToolAgentError, match="both be None or both be integers"):
            agent._make_turn(
                prompt="run",
                raw_response="raw",
                final_response="final",
                blackboard_start=None,
                blackboard_end=1,
            )

    @pytest.mark.parametrize(
        ("blackboard_start", "blackboard_end"),
        [
            (-1, 1),
            (2, 1),
            (True, 1),
            (0, False),
            ("0", 1),
            (0, "1"),
        ],
    )
    def test_make_turn_rejects_invalid_blackboard_span(
        self,
        blackboard_start: Any,
        blackboard_end: Any,
    ) -> None:
        agent = make_agent()

        with pytest.raises(ToolAgentError, match="blackboard_start and blackboard_end"):
            agent._make_turn(
                prompt="run",
                raw_response="raw",
                final_response="final",
                blackboard_start=blackboard_start,
                blackboard_end=blackboard_end,
            )

    def test_make_turn_rejects_unexpected_metadata(self) -> None:
        agent = make_agent()

        with pytest.raises(ToolAgentError, match="unexpected metadata"):
            agent._make_turn(
                prompt="run",
                raw_response="raw",
                final_response="final",
                blackboard_start=0,
                blackboard_end=1,
                unexpected=True,
            )

    def test_render_turn_with_none_span_returns_base_user_assistant_pair(self) -> None:
        agent = make_agent()
        turn = ToolAgentTurn(
            prompt="run",
            raw_response="raw response",
            final_response="final response",
            blackboard_start=None,
            blackboard_end=None,
        )

        rendered = agent.render_turn(turn)

        assert rendered == [
            {"role": "user", "content": "run"},
            {"role": "assistant", "content": "raw response"},
        ]

    def test_render_turn_with_empty_span_returns_base_user_assistant_pair(self) -> None:
        agent = make_agent()
        turn = ToolAgentTurn(
            prompt="run",
            raw_response="raw response",
            final_response="final response",
            blackboard_start=0,
            blackboard_end=0,
        )

        rendered = agent.render_turn(turn)

        assert rendered == [
            {"role": "user", "content": "run"},
            {"role": "assistant", "content": "raw response"},
        ]

    def test_render_turn_rejects_span_beyond_current_blackboard(self) -> None:
        agent = make_agent()
        turn = ToolAgentTurn(
            prompt="run",
            raw_response="raw response",
            final_response="final response",
            blackboard_start=0,
            blackboard_end=1,
        )

        with pytest.raises(ToolAgentError, match="Invalid blackboard span"):
            agent.render_turn(turn)
