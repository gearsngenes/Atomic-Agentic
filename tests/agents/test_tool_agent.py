from __future__ import annotations

import json
from typing import Any

import pytest

from atomic_agentic.agents.toolagent import ToolAgent
from atomic_agentic.agents.planact import PlanActAgent
from atomic_agentic.agents.react import ReActAgent
from atomic_agentic.constants.agents import RETURN_TOOL_NAME, RETURN_VALUE_FIELD
from atomic_agentic.exceptions import ToolAgentError, ToolRegistrationError
from atomic_agentic.models.agents.blackboard_models import ConstantSpec
from atomic_agentic.models.agents.prompts import PromptConfig
from atomic_agentic.models.a2a_sdk import A2AtomicSkillMetadata
from atomic_agentic.models.parameters import ParamSpec
from atomic_agentic.a2a import A2AClientHub
from atomic_agentic.core.Invokable import AtomicInvokable
from atomic_agentic.tools import Tool
from atomic_agentic.utils.core import apply_name_filter, validate_name_filter

from .conftest import (
    FakeLLMEngine,
    add,
    multiply,
    arg,
    make_planact_agent,
    make_react_agent,
    planact_plan_json,
    react_step_json,
    package_tool_result,
)


# --------------------------------------------------------------------------- #
# Local helpers -- ToolAgent itself is abstract; PlanActAgent is used
# throughout as the concrete stand-in to exercise ToolAgent's own concrete
# surface. `_plain_agent` deliberately does NOT register the math tools
# conftest's `make_planact_agent` always registers -- several tests here
# register "add"/"multiply" themselves under bare names and would collide
# with that fixture's own pre-registration.
# --------------------------------------------------------------------------- #
def _plain_agent(**kwargs: Any) -> PlanActAgent:
    return PlanActAgent(
        name="tests",
        namespace="tests",
        description="Plain PlanActAgent for ToolAgent-level tests.",
        llm_engine=FakeLLMEngine([]),
        **kwargs,
    )


def _scripted_plan(
    *, plan: list[dict[str, Any]], return_value: Any = None, summary: str = "Running the plan."
) -> dict[str, Any]:
    """Round-trips `conftest.planact_plan_json`'s JSON text back into a real
    dict -- FakeLLMEngine never auto-parses scripted responses (see
    test_planact.py's own `scripted()` for the identical pattern)."""
    return json.loads(planact_plan_json(plan=plan, return_value=return_value, summary=summary))


def _a2a_sdk_skill_metadata(*, remote_name: str) -> A2AtomicSkillMetadata:
    return A2AtomicSkillMetadata(
        remote_name=remote_name,
        description=f"Remote skill {remote_name}.",
        extra_description="",
        params=(
            ParamSpec(name="a", index=0, kind=ParamSpec.POSITIONAL_OR_KEYWORD, type="int"),
            ParamSpec(name="b", index=1, kind=ParamSpec.POSITIONAL_OR_KEYWORD, type="int"),
        ),
        return_type="int",
    )


class FakeA2AClientHub(A2AClientHub):
    """Minimal A2AClientHub subclass that skips real network construction --
    same precedent as FakeMCPClientHub/FakePyA2AtomicClient elsewhere in this
    test suite. Re-verified against the current A2AClientHub/A2AProxyTool
    interface (agent_card/transport_mode/get_atomic_skills/call_atomic_skill)
    before reuse -- A2AClientHub is a plain class (not ABC), so overriding
    __init__ wholesale and never calling super().__init__() is safe."""

    def __init__(
        self,
        *,
        skills: dict[str, A2AtomicSkillMetadata] | None = None,
        include_names: list[str] | None = None,
        exclude_names: list[str] | None = None,
    ) -> None:
        raw_skills = (
            {"add": _a2a_sdk_skill_metadata(remote_name="add")} if skills is None else skills
        )
        resolved_include, resolved_exclude = validate_name_filter(include_names, exclude_names)
        self._skills = apply_name_filter(raw_skills, resolved_include, resolved_exclude)
        self._card = type("FakeCard", (), {"name": "FakeA2AAgent", "description": ""})()
        self.skill_calls: list[tuple[str, dict[str, Any]]] = []

    @property
    def agent_card(self) -> Any:
        return self._card

    @property
    def transport_mode(self) -> str:
        return "JSONRPC"

    @property
    def base_url(self) -> str:
        return "http://example.test/a2a-sdk"

    @property
    def persistent(self) -> bool:
        return False

    def get_atomic_skills(self) -> dict[str, A2AtomicSkillMetadata]:
        return dict(self._skills)

    def call_atomic_skill(self, skill_id: str, inputs: dict) -> Any:
        self.skill_calls.append((skill_id, dict(inputs)))
        return 17


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #
class TestToolAgentConstruction:
    @pytest.mark.parametrize("value", [None, 0, 1, 5])
    def test_tool_calls_limit_accepts_none_and_non_negative_int(self, value: int | None) -> None:
        agent = _plain_agent(tool_calls_limit=value)
        assert agent.tool_calls_limit == value

    @pytest.mark.parametrize("value", [-1, "1", 1.5])
    def test_tool_calls_limit_rejects_negative_or_non_int(self, value: Any) -> None:
        with pytest.raises(ToolAgentError, match="tool_calls_limit"):
            _plain_agent(tool_calls_limit=value)

    def test_tool_calls_limit_setter_validates_after_construction(self) -> None:
        agent = _plain_agent()
        with pytest.raises(ToolAgentError, match="tool_calls_limit"):
            agent.tool_calls_limit = -1

    def test_regeneration_limit_defaults_to_five(self) -> None:
        agent = _plain_agent()
        assert agent.regeneration_limit == 5

    @pytest.mark.parametrize("value", [-1, "1", 1.5])
    def test_regeneration_limit_rejects_negative_or_non_int(self, value: Any) -> None:
        with pytest.raises(ToolAgentError, match="regeneration_limit"):
            _plain_agent(regeneration_limit=value)

    def test_regeneration_limit_is_read_only(self) -> None:
        agent = _plain_agent()
        with pytest.raises(AttributeError):
            agent.regeneration_limit = 1  # type: ignore[misc]

    @pytest.mark.parametrize("value", [None, 1, 5])
    def test_tool_concurrency_limit_accepts_none_and_positive_int(self, value: int | None) -> None:
        agent = _plain_agent(tool_concurrency_limit=value)
        assert agent.tool_concurrency_limit == value

    @pytest.mark.parametrize("value", [0, -1, "1", 1.5])
    def test_tool_concurrency_limit_rejects_zero_negative_or_non_int(self, value: Any) -> None:
        with pytest.raises(ToolAgentError, match="tool_concurrency_limit"):
            _plain_agent(tool_concurrency_limit=value)

    def test_tool_concurrency_limit_setter_validates_after_construction(self) -> None:
        agent = _plain_agent()
        with pytest.raises(ToolAgentError, match="tool_concurrency_limit"):
            agent.tool_concurrency_limit = 0

    def test_tools_construction_kwarg_registers_via_register_tools(self) -> None:
        agent = _plain_agent(tools=[add, multiply])
        assert agent.has_tool("add")
        assert agent.has_tool("multiply")

    def test_constants_construction_kwargs_register_via_register_constants(self) -> None:
        agent = _plain_agent(
            constants=[1, "two"],
            constant_aliases=["a", "b"],
            constant_descriptions=["First.", "Second."],
        )
        assert agent.get_constant("A").value == 1
        assert agent.get_constant("B").description == "Second."

    def test_tool_instructions_defaults_to_none(self) -> None:
        agent = _plain_agent()
        assert agent.tool_instructions is None

    def test_tool_instructions_string_normalized(self) -> None:
        agent = _plain_agent(tool_instructions="Always say hi.")
        assert agent.tool_instructions == "Always say hi."

    def test_tool_instructions_whitespace_only_normalizes_to_none(self) -> None:
        agent = _plain_agent(tool_instructions="   ")
        assert agent.tool_instructions is None

    def test_tool_instructions_promptconfig_passthrough(self) -> None:
        config = PromptConfig(template="Static guidance.", description="d")
        agent = _plain_agent(tool_instructions=config)
        assert agent.tool_instructions == "Static guidance."

    def test_tool_instructions_invalid_type_raises(self) -> None:
        with pytest.raises(TypeError, match="tool_instructions"):
            _plain_agent(tool_instructions=123)  # type: ignore[arg-type]

    def test_tool_instructions_has_no_setter(self) -> None:
        agent = _plain_agent()
        with pytest.raises(AttributeError):
            agent.tool_instructions = "x"  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# Abstract contract
# --------------------------------------------------------------------------- #
class TestToolAgentAbstractContract:
    """ToolAgent re-abstracts every Agent task-lifecycle hook -- confirmed
    live via `ToolAgent.__abstractmethods__` (8 entries, no shared body for
    any of them): _initialize_task/think/async_think/prepare/async_prepare/
    act/async_act/_render_task_messages. There is no concrete, final act()/
    async_act() on ToolAgent anymore -- that shared generic loop was removed;
    each concrete family (PlanActAgent/ReActAgent/ScriptActAgent) owns its
    own full lifecycle now."""

    _EXPECTED_ABSTRACTMETHODS = frozenset({
        "_initialize_task", "think", "async_think", "prepare", "async_prepare",
        "act", "async_act", "_render_task_messages",
    })

    def test_toolagent_cannot_be_instantiated_directly(self) -> None:
        with pytest.raises(TypeError):
            ToolAgent(  # type: ignore[abstract]
                name="a", namespace="tests", description="d", llm_engine=FakeLLMEngine([]),
            )

    def test_toolagent_reabstracts_full_lifecycle_hook_set(self) -> None:
        assert ToolAgent.__abstractmethods__ == self._EXPECTED_ABSTRACTMETHODS

    def test_planact_agent_implements_every_required_hook(self) -> None:
        assert PlanActAgent.__abstractmethods__ == frozenset()

    def test_react_agent_implements_every_required_hook(self) -> None:
        assert ReActAgent.__abstractmethods__ == frozenset()

    def test_subclass_missing_hooks_cannot_instantiate(self) -> None:
        class _IncompleteAgent(ToolAgent):
            def _initialize_task(self, *, turns, prompt, inputs):  # type: ignore[override]
                raise NotImplementedError

            def think(self, task):  # type: ignore[override]
                return task

            async def async_think(self, task):  # type: ignore[override]
                return task
            # prepare/async_prepare/act/async_act/_render_task_messages
            # intentionally omitted.

        with pytest.raises(TypeError):
            _IncompleteAgent(  # type: ignore[abstract]
                name="a", namespace="tests", description="d", llm_engine=FakeLLMEngine([]),
            )


# --------------------------------------------------------------------------- #
# Namespace
# --------------------------------------------------------------------------- #
class TestToolAgentNamespace:
    def test_namespace_is_required(self) -> None:
        with pytest.raises(TypeError):
            PlanActAgent(  # type: ignore[call-arg]
                name="a",
                description="d",
                llm_engine=FakeLLMEngine([]),
            )

    def test_plan_act_agent_namespace_explicit(self) -> None:
        agent = PlanActAgent(
            name="a", namespace="planner_ns", description="d", llm_engine=FakeLLMEngine([]),
        )
        assert agent.namespace == "planner_ns"

    def test_react_agent_namespace_explicit(self) -> None:
        agent = ReActAgent(
            name="a", namespace="react_ns", description="d", llm_engine=FakeLLMEngine([]),
        )
        assert agent.namespace == "react_ns"


# --------------------------------------------------------------------------- #
# Base Agent post_invoke/post_result_key routing, exercised through
# PlanActAgent/ReActAgent with current wire-format scripted responses.
# --------------------------------------------------------------------------- #
class TestToolAgentPostInvokeRouting:
    def test_planact_agent_supports_post_invoke_passthrough(self) -> None:
        agent = make_planact_agent(
            [_scripted_plan(plan=[], return_value=5)],
            post_invoke=package_tool_result,
        )

        result = agent.invoke({"prompt": "run plan", "label": "planact"})

        assert result.result == {"label": "planact", "result": 5}

    def test_react_agent_supports_post_invoke_passthrough(self) -> None:
        agent = make_react_agent(
            [json.loads(react_step_json(call=RETURN_TOOL_NAME, arguments=[arg(RETURN_VALUE_FIELD, 7)]))],
            tool_calls_limit=1,
            post_invoke=package_tool_result,
        )

        result = agent.invoke({"prompt": "run react", "label": "react"})

        assert result.result == {"label": "react", "result": 7}


# --------------------------------------------------------------------------- #
# Tool registration
# --------------------------------------------------------------------------- #
class TestToolRegistration:
    def test_register_callable_adds_tool_under_bare_name(self) -> None:
        agent = _plain_agent()

        registered = agent.register_tool(add)

        assert registered is True
        assert agent.has_tool("add")
        assert agent.get_tool("add").invoke({"x": 1, "y": 2}).result == 3

    def test_register_tool_instance_stores_directly_not_retoolified(self) -> None:
        agent = _plain_agent()
        tool = Tool(function=add, name="adder", namespace="myns", description="Add values.")

        registered = agent.register_tool(tool)

        assert registered is True
        assert agent.get_tool("adder") is tool

    def test_register_callable_with_alias_uses_alias_as_effective_id(self) -> None:
        agent = _plain_agent()

        agent.register_tool(add, alias="plus")

        assert agent.has_tool("plus")
        assert not agent.has_tool("add")

    def test_register_duplicate_raises_by_default(self) -> None:
        agent = _plain_agent()
        agent.register_tool(add)

        with pytest.raises(ToolRegistrationError, match="already registered"):
            agent.register_tool(add)

    def test_register_duplicate_skip_returns_false_and_keeps_original(self) -> None:
        agent = _plain_agent()
        agent.register_tool(add, alias="calc")

        result = agent.register_tool(multiply, alias="calc", name_collision_policy="skip")

        assert result is False
        assert agent.get_tool("calc").invoke({"x": 2, "y": 3}).result == 5

    def test_register_duplicate_replace_replaces_tool(self) -> None:
        agent = _plain_agent()
        agent.register_tool(add, alias="calc")

        result = agent.register_tool(multiply, alias="calc", name_collision_policy="replace")

        assert result is True
        assert agent.get_tool("calc").invoke({"x": 3, "y": 4}).result == 12

    def test_register_invalid_collision_policy_raises(self) -> None:
        agent = _plain_agent()

        with pytest.raises(ToolRegistrationError, match="name_collision_policy"):
            agent.register_tool(add, name_collision_policy="bogus")  # type: ignore[arg-type]

    def test_register_invalid_alias_raises(self) -> None:
        agent = _plain_agent()

        with pytest.raises(ToolRegistrationError, match="alias"):
            agent.register_tool(add, alias="not a valid alias!")

    def test_register_unsupported_type_raises(self) -> None:
        agent = _plain_agent()
        with pytest.raises(ToolRegistrationError, match="unsupported component type"):
            agent.register_tool(42)  # type: ignore[arg-type]

    def test_get_tool_unknown_raises(self) -> None:
        agent = _plain_agent()
        with pytest.raises(ToolAgentError, match="unknown tool"):
            agent.get_tool("missing")

    def test_remove_tool_returns_true_then_false(self) -> None:
        agent = _plain_agent()
        agent.register_tool(add)

        assert agent.remove_tool("add") is True
        assert agent.remove_tool("add") is False

    def test_remove_reserved_tool_raises(self) -> None:
        agent = _plain_agent()
        with pytest.raises(ToolRegistrationError, match="reserved"):
            agent.remove_tool("make_sequence")

    def test_register_reserved_tool_name_raises(self) -> None:
        agent = _plain_agent()
        with pytest.raises(ToolRegistrationError, match="reserved"):
            agent.register_tool(add, alias="make_sequence")

    def test_clear_tools_keeps_reserved_tools_only(self) -> None:
        agent = _plain_agent()
        agent.register_tool(add)
        agent.register_tool(multiply)

        agent.clear_tools()

        remaining = agent.list_tools()
        assert set(remaining) == {"make_sequence", "make_dict"}

    def test_list_tools_returns_shallow_copy(self) -> None:
        agent = _plain_agent()
        agent.register_tool(add)

        snapshot = agent.list_tools()
        snapshot.clear()

        assert agent.has_tool("add")

    def test_get_tool_returns_atomic_invokable(self) -> None:
        agent = _plain_agent()
        agent.register_tool(add)
        assert isinstance(agent.get_tool("add"), AtomicInvokable)

    def test_actions_context_lists_registered_tools(self) -> None:
        agent = _plain_agent()
        agent.register_tool(add)

        context = agent.actions_context()

        assert "add(" in context

    def test_actions_context_renders_alias_in_place_of_bare_name(self) -> None:
        agent = _plain_agent()
        agent.register_tool(add, alias="plus")

        context = agent.actions_context()

        assert "plus(" in context
        assert "add(" not in context

    def test_register_tools_batch_registers_callables_under_bare_names(self) -> None:
        agent = _plain_agent()

        result = agent.register_tools([add, multiply])

        assert result is True
        assert agent.has_tool("add")
        assert agent.has_tool("multiply")

    def test_register_tools_aliases_length_mismatch_raises(self) -> None:
        agent = _plain_agent()
        with pytest.raises(ValueError, match="aliases must be the same length"):
            agent.register_tools([add, multiply], aliases=["only_one"])

    def test_register_tools_intra_batch_duplicate_raises(self) -> None:
        agent = _plain_agent()
        with pytest.raises(ToolRegistrationError, match="duplicate effective id"):
            agent.register_tools([add, add])

    def test_register_tools_skip_mode_excludes_existing_from_outcome(self) -> None:
        agent = _plain_agent()
        agent.register_tool(add)

        result = agent.register_tools([add, multiply], name_collision_policy="skip")

        assert result is False
        assert agent.has_tool("multiply")

    def test_register_tools_replace_mode_overwrites_existing(self) -> None:
        agent = _plain_agent()
        agent.register_tool(add, alias="calc")

        result = agent.register_tools([multiply], aliases=["calc"], name_collision_policy="replace")

        assert result is True
        assert agent.get_tool("calc").invoke({"x": 3, "y": 4}).result == 12

    def test_register_tools_mixed_invokables_and_callables(self) -> None:
        agent = _plain_agent()
        tool = Tool(function=multiply, name="mult", namespace="myns", description="Mult.")

        agent.register_tools([add, tool])

        assert agent.has_tool("add")
        assert agent.has_tool("mult")

    def test_register_tools_raise_mode_rejects_existing_toolbox_entry(self) -> None:
        agent = _plain_agent()
        agent.register_tool(add)

        with pytest.raises(ToolRegistrationError, match="already registered"):
            agent.register_tools([add])


# --------------------------------------------------------------------------- #
# Hub-expansion registration (A2AClientHub) via register_tools
# --------------------------------------------------------------------------- #
class TestA2AHubRegistration:
    """An A2AClientHub expands into one skill-mode tool per discovered
    Atomic skill, plus exactly one trailing generic-mode tool -- each under
    its own intrinsic bare name (never the dotted full_name), and never
    aliasable. Verified directly against the current register_tools/
    A2AProxyTool/toolify bodies, not assumed from the old (dotted-full_name)
    behavior."""

    def test_register_tools_with_a2a_hub_registers_skills_plus_generic(self) -> None:
        agent = _plain_agent()
        hub = FakeA2AClientHub(
            skills={
                "add": _a2a_sdk_skill_metadata(remote_name="add"),
                "multiply": _a2a_sdk_skill_metadata(remote_name="multiply"),
            }
        )

        agent.register_tools([hub])

        assert agent.has_tool("add")
        assert agent.has_tool("multiply")
        assert agent.has_tool("send_parts")
        assert set(agent.list_tools()) == {"make_sequence", "make_dict", "add", "multiply", "send_parts"}

    def test_register_tools_hub_include_names_filters_skills_only(self) -> None:
        """include_names is a hub-construction-time filter on
        get_atomic_skills() only -- the trailing generic tool is unaffected."""
        agent = _plain_agent()
        hub = FakeA2AClientHub(
            skills={
                "add": _a2a_sdk_skill_metadata(remote_name="add"),
                "multiply": _a2a_sdk_skill_metadata(remote_name="multiply"),
            },
            include_names=["add"],
        )

        agent.register_tools([hub])

        assert agent.has_tool("add")
        assert not agent.has_tool("multiply")
        assert agent.has_tool("send_parts")

    def test_register_tools_a2a_hub_zero_skills_still_registers_generic(self) -> None:
        agent = _plain_agent()
        hub = FakeA2AClientHub(skills={})

        agent.register_tools([hub])

        assert agent.has_tool("send_parts")
        assert not agent.has_tool("add")

    def test_a2a_hub_include_names_empty_list_raises_at_construction(self) -> None:
        with pytest.raises(ValueError):
            FakeA2AClientHub(include_names=[])

    def test_register_tools_a2a_hub_registered_tool_invokes_fake_hub(self) -> None:
        agent = _plain_agent()
        hub = FakeA2AClientHub()

        agent.register_tools([hub])
        tool = agent.get_tool("add")

        assert tool.invoke({"a": 3, "b": 4}).result == 17
        assert hub.skill_calls == [("add", {"a": 3, "b": 4})]

    def test_register_tools_hub_entry_cannot_take_alias(self) -> None:
        agent = _plain_agent()
        hub = FakeA2AClientHub()

        with pytest.raises(ValueError, match="cannot take an alias"):
            agent.register_tools([hub], aliases=["custom"])


# --------------------------------------------------------------------------- #
# Constant registration
# --------------------------------------------------------------------------- #
class TestConstantRegistration:
    def test_register_constant_with_alias_stores_normalized_spec(self) -> None:
        agent = _plain_agent()

        registered = agent.register_constant(
            {"user": "Ada"}, alias="user_context", description="Current user context."
        )

        assert registered is True
        assert agent.has_constant("user_context") is True
        spec = agent.get_constant("USER_CONTEXT")
        assert isinstance(spec, ConstantSpec)
        assert spec.name == "K_USER_CONTEXT"
        assert spec.value == {"user": "Ada"}
        assert spec.description == "Current user context."
        assert spec.type == "dict"

    def test_register_constant_auto_names_with_incrementing_counter(self) -> None:
        agent = _plain_agent()

        agent.register_constant(1)
        agent.register_constant(2)

        assert agent.get_constant("K_0").name == "K_0"
        assert agent.get_constant("K_0").value == 1
        assert agent.get_constant("K_1").name == "K_1"
        assert agent.get_constant("K_1").value == 2

    def test_register_constant_blank_description_defaults_to_no_details(self) -> None:
        agent = _plain_agent()
        agent.register_constant(5, alias="VALUE", description="   ")
        assert agent.get_constant("VALUE").description == "No details"

    def test_constants_property_returns_shallow_copy(self) -> None:
        agent = _plain_agent()
        agent.register_constant(1, alias="VALUE")

        constants = agent.constants
        constants.clear()

        assert len(agent.constants) == 1
        assert agent.has_constant("VALUE") is True

    def test_register_constant_duplicate_raises_by_default(self) -> None:
        agent = _plain_agent()
        agent.register_constant(1, alias="VALUE")

        with pytest.raises(ToolAgentError, match="constant already registered"):
            agent.register_constant(2, alias="VALUE")

    def test_register_constant_duplicate_skip_returns_false_and_keeps_original(self) -> None:
        agent = _plain_agent()
        agent.register_constant(1, alias="VALUE")

        result = agent.register_constant(2, alias="VALUE", name_collision_policy="skip")

        assert result is False
        assert agent.get_constant("VALUE").value == 1

    def test_register_constant_duplicate_replace_overwrites(self) -> None:
        agent = _plain_agent()
        agent.register_constant(1, alias="VALUE")

        result = agent.register_constant(2, alias="VALUE", name_collision_policy="replace")

        assert result is True
        assert agent.get_constant("VALUE").value == 2

    def test_register_constant_duplicate_suffix_renames(self) -> None:
        agent = _plain_agent()
        agent.register_constant(1, alias="VALUE")

        result = agent.register_constant(2, alias="VALUE", name_collision_policy="suffix")

        assert result is True
        assert agent.get_constant("VALUE").value == 1
        assert agent.get_constant("VALUE_0").value == 2

    def test_register_constant_invalid_alias_raises(self) -> None:
        agent = _plain_agent()
        with pytest.raises(ToolAgentError, match="alias must be None or a non-empty string"):
            agent.register_constant(1, alias="   ")

    def test_register_constant_invalid_collision_policy_raises(self) -> None:
        agent = _plain_agent()
        with pytest.raises(ToolRegistrationError, match="name_collision_policy"):
            agent.register_constant(1, alias="VALUE", name_collision_policy="bogus")  # type: ignore[arg-type]

    def test_register_constants_batch_adds_all_with_aliases_and_descriptions(self) -> None:
        agent = _plain_agent()

        result = agent.register_constants(
            [1, "two", (1, 2)],
            aliases=["a", "b", "c"],
            descriptions=[None, "Second value.", "Coordinates."],
        )

        assert result is True
        assert agent.get_constant("A").value == 1
        assert agent.get_constant("B").description == "Second value."
        assert agent.get_constant("C").value == (1, 2)

    def test_register_constants_aliases_length_mismatch_raises(self) -> None:
        agent = _plain_agent()
        with pytest.raises(ValueError, match="aliases must be the same length"):
            agent.register_constants([1, 2], aliases=["a"])

    def test_register_constants_descriptions_length_mismatch_raises(self) -> None:
        agent = _plain_agent()
        with pytest.raises(ValueError, match="descriptions must be the same length"):
            agent.register_constants([1, 2], descriptions=["only one"])

    def test_register_constants_intra_batch_duplicate_raises(self) -> None:
        agent = _plain_agent()
        with pytest.raises(ToolAgentError, match="duplicate constant name in batch"):
            agent.register_constants([1, 2], aliases=["VALUE", "VALUE"])

    def test_register_constants_suffix_resolves_intra_batch_and_existing_collisions(self) -> None:
        agent = _plain_agent()
        agent.register_constant(0, alias="VALUE")

        agent.register_constants([1, 2], aliases=["VALUE", "VALUE"], name_collision_policy="suffix")

        assert agent.get_constant("VALUE").value == 0
        assert agent.get_constant("VALUE_0").value == 1
        assert agent.get_constant("VALUE_1").value == 2

    def test_remove_constant_returns_true_then_false(self) -> None:
        agent = _plain_agent()
        agent.register_constant(1, alias="A")

        assert agent.remove_constant("A") is True
        assert agent.remove_constant("A") is False

    def test_clear_constants_removes_all(self) -> None:
        agent = _plain_agent()
        agent.register_constants([1, 2], aliases=["A", "B"])

        agent.clear_constants()

        assert agent.constants == {}

    def test_get_constant_unknown_raises(self) -> None:
        agent = _plain_agent()
        with pytest.raises(ToolAgentError, match="unknown constant"):
            agent.get_constant("MISSING")

    def test_update_constant_description_replaces_only_description(self) -> None:
        agent = _plain_agent()
        agent.register_constant({"a": 1}, alias="PAYLOAD", description="Old.")

        agent.update_constant_description("PAYLOAD", "New description.")

        spec = agent.get_constant("PAYLOAD")
        assert spec.description == "New description."
        assert spec.value == {"a": 1}
        assert spec.name == "K_PAYLOAD"

    def test_update_constant_description_unknown_raises(self) -> None:
        agent = _plain_agent()
        with pytest.raises(ToolAgentError, match="unknown constant"):
            agent.update_constant_description("MISSING", "text")

    def test_update_constant_description_rejects_blank(self) -> None:
        agent = _plain_agent()
        agent.register_constant(1, alias="A")
        with pytest.raises(ToolAgentError, match="description must be a non-empty string"):
            agent.update_constant_description("A", "   ")

    def test_constants_context_hides_values_and_renders_metadata(self) -> None:
        agent = _plain_agent()
        agent.register_constant("super-secret-value", alias="SECRET", description="Sensitive value.")
        agent.register_constant(3, alias="UNLABELED")

        context = agent.constants_context()

        assert "K_SECRET: str" in context
        assert "Sensitive value." in context
        assert "K_UNLABELED: int" in context
        assert "No details" in context
        assert "super-secret-value" not in context

    def test_constants_context_empty_registry_message(self) -> None:
        agent = _plain_agent()
        assert agent.constants_context() == "No constants registered."


# --------------------------------------------------------------------------- #
# Shared rendering surface (actions_context/constants_context helpers,
# _render_system_message/_extra_system_context, render_turn/_turn_position,
# _copy_for_task_namespace) -- new 5 coverage: these are genuinely
# ToolAgent-owned and were never directly tested anywhere else.
# --------------------------------------------------------------------------- #
class TestDocstringBlockRendering:
    def test_single_line_description_closes_on_same_line(self) -> None:
        assert ToolAgent._render_docstring_block("hello") == '    """hello"""'

    def test_multi_line_description_indents_continuation(self) -> None:
        block = ToolAgent._render_docstring_block("first\nsecond")
        assert block == '    """first\n    second\n    """'

    def test_blank_continuation_lines_left_bare(self) -> None:
        block = ToolAgent._render_docstring_block("first\n\nthird")
        assert block == '    """first\n\n    third\n    """'


class TestSystemMessageRendering:
    def test_extra_system_context_defaults_to_empty_dict(self) -> None:
        plan_agent = _plain_agent()
        react_agent = ReActAgent(
            name="r", namespace="tests", description="d", llm_engine=FakeLLMEngine([]),
        )
        assert plan_agent._extra_system_context() == {}
        assert react_agent._extra_system_context() == {}

    def test_render_system_message_injects_tools_and_constants(self) -> None:
        agent = _plain_agent()
        agent.register_tool(add)
        agent.register_constant(7, alias="SEVEN", description="Lucky number.")
        task = agent._initialize_task(turns=[], prompt="do work", inputs={})

        rendered = agent._render_system_message(task)

        assert len(rendered) == 1
        assert rendered[0]["role"] == "system"
        assert "add(" in rendered[0]["content"]
        assert "K_SEVEN: int" in rendered[0]["content"]

    def test_render_system_message_empty_when_system_prompt_name_none(self) -> None:
        agent = _plain_agent()
        task = agent._initialize_task(turns=[], prompt="do work", inputs={})
        task.system_prompt_name = None

        assert agent._render_system_message(task) == []

    def test_render_system_message_omits_banner_when_tool_instructions_unset(self) -> None:
        agent = _plain_agent()
        task = agent._initialize_task(turns=[], prompt="do work", inputs={})

        content = agent._render_system_message(task)[0]["content"]

        assert "ADDITIONAL TOOL INSTRUCTIONS" not in content

    def test_render_system_message_appends_tool_instructions_banner_when_set(self) -> None:
        agent = _plain_agent(tool_instructions="Always say hi.")
        task = agent._initialize_task(turns=[], prompt="do work", inputs={})

        content = agent._render_system_message(task)[0]["content"]

        assert "ADDITIONAL TOOL INSTRUCTIONS" in content
        assert "Always say hi." in content
        assert content.index("ADDITIONAL TOOL INSTRUCTIONS") > content.index("AVAILABLE TOOLS")

    def test_render_system_message_tool_instructions_promptconfig_shares_render_context(self) -> None:
        config = PromptConfig(template="See: {TOOLS}", description="d")
        agent = _plain_agent(tool_instructions=config)
        agent.register_tool(add)
        task = agent._initialize_task(turns=[], prompt="do work", inputs={})

        content = agent._render_system_message(task)[0]["content"]

        banner = content[content.index("ADDITIONAL TOOL INSTRUCTIONS"):]
        assert "add(" in banner

    def test_tool_instructions_field_becomes_declared_parameter(self) -> None:
        agent = _plain_agent(tool_instructions="Repeat {loops} times.")
        assert "loops" in {p.name for p in agent.parameters}

    def test_render_system_message_tool_instructions_renders_against_task_inputs(self) -> None:
        agent = _plain_agent(tool_instructions="Repeat {loops} times.")
        task = agent._initialize_task(
            turns=[], prompt="do work", inputs={"prompt": "do work", "loops": 3}
        )

        content = agent._render_system_message(task)[0]["content"]

        banner = content[content.index("ADDITIONAL TOOL INSTRUCTIONS"):]
        assert "Repeat 3 times." in banner

    def test_invoke_missing_tool_instructions_field_raises_from_render_not_entry(self) -> None:
        """Declaring a `tool_instructions` field as an `extra_parameters`
        entry (this fix) only widens `filter_inputs` to retain a
        caller-supplied value under that name -- `filter_inputs` (and
        `Agent.invoke` generally) injects declared defaults but never raises
        for a declared, non-defaulted parameter the caller simply omits.
        A field with no explicit default (the case here -- a bare
        `tool_instructions` string never attaches `field_specs`) is
        therefore still *discovered* missing only once rendering actually
        needs it, exactly like the already-shipped
        `BasicAgent.role_prompt`/`ThinkingAgent.thinking_instructions`
        precedent behaves today for the same scenario (confirmed by direct
        reproduction, not assumed). So omitting `loops` still raises
        `PromptConfig.render`'s `ValueError` -- this test pins down that
        real, current behavior rather than a hoped-for distinct entry-level
        error that no code path in this codebase actually produces."""
        agent = _plain_agent(tool_instructions="Repeat {loops} times.")

        with pytest.raises(ValueError, match=r"PromptConfig\.render.*loops"):
            agent.invoke({"prompt": "do work"})

    def test_render_system_message_base_prompt_unaffected_by_task_inputs(self) -> None:
        agent = _plain_agent(tool_instructions="Repeat {loops} times.")
        task = agent._initialize_task(
            turns=[], prompt="do work", inputs={"prompt": "do work", "loops": 3}
        )
        content = agent._render_system_message(task)[0]["content"]

        baseline_agent = _plain_agent()
        baseline_task = baseline_agent._initialize_task(
            turns=[], prompt="do work", inputs={"prompt": "do work", "loops": 3}
        )
        baseline_content = baseline_agent._render_system_message(baseline_task)[0]["content"]

        # The base prompt's own rendering (TOOLS/CONSTANTS only) is a
        # byte-identical prefix of the tool_instructions-bearing render --
        # the banner is appended after, never templated in. No stray "3"
        # leaks into that shared prefix.
        assert content.startswith(baseline_content)
        assert "3" not in baseline_content


class TestRenderTurnAndTurnPosition:
    def test_render_turn_labels_turns_by_position(self) -> None:
        agent = make_planact_agent(
            [
                _scripted_plan(plan=[], return_value=1),
                _scripted_plan(plan=[], return_value=2),
            ],
            context_enabled=True,
        )

        agent.invoke({"prompt": "first"})
        agent.invoke({"prompt": "second"})

        turns = agent.get_conversation()
        assert len(turns) == 2

        first_messages = agent.render_turn(turns[0])
        second_messages = agent.render_turn(turns[1])

        assert first_messages[-1]["content"].startswith("task_result_0: int = ")
        assert second_messages[-1]["content"].startswith("task_result_1: int = ")

    def test_turn_position_is_zero_for_a_root_turn(self) -> None:
        agent = make_planact_agent([_scripted_plan(plan=[], return_value=1)], context_enabled=True)
        agent.invoke({"prompt": "only turn"})

        turn = agent.get_conversation()[0]

        assert agent._turn_position(turn) == 0


class TestCopyForTaskNamespace:
    @pytest.mark.parametrize("value", [1, 1.5, "text", True, None, b"bytes", 5 + 2j])
    def test_atomic_immutable_values_returned_as_is(self, value: Any) -> None:
        agent = _plain_agent()
        assert agent._copy_for_task_namespace(value) is value

    def test_mutable_value_is_deep_copied(self) -> None:
        agent = _plain_agent()
        original = {"items": [1, 2, 3]}

        copied = agent._copy_for_task_namespace(original)

        assert copied == original
        assert copied is not original
        assert copied["items"] is not original["items"]


# --------------------------------------------------------------------------- #
# to_dict() diagnostics ToolAgent itself owns.
# --------------------------------------------------------------------------- #
class TestToolAgentToDict:
    def test_to_dict_includes_execution_knobs(self) -> None:
        agent = _plain_agent(tool_calls_limit=3, regeneration_limit=2, tool_concurrency_limit=4)

        d = agent.to_dict()

        assert d["tool_calls_limit"] == 3
        assert d["regeneration_limit"] == 2
        assert d["tool_concurrency_limit"] == 4

    def test_to_dict_includes_tools_mapping(self) -> None:
        agent = _plain_agent()
        agent.register_tool(add)

        d = agent.to_dict()

        assert "add" in d["tools"]
        assert "make_sequence" in d["tools"]
