from __future__ import annotations

from typing import Any

import pytest

from atomic_agentic.agents.data_classes import AgentTurn, ToolAgentTurn, BlackboardSlot
from atomic_agentic.core.sentinels import NO_VAL


class TestAgentTurn:
    def test_to_dict_returns_prompt_raw_response_and_final_response(self) -> None:
        turn = AgentTurn(
            prompt="write a summary",
            raw_response="raw assistant text",
            final_response={"final": "processed assistant text"},
        )

        assert turn.to_dict() == {
            "prompt": "write a summary",
            "raw_response": "raw assistant text",
            "final_response": {"final": "processed assistant text"},
        }

    def test_to_dict_returns_independent_top_level_mapping(self) -> None:
        turn = AgentTurn(
            prompt="collect items",
            raw_response="raw",
            final_response={"items": ["a", "b"]},
        )

        data = turn.to_dict()
        data["prompt"] = "mutated"

        assert turn.prompt == "collect items"
        assert turn.final_response == {"items": ["a", "b"]}


class TestToolAgentTurn:
    def test_to_dict_includes_agent_turn_fields_and_blackboard_span(self) -> None:
        turn = ToolAgentTurn(
            prompt="run tools",
            raw_response=42,
            final_response={"value": 42},
            blackboard_start=3,
            blackboard_end=6,
        )

        assert turn.to_dict() == {
            "prompt": "run tools",
            "raw_response": 42,
            "final_response": {"value": 42},
            "blackboard_start": 3,
            "blackboard_end": 6,
        }

    def test_blackboard_span_defaults_to_none(self) -> None:
        turn = ToolAgentTurn(
            prompt="run without context",
            raw_response="raw",
            final_response="final",
        )

        assert turn.blackboard_start is None
        assert turn.blackboard_end is None
        assert turn.to_dict() == {
            "prompt": "run without context",
            "raw_response": "raw",
            "final_response": "final",
            "blackboard_start": None,
            "blackboard_end": None,
        }


class TestBlackboardSlot:
    def test_default_slot_is_empty_with_default_metadata(self) -> None:
        slot = BlackboardSlot(step=0)

        assert slot.step == 0
        assert slot.tool is NO_VAL
        assert slot.args is NO_VAL
        assert slot.resolved_args is NO_VAL
        assert slot.result is NO_VAL
        assert slot.error is NO_VAL
        assert slot.status == "empty"
        assert slot.step_dependencies == ()
        assert slot.await_step is NO_VAL
        assert slot.is_empty() is True
        assert slot.is_planned() is False
        assert slot.is_prepared() is False
        assert slot.is_executed() is False
        assert slot.is_failed() is False

    @pytest.mark.parametrize(
        ("status", "method_name"),
        [
            ("empty", "is_empty"),
            ("planned", "is_planned"),
            ("prepared", "is_prepared"),
            ("executed", "is_executed"),
            ("failed", "is_failed"),
        ],
    )
    def test_status_helpers_are_status_based(self, status: str, method_name: str) -> None:
        slot = BlackboardSlot(step=0, status=status)

        assert getattr(slot, method_name)() is True

    @pytest.mark.parametrize("step", [-1, "0", 1.5, True])
    def test_rejects_invalid_step(self, step: Any) -> None:
        with pytest.raises(ValueError, match="step"):
            BlackboardSlot(step=step)  # type: ignore[arg-type]

    def test_rejects_invalid_status(self) -> None:
        with pytest.raises(ValueError, match="status"):
            BlackboardSlot(step=0, status="done")

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            (NO_VAL, ()),
            (None, ()),
            (2, (2,)),
            ([2, 0, 2], (0, 2)),
            ((3, 1), (1, 3)),
            ({5, 4, 5}, (4, 5)),
            (frozenset({8, 7}), (7, 8)),
        ],
    )
    def test_normalizes_step_dependencies(self, raw: Any, expected: tuple[int, ...]) -> None:
        slot = BlackboardSlot(step=0, step_dependencies=raw)

        assert slot.step_dependencies == expected

    @pytest.mark.parametrize("deps", [[-1], [0, "1"], [True], object()])
    def test_rejects_invalid_step_dependencies(self, deps: Any) -> None:
        with pytest.raises(ValueError, match="step_dependencies"):
            BlackboardSlot(step=0, step_dependencies=deps)

    @pytest.mark.parametrize("await_step", [-1, "0", 1.5, True, None])
    def test_rejects_invalid_await_step(self, await_step: Any) -> None:
        with pytest.raises(ValueError, match="await_step"):
            BlackboardSlot(step=0, await_step=await_step)

    def test_copy_returns_shallow_copy_with_all_fields(self) -> None:
        args = {"x": [1, 2]}
        result = {"value": 3}
        slot = BlackboardSlot(
            step=4,
            tool="Tool.tests.add",
            args=args,
            resolved_args=args,
            result=result,
            error=NO_VAL,
            status="executed",
            step_dependencies=(1, 2),
            await_step=1,
        )

        copied = slot.copy()

        assert copied is not slot
        assert copied == slot
        assert copied.args is args
        assert copied.resolved_args is args
        assert copied.result is result

    def test_from_dict_constructs_slot_with_alias_await(self) -> None:
        slot = BlackboardSlot.from_dict(
            {
                "step": 3,
                "tool": "Tool.tests.multiply",
                "args": {"x": "<<__s0__>>", "y": 2},
                "resolved_args": {"x": 4, "y": 2},
                "result": 8,
                "error": NO_VAL,
                "status": "executed",
                "step_dependencies": [0, 2, 2],
                "await": 2,
            }
        )

        assert slot.step == 3
        assert slot.tool == "Tool.tests.multiply"
        assert slot.args == {"x": "<<__s0__>>", "y": 2}
        assert slot.resolved_args == {"x": 4, "y": 2}
        assert slot.result == 8
        assert slot.error is NO_VAL
        assert slot.status == "executed"
        assert slot.step_dependencies == (0, 2)
        assert slot.await_step == 2

    def test_from_dict_defaults_optional_fields(self) -> None:
        slot = BlackboardSlot.from_dict({"step": 0})

        assert slot.tool is NO_VAL
        assert slot.args is NO_VAL
        assert slot.resolved_args is NO_VAL
        assert slot.result is NO_VAL
        assert slot.error is NO_VAL
        assert slot.status == "empty"
        assert slot.step_dependencies == ()
        assert slot.await_step is NO_VAL

    def test_from_dict_rejects_non_mapping(self) -> None:
        with pytest.raises(TypeError, match="requires a mapping"):
            BlackboardSlot.from_dict([("step", 0)])  # type: ignore[arg-type]

    def test_from_dict_rejects_missing_step(self) -> None:
        with pytest.raises(ValueError, match="missing required key"):
            BlackboardSlot.from_dict({"tool": "Tool.tests.add"})

    def test_from_dict_rejects_unsupported_keys(self) -> None:
        with pytest.raises(ValueError, match="unsupported keys"):
            BlackboardSlot.from_dict({"step": 0, "extra": True})

    def test_from_dict_rejects_both_await_keys(self) -> None:
        with pytest.raises(ValueError, match="both 'await' and 'await_step'"):
            BlackboardSlot.from_dict({"step": 0, "await": 0, "await_step": 0})

    def test_to_dict_shape_includes_status_and_dependency_fields(self) -> None:
        slot = BlackboardSlot(
            step=0,
            tool="Tool.tests.add",
            args={"x": 1, "y": 2},
            resolved_args={"x": 1, "y": 2},
            result=3,
            error=NO_VAL,
            status="executed",
            step_dependencies=(0,),
            await_step=0,
        )

        assert slot.to_dict() == {
            "step": 0,
            "tool": "Tool.tests.add",
            "args": {"x": 1, "y": 2},
            "resolved_args": {"x": 1, "y": 2},
            "result": 3,
            "error": NO_VAL,
            "status": "executed",
            "step_dependencies": (0,),
            "await_step": 0,
        }
