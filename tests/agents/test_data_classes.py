from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from atomic_agentic.models.agents.records import AgentRecord, LLMRecord, ToolAgentRecord
from atomic_agentic.models.agents.blackboard_models import BlackboardSlot, ConstantSpec
from atomic_agentic.core.constants import NO_VAL
from atomic_agentic.models.results import AtomicResult, LLMModelData, LLMResult, TokenUsage


def make_token_usage(*, input_tokens: int = 10, generated_tokens: int = 5) -> TokenUsage:
    return TokenUsage(
        input_tokens=input_tokens,
        generated_tokens=generated_tokens,
        total_tokens=input_tokens + generated_tokens,
    )


def make_model_data(*, provider: str = "openai") -> LLMModelData:
    return LLMModelData(provider=provider)


def make_llm_result(*, text: str = "generated text", invoker_id: str = "engine-1") -> LLMResult:
    started_at = datetime.now(timezone.utc)
    return LLMResult(
        result=text,
        invoker_id=invoker_id,
        started_at=started_at,
        ended_at=started_at + timedelta(seconds=1),
        token_usage=make_token_usage(),
        model_data=make_model_data(),
    )


def make_llm_record(*, user_prompt: str = "generate a response", text: str = "generated text") -> LLMRecord:
    return LLMRecord(user_prompt=user_prompt, llm_result=make_llm_result(text=text))


def make_atomic_result(*, value: Any = 5, invoker_id: str = "tool-1") -> AtomicResult:
    started_at = datetime.now(timezone.utc)
    return AtomicResult(
        result=value,
        invoker_id=invoker_id,
        started_at=started_at,
        ended_at=started_at + timedelta(seconds=1),
    )


class TestConstantSpec:
    def test_valid_spec_normalizes_name_description_and_derives_type(self) -> None:
        value = {"items": [1, 2]}

        spec = ConstantSpec(
            name=" PAYLOAD ",
            value=value,
            description=" Runtime payload. ",
            inline_limit=20,
        )

        assert spec.name == "PAYLOAD"
        assert spec.value is value
        assert spec.description == "Runtime payload."
        assert spec.inline_limit == 20
        assert spec.type == "dict"

    def test_blank_description_normalizes_to_none(self) -> None:
        spec = ConstantSpec(name="VALUE", value=1, description="   ")

        assert spec.description is None

    def test_to_dict_includes_value_metadata_and_derived_type(self) -> None:
        spec = ConstantSpec(
            name="THRESHOLD",
            value=0.75,
            description="Decision threshold.",
            inline_limit=8,
        )

        assert spec.to_dict() == {
            "name": "THRESHOLD",
            "value": 0.75,
            "description": "Decision threshold.",
            "inline_limit": 8,
            "type": "float",
        }

    @pytest.mark.parametrize(
        "name",
        ["", " ", "1VALUE", "bad-name", "bad name", None],
    )
    def test_rejects_invalid_name(self, name: Any) -> None:
        with pytest.raises(ValueError, match="ConstantSpec.name"):
            ConstantSpec(name=name, value=1)  # type: ignore[arg-type]

    @pytest.mark.parametrize("description", [1, True, object()])
    def test_rejects_non_string_description(self, description: Any) -> None:
        with pytest.raises(TypeError, match="description"):
            ConstantSpec(name="VALUE", value=1, description=description)  # type: ignore[arg-type]

    @pytest.mark.parametrize("inline_limit", [0, -1, 1.5, True, "5"])
    def test_rejects_invalid_inline_limit(self, inline_limit: Any) -> None:
        with pytest.raises(ValueError, match="inline_limit"):
            ConstantSpec(name="VALUE", value=1, inline_limit=inline_limit)  # type: ignore[arg-type]

    def test_is_frozen(self) -> None:
        spec = ConstantSpec(name="VALUE", value=1)

        with pytest.raises(FrozenInstanceError):
            spec.name = "OTHER"  # type: ignore[misc]


class TestLLMRecord:
    def test_valid_record_exposes_user_prompt_and_llm_result(self) -> None:
        llm_result = make_llm_result(text="hello world")

        record = LLMRecord(user_prompt="say hello", llm_result=llm_result)

        assert record.user_prompt == "say hello"
        assert record.llm_result is llm_result

    def test_to_dict_returns_user_prompt_and_serialized_llm_result(self) -> None:
        llm_result = make_llm_result(text="hello world")
        record = LLMRecord(user_prompt="say hello", llm_result=llm_result)

        assert record.to_dict() == {
            "user_prompt": "say hello",
            "llm_result": llm_result.to_dict(),
        }

    def test_rejects_non_string_user_prompt(self) -> None:
        with pytest.raises(TypeError, match="user_prompt"):
            LLMRecord(user_prompt=123, llm_result=make_llm_result())  # type: ignore[arg-type]

    def test_rejects_non_llm_result_llm_result(self) -> None:
        with pytest.raises(TypeError, match="llm_result"):
            LLMRecord(user_prompt="say hello", llm_result="not an llm result")  # type: ignore[arg-type]

    def test_is_frozen(self) -> None:
        record = make_llm_record()

        with pytest.raises(FrozenInstanceError):
            record.user_prompt = "other"  # type: ignore[misc]


class TestAgentRecord:
    def test_to_dict_returns_explicit_canonical_shape(self) -> None:
        llm_record = make_llm_record(user_prompt="write a summary", text="raw assistant text")
        record = AgentRecord(
            user_prompt="write a summary",
            generated_response="raw assistant text",
            final_response={"final": "processed assistant text"},
            llm_records=(llm_record,),
            run_id="run-123",
        )

        assert record.to_dict() == {
            "user_prompt": "write a summary",
            "generated_response": "raw assistant text",
            "final_response": {"final": "processed assistant text"},
            "llm_records": [llm_record.to_dict()],
            "run_id": "run-123",
        }

    def test_to_dict_returns_independent_top_level_mapping(self) -> None:
        record = AgentRecord(
            user_prompt="collect items",
            generated_response="raw",
            final_response={"items": ["a", "b"]},
            llm_records=(make_llm_record(),),
            run_id="run-456",
        )

        data = record.to_dict()
        data["user_prompt"] = "mutated"

        assert record.user_prompt == "collect items"
        assert record.final_response == {"items": ["a", "b"]}

    def test_normalizes_llm_records_into_immutable_tuple(self) -> None:
        llm_record = make_llm_record()

        record = AgentRecord(
            user_prompt="run",
            generated_response="raw",
            final_response="final",
            llm_records=[llm_record, llm_record],
            run_id="run-789",
        )

        assert record.llm_records == (llm_record, llm_record)
        assert isinstance(record.llm_records, tuple)

    def test_normalizes_run_id_by_stripping_whitespace(self) -> None:
        record = AgentRecord(
            user_prompt="run",
            generated_response="raw",
            final_response="final",
            llm_records=(make_llm_record(),),
            run_id="  run-789  ",
        )

        assert record.run_id == "run-789"

    def test_rejects_non_string_user_prompt(self) -> None:
        with pytest.raises(TypeError, match="user_prompt"):
            AgentRecord(
                user_prompt=123,  # type: ignore[arg-type]
                generated_response="raw",
                final_response="final",
                llm_records=(make_llm_record(),),
                run_id="run-1",
            )

    def test_rejects_non_sequence_llm_records(self) -> None:
        with pytest.raises(TypeError, match="llm_records"):
            AgentRecord(
                user_prompt="run",
                generated_response="raw",
                final_response="final",
                llm_records=make_llm_record(),  # type: ignore[arg-type]
                run_id="run-1",
            )

    def test_rejects_empty_llm_records(self) -> None:
        with pytest.raises(ValueError, match="llm_records"):
            AgentRecord(
                user_prompt="run",
                generated_response="raw",
                final_response="final",
                llm_records=(),
                run_id="run-1",
            )

    def test_rejects_non_llm_record_items_in_llm_records(self) -> None:
        with pytest.raises(TypeError, match="llm_records"):
            AgentRecord(
                user_prompt="run",
                generated_response="raw",
                final_response="final",
                llm_records=(make_llm_record(), "not a record"),
                run_id="run-1",
            )

    def test_rejects_non_string_run_id(self) -> None:
        with pytest.raises(TypeError, match="run_id"):
            AgentRecord(
                user_prompt="run",
                generated_response="raw",
                final_response="final",
                llm_records=(make_llm_record(),),
                run_id=123,  # type: ignore[arg-type]
            )

    def test_rejects_blank_run_id(self) -> None:
        with pytest.raises(ValueError, match="run_id"):
            AgentRecord(
                user_prompt="run",
                generated_response="raw",
                final_response="final",
                llm_records=(make_llm_record(),),
                run_id="   ",
            )

    def test_is_frozen(self) -> None:
        record = AgentRecord(
            user_prompt="run",
            generated_response="raw",
            final_response="final",
            llm_records=(make_llm_record(),),
            run_id="run-1",
        )

        with pytest.raises(FrozenInstanceError):
            record.user_prompt = "other"  # type: ignore[misc]


class TestToolAgentRecord:
    def test_is_an_agent_record(self) -> None:
        record = ToolAgentRecord(
            user_prompt="run tools",
            generated_response=42,
            final_response={"value": 42},
            llm_records=(make_llm_record(),),
            run_id="run-1",
            blackboard_start=3,
            blackboard_end=6,
        )

        assert isinstance(record, AgentRecord)
        assert record.blackboard_start == 3
        assert record.blackboard_end == 6

    def test_to_dict_returns_inherited_agent_record_shape(self) -> None:
        llm_record = make_llm_record()
        record = ToolAgentRecord(
            user_prompt="run tools",
            generated_response=42,
            final_response={"value": 42},
            llm_records=(llm_record,),
            run_id="run-1",
            blackboard_start=3,
            blackboard_end=6,
        )

        # ToolAgentRecord does not override AgentRecord.to_dict(); the
        # blackboard span is exposed only through the dataclass attributes.
        assert record.to_dict() == {
            "user_prompt": "run tools",
            "generated_response": 42,
            "final_response": {"value": 42},
            "llm_records": [llm_record.to_dict()],
            "run_id": "run-1",
        }

    def test_blackboard_span_defaults_to_none(self) -> None:
        record = ToolAgentRecord(
            user_prompt="run without context",
            generated_response="raw",
            final_response="final",
            llm_records=(make_llm_record(),),
            run_id="run-2",
        )

        assert record.blackboard_start is None
        assert record.blackboard_end is None

    def test_inherits_agent_record_validation(self) -> None:
        with pytest.raises(TypeError, match="user_prompt"):
            ToolAgentRecord(
                user_prompt=123,  # type: ignore[arg-type]
                generated_response="raw",
                final_response="final",
                llm_records=(make_llm_record(),),
                run_id="run-3",
            )

    def test_is_frozen(self) -> None:
        record = ToolAgentRecord(
            user_prompt="run",
            generated_response="raw",
            final_response="final",
            llm_records=(make_llm_record(),),
            run_id="run-4",
        )

        with pytest.raises(FrozenInstanceError):
            record.blackboard_start = 1  # type: ignore[misc]


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

    def test_result_field_stores_full_atomic_result_envelope(self) -> None:
        envelope = make_atomic_result(value=8)

        slot = BlackboardSlot(step=2, result=envelope, status="executed")

        assert slot.is_executed() is True
        assert slot.result is envelope
        assert isinstance(slot.result, AtomicResult)
        # Consumers unwrap the caller-facing payload explicitly.
        assert slot.result.result == 8

    def test_copy_preserves_atomic_result_envelope_identity(self) -> None:
        envelope = make_atomic_result(value=13)
        slot = BlackboardSlot(step=1, result=envelope, status="executed")

        copied = slot.copy()

        assert copied.result is envelope
        assert copied.result.result == 13

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
