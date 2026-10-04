from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from atomic_agentic.constants.agents import (
    ATTR_CALL_ALIAS,
    PY_BUILTIN_ALIAS,
    RETURN_ALIAS,
    RHS_ASSIGN_ALIAS,
)
from atomic_agentic.models.agents.records import (
    AgentRecord,
    LLMRecord,
    ScriptActAgentRecord,
    ScriptActAgentToolUsage,
    ThinkingAgentRecord,
    ToolAgentRecord,
)
from atomic_agentic.models.agents.blackboard_models import ToolStatement, ConstantSpec
from atomic_agentic.constants.core import NO_VAL
from atomic_agentic.models.results import LLMModelData, LLMResult, TokenUsage
from atomic_agentic.models.results.agents import AgentResult, ToolUsageRecord


def make_token_usage(*, input_tokens: int = 10, generated_tokens: int = 5) -> TokenUsage:
    return TokenUsage(
        input_tokens=input_tokens,
        generated_tokens=generated_tokens,
        total_tokens=input_tokens + generated_tokens,
        response_tokens=generated_tokens,
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


def make_llm_record(*, text: str = "generated text") -> LLMRecord:
    return LLMRecord(
        messages=({"role": "user", "content": "generate a response"},),
        llm_result=make_llm_result(text=text),
    )


def make_agent_result(*, value: Any = "final output") -> AgentResult:
    started_at = datetime.now(timezone.utc)
    return AgentResult(
        result=value,
        invoker_id="agent-1",
        started_at=started_at,
        ended_at=started_at + timedelta(seconds=1),
        llm_token_usage=(make_token_usage(),),
        llm_model_data=make_model_data(),
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
    def test_valid_record_stores_messages_and_llm_result(self) -> None:
        llm_result = make_llm_result(text="hello world")
        record = LLMRecord(
            messages=[{"role": "user", "content": "say hello"}],
            llm_result=llm_result,
        )
        assert record.messages == ({"role": "user", "content": "say hello"},)
        assert record.llm_result is llm_result

    def test_messages_list_normalized_to_tuple(self) -> None:
        record = LLMRecord(
            messages=[{"role": "user", "content": "hi"}],
            llm_result=make_llm_result(),
        )
        assert isinstance(record.messages, tuple)

    def test_multi_message_delta_stored_in_order(self) -> None:
        msgs = [
            {"role": "user", "content": "original task"},
            {"role": "assistant", "content": "running plan"},
            {"role": "user", "content": "next step"},
        ]
        record = LLMRecord(messages=msgs, llm_result=make_llm_result())
        assert record.messages == tuple(msgs)

    def test_to_dict_serializes_messages_as_list(self) -> None:
        llm_result = make_llm_result(text="hello world")
        msgs = [{"role": "user", "content": "say hello"}]
        record = LLMRecord(messages=msgs, llm_result=llm_result)
        assert record.to_dict() == {
            "messages": msgs,
            "llm_result": llm_result.to_dict(),
            "system_prompt_name": None,
        }

    def test_system_prompt_name_defaults_to_none(self) -> None:
        record = make_llm_record()
        assert record.system_prompt_name is None

    def test_system_prompt_name_accepts_string(self) -> None:
        record = LLMRecord(
            messages=({"role": "user", "content": "hi"},),
            llm_result=make_llm_result(),
            system_prompt_name="role",
        )
        assert record.system_prompt_name == "role"

    def test_system_prompt_name_rejects_non_string(self) -> None:
        with pytest.raises(TypeError, match="system_prompt_name"):
            LLMRecord(
                messages=({"role": "user", "content": "hi"},),
                llm_result=make_llm_result(),
                system_prompt_name=123,  # type: ignore[arg-type]
            )

    def test_to_dict_includes_system_prompt_name(self) -> None:
        record = LLMRecord(
            messages=({"role": "user", "content": "hi"},),
            llm_result=make_llm_result(),
            system_prompt_name="role",
        )
        assert record.to_dict()["system_prompt_name"] == "role"

    def test_rejects_string_as_messages(self) -> None:
        with pytest.raises(TypeError, match="messages"):
            LLMRecord(messages="not a list", llm_result=make_llm_result())  # type: ignore[arg-type]

    def test_rejects_empty_messages(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            LLMRecord(messages=[], llm_result=make_llm_result())

    def test_rejects_non_dict_element(self) -> None:
        with pytest.raises(TypeError, match=r"messages\[0\]"):
            LLMRecord(messages=["not a dict"], llm_result=make_llm_result())  # type: ignore[arg-type]

    def test_rejects_empty_dict_element(self) -> None:
        with pytest.raises(ValueError, match=r"messages\[0\]"):
            LLMRecord(messages=[{}], llm_result=make_llm_result())

    def test_rejects_non_string_key(self) -> None:
        with pytest.raises(TypeError, match="non-string key"):
            LLMRecord(messages=[{1: "val"}], llm_result=make_llm_result())  # type: ignore[arg-type]

    def test_rejects_non_string_value(self) -> None:
        with pytest.raises(TypeError, match="non-string value"):
            LLMRecord(messages=[{"role": 123}], llm_result=make_llm_result())  # type: ignore[arg-type]

    def test_rejects_non_llm_result(self) -> None:
        with pytest.raises(TypeError, match="llm_result"):
            LLMRecord(
                messages=[{"role": "user", "content": "hi"}],
                llm_result="not a result",  # type: ignore[arg-type]
            )

    def test_is_frozen(self) -> None:
        record = make_llm_record()
        with pytest.raises(FrozenInstanceError):
            record.messages = ({"role": "user", "content": "other"},)  # type: ignore[misc]


class TestAgentRecord:
    def test_valid_record_stores_prompt_and_response(self) -> None:
        record = AgentRecord(
            user_prompt="write a summary",
            generated_response="raw assistant text",
        )
        assert record.user_prompt == "write a summary"
        assert record.generated_response == "raw assistant text"
        assert record.final_result is None

    def test_final_result_accepts_agent_result(self) -> None:
        agent_result = make_agent_result()
        record = AgentRecord(
            user_prompt="run",
            generated_response="raw",
            final_result=agent_result,
        )
        assert record.final_result is agent_result

    def test_to_dict_with_none_final_result(self) -> None:
        record = AgentRecord(
            user_prompt="run",
            generated_response="raw",
        )
        assert record.to_dict() == {
            "user_prompt": "run",
            "inputs": {},
            "generated_response": "raw",
            "final_result": None,
            "llm_records": [],
            "prev_run_id": None,
            "child_ids": [],
        }

    def test_to_dict_with_agent_result(self) -> None:
        agent_result = make_agent_result(value="done")
        record = AgentRecord(
            user_prompt="run",
            generated_response="raw",
            final_result=agent_result,
        )
        d = record.to_dict()
        assert d["user_prompt"] == "run"
        assert d["generated_response"] == "raw"
        assert d["final_result"] == agent_result.to_dict()

    def test_to_dict_includes_inputs_field(self) -> None:
        record = AgentRecord(
            user_prompt="run",
            generated_response="raw",
            inputs={"lang": "English"},
        )
        assert record.to_dict()["inputs"] == {"lang": "English"}

    def test_inputs_shallow_copied_on_construction(self) -> None:
        ctx = {"key": "val"}
        record = AgentRecord(user_prompt="x", generated_response="y", inputs=ctx)
        ctx["key"] = "mutated"
        assert record.inputs["key"] == "val"

    def test_rejects_non_string_user_prompt(self) -> None:
        with pytest.raises(TypeError, match="user_prompt"):
            AgentRecord(user_prompt=123, generated_response="raw")  # type: ignore[arg-type]

    def test_accepts_string_user_prompt(self) -> None:
        record = AgentRecord(user_prompt="plain string", generated_response="raw")
        assert record.user_prompt == "plain string"

    def test_is_frozen(self) -> None:
        record = AgentRecord(user_prompt="run", generated_response="raw")
        with pytest.raises(FrozenInstanceError):
            record.generated_response = "other"  # type: ignore[misc]

    def test_llm_records_defaults_to_empty_tuple(self) -> None:
        record = AgentRecord(user_prompt="hi", generated_response="there")
        assert record.llm_records == ()

    def test_llm_records_populated_on_completed_record(self) -> None:
        llm_rec = make_llm_record()
        record = AgentRecord(
            user_prompt="hi",
            generated_response="there",
            llm_records=(llm_rec,),
        )
        assert record.llm_records == (llm_rec,)

    def test_llm_records_normalizes_list_to_tuple(self) -> None:
        llm_rec = make_llm_record()
        record = AgentRecord(
            user_prompt="hi",
            generated_response="there",
            llm_records=[llm_rec],
        )
        assert isinstance(record.llm_records, tuple)

    def test_rejects_non_llm_record_item_in_llm_records(self) -> None:
        with pytest.raises(TypeError, match="LLMRecord"):
            AgentRecord(
                user_prompt="hi",
                generated_response="there",
                llm_records=("not a record",),
            )

    def test_to_dict_includes_llm_records(self) -> None:
        llm_rec = make_llm_record()
        record = AgentRecord(
            user_prompt="hi",
            generated_response="there",
            llm_records=(llm_rec,),
        )
        d = record.to_dict()
        assert "llm_records" in d
        assert isinstance(d["llm_records"], list)
        assert d["llm_records"] == [llm_rec.to_dict()]

    def test_prev_defaults_to_none(self) -> None:
        record = AgentRecord(user_prompt="x", generated_response="y")
        assert record.prev is None

    def test_prev_accepts_agent_record_instance(self) -> None:
        completed = AgentRecord(
            user_prompt="first",
            generated_response="response",
            final_result=make_agent_result(),
        )
        record = AgentRecord(user_prompt="second", generated_response="r2", prev=completed)
        assert record.prev is completed

    def test_prev_rejects_non_agent_record(self) -> None:
        with pytest.raises(TypeError, match="prev"):
            AgentRecord(user_prompt="x", generated_response="y", prev="bad")  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad_prev", [0, "string", object(), []])
    def test_prev_rejects_non_agent_record_parametrize(self, bad_prev: Any) -> None:
        with pytest.raises(TypeError, match="prev"):
            AgentRecord(user_prompt="x", generated_response="y", prev=bad_prev)  # type: ignore[arg-type]

    def test_to_dict_prev_run_id_is_none_when_no_prev(self) -> None:
        record = AgentRecord(user_prompt="x", generated_response="y")
        assert record.to_dict()["prev_run_id"] is None

    def test_to_dict_prev_run_id_matches_prev_final_result_run_id(self) -> None:
        agent_result = make_agent_result()
        prev_record = AgentRecord(
            user_prompt="first",
            generated_response="r1",
            final_result=agent_result,
        )
        record = AgentRecord(user_prompt="second", generated_response="r2", prev=prev_record)
        assert record.to_dict()["prev_run_id"] == agent_result.run_id

    def test_children_defaults_to_empty_list(self) -> None:
        record = AgentRecord(user_prompt="x", generated_response="y")
        assert record.children == []

    def test_children_accepts_agent_record_instances(self) -> None:
        child = AgentRecord(
            user_prompt="child", generated_response="r", final_result=make_agent_result()
        )
        record = AgentRecord(user_prompt="x", generated_response="y", children=[child])
        assert record.children == [child]

    def test_children_rejects_non_list_tuple(self) -> None:
        with pytest.raises(TypeError, match="children"):
            AgentRecord(user_prompt="x", generated_response="y", children="not a list")  # type: ignore[arg-type]

    def test_children_rejects_non_agent_record_element(self) -> None:
        with pytest.raises(TypeError, match="children"):
            AgentRecord(user_prompt="x", generated_response="y", children=["not a record"])  # type: ignore[list-item]

    def test_children_not_normalized_to_tuple(self) -> None:
        # Deliberately mutable -- unlike llm_records, children is appended to
        # in place after construction (Agent._commit_emit's fork-vs-continue
        # bookkeeping), so it must stay a real list, not a frozen tuple.
        record = AgentRecord(user_prompt="x", generated_response="y")
        assert isinstance(record.children, list)
        record.children.append(
            AgentRecord(user_prompt="c", generated_response="r", final_result=make_agent_result())
        )
        assert len(record.children) == 1

    def test_to_dict_child_ids_reflects_current_children(self) -> None:
        child_result = make_agent_result()
        child = AgentRecord(user_prompt="child", generated_response="r", final_result=child_result)
        record = AgentRecord(user_prompt="x", generated_response="y", children=[child])
        assert record.to_dict()["child_ids"] == [child_result.run_id]


class TestToolAgentRecord:
    def test_is_an_agent_record(self) -> None:
        statements = (ToolStatement(identifier="x", tool="add"),)
        failed = (ToolStatement(identifier="y", tool="add", exception=ValueError("boom")),)
        record = ToolAgentRecord(
            user_prompt="run tools",
            generated_response=42,
            statements=statements,
            failed_statements=failed,
            regenerations_used=2,
        )
        assert isinstance(record, AgentRecord)
        assert record.statements == statements
        assert record.failed_statements == failed
        assert record.regenerations_used == 2

    def test_fields_default_to_empty(self) -> None:
        record = ToolAgentRecord(user_prompt="run tools", generated_response=42)

        assert record.statements == ()
        assert record.failed_statements == ()
        assert record.regenerations_used == 0

    def test_statements_normalized_to_tuple(self) -> None:
        statements = [ToolStatement(identifier="x", tool="add")]
        record = ToolAgentRecord(user_prompt="run", generated_response=1, statements=statements)

        assert isinstance(record.statements, tuple)

    def test_rejects_non_tool_statement_in_statements(self) -> None:
        with pytest.raises(TypeError, match="ToolStatement"):
            ToolAgentRecord(
                user_prompt="run", generated_response=1, statements=("not a statement",)  # type: ignore[arg-type]
            )

    def test_rejects_non_tool_statement_in_failed_statements(self) -> None:
        with pytest.raises(TypeError, match="ToolStatement"):
            ToolAgentRecord(
                user_prompt="run", generated_response=1,
                failed_statements=("not a statement",),  # type: ignore[arg-type]
            )

    def test_to_dict_includes_statements_and_regenerations_used(self) -> None:
        statements = (ToolStatement(identifier="x", tool="add", args=(_const(1), _const(2))),)
        record = ToolAgentRecord(
            user_prompt="run tools", generated_response=42, statements=statements, regenerations_used=1,
        )
        d = record.to_dict()
        assert d["statements"] == [s.to_dict() for s in statements]
        assert d["failed_statements"] == []
        assert d["regenerations_used"] == 1

    def test_inherits_agent_record_validation(self) -> None:
        with pytest.raises(TypeError, match="user_prompt"):
            ToolAgentRecord(user_prompt=123, generated_response="raw")  # type: ignore[arg-type]

    def test_is_frozen(self) -> None:
        record = ToolAgentRecord(user_prompt="run", generated_response="raw")
        with pytest.raises(FrozenInstanceError):
            record.regenerations_used = 1  # type: ignore[misc]

    def test_render_as_code_groups_by_batch(self) -> None:
        a = ToolStatement(identifier="a", tool="tool_a", batch_index=0)
        b = ToolStatement(identifier="b", tool="tool_b", batch_index=1)
        record = ToolAgentRecord(user_prompt="do it", generated_response=None, statements=(a, b))

        rendered = record.render_as_code()

        assert "# Batch 0:" in rendered
        assert "# Batch 1:" in rendered

    def test_usage_report_counts_dispatched_calls_by_tool_identity(self) -> None:
        statements = (
            ToolStatement(identifier="a", tool="add"),
            ToolStatement(identifier="b", tool="add"),
        )
        failed = (ToolStatement(identifier="c", tool="add", exception=ValueError("boom")),)
        record = ToolAgentRecord(
            user_prompt="do it", generated_response=None, statements=statements, failed_statements=failed,
        )

        report = record.usage_report()

        assert report.total_dispatched == 3
        assert report.total_failed == 1
        assert report.by_tool == (ToolUsageRecord(tool_name="add", call_count=3),)

    def test_usage_report_excludes_return_alias(self) -> None:
        statements = (
            ToolStatement(identifier="a", tool="add"),
            ToolStatement(identifier=None, tool=RETURN_ALIAS, kwargs={"val": _const(1)}),
        )
        record = ToolAgentRecord(user_prompt="do it", generated_response=1, statements=statements)

        report = record.usage_report()

        assert report.total_dispatched == 1
        assert report.by_tool == (ToolUsageRecord(tool_name="add", call_count=1),)


class TestThinkingAgentRecord:
    def test_is_an_agent_record(self) -> None:
        record = ThinkingAgentRecord(
            user_prompt="write a poem",
            generated_response="a poem",
            thoughts=("first thought", "second thought"),
        )
        assert isinstance(record, AgentRecord)
        assert record.thoughts == ("first thought", "second thought")

    def test_thoughts_defaults_to_empty_tuple(self) -> None:
        record = ThinkingAgentRecord(
            user_prompt="write a poem",
            generated_response="a poem",
        )
        assert record.thoughts == ()

    def test_thoughts_normalizes_list_to_tuple(self) -> None:
        record = ThinkingAgentRecord(
            user_prompt="write a poem",
            generated_response="a poem",
            thoughts=["a thought", {"k": "v"}, 3],
        )
        assert record.thoughts == ("a thought", {"k": "v"}, 3)
        assert isinstance(record.thoughts, tuple)

    def test_thoughts_rejects_non_sequence(self) -> None:
        with pytest.raises(TypeError, match="thoughts"):
            ThinkingAgentRecord(
                user_prompt="write a poem",
                generated_response="a poem",
                thoughts="not a sequence",  # type: ignore[arg-type]
            )

    def test_to_dict_includes_thoughts(self) -> None:
        record = ThinkingAgentRecord(
            user_prompt="write a poem",
            generated_response="a poem",
            thoughts=["a thought", {"k": "v"}, 3],
        )
        assert record.to_dict() == {
            "user_prompt": "write a poem",
            "inputs": {},
            "generated_response": "a poem",
            "final_result": None,
            "llm_records": [],
            "prev_run_id": None,
            "child_ids": [],
            "thoughts": ["a thought", {"k": "v"}, 3],
        }

    def test_inherits_agent_record_validation(self) -> None:
        with pytest.raises(TypeError, match="user_prompt"):
            ThinkingAgentRecord(user_prompt=123, generated_response="raw")  # type: ignore[arg-type]

    def test_is_frozen(self) -> None:
        record = ThinkingAgentRecord(user_prompt="run", generated_response="raw")
        with pytest.raises(FrozenInstanceError):
            record.thoughts = ("x",)  # type: ignore[misc]


def _name(identifier: str) -> ast.Name:
    return ast.Name(id=identifier, ctx=ast.Load())


def _const(value: object) -> ast.Constant:
    return ast.Constant(value=value)


class TestToolStatement:
    """
    Covers ToolStatement's own construction/validation/serialization --
    the unified replacement for the pre-toolstatement-unification
    BlackboardSlot (no separate status string, no step_dependencies stored
    field, no copy()/from_dict() -- dependencies are derived on demand via
    extract_identifiers, and a ToolStatement is never reconstructed from a
    plain dict the way a BlackboardSlot once was).
    """

    def test_defaults(self) -> None:
        stmt = ToolStatement(identifier=None, tool="add")

        assert stmt.identifier is None
        assert stmt.tool == "add"
        assert stmt.args == ()
        assert stmt.kwargs == {}
        assert stmt.batch_index is None
        assert stmt.result is None
        assert stmt.exception is None

    def test_args_normalized_to_tuple(self) -> None:
        stmt = ToolStatement(identifier=None, tool="add", args=[_const(1), _const(2)])

        assert isinstance(stmt.args, tuple)

    def test_rejects_invalid_identifier(self) -> None:
        with pytest.raises(ValueError, match="identifier"):
            ToolStatement(identifier="not an identifier", tool="add")

    def test_rejects_empty_tool(self) -> None:
        with pytest.raises(ValueError, match="tool"):
            ToolStatement(identifier=None, tool="")

    def test_rejects_non_sequence_args(self) -> None:
        with pytest.raises(TypeError, match="args"):
            ToolStatement(identifier=None, tool="add", args="not a sequence")  # type: ignore[arg-type]

    def test_rejects_non_dict_kwargs(self) -> None:
        with pytest.raises(TypeError, match="kwargs"):
            ToolStatement(identifier=None, tool="add", kwargs=["not a dict"])  # type: ignore[arg-type]

    def test_to_dict_renders_ast_expr_args_as_source_text(self) -> None:
        stmt = ToolStatement(identifier="y", tool="add", args=(_name("x"), _const(2)))

        d = stmt.to_dict()

        assert d["args"] == ["x", "2"]
        assert d["identifier"] == "y"
        assert d["tool"] == "add"
        assert d["result"] is None
        assert d["exception"] is None

    def test_to_dict_renders_exception_as_repr(self) -> None:
        stmt = ToolStatement(identifier="f", tool="add", exception=ValueError("boom"))

        assert stmt.to_dict()["exception"] == repr(ValueError("boom"))

    def test_to_code_renders_a_plain_call(self) -> None:
        stmt = ToolStatement(identifier="y", tool="add", args=(_name("x"), _const(2)))

        assert stmt.to_code() == "y = add(x, 2)"

    def test_to_code_renders_return_from_kwargs(self) -> None:
        stmt = ToolStatement(identifier=None, tool=RETURN_ALIAS, kwargs={"val": _name("w")})

        assert stmt.to_code() == "return w"

    def test_to_code_renders_return_from_positional_args(self) -> None:
        # ReActAgent's real, model-authored return call may supply its one
        # argument positionally instead of by keyword.
        stmt = ToolStatement(identifier=None, tool=RETURN_ALIAS, args=(_name("w"),))

        assert stmt.to_code() == "return w"

    def test_to_code_renders_rhs_assign(self) -> None:
        stmt = ToolStatement(identifier="x", tool=RHS_ASSIGN_ALIAS, kwargs={"val": _const(1)})

        assert stmt.to_code() == "x = 1"


class TestScriptActAgentRecordSerialization:
    def test_to_dict_handles_every_slot_shape_without_crashing(self) -> None:
        # Regression guard for the documented slots=True-under-inheritance
        # super() gotcha (bare super() raises for a slotted dataclass in an
        # inheritance chain) -- exercises ToolStatement.to_dict() across
        # every slot shape, plus ScriptActAgentRecord.to_dict()'s own super()
        # call.
        statements = (
            ToolStatement(identifier="x", tool=RHS_ASSIGN_ALIAS, kwargs={"val": _const(1)}),
            ToolStatement(identifier="y", tool="add", args=(_name("x"), _const(2))),
            ToolStatement(identifier="z", tool=PY_BUILTIN_ALIAS, args=("len", _name("y"))),
            ToolStatement(identifier="w", tool=ATTR_CALL_ALIAS, args=(_name("y"), "method")),
            ToolStatement(identifier=None, tool=RETURN_ALIAS, kwargs={"val": _name("w")}),
        )
        failed = (ToolStatement(identifier="f", tool="add", exception=ValueError("boom")),)
        record = ScriptActAgentRecord(
            user_prompt="do it", generated_response=5, statements=statements, failed_statements=failed,
        )

        d = record.to_dict()

        assert len(d["statements"]) == 5
        assert len(d["failed_statements"]) == 1
        assert d["failed_statements"][0]["exception"] == repr(ValueError("boom"))

    def test_to_dict_renders_py_builtin_args_as_json_serializable(self) -> None:
        slot = ToolStatement(identifier="z", tool=PY_BUILTIN_ALIAS, args=("len", _name("y")))

        d = slot.to_dict()

        assert d["args"] == ["len", "y"]

    def test_to_dict_renders_attr_call_args_with_unparsed_object_source(self) -> None:
        slot = ToolStatement(identifier="w", tool=ATTR_CALL_ALIAS, args=(_name("y"), "method", _const(1)))

        d = slot.to_dict()

        # Every entry still an ast.expr node -- including a plain
        # ast.Constant -- is rendered via ast.unparse (source text, not the
        # raw value); only method_name (already a bare str, args[1]) passes
        # through unchanged.
        assert d["args"] == ["y", "method", "1"]

    def test_render_as_code_groups_by_batch(self) -> None:
        a = ToolStatement(identifier="a", tool="tool_a", batch_index=0)
        b = ToolStatement(identifier="b", tool="tool_b", batch_index=1)
        record = ScriptActAgentRecord(user_prompt="do it", generated_response=None, statements=(a, b))

        rendered = record.render_as_code()

        assert "# Batch 0:" in rendered
        assert "# Batch 1:" in rendered


class TestScriptActAgentToolUsage:
    def test_computes_all_five_metrics_from_statements(self) -> None:
        binop_val = ast.parse("1 + 2", mode="eval").body
        statements = (
            ToolStatement(identifier="a", tool="add"),
            ToolStatement(identifier="b", tool="add"),
            ToolStatement(identifier="c", tool=PY_BUILTIN_ALIAS, args=("len", _name("a"))),
            ToolStatement(identifier="d", tool=ATTR_CALL_ALIAS, args=(_name("a"), "method")),
            ToolStatement(identifier="e", tool=RHS_ASSIGN_ALIAS, kwargs={"val": binop_val}),
            ToolStatement(identifier="f", tool=RHS_ASSIGN_ALIAS, kwargs={"val": _const(1)}),
        )
        record = ScriptActAgentRecord(user_prompt="do it", generated_response=None, statements=statements)

        usage = record.dispatch_breakdown()

        assert usage == ScriptActAgentToolUsage(
            registered_tool_calls=2, builtin_calls=1, attribute_calls=1,
            binop_count=1, rhs_assignment_count=2,
        )

    def test_failed_registered_tool_call_is_counted(self) -> None:
        # tool_usage() must count a failed dispatched call (from
        # failed_statements) the same as a succeeded one, since
        # tool_calls_used counted it unconditionally when it ran.
        statements = (ToolStatement(identifier="a", tool="add"),)
        failed = (ToolStatement(identifier="b", tool="add", exception=RuntimeError("boom")),)
        record = ScriptActAgentRecord(
            user_prompt="do it", generated_response=None, statements=statements, failed_statements=failed,
        )

        usage = record.dispatch_breakdown()

        assert usage.registered_tool_calls == 2

    def test_failed_builtin_and_attr_calls_are_counted(self) -> None:
        failed = (
            ToolStatement(identifier="a", tool=PY_BUILTIN_ALIAS, args=("len", _name("x")), exception=ValueError()),
            ToolStatement(identifier="b", tool=ATTR_CALL_ALIAS, args=(_name("x"), "m"), exception=ValueError()),
        )
        record = ScriptActAgentRecord(user_prompt="do it", generated_response=None, failed_statements=failed)

        usage = record.dispatch_breakdown()

        assert usage.builtin_calls == 1
        assert usage.attribute_calls == 1

    def test_failed_statements_never_affect_binop_or_rhs_assignment_counts(self) -> None:
        statements = (
            ToolStatement(identifier="a", tool=RHS_ASSIGN_ALIAS, kwargs={"val": _const(1)}),
        )
        without_failures = ScriptActAgentRecord(
            user_prompt="do it", generated_response=None, statements=statements,
        )
        with_failures = ScriptActAgentRecord(
            user_prompt="do it", generated_response=None, statements=statements,
            failed_statements=(ToolStatement(identifier="b", tool="add", exception=ValueError()),),
        )

        assert without_failures.dispatch_breakdown().binop_count == with_failures.dispatch_breakdown().binop_count
        assert (
            without_failures.dispatch_breakdown().rhs_assignment_count
            == with_failures.dispatch_breakdown().rhs_assignment_count
        )

    def test_empty_record_has_all_zero_metrics(self) -> None:
        record = ScriptActAgentRecord(user_prompt="do it", generated_response=None)

        assert record.dispatch_breakdown() == ScriptActAgentToolUsage(
            registered_tool_calls=0, builtin_calls=0, attribute_calls=0,
            binop_count=0, rhs_assignment_count=0,
        )
