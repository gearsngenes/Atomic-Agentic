from __future__ import annotations

from types import SimpleNamespace

import ast
from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from atomic_agentic.constants.agents import ATTR_CALL_ALIAS, PY_BUILTIN_ALIAS
from atomic_agentic.models.agents.records import (
    AgentRecord,
    LLMRecord,
    PlanActAgentRecord,
    ThinkingAgentRecord,
    ToolAgentRecord,
)
from atomic_agentic.models.agents.blackboard_models import Statement, Step
from atomic_agentic.models.agents.context import ContextVariable
from atomic_agentic.constants.core import NO_VAL
from atomic_agentic.models.results import LLMModelData, LLMResult, TokenUsage
from atomic_agentic.models.results.agents import ToolUsageRecord


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


class TestContextVariable:
    def test_valid_spec_normalizes_name_description_and_derives_type(self) -> None:
        value = {"items": [1, 2]}

        spec = ContextVariable(
            name=" PAYLOAD ",
            value=value,
            description=" Runtime payload. ",
        )

        assert spec.name == "PAYLOAD"
        assert spec.value is value
        assert spec.description == "Runtime payload."
        assert spec.type == "dict"

    def test_blank_description_normalizes_to_none(self) -> None:
        spec = ContextVariable(name="VALUE", value=1, description="   ")

        assert spec.description is None

    def test_to_dict_includes_value_metadata_and_derived_type(self) -> None:
        spec = ContextVariable(
            name="THRESHOLD",
            value=0.75,
            description="Decision threshold.",
        )

        assert spec.to_dict() == {
            "name": "THRESHOLD",
            "value": 0.75,
            "description": "Decision threshold.",
            "readonly": False,
            "render_mode": "s",
            "preview_limit": None,
            "type": "float",
        }

    def test_render_mode_defaults_to_str(self) -> None:
        assert ContextVariable(name="V", value=1).render_mode == "s"

    @pytest.mark.parametrize("mode", ["s", "r", "a", "j", "p"])
    def test_render_mode_accepts_valid_values_and_survives_replace(
        self, mode: str
    ) -> None:
        spec = ContextVariable(name="V", value=1, render_mode=mode)

        assert spec.render_mode == mode
        assert replace(spec, value=2).render_mode == mode
        assert spec.to_dict()["render_mode"] == mode

    @pytest.mark.parametrize("mode", ["x", "S", "", 1, ["s"], None])
    def test_render_mode_rejects_invalid_values(self, mode: Any) -> None:
        with pytest.raises(ValueError, match="render_mode"):
            ContextVariable(name="V", value=1, render_mode=mode)

    def test_preview_limit_defaults_to_none(self) -> None:
        assert ContextVariable(name="V", value=1).preview_limit is None

    @pytest.mark.parametrize("limit", [None, 0, 5])
    def test_preview_limit_accepts_valid_values_and_survives_replace(
        self, limit: int | None
    ) -> None:
        spec = ContextVariable(name="V", value=1, preview_limit=limit)

        assert spec.preview_limit == limit
        assert replace(spec, value=2).preview_limit == limit
        assert spec.to_dict()["preview_limit"] == limit

    @pytest.mark.parametrize("limit", [-1, True, False, 1.5, "5"])
    def test_preview_limit_rejects_invalid_values(self, limit: Any) -> None:
        with pytest.raises(ValueError, match="preview_limit"):
            ContextVariable(name="V", value=1, preview_limit=limit)

    @pytest.mark.parametrize(
        "name",
        ["", " ", "1VALUE", "bad-name", "bad name", None],
    )
    def test_rejects_invalid_name(self, name: Any) -> None:
        with pytest.raises(ValueError, match="ContextVariable.name"):
            ContextVariable(name=name, value=1)  # type: ignore[arg-type]

    @pytest.mark.parametrize("description", [1, True, object()])
    def test_rejects_non_string_description(self, description: Any) -> None:
        with pytest.raises(TypeError, match="description"):
            ContextVariable(name="VALUE", value=1, description=description)  # type: ignore[arg-type]

    def test_is_frozen(self) -> None:
        spec = ContextVariable(name="VALUE", value=1)

        with pytest.raises(FrozenInstanceError):
            spec.name = "OTHER"  # type: ignore[misc]

    def test_type_tracks_value_through_replace(self) -> None:
        spec = ContextVariable(name="X", value=1)
        assert spec.type == "int"

        updated = replace(spec, value="now a string")

        assert updated.type == "str"
        assert spec.type == "int"

    def test_readonly_defaults_false_and_rejects_non_bool(self) -> None:
        assert ContextVariable(name="X", value=1).readonly is False

        for bad in (0, 1, "yes", None):
            with pytest.raises(TypeError, match="readonly"):
                ContextVariable(name="X", value=1, readonly=bad)  # type: ignore[arg-type]

    def test_readonly_survives_replace_and_appears_in_to_dict_and_equality(self) -> None:
        spec = ContextVariable(name="X", value=1, readonly=True)

        assert replace(spec, value=2).readonly is True
        assert spec.to_dict()["readonly"] is True
        assert spec != ContextVariable(name="X", value=1)


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
        }

    def test_has_no_system_prompt_name(self) -> None:
        record = make_llm_record()
        assert not hasattr(record, "system_prompt_name")
        assert "system_prompt_name" not in record.to_dict()

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
            run_id="run-1",
            user_prompt="write a summary",
            generated_response="raw assistant text",
        )
        assert record.user_prompt == "write a summary"
        assert record.generated_response == "raw assistant text"
        assert isinstance(record.run_id, str) and record.run_id

    def test_run_id_is_required(self) -> None:
        with pytest.raises(TypeError, match="run_id"):
            AgentRecord(user_prompt="run", generated_response="raw")  # type: ignore[call-arg]

    def test_run_id_accepts_explicit_value(self) -> None:
        record = AgentRecord(user_prompt="run", generated_response="raw", run_id="abc123")
        assert record.run_id == "abc123"

    def test_agent_name_defaults_to_none(self) -> None:
        record = AgentRecord(run_id="run-3", user_prompt="x", generated_response="y")
        assert record.agent_name is None

    def test_agent_name_is_keyword_only_and_serialized(self) -> None:
        record = AgentRecord(run_id="run-4", user_prompt="x", generated_response="y", agent_name="ns.a;123")
        assert record.agent_name == "ns.a;123"
        assert record.to_dict()["agent_name"] == "ns.a;123"

    def test_llm_records_rejects_str(self) -> None:
        with pytest.raises(TypeError, match="llm_records"):
            AgentRecord(run_id="run-5", user_prompt="x", generated_response="y", llm_records="abc")  # type: ignore[arg-type]

    def test_to_dict_has_no_final_result_key(self) -> None:
        record = AgentRecord(
            run_id="run-6",
            user_prompt="run",
            generated_response="raw",
        )
        d = record.to_dict()
        assert "final_result" not in d
        assert d == {
            "user_prompt": "run",
            "inputs": {},
            "generated_response": "raw",
            "run_id": record.run_id,
            "agent_name": None,
            "llm_records": [],
            "prev_run_id": None,
            "child_ids": [],
        }

    def test_to_dict_includes_inputs_field(self) -> None:
        record = AgentRecord(
            run_id="run-7",
            user_prompt="run",
            generated_response="raw",
            inputs={"lang": "English"},
        )
        assert record.to_dict()["inputs"] == {"lang": "English"}

    def test_inputs_shallow_copied_on_construction(self) -> None:
        ctx = {"key": "val"}
        record = AgentRecord(run_id="run-8", user_prompt="x", generated_response="y", inputs=ctx)
        ctx["key"] = "mutated"
        assert record.inputs["key"] == "val"

    def test_rejects_non_string_user_prompt(self) -> None:
        with pytest.raises(TypeError, match="user_prompt"):
            AgentRecord(run_id="run-9", user_prompt=123, generated_response="raw")  # type: ignore[arg-type]

    def test_accepts_string_user_prompt(self) -> None:
        record = AgentRecord(run_id="run-10", user_prompt="plain string", generated_response="raw")
        assert record.user_prompt == "plain string"

    def test_is_frozen(self) -> None:
        record = AgentRecord(run_id="run-11", user_prompt="run", generated_response="raw")
        with pytest.raises(FrozenInstanceError):
            record.generated_response = "other"  # type: ignore[misc]

    def test_llm_records_defaults_to_empty_tuple(self) -> None:
        record = AgentRecord(run_id="run-12", user_prompt="hi", generated_response="there")
        assert record.llm_records == ()

    def test_llm_records_populated_on_completed_record(self) -> None:
        llm_rec = make_llm_record()
        record = AgentRecord(
            run_id="run-13",
            user_prompt="hi",
            generated_response="there",
            llm_records=(llm_rec,),
        )
        assert record.llm_records == (llm_rec,)

    def test_llm_records_normalizes_list_to_tuple(self) -> None:
        llm_rec = make_llm_record()
        record = AgentRecord(
            run_id="run-14",
            user_prompt="hi",
            generated_response="there",
            llm_records=[llm_rec],
        )
        assert isinstance(record.llm_records, tuple)

    def test_to_dict_includes_llm_records(self) -> None:
        llm_rec = make_llm_record()
        record = AgentRecord(
            run_id="run-15",
            user_prompt="hi",
            generated_response="there",
            llm_records=(llm_rec,),
        )
        d = record.to_dict()
        assert "llm_records" in d
        assert isinstance(d["llm_records"], list)
        assert d["llm_records"] == [llm_rec.to_dict()]

    def test_prev_defaults_to_none(self) -> None:
        record = AgentRecord(run_id="run-16", user_prompt="x", generated_response="y")
        assert record.prev is None

    def test_prev_accepts_agent_record_instance(self) -> None:
        completed = AgentRecord(
            run_id="run-17",
            user_prompt="first",
            generated_response="response",
        )
        record = AgentRecord(run_id="run-18", user_prompt="second", generated_response="r2", prev=completed)
        assert record.prev is completed

    def test_to_dict_prev_run_id_is_none_when_no_prev(self) -> None:
        record = AgentRecord(run_id="run-19", user_prompt="x", generated_response="y")
        assert record.to_dict()["prev_run_id"] is None

    def test_to_dict_prev_run_id_matches_prev_run_id(self) -> None:
        prev_record = AgentRecord(
            run_id="run-20",
            user_prompt="first",
            generated_response="r1",
        )
        record = AgentRecord(run_id="run-21", user_prompt="second", generated_response="r2", prev=prev_record)
        assert record.to_dict()["prev_run_id"] == prev_record.run_id

    def test_children_defaults_to_empty_list(self) -> None:
        record = AgentRecord(run_id="run-22", user_prompt="x", generated_response="y")
        assert record.children == []

    def test_children_accepts_agent_record_instances(self) -> None:
        child = AgentRecord(run_id="run-23", user_prompt="child", generated_response="r")
        record = AgentRecord(run_id="run-24", user_prompt="x", generated_response="y", children=[child])
        assert record.children == [child]

    def test_children_not_normalized_to_tuple(self) -> None:
        # Deliberately mutable -- unlike llm_records, children is appended to
        # in place after construction (Agent.record's children
        # bookkeeping), so it must stay a real list, not a frozen tuple.
        record = AgentRecord(run_id="run-25", user_prompt="x", generated_response="y")
        assert isinstance(record.children, list)
        record.children.append(AgentRecord(run_id="run-26", user_prompt="c", generated_response="r"))
        assert len(record.children) == 1

    def test_to_dict_child_ids_reflects_current_children(self) -> None:
        child = AgentRecord(run_id="run-27", user_prompt="child", generated_response="r")
        record = AgentRecord(run_id="run-28", user_prompt="x", generated_response="y", children=[child])
        assert record.to_dict()["child_ids"] == [child.run_id]


def _name(identifier: str) -> ast.Name:
    return ast.Name(id=identifier, ctx=ast.Load())


def _const(value: object) -> ast.Constant:
    return ast.Constant(value=value)


def _statement(
    tool: str = "add", target: str | None = "x", *, ordinal: int = 0, status: str = "ran",
    args: tuple = (), error: Exception | None = None,
) -> Statement:
    """A one-step tool statement in the given status (a failed one carries ``error``)."""
    output = SimpleNamespace(result=1, to_dict=lambda: {"result": 1}) if status == "ran" else NO_VAL
    step = Step(kind="tool", tool=tool, args=args, output=output, status=status, error=error)
    kind = "assign" if target is not None else "call"
    return Statement(ordinal=ordinal, source="", kind=kind, target=target, steps=(step,))


def _failed(tool: str = "add", target: str | None = "y", *, ordinal: int = 1) -> Statement:
    return _statement(tool, target, ordinal=ordinal, status="failed", error=ValueError("boom"))


class TestToolAgentRecord:
    def test_is_an_agent_record(self) -> None:
        statements = (_statement(),)
        failed = (_failed(),)
        record = ToolAgentRecord(
            run_id="run-29",
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
        record = ToolAgentRecord(run_id="run-30", user_prompt="run tools", generated_response=42)

        assert record.statements == ()
        assert record.failed_statements == ()
        assert record.regenerations_used == 0

    def test_statements_normalized_to_tuple(self) -> None:
        statements = [_statement()]
        record = ToolAgentRecord(run_id="run-31", user_prompt="run", generated_response=1, statements=statements)

        assert isinstance(record.statements, tuple)

    def test_rejects_non_statement_in_statements(self) -> None:
        with pytest.raises(TypeError, match="Statement"):
            ToolAgentRecord(
                run_id="run-32",
                user_prompt="run", generated_response=1, statements=("not a statement",)  # type: ignore[arg-type]
            )

    def test_rejects_non_statement_in_failed_statements(self) -> None:
        with pytest.raises(TypeError, match="Statement"):
            ToolAgentRecord(
                run_id="run-33",
                user_prompt="run", generated_response=1,
                failed_statements=("not a statement",),  # type: ignore[arg-type]
            )

    def test_to_dict_includes_statements_and_regenerations_used(self) -> None:
        statements = (_statement(args=(_const(1), _const(2))),)
        record = ToolAgentRecord(
            run_id="run-34",
            user_prompt="run tools", generated_response=42, statements=statements, regenerations_used=1,
        )
        d = record.to_dict()
        assert d["statements"] == [s.to_dict() for s in statements]
        assert d["failed_statements"] == []
        assert d["regenerations_used"] == 1

    def test_inherits_agent_record_validation(self) -> None:
        with pytest.raises(TypeError, match="user_prompt"):
            ToolAgentRecord(run_id="run-35", user_prompt=123, generated_response="raw")  # type: ignore[arg-type]

    def test_is_frozen(self) -> None:
        record = ToolAgentRecord(run_id="run-36", user_prompt="run", generated_response="raw")
        with pytest.raises(FrozenInstanceError):
            record.regenerations_used = 1  # type: ignore[misc]

    def test_render_as_code_merges_done_and_failed_in_source_order(self) -> None:
        done = Statement(
            ordinal=0, source="a = add(1, 2)", kind="assign", target="a",
            steps=(Step(kind="tool", tool="add", output=SimpleNamespace(result=3, to_dict=lambda: {"result": 3}), status="ran"),),
        )
        failed = Statement(
            ordinal=1, source="b = boom(1)", kind="assign", target="b",
            steps=(Step(kind="tool", tool="boom", status="failed", error=ValueError("bad")),),
        )
        later = Statement(
            ordinal=2, source="c = add(a, 1)", kind="assign", target="c",
            steps=(Step(kind="tool", tool="add", output=SimpleNamespace(result=3, to_dict=lambda: {"result": 3}), status="ran"),),
        )
        record = ToolAgentRecord(
            run_id="run-37", user_prompt="do it", generated_response=None,
            statements=(done, later), failed_statements=(failed,),
        )

        lines = record.render_as_code().split("\n")

        assert lines[0] == "a = add(1, 2)"
        assert lines[1].startswith("b = boom(1)  # FAILED")
        assert lines[2] == "c = add(a, 1)"

    def test_usage_report_counts_ran_and_failed_tool_steps_by_identity(self) -> None:
        statements = (_statement(target="a"), _statement(target="b", ordinal=1))
        failed = (_failed(target="c", ordinal=2),)
        record = ToolAgentRecord(
            run_id="run-38",
            user_prompt="do it", generated_response=None, statements=statements, failed_statements=failed,
        )

        report = record.usage_report()

        assert report.total_dispatched == 3
        assert report.total_failed == 1
        assert report.by_tool == (ToolUsageRecord(tool_name="add", call_count=3),)

    def test_usage_report_excludes_eval_steps(self) -> None:
        ret = Statement(
            ordinal=1, source="return 1", kind="return", target=None,
            steps=(Step(kind="eval", args=(_const(1),), status="ran"),),
        )
        record = ToolAgentRecord(
            run_id="run-39", user_prompt="do it", generated_response=1,
            statements=(_statement(target="a"), ret),
        )

        report = record.usage_report()

        assert report.total_dispatched == 1
        assert report.by_tool == (ToolUsageRecord(tool_name="add", call_count=1),)

    def test_usage_report_is_build_usage_report_over_the_record_statements(self) -> None:
        from atomic_agentic.utils.agents import build_usage_report

        statements = (_statement(target="a"),)
        failed = (_failed("sub", "b"),)
        record = ToolAgentRecord(
            run_id="run-40",
            user_prompt="do it", generated_response=None, statements=statements, failed_statements=failed,
        )

        assert record.usage_report() == build_usage_report((*statements, *failed))
        assert build_usage_report([]).by_tool == ()


class TestThinkingAgentRecord:
    def test_is_an_agent_record(self) -> None:
        record = ThinkingAgentRecord(
            run_id="run-41",
            user_prompt="write a poem",
            generated_response="a poem",
            thoughts=("first thought", "second thought"),
        )
        assert isinstance(record, AgentRecord)
        assert record.thoughts == ("first thought", "second thought")

    def test_thoughts_defaults_to_empty_tuple(self) -> None:
        record = ThinkingAgentRecord(
            run_id="run-42",
            user_prompt="write a poem",
            generated_response="a poem",
        )
        assert record.thoughts == ()

    def test_thoughts_normalizes_list_to_tuple(self) -> None:
        record = ThinkingAgentRecord(
            run_id="run-43",
            user_prompt="write a poem",
            generated_response="a poem",
            thoughts=["a thought", {"k": "v"}, 3],
        )
        assert record.thoughts == ("a thought", {"k": "v"}, 3)
        assert isinstance(record.thoughts, tuple)

    def test_thoughts_rejects_non_sequence(self) -> None:
        with pytest.raises(TypeError, match="thoughts"):
            ThinkingAgentRecord(
                run_id="run-44",
                user_prompt="write a poem",
                generated_response="a poem",
                thoughts="not a sequence",  # type: ignore[arg-type]
            )

    def test_to_dict_includes_thoughts(self) -> None:
        record = ThinkingAgentRecord(
            run_id="run-45",
            user_prompt="write a poem",
            generated_response="a poem",
            thoughts=["a thought", {"k": "v"}, 3],
        )
        d = record.to_dict()
        assert "final_result" not in d
        assert d == {
            "user_prompt": "write a poem",
            "inputs": {},
            "generated_response": "a poem",
            "run_id": record.run_id,
            "agent_name": None,
            "llm_records": [],
            "prev_run_id": None,
            "child_ids": [],
            "thoughts": ["a thought", {"k": "v"}, 3],
        }

    def test_inherits_agent_record_validation(self) -> None:
        with pytest.raises(TypeError, match="user_prompt"):
            ThinkingAgentRecord(run_id="run-46", user_prompt=123, generated_response="raw")  # type: ignore[arg-type]

    def test_is_frozen(self) -> None:
        record = ThinkingAgentRecord(run_id="run-47", user_prompt="run", generated_response="raw")
        with pytest.raises(FrozenInstanceError):
            record.thoughts = ("x",)  # type: ignore[misc]


class TestPlanActAgentRecord:
    def test_is_a_tool_agent_record_with_repair_rounds(self) -> None:
        record = PlanActAgentRecord(
            run_id="run-48", user_prompt="do it", generated_response=5,
            statements=(_statement(),), repair_rounds_used=2,
        )

        assert isinstance(record, ToolAgentRecord)
        assert record.repair_rounds_used == 2
        assert PlanActAgentRecord(run_id="run-48b", user_prompt="x", generated_response=1).repair_rounds_used == 0

    def test_to_dict_handles_every_step_shape_without_crashing(self) -> None:
        # Regression guard for the documented slots=True-under-inheritance
        # super() gotcha -- exercises Statement.to_dict() across every step
        # shape, plus PlanActAgentRecord.to_dict()'s own super() call.
        statements = (
            Statement(
                ordinal=0, source="x = 1", kind="assign", target="x",
                steps=(Step(kind="eval", args=(_const(1),), output=1, status="ran"),),
            ),
            _statement("add", "y", ordinal=1, args=(_name("x"), _const(2))),
            _statement(PY_BUILTIN_ALIAS, "z", ordinal=2, args=("len", _name("y"))),
            _statement(ATTR_CALL_ALIAS, "w", ordinal=3, args=(_name("y"), "method")),
            Statement(
                ordinal=4, source="return w", kind="return", target=None,
                steps=(Step(kind="eval", args=(_name("w"),), output=5, status="ran"),),
            ),
        )
        failed = (_failed("add", "f", ordinal=5),)
        record = PlanActAgentRecord(
            run_id="run-49",
            user_prompt="do it", generated_response=5, statements=statements, failed_statements=failed,
            repair_rounds_used=1,
        )

        d = record.to_dict()

        assert len(d["statements"]) == 5
        assert len(d["failed_statements"]) == 1
        assert d["failed_statements"][0]["steps"][0]["error"] == repr(ValueError("boom"))
        assert d["repair_rounds_used"] == 1
        assert "dispatch_breakdown" not in dir(record)
