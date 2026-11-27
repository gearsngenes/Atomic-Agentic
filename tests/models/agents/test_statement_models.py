"""Step and Statement: frozen shape and debugging snapshots."""

from __future__ import annotations

import ast
from types import SimpleNamespace
from dataclasses import FrozenInstanceError, replace

import pytest

from atomic_agentic.constants.core import NO_VAL
from atomic_agentic.models.agents import Statement, Step


def _step(**kw) -> Step:
    base = dict(
        kind="tool", tool="add",
        args=(ast.Constant(1), ast.Name("x", ast.Load())),
    )
    base.update(kw)
    return Step(**base)


def test_step_defaults_are_unrun():
    s = _step()
    assert s.status == "not_run"
    assert s.output is NO_VAL and s.value is NO_VAL
    assert s.result is None and s.error is None
    assert s.name is None


def _src(text: str) -> ast.expr:
    return ast.parse(text, mode="eval").body


def test_step_text_renders_each_step_shape():
    assert _step().text == "add(1, x)"
    assert _step(kwargs={"b": _src("y + 1")}).text == "add(1, x, b=y + 1)"
    star = ast.Starred(value=ast.Name("xs", ast.Load()), ctx=ast.Load())
    assert _step(args=(star,), kwargs={"**": _src("kw")}).text == "add(*xs, **kw)"
    assert _step(tool="py_builtin", args=("len", _src("s"))).text == "len(s)"
    assert _step(tool="attr_call", args=(_src("s"), "split", _src("','"))).text == "s.split(',')"
    assert Step(kind="eval", args=(_src("a + 1"),)).text == "a + 1"


def test_step_and_statement_are_frozen():
    s = _step()
    with pytest.raises(FrozenInstanceError):
        s.status = "ran"
    st = Statement(ordinal=0, source="x = add(1, x)", kind="assign", target="x", steps=(s,))
    with pytest.raises(FrozenInstanceError):
        st.ordinal = 1


def test_replace_returns_a_new_object_and_leaves_the_original():
    s = _step()
    ran = replace(s, kind="eval", status="ran", output=3)
    assert ran is not s
    assert s.status == "not_run" and s.value is NO_VAL
    assert ran.status == "ran" and ran.value == 3 and ran.result is None


def test_tool_step_value_and_result_derive_from_output():
    envelope = SimpleNamespace(result=[1], to_dict=lambda: {"result": [1]})
    s = replace(_step(), output=envelope, status="ran")
    assert s.result is envelope and s.value is envelope.result
    assert s.to_dict()["result"] == {"result": [1]} and s.to_dict()["value"] == [1]


def test_step_to_dict_renders_ast_values_error_and_no_val():
    s = _step(kwargs={"b": ast.Name("y", ast.Load())}, error=ValueError("bad"), status="failed")
    d = s.to_dict()
    assert d["args"] == ["1", "x"]
    assert d["kwargs"] == {"b": "y"}
    assert d["value"] is None
    assert d["error"] == "ValueError('bad')"
    assert d["status"] == "failed"


def test_statement_to_dict_nests_steps():
    s = _step(kind="eval", output=7, status="ran")
    st = Statement(ordinal=2, source="x = add(1, x)", kind="assign", target="x", steps=(s,))
    d = st.to_dict()
    assert d["ordinal"] == 2 and d["target"] == "x" and d["value"] == 7
    assert d["steps"][0]["tool"] == "add"


def test_statement_status_and_value_are_derived_from_steps():
    hoisted = _step(name="_SUB_0", kind="eval")
    own = _step(kind="eval")
    st = Statement(ordinal=0, source="x = add(add(1, x), 1)", kind="assign", target="x", steps=(hoisted, own))
    assert st.status == "not_run" and st.value is NO_VAL

    done = replace(st, steps=(replace(hoisted, status="ran", output=2), replace(own, status="ran", output=3)))
    assert done.status == "done" and done.value == 3

    failed = replace(st, steps=(replace(hoisted, status="ran", output=2), replace(own, status="failed")))
    assert failed.status == "failed" and failed.value is NO_VAL
