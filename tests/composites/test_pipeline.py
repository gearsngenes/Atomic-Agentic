from __future__ import annotations

import asyncio
import logging
from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from atomic_agentic.composites import Pipeline
from atomic_agentic.invokables.prompt_template import PromptTemplate
from atomic_agentic.models.results import AtomicResult
from atomic_agentic.tools.base import Tool

PIPELINE_LOGGER = "atomic_agentic.composites.pipeline"


def add_one(value: int) -> dict[str, int]:
    """Add one and report it as first."""
    return {"first": value + 1}


def double(first: int) -> dict[str, int]:
    """Double first and report it as second."""
    return {"second": first * 2}


def label(second: int) -> str:
    """Render second as text."""
    return f"value={second}"


def triple(value: int) -> int:
    """Return value times three (a non-mapping)."""
    return value * 3


def pair(value: int) -> tuple[int, int]:
    """Return value and value plus one."""
    return (value, value + 1)


def make_ab(value: int) -> dict[str, int]:
    """Return a two-key mapping."""
    return {"a": value, "b": value + 1}


def take_a(a: int) -> int:
    """Return a."""
    return a


def need_b(b: int) -> int:
    """Return b."""
    return b


def add_xy(x: int, y: int) -> int:
    """Return x plus y."""
    return x + y


def make_text(value: int) -> str:
    """Return text."""
    return f"v{value}"


def make_tool(function: Any, name: str) -> Tool:
    return Tool(function=function, name=name, description=f"Test tool {name}.")


class _ExtraDescriptionTool(Tool):
    def __init__(self, *args: Any, extra: str, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._extra = extra

    def _extra_description(self) -> str:
        return self._extra


def build(steps: list[Any], name: str = "pipe") -> Pipeline:
    return Pipeline(name, "Test pipeline.", steps)


class TestConstruction:
    def test_rejects_non_list_steps(self) -> None:
        with pytest.raises(TypeError, match="non-empty list"):
            build((add_one,))  # type: ignore[arg-type]

    def test_rejects_empty_steps(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            build([])

    def test_rejects_wrong_tuple_length(self) -> None:
        with pytest.raises(ValueError, match="length 3"):
            build([(add_one, "k", "extra")])

    @pytest.mark.parametrize("entry", [42, None, object()])
    def test_rejects_unsupported_entry(self, entry: Any) -> None:
        with pytest.raises(TypeError):
            build([entry])

    def test_rejects_bad_binding_type(self) -> None:
        with pytest.raises(TypeError):
            build([(add_one, 5)])

    def test_rejects_duplicate_destinations(self) -> None:
        with pytest.raises(ValueError, match="duplicate"):
            build([(make_ab, {"a": "x", "b": "x"}), add_xy])

    def test_none_binding_equals_bare(self) -> None:
        pipe = build([(add_one, None), double])
        assert pipe.steps[0].binding is None

    def test_duplicate_instance_and_single_step_are_legal(self) -> None:
        step = make_tool(triple, "triple_step")
        pipe = build([(step, "value"), step])
        assert pipe.steps[0].invokable is pipe.steps[1].invokable
        single = build([step])
        assert len(single.steps) == 1

    def test_rejects_invalid_name(self) -> None:
        with pytest.raises(ValueError):
            Pipeline("bad name", "Test pipeline.", [add_one])


class TestCoercion:
    def test_str_step_becomes_prompt_template(self) -> None:
        pipe = build(["Hello {who}"], name="p")
        assert isinstance(pipe.steps[0].invokable, PromptTemplate)
        assert pipe.steps[0].invokable.name == "p_step_0"
        assert pipe.invoke({"who": "AA"}).result == "Hello AA"

    def test_callable_step_becomes_named_tool(self) -> None:
        pipe = build([add_one, double], name="p")
        step = pipe.steps[1].invokable
        assert isinstance(step, Tool)
        assert step.name == "p_step_1"
        assert step.description.startswith("Step 1 of pipeline p")

    def test_invokable_step_kept_as_is(self) -> None:
        tool = make_tool(add_one, "mine")
        pipe = build([tool])
        assert pipe.steps[0].invokable is tool


class TestSteps:
    def test_steps_is_tuple_of_frozen_steps(self) -> None:
        pipe = build([add_one, (double, "second")])
        assert isinstance(pipe.steps, tuple)
        assert all(isinstance(s, Pipeline.Step) for s in pipe.steps)
        with pytest.raises(FrozenInstanceError):
            pipe.steps[0].binding = "x"  # type: ignore[misc]


class TestSchema:
    def test_parameters_come_from_step_zero_regardless_of_binding(self) -> None:
        first = make_tool(add_one, "first")
        bare = build([first, double])
        bound = build([(first, "first"), double])
        assert bare.parameters == first.parameters
        assert bound.parameters == first.parameters
        assert not bare.has_varkwargs

    def test_return_type_bare_is_last_steps(self) -> None:
        last = make_tool(label, "last")
        pipe = build([add_one, double, last])
        assert pipe.return_type == last.return_type

    @pytest.mark.parametrize("binding", ["k", ["a"], {"a": "x"}, {}])
    def test_return_type_bound_is_dict(self, binding: Any) -> None:
        pipe = build([(make_ab, binding)])
        assert pipe.return_type == ("dict[str, Any]",)


class TestHandoff:
    def test_bare_mapping_passes_through(self) -> None:
        pipe = build([add_one, double, label])
        assert pipe.invoke({"value": 1}).result == "value=4"

    def test_str_binding_stores_non_mapping_under_key(self) -> None:
        pipe = build([(triple, "value"), add_one])
        assert pipe.invoke({"value": 2}).result == {"first": 7}

    def test_list_binding_filters(self) -> None:
        pipe = build([(make_ab, ["a"]), take_a])
        assert pipe.invoke({"value": 5}).result == 5

    def test_dict_binding_renames(self) -> None:
        pipe = build([(make_ab, {"a": "value"}), triple])
        assert pipe.invoke({"value": 5}).result == 15

    def test_int_selectors_unpack_tuple(self) -> None:
        pipe = build([(pair, {0: "x", 1: "y"}), add_xy])
        assert pipe.invoke({"value": 4}).result == 9

    def test_missing_selector_is_skipped_with_debug_log(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        pipe = build([(make_ab, {"a": "a", "zzz": "zzz"}), take_a])
        with caplog.at_level(logging.DEBUG, logger=PIPELINE_LOGGER):
            assert pipe.invoke({"value": 5}).result == 5
        assert any("zzz" in r.getMessage() for r in caplog.records)

    def test_str_binding_does_not_cover_next_steps_parameters(self) -> None:
        pipe = build([(make_ab, "blob"), add_xy])
        with pytest.raises(TypeError):
            pipe.invoke({"value": 1})


class TestFailurePaths:
    def test_bare_non_mapping_fails_in_next_step(self) -> None:
        pipe = build([triple, triple])
        with pytest.raises(TypeError):
            pipe.invoke({"value": 1})

    def test_selectors_on_str_response_raise(self) -> None:
        pipe = build([(make_text, ["x"]), take_a])
        with pytest.raises(TypeError, match="str/bytes"):
            pipe.invoke({"value": 1})

    def test_missing_required_key_in_next_step_raises(self) -> None:
        pipe = build([(make_ab, ["a"]), need_b])
        with pytest.raises(TypeError):
            pipe.invoke({"value": 1})

    def test_no_construction_time_destination_check(self) -> None:
        pipe = build([(make_ab, {"a": "nope"}), triple])
        assert pipe.steps[0].binding == {"a": "nope"}


class TestReturn:
    def test_last_bare_returns_raw_response(self) -> None:
        assert build([triple]).invoke({"value": 2}).result == 6

    def test_last_bound_partial_dict_is_silent(self) -> None:
        pipe = build([(make_ab, {"a": "a", "zzz": "z"})])
        assert pipe.invoke({"value": 2}).result == {"a": 2}

    def test_result_is_plain_atomic_result(self) -> None:
        result = build([add_one]).invoke({"value": 1})
        assert type(result) is AtomicResult
        assert not hasattr(result, "trace")


class TestExtraDescription:
    def test_chains_from_last_step_when_bare(self) -> None:
        last = _ExtraDescriptionTool(
            function=label, name="last", description="Last.", extra="EXTRA"
        )
        assert build([add_one, double, last])._extra_description() == "EXTRA"
        assert "EXTRA" in build([add_one, double, last]).description

    def test_empty_when_last_is_bound(self) -> None:
        last = _ExtraDescriptionTool(
            function=label, name="last", description="Last.", extra="EXTRA"
        )
        assert build([add_one, double, (last, "out")])._extra_description() == ""


class TestAsync:
    def test_async_mirrors_sync_handoff(self) -> None:
        pipe = build([(pair, {0: "x", 1: "y"}), add_xy])
        assert asyncio.run(pipe.async_invoke({"value": 4})).result == 9

    def test_async_bare_chain(self) -> None:
        pipe = build([add_one, double, label])
        assert asyncio.run(pipe.async_invoke({"value": 1})).result == "value=4"

    def test_async_str_binding_and_bound_last(self) -> None:
        pipe = build([(triple, "value"), (add_one, ["first"])])
        assert asyncio.run(pipe.async_invoke({"value": 2})).result == {"first": 7}


class TestLogging:
    def test_debug_only(self, caplog: pytest.LogCaptureFixture) -> None:
        pipe = build([add_one, double])
        with caplog.at_level(logging.DEBUG, logger=PIPELINE_LOGGER):
            pipe.invoke({"value": 1})
        records = [r for r in caplog.records if r.name == PIPELINE_LOGGER]
        assert records
        assert all(r.levelno == logging.DEBUG for r in records)


class TestSerialization:
    def test_to_dict_carries_bindings(self) -> None:
        pipe = build([add_one, (make_ab, {"a": "x"}), (triple, "value")])
        data = pipe.to_dict()
        assert data["step_count"] == 3
        assert [s["binding"] for s in data["steps"]] == [None, {"a": "x"}, "value"]
        assert data["steps"][0]["invokable"]["name"] == "pipe_step_0"
        assert "return_index" not in data


class TestSurface:
    def test_export(self) -> None:
        import atomic_agentic.composites as composites

        assert composites.__all__ == ["Pipeline"]
        assert composites.Pipeline is Pipeline

    def test_workflows_package_still_imports(self) -> None:
        import atomic_agentic.workflows  # noqa: F401
