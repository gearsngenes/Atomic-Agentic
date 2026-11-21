from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from atomic_agentic.composites import Pipeline
from atomic_agentic.invokables.prompt_template import PromptTemplate
from atomic_agentic.models.parameters import ParamSpec
from atomic_agentic.models.results import AtomicResult
from atomic_agentic.tools.base import Tool


def add_one(value: int) -> dict[str, int]:
    """Add one and report it as first."""
    return {"first": value + 1}


def double(first: int) -> dict[str, int]:
    """Double first and report it as second."""
    return {"second": first * 2}


def label(second: int) -> str:
    """Render second as text."""
    return f"value={second}"


def add_bonus(value: int, bonus: int = 0) -> int:
    """Add bonus to value."""
    return value + bonus


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


def add_xy(x: int, y: int) -> int:
    """Return x plus y."""
    return x + y


def make_text(value: int) -> str:
    """Return text."""
    return f"v{value}"


def add_first_and_tripled(first: int, tripled: int) -> int:
    """Return first plus tripled."""
    return first + tripled


def add_value_and_second(value: int, second: int) -> int:
    """Return value plus second."""
    return value + second


def collect_kwargs(**kwargs: Any) -> dict[str, Any]:
    """Return the keyword payload."""
    return dict(kwargs)


def add_to_n(n: int) -> dict[str, int]:
    """Increment n."""
    return {"n": n + 1}


def make_tool(function: Any, name: str) -> Tool:
    return Tool(function=function, name=name, description=f"Test tool {name}.")


def build(steps: list[Any], name: str = "pipe", **kwargs: Any) -> Pipeline:
    return Pipeline(name, "Test pipeline.", steps, **kwargs)


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

    def test_empty_last_binding_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one destination"):
            build([add_one, (make_ab, {})])

    def test_empty_intermediate_binding_is_allowed(self) -> None:
        pipe = build([(add_one, {}), triple])
        assert pipe.steps[0].binding == {}

    def test_rejects_invalid_name(self) -> None:
        with pytest.raises(ValueError):
            Pipeline("bad name", "Test pipeline.", [add_one])


class TestCoercion:
    def test_str_step_becomes_prompt_template(self) -> None:
        pipe = build(["Hello {who}"], name="p")
        assert isinstance(pipe.steps[0].invokable, PromptTemplate)
        assert pipe.steps[0].invokable.name == "p_0"
        assert pipe.invoke({"who": "AA"}).result == "Hello AA"

    def test_callable_step_becomes_named_tool(self) -> None:
        pipe = build([add_one, double], name="p")
        step = pipe.steps[1].invokable
        assert isinstance(step, Tool)
        assert step.name == "p_1"
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

    @pytest.mark.parametrize("binding", [["a"], {"a": "x"}])
    def test_return_type_bound_collection_is_dict(self, binding: Any) -> None:
        pipe = build([(make_ab, binding)])
        assert pipe.return_type == ("dict[str, Any]",)

    def test_return_type_bound_str_is_any(self) -> None:
        assert build([(triple, "k")]).return_type == ("Any",)

    def test_explicit_parameters_forward_undeclared_inputs(self) -> None:
        extra = [
            ParamSpec(name="value", index=0, kind=ParamSpec.KEYWORD_ONLY, type=("Any",)),
            ParamSpec(name="bonus", index=1, kind=ParamSpec.KEYWORD_ONLY, type=("Any",)),
        ]
        pipe = build([(triple, "value"), (add_bonus, "total")], parameters=extra)
        assert pipe.parameters == extra
        assert pipe.invoke({"value": 2, "bonus": 10}).result == 16

    def test_undeclared_inputs_are_dropped_without_parameters(self) -> None:
        pipe = build([(triple, "value"), (add_bonus, "total")])
        assert pipe.invoke({"value": 2, "bonus": 10}).result == 6


class TestState:
    def test_bare_mapping_merges_into_state(self) -> None:
        pipe = build([add_one, double, label])
        assert pipe.invoke({"value": 1}).result == "value=4"

    def test_bare_non_mapping_intermediate_raises(self) -> None:
        pipe = build([triple, triple])
        with pytest.raises(TypeError, match="unbound"):
            pipe.invoke({"value": 1})

    def test_bound_intermediate_stores_its_keys(self) -> None:
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

    def test_later_step_reads_non_adjacent_key(self) -> None:
        # step 2 reads "first" (step 0) and "tripled" (step 1), skipping adjacency
        pipe = build([add_one, (triple, "tripled"), add_first_and_tripled])
        assert pipe.invoke({"value": 1}).result == 2 + 3

    def test_step_zero_inputs_stay_visible(self) -> None:
        pipe = build([add_one, double, add_value_and_second])
        assert pipe.invoke({"value": 1}).result == 1 + 4

    def test_str_binding_does_not_cover_next_steps_parameters(self) -> None:
        pipe = build([(make_ab, "blob"), add_xy])
        with pytest.raises(TypeError):
            pipe.invoke({"value": 1})

    def test_extra_inputs_reach_kwargs_step_zero(self) -> None:
        pipe = build([collect_kwargs])
        assert pipe.invoke({"p": 1, "q": 2}).result == {"p": 1, "q": 2}


class TestReturn:
    def test_last_bare_returns_raw_response_of_any_type(self) -> None:
        assert build([triple]).invoke({"value": 2}).result == 6
        assert build([make_text]).invoke({"value": 2}).result == "v2"
        assert build([pair]).invoke({"value": 2}).result == (2, 3)

    def test_last_bound_str_returns_stored_value(self) -> None:
        assert build([(triple, "k")]).invoke({"value": 2}).result == 6

    def test_last_bound_list_returns_dict(self) -> None:
        assert build([(make_ab, ["a"])]).invoke({"value": 2}).result == {"a": 2}

    def test_last_bound_dict_returns_destinations(self) -> None:
        pipe = build([(make_ab, {"a": "x", "b": "y"})])
        assert pipe.invoke({"value": 2}).result == {"x": 2, "y": 3}

    def test_missing_last_selector_raises_key_error(self) -> None:
        pipe = build([(make_ab, {"a": "a", "zzz": "z"})])
        with pytest.raises(KeyError, match="z"):
            pipe.invoke({"value": 2})

    def test_result_is_plain_atomic_result(self) -> None:
        result = build([add_one]).invoke({"value": 1})
        assert type(result) is AtomicResult
        assert not hasattr(result, "trace")


class TestDepth:
    def test_more_than_fifty_steps_run(self) -> None:
        pipe = build([add_to_n for _ in range(60)])
        # the last step is bare, so the raw response is the last mapping
        assert pipe.invoke({"n": 0}).result == {"n": 60}

    def test_single_step_is_valid(self) -> None:
        assert build([add_one]).invoke({"value": 1}).result == {"first": 2}

    def test_duplicate_instances_run_twice(self) -> None:
        step = make_tool(add_to_n, "bump")
        assert build([step, step]).invoke({"n": 0}).result == {"n": 2}


class TestAsync:
    def test_async_mirrors_sync(self) -> None:
        pipe = build([(pair, {0: "x", 1: "y"}), add_xy])
        assert asyncio.run(pipe.async_invoke({"value": 4})).result == 9
        assert pipe.invoke({"value": 4}).result == 9

    def test_async_bare_chain(self) -> None:
        pipe = build([add_one, double, label])
        assert asyncio.run(pipe.async_invoke({"value": 1})).result == "value=4"

    def test_async_str_binding_and_bound_last(self) -> None:
        pipe = build([(triple, "value"), (add_one, ["first"])])
        assert asyncio.run(pipe.async_invoke({"value": 2})).result == {"first": 7}
        pipe = build([(triple, "value"), (triple, "out")])
        assert asyncio.run(pipe.async_invoke({"value": 2})).result == 18


class TestSerialization:
    def test_to_dict_carries_bindings(self) -> None:
        pipe = build([add_one, (make_ab, {"a": "x"}), (triple, "value")])
        data = pipe.to_dict()
        assert data["step_count"] == 3
        assert [s["binding"] for s in data["steps"]] == [None, {"a": "x"}, "value"]
        assert data["steps"][0]["invokable"]["name"] == "pipe_0"
        assert "return_index" not in data


class TestSurface:
    def test_export(self) -> None:
        import atomic_agentic.composites as composites

        assert composites.__all__ == ["Graph", "Pipeline", "Router"]
        assert composites.Pipeline is Pipeline

    def test_workflows_package_still_imports(self) -> None:
        import atomic_agentic.workflows  # noqa: F401
