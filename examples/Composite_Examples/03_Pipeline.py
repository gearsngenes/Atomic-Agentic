"""
03_Pipeline.py

Pipeline: a fixed chain of steps with plain functions (no LLM, no keys).

Demonstrates:
- a bare chain: each non-last step returns a dict that merges into shared state
- bound steps and how the LAST step's binding shapes the result and return_type
- the optional parameters= override for an input only a later step declares
- a Pipeline used as a step of another Pipeline, and to_dict()

Run from the repo root:
    python examples/Composite_Examples/03_Pipeline.py
"""

from __future__ import annotations

from typing import Any

from atomic_agentic.composites import Pipeline
from atomic_agentic.models.parameters import ParamSpec


# ---------------------------------------------------------------------
# Step functions
# ---------------------------------------------------------------------
def add_and_carry(a: int, b: int) -> dict[str, int]:
    return {"total": a + b, "carried": a}


def multiply(total: int, carried: int) -> dict[str, int]:
    return {"product": total * carried}


def to_message(product: int) -> str:
    return f"The product is {product}"


def stats(product: int) -> dict[str, Any]:
    return {"is_even": product % 2 == 0, "digits": len(str(product)), "extra": "unused"}


def scale(value: int) -> dict[str, int]:
    return {"scaled": value * 10}


def tag(scaled: int, prefix: str = "item") -> str:
    return f"{prefix}-{scaled}"


def bare_chain_demo() -> None:
    print("\n=== 1. Bare chain ===")
    pipe = Pipeline("bare_chain", "Add, multiply, then phrase.", [add_and_carry, multiply, to_message])
    result = pipe.invoke({"a": 2, "b": 5})
    print("result    :", result.result)
    print("invoker_id:", result.invoker_id)
    print("type      :", type(result).__name__, "(no trace)")
    print("return_type:", pipe.return_type)


def binding_demo() -> None:
    print("\n=== 2. The last step's binding ===")
    head = [add_and_carry, multiply]
    shapes = {
        "bare": head + [to_message],
        "str": head + [(to_message, "message")],
        "list": head + [(stats, ["is_even", "digits"])],
        "dict": head + [(stats, {"is_even": "even", "digits": "size"})],
    }
    for label, steps in shapes.items():
        pipe = Pipeline(f"shape_{label}", f"Last step bound as {label}.", steps)
        result = pipe.invoke({"a": 2, "b": 5}).result
        print(f"{label:5}: {result!r:55} return_type={pipe.return_type}")


def parameters_demo() -> None:
    print("\n=== 3. parameters= override ===")
    steps = [scale, tag]
    plain = Pipeline("tagger", "Scale then tag.", steps)
    print("declared inputs  :", [p.name for p in plain.parameters])
    print("without override :", plain.invoke({"value": 3, "prefix": "box"}).result)

    # step 0 declares only "value"; declaring "prefix" lets it reach the later step
    declared = [
        ParamSpec(name="value", index=0, kind=ParamSpec.KEYWORD_ONLY, type=("int",)),
        ParamSpec(name="prefix", index=1, kind=ParamSpec.KEYWORD_ONLY, type=("str",)),
    ]
    forwarded = Pipeline("tagger_with_prefix", "Scale then tag.", steps, parameters=declared)
    print("declared inputs  :", [p.name for p in forwarded.parameters])
    print("with override    :", forwarded.invoke({"value": 3, "prefix": "box"}).result)


def nesting_demo() -> None:
    print("\n=== 4. A Pipeline as a step ===")
    inner = Pipeline("inner", "Add, multiply, phrase.", [add_and_carry, multiply, to_message])
    outer = Pipeline(
        "outer",
        "Run the inner pipeline, then shout the message.",
        [(inner, "message"), (lambda message: message.upper(), "shouted")],
    )
    print("result:", outer.invoke({"a": 3, "b": 4}).result)
    data = outer.to_dict()
    print("to_dict keys :", sorted(data))
    print("outer steps  :", [(s["invokable"]["type"], s["binding"]) for s in data["steps"]])


def main() -> None:
    bare_chain_demo()
    binding_demo()
    parameters_demo()
    nesting_demo()


if __name__ == "__main__":
    main()
