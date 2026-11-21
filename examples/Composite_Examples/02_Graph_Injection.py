"""
02_Graph_Injection.py

Building a Graph by injection with plain functions (no LLM, no keys).

Demonstrates:
- add_sequence on an empty graph, then bridging a second chain onto an
  existing node with start_from / end_at
- add_sequence(..., repeat_until={...}): a loop that runs until state matches,
  with the depth cap as the safety net for a loop that never matches
- add_parallel with fan_out / fan_in: branches that run in one superstep and a
  merge node that runs once after them, with `returns` projecting named results

Run from the repo root:
    python examples/Composite_Examples/02_Graph_Injection.py
"""

from __future__ import annotations

from atomic_agentic.composites import Graph


# ---------------------------------------------------------------------
# 1. add_sequence: a chain, then a second chain bridged onto it
# ---------------------------------------------------------------------
def add_one(x: int) -> dict:
    return {"x": x + 1}


def double(x: int) -> dict:
    return {"x": x * 2}


def square(x: int) -> dict:
    return {"x": x * x}


def negate(x: int) -> dict:
    return {"x": -x}


def sequence_demo() -> None:
    print("\n=== 1. add_sequence and bridging ===")
    graph = Graph("arithmetic", "Arithmetic chains over one state key.", returns="x")
    graph.add_sequence({"add_one": add_one, "double": double, "square": square})
    graph.set_start("add_one")
    result = graph.invoke({"x": 2})
    print("(2 + 1) * 2 squared =", result.result, "| trace:", result.trace)

    # start_from hangs the new chain off the end of the existing one
    graph.add_sequence({"negate": negate}, start_from="square")
    result = graph.invoke({"x": 2})
    print("then negated        =", result.result, "| trace:", result.trace)

    # end_at routes the new chain's last node into an existing node
    bridged = Graph("bridged", "A chain that joins an existing node.", returns="x")
    bridged.add_node("square", square)
    bridged.add_sequence({"add_one": add_one, "double": double}, end_at="square")
    bridged.set_start("add_one")
    result = bridged.invoke({"x": 2})
    print("end_at join         =", result.result, "| trace:", result.trace)


# ---------------------------------------------------------------------
# 2. repeat_until: loop until the state matches
# ---------------------------------------------------------------------
def tick(count: int = 0) -> dict:
    return {"count": count + 1}


def check(count: int) -> dict:
    return {"done": count >= 3}


def loop_demo() -> None:
    print("\n=== 2. repeat_until ===")
    loop = Graph("counter_loop", "Count until done.", returns="count", max_traversal_depth=10)
    loop.add_sequence({"tick": tick, "check": check}, repeat_until={"done": True})
    loop.set_start("tick")
    result = loop.invoke({})
    iterations = sum(1 for names in result.trace if names == ("tick",))
    print("count =", result.result, "| iterations =", iterations)
    print("trace:", [names[0] for names in result.trace])

    # a condition that never matches is stopped by the depth cap
    stuck = Graph("stuck_loop", "Never done.", max_traversal_depth=6)
    stuck.add_sequence({"tick": tick, "check": check}, repeat_until={"done": "never"})
    stuck.set_start("tick")
    try:
        stuck.invoke({})
    except RuntimeError as exc:
        print("RuntimeError:", exc)


# ---------------------------------------------------------------------
# 3. add_parallel: fan_out, branches in one superstep, fan_in
# ---------------------------------------------------------------------
def load(text: str) -> dict:
    return {"text": text.strip()}


def word_stats(text: str) -> dict:
    words = text.split()
    return {"word_count": len(words), "unique_words": len({w.lower() for w in words})}


def line_stats(text: str) -> dict:
    lines = text.splitlines()
    return {"line_count": len(lines), "non_empty_lines": len([ln for ln in lines if ln.strip()])}


def text_flags(text: str) -> dict:
    return {"has_question": "?" in text, "is_long": len(text) > 40}


def merge_report(word_count: int, line_count: int, has_question: bool) -> dict:
    return {"report": f"{word_count} words, {line_count} lines, question={has_question}"}


def parallel_demo() -> None:
    print("\n=== 3. add_parallel ===")
    graph = Graph(
        "text_analysis",
        "Analyse text in parallel branches.",
        returns=["word_count", "line_count", "has_question", "report"],
    )
    graph.add_node("load", load)
    graph.add_node("merge_report", merge_report)
    graph.add_parallel(
        {"word_stats": word_stats, "line_stats": line_stats, "text_flags": text_flags},
        fan_out="load",
        fan_in="merge_report",
    )
    graph.set_start("load")
    result = graph.invoke({"text": "Is this one superstep?\nYes, it is.\n"})
    for step, names in enumerate(result.trace, start=1):
        print(f"  superstep {step}: {names}")
    fired = {}
    for names in result.trace:
        for name in names:
            fired[name] = fired.get(name, 0) + 1
    print("runs per node:", fired)
    print("result:", result.result)


def main() -> None:
    sequence_demo()
    loop_demo()
    parallel_demo()


if __name__ == "__main__":
    main()
