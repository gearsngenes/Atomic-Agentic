"""06_replanning_demo.py

Demonstrates ScriptActAgent's framework-only repair mechanic (the
`scriptact-repair-rework` pass) live: a task deliberately shaped so the
model is coerced -- unknowingly, not as a "handle this edge case" test --
into dispatching a call that raises, forcing a real repair round, then
recovers on its own using nothing but the failure feedback it's shown.

The trap: `divide(numerator, denominator)` is an ordinary, unguarded
`a / b`. The task hands the model three same-shaped rows of exam data to
average via `divide(total_points, number_of_students)` -- one row just
happens to have `number_of_students=0` (nobody in that class took the
quiz). Nothing in the task calls out that row as special, so the most
natural first plan calls `divide(0, 0)` for it exactly like the other two,
raising `ZeroDivisionError` on dispatch. Only *after* seeing that failure
(via the new `# THIS BATCH FAILED:` section `ONESHOT_PLANNER_PROMPT` now
teaches) does the model have any reason to special-case it -- there is no
way to dodge the failure on the first attempt without being told the
answer up front, which would defeat the point.

After the run, prints `result.repair_rounds_used`/`result.regenerations_used`
(new this pass -- `ScriptActAgentResult`/`ScriptActAgentRecord` didn't
surface either counter past the ephemeral task before) alongside the
reconstructed script and the raised failure itself, so all three ways of
inspecting what happened -- result, record, live exception -- agree.
"""
import logging

from atomic_agentic.agents import ScriptActAgent

from shared_engine import llm_engine

logging.basicConfig(level=logging.INFO)

print("Testing ScriptActAgent's framework-triggered repair mechanic")


def divide(numerator: int, denominator: int) -> float:
    """Divide numerator by denominator. No zero-check -- a genuine
    unguarded division, the way a first draft of this function usually
    looks before anyone hits the edge case."""
    return numerator / denominator


# ──────────────────────────  SET-UP  ───────────────────────────
agent = ScriptActAgent(
    name="Test_ScriptActAgent",
    namespace="examples",
    description="Testing framework-only repair rounds via a coerced divide-by-zero.",
    llm_engine=llm_engine,
    context_enabled=True,
    replanning_limit=2,
    fail_fast=False,
)
agent.register_tool(divide)

# ──────────────────────────  TASK  ─────────────────────────────
# Class C's row is presented exactly like A and B's -- same shape, same
# wording -- so nothing hints it needs different handling. Whatever the
# model does about it happens only in the repair round, driven by the
# actual ZeroDivisionError it's shown, not a rule it was handed in advance.
task_prompt = """
Three classes took a pop quiz. For each class, compute its average score
as divide(total_points, number_of_students), then print each result as
'<class>: <average>'.

- Class A: total_points=170, number_of_students=2
- Class B: total_points=240, number_of_students=3
- Class C: total_points=0, number_of_students=0
"""

print("\n⇢ Executing quiz-average demo …")
result = agent.invoke({"prompt": task_prompt})

print("\n=== FINAL AGENT RESULT ===")
print(result.result)

print("\n=== BUDGET ACCOUNTING (new this pass) ===")
print(f"regenerations_used: {result.regenerations_used}")
print(f"repair_rounds_used: {result.repair_rounds_used}")

record = agent.get_conversation()[-1]
print("\n=== EXECUTED SCRIPT (successful statements only) ===")
print(record.render_as_code())

if record.failed_statements:
    print("\n=== WHAT ACTUALLY FAILED (triggered the repair round) ===")
    for slot in record.failed_statements:
        print(f"{slot.tool}(args={slot.args!r}) -> {slot.exception!r}")
else:
    print("\n(No failure recorded -- the model somehow avoided the trap on round 1.)")
