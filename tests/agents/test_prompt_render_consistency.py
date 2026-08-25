"""
Regression guard against prompt-text-vs-real-rendered-output drift.

Found during the v2.0.0a30 pre-release integrity audit: PLANNER_PROMPT
claimed prior assistant turns look like "CACHE STEPS #X-Y PRODUCED:", while
ToolAgent.render_turn() actually rendered "CACHED STEPS [0, 1, 2] PRODUCED:"
-- silently wrong across two releases. Worse, test_react.py's
test_injects_running_plan_after_first_step was passing the whole time only
because its assertion coincidentally matched the *prompt's own* stale
example text (concatenated into the joined message string it searched),
not the real rendered snapshot it claimed to test.

Each test here extracts the header phrase directly from the live prompt
template -- never a second hardcoded copy of "what the prompt is supposed
to say" -- and asserts that exact phrase appears in real rendered output
from an actual scripted agent invocation. Drift on either side (prompt
text edited without updating the renderer, or vice versa) fails here
immediately instead of silently.
"""
from __future__ import annotations

import re

from .conftest import make_agent, make_react_agent, register_math_tools, react_step_json, FakeLLMEngine

from atomic_agentic.agents.toolagent import return_tool
from atomic_agentic.agents.prompts import PLANNER_PROMPT, ORCHESTRATOR_PROMPT


def _extract_quoted_header(template: str, anchor: str) -> str:
    """Pull the quoted example-header string containing ``anchor`` out of a
    prompt template, e.g. '"CACHED STEPS [i, j, ...] PRODUCED:"' ->
    'CACHED STEPS [i, j, ...] PRODUCED:'. Fails loudly (not a silent None)
    if the prompt no longer contains such a quoted string -- that absence
    is itself a drift signal worth catching here."""
    match = re.search(rf'"([^"]*{re.escape(anchor)}[^"]*)"', template)
    assert match is not None, (
        f"{anchor!r} not found as a quoted example header in the prompt "
        "template -- the prompt may have been reworded without updating "
        "this test."
    )
    return match.group(1)


class TestPlannerPromptCachedStepsMatchesRenderTurn:
    """PLANNER_PROMPT's '# CONTEXT YOU MAY SEE' claims about
    ToolAgent.render_turn()'s cross-turn cache rendering, checked against a
    real scripted invocation rather than a second hardcoded copy of the
    expected text. render_turn() is shared by PlanActAgent and ReActAgent
    alike (defined once on ToolAgent), so PLANNER_PROMPT is as valid a
    place to anchor this as ORCHESTRATOR_PROMPT would be."""

    def test_all_executed_header_matches_real_render_turn_output(self) -> None:
        header = _extract_quoted_header(PLANNER_PROMPT.template, "CACHED STEPS")
        # header looks like "CACHED STEPS [i, j, ...] PRODUCED:" -- the
        # bracketed middle is illustrative, not literal; check the static
        # text around it.
        prefix = header.split("[")[0]
        suffix = header.rsplit("]", 1)[1]

        agent = make_agent(context_enabled=True)
        keys = register_math_tools(agent)
        agent.set_script([
            [{"tool": keys["add"], "args": {"x": 2, "y": 3}}],
            [{"tool": return_tool.full_name, "args": {"val": "|STEP.0|"}}],
        ])
        result = agent.invoke({"prompt": "run"})
        assert result.result == 5

        rendered = agent.render_turn(agent.records[0])[1]["content"]
        assert prefix in rendered
        assert suffix in rendered
        # And the bracketed middle really is a Python list of ints, not the
        # prompt's own illustrative "[i, j, ...]" placeholder text.
        assert re.search(r"CACHED STEPS \[\d+(, \d+)*\] PRODUCED:", rendered)

    def test_failed_steps_header_matches_real_render_turn_output(self) -> None:
        header = _extract_quoted_header(PLANNER_PROMPT.template, "FAILED STEPS")
        prefix = header.split("[")[0]

        agent = make_agent(context_enabled=True, fail_fast=False)
        register_math_tools(agent)
        agent.set_script([
            [
                {"tool": "Tool.tests.fail_tool", "args": {}},
                {"tool": "Tool.tests.add", "args": {"x": 3, "y": 4}},
            ],
            [{"tool": return_tool.full_name, "args": {"val": "|STEP.1|"}}],
        ])
        result = agent.invoke({"prompt": "run"})
        assert result.result == 7

        rendered = agent.render_turn(agent.records[0])[1]["content"]
        assert prefix in rendered
        assert re.search(r"FAILED STEPS \[\d+(, \d+)*\]:", rendered)


class TestOrchestratorPromptStepsSnapshotMatchesRender:
    """ORCHESTRATOR_PROMPT's own worked '# EXAMPLE' running-plan snapshot
    header, checked against ReActAgent._render_task_messages()'s real
    output rather than a second hardcoded copy of the expected text."""

    def test_steps_so_far_header_matches_real_snapshot_output(self) -> None:
        match = re.search(r"(STEPS )\d+-\d+( SO FAR:)", ORCHESTRATOR_PROMPT.template)
        assert match is not None, (
            "'STEPS N-M SO FAR:' not found in ORCHESTRATOR_PROMPT's worked "
            "example -- the prompt may have been reworded without updating "
            "this test."
        )
        prefix, suffix = match.group(1), match.group(2)

        agent = make_react_agent(
            [
                react_step_json(step=0, tool="Tool.tests.add", args={"x": 2, "y": 3}),
                react_step_json(step=1, tool=return_tool.full_name, args={"val": "|STEP.0|"}),
            ],
            tool_calls_limit=1,
        )
        result = agent.invoke({"prompt": "run"})
        assert result.result == 5

        engine = agent.llm_engine
        assert isinstance(engine, FakeLLMEngine)
        second_call_text = "\n".join(message["content"] for message in engine.calls[1])
        assert prefix in second_call_text
        assert suffix in second_call_text
        assert re.search(rf"{re.escape(prefix)}\d+-\d+{re.escape(suffix)}", second_call_text)
