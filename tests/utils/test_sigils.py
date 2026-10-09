from __future__ import annotations

from atomic_agentic.models.agents.blackboard_models import ToolStatement
from atomic_agentic.utils.sigils import render_cache_snapshot


# --------------------------------------------------------------------------- #
# render_cache_snapshot -- "$" prefix coverage only. Mirrors
# tests/utils/test_script.py's own same-named-function tests in shape
# (empty-input, bound-identifiers, truncation); this file stays scoped to
# the one behavior this task changed, not a full sigils.py sweep.
# --------------------------------------------------------------------------- #
class TestRenderCacheSnapshotSigil:
    def test_empty_completed_returns_empty_string(self) -> None:
        assert render_cache_snapshot([], {}, None) == ""

    def test_bound_name_is_prefixed_with_sigil(self) -> None:
        stmt = ToolStatement(identifier="a", tool="tool_a")

        rendered = render_cache_snapshot([stmt], {"a": 1}, None)

        assert "$a: int = 1" in rendered
        assert "\na: int = 1" not in rendered
        assert not rendered.startswith("```\nCached values:\na:")

    def test_multiple_bound_names_all_sigiled(self) -> None:
        a = ToolStatement(identifier="a", tool="tool_a")
        b = ToolStatement(identifier="b", tool="tool_b")

        rendered = render_cache_snapshot([a, b], {"a": 1, "b": "two"}, None)

        assert "$a: int = 1" in rendered
        assert "$b: str = \"two\"" in rendered
