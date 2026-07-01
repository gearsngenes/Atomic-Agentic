from __future__ import annotations

from atomic_agentic.constants.agents import RUN_ID_PARAM


class TestRunIdParam:
    def test_run_id_param_has_non_empty_description(self) -> None:
        assert isinstance(RUN_ID_PARAM.description, str)
        assert RUN_ID_PARAM.description.strip() != ""

    def test_run_id_param_identity(self) -> None:
        assert RUN_ID_PARAM.name == "run_id"
        assert RUN_ID_PARAM.kind == "KEYWORD_ONLY"
        assert RUN_ID_PARAM.default is None
