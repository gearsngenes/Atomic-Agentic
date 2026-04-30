from __future__ import annotations

from atomic_agentic.agents.data_classes import AgentTurn, ToolAgentTurn


class TestAgentTurn:
    def test_to_dict_returns_prompt_raw_response_and_final_response(self) -> None:
        turn = AgentTurn(
            prompt="write a summary",
            raw_response="raw assistant text",
            final_response={"final": "processed assistant text"},
        )

        assert turn.to_dict() == {
            "prompt": "write a summary",
            "raw_response": "raw assistant text",
            "final_response": {"final": "processed assistant text"},
        }

    def test_to_dict_returns_independent_top_level_mapping(self) -> None:
        turn = AgentTurn(
            prompt="collect items",
            raw_response="raw",
            final_response={"items": ["a", "b"]},
        )

        data = turn.to_dict()
        data["prompt"] = "mutated"

        assert turn.prompt == "collect items"
        assert turn.final_response == {"items": ["a", "b"]}


class TestToolAgentTurn:
    def test_to_dict_includes_agent_turn_fields_and_blackboard_span(self) -> None:
        turn = ToolAgentTurn(
            prompt="run tools",
            raw_response=42,
            final_response={"value": 42},
            blackboard_start=3,
            blackboard_end=6,
        )

        assert turn.to_dict() == {
            "prompt": "run tools",
            "raw_response": 42,
            "final_response": {"value": 42},
            "blackboard_start": 3,
            "blackboard_end": 6,
        }

    def test_blackboard_span_defaults_to_none(self) -> None:
        turn = ToolAgentTurn(
            prompt="run without context",
            raw_response="raw",
            final_response="final",
        )

        assert turn.blackboard_start is None
        assert turn.blackboard_end is None
        assert turn.to_dict() == {
            "prompt": "run without context",
            "raw_response": "raw",
            "final_response": "final",
            "blackboard_start": None,
            "blackboard_end": None,
        }