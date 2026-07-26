from __future__ import annotations

from atomic_agentic.models.agents.thought_models import AgentThought


class TestAgentThought:
    def test_stores_observation_question_and_answer(self) -> None:
        thought = AgentThought(
            observation="The task is a poem about Paris.",
            question="What rhyming scheme should the poem use?",
            answer="AABB.",
        )
        assert thought.observation == "The task is a poem about Paris."
        assert thought.question == "What rhyming scheme should the poem use?"
        assert thought.answer == "AABB."

    def test_observation_accepts_none(self) -> None:
        thought = AgentThought(observation=None, question="Why?", answer="Because.")
        assert thought.observation is None

    def test_to_dict_returns_all_fields(self) -> None:
        thought = AgentThought(observation="obs", question="q", answer="a")
        assert thought.to_dict() == {
            "observation": "obs",
            "question": "q",
            "answer": "a",
        }

    def test_to_dict_preserves_none_observation(self) -> None:
        thought = AgentThought(observation=None, question="q", answer="a")
        assert thought.to_dict()["observation"] is None
