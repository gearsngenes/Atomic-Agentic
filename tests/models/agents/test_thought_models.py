from __future__ import annotations

from atomic_agentic.models.agents.thought_models import AgentThought


class TestAgentThought:
    def test_stores_category_and_content(self) -> None:
        thought = AgentThought(category="OBSERVATION", content="The task is a poem about Paris.")
        assert thought.category == "OBSERVATION"
        assert thought.content == "The task is a poem about Paris."

    def test_to_dict_returns_all_fields(self) -> None:
        thought = AgentThought(category="QUESTION", content="What rhyming scheme should the poem use?")
        assert thought.to_dict() == {
            "category": "QUESTION",
            "content": "What rhyming scheme should the poem use?",
        }
