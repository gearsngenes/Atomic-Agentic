import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from modules.A2Agents import A2AServerAgent
from modules.Agents import Agent
from modules.LLMEngines import OpenAIEngine

def main():
    """
    Trivia A2A server
    - Seed is a schema-driven Agent that expects a mapping (e.g., {"prompt": "..."}).
    - A2AServerAgent wraps the seed and exposes message-level function responses.
    """
    llm = OpenAIEngine(model="gpt-4o-mini")

    seed = Agent(
        name="TriviaAgent",
        llm_engine=llm,
        role_prompt=(
            "You respond to a 'prompt' about animals with ONE short fun fact. "
            "If the prompt does not include an animal, default to dogs."
        ),
        description="Gives a single fun animal fact.",
        context_enabled=False,
    )

    server = A2AServerAgent(
        seed=seed,
        name="Trivia Agent",
        description="Animal trivia over A2A",
        host="localhost",
        port=6000,
        version="1.0.0",
    )
    server.run(debug=True)

if __name__ == "__main__":
    main()
