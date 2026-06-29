from __future__ import annotations

import logging

from dotenv import load_dotenv

from atomic_agentic.a2a import PyA2AtomicHost
from atomic_agentic.agents import BasicAgent
from atomic_agentic.engines.LLMEngines import OpenAIEngine

load_dotenv()


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    llm = OpenAIEngine(model="gpt-4o-mini")

    trivia_agent = BasicAgent(
        name="TriviaAgent",
        namespace="trivia_host",
        description="A concise trivia agent that returns one interesting fact about the user's topic.",
        llm_engine=llm,
        role_prompt=(
            "You are a concise trivia specialist. "
            "When given a prompt or topic, return one accurate and interesting trivia fact "
            "in 2-4 sentences."
        ),
        context_enabled=False,
    )

    host = PyA2AtomicHost(
        invokables=[trivia_agent],
        name="trivia_host",
        description="PyA2AtomicHost exposing the TriviaAgent invokable.",
        version="1.0.0",
        host="localhost",
        port=6000,
    )
    host.run_server(debug=True)


if __name__ == "__main__":
    main()
