# trivia_host_server.py
"""
Trivia A2A Host Server
----------------------
A basic (non-tool-using) Agent hosted over A2A using A2AgentHost.

Run:
  python trivia_host_server.py
Then test:
  python a2a_proxy_client.py trivia
"""
from dotenv import load_dotenv

from atomic_agentic.agents import Agent
from atomic_agentic.a2a import A2AgentHost
from atomic_agentic.engines.LLMEngines import OpenAIEngine

load_dotenv()


def main() -> None:
    llm = OpenAIEngine(model="gpt-4o-mini")

    seed = Agent(
        name="TriviaAgent",
        description="Trivia expert. Returns accurate, concise facts.",
        llm_engine=llm,
        context_enabled=False,
        role_prompt=(
            "You are a trivia expert.\n"
            "The user provides a 'prompt'.\n"
            "Respond with 1–3 sentences, factually accurate, no fluff.\n"
            "If the prompt asks for multiple items, comply, but keep it concise.\n"
        ),
    )

    host = A2AgentHost(seed_agent=seed, host="localhost", port=6000, version="1.0.0")
    host.run(debug=True)


if __name__ == "__main__":
    main()
