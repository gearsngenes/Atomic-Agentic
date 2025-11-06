import sys
from pathlib import Path

# Ensure local modules can be imported when running from examples/
sys.path.append(str(Path(__file__).resolve().parents[2]))

from modules.A2Agents import A2AServerAgent
from modules.Agents import Agent
from modules.LLMEngines import OpenAIEngine

def main():
    """
    Exercise: run a Shakespearean server that exposes a seed Agent over A2A.

    New server contract:
      - Construct A2AServerAgent with (seed, name, description, url, version).
      - The URL MUST be an absolute URL (with scheme). This goes into Agent Card.
      - Start with .run(host, port, debug).
      - Server routes function calls in handle_task:
          'invoke'         -> seed.invoke(payload: Mapping)
          'arguments_map'  -> exports tool signature (JSON-safe)
          'agent_meta'     -> basic metadata
    """

    # Choose your OpenAI model here; the base Agent requires the engine to return str.
    # Be sure your environment is configured with proper API credentials.
    llm = OpenAIEngine(model="gpt-4o-mini")

    seed = Agent(
        name="ShakespeareAgent",
        llm_engine=llm,
        role_prompt="You are a helpful assistant that responds only in Shakespearean English.",
        description="Responds in Shakespearean English.",
        context_enabled=False,
    )

    # IMPORTANT: URL must include scheme and correct port for the Agent Card
    server = A2AServerAgent(
        seed=seed,
        name="Shakespeare Agent",
        description="Shakespearean English responses via A2A.",
        host="localhost",
        port=5000,
        version="1.0.0",
    )

    # Run HTTP server (matching the URL's port)
    server.run(debug=True)

if __name__ == "__main__":
    main()
