import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from modules.A2Agents import A2AServerAgent
from modules.Agents import Agent
from modules.LLMEngines import OpenAIEngine

llm = OpenAIEngine(model = "gpt-4o-mini")
seed = Agent(
    name = "ShakespeareAgent",
    llm_engine = llm,
    role_prompt = "You are a helpful assistant that responds only in Shakespearean English.",
    description= "Responds in Shakespearean English.",
    context_enabled=False)

if __name__ == "__main__":
    A2AServerAgent(seed, host = "localhost", port = 5000).run()