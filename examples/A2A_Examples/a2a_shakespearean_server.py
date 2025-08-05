import os
import sys
from pathlib import Path
from python_a2a import A2AServer, agent, run_server, TaskStatus, TaskState
from dotenv import load_dotenv
load_dotenv()

# Make sure Python can find your `modules/` folder
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from modules.Agents import Agent
from modules.LLMEngines import OpenAIEngine
from modules.Agents import Agent

llm = OpenAIEngine(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
my_agent = Agent(
    name="ShakespeareAgent",
    description="Responds in Shakespearean English.",
    llm_engine=llm,
    role_prompt="You are a helpful assistant that responds only in Shakespearean English.",
    context_enabled=True
)

@agent(
    name="ShakespeareAgent",
    description="An AI assistant that speaks only in Shakespearean English",
    version="1.0.0",
    url="http://localhost:5000"      # clients will connect here
)
class ShakespeareA2AServer(A2AServer):
    def __init__(self):
        super().__init__(google_a2a_compatible=True)        

    def handle_task(self, task):
        # 4) Unpack user text
        content = task.message.get("content", {}) or {}
        user_text = content.get("text", "")

        # 5) Invoke the Agent
        response_text = my_agent.invoke(user_text)

        # 6) Wrap and complete
        task.artifacts = [{
            "parts": [
                {"type": "text", "text": response_text}
            ]
        }]
        task.status = TaskStatus(state=TaskState.COMPLETED)
        return task

if __name__ == "__main__":
    # 7) Bind to all interfaces but advertise localhost for local tests
    run_server(
        ShakespeareA2AServer(),
        host="0.0.0.0",
        port=5000,
        debug=True
    )