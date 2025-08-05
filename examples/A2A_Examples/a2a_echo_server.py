import uuid
from python_a2a import A2AServer, agent, run_server, TaskStatus, TaskState

@agent(
    name="EchoAgent",
    description="An A2A echo agent that echoes back the user response but capitalized",
    version="1.0.0",
    # Must be the address your clients will use!
    url="http://localhost:6000"    
)
class EchoServer(A2AServer):
    def handle_task(self, task):
        # 1) Extract the incoming text from the 'content' field
        content = task.message.get("content", {}) or {}
        text = content.get("text", "").upper()

        # 2) Echo it back in the artifact
        task.artifacts = [{
            "parts": [
                {"type": "text", "text": text}
            ]
        }]

        # 3) Mark the task complete
        task.status = TaskStatus(state=TaskState.COMPLETED)
        return task

if __name__ == "__main__":
    # Bind to 0.0.0.0 so any interface (including localhost) is served,
    # but advertise localhost in the Agent Card so your test client can reach it.
    run_server(EchoServer(), host="0.0.0.0", port=6000, debug=True)
