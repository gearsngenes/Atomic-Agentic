from modules.Agents import Agent, ToolAgent
from python_a2a import A2AClient, A2AServer, agent, TaskStatus, TaskState, run_server
from typing import Any

class A2AProxyAgent(Agent):
    def __init__(self, a2a_host: str):
        self._client = A2AClient(a2a_host)
        agent_card = self._client.get_agent_card()
        self._name, self._description = agent_card.name, agent_card.description
        self._context_enabled = False
        self._llm_engine = None
        self._history = []
    def attach(self, file_path: str):
        raise NotImplementedError("A2AProxyAgent does not support file attachments.")
    def detach(self, file_path: str):
        raise NotImplementedError("A2AProxyAgent does not support file attachments.")
    def invoke(self, prompt:str):
        response = self._client.ask(prompt)
        return response
    @property
    def description(self):
        return self._description
    @property
    def name(self):
        return self._name

class A2AServerAgent(Agent):
    def __init__(self, seed:Any, host: str = "localhost", port: int=5000):
        if isinstance(seed, Agent):
            self._seed = seed
            self._context_enabled = seed.context_enabled
            self._llm_engine = seed.llm_engine
            self._name = seed.name
            self._role_prompt = seed.role_prompt
            self._description = seed.description
            self._history = seed.history
            self._port = port
            self._host = host
        else:
            raise ValueError("A2AHostAgent must be initialized with an Agent instance as seed.")
        outer = self
        @agent(
            name=seed.name,
            description=seed.description,
            version="1.0.0",
            url=f"http://{self._host}:{self._port}"
        )
        class A2AServerWrapper(A2AServer):
            def handle_task(self, task):
                prompt = task.message.get("content", {}).get("text", "")
                response_text = outer.invoke(prompt)
                task.artifacts = [{"parts": [{"type": "text", "text": response_text}]}]
                task.status = TaskStatus(state=TaskState.COMPLETED)
                return task
        self._server = A2AServerWrapper(port=port)
    def attach(self, file_path: str):
        self._seed.attach(file_path)
    def detach(self, file_path: str):
        self._seed.detach(file_path)
    def register(self, tool: Any):
        if not isinstance(tool, ToolAgent):
            raise ValueError("Only ToolAgent instances can register methods.")
        self._seed.register(tool)
    def invoke(self, prompt:str):
        response = self._seed.invoke(prompt)
        if self._context_enabled:
            self._history.append({"role": "user", "content": prompt})
            self._history.append({"role": "assistant", "content": response})
        return response
    def run(self, debug: bool = True):
        run_server(self._server, host=self._host, port=self._port, debug=debug)