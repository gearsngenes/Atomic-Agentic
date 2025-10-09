"""
A2Agents
========

A2A-enabled agent adapters that allow:
1) **Proxying** requests to a remote agent (A2AProxyAgent).
2) **Serving** a local seed Agent over the A2A protocol (A2AServerAgent).

Behavioral Notes
----------------
- This module refines documentation **only**. There are **no functional changes**.
- Both classes follow the same high-level Agent contract used elsewhere in the codebase:
  `attach()`, `detach()`, and `invoke()` shape, plus `name`/`description` accessors.
"""

from modules.Agents import Agent
from modules.ToolAgents import ToolAgent
from python_a2a import A2AClient, A2AServer, agent, TaskStatus, TaskState, run_server
from typing import Any


class A2AProxyAgent(Agent):
    """
    Read-only proxy to a remote A2A agent.

    Construction
    ------------
    A2AProxyAgent(a2a_host: str)
        - Connects to the remote A2A endpoint and fetches its "agent card".
        - Populates `.name` and `.description` from the remote metadata.
        - Disables local conversation context and LLM engine (this class forwards calls).

    Capabilities
    ------------
    - `invoke(prompt)` forwards the prompt to the remote agent via `A2AClient.ask(...)`.
    - File attachment methods are **not supported** for a proxy and raise errors.
    - Properties `name` and `description` are derived from the remote agent card.
    """

    def __init__(self, a2a_host: str):
        self._client = A2AClient(a2a_host)
        agent_card = self._client.get_agent_card()
        self._name, self._description = agent_card.name, agent_card.description
        self._context_enabled = False
        self._llm_engine = None
        self._history = []

    def attach(self, file_path: str):
        """A proxy cannot manage local file attachments."""
        raise NotImplementedError("A2AProxyAgent does not support file attachments.")

    def detach(self, file_path: str):
        """A proxy cannot manage local file attachments."""
        raise NotImplementedError("A2AProxyAgent does not support file attachments.")

    def invoke(self, prompt: str):
        """
        Forward the prompt to the remote A2A agent and return its string response.

        Parameters
        ----------
        prompt : str
            The user input to send to the remote agent.

        Returns
        -------
        str
            The remote agent's response.
        """
        response = self._client.ask(prompt)
        return response

    @property
    def description(self):
        """Remote agent description as reported by the A2A agent card."""
        return self._description

    @property
    def name(self):
        """Remote agent name as reported by the A2A agent card."""
        return self._name


class A2AServerAgent(Agent):
    """
    Host an existing local `Agent` instance as an A2A server.

    Construction
    ------------
    A2AServerAgent(seed: Any, host: str = "localhost", port: int = 5000)
        - `seed` must be an instance of `Agent`. Its state (context flag, LLM engine,
          name, role prompt, description, and history) is mirrored into this server agent.
        - Defines an A2A server wrapper class with the same name/description/version
          and binds `handle_task` to route incoming A2A tasks to `outer.invoke(...)`.

    Capabilities
    ------------
    - `run(debug=True)` starts the A2A HTTP server on the configured host/port.
    - `invoke(prompt)` calls the underlying seed agent's `invoke(...)` and mirrors
      history when the seed had context enabled.
    - `attach`/`detach` delegate to the seed agent.
    - `register(tool)` ensures that only `ToolAgent` instances can be registered
      against the seed (mirrors prior behavior/guardrails).
    """

    def __init__(self, seed: Any, host: str = "localhost", port: int = 5000):
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

        # Bind the outer agent for the wrapper's task handler closure.
        outer = self

        @agent(
            name=seed.name,
            description=seed.description,
            version="1.0.0",
            url=f"http://{self._host}:{self._port}"
        )
        class A2AServerWrapper(A2AServer):
            """
            Concrete A2A server wrapper exposing `seed` over HTTP.

            `handle_task` extracts the prompt text from the incoming task payload,
            calls the outer agent's `invoke`, attaches the result as text content,
            and marks the task state as COMPLETED.
            """
            def handle_task(self, task):
                prompt = task.message.get("content", {}).get("text", "")
                response_text = outer.invoke(prompt)
                task.artifacts = [{"parts": [{"type": "text", "text": response_text}]}]
                task.status = TaskStatus(state=TaskState.COMPLETED)
                return task

        self._server = A2AServerWrapper(port=port)

    # --- Delegations to the seed agent ---------------------------------------

    def attach(self, file_path: str):
        """Delegate file attachment to the underlying seed agent."""
        self._seed.attach(file_path)

    def detach(self, file_path: str):
        """Delegate file detachment to the underlying seed agent."""
        self._seed.detach(file_path)

    def register(self, tool: Any):
        """
        Register a tool-bearing agent with the seed.

        Raises
        ------
        ValueError
            If `tool` is not an instance of `ToolAgent`.
        """
        if isinstance(self._seed, ToolAgent):
            self._seed.register(tool)
        else:
            raise ValueError("Only ToolAgent instances can register methods.")

    def invoke(self, prompt: str):
        """
        Invoke the underlying seed agent and (optionally) mirror history.

        Parameters
        ----------
        prompt : str
            The input text forwarded to the seed agent.

        Returns
        -------
        str
            The seed agent's response.
        """
        response = self._seed.invoke(prompt)
        if self._context_enabled:
            self._history.append({"role": "user", "content": prompt})
            self._history.append({"role": "assistant", "content": response})
        return response

    def run(self, debug: bool = True):
        """
        Start serving the wrapped agent over A2A HTTP.

        Parameters
        ----------
        debug : bool
            Whether to launch the server in debug mode.
        """
        run_server(self._server, host=self._host, port=self._port, debug=debug)
