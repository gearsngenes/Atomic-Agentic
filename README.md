# Atomic-Agentic

Atomic-Agentic is a Python toolkit for building, orchestrating, and experimenting with agentic AI structures. It provides a framework to modularly build advanced design patterns with agents and perform advancedorchestration. This allows us to construct powerful architecture with agentic AI while also being approachable for newcomers.

## Mission Statement
Atomic-Agentic is built around the philosophy that **agentic AI should exist in an Object-Oriented and Design-Pattern-centric framework**. That agents can and should exist
as clear and distinct instances. If we can design them to abide: breaking down agent behaviors into modular, composable, and reusable classes that follow object-oriented programming (OOP) principles. This makes it easy to extend, customize, and orchestrate agents for complex workflows.

This repository aims to:
- Offer reusable, composable agent classes for AI workflows.
- Encourage best practices in agentic design, such as clear role separation and tool/plugin integration.
- Provide simple, programmatic interfaces for building, running, and extending agent-based systems.
- Supply practical examples and templates for real-world agentic applications.


Install dependencies:

```bash
pip install -r requirements.txt
```

Project layout:

```
Atomic-Agentic/
├── modules/
│   ├── Agents.py           # Core agent classes
│   ├── PlannerAgents.py    # Planner agents (plan + execute)
│   ├── OrchestratorAgents.py # Orchestrator agents (step-by-step decisions)
│   ├── Plugins.py          # Plugin toolsets
│   ├── LLMEngines.py       # Model backends (OpenAI, LlamaCPP, etc.)
│   └── Prompts.py          # System prompts for agent behavior
└── examples/               # Sample usage scripts
```

---

## 🧠 1. LLM Engines

LLM engines are framework-agnostic wrappers around specific large language models. Instead of hard-coding OpenAI or LlamaCPP logic into every agent, Atomic-Agentic uses a consistent interface:

```python
class LLMEngine(ABC):
    @abstractmethod
    def invoke(self, messages: list[dict]) -> str:
        ...
```

This ensures that any agent or planner in the framework can swap between OpenAI, HuggingFace, Bedrock, Gemini, or local models like Llama without changing its logic.

All engines use OpenAI-compatible message format:
```python
[
  { "role": "system", "content": "..." },
  { "role": "user", "content": "..." },
]
```

### Example: OpenAI
```python
from modules.LLMEngines import OpenAIEngine

engine = OpenAIEngine(api_key="...", model="gpt-4o")
response = engine.invoke([
  {"role": "system", "content": "You are a haiku generator."},
  {"role": "user", "content": "Write one about summer rain."}
])
print(response)
```

### Example: LlamaCpp (local models)
```python
from modules.LLMEngines import LlamaCppEngine

engine = LlamaCppEngine(model_path="./models/mistral.gguf")
response = engine.invoke([
  {"role": "system", "content": "You are a coding assistant."},
  {"role": "user", "content": "Write a bubble sort in Python."}
])
print(response)
```

This abstraction is crucial for experimentation, benchmarking, and deployment flexibility.

---

## 🧠 2. Agent

The `Agent` class is the most foundational unit in Atomic-Agentic. It represents a single prompt-and-response loop with an LLM, but is also capable of:

- Storing context (chat memory) across invocations
- Defining a `role_prompt` that governs its personality or task
- Tightly coupling with any LLM backend (via `LLMEngine`)
- Serving as a callable sub-agent in tool-based workflows

The `role_prompt` acts as the agent’s "instruction manual"—it guides every response the model generates. Setting this prompt effectively allows you to create specialists: tutors, debuggers, poets, analysts, and more.

If `context_enabled=True`, the agent remembers previous user and assistant messages across multiple calls. This makes it suitable for conversational interfaces and stateful interactions.

### Example
```python
from modules.Agents import Agent
from modules.LLMEngines import OpenAIEngine

llm = OpenAIEngine(api_key="...", model="gpt-4o")

agent = Agent(
    name="TravelBot",
    description="Helps plan vacations with recommendations and local tips.",
    llm_engine=llm,
    role_prompt="You are a friendly travel planner who gives useful suggestions for trips.",
    context_enabled=True
)

print(agent.invoke("I'm going to Japan next month. Where should I visit?"))
print(agent.invoke("And what should I eat in Osaka?"))  # Memory carries over!
```

This atomic agent is useful as a building block for more complex orchestrators and planners. You can compose or register these agents inside more complex agents, and their `.description` will be included automatically.

---

## 🧠 3. PrePostAgent

Wraps an agent with a chain of input/output processing functions.

```python
from modules.Agents import PrePostAgent
from modules.LLMEngines import OpenAIEngine

def clean_input(text: str) -> str:
    return text.replace("Q: ", "")

def emphasize_output(text: str) -> str:
    return text.upper()

engine = OpenAIEngine(api_key="...", model="gpt-4o")
agent = PrePostAgent("Translator", "French translator", engine)
agent.add_prestep(clean_input)
agent.add_poststep(emphasize_output)

print(agent.invoke("Q: How are you?"))
```

---

## 🧠 4. ChainSequenceAgent

Chains multiple agents. Each one receives the output of the previous.

```python
from modules.Agents import Agent, ChainSequenceAgent
from modules.LLMEngines import OpenAIEngine

engine = OpenAIEngine(api_key="...", model="gpt-4o")
a = Agent("Outline", "Outlines a story", engine)
b = Agent("Write", "Writes a paragraph", engine)

chain = ChainSequenceAgent("StoryChain")
chain.add(a)
chain.add(b)

print(chain.invoke("A wizard travels through time."))
```

---

## 🧠 5. PlannerAgent

Creates a multi-step plan and executes each tool step using plugins, agents, or MCP servers.

```python
from modules.PlannerAgents import PlannerAgent
from modules.Plugins import MathPlugin, ConsolePlugin
from modules.LLMEngines import OpenAIEngine

planner = PlannerAgent("MathPlanner", "Does math", OpenAIEngine("gpt-4o", api_key="..."), allow_agentic=True, allow_mcpo=True)
planner.register(MathPlugin())
planner.register(ConsolePlugin())

result = planner.invoke("Add 10 and 5, then print it.")
```

---

## 🧠 6. OrchestratorAgent

Generates and executes one step at a time. Unlike the PlannerAgent where the entire plan is created once by the agent, the OrchestratorAgent executes tasks one step at a time, allowing it **to course-correct or change its next step** based on the results of the step it generated.

```python
from modules.OrchestratorAgents import OrchestratorAgent
from modules.Plugins import MathPlugin
from modules.LLMEngines import OpenAIEngine

orch = OrchestratorAgent("MathStepper", "Step-by-step solver", OpenAIEngine("gpt-4o", api_key="..."), allow_mcpo=True)
orch.register(MathPlugin())

result = orch.invoke("Multiply 5 by 10, then if the result is even, return 'EVEN', else return 'ODD'.")
```

---

## 🧩 Registering Tools: Plugins, Agents, Functions, and MCP Servers

All `ToolAgent` subclasses (like `PlannerAgent` and `OrchestratorAgent`) support adding tools to their toolbox. Tools can be:

- ✅ Individual Python functions
- ✅ Plugin collections (e.g., `MathPlugin`)
- ✅ Other agents (if `allow_agentic=True`)
- ✅ External MCP servers (if `allow_mcpo=True`)

When creating the agent, pass these options via the constructor:

```python
from modules.PlannerAgents import PlannerAgent
from modules.LLMEngines import OpenAIEngine

llm = OpenAIEngine(api_key="...", model="gpt-4o")

# Enable planner to register both agents and MCP servers
planner = PlannerAgent(
    name="MultiToolPlanner",
    description="Plans using local tools, agents, and remote servers.",
    llm_engine=llm,
    allow_agentic=True,
    allow_mcpo=True
)
```

### 🔧 Registering tools

```python
# Register a standalone function
def shout(text: str) -> str:
    return text.upper()

planner.register(shout, "Converts text to uppercase.")

# Register a plugin
from modules.Plugins import MathPlugin
planner.register(MathPlugin())

# Register another agent
from modules.Agents import Agent
assistant = Agent(name="Summarizer",
    description="Summarizes text",
    role_prompt="Synthesize a concise, communicative, and accurate summary of any body of text you receive.",
    llm_engine=llm)
planner.register(assistant)

# Register an MCP tool server (OpenAPI-compliant)
planner.register("http://localhost:8000")
```

Once registered, these tools are available to the agent’s planning or orchestration logic automatically.

---

## 💡 Prompt Schema

All tool-using agents follow a strict format when planning:

### PlannerAgent Step Format:
Below is the example of decomposing the task: calculate 10 * (1 + 2):
```json
[
  { "function": "__dev_tools__.add", "args": { "a": 1, "b": 2 } },
  { "function": "__dev_tools__.multiply", "args": { "a": "{{step0}}", "b": 10 } },
  { "function": "__dev_tools__._return", "args": { "val": "{{step1}}" } }
]
```

### OrchestratorAgent Step Format:
The example below for a step asking for the product of 5 and 10.
```json
{
  "step_call": {
    "function": "__plugin_MathPlugin__.multiply",
    "args": { "a": 5, "b": 10 },
    "source": "__plugin_MathPlugin__"
  },
  "explanation": "Multiply to start solving.",
  "decision_point": false,
  "status": "INCOMPLETE"
}
```

---

## ✅ Summary

| Class              | Description                          | Multi-step | Uses tools | Calls agents | MCP support |
|-------------------|--------------------------------------|------------|------------|--------------|-------------|
| `Agent`           | Basic text-to-text                   | ❌         | ❌         | ❌           | ❌          |
| `PrePostAgent`    | Adds input/output processing         | ❌         | ❌         | ❌           | ❌          |
| `ChainSequence`   | Static chain of agents               | ✅         | ❌         | ✅           | ❌          |
| `PlannerAgent`    | Generates and runs full plan         | ✅         | ✅         | ✅*          | ✅**         |
| `OrchestratorAgent` | Decides & executes next step         | ✅         | ✅         | ✅*          | ✅**         |

*Requires `allow_agentic=True`\
**Requires `allow_mcpo=True`

