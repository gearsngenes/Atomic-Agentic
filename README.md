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

## Philosophy & Inspiration: Atomizing Agentic AI for OOP
Atomic-Agentic is built on five core principles:

1. **Platform-Agnostic LLMs:** Agents are designed to work seamlessly with any large language model (LLM)—whether Azure, Bedrock, Huggingface, OpenAI, or others. The underlying LLMs are abstracted so that, regardless of provider, agents interact with them in a unified, consistent way.

2. **Distinct Agent Instances:** Every agent instance has a clear and distinct set of attributes and capabilities, tied to its LLM model, class, and registered tools. This separation ensures that each agent's behavior and abilities are explicit and easily managed.

3. **Extensible Functionality:** Agent functionality isn't limited to what's defined in the base classes. Thanks to OOP, users can create subclasses for custom behaviors. For advanced agents, you can also register tools and plugins to dynamically extend their capabilities—allowing agents to interact with new environments or perform new tasks without modifying core code.

4. **Reusability:** By defining agents as OOP classes, you can reuse agent logic across different workflows and projects. This modularity makes it easy to adapt, extend, and share agentic components.

5. **Composability & Orchestration:** Advanced agent classes can build structures composed of other agents or orchestrations of agents. These composite agents can themselves be treated as singular agents, enabling scalable, hierarchical, and collaborative workflows.

## Repository Structure

```
Atomic-Agentic/
│
├── modules/
│   ├── Agents.py         # Core agent class and agent subclasses
│   ├── PlannerAgents.py  # Planner Agent and its subclasses
│   ├── LLMEngines.py     # Language model wrapper platform-specific subclasses
│   ├── PlanExecutors.py  # Execution logic for Planner-Agent generated plans
│   ├── Plugins.py        # Plugin/batch-tool system for agents
│   └── Prompts.py        # Prompt templates and utilities
│
├── examples/
│   ├── Agent_Examples/
│   │   ├── openai_agent_test.py    # Example: OpenAI agent usage
│   │   └── slm_agent_test.py       # Example: SLM agent usage
│   ├── Planner_Examples/
│   │   ├── 01_plugins_test.py          # Generate plans with plugin methods
│   │   ├── 02_async_planner_test.py    # Asynchronous plan execution
│   │   ├── 03_agentic_story_builder.py # Multi-agent story workflow
│   │   ├── 04_planner_delegator.py     # Delegating tasks to individual planners
│   │   ├── 05_mcpo_planner_test.py     # Running a test server
|   |   └── sample_mcp_server.py        # Run this in a separate terminal
│   │   
│   ├── PrePost_Examples
│   │   └── 01_coder_build_and_run.py  # Generates & executes code
│   └── ChainSequence_Examples/
│       ├── 01_unit_chain_programmer.py       # Single unit ChainSequence agent programming
│       ├── 02_cpp_to_python_chainsequence.py # C++ to Python ChainSequence agent
│       └── 03_chainsequence_story_builder.py # ChainSequence agent story builder
│
├── README.md        # <---- You are here
└── requirements.txt # Python requirements
```

## Key Classes
Now, let's run through some of the key classes that are used in Atomic-Agentic, to familiarize ourselves with it.
### Agents
Atomic-Agentic agents are designed to be modular and composable, progressing from simple wrappers to complex orchestrators:

**1. LLMEngine** (`modules/LLMEngines.py`)
The LLM Engine is the simplest unit of language models in Atomic-Agentic. They offer provider and model-agnostic interface to call LLMs. At the moment, subclasses tailoring the engines to Llama-CPP and OpenAI are available.
- **Stateless**: Does not retain any context of prior messages sent to it.
- **Polymorphic**: Provides subclasses to formatted for various platforms (OpenAI, Azure, Huggingface, Bedrock, etc.).
- **Reusable**: Can be used as a standalone llm object or as the core of more advanced agents.

**Example:**
```python
from modules.LLMEngines import OpenAIEngine

llm_engine = OpenAIEngine(
   api_key = "your-api-key",
   model = "your-model")

# Use the llm_engine directly
response = llm_engine.invoke("What is the capital of France?")
print(response)
```

**2. Agent** (`modules/Agents.py`)
The base Agent class, representing an atomic unit of reasoning and action.
- Encapsulates a role, prompt, and interaction logic.
- Each instance is tied to a specific LLMEngine instance and can be extended with custom attributes or behaviors.

**Example:**
```python
from modules.LLMEngines import OpenAIEngine
from modules.Agents import Agent

llm_engine = OpenAIEngine(api_key="your-api-key", model="your-model")

my_agent = Agent(
   name = "Fact_Checker"
   description="...",
   llm_engine=llm_engine,
   role_prompt = "You are a helpful fact-checking assistant. Whenever you see a mistake, start your response with 'WAIT.' before continuing. .",
   context_enabled = True # Mark as true to remember prior messages
)

# Use the agent to process a query
q1_result = my_agent.invoke("Is the Eiffel Tower in Canada?")
q2_result = my_agent.invoke("If not, then where?")
print(f"{q1_result}\n{q2_result}")
```

**3.🧼 PrePostAgent: Input & Output Processing Agent**
The PrePostAgent is a specialized subclass of the base Agent, designed to allow custom preprocessing and postprocessing logic to be injected before and after the LLM's response. This gives users a flexible and deterministic way to manipulate input and output data surrounding the core LLM invocation.

Preprocessors: Functions that transform the raw user prompt before it’s sent to the LLM.

Postprocessors: Functions that transform the LLM response before it’s returned.

Composable Steps: Both processors are defined as ordered lists of Python functions and can be updated incrementally or all at once.

This is especially useful for tasks such as:

- Auto-formatting user input (e.g., strip whitespace, enforce structure)
- Parsing or validating output (e.g., extracting JSON, evaluating code)
- Adding dynamic control logic around a standard agent

Key Properties:
- .preprocessors — Returns a list of pre-processing functions
- .postprocessors — Returns a list of post-processing functions
- .add_prestep(func) — Adds a new preprocessor function
- .add_poststep(func) — Adds a new postprocessor function

**Example:**
Here, we have an example of a pre-post agent being used to remove a specific prefix from any input the agent receives before answering. This allows us to to have more control and cleaner inputs for our Agent.
```python
from modules.LLMEngines import OpenAIEngine
from modules.Agents import PrePostAgent

# Preprocessing tool — removes 'Input: ' prefix if present
def strip_translate_prefix(text: str) -> str:
    return text.replace("Input: ", "").strip()

# Postprocessing tool — converts all output to uppercase
def shoutify(text: str) -> str:
    return text.upper()

# LLM Engine
engine = OpenAIEngine(api_key="...", model="gpt-4o")

# Define PrePost agent
translator = PrePostAgent(
    name="UppercaseTranslator",
    description="...",
    llm_engine=engine,
    role_prompt="You are a translator that translates English text to French."
)

# Add preprocessing and postprocessing functions
translator.add_prestep(strip_translate_prefix)
translator.add_poststep(shoutify)

# Run the agent
response = translator.invoke("Input: Hello, how are you?")
print(response)
# Output (uppercase French translation): BONJOUR, COMMENT ÇA VA ?
```
This pattern is powerful for tasks where deterministic transformation or formatting of inputs/outputs is needed around otherwise generative agents. You can build a stack of custom behaviors without modifying the agent’s core logic.

Let me know if you'd like this formatted into a pull-request-style snippet or if you want similar entries added for other classes.

**4. ChainSequenceAgent** (see `ChainSequence_Examples/`)

As their name implies, the ChainSequence can chain together multiple individual agents in a sequence one large composite agent to perform more advanced tasks or reasoning. This agent only needs to be invoked once to run the entire chain. You can also 'add' or 'pop' agents from the chain to edit it.

**Example:**
Below is an abbreviated version of the ChainSequence Storybuilder example.

```python
from modules.LLMEngines import OpenAIEngine
from modules.Agents import Agent, ChainSequence

# Define two simple agents, assume the rest of their features are already implemented
outliner = Agent(name = "story-outliner", description="...", ...)
writer = Agent(name = "story-writer", description="...", ...)

#Create a chain in a ChainSequence
story_chain = ChainSequenceAgent("Story-Builder-chain") # The description uses that of the agents you add
story_chain.add(outliner)
story_chain.add(writer)

# Use the ChainSequence agent to process a story idea
final_story = story_chain.invoke("A detective solves a mystery in Paris.")
print(final_story)
```

**4. PlannerAgent** (`modules/Agents.py`)

The most advanced agent capable of orchestrating tools and other agents in multi-step plans. However, unlike ChainSequences that create chains of a fixed order of execution, the PlannerAgent creates a chain of execution *at runtime*.
- Registers tools/plugins with textual description as context
- Like the ChainSequenceAgent, it can orchestrate tasks involving other agents (including other planners), but it is non-deterministic, and determined at runtime whether the registered agent is used.
- Feeds the descriptions and user prompts to the llm to generate an executable workflow of function calls and their arguments.
- Supports running the generated plans synchronously and asynchronously, for scalable, concurrent tasks.

**Example:**
```python
from modules.LLMEngines import OpenAIEngine
from modules.Agents import PlannerAgent
from modules.Plugins import ConsolePlugin, MathPlugin

# Define a planner agent, assume pre-defined LLM_engine
planner = PlannerAgent(name="Workflow Planner", description="...", llm_engine=llm_engine)

# Register plugins/tools
planner.register(ConsolePlugin()) # for methods related to logging, printing, etc.
planner.register(MathPlugin()) # for basic math methods

# Use the planner agent to generate and execute a plan
planner_result = planner.invoke("Calculate the product of 42 and 58, then print the result.")
# planner-result might be 'None' depending on exactly what the plan planner generates
print(planner_result)
```

### Plugins & Tools
Plugins extend PlannerAgent capabilities, allowing agents to interact with their environment or perform specialized tasks. The plugin system is designed for easy registration and dynamic extension:

**Plugin System** (`modules/Plugins.py`)
- Agents can register plugins at runtime, adding new methods or tools without modifying the agent's core code.
- Plugins are analogous to electrons: they orbit agents and enable interactions (I/O, math, external APIs, etc.).

**Example Plugins:**
- `ConsolePlugin`: Enables agents to print to the console or interact with user input/output.
- `MathPlugin`: Adds basic math operations and calculation capabilities.
- `PythonPlugin`: Adds additional python execution functionalities, like the exec() method.
- `ParserPlugin`: Adds methods for extracting and parsing text into python objects
- Custom plugins: Users can define their own plugins for domain-specific tasks (e.g., file I/O, web requests, data processing).


Plugins can be registered to any agent, including PlannerAgents and ChainSequences, making the system highly extensible and adaptable to new environments or requirements.

## 🧠 Agent Descriptions Now Handled Automatically

A recent update simplifies how you register agents inside planner or orchestrator agents. Previously, when you registered an agent to another, you needed to manually provide a description:

```python
agentic.register(helper_agent, description="Summarizes a given passage.")
```

Now, **you don’t need to provide a description at all**. Agents expose a `.description` property that generates a Markdown-formatted string based on their type and declared description.

This means:

- `AgenticPlannerAgent`, `McpoPlannerAgent`, and `AgenticOrchestratorAgent` will automatically embed a subagent’s description when registering.
- You no longer have to repeat information or worry about formatting descriptions manually.
- This is especially useful when you're orchestrating many agents or dynamically building agent registries.

**New Example:**

```python
from modules.PlannerAgents import AgenticPlannerAgent
from modules.Agents import Agent
from modules.LLMEngines import OpenAIEngine

llm = OpenAIEngine(api_key="...", model="gpt-4o")

meta = AgenticPlannerAgent("MetaPlanner", description="Handles subtasks by delegating to helpers.", llm_engine=llm)

# Define a helper agent
summarizer = Agent(description="...", 
    name="Summarizer",
    description="Summarizes English passages into concise bullets.",
    llm_engine=llm,
    role_prompt="Summarize input into bullet points."
)

# No need to pass `description=` explicitly anymore!
meta.register(summarizer)

result = meta.invoke("Summarize: The Eiffel Tower is in Paris and was built in 1889.")
```

Under the hood, this will register a function like:
```
__agent_Summarizer__.invoke(text: str) → str — Agent description: ~~Agent Summarizer~~
A generic Agent for on Text-Text responses. Description: Summarizes English passages into concise bullets.
```

This makes your orchestration logic cleaner, avoids redundancy, and ensures descriptions stay consistent with the agents themselves.

## Planner Agents: Tool-Oriented Reasoners

Planner Agents are one of the most powerful constructs in Atomic-Agentic. Unlike reactive agents (like `Agent` or `ChainSequenceAgent`), planners **autonomously generate execution plans**—dynamic chains of function calls based on the task prompt and available tools.

### 🧠 `PlannerAgent`: The Runtime Plan Generator

The `PlannerAgent` generates and executes structured multi-step plans using developer-defined tools and plugin methods. It is the first agent capable of autonomous orchestration based on a user’s natural language instruction.

- **Tool Registration**: Supports registering functions or `Plugin` collections.
- **Plan Execution**: Steps are executed one-by-one (synchronously or asynchronously).
- **No agent-to-agent orchestration yet**—just local tool use.

**Example:**
```python
from modules.PlannerAgents import PlannerAgent
from modules.LLMEngines import OpenAIEngine
from modules.Plugins import MathPlugin, ConsolePlugin

llm = OpenAIEngine(api_key="...", model="gpt-4o")
planner = PlannerAgent(name = "SimplePlanner", description="...",  llm_engine=llm)

planner.register(MathPlugin())
planner.register(ConsolePlugin())

result = planner.invoke("Multiply 8 by 6, then print the result.")
```

---

### 🔁 `AgenticPlannerAgent`: Adds Multi-Agent Planning

This subclass expands `PlannerAgent` by allowing it to **invoke other registered agents**—essentially treating each agent as a callable tool (`<agent_name>.invoke`). This enables planners to **delegate subtasks** to other agents dynamically.

- **Agent Registration**: Register other agents and invoke them via plan steps.
- **Granular Mode**: Optional flag (`granular=True`) allows this planner to also register plugins/methods like a basic `PlannerAgent`.

**Example:**
```python
from modules.PlannerAgents import AgenticPlannerAgent
from modules.Agents import Agent
from modules.LLMEngines import OpenAIEngine

llm = OpenAIEngine(api_key="...", model="gpt-4o")
agentic = AgenticPlannerAgent("MetaPlanner", description="...", llm_engine=llm, granular=True)

helper = Agent("SimpleHelper", description="...", llm_engine=llm, role_prompt="Summarize input text.")
agentic.register(helper, description="Summarizes a given passage.")

# This will delegate the summarization to the registered helper agent
agentic.invoke("Summarize: 'The Eiffel Tower is located in Paris. It was built in 1889.'")
```

---

### 🌐 `McpoPlannerAgent`: Calls External MCP Servers

With MCP becoming an integral part of Agentic AI communication and tool-use, I wanted to make sure that Atomic-Agentic could be compatible with MCP or its extensions. Therfore, introducing the MCP-O Planner Agent. This is the most powerful subclass. In addition to tools and agents, the `McpoPlannerAgent` can **call external tools hosted on any OpenAPI-compliant MCP-O server**. These external tools appear as remote function HTTP-compatible endpoints (`/add`, `/classify`, etc.), but connect to MCP-running servers. This allows us to retrieve tool-context remotely, and could potentially remove the burden of providing context on developers.

- **MCP-O Tool Registration**: Use `register("http://localhost:8000")` to pull in tool paths from a server's OpenAPI schema.
- **MCP Server Invocation**: The plan can include steps that send POST requests to any known MCP-O server.

**Example:**

First and foremost, make sure you have a link to a working MCP server hosted on MCPO. In our `Planner_Examples/` folder, there is a `sample_mcp_server.py` file that you can run. Make sure to start the server before running any agents that use this server, using the command seen here:

`mcpo --port 8000 -- python sample_server.py`


As a loose example, here is how we'd use an instance of McpoPlannerAgent.

```python
from modules.PlannerAgents import McpoPlannerAgent
from modules.LLMEngines import OpenAIEngine

llm = OpenAIEngine(api_key="...", model="gpt-4o")
mcpo = McpoPlannerAgent("CrossServerPlanner", description="...", llm_engine=llm)

# Register an MCP server
mcpo.register("http://localhost:8000") # our sample server

# The following user prompt will invoke one of the external tools (like /add)
result = mcpo.invoke("Use the MCP server to calculate the derivative of x**3 at x = 1")
```

---

### Summary Table

| Class Name            | Key Features                                                       | Tools Allowed           | Can Call Agents | Can Call MCP Servers |
|----------------------|---------------------------------------------------------------------|--------------------------|------------------|-----------------------|
| `PlannerAgent`        | Basic multi-step plans using registered methods/plugins            | ✅ Plugins + functions   | ❌               | ❌                    |
| `AgenticPlannerAgent` | Adds agent-to-agent tool usage via `<agent_name>.invoke` syntax    | ✅ (if `granular=True`)  | ✅               | ❌                    |
| `McpoPlannerAgent`    | Adds support for external MCP-O tool servers                        | ✅ + Agents + MCP tools  | ✅               | ✅                    |
