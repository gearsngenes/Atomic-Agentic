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
│   ├── LLMEngines.py      # Language model wrapper platform-specific subclasses
│   ├── PlanExecutors.py  # Execution logic for Planner-Agent generated plans
│   ├── Plugins.py        # Plugin/batch-tool system for agents
│   └── Prompts.py        # Prompt templates and utilities
│
├── examples/
│   ├── Agent_Examples/
│   │   ├── openai_agent_test.py    # Example: OpenAI agent usage
│   │   └── slm_agent_test.py       # Example: SLM agent usage
│   ├── Planner_Examples/
│   │   ├── 01_plugins_test.py         # Generate plans with plugin methods
│   │   ├── 02_async_planner_test.py   # Asynchronous plan execution
│   │   ├── 03_agentic_story_builder.py   # Multi-agent story workflow
│   │   └── 04_orchestrating_planners.py  # Orchestrating planners
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
   llm_engine=llm_engine,
   role_prompt = "You are a helpful fact-checking assistant. Whenever you see a mistake, start your response with 'WAIT.' before continuing. .",
   context_enabled = True # Mark as true to remember prior messages
)

# Use the agent to process a query
q1_result = my_agent.invoke("Is the Eiffel Tower in Canada?")
q2_result = my_agent.invoke("If not, then where?")
print(f"{q1_result}\n{q2_result}")
```

**3. PrePostAgent** see (`/PrePost_Examples/`)

This is the most basic of the Tool-Using agents in Atomic-Agentic. It can register specific methods and arrange them in specific orders before and after the LLM invocation.
- `.preprocessors` for adding any operations to apply to the input before the LLM
- `.postprocessors` for adding any operations to the output of the LLM.
- Provides extra flexibility and deterministic logic to filtering inputs and outputs for agents.

**4. ChainSequenceAgent** (see `ChainSequence_Examples/`)

As their name implies, the ChainSequence can chain together multiple individual agents in a sequence one large composite agent to perform more advanced tasks or reasoning. This agent only needs to be invoked once to run the entire chain. You can also 'add' or 'pop' agents from the chain to edit it.

**Example:**
Below is an abbreviated version of the ChainSequence Storybuilder example.

```python
from modules.LLMEngines import OpenAIEngine
from modules.Agents import Agent, ChainSequence

# Define two simple agents, assume 
outliner = Agent(...)
writer = Agent(...)

#Create a chain in a ChainSequence
story_chain = ChainSequenceAgent("Story-Builder-chain")
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
planner = PlannerAgent(llm_engine=llm_engine, name="Workflow Orchestrator")

# Register plugins/tools
planner.register_plugin(ConsolePlugin()) # for methods related to logging, printing, etc.
planner.register_plugin(MathPlugin()) # for basic math methods

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


## Getting Started

1. **Install dependencies**
   Once you've cloned the repository, install the python dependencies like so:
   ```
   pip install -r requirements.txt
   ```
2. **Run examples**
   Navigate to any of the available examples/ sub folders and run the example. For instance, the basic OpenAI agent example would be run like so:
   ```
   python .\examples\Agent_Examples\openai_agent_test.py
   ```