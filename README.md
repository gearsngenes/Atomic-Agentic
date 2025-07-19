# Atomic-Agentic

Atomic-Agentic is a Python toolkit for building, orchestrating, and experimenting with agentic AI structures. It provides a set of modular agent classes and planning/orchestration tools, designed to be both powerful for advanced users and approachable for newcomers.

## Mission Statement
Atomic-Agentic is built around the philosophy that **agentic AI should exist in an Object-Oriented framework**. That agents can and should exist
as clear and distinct instances. If we can design them to abide: breaking down agent behaviors into modular, composable, and reusable classes that follow object-oriented programming (OOP) principles. This makes it easy to extend, customize, and orchestrate agents for complex workflows.

This repository aims to:
- Offer reusable, composable agent classes for AI workflows.
- Encourage best practices in agentic design, such as clear role separation and tool/plugin integration.
- Provide simple, programmatic interfaces for building, running, and extending agent-based systems.
- Supply practical examples and templates for real-world agentic applications.

## Philosophy & Inspiration: Atomizing Agentic AI for OOP
A little context for what Atomic-Agentic encourages.
### Core Design Philosophy
Atomic-Agentic is built on five core principles:

1. **Platform-Agnostic LLMs:** Agents are designed to work seamlessly with any large language model (LLM)—whether Azure, Bedrock, Huggingface, OpenAI, or others. The underlying LLMs are abstracted so that, regardless of provider, agents interact with them in a unified, consistent way.

2. **Distinct Agent Instances:** Every agent instance has a clear and distinct set of attributes and capabilities, tied to its LLM model, class, and registered tools. This separation ensures that each agent's behavior and abilities are explicit and easily managed.

3. **Extensible Functionality:** Agent functionality isn't limited to what's defined in the base classes. Thanks to OOP, users can create subclasses for custom behaviors. For advanced agents, you can also register tools and plugins to dynamically extend their capabilities—allowing agents to interact with new environments or perform new tasks without modifying core code.

4. **Reusability:** By defining agents as OOP classes, you can reuse agent logic across different workflows and projects. This modularity makes it easy to adapt, extend, and share agentic components.

5. **Composability & Orchestration:** Advanced agent classes can build structures composed of other agents or orchestrations of agents. These composite agents can themselves be treated as singular agents, enabling scalable, hierarchical, and collaborative workflows.

### Inspiration from Nature: Agents as Atoms & Molecules
The inspiration for Atomic-Agentic comes from chemistry:
- **LLMs = Atomic Nuclei:** The LLM is the nucleus at the center of each agent, holding it together and providing its core intelligence.
- **Agents = Atoms & Molecules:** Each agent is like an atom, with its own identity and properties. When agents interact or bond, they form molecules—composite agents or orchestrations with emergent behaviors.
- **Tools/Plugins = Electrons:** The tools and plugins registered to agents are like electrons orbiting atoms and molecules. They enable agents to interact with their environment and with each other, just as electrons drive chemical reactions and bonding.

This chemistry-inspired, OOP-friendly design makes it easy to build, extend, and orchestrate agentic AI systems that are modular, reusable, and platform-agnostic.

## Repository Structure

```
Atomic-Agentic/
│
├── modules/
│   ├── Agents.py         # Core agent class and agent subclasses
│   ├── LLMNuclei.py      # Language model wrapper platform-specific subclasses
│   ├── PlanExecutors.py  # Execution logic for Planner-Agent generated plans
│   ├── Plugins.py        # Plugin/batch-tool system for agents
│   └── Prompts.py        # Prompt templates and utilities
│
├── examples/
│   ├── Agent_Examples/
│   │   ├── openai_agent_test.py    # Example: OpenAI agent usage
│   │   └── slm_agent_test.py       # Example: SLM agent usage
│   ├── Planner_Examples/
│   │   ├── 01_plugins_test.py      # Generate plans with plugin methods
│   │   ├── 02_async_planner_test.py  # Asynchronous plan execution
│   │   ├── 03_agentic_story_builder.py   # Multi-agent story workflow
│   │   └── 04_orchestrating_planners.py  # Orchestrating planners
│   ├── Polymer_Examples/
│   │   ├── 01_monomer_programmer.py    # Monomer agent programming
│   │   ├── 02_cpp_to_python_polymer.py # C++ to Python polymer agent
│   │   └── 03_polymer_story_builder.py # Polymer agent story builder
│   └── output_markdowns/
│       ├── planner_story.md
│       └── polymer_story.md
│
├── README.md                       # <---- You are here
└── requirements.txt                # Python requirements
```

## Key Classes
Now, let's run through some of the key classes that are used in Atomic-Agentic, to familiarize ourselves with it.
### Agents
Atomic-Agentic agents are designed to be modular and composable, progressing from simple wrappers to complex orchestrators:

**1. LLMNucleus** (`modules/LLMNuclei.py`)
The LLM nucleus is analogous to the nucleus of an atom: the simplest unit of language models in Atomic-Agentic. At the moment, subclasses for Llama-CPP and OpenAI are available.
- Does not retain any context of prior messages sent to it.
- Provides subclasses to initialize LLMs from various platforms (OpenAI, Azure, Huggingface, Bedrock, etc.), while providing a unified interface for prompt completion and model interaction.
- Can be used as a standalone llm object or as the core of more advanced agents.

**Example:**
```python
from modules.LLMNuclei import OpenAINucleus

nucleus = OpenAINucleus(
   api_key = "your-api-key",
   model = "your-model")

# Use the nucleus directly
response = nucleus.invoke("What is the capital of France?")
print(response)
```

**2. Agent** (`modules/Agents.py`)
The base Agent class, representing an atomic unit of reasoning and action.
- Encapsulates a role, prompt, and interaction logic.
- Each instance is tied to a specific LLMNucleus instance and can be extended with custom attributes or behaviors.

**Example:**
```python
from modules.LLMNuclei import OpenAINucleus
from modules.Agents import Agent

nucleus = OpenAINucleus(api_key="your-api-key", model="your-model")

my_agent = Agent(
   name = "Fact_Checker"
   nucleus=nucleus,
   role_prompt = "You are a helpful fact-checking assistant.",
   context_enabled = True # Mark as true to remember prior messages
)

# Use the agent to process a query
q1_result = my_agent.invoke("Is the Eiffel Tower in Canada?")
q2_result = my_agent.invoke("If not, then where?")
print(f"{q1_result}\n{q2_result}")
```

**3. PolymerAgent** (see `Polymer_Examples/`)

As their chemistry-inspired namesake implies, the PolymerAgent can chain together multiple individual agents into one large composite agent to perform more advanced tasks or reasoning. In terms of design patterns, the PolymerAgent acts as a doubly linked list.
- Enables multi-step reasoning, transformation, and collaborative workflows.
- Allows users to register specific methods to clean up or preprocess PolymerAgent outputs before feeding them to the next agent in the chain.
- Each PolymerAgent must register an agent to dictate its specific logic. That internal 'seed' agent can be a base Agent, or in itself be another polymer agent chain, or even a planner agent.

**Example:**
```python
from modules.LLMNuclei import OpenAINucleus
from modules.Agents import Agent, PolymerAgent

nucleus = OpenAINucleus(api_key="your-api-key", model="your-model")

# Define two simple agents
outliner = Agent(nucleus=nucleus, name="Outliner", role_prompt="You are a story outliner that creates full writing outlines from initial story ideas.")
writer = Agent(nucleus=nucleus, name="Writer", role_prompt="You turn story outlines into full, markdown formatted stories.")

#Create a chain in a PolymerAgent
story_chain = PolymerAgent(outliner)
story_chain.talks_to(PolymerAgent(writer))

# Use the polymer agent to process a story idea
final_story = story_chain.invoke("A detective solves a mystery in Paris.")
print(final_story)
```

**4. PlannerAgent** (`modules/Agents.py`)

The most advanced agent capable of orchestrating tools and other agents in multi-step plans. However, unlike PolymerAgents that create chains of a fixed order of execution, the PlannerAgent creates a chain of execution *at runtime*.
- Registers tools/plugins with textual description as context
- Like the Polymer Agent, it can orchestrate tasks involving other agents (including other planners), but it is non-deterministic, and determined at runtime whether the registered agent is used.
- Feeds the descriptions and user prompts to the llm to generate an executable workflow of function calls and their arguments.
- Supports running the generated plans synchronously and asynchronously, for scalable, concurrent tasks.

**Example:**
```python
from modules.LLMNuclei import OpenAINucleus
from modules.Agents import PlannerAgent
from modules.Plugins import ConsolePlugin, MathPlugin

nucleus = OpenAINucleus(api_key="your-api-key", model="gpt-3.5-turbo")

# Define a planner agent
planner = PlannerAgent(nucleus=nucleus, name="Workflow Orchestrator")

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


Plugins can be registered to any agent, including PlannerAgents and PolymerAgents, making the system highly extensible and adaptable to new environments or requirements.


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