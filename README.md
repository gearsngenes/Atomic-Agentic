# Atomic-Agentic

Atomic-Agentic is a Python toolkit for building, orchestrating, and experimenting with agentic AI structures. It provides a set of modular agent classes and planning/orchestration tools, designed to be both powerful for advanced users and approachable for newcomers.

## Purpose

This repository aims to:
- Offer reusable, composable agent classes for AI workflows.
- Encourage best practices in agentic design, such as clear role separation and tool/plugin integration.
- Provide simple, programmatic interfaces for building, running, and extending agent-based systems.
- Supply practical examples and templates for real-world agentic applications.

## Repository Structure

```
Atomic-Agentic/
│
├── modules/
│   ├── Agents.py         # Core agent and planner classes
│   ├── PlanExecutors.py  # Execution logic for generated plans
│   ├── Plugins.py        # Plugin/batch-tool system for agents
│   └── Prompts.py        # Prompt templates and utilities
│
│
├── examples/
|   ├── plugins_test.py             # Generate plans with plugin methods
|   ├── agentic_story_builder.py    # Execute agentic plans for storytelling
│   ├── test_async_planner.py       # Asynchronous plan execution 
│   └── orchestrating_planners.py   # Orchestrating other planners
│
├── README.md                       # <---- You are here
└── requirements.txt                # The python requirements
```

## Key Agent Classes & Capabilities

- **Agent** (`modules/Agents.py`):  
  The base class for all agents. Encapsulates a role, prompt, and interaction logic. Can be extended for specialized behaviors.

- **PlannerAgent** (`modules/Agents.py`):  
  An advanced agent that can orchestrate other agents, manage multi-step plans, and coordinate tool/plugin usage. Useful for workflows requiring planning, delegation, or iterative refinement.

- **Plugins/Tools** (`modules/Plugins.py`):  
  Agents can be extended with plugins (`ConsolePlugin`, `MathPlugin`, etc.) to provide additional capabilities, such as basic math operations, printing,user I/O, or custom tool integration.

## Example Walkthroughs

### 1. `plugins_test.py`
**Purpose:**  
This file demonstrates the plugin system for agents. It tests how agents can be extended with additional capabilities (plugins/tools), such as printing to the console or performing custom actions.

**What it does:**
- Shows how to register and use plugins with agents.
- Validates that agents can interact with external tools, making them more flexible and extensible.
- Serves as a foundation for understanding how agents can be augmented with new behaviors.

---

### 2. `examples/agentic_story_builder.py`
**Purpose:**  
A practical, multi-agent workflow for collaborative story writing, using the agent and plugin system.

**What it does:**
- Defines three agent roles: Outliner, Writer, Reviewer.
- Uses a `PlannerAgent` to orchestrate the workflow: outline → draft → review → revise (repeat as needed).
- Demonstrates agent-to-agent communication and iterative improvement.
- Utilizes plugins (e.g., for console output).
- Saves the final story draft to a markdown file.

**How it builds on the previous example:**
- Moves from simple plugin testing to a real-world, multi-agent application.
- Shows how plugins and agents work together in a coordinated pipeline.

---

### 3. `test_async_planner.py`
**Purpose:**  
Tests asynchronous planning and execution capabilities for agents.

**What it does:**
- Demonstrates how agents and planners can operate asynchronously, allowing for concurrent or non-blocking workflows.
- Likely includes tests for async task execution, agent coordination, and result collection.

**How it builds on the previous examples:**
- Extends the agent and planner concepts to support async operations.
- Prepares the groundwork for more complex, scalable, or real-time agentic systems.

---

### 4. `orchestrating_planners.py`
**Purpose:**  
Showcases advanced orchestration logic for planners and agents.

**What it does:**
- Provides higher-level orchestration patterns, possibly including dynamic plan generation, agent delegation, and complex tool usage.
- May demonstrate how to compose multiple planners, manage dependencies, or handle more sophisticated workflows.

**How it builds on the previous examples:**
- Integrates plugins, agents, and async planning into a unified orchestration framework.
- Illustrates best practices for building robust, extensible agentic systems.

---

## How to Use

1. **Install dependencies**  
   ```
   pip install -r requirements.txt
   ```

2. **Run the example**  
   From the project root:
   ```
   python examples/agentic_story_builder.py
   ```
   Or, if you encounter import errors, try:
   ```
   python -m examples.agentic_story_builder
   ```

3. **Follow the prompts**  
   - Enter a story idea and number of review cycles.
   - The script will generate, review, and iterate on a story, saving the result in `output_markdowns/`.

## Extending & Customizing

- Add new agent roles by subclassing `Agent` or `PlannerAgent`.
- Create new plugins/tools in `modules/Plugins.py`.
- Use or modify prompt templates in `modules/Prompts.py`.
- Explore and adapt the orchestration logic in `orchestrating_planners.py` and the example scripts.

## Tests

- `plugins_test.py` and `test_async_planner.py` provide test coverage for plugins and async planning logic.