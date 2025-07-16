import os, json, re, inspect, logging
from dotenv import load_dotenv
from openai import OpenAI
from typing import get_type_hints
from typing import Any
load_dotenv()

# internal imports
from atomic_agents.Plugins import *
from atomic_agents.PlanExecutors import PlanExecutor, AsyncPlanExecutor
import atomic_agents.Prompts as Prompts

# ────────────────────────────────────────────────────────────────
# 1.  Agent  (LLM responds to prompts)
# ────────────────────────────────────────────────────────────────
class Agent:
    def __init__(self, name, role_prompt: str = Prompts.DEFAULT_PROMPT, llm=None, model: str = "gpt-4o-mini", context_enabled: bool = False):
        self.name = name
        self.llm = llm or OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.role_prompt = role_prompt
        self.context_enabled = context_enabled
        self.history = []

    def invoke(self, prompt: str) -> str:
        messages = [{"role": "system", "content": self.role_prompt}]
        if self.context_enabled:
            messages.extend(self.history)  # Include previous messages if context is enabled
        messages.append({"role": "user", "content": prompt})
        response = self.llm.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.1,
        )
        response = response.choices[0].message.content.strip()
        if self.context_enabled:
            self.history.append({"role": "user", "content": prompt})
            self.history.append({"role": "assistant", "content": response})
        return response

# ────────────────────────────────────────────────────────────────
# 2.  Planner Agent  (plans and executes)
# ────────────────────────────────────────────────────────────────
class PlannerAgent(Agent):
    """
    Generates and runs an executable plan as a tool.
    Additionally, can register other agents' invoke()
    methods as tools which it can run`.
    """

    def __init__(self, name: str, model: str = "gpt-4o-mini", is_async = True, debug = False):
        super().__init__(name = name, role_prompt = Prompts.AGENTIC_PLANNER_PROMPT, model=model)

        # registries -------------------------------------------------
        self.toolbox: dict[str, dict] = {}
        self.plugin_list: list[str] = []

        # always provide a simple "return" helper
        self.register_tool("return", lambda val: val, "return(val:any) – identity helper for the final step.")
        self.debug = debug
        
        # debug / executor
        self.executor = AsyncPlanExecutor(debug=debug) if is_async else PlanExecutor(debug=debug)

    @staticmethod
    def _build_signature(func: callable) -> str:
        sig   = inspect.signature(func)
        hints = get_type_hints(func)
        params = [
            f"{n}: {hints.get(n, Any).__name__}"
            + (f" = {p.default!r}" if p.default is not inspect._empty else "")
            for n, p in sig.parameters.items() if n != "self"
        ]
        rtype = hints.get("return", Any).__name__
        return f"({', '.join(params)}) → {rtype}"

    def register_tool(self, name: str, func: callable, description: str, plugin: str = "", agent: str ="") -> None:
        self.toolbox[name] = {
            "agent":        agent if agent else "__base_agent__",
            "plugin":       plugin if plugin else "__base_registry__",
            "callable":     func,
            "description":  f"{name}{self._build_signature(func)} — {description}",
            "is_async":     inspect.iscoroutinefunction(func),
        }

    def register_plugin(self, plugin: Plugin) -> None:
        pname = plugin.__class__.__name__
        if pname in self.plugin_list:
            raise ValueError(f"Plugin “{pname}” already registered.")
        self.plugin_list.append(pname)
        for method_name, meta in plugin.method_map().items():
            self.register_tool(
                name =          method_name,
                func =          meta["callable"],
                description =   meta["description"],
                plugin =        pname
            )

    def register_agent(self, agent: Agent, description: str | None = None,) -> None:
        """
        Expose `agent.invoke` as a tool called  '<Alias>.invoke'.

        Parameters
        ----------
        agent        : the BasicAgent (or subclass) to expose
        description  : optional extra text appended to auto-description
        """
        # Build signature string "(arg: type, …) → return"
        signature = self._build_signature(agent.invoke)

        # Compose description
        desc_lines = [
            f"{agent.name}.invoke signature: {signature} -- "
            f"Invokes the {agent.name} agent.",
            f"Agent description: {description}",
        ]

        self.register_tool(
            name =          f"{agent.name}.invoke",
            func =          agent.invoke, # bound method
            description =   "\n".join(desc_lines),
            agent=          agent.name,
        )

    def create_plan(self, prompt: str) -> dict:
        methods_block = "\n".join(
            f"-{name}: {meta['description']}" for name, meta in self.toolbox.items()
        )

        # 2. Craft user prompt ---------------------------------------
        user_prompt = (
            f"AVAILABLE METHODS\n{methods_block}\n\n"
            f"TASK\n{prompt}\n\n"
            "Respond *only* with the JSON plan."
        )

        # 3. Ask the LLM via BasicAgent ------------------------------
        raw = super().invoke(user_prompt).strip()
        raw = re.sub(r'^```[a-zA-Z]*|```$', '', raw)  # scrub fences if any

        try:
            steps = json.loads(raw)
        except Exception as e:
            raise ValueError(f"JSON parse error: {e}\nLLM output:\n{raw}")

        unknown = [s["function"] for s in steps if s["function"] not in self.toolbox]
        if unknown:
            raise ValueError(f"Plan uses unknown functions: {unknown}")
        
        plan = {"steps": steps, "tools": self.toolbox}
        
        # guarantee last step is "return"
        if not plan["steps"] or plan["steps"][-1]["function"] != "return":
            last_index = len(plan["steps"]) - 1
            plan["steps"].append(
                {
                    "function": "return",
                    "args": {"val": f"{{{{step{last_index}}}}}" if plan["steps"][-1] else None},
                }
            )
        return plan
    
    def execute_plan(self, plan: dict) -> dict:
        return self.executor.execute(plan)
    
    def invoke(self, prompt: str) -> dict:
        """
        Returns
        -------
        { "steps": list[dict], "tools": dict[str, meta] }
        """      
        plan = self.create_plan(prompt)
        if self.debug:
            plan_steps = "\n".join(f"STEP {i}:\n{json.dumps(s, indent = 2)}" for i, s in enumerate(plan["steps"]))
            logging.info(f"[PLAN] STEPS BELOW:\n\n{plan_steps}")
        plan_result = self.execute_plan(plan)
        return plan_result
    
