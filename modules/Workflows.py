from __future__ import annotations
from modules.Agents import Agent
import modules.Prompts as Prompts
from modules.LLMEngines import *
from abc import ABC, abstractmethod
import asyncio
import logging
import json, re
from modules.Tools import Tool
import inspect
from typing import Callable, Optional


import inspect, json, logging
from typing import Any, Tuple, Dict

# A call-spec is a (args, kwargs) tuple: (list|tuple, dict)
def _is_callspec(obj: Any) -> bool:
    return (
        isinstance(obj, tuple) and
        len(obj) == 2 and
        isinstance(obj[0], (list, tuple)) and
        isinstance(obj[1], dict)
    )

def _normalize_to_single_payload(*args: Any, **kwargs: Any) -> Any:
    """
    If kwargs or multiple args -> return (list(args), kwargs) as a call-spec.
    If exactly one positional arg and no kwargs -> return that value.
    If nothing -> empty string (or raise, your choice).
    """
    if kwargs or len(args) > 1:
        return (list(args), dict(kwargs))
    if len(args) == 1:
        return args[0]
    return ""  # or: raise TypeError("invoke requires at least one argument")

def _pretty_prompt(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)


class Workflow(ABC):
    def __init__(self, name: str, description: str):
        self._name = name
        self._description = description

    @property
    def name(self) -> str: return self._name
    
    @property
    def description(self) -> str: return self._description

    @abstractmethod
    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        pass

    @abstractmethod
    def clear_memory(self) -> None:
        pass

class SingleAgent(Workflow):
    def __init__(self, agent: Agent):
        super().__init__(agent.name, agent.description)
        self.agent = agent
    def clear_memory(self):
        if self.agent and self.agent.context_enabled:
            self.agent.clear_memory()
    def invoke(self, *args, **kwargs):
        logging.info(f"\n+---{'-'*len(self._name + ' Starting')}---+"
                     f"\n|   {self._name} Starting   |"
                     f"\n+---{'-'*len(self._name + ' Starting')}---+")
        if not self.agent:
            raise ValueError("No agent registered in this workflow.")
        payload = _normalize_to_single_payload(*args, **kwargs)
        text = _pretty_prompt(payload)
        result = self.agent.invoke(str(text))
        logging.info(   f"\n+---{'-'*len(self._name + ' Finished')}---+"
                        f"\n|   {self._name} Finished   |"
                        f"\n+---{'-'*len(self._name + ' Finished')}---+\n")
        return result

class ToolFlow(Workflow):
    def __init__(self, tool: Tool):
        super().__init__(tool.name, tool.description)
        self.tool = tool
        self._sig = inspect.signature(tool.func)

        params = list(self._sig.parameters.values())
        self._has_var_pos = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params)
        bindable = [p for p in params if p.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY
        )]
        self._is_single_param = (len(bindable) == 1 and not self._has_var_pos)
        self._single_param = bindable[0] if self._is_single_param else None

    def clear_memory(self):
        self.tool.clear_memory()

    def _bind_single(self, value: Any) -> inspect.BoundArguments:
        p = self._single_param
        assert p is not None
        if p.kind == inspect.Parameter.KEYWORD_ONLY:
            return self._sig.bind(**{p.name: value})
        return self._sig.bind(value)

    def _bind_from_payload(self, payload: Any) -> inspect.BoundArguments:
        # Single-param: treat any payload (scalar/dict/list/tuple) as the single argument
        if self._is_single_param:
            return self._bind_single(payload)

        # Multi-param:
        if _is_callspec(payload):
            args, kwargs = payload
            return self._sig.bind(*list(args), **kwargs)
        if isinstance(payload, dict):
            return self._sig.bind(**payload)          # kwargs
        if isinstance(payload, (list, tuple)):
            return self._sig.bind(*list(payload))     # positional
        # scalar fallback: use 'prompt' if available, else positional single
        if "prompt" in self._sig.parameters:
            return self._sig.bind(prompt=payload)
        return self._sig.bind(payload)

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        payload = _normalize_to_single_payload(*args, **kwargs)
        ba = self._bind_from_payload(payload)
        ba.apply_defaults()
        fn = self.tool.func
        return fn(*ba.args, **ba.kwargs)

class ChainOfThought(Workflow):
    def __init__(self, name: str, description: str, steps: list[Agent|Workflow|Tool] = []):
        super().__init__(name, description)
        self._steps: list[Workflow] = []
        for s in (steps or []):
            if isinstance(s, Workflow):
                self._steps.append(s)
            elif isinstance(s, Agent):
                self._steps.append(SingleAgent(s))
            elif isinstance(s, Tool):
                self._steps.append(ToolFlow(s))
            else:
                raise TypeError(f"Unsupported step type: {type(s)}")

    def insert_step(self, step: Agent|Workflow|Tool, position: int | None = None):
        if isinstance(step, Agent): step = SingleAgent(step)
        if isinstance(step, Tool):  step = ToolFlow(step)
        if position is None: self._steps.append(step)
        else: self._steps.insert(position, step)

    def pop(self, position: int = -1) -> Workflow:
        if not self._steps: raise ValueError("No steps to remove.")
        return self._steps.pop(position)

    def clear_memory(self):
        for step in self._steps: step.clear_memory()

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        logging.info(f"\n+---{'-'*len(self._name + ' Starting')}---+"
                     f"\n|   {self._name} Starting   |"
                     f"\n+---{'-'*len(self._name + ' Starting')}---+")
        if not self._steps:
            raise ValueError("No steps registered in this workflow.")

        current = _normalize_to_single_payload(*args, **kwargs)

        for step in self._steps:
            logging.info(f"[WORKFLOW] Invoking: {step.name}")
            current = step.invoke(current)

        logging.info(f"\n+---{'-'*len(self._name + ' Finished')}---+"
                     f"\n|   {self._name} Finished   |"
                     f"\n+---{'-'*len(self._name + ' Finished')}---+\n")
        return current

class MakerChecker(Workflow):
    """
    Maker-Checker pattern (agents only), *args/**kwargs compatible.
    """
    def __init__(self, name: str, description: str,
                 maker_instructions: str, checker_criteria: str,
                 maker_llm: LLMEngine, checker_llm: LLMEngine,
                 max_revisions: int = 1):
        super().__init__(name, description)
        if max_revisions < 0:
            raise ValueError("max_revisions must be >= 0")
        self.max_revisions = max_revisions

        # Build role (SYSTEM) prompts via templates
        maker_system = Prompts.MAKER_SYSTEM_TEMPLATE.format(
            maker_instructions=maker_instructions
        )
        checker_system = Prompts.CHECKER_SYSTEM_TEMPLATE.format(
            checker_criteria=checker_criteria
        )

        # Create Agents (context enabled)
        self.maker = Agent(
            name=f"{name}::Maker",
            description="Maker in Maker-Checker workflow",
            role_prompt=maker_system,
            llm_engine=maker_llm,
            context_enabled=True,
        )
        self.checker = Agent(
            name=f"{name}::Checker",
            description="Checker in Maker-Checker workflow",
            role_prompt=checker_system,
            llm_engine=checker_llm,
            context_enabled=True,
        )

    def clear_memory(self):
        self.maker.clear_memory()
        self.checker.clear_memory()

    # --- internal helpers ---
    @staticmethod
    def _parse_checker_json(raw: str) -> dict:
        s = raw.strip()
        # strip code fences if model adds them
        if s.startswith("```"):
            s = re.sub(r"^```(?:json)?\s*", "", s)
            s = re.sub(r"\s*```$", "", s)
        try:
            obj = json.loads(s)
        except Exception as e:
            raise ValueError(f"Checker did not return valid JSON: {e}\nRaw:\n{s}")
        # strict keys/types
        if not (isinstance(obj, dict) and "approved" in obj and "revisions" in obj):
            raise ValueError(f"Checker JSON missing required keys. Got: {obj}")
        if not isinstance(obj["approved"], bool):
            raise ValueError(f"'approved' must be boolean. Got: {type(obj['approved']).__name__}")
        if not isinstance(obj["revisions"], str):
            raise ValueError(f"'revisions' must be string. Got: {type(obj['revisions']).__name__}")
        return obj

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        logging.info(
            f"\n+---{'-'*len(self._name + ' Starting')}---+"
            f"\n|   {self._name} Starting   |"
            f"\n+---{'-'*len(self._name + ' Starting')}---+"
        )
        
        payload = _normalize_to_single_payload(*args, **kwargs)
        text = _pretty_prompt(payload)
        maker_user = f"Given the user's new request, create a first draft:\n{text}"
        rounds: list[dict] = []
        draft = ""
        approved = False

        # Prepare first maker user message (no revisions yet)
        logging.info(f"{self.name} Making first draft")
        draft = self.maker.invoke(maker_user)

        for i in range(1, self.max_revisions + 1):
            # Checker reviews
            logging.info(f"{self.name} Reviewing {i}th draft")
            checker_user = f"Review the following draft against your criteria:\n{draft}"
            verdict_raw = self.checker.invoke(checker_user)
            verdict = self._parse_checker_json(str(verdict_raw))
            approved = verdict["approved"]
            revisions = verdict["revisions"]

            rounds.append({
                "approved": approved,
                "revisions": revisions,
                "draft": draft,
            })
            if approved:
                break
            # Make a new revised draft
            logging.info(f"{self.name} Re-Making {i}th draft")
            maker_user = f"Given the below revision notes, make a new draft:\n{revisions}"
            draft = self.maker.invoke(maker_user)

        logging.info(
            f"\n+---{'-'*len(self._name + ' Finished')}---+"
            f"\n|   {self._name} Finished   |"
            f"\n+---{'-'*len(self._name + ' Finished')}---+\n"
        )
        return rounds, draft

class Selector(Workflow):
    def __init__(
        self, name: str, description: str, branches: List[Agent|Tool|Workflow], decider: LLMEngine|Agent|Workflow|Tool):
        super().__init__(name, description)

        # Wrap branches as SingleAgent for uniform .invoke(*args, **kwargs)
        self.branches: List[Workflow] = [
            SingleAgent(b) if isinstance(b, Agent) else
            (ToolFlow(b) if isinstance(b, Tool) else b) for b in (branches or [])]

        # Build decider Agent with SYSTEM role-prompt that lists branches
        self.custom_agent = isinstance(decider, Agent)
        if isinstance(decider, LLMEngine):
            self.decider = SingleAgent(Agent(
                name=f"{name}::Selector",
                description="Branch selection agent",
                role_prompt=self._build_decider_system_prompt(),
                llm_engine=decider,
            ))
        elif self.custom_agent:
            self.decider = SingleAgent(decider)
        elif isinstance(decider, Tool):
            self.decider = ToolFlow(decider)
        else:
            self.decider = decider

    # ---- internal helpers ----

    def _build_decider_system_prompt(self) -> str:
        # name: description lines, one per branch (stable, readable)
        branch_lines = ",\n".join(
            f"{b.name}: {b.description}" for b in self.branches
        )
        return Prompts.CONDITIONAL_DECIDER_PROMPT.format(branches=branch_lines)

    def _update_decider_prompt(self) -> None:
        # refresh the decider's SYSTEM prompt when branch set changes
        self.decider.agent.role_prompt = self._build_decider_system_prompt()

    @staticmethod
    def _extract_decision(raw: Any) -> str:
        """
        Accept plain strings OR tiny JSON like {"workflow": "..."} / {"choice": "..."}.
        Fallback: first non-empty line of the text.
        """
        text = str(raw).strip()
        # strip fences if present
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        # try JSON
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                for k in ("workflow", "choice", "selected", "name"):
                    v = obj.get(k)
                    if isinstance(v, str) and v.strip():
                        return v.strip()
        except Exception:
            pass
        # fallback to first line
        return text.splitlines()[0].strip()

    # ---- public API ----

    def add_branch(self, branch: Agent|Tool|Workflow) -> None:
        if isinstance(branch, Agent):
            new_branch = SingleAgent(branch)
        elif isinstance(branch, Tool):
            new_branch = ToolFlow(branch)
        else:
            new_branch = branch
        self.branches.append(new_branch)
        if isinstance(self.decider, SingleAgent) and not self.custom_agent:
            self._update_decider_prompt()

    def remove_branch(self, branch_name: str):
        removed = None
        kept = []
        for b in self.branches:
            if b.name == branch_name and removed is None:
                removed = b
            else:
                kept.append(b)
        self.branches = kept
        self._update_decider_prompt()
        return removed

    def clear_memory(self):
        self.decider.clear_memory()
        for b in self.branches:
            b.clear_memory()

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        logging.info(
            f"\n+---{'-'*len(self._name + ' Starting')}---+"
            f"\n|   {self._name} Starting   |"
            f"\n+---{'-'*len(self._name + ' Starting')}---+"
        )

        if not self.decider or not self.branches:
            raise ValueError("Decider and branches must be set.")
        logging.info(f"[WORKFLOW] Selecting branch via decider on {self._name}")
        decision_raw = self.decider.invoke(*args, **kwargs)
        choice_name = self._extract_decision(decision_raw)
        logging.info(f"[WORKFLOW] Decider chose: {choice_name}")

        # route to the chosen branch with the ORIGINAL payload
        for branch in self.branches:
            if branch.name == choice_name:
                result = branch.invoke(*args, **kwargs)
                logging.info(
                    f"\n+---{'-'*len(self._name + ' Finished')}---+"
                    f"\n|   {self._name} Finished   |"
                    f"\n+---{'-'*len(self._name + ' Finished')}---+\n"
                )
                return result

        raise ValueError(f"Decider chose an unknown branch: {choice_name}")

class Delegator(Workflow):
    def __init__(
        self,
        name: str, description: str,
        delegator_component: LLMEngine | Agent | Tool | Workflow,
        branches: list[Agent | Workflow | Tool],
    ):
        super().__init__(name, description)

        # Normalize branches to ORDERED LIST preserving order
        self.branches: list[Workflow] = [
            (SingleAgent(b) if isinstance(b, Agent)
             else (ToolFlow(b) if isinstance(b, Tool) else b))
            for b in branches
        ]

        # Build the task master once; keep only a boolean to know if it's our internal agent
        self._is_internal_agent: bool = False

        if isinstance(delegator_component, LLMEngine):
            # Internal agent we OWN; update its role-prompt whenever branches change
            internal_agent = Agent(
                name=f"{name}.delegator",
                description="Delegator decider (internal)",
                llm_engine=delegator_component,
                role_prompt=Prompts.DELEGATOR_SYSTEM_PROMPT,
            )
            self.task_master: Workflow = SingleAgent(internal_agent)
            self._is_internal_agent = True
            self._refresh_internal_prompt()
        elif isinstance(delegator_component, Agent):
            self.task_master: Workflow = SingleAgent(delegator_component)
        elif isinstance(delegator_component, Tool):
            self.task_master: Workflow = ToolFlow(delegator_component)
        elif isinstance(delegator_component, Workflow):
            self.task_master: Workflow = delegator_component
        else:
            raise TypeError("delegator_component must be LLMEngine | Agent | Tool | Workflow")

    # ---------------- Branch management ----------------

    def add_branch(self, branch: Agent | Workflow | Tool):
        wf = (SingleAgent(branch) if isinstance(branch, Agent)
              else (ToolFlow(branch) if isinstance(branch, Tool) else branch))
        self.branches.append(wf)
        if self._is_internal_agent:
            self._refresh_internal_prompt()

    def remove_branch(self, branch_name: str):
        idx = next((i for i, b in enumerate(self.branches) if b._name == branch_name), None)
        removed = None
        if idx is not None:
            removed = self.branches.pop(idx)
            if self._is_internal_agent:
                self._refresh_internal_prompt()
        return removed

    def clear_memory(self):
        # Clear task_master memory if supported
        self.task_master.clear_memory()
        # Clear branches
        for b in self.branches:
            b.clear_memory()

    # ---------------- Internal helpers ----------------

    def _refresh_internal_prompt(self):
        """Update the internal decider's role-prompt to include branch list & rules."""
        branch_list = [{"name": b.name, "description": b.description} for b in self.branches]
        dynamic = (
            f"{Prompts.DELEGATOR_SYSTEM_PROMPT}\n\n"
            f"BRANCHES:\n{json.dumps(branch_list, ensure_ascii=False)}\n"
            f"(Remember: include every branch; use null to indicate skipping.)"
        )
        if isinstance(self.task_master, SingleAgent): self.task_master.agent.role_prompt = dynamic

    async def _invoke_branch_async(self, branch: Workflow, payload):
        """
        Invoke a branch workflow with normalized payload conventions.
        For ToolFlow: bind using signature-aware logic (as in ToolFlow.invoke).
        For other Workflows/Agents: pass through payload as a single argument,
        unless payload is an explicit callspec [args, kwargs].
        """
        loop = asyncio.get_running_loop()

        # Skip sentinel
        if payload is None:
            return None

        def _call():
            if isinstance(branch, ToolFlow):
                # Tool binding follows the ToolFlow's own normalization
                if _is_callspec(payload):
                    args, kwargs = payload
                    return branch.invoke(*args, **kwargs)
                if isinstance(payload, dict):
                    return branch.invoke(payload)
                if isinstance(payload, (list, tuple)):
                    return branch.invoke(*payload)
                return branch.invoke(payload)
            else:
                # SingleAgent / other Workflows
                if _is_callspec(payload):
                    args, kwargs = payload
                    return branch.invoke(*args, **kwargs)
                return branch.invoke(payload)

        logging.info(f"[WORKFLOW] Invoking branch: {branch._name}")
        return await loop.run_in_executor(None, _call)

    # ---------------- Public API ----------------

    def invoke(self, *args, **kwargs):
        logging.info(
            f"\n+---{'-' * (len(self._name) + 9)}---+"
            f"\n|   {self._name} Starting   |"
            f"\n+---{'-' * (len(self._name) + 9)}---+"
        )

        if not self.branches:
            raise ValueError("Delegator has no branches to execute.")

        # Build decider call exactly once through the unified task_master
        raw = self.task_master.invoke(*args, **kwargs)

        # Normalize decider output to dict
        if isinstance(raw, dict):
            decider_output = raw
        elif isinstance(raw, str):
            cleaned = re.sub(r"```json(.*?)```", r"\1", raw, flags=re.DOTALL).strip()
            try:
                decider_output = json.loads(cleaned)
            except Exception as e:
                raise ValueError(
                    f"Delegator decider returned a string that is not valid JSON dict: {e}\nOutput was:\n{raw}"
                )
        else:
            raise TypeError(
                "Decider must return a dict (branch_name -> payload) or a JSON-string of that dict."
            )

        # Validate/complete mapping to cover ALL branches
        name_to_index = {b._name: i for i, b in enumerate(self.branches)}
        payloads = [None] * len(self.branches)  # default None (skip)

        for k, v in (decider_output or {}).items():
            idx = name_to_index.get(k)
            if idx is None:
                logging.warning(f"[WORKFLOW] Decider produced unknown branch '{k}' — skipping.")
                continue
            payloads[idx] = v

        # Fan-out concurrently (Delegator is ALWAYS concurrent)
        async def _fanout():
            tasks = [
                self._invoke_branch_async(self.branches[i], payloads[i])
                for i in range(len(self.branches))
            ]
            return await asyncio.gather(*tasks, return_exceptions=False)

        try:
            # Use a fresh loop to avoid conflicts with any outer loops
            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                results = loop.run_until_complete(_fanout())
            finally:
                asyncio.set_event_loop(None)
                loop.close()
        except RuntimeError:
            # Fallback: if event loop policy prevents new loops, try asyncio.run
            results = asyncio.run(_fanout())

        logging.info(
            f"\n+---{'-' * (len(self._name) + 9)}---+"
            f"\n|   {self._name} Finished   |"
            f"\n+---{'-' * (len(self._name) + 9)}---+\n"
        )
        # Tuple aligned to branch order (includes None for skipped)
        return tuple(results)
