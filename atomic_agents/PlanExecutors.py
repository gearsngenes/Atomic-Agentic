import asyncio, inspect, re, json, logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List
# ────────────────────────────────────────────────────────────────
# 1.  Common helper mix-in
# ────────────────────────────────────────────────────────────────
class _ExecutorBase:
    """
    Shared placeholder-resolution & step bookkeeping for both sync/async.
    """

    TOKEN = re.compile(r"\{\{step(\d+)\}\}")       # all occurrences
    TOKEN_FULL = re.compile(r"^\{\{step(\d+)\}\}$")  # whole-field match

    def __init__(self):
        self.ctx: Dict[str, Dict[str, Any]] = {}   # stepN → {"result": …}

    # ----------------------------------------
    def _resolve(self, obj: Any) -> Any:
        """
        Recursively replace '{{stepN}}' tokens with the referenced result.
        """
        if isinstance(obj, str):
            whole = self.TOKEN_FULL.match(obj)
            if whole:                                       # full substitution
                return self.ctx[f"step{whole.group(1)}"]["result"]
            return self.TOKEN.sub(
                lambda m: str(self.ctx[f"step{m.group(1)}"]["result"]), obj
            )
        if isinstance(obj, list):
            return [self._resolve(x) for x in obj]
        if isinstance(obj, dict):
            return {k: self._resolve(v) for k, v in obj.items()}
        return obj

    # ----------------------------------------
    @staticmethod
    def _is_async_fn(fn: Any) -> bool:
        return inspect.iscoroutinefunction(fn)

# ────────────────────────────────────────────────────────────────
# 2.  Synchronous executor
# ────────────────────────────────────────────────────────────────
class PlanExecutor(_ExecutorBase):
    """
    Blocking execution: steps run sequentially on the caller’s thread.
    """
    def __init__(self, debug=False):
        super().__init__()
        self.debug = debug
    def execute(self, plan: Dict[str, Any]) -> Any:
        steps: List[Dict] = plan["steps"]
        tools: Dict[str, Dict] = plan["tools"]

        for idx, step in enumerate(steps):
            fn_meta = tools[step["function"]]
            fn      = fn_meta["callable"]
            if self._is_async_fn(fn):
                raise RuntimeError(
                    f"Function '{step['function']}' is async; "
                    "use AsyncPlanExecutor instead."
                )
            if self.debug:
                logging.info(f"[TOOL] {step["function"]}, args: {step["args"]}")
            args    = self._resolve(step.get("args", {}))
            result  = fn(**args)
            self.ctx[f"step{idx}"] = {**step, "result": result}
        final_result = self.ctx[f"step{len(steps)-1}"]["result"]
        return final_result

# ────────────────────────────────────────────────────────────────
# 3.  Asynchronous executor
# ────────────────────────────────────────────────────────────────
class AsyncPlanExecutor(_ExecutorBase):
    """
    Executes as many independent steps concurrently as dependencies allow.

    •  invoke_async → coroutine for use inside an event loop
    •  invoke      → convenience wrapper (runs its own loop)
    """

    def __init__(self, debug = False, max_workers: int | None = None):
        super().__init__()
        self._pool = ThreadPoolExecutor(max_workers=max_workers)
        self.debug = debug

    # ------------- public API -----------------------------------
    async def execute_async(self, plan: Dict[str, Any]) -> Any:
        steps  = plan["steps"]
        tools  = plan["tools"]

        # Build dependency graph: step i ↦ set(indices it depends on)
        deps = {
            i: {int(n) for n in self.TOKEN.findall(json.dumps(step.get("args", {})))}
            for i, step in enumerate(steps)
        }
        remaining = set(range(len(steps)))
        completed = set()

        async def _run_one(i: int):
            step   = steps[i]
            fn_meta = tools[step["function"]]
            fn     = fn_meta["callable"]
            args   = self._resolve(step.get("args", {}))
            if self.debug:
                logging.info(f"[TOOL] {step["function"]}, args: {step["args"]}")
            if self._is_async_fn(fn):
                return await fn(**args)
            # run blocking code in thread-pool
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(self._pool, lambda: fn(**args))

        while remaining:
            ready = [i for i in remaining if deps[i] <= completed]
            if not ready:
                raise RuntimeError("Circular dependency detected in plan.")

            coros = {i: _run_one(i) for i in ready}
            results = await asyncio.gather(*coros.values())

            for idx, res in zip(coros.keys(), results):
                self.ctx[f"step{idx}"] = {**steps[idx], "result": res}
                completed.add(idx)
                remaining.remove(idx)

        return self.ctx[f"step{len(steps)-1}"]["result"]

    # ------------- sync wrapper ---------------------------------
    def execute(self, plan: Dict[str, Any]) -> Any:
        """
        Convenience for non-async callers. Spawns a fresh event loop.
        """
        return asyncio.run(self.execute_async(plan))
