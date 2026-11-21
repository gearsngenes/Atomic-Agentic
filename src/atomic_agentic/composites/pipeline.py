from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

from ..invokables.atomic_invokable import AtomicInvokable
from ..tools.toolification import to_invokable
from ..utils.binding import Binding, normalize_binding
from .graph import Graph

logger = logging.getLogger(__name__)

__all__ = ["Pipeline"]


class Pipeline(AtomicInvokable):
    """Fixed ordered chain of invokables, run as a linear private ``Graph``.

    Construction contract
    ---------------------
    - ``steps`` is a non-empty list. Each entry is an invokable, a ``str``
      (becomes a ``PromptTemplate``), a callable (becomes a ``Tool``), or a
      2-tuple ``(entry, binding)``. ``(entry, None)`` is the same as a bare
      entry. Duplicate instances and a single step are legal.
    - Entries are coerced through the shared ``to_invokable``; a coerced step
      is named ``<pipeline name>_{i}`` and the private graph's node names match.
    - The pipeline holds its graph by composition and has no mutators: topology
      and step instances are fixed at construction (shallow immutability:
      nested invokables keep their own mutability).
    - ``parameters`` are step 0's, copied once. ``return_type`` is the last
      step's when it is bare, ``dict[str, Any]`` when it is bound.

    Runtime contract
    ----------------
    - State flows by binding through the graph's shared state: a later step may
      read any earlier bound key by name, and the call inputs stay visible to
      every step. A bare intermediate step must return a mapping (merged into
      state) or ``TypeError`` is raised by the graph; give it a binding to store
      a non-mapping response.
    - A bare last step's raw response is the result. A bound last step returns a
      dict of its destination keys; a selector that never produced a value
      raises ``KeyError``. The result is a plain ``AtomicResult`` (no trace).
    """

    @dataclass(frozen=True)
    class Step:
        """One configured step: the invokable and its output binding (None = bare)."""

        invokable: AtomicInvokable
        binding: Binding | None = None

    def __init__(
        self,
        name: str,
        description: str,
        steps: list[AtomicInvokable | str | Callable[..., Any] | tuple[Any, Binding | None]],
    ) -> None:
        # 1. steps shape
        if not isinstance(steps, list):
            raise TypeError(f"steps must be a non-empty list, got {type(steps)!r}")
        if not steps:
            raise ValueError("steps must not be empty")

        # 2. unpack, normalize and coerce each step
        built: list[Pipeline.Step] = []
        for index, entry in enumerate(steps):
            if isinstance(entry, tuple):
                if len(entry) != 2:
                    raise ValueError(
                        f"step {index} tuple must be (step, binding); got length {len(entry)}"
                    )
                obj, raw_binding = entry
            else:
                obj, raw_binding = entry, None
            binding = normalize_binding(raw_binding, label=f"{name} step {index} binding")
            invokable = to_invokable(
                obj,
                name=f"{name}_{index}",
                description=f"Step {index} of pipeline {name}",
            )
            built.append(Pipeline.Step(invokable, binding))
        steps_tuple = tuple(built)

        # 3. the last step must name a destination when bound
        last = steps_tuple[-1]
        if last.binding is not None and not last.binding:
            raise ValueError(f"{name}: the last step's binding must name at least one destination")

        # 4. graph nodes: a bare last step is stored under the reserved result key
        node_names = [f"{name}_{i}" for i in range(len(steps_tuple))]
        nodes: dict[str, Any] = {}
        for node_name, step in zip(node_names, steps_tuple):
            graph_binding = step.binding
            if step is last and graph_binding is None:
                graph_binding = "__result__"
            nodes[node_name] = (step.invokable, graph_binding)

        # 5. returns: raw response when bare, a dict of destinations when bound
        if last.binding is None:
            returns: str | list[str] = "__result__"
        elif isinstance(last.binding, str):
            returns = [last.binding]
        else:
            returns = list(last.binding.values())

        # 6. private graph
        parameters = steps_tuple[0].invokable.parameters
        graph = Graph(
            name,
            description,
            nodes,
            edges=list(zip(node_names, node_names[1:])),
            start=node_names[0],
            parameters=parameters,
            returns=returns,
            max_traversal_depth=len(steps_tuple),
        )

        # 7. base init
        return_type = last.invokable.return_type if last.binding is None else "dict[str, Any]"
        super().__init__(
            name=name,
            description=description,
            parameters=parameters,
            return_type=return_type,
        )
        self._graph = graph
        self._steps: tuple[Pipeline.Step, ...] = steps_tuple

    @property
    def steps(self) -> tuple[Pipeline.Step, ...]:
        """The fixed tuple of configured ``Step`` objects."""
        return self._steps

    def _invoke_impl(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Run the private graph and return its selected result."""
        return {"result": self._graph.invoke(inputs).result}

    async def _async_invoke_impl(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Async twin of ``_invoke_impl``."""
        return {"result": (await self._graph.async_invoke(inputs)).result}

    def to_dict(self) -> dict[str, Any]:
        """Serialize the pipeline with each step and its binding."""
        data = super().to_dict()
        data.update(
            {
                "steps": [
                    {
                        "invokable": step.invokable.to_dict(),
                        "binding": dict(step.binding)
                        if isinstance(step.binding, dict)
                        else step.binding,
                    }
                    for step in self._steps
                ],
                "step_count": len(self._steps),
            }
        )
        return data
