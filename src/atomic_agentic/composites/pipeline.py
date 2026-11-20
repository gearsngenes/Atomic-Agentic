from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

from ..invokables.atomic_invokable import AtomicInvokable
from ..tools.toolification import to_invokable
from ..utils.binding import Binding, normalize_binding, slice_response

logger = logging.getLogger(__name__)

__all__ = ["Pipeline"]


class Pipeline(AtomicInvokable):
    """Fixed ordered chain of invokables; each step's response feeds the next.

    Construction contract
    ---------------------
    - ``steps`` is a non-empty list. Each entry is an invokable, a ``str``
      (becomes a ``PromptTemplate``), a callable (becomes a ``Tool``), or a
      2-tuple ``(entry, binding)``. ``(entry, None)`` is the same as a bare
      entry. Duplicate instances and a single step are legal.
    - Entries are coerced through the shared ``to_invokable``; a coerced step
      is named ``<pipeline name>_step_{i}``.
    - Topology and step instances are fixed at construction (shallow
      immutability: nested invokables keep their own mutability).
    - ``parameters`` are step 0's, copied once; step 0's binding only shapes
      its output. ``return_type`` is the last step's when its binding is bare,
      ``dict[str, Any]`` whenever a binding is present.

    Runtime contract
    ----------------
    - Inputs go to step 0. A non-last step's response is sliced by its binding
      (shared grammar, ``on_missing="skip"``) or passed through raw when bare,
      and becomes the next step's inputs.
    - The last step's response is the pipeline's response: raw when bare, a
      dict when bound. The result is a plain ``AtomicResult``.
    - ``Pipeline`` validates no handoff itself. Binding failures come from
      ``slice_response`` and everything else from the next step's
      ``filter_inputs`` (all ``TypeError``).
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
        if not isinstance(steps, list):
            raise TypeError(f"steps must be a non-empty list, got {type(steps)!r}")
        if not steps:
            raise ValueError("steps must not be empty")

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
                name=f"{name}_step_{index}",
                description=f"Step {index} of pipeline {name}",
            )
            built.append(Pipeline.Step(invokable, binding))
        steps_tuple = tuple(built)

        last = steps_tuple[-1]
        return_type = last.invokable.return_type if last.binding is None else "dict[str, Any]"

        super().__init__(
            name=name,
            description=description,
            parameters=steps_tuple[0].invokable.parameters,
            return_type=return_type,
        )
        self._steps: tuple[Pipeline.Step, ...] = steps_tuple

    @property
    def steps(self) -> tuple[Pipeline.Step, ...]:
        """The fixed tuple of configured ``Step`` objects."""
        return self._steps

    def _handoff(self, index: int, response: Any) -> Any:
        """Apply step ``index``'s binding to its response (shared by sync and async)."""
        binding = self._steps[index].binding
        if binding is None:
            return response
        bound = slice_response(
            response,
            binding,
            label=f"{self.full_name} step {index} binding",
            on_missing="skip",
        )
        if isinstance(binding, dict):
            skipped = set(binding.values()) - set(bound)
            if skipped:
                logger.debug(
                    "%s: step %d skipped missing selector(s) for %s",
                    self.full_name,
                    index,
                    sorted(skipped),
                )
        return bound

    def _invoke_impl(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Run the steps in order, piping each handoff into the next step."""
        current: Any = inputs
        for index, step in enumerate(self._steps):
            logger.debug(
                "%s: step %d (%s)", self.full_name, index, step.invokable.full_name
            )
            result = step.invokable.invoke(current)
            current = self._handoff(index, result.result)
        return {"result": current}

    async def _async_invoke_impl(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Async twin of ``_invoke_impl``."""
        current: Any = inputs
        for index, step in enumerate(self._steps):
            logger.debug(
                "[Async %s]: step %d (%s)", self.full_name, index, step.invokable.full_name
            )
            result = await step.invokable.async_invoke(current)
            current = self._handoff(index, result.result)
        return {"result": current}

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
