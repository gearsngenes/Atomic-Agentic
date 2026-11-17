from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Collection, Mapping
from typing import Any, ClassVar, Dict

from ..exceptions import ExecutionError
from ..core.atomic_invokable import AtomicInvokable
from ..models.parameters import ParamSpec
from ..models.results.workflows import WorkflowResult

__all__ = [
    "Workflow",
]

class Workflow(AtomicInvokable, ABC):
    """
    Base workflow primitive focused on orchestration and result tracing.

    Contract
    --------
    - Inputs are dict-first and filtered through ``AtomicInvokable.filter_inputs()``.
    - Subclasses implement ``_run()`` and may optionally override ``_async_run()``.
    - Both run hooks must return ``(payload, result_kwargs)`` where:
        * ``payload`` is the caller-facing result value (any type)
        * ``result_kwargs`` is a dict of per-kind extra fields forwarded to
          the result subclass constructor — including, when the executing
          subclass populates it, a ``trace`` entry.
    - ``_invoke_impl`` / ``_async_invoke_impl`` merge the two into the payload
      dict, and the inherited ``AtomicInvokable`` template builds ``result_cls``
      from it. Subclasses with fixed per-instance fields include them in
      ``result_kwargs``. Whether ``trace`` is populated is controlled by
      ``include_trace``.
    """

    result_cls: ClassVar[type[WorkflowResult]] = WorkflowResult

    def __init__(
        self,
        name: str,
        description: str,
        parameters: list[ParamSpec],
        return_type: str | Collection[str],
        *,
        include_trace: bool = True,
    ) -> None:
        super().__init__(
            name=name,
            description=description,
            parameters=parameters,
            return_type=return_type,
        )

        self.include_trace = include_trace

    # ------------------------------------------------------------------ #
    # Trace toggle
    # ------------------------------------------------------------------ #
    @property
    def include_trace(self) -> bool:
        """Whether this workflow's results populate their ``trace`` field."""
        return self._include_trace

    @include_trace.setter
    def include_trace(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError(
                f"include_trace must be a bool, got {type(value).__name__}."
            )
        self._include_trace = value

    # ------------------------------------------------------------------ #
    # Subclass run hooks
    # ------------------------------------------------------------------ #
    @abstractmethod
    def _run(self, inputs: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
        """
        Execute the workflow's core synchronous logic.

        Returns
        -------
        tuple[Any, dict[str, Any]]
            ``(payload, result_kwargs)`` where:
            - ``payload`` is the caller-facing workflow output (any type)
            - ``result_kwargs`` is a plain dict of per-kind extra fields
              forwarded to the result constructor
        """
        raise NotImplementedError

    async def _async_run(
        self,
        inputs: Mapping[str, Any],
    ) -> tuple[Any, dict[str, Any]]:
        """Default async compatibility wrapper around ``_run()``."""
        return await asyncio.to_thread(self._run, inputs)

    # ------------------------------------------------------------------ #
    # Invoke hooks
    # ------------------------------------------------------------------ #
    def _invoke_impl(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Run ``_run``; payload = ``result`` plus the per-kind fields it returns."""
        try:
            payload, result_kwargs = self._run(inputs)
        except Exception as exc:
            raise ExecutionError(
                f"{type(self).__name__}._run failed: {exc}"
            ) from exc
        return {"result": payload, **result_kwargs}

    async def _async_invoke_impl(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Async twin of ``_invoke_impl`` over ``_async_run``."""
        try:
            payload, result_kwargs = await self._async_run(inputs)
        except Exception as exc:
            raise ExecutionError(
                f"{type(self).__name__}._async_run failed: {exc}"
            ) from exc
        return {"result": payload, **result_kwargs}

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        """Minimal diagnostic snapshot."""
        data = super().to_dict()
        data.update(
            {
                "include_trace": self._include_trace,
            }
        )
        return data
