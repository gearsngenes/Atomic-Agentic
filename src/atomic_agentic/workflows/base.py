from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any, Dict

from ..exceptions import ExecutionError
from ..core.Invokable import AtomicInvokable
from ..models.parameters import ParamSpec
from ..models.results.workflows import WorkflowResult

logger = logging.getLogger(__name__)

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
          the result subclass constructor via ``make_result()`` — including,
          when the executing subclass populates it, a ``trace`` entry.
    - Public ``invoke()`` / ``async_invoke()`` wrap the payload in a
      ``WorkflowResult``-family envelope. Whether that envelope's ``trace``
      field is populated is controlled by ``include_trace``.
    """

    def __init__(
        self,
        name: str,
        namespace: str,
        description: str,
        parameters: list[ParamSpec],
        return_type: str,
        include_trace: bool = True,
    ) -> None:
        super().__init__(
            name=name,
            namespace=namespace,
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
              forwarded to the ``WorkflowResult`` subclass constructor
        """
        raise NotImplementedError

    async def _async_run(
        self,
        inputs: Mapping[str, Any],
    ) -> tuple[Any, dict[str, Any]]:
        """Default async compatibility wrapper around ``_run()``."""
        return await asyncio.to_thread(self._run, inputs)

    # ------------------------------------------------------------------ #
    # Result construction
    # ------------------------------------------------------------------ #
    def make_result(
        self,
        result: Any,
        started_at: datetime,
        ended_at: datetime,
        **result_kwargs: Any,
    ) -> WorkflowResult:
        """Construct a WorkflowResult envelope for this workflow's invocation.

        Subclasses override this to fix their own ``result_cls``. This base
        override is required — without it the fallback is
        ``AtomicInvokable.make_result()`` which hardcodes ``result_cls=AtomicResult``.
        """
        return self._make_result(
            result=result,
            started_at=started_at,
            ended_at=ended_at,
            result_cls=WorkflowResult,
            **result_kwargs,
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def invoke(self, inputs: Mapping[str, Any]) -> WorkflowResult:
        """Synchronously invoke the workflow."""
        logger.info("[%s started]", self.full_name)

        started_at = datetime.now(timezone.utc)
        filtered_inputs = self.filter_inputs(inputs)

        try:
            payload, result_kwargs = self._run(filtered_inputs)
        except Exception as exc:
            raise ExecutionError(
                f"{type(self).__name__}._run failed: {exc}"
            ) from exc

        ended_at = datetime.now(timezone.utc)
        workflow_result = self.make_result(
            result=payload,
            started_at=started_at,
            ended_at=ended_at,
            **result_kwargs,
        )

        logger.info("[%s finished]", self.full_name)
        return workflow_result

    async def async_invoke(self, inputs: Mapping[str, Any]) -> WorkflowResult:
        """Asynchronously invoke the workflow."""
        logger.info("[Async %s started]", self.full_name)

        started_at = datetime.now(timezone.utc)
        filtered_inputs = self.filter_inputs(inputs)

        try:
            payload, result_kwargs = await self._async_run(filtered_inputs)
        except Exception as exc:
            raise ExecutionError(
                f"{type(self).__name__}._async_run failed: {exc}"
            ) from exc

        ended_at = datetime.now(timezone.utc)
        workflow_result = self.make_result(
            result=payload,
            started_at=started_at,
            ended_at=ended_at,
            **result_kwargs,
        )

        logger.info("[Async %s finished]", self.full_name)
        return workflow_result

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
