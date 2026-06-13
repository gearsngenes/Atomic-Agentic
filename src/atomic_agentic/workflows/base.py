from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from ..core.Exceptions import ExecutionError
from ..core.Invokable import AtomicInvokable
from ..core.Parameters import ParamSpec
from dataclasses import dataclass
from typing import Any
from ..results.workflows import WorkflowResult

logger = logging.getLogger(__name__)

__all__ = [
    "WorkflowCheckpoint",
    "Workflow",
]

@dataclass(frozen=True, slots=True)
class WorkflowCheckpoint:
    """A single workflow invocation record."""

    inputs: dict[str, Any]
    result: WorkflowResult

class Workflow(AtomicInvokable, ABC):
    """
    Base workflow primitive focused on orchestration and checkpointing.

    Contract
    --------
    - Inputs are dict-first and filtered through ``AtomicInvokable.filter_inputs()``.
    - Subclasses implement ``_run()`` and may optionally override ``_async_run()``.
    - Both run hooks must return ``(payload, result_kwargs)`` where:
        * ``payload`` is the caller-facing result value (any type)
        * ``result_kwargs`` is a dict of per-kind extra fields forwarded to
          the result subclass constructor via ``make_result()``
    - Public ``invoke()`` / ``async_invoke()`` wrap the payload in a
      ``WorkflowResult``-family envelope and record a ``WorkflowCheckpoint``.
    """

    def __init__(
        self,
        name: str,
        description: str,
        parameters: list[ParamSpec],
        return_type: str,
        *,
        filter_extraneous_inputs: bool = True,
    ) -> None:
        super().__init__(
            name=name,
            description=description,
            parameters=parameters,
            return_type=return_type,
            filter_extraneous_inputs=filter_extraneous_inputs,
        )

        self._checkpoints: list[WorkflowCheckpoint] = []

    # ------------------------------------------------------------------ #
    # Checkpoint properties
    # ------------------------------------------------------------------ #
    @property
    def checkpoints(self) -> list[WorkflowCheckpoint]:
        """Return a shallow copy of recorded checkpoints."""
        return list(self._checkpoints)

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
    def get_checkpoint(
        self, key: int | str | WorkflowResult
    ) -> Optional[WorkflowCheckpoint]:
        """Return the checkpoint matching the given key.

        Parameters
        ----------
        key:
            - ``int`` — positional index into the checkpoint list. Raises
              ``IndexError`` naturally for out-of-range access.
            - ``WorkflowResult`` — identity match on ``checkpoint.result``.
              Returns the first match or ``None``.
            - ``str`` — run id match on ``checkpoint.result.run_id``.
              Returns the first match or ``None``.
        """
        if isinstance(key, int):
            return self._checkpoints[key]
        if isinstance(key, WorkflowResult):
            for checkpoint in self._checkpoints:
                if checkpoint.result is key:
                    return checkpoint
            return None
        for checkpoint in self._checkpoints:
            if checkpoint.result.run_id == key:
                return checkpoint
        return None

    def invoke(self, inputs: Mapping[str, Any]) -> WorkflowResult:
        """Synchronously invoke the workflow."""
        with self._invoke_lock:
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

            self._checkpoints.append(
                WorkflowCheckpoint(result=workflow_result, inputs=dict(filtered_inputs))
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

        self._checkpoints.append(
            WorkflowCheckpoint(result=workflow_result, inputs=dict(filtered_inputs))
        )

        logger.info("[Async %s finished]", self.full_name)
        return workflow_result

    # ------------------------------------------------------------------ #
    # Memory / serialization
    # ------------------------------------------------------------------ #
    def clear_memory(self) -> None:
        """Clear workflow-owned checkpoint history."""
        self._checkpoints.clear()

    def to_dict(self) -> Dict[str, Any]:
        """Minimal diagnostic snapshot."""
        data = super().to_dict()
        data.update(
            {
                "checkpoint_count": len(self._checkpoints),
                "runs": [
                    checkpoint.result.run_id for checkpoint in self._checkpoints
                ],
            }
        )
        return data
