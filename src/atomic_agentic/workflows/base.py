from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any, Dict, Generic, Optional
from uuid import uuid4

from ..core.Exceptions import ExecutionError, ValidationError
from ..core.Invokable import AtomicInvokable
from ..core.Parameters import ParamSpec
from .metadata import M, WorkflowCheckpoint, WorkflowRunMetadata

logger = logging.getLogger(__name__)

__all__ = [
    "FlowResultDict",
    "WorkflowCheckpoint",
    "Workflow",
]


class FlowResultDict(dict[str, Any]):
    """Dict-like workflow result carrying a non-item ``run_id`` attribute."""

    __slots__ = ("run_id",)

    def __init__(self, *args: Any, run_id: str, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.run_id = run_id

    def copy(self) -> FlowResultDict:
        """Return a shallow copy preserving ``run_id``."""
        return type(self)(self, run_id=self.run_id)


class Workflow(AtomicInvokable, ABC, Generic[M]):
    """
    Base workflow primitive focused on orchestration and checkpointing.

    Contract
    --------
    - Inputs are dict-first and filtered through ``AtomicInvokable.filter_inputs()``.
    - Subclasses implement ``_run()`` and may optionally override ``_async_run()``.
    - Both run hooks must return ``(metadata, result)`` where:
        * ``metadata`` is a typed ``WorkflowRunMetadata`` record
        * ``result`` is a mapping
    - Public ``invoke()`` / ``async_invoke()`` wrap the final mapping in
      ``FlowResultDict`` and record a checkpoint.
    """

    def __init__(
        self,
        name: str,
        description: str,
        parameters: list[ParamSpec],
        *,
        filter_extraneous_inputs: bool = True,
    ) -> None:
        super().__init__(
            name=name,
            description=description,
            parameters=parameters,
            return_type="FlowResultDict[str, Any]",
            filter_extraneous_inputs=filter_extraneous_inputs,
        )

        self._checkpoints: list[WorkflowCheckpoint[M]] = []

    # ------------------------------------------------------------------ #
    # Checkpoint properties
    # ------------------------------------------------------------------ #
    @property
    def checkpoints(self) -> list[WorkflowCheckpoint[M]]:
        """Return a shallow copy of recorded checkpoints."""
        return list(self._checkpoints)

    @property
    def latest_run(self) -> Optional[str]:
        """Return the most recent checkpoint run id, if any."""
        return self._checkpoints[-1].run_id if self._checkpoints else None

    # ------------------------------------------------------------------ #
    # Subclass run hooks
    # ------------------------------------------------------------------ #
    @abstractmethod
    def _run(self, inputs: Mapping[str, Any]) -> tuple[M, Mapping[str, Any]]:
        """
        Execute the workflow's core synchronous logic.

        Returns
        -------
        tuple[M, Mapping[str, Any]]
            ``(metadata, result)`` where:
            - ``metadata`` is the typed checkpoint metadata for this run
            - ``result`` is the final mapping-shaped workflow output
        """
        raise NotImplementedError

    async def _async_run(
        self,
        inputs: Mapping[str, Any],
    ) -> tuple[M, Mapping[str, Any]]:
        """
        Default async compatibility wrapper around ``_run()``.

        Subclasses with true native async orchestration should override this.
        """
        return await asyncio.to_thread(self._run, inputs)

    # ------------------------------------------------------------------ #
    # Validation / normalization helpers
    # ------------------------------------------------------------------ #
    def _normalize_run_output(
        self,
        *,
        metadata: M,
        result: Mapping[str, Any],
        run_id: str,
    ) -> tuple[M, FlowResultDict]:
        """
        Snapshot validated run outputs into stable base-owned containers.
        """
        if not isinstance(metadata, WorkflowRunMetadata):
            raise ValidationError(
                f"{type(self).__name__}._run returned invalid metadata type: {type(metadata)!r}"
            )

        if not isinstance(result, Mapping):
            raise ValidationError(
                f"{type(self).__name__}._run returned non-mapping result: {type(result)!r}"
            )

        result_dict = FlowResultDict(dict(result), run_id=run_id)
        return metadata, result_dict

    def _checkpoint(
        self,
        *,
        run_id: str,
        started_at: datetime,
        ended_at: datetime,
        inputs: Mapping[str, Any],
        result: Mapping[str, Any],
        metadata: M,
    ) -> WorkflowCheckpoint[M]:
        """
        Construct and append one checkpoint record.

        This is base-owned and shared by sync + async invoke paths.
        """
        checkpoint = WorkflowCheckpoint(
            run_id=run_id,
            started_at=started_at,
            ended_at=ended_at,
            elapsed_s=(ended_at - started_at).total_seconds(),
            inputs=dict(inputs),
            result=dict(result),
            metadata=metadata,
        )
        self._checkpoints.append(checkpoint)
        return checkpoint

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def get_checkpoint(self, run_id: str) -> Optional[WorkflowCheckpoint[M]]:
        """Return the checkpoint matching the given run id, if any."""
        for checkpoint in self._checkpoints:
            if checkpoint.run_id == run_id:
                return checkpoint
        return None

    def invoke(self, inputs: Mapping[str, Any]) -> FlowResultDict:
        """
        Synchronously invoke the workflow.

        Lifecycle
        ---------
        1. Filter inputs.
        2. Record timing + run id.
        3. Execute ``_run()``.
        4. Normalize metadata/result.
        5. Record checkpoint.
        6. Return ``FlowResultDict``.
        """
        with self._invoke_lock:
            logger.info("[%s started]", self.full_name)

            filtered_inputs = self.filter_inputs(inputs)
            started_at = datetime.now(timezone.utc)
            run_id = uuid4().hex

            try:
                metadata, result = self._run(filtered_inputs)
            except Exception as exc:
                raise ExecutionError(f"{type(self).__name__}._run failed: {exc}") from exc

            metadata_obj, flow_result = self._normalize_run_output(
                metadata=metadata,
                result=result,
                run_id=run_id,
            )

            ended_at = datetime.now(timezone.utc)
            self._checkpoint(
                run_id=run_id,
                started_at=started_at,
                ended_at=ended_at,
                inputs=filtered_inputs,
                result=flow_result,
                metadata=metadata_obj,
            )

            logger.info("[%s finished]", self.full_name)
            return flow_result

    async def async_invoke(self, inputs: Mapping[str, Any]) -> FlowResultDict:
        """
        Asynchronously invoke the workflow.

        This mirrors ``invoke()`` but dispatches through ``_async_run()``.
        """
        logger.info("[Async %s started]", self.full_name)

        filtered_inputs = self.filter_inputs(inputs)
        started_at = datetime.now(timezone.utc)
        run_id = uuid4().hex

        try:
            metadata, result = await self._async_run(filtered_inputs)
        except Exception as exc:
            raise ExecutionError(f"{type(self).__name__}._async_run failed") from exc

        metadata_obj, flow_result = self._normalize_run_output(
            metadata=metadata,
            result=result,
            run_id=run_id,
        )

        ended_at = datetime.now(timezone.utc)
        self._checkpoint(
            run_id=run_id,
            started_at=started_at,
            ended_at=ended_at,
            inputs=filtered_inputs,
            result=flow_result,
            metadata=metadata_obj,
        )

        logger.info("[Async %s finished]", self.full_name)
        return flow_result

    # ------------------------------------------------------------------ #
    # Memory / serialization
    # ------------------------------------------------------------------ #
    def clear_memory(self) -> None:
        """Clear workflow-owned checkpoint history."""
        self._checkpoints.clear()

    def to_dict(self) -> Dict[str, Any]:
        """
        Minimal diagnostic snapshot.

        Checkpoints are intentionally omitted here to keep the base snapshot light.
        """
        data = super().to_dict()
        data.update(
            {
                "checkpoint_count": len(self._checkpoints),
                "runs": [checkpoint.run_id for checkpoint in self._checkpoints],
            }
        )
        return data
