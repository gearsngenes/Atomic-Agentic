from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Optional

from .StructuredInvokable import StructuredInvokable
from ..core.Exceptions import ValidationError
from .base import FlowResultDict, Workflow
from .basic import BasicFlow

logger = logging.getLogger(__name__)

class SequentialFlow(Workflow):
    """Execute a fixed ordered sequence of workflow-shaped steps.

    Step normalization
    ------------------
    - Existing ``Workflow`` instances are kept as-is.
    - ``StructuredInvokable`` instances are wrapped once in ``BasicFlow``.

    Construction contract
    ---------------------
    - ``steps`` must be a non-empty ``list[Workflow | StructuredInvokable]``.
    - The topology is fixed at construction.
    - The configured step instances are fixed at construction.
    - No post-construction step mutation API is provided.
    - ``return_index`` selects which executed step result becomes the outer
      workflow result. This is selection-only mutability; it does not alter
      topology or execution order.

    Runtime contract
    ----------------
    - Inputs are passed to the first step.
    - Each step's mapping result is passed directly to the next step.
    - All configured steps execute on every run.
    - The step selected by ``return_index`` determines the final step result
      returned to the workflow base, which then wraps it in the outer
      ``FlowResultDict`` and records the sequential checkpoint.

    Metadata
    --------
    Per-run metadata contains:

    - ``step_records``: list[dict[str, Any]]
        One record per executed step, each containing:
        - ``step``: zero-based step index for that run
        - ``instance_id``: executed step instance id
        - ``full_name``: executed step full name
        - ``run_id``: that step's workflow run id
    - ``return_index``:
        The configured return index for that run.
    - ``resolved_return_step``:
        The resolved absolute step index selected for return.
    - ``returned_step_run_id``:
        The child step run id whose result became the outer sequential result.

    Retrieval helpers
    -----------------
    - ``get_step_records(run_id)`` returns the stored step records for a parent
      sequential run, or ``None`` if the parent run is not found.
    - ``get_step_results(run_id)`` resolves those records back into child step
      checkpoint results and returns ``list[result | None]``. ``None`` is used
      when the child run id no longer resolves to a retained child checkpoint.
    - ``get_step_result(run_id, step_index)`` is a convenience wrapper over
      ``get_step_results(run_id)``.

    Notes
    -----
    This class enforces *fixed sequence topology*, but this is still only
    shallow graph immutability. Nested step objects may retain their own broader
    AA mutability elsewhere.
    """

    def __init__(
        self,
        name: str,
        description: str,
        steps: list[Workflow | StructuredInvokable],
        *,
        return_index: int = -1,
        filter_extraneous_inputs: Optional[bool] = None,
    ) -> None:
        if not isinstance(steps, list):
            raise TypeError(
                f"steps must be a non-empty list[Workflow | StructuredInvokable], got {type(steps)!r}"
            )
        if not steps:
            raise ValueError("steps must not be empty")

        normalized_steps = tuple(self._normalize_step(step) for step in steps)

        resolved_filter = (
            filter_extraneous_inputs
            if filter_extraneous_inputs is not None
            else normalized_steps[0].filter_extraneous_inputs
        )

        super().__init__(
            name=name,
            description=description,
            parameters=normalized_steps[0].parameters,
            filter_extraneous_inputs=resolved_filter,
        )

        self._steps: tuple[Workflow, ...] = normalized_steps
        self.return_index = return_index

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #
    @property
    def steps(self) -> tuple[Workflow, ...]:
        """Return the fixed normalized step tuple."""
        return self._steps

    @property
    def return_index(self) -> int:
        """Configured step index whose result becomes the outer flow result."""
        return self._return_index

    @return_index.setter
    def return_index(self, value: int) -> None:
        """Validate and store the configured return step index."""
        if not isinstance(value, int):
            raise TypeError(f"return_index must be an int, got {type(value)!r}")
        self._resolve_step_index(value)
        self._return_index = value

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalize_step(step: Workflow | StructuredInvokable) -> Workflow:
        """Normalize one configured step into a workflow-shaped step."""
        if isinstance(step, Workflow):
            return step
        if isinstance(step, StructuredInvokable):
            return BasicFlow(component=step)
        raise TypeError(
            "SequentialFlow steps must be Workflow or StructuredInvokable, "
            f"got {type(step)!r}"
        )

    def _resolve_step_index(self, index: int) -> int:
        """Resolve a configured step index using normal Python negative indexing."""
        if not isinstance(index, int):
            raise TypeError(f"step index must be an int, got {type(index)!r}")

        length = len(self._steps)
        resolved = index if index >= 0 else length + index
        if resolved < 0 or resolved >= length:
            raise IndexError(
                f"step index {index} out of range for {length} configured step(s)"
            )
        return resolved

    # ------------------------------------------------------------------ #
    # Run-oriented retrieval
    # ------------------------------------------------------------------ #
    def get_step_records(self, run_id: str) -> Optional[list[dict[str, Any]]]:
        """Return the stored step records for one sequential run.

        Returns ``None`` when the parent sequential checkpoint is not found.

        The returned list is a shallow-copied snapshot of the stored metadata
        records. Each record is validated to contain the current required fields.
        """
        checkpoint = self.get_checkpoint(run_id)
        if checkpoint is None:
            return None

        raw_records = checkpoint.metadata.get("step_records")
        if not isinstance(raw_records, list):
            raise ValidationError(
                f"{self.full_name}: checkpoint metadata missing valid 'step_records' list "
                f"for run_id {run_id!r}"
            )

        validated_records: list[dict[str, Any]] = []

        for record_index, record in enumerate(raw_records):
            if not isinstance(record, Mapping):
                raise ValidationError(
                    f"{self.full_name}: step_records[{record_index}] must be a mapping, "
                    f"got {type(record)!r}"
                )

            step_index = record.get("step")
            instance_id = record.get("instance_id")
            full_name = record.get("full_name")
            child_run_id = record.get("run_id")

            if not isinstance(step_index, int) or step_index < 0:
                raise ValidationError(
                    f"{self.full_name}: step_records[{record_index}]['step'] must be an int >= 0, "
                    f"got {step_index!r}"
                )
            if not isinstance(instance_id, str) or not instance_id.strip():
                raise ValidationError(
                    f"{self.full_name}: step_records[{record_index}]['instance_id'] must be a "
                    f"non-empty string, got {instance_id!r}"
                )
            if not isinstance(full_name, str) or not full_name.strip():
                raise ValidationError(
                    f"{self.full_name}: step_records[{record_index}]['full_name'] must be a "
                    f"non-empty string, got {full_name!r}"
                )
            if not isinstance(child_run_id, str) or not child_run_id.strip():
                raise ValidationError(
                    f"{self.full_name}: step_records[{record_index}]['run_id'] must be a "
                    f"non-empty string, got {child_run_id!r}"
                )

            validated_records.append(dict(record))

        return validated_records

    def get_step_results(
        self,
        run_id: str,
    ) -> Optional[list[dict[str, Any] | None]]:
        """Return child step results for one sequential run.

        This method uses :meth:`get_step_records` as the source of truth.

        Returns
        -------
        Optional[list[dict[str, Any] | None]]
            - ``None`` if the parent sequential run is not found
            - otherwise, one entry per stored step record
            - each entry is the child step checkpoint result dict, or ``None`` if
              that child run id no longer resolves to a retained child checkpoint
        """
        step_records = self.get_step_records(run_id)
        if step_records is None:
            return None

        step_results: list[dict[str, Any] | None] = []

        for record_index, record in enumerate(step_records):
            step_index = record["step"]
            if step_index >= len(self._steps):
                raise ValidationError(
                    f"{self.full_name}: step_records[{record_index}] references step index "
                    f"{step_index}, but only {len(self._steps)} configured steps exist"
                )

            step: Workflow = self._steps[step_index]

            recorded_instance_id = record["instance_id"]
            if step.instance_id != recorded_instance_id:
                raise ValidationError(
                    f"{self.full_name}: step_records[{record_index}] instance_id mismatch for "
                    f"step {step_index}: recorded {recorded_instance_id!r}, current {step.instance_id!r}"
                )

            child_checkpoint = step.get_checkpoint(record["run_id"])
            if child_checkpoint is None:
                step_results.append(None)
            else:
                step_results.append(dict(child_checkpoint.result))

        return step_results

    def get_step_result(
        self,
        run_id: str,
        step_index: int,
    ) -> Optional[dict[str, Any]]:
        """Return one child step result for one sequential run.

        This method uses :meth:`get_step_results` as its source of truth.

        Returns
        -------
        Optional[dict[str, Any]]
            - ``None`` if the parent sequential run is not found
            - the resolved child result dict for the requested step
            - ``None`` if the child checkpoint for that recorded step run id is
              no longer available

        Raises
        ------
        TypeError
            If ``step_index`` is not an int.
        IndexError
            If ``step_index`` is out of range for the configured steps.
        """
        resolved_index = self._resolve_step_index(step_index)

        step_results = self.get_step_results(run_id)
        if step_results is None:
            return None

        return step_results[resolved_index]

    # ------------------------------------------------------------------ #
    # Workflow run hooks
    # ------------------------------------------------------------------ #
    def _run(
        self,
        inputs: Mapping[str, Any],
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        """Synchronously execute all configured steps and return the selected step result."""
        if not self._steps:
            raise ValidationError(
                f"{self.full_name}: cannot execute an empty SequentialFlow"
            )

        running_result: Mapping[str, Any] = inputs
        step_records: list[dict[str, Any]] = []
        step_results: list[FlowResultDict] = []

        for index, step in enumerate(self._steps):
            logger.info("%s: invoking step %d (%s)", self.full_name, index, step.full_name)
            result = step.invoke(running_result)

            if not isinstance(result, FlowResultDict):
                raise ValidationError(
                    f"{self.full_name}: step {index} ({step.full_name}) returned "
                    f"{type(result)!r}, expected FlowResultDict"
                )

            step_records.append(
                {
                    "step": index,
                    "instance_id": step.instance_id,
                    "full_name": step.full_name,
                    "run_id": result.run_id,
                }
            )
            step_results.append(result)
            running_result = result

        resolved_return_step = self._resolve_step_index(self._return_index)
        returned_result = step_results[resolved_return_step]

        metadata = {
            "step_records": step_records,
            "return_index": self._return_index,
            "resolved_return_step": resolved_return_step,
            "returned_step_run_id": step_records[resolved_return_step]["run_id"],
        }
        return metadata, returned_result

    async def _async_run(
        self,
        inputs: Mapping[str, Any],
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        """Asynchronously execute all configured steps and return the selected step result."""
        if not self._steps:
            raise ValidationError(
                f"{self.full_name}: cannot execute an empty SequentialFlow"
            )

        running_result: Mapping[str, Any] = inputs
        step_records: list[dict[str, Any]] = []
        step_results: list[FlowResultDict] = []

        for index, step in enumerate(self._steps):
            logger.info(
                "[Async %s]: invoking step %d (%s)",
                self.full_name,
                index,
                step.full_name,
            )
            result = await step.async_invoke(running_result)

            if not isinstance(result, FlowResultDict):
                raise ValidationError(
                    f"{self.full_name}: async step {index} ({step.full_name}) returned "
                    f"{type(result)!r}, expected FlowResultDict"
                )

            step_records.append(
                {
                    "step": index,
                    "instance_id": step.instance_id,
                    "full_name": step.full_name,
                    "run_id": result.run_id,
                }
            )
            step_results.append(result)
            running_result = result

        resolved_return_step = self._resolve_step_index(self._return_index)
        returned_result = step_results[resolved_return_step]

        metadata = {
            "step_records": step_records,
            "return_index": self._return_index,
            "resolved_return_step": resolved_return_step,
            "returned_step_run_id": step_records[resolved_return_step]["run_id"],
        }
        return metadata, returned_result

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        """Serialize the fixed sequence and its configured selection policy."""
        data = super().to_dict()
        data.update(
            {
                "steps": [step.to_dict() for step in self._steps],
                "step_count": len(self._steps),
                "return_index": self._return_index,
            }
        )
        return data
