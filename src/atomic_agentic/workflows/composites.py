from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence, Union, List

from ..core.Invokable import AtomicInvokable
from .base import Workflow, BundlingPolicy, MappingPolicy, AbsentValPolicy
from .basic import BasicFlow


class SequentialFlow(Workflow):
    """
    A composite Workflow that executes a sequence of steps deterministically.

    Design goals / semantics
    ------------------------
    - **Normalization boundary:** every configured step is wrapped into a `BasicFlow`,
      even if the original component is already a `Workflow`. This guarantees a
      consistent packaging boundary across heterogeneous components (Tools, Agents,
      Workflows).
    - **Schema handoff wiring:** for i in [0..n-2], the upstream wrapper's
      `output_schema` is set to the downstream wrapper's `input_schema`.
      This makes each step package its output into exactly the mapping shape the
      next step expects as inputs.
    - **No compatibility enforcement:** this class does *not* validate that step i
      can satisfy step i+1. Any incompatibilities are surfaced naturally at runtime
      through the invoked wrapper's packaging/validation rules.

    Invocation contract
    -------------------
    `_invoke(inputs)` runs each step with the prior step's packaged output as the
    next step's inputs. The raw result returned by `_invoke` is the final step's
    packaged output mapping.

    Metadata
    --------
    The metadata returned by `_invoke` contains:

    - `midwork_checkpoints`: a list of integer indices, one per executed step, where
      each index refers to the checkpoint created inside that step wrapper during
      this run (i.e., `len(step.checkpoints) - 1` right after invocation).

    Parameters
    ----------
    name, description:
        Standard workflow identity fields.
    steps:
        Optional list of AtomicInvokable components (Tool/Agent/Workflow). Each is
        wrapped into a `BasicFlow` internally.
    output_schema:
        Optional workflow output schema for the SequentialFlow itself. This does not
        override per-step wiring. If omitted, defaults to the base Workflow default.
    bundling_policy, mapping_policy, absent_val_policy, default_absent_val:
        Packaging/validation policies for the SequentialFlow's *own* packaging boundary.
        Step wrapper policies currently use BasicFlow defaults (to be refined separately).

    Step management
    ---------------
    This class provides small, explicit mutation helpers (append/extend/insert/pop/remove/
    replace/clear). These methods rebuild the internal wrapper list and re-apply schema
    handoff wiring each time (still no compatibility enforcement).
    """

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        name: str,
        description: str,
        *,
        output_schema: Optional[Union[list[str], Mapping[str, Any]]] = None,
        steps: Optional[list[AtomicInvokable]] = None,
        bundling_policy: BundlingPolicy = BundlingPolicy.BUNDLE,
        mapping_policy: MappingPolicy = MappingPolicy.STRICT,
        absent_val_policy: AbsentValPolicy = AbsentValPolicy.RAISE,
        default_absent_val: Any = None,
    ) -> None:
        super().__init__(
            name=name,
            description=description,
            output_schema=output_schema,
            bundling_policy=bundling_policy,
            mapping_policy=mapping_policy,
            absent_val_policy=absent_val_policy,
            default_absent_val=default_absent_val,
        )
        self._steps: List[BasicFlow] = []
        self.steps = steps

    # ------------------------------------------------------------------ #
    # Steps Properties
    # ------------------------------------------------------------------ #
    @property
    def steps(self) -> list[BasicFlow]:
        """The normalized list of step wrappers (always `BasicFlow`)."""
        return list(self._steps)

    @steps.setter
    def steps(self, steps: Optional[list[AtomicInvokable]]) -> None:
        prepared_steps: list[BasicFlow] = []

        # Empty steps => no-op sequential flow
        if not steps:
            self._steps = prepared_steps
            self._arguments_map, self._return_type = self.build_args_returns()
            self._is_persistible = self._compute_is_persistible()
            return

        # Normalize everything into BasicFlow wrappers.
        prepared_steps = [BasicFlow(component=step) for step in steps]

        # Wire output->input schema between adjacent wrappers.
        for i in range(len(prepared_steps) - 1):
            prepared_steps[i].output_schema = prepared_steps[i + 1].input_schema

        self._steps = prepared_steps
        self._arguments_map, self._return_type = self.build_args_returns()
        self._is_persistible = self._compute_is_persistible()

    # ------------------------------------------------------------------ #
    # Step management APIs
    # ------------------------------------------------------------------ #
    def append_step(self, step: AtomicInvokable) -> None:
        """Append a new step to the end of the sequence."""
        steps = [_.component for _ in self.steps] + [step]
        self.steps = steps

    def extend(self, steps: Sequence[AtomicInvokable]) -> None:
        """Append multiple steps to the end of the sequence."""
        components = [_.component for _ in self.steps] + list(steps)
        self.steps = components

    def insert(self, index: int, step: AtomicInvokable) -> None:
        """Insert a step at the given index (supports negative indices like list.insert)."""
        components = [_.component for _ in self.steps]
        components.insert(index, step)
        self.steps = components

    def replace(self, index: int, step: AtomicInvokable) -> AtomicInvokable:
        """
        Replace the step at `index` and return the removed component.
        Raises IndexError if index is out of range.
        """
        components = [_.component for _ in self.steps]
        removed = components[index]  # may raise IndexError
        components[index] = step
        self.steps = components
        return removed

    def pop(self, index: int = -1) -> AtomicInvokable:
        """
        Remove and return the component at `index` (default last).
        Raises IndexError if index is out of range.
        """
        components = [_.component for _ in self.steps]
        removed = components.pop(index)  # may raise IndexError
        self.steps = components if components else None
        return removed

    def clear_steps(self) -> None:
        """Remove all steps (becomes an identity/no-op sequential flow)."""
        self.steps = None

    # ------------------------------------------------------------------ #
    # Workflow Helpers
    # ------------------------------------------------------------------ #
    def build_args_returns(self):
        base_args, base_ret = super().build_args_returns()

        # If we don't have steps (or haven't been initialized yet), fall back to base.
        if not self._steps:
            return base_args, base_ret

        # Expose the first step wrapper's arguments_map as this SequentialFlow's inputs.
        return self._steps[0].arguments_map, base_ret

    def _compute_is_persistible(self):
        if not self._steps:
            return True
        return all(step.is_persistible for step in self._steps)

    def _invoke(self, inputs: Mapping[str, Any]):
        if not self._steps:
            return {"midwork_checkpoints": []}, inputs

        checkpoint_indices: list[int] = []
        running_result: Mapping[str, Any] = inputs

        for step in self._steps:
            running_result = step.invoke(running_result)
            checkpoint_indices.append(len(step.checkpoints) - 1)

        return {"midwork_checkpoints": checkpoint_indices}, running_result
