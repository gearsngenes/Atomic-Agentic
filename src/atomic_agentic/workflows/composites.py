from ..core.Invokable import AtomicInvokable
from .base import Workflow, BundlingPolicy, MappingPolicy, AbsentValPolicy
from .basic import BasicFlow
from typing import(
    List,
    Optional,
)


class SequentialFlow(Workflow):
    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def __init__(self,
                 name,
                 description,
                 *,
                 output_schema = None,
                 steps: Optional[List[AtomicInvokable]] = None,
                 bundling_policy = BundlingPolicy.BUNDLE,
                 mapping_policy = MappingPolicy.STRICT,
                 absent_val_policy = AbsentValPolicy.RAISE,
                 default_absent_val = None):
        self.steps = steps
        super().__init__(name,
                         description,
                         output_schema=output_schema,
                         bundling_policy=bundling_policy,
                         mapping_policy=mapping_policy,
                         absent_val_policy=absent_val_policy,
                         default_absent_val=default_absent_val)

    # ------------------------------------------------------------------ #
    # StepFlow Properties
    # ------------------------------------------------------------------ #    
    @property
    def steps(self) -> List[AtomicInvokable]:
        return self._steps
    @steps.setter
    def steps(self, steps: Optional[List[AtomicInvokable]]) -> None:
        prepared_steps: List[BasicFlow] = []
        # Empty steps
        if steps is None or not len(steps):
            self._steps = prepared_steps
            return
        # Normalize to basic workflows
        prepared_steps.extend([BasicFlow(component=step) for step in steps])
        # Map previous step output schemas to next step input schemas
        for i in range(len(steps) - 1):
            prepared_steps[i].output_schema = prepared_steps[i+1].input_schema
        # Set self._steps to new steps
        self._steps = prepared_steps
        # Rebuild arguments-map and return type & persistibility
        self._arguments_map, self._return_type = self.build_args_returns()
        self._is_persistible = self._compute_is_persistible()

    # ------------------------------------------------------------------ #
    # Workflow Helpers
    # ------------------------------------------------------------------ #
    def build_args_returns(self):
        base_args, base_ret = super().build_args_returns()
        if not self.steps:
            return base_args, base_ret
        return self.steps[0].arguments_map, base_ret

    def _compute_is_persistible(self):
        if not self.steps:
            return True
        return all([step.is_persistible for step in self.steps])

    def _invoke(self, inputs):
        # if no steps, return original inputs
        if not self.steps:
            return inputs
        # initialize partial results tracking
        partial_results = []
        running_result = inputs
        # run each step deterministically, adding partial steps to metadata
        for step in self.steps:
            running_result = step.invoke(running_result)
            partial_results.append(running_result)
        # return metadata and final result
        metadata = {"partial_results":partial_results}
        return metadata, running_result