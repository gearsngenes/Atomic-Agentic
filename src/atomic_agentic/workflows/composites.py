from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence, Union, List
import logging

from ..core.Invokable import AtomicInvokable
from .base import Workflow, BundlingPolicy, AbsentValPolicy
from .basic import BasicFlow
from ..core.Parameters import ParamSpec

logger = logging.getLogger(__name__)

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
    bundling_policy, absent_val_policy, default_absent_val:
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
        filter_extraneous_inputs: Optional[bool] = None,
        steps: Optional[list[AtomicInvokable]] = None,
        *,
        output_schema: Optional[Union[type, List[Union[str, ParamSpec]], Mapping[str, Any]]] = None,
        bundling_policy: BundlingPolicy = BundlingPolicy.BUNDLE,
        absent_val_policy: AbsentValPolicy = AbsentValPolicy.RAISE,
        default_absent_val: Any = None,
    ) -> None:
        steps: List[AtomicInvokable] = steps or []
        self._steps: List[BasicFlow] = [BasicFlow(component=step) for step in steps]
        filter = filter_extraneous_inputs if filter_extraneous_inputs is not None else (
            self._steps[0].filter_extraneous_inputs if steps else False)
        super().__init__(
            name=name,
            description=description,
            parameters=steps[0].parameters if steps else [],
            output_schema=output_schema,
            bundling_policy=bundling_policy,
            absent_val_policy=absent_val_policy,
            default_absent_val=default_absent_val,
            filter_extraneous_inputs=filter,
        )
        self._rewire_steps()

    # ------------------------------------------------------------------ #
    # Steps Properties
    # ------------------------------------------------------------------ #
    @property
    def steps(self) -> list[BasicFlow]:
        """The normalized list of step wrappers (always `BasicFlow`)."""
        return list(self._steps)

    @steps.setter
    def steps(self, steps: Optional[list[AtomicInvokable]]) -> None:
        if not steps:
            self._steps = []
        else:
            self._steps = [BasicFlow(component=step) for step in steps]
        self._rewire_steps()

    # ------------------------------------------------------------------ #
    # Step management APIs
    # ------------------------------------------------------------------ #
    def append_step(self, step: AtomicInvokable) -> None:
        """Append a new step to the end of the sequence."""
        self._steps.append(BasicFlow(component=step))
        self._rewire_steps()

    def extend(self, steps: Sequence[AtomicInvokable]) -> None:
        """Append multiple steps to the end of the sequence."""
        self._steps.extend([BasicFlow(component=step) for step in steps])
        self._rewire_steps()

    def insert(self, index: int, step: AtomicInvokable) -> None:
        """Insert a step at the given index (supports negative indices like list.insert)."""
        self._steps.insert(index, BasicFlow(component=step))
        self._rewire_steps()

    def replace(self, index: int, step: AtomicInvokable) -> AtomicInvokable:
        """
        Replace the step at `index` and return the removed component.
        Raises IndexError if index is out of range.
        """
        removed = self._steps[index].component
        self._steps[index].component = step
        self._rewire_steps()
        return removed

    def pop(self, index: int = -1) -> AtomicInvokable:
        """
        Remove and return the component at `index` (default last).
        Raises IndexError if index is out of range.
        """
        removed = self._steps.pop(index)
        self._rewire_steps()
        return removed

    def clear_steps(self) -> None:
        """Remove all steps (becomes an identity/no-op sequential flow)."""
        self.steps = None

    # ------------------------------------------------------------------ #
    # Workflow Helpers
    # ------------------------------------------------------------------ #
    def _rewire_steps(self) -> None:
        """Re-apply output->input schema wiring between adjacent wrappers."""
        for i in range(len(self._steps)-1):
            self._steps[i].output_schema = self._steps[i+1].parameters
        for i in range(len(self._steps)):
            # Configure step policies to fixed values
            self._steps[i].bundling_policy = BundlingPolicy.BUNDLE
            self._steps[i].absent_val_policy = AbsentValPolicy.DROP
            self._steps[i].default_absent_val = None
        # Remove output schema from last step
        if self._steps:
            self._steps[-1].output_schema = self.output_schema
            self._steps[-1].absent_val_policy = self.absent_val_policy
            self._steps[-1].default_absent_val = self.default_absent_val
        # Update SequentialFlow parameters to match first step
        self._parameters = self._steps[0].parameters if self._steps else []

    def _invoke(self, inputs: Mapping[str, Any]):
        if not self._steps:
            return {"midwork_checkpoints": []}, inputs

        checkpoint_indices: list[int] = []
        running_result: Mapping[str, Any] = inputs

        for i, step in enumerate(self._steps):
            logger.info(f"{self.full_name}: invoking step {i}")
            running_result = step.invoke(running_result)
            checkpoint_indices.append(len(step.checkpoints) - 1)

        return {"midwork_checkpoints": checkpoint_indices}, running_result


class MakerCheckerFlow(Workflow):
    """
    A composite Workflow implementing a Maker–Checker (optionally Judge) revision loop.
    
    Overview
    --------
    The Maker–Checker–Judge pattern implements iterative refinement where:
    
    1. **Maker** produces an initial draft from inputs
    2. **Checker** reviews/validates the draft and provides feedback
    3. **Judge** (optional) decides if the draft is acceptable; if not, loop continues
    4. **Maker** revises using checker feedback; process repeats up to ``max_revisions``
    
    Loop Semantics
    --------------
    - **Initialization**: Maker produces initial draft from ``inputs``
    - **Iteration**: For each revision round (up to ``max_revisions``):
      
      1. Checker reviews current draft; outputs become next revision inputs for Maker
      2. If Judge is present:
         
         - Judge examines checker output
         - Judge must return a **boolean** (True = accept, False = continue)
         - If True, loop terminates early; final draft is returned
      
      3. Maker produces revised draft from checker feedback
    
    - **Completion**: After ``max_revisions`` iterations or early judge acceptance,
      the final draft (`~Any`) is returned
    
    Checkpoints & Metadata
    ----------------------
    Metadata dict contains:
    
    - ``maker_checkpoints`` (list[int]): checkpoint indices from each Maker invocation
      (length is ``num_revisions + 1``: initial + revisions)
    - ``checker_checkpoints`` (list[int]): checkpoint indices from each Checker run
    - ``judge_checkpoints`` (list[int] or None): checkpoint indices from Judge runs;
      None if no Judge was configured
    - ``iterations_run`` (int): actual number of revision rounds executed
    - ``stopped_early`` (bool): True if Judge accepted before ``max_revisions``
    
    Design Notes
    ~~~~~~~~~~~~
    - If ``max_revisions=0``, only Maker runs; Checker and Judge are skipped
    - Judge output is treated as a single boolean value (first value from output dict)
    - All steps (Maker, Checker, Judge) are wrapped as ``BasicFlow`` internally
    - Input schema comes from Maker; output schema is configurable
    """

    def __init__(
        self,
        name: str,
        description: str,
        maker: AtomicInvokable,
        checker: AtomicInvokable,
        filter_extraneous_inputs: Optional[bool] = None,
        judge: Optional[AtomicInvokable] = None,
        max_revisions: int = 1,
        *,
        output_schema: Optional[Union[list[str], Mapping[str, Any]]] = None,
        bundling_policy: BundlingPolicy = BundlingPolicy.BUNDLE,
        absent_val_policy: AbsentValPolicy = AbsentValPolicy.RAISE,
        default_absent_val: Any = None,
    ) -> None:
        # ------------------------------------------------------------
        # PREPARE attributes needed by build_args_returns()
        # ------------------------------------------------------------
        self._maker: BasicFlow = BasicFlow(component=maker)
        self._checker: BasicFlow = BasicFlow(component=checker)
        self._judge: Optional[BasicFlow] = BasicFlow(component=judge, bundling_policy=BundlingPolicy.UNBUNDLE) if judge is not None else None
        self._max_revisions: int = max_revisions
        filter = filter_extraneous_inputs if filter_extraneous_inputs is not None else self._maker.filter_extraneous_inputs
        # ------------------------------------------------------------
        # Base Workflow init (will call build_args_returns)
        # ------------------------------------------------------------
        super().__init__(
            name=name,
            description=description,
            parameters=self._maker.parameters,
            output_schema=output_schema,
            bundling_policy=bundling_policy,
            absent_val_policy=absent_val_policy,
            default_absent_val=default_absent_val,
            filter_extraneous_inputs=filter,
        )
        self._rebuild()

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #
    @property
    def maker(self) -> BasicFlow:
        return self._maker

    @maker.setter
    def maker(self, candidate: AtomicInvokable) -> None:
        self._maker = BasicFlow(component=candidate,
                                absent_val_policy=AbsentValPolicy.DROP)
        self._rebuild()

    @property
    def checker(self) -> BasicFlow:
        return self._checker

    @checker.setter
    def checker(self, candidate: AtomicInvokable) -> None:
        self._checker = BasicFlow(component=candidate,
                                  absent_val_policy=AbsentValPolicy.DROP)
        self._rebuild()

    @property
    def judge(self) -> Optional[BasicFlow]:
        return self._judge

    @judge.setter
    def judge(self, candidate: Optional[AtomicInvokable]) -> None:
        self._judge = BasicFlow(component=candidate,
                                absent_val_policy=AbsentValPolicy.DROP,
                                bundling_policy=BundlingPolicy.UNBUNDLE,
                                filter_extraneous_inputs=True) if candidate is not None else None
        self._rebuild()

    @property
    def max_revisions(self) -> int:
        return self._max_revisions

    @max_revisions.setter
    def max_revisions(self, value: int) -> None:
        if not isinstance(value, int):
            raise TypeError("max_revisions must be an int")
        if value < 0:
            raise ValueError("max_revisions must be >= 0")
        self._max_revisions = value

    # ------------------------------------------------------------------ #
    # Wiring / validation
    # ------------------------------------------------------------------ #
    def _rebuild(self) -> None:
        """Re-apply wiring and re-compute args/returns."""
        # Wire maker <-> checker
        self._checker.output_schema = self._maker.parameters
        self._maker.output_schema = self._checker.parameters
        self._parameters = list(self._maker.parameters)

    # ------------------------------------------------------------------ #
    # Invocation
    # ------------------------------------------------------------------ #
    def _invoke(self, inputs: Mapping[str, Any]):
        maker_ckpts: list[int] = []
        checker_ckpts: list[int] = []
        judge_ckpts: Optional[list[int]] = [] if self._judge is not None else None

        stopped_early = False

        # Initial draft
        logger.info(f"{self.full_name}: invoking self.maker for initial draft")
        draft = self._maker.invoke(inputs)
        maker_ckpts.append(len(self._maker.checkpoints) - 1)

        if self._max_revisions == 0:
            return {
                "maker_checkpoints": maker_ckpts,
                "checker_checkpoints": checker_ckpts,
                "judge_checkpoints": judge_ckpts,
                "iterations_run": 0,
                "stopped_early": False,
            }, draft

        for _ in range(self._max_revisions):
            # Checker
            logger.info(f"{self.full_name}: invoking self.checker for revision {_+1}")
            next_inputs = self._checker.invoke(draft)
            checker_ckpts.append(len(self._checker.checkpoints) - 1)

            # Judge (optional)
            if self._judge is not None:
                logger.info(f"{self.full_name}: self.judge inspecting revision {_+1}")
                judge_out = self._judge.invoke(next_inputs)
                judge_ckpts.append(len(self._judge.checkpoints) - 1)

                decision = next(iter(judge_out.values()))
                if not isinstance(decision, bool):
                    raise TypeError("Judge must return a boolean")

                if decision:
                    logger.info(f"{self.full_name}: judge accepted revision {_+1}, stopping early")
                    stopped_early = True
                    break

            # Maker rework
            logger.info(f"{self.full_name}: applying feedback from revision {_+1} for draft {_+2}")
            draft = self._maker.invoke(next_inputs)
            maker_ckpts.append(len(self._maker.checkpoints) - 1)

        return {
            "maker_checkpoints": maker_ckpts,
            "checker_checkpoints": checker_ckpts,
            "judge_checkpoints": judge_ckpts,
            "iterations_run": len(checker_ckpts),
            "stopped_early": stopped_early,
        }, draft
