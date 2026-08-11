from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from datetime import datetime
from typing import Any

from ..exceptions import WorkflowError
from ..core.Invokable import AtomicInvokable
from ..utils.core import run_coro_sync
from ..utils.parameters import (
    build_parameter_reports,
    apply_parameter_reports,
    insert_by_category,
)
from ..constants.core import IDENTIFIER_PATTERN
from ..models.results import AtomicResult
from ..models.results.workflows import ParallelFlowResult
from .base import Workflow

logger = logging.getLogger(__name__)

__all__ = ["ParallelFlow"]


class ParallelFlow(Workflow):
    """Execute a fixed set of branches concurrently and project one outer result.

    Construction contract
    ----------------------
    - ``branches`` is a non-empty ``list[AtomicInvokable]``. No wrapping,
      no dict-input option — branches are always positional.
    - ``parameters`` is not a constructor option. The outer schema is derived
      by reconciling every branch's own parameters as true N-way peers, not a
      left-to-right fold: a name with no type compatible across every branch
      that declares it, or an incompatible kind, raises ``WorkflowError``
      immediately. A name declared identically everywhere it appears is used
      as-is. A name that's compatible but not identical across branches is
      reconciled into one parameter using the full compatible-type witness set
      (which may be broader than any single branch's own declared type) with
      kind/default/description taken from the earliest-declaring branch that
      has it; every such name across one construction call is reported in
      exactly one grouped warning, not one per name. Two branches
      independently declaring a same-kind variadic under different names still
      cannot both survive into the final schema, but this is now caught by the
      shared parameter-order validator every ``AtomicInvokable`` already goes
      through (raises ``SchemaError``, a ``WorkflowError`` subclass) rather
      than a bespoke pre-check naming the specific branches involved -- a
      minor message-quality regression accepted here, matching the same
      trade-off `Agent`'s own Pass 2 rework already made.
    - ``selected_branches`` is a ``list[int]`` of branch positions (each
      ``0 <= index < len(branches)``, no negative wraparound, no
      duplicates) choosing which branches feed the projected result, and
      in what order. ``None`` or ``[]`` both mean "no branches selected" —
      a legal, meaningful state, not an error.
    - ``result_mode`` (``SCALAR``/``LIST``/``TUPLE``/``SET``/``DICT`` class
      constants) selects the projection shape. ``SCALAR`` allows at most
      one selected branch (zero is legal — see Runtime contract).
    - ``result_keys`` is a ``list[str]`` of labels, required exactly when
      ``result_mode='dict'`` and must match ``len(selected_branches)``
      (raise otherwise). Must be ``None``/empty for every other mode
      (raise if provided). No auto-derivation from anywhere — always
      explicit.
    - ``include_trace`` controls whether invocation results populate
      ``trace`` with the full per-branch ``AtomicResult`` tuple (ALL
      branches, not just selected ones). Defaults to ``True``; mutable
      post-construction via the inherited ``include_trace`` property.
    - Branch topology, parameter schema, and output projection are all
      fixed at construction.

    Runtime contract
    ------------------
    - The same outer input mapping is broadcast, unchanged, to every branch.
    - All configured branches always execute, concurrently — selection only
      affects what's projected into the outer result, not what runs.
    - The final outer result is projected from the branches named by
      ``selected_branches``, shaped by ``result_mode``:
        - ``SCALAR``: one selected branch -> that branch's payload; zero
          selected -> ``None``.
        - ``LIST`` / ``TUPLE`` / ``SET``: the selected payloads in that
          container type (empty selection -> empty container).
        - ``DICT``: ``dict(zip(result_keys, payloads))`` (empty selection
          -> ``{}``).

    Result
    ------
    Per-run result fields (see ``ParallelFlowResult``):

    - ``trace``: full ordered tuple of every branch's own ``AtomicResult``
      (all branches, declaration order), when ``include_trace`` is enabled.
    - ``result_mode``: fixed projection mode, injected directly by
      ``make_result()``.
    - ``selected_indices``: fixed branch positions selected for
      projection, injected directly by ``make_result()``.
    - ``result_keys``: single source of truth for the projection's labels,
      computed once at construction and mirrored verbatim onto the result.
      For ``DICT`` mode it's the (validated) ``result_keys`` constructor
      value; for every other mode it's exactly ``selected_indices`` — the
      same tuple, not a separate concept.
    """

    SCALAR = "scalar"
    LIST = "list"
    TUPLE = "tuple"
    SET = "set"
    DICT = "dict"

    _OUTPUT_TYPES = {SCALAR, LIST, TUPLE, SET, DICT}

    def __init__(
        self,
        name: str,
        namespace: str,
        description: str,
        branches: list[AtomicInvokable] | tuple[AtomicInvokable, ...],
        *,
        result_mode: str = LIST,
        selected_branches: list[int] | None = None,
        result_keys: list[str] | None = None,
        include_trace: bool = True,
    ) -> None:
        # 1. branches: always a plain list[AtomicInvokable], no wrapping.
        if not isinstance(branches, (list, tuple)):
            raise TypeError(
                f"branches must be a non-empty list or tuple[AtomicInvokable], got {type(branches)!r}"
            )
        if not branches:
            raise ValueError("branches must not be empty")
        for index, branch in enumerate(branches):
            if not isinstance(branch, AtomicInvokable):
                raise TypeError(
                    f"ParallelFlow branches must be AtomicInvokable, "
                    f"got {type(branch)!r} at position {index}"
                )
        branch_items = tuple(branches)
        branch_count = len(branch_items)

        # 2. Outer schema is fully derived from branches — no override.
        # True N-way peer reconciliation (not a left-to-right fold):
        # every branch's parameters are judged against every other
        # branch simultaneously.
        sources = [list(branch.parameters) for branch in branch_items]
        source_labels = tuple(
            f"branch {i} ({branch.full_name})"
            for i, branch in enumerate(branch_items)
        )
        reports = build_parameter_reports(sources)
        constructed = apply_parameter_reports(
            reports, source_labels, error_cls=WorkflowError, stacklevel=3
        )
        declared_parameters = insert_by_category([], constructed)

        # 3. result_mode must be one of the known projection shapes.
        normalized_mode = str(result_mode).strip().lower()
        if normalized_mode not in self._OUTPUT_TYPES:
            raise ValueError(
                f"result_mode must be one of {sorted(self._OUTPUT_TYPES)!r}, got {result_mode!r}"
            )

        # 4. selected_branches: None/[] both mean "select nothing" — a
        #    legal state, not an error. Every item must be an in-range,
        #    non-negative int (no wraparound); no duplicates.
        if selected_branches is None:
            selected_branches = []
        if not isinstance(selected_branches, (list, tuple)):
            raise TypeError(
                f"selected_branches must be a list or tuple or None, got {type(selected_branches)!r}"
            )
        for raw in selected_branches:
            if not isinstance(raw, int) or isinstance(raw, bool):
                raise TypeError(
                    f"selected_branches items must be int, got {type(raw)!r}"
                )
            if raw < 0 or raw >= branch_count:
                raise IndexError(
                    f"selected_branches index {raw} out of range for "
                    f"{branch_count} configured branch(es)"
                )
        if len(selected_branches) != len(set(selected_branches)):
            raise ValueError("selected_branches must not select the same branch twice")
        selected_indices: tuple[int, ...] = tuple(selected_branches)

        # 5. SCALAR allows at most one selected branch (zero is legal —
        #    invoke() then returns None).
        if normalized_mode == self.SCALAR and len(selected_indices) > 1:
            raise ValueError("result_mode='scalar' allows at most one selected branch")

        # 6. result_keys: required (and validated) exactly for DICT mode,
        #    forbidden otherwise. Never auto-derived — always explicit.
        #    Either way, self._result_keys ends up as ONE of two things:
        #    the validated label list (DICT), or selected_indices itself
        #    (every other mode) — no third case, no None special-casing.
        if normalized_mode == self.DICT:
            if result_keys is None:
                result_keys = []
            if not isinstance(result_keys, (list, tuple)):
                raise TypeError(
                    f"result_keys must be a list[str] or tuple[str] or None, got {type(result_keys)!r}"
                )
            if len(result_keys) != len(selected_indices):
                raise ValueError(
                    "result_keys must match the length of selected_branches "
                    f"({len(result_keys)} given, {len(selected_indices)} selected)"
                )

            cleaned_keys: list[str] = []
            for index, raw_name in enumerate(result_keys):
                if not isinstance(raw_name, str) or not raw_name.strip():
                    raise ValueError(f"result_keys[{index}] must be a non-empty string")
                cleaned = raw_name.strip()
                if not IDENTIFIER_PATTERN.match(cleaned):
                    raise ValueError(
                        f"result_keys[{index}]={cleaned!r} is not a valid parameter-style name"
                    )
                cleaned_keys.append(cleaned)
            if len(cleaned_keys) != len(set(cleaned_keys)):
                raise ValueError("result_keys must be unique")

            resolved_result_keys: tuple[int, ...] | tuple[str, ...] = tuple(cleaned_keys)
            resolved_return_type = "dict"
        else:
            if result_keys is not None:
                raise ValueError(
                    f"result_keys must be None when result_mode={normalized_mode!r}"
                )
            resolved_result_keys = selected_indices
            if normalized_mode == self.SCALAR:
                # "None" (the type string) mirrors utils/parameters.py's own
                # NoneType -> "None" formatting convention.
                resolved_return_type = (
                    branch_items[selected_indices[0]].return_type if selected_indices else "None"
                )
            else:
                resolved_return_type = normalized_mode

        super().__init__(
            name=name,
            namespace=namespace,
            description=description,
            parameters=declared_parameters,
            return_type=resolved_return_type,
            include_trace=include_trace,
        )

        self._branches: tuple[AtomicInvokable, ...] = branch_items
        self._result_mode: str = normalized_mode
        self._selected_indices: tuple[int, ...] = selected_indices
        self._result_keys: tuple[int, ...] | tuple[str, ...] = resolved_result_keys

    # ------------------------------------------------------------------ #
    # Read-only topology / contract properties
    # ------------------------------------------------------------------ #
    @property
    def branches(self) -> tuple[AtomicInvokable, ...]:
        """Return the fixed branch tuple, declaration order."""
        return self._branches

    @property
    def result_mode(self) -> str:
        """Return the fixed output projection mode."""
        return self._result_mode

    @property
    def selected_indices(self) -> tuple[int, ...]:
        """Return the fixed branch indices selected for projection, in
        projection order. May be empty (no branch selected)."""
        return self._selected_indices

    @property
    def result_keys(self) -> tuple[int, ...] | tuple[str, ...]:
        """Return the fixed projection labels — the single source of truth
        also mirrored onto ``ParallelFlowResult.result_keys``. For DICT
        mode, the validated label strings; for every other mode, exactly
        ``selected_indices`` (same tuple, not a separate value)."""
        return self._result_keys

    def _extra_description(self) -> str:
        """Describe the fixed output projection shape.

        A branch's own extra description is only inlined in the SCALAR case
        with a branch selected, where the result IS that one branch's
        payload verbatim (a true 1:1 passthrough, not an aggregation).
        DICT/LIST/TUPLE/SET describe structural shape facts instead of
        inlining any branch content.
        """
        if self._result_mode == self.SCALAR:
            if not self._selected_indices:
                return "Returns None (no branch selected)."
            return self._branches[self._selected_indices[0]]._extra_description()

        if self._result_mode == self.DICT:
            entries = ", ".join(
                f"{name} ({self._branches[index].return_type})"
                for name, index in zip(self._result_keys, self._selected_indices)
            )
            return f"Returns a dict of {len(self._selected_indices)} keys: {entries}"

        return f"Returns a {self._result_mode} of {len(self._selected_indices)} branch outputs."

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _project_result(self, branch_results: list[AtomicResult]) -> Any:
        """Project executed branch results into the configured outer result."""
        payloads = [branch_results[index].result for index in self._selected_indices]

        if self._result_mode == self.SCALAR:
            return payloads[0] if payloads else None
        if self._result_mode == self.LIST:
            return list(payloads)
        if self._result_mode == self.TUPLE:
            return tuple(payloads)
        if self._result_mode == self.SET:
            return set(payloads)

        # DICT: self._result_keys IS the label list in this mode.
        return dict(zip(self._result_keys, payloads))

    # ------------------------------------------------------------------ #
    # Result construction
    # ------------------------------------------------------------------ #
    def make_result(
        self,
        result: Any,
        started_at: datetime,
        ended_at: datetime,
        **result_kwargs: Any,
    ) -> ParallelFlowResult:
        """Construct a ParallelFlowResult envelope, injecting the fixed
        result_mode/selected_indices/result_keys directly — all constant
        facts about this instance, not per-run data."""
        return self._make_result(
            result=result,
            started_at=started_at,
            ended_at=ended_at,
            result_cls=ParallelFlowResult,
            result_mode=self._result_mode,
            selected_indices=self._selected_indices,
            result_keys=self._result_keys,
            **result_kwargs,
        )

    # ------------------------------------------------------------------ #
    # Workflow run hooks
    # ------------------------------------------------------------------ #
    def _run(self, inputs: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
        """Synchronously execute all configured branches concurrently.

        Sync is the async path bridged, not a second implementation --
        matches ToolAgent.act()'s own run_coro_sync(async gather) pattern.
        Branches without a native async_invoke override already run their
        real .invoke() inside a background thread by default
        (AtomicInvokable.async_invoke), so a separate thread-pool mechanism
        here would just duplicate that.
        """
        return run_coro_sync(self._async_run(inputs))

    async def _async_run(self, inputs: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
        """Asynchronously execute all configured branches concurrently."""
        branch_inputs = [dict(inputs) for _ in self._branches]

        async def run_one(index: int, branch: AtomicInvokable, branch_input: dict[str, Any]) -> AtomicResult:
            try:
                return await branch.async_invoke(branch_input)
            except Exception as exc:
                raise RuntimeError(
                    f"{self.full_name}: async branch {index} ({branch.full_name}) failed during invoke"
                ) from exc

        branch_results = list(
            await asyncio.gather(
                *[
                    run_one(index, branch, branch_inputs[index])
                    for index, branch in enumerate(self._branches)
                ]
            )
        )

        return self._project_result(branch_results), {
            "trace": tuple(branch_results) if self.include_trace else None,
        }

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        """Serialize the fixed branch topology plus output projection."""
        data = super().to_dict()
        data.update(
            {
                "branches": [branch.to_dict() for branch in self._branches],
                "branch_count": len(self._branches),
                "result_mode": self._result_mode,
                "selected_indices": list(self._selected_indices),
                "result_keys": list(self._result_keys),
            }
        )
        return data
