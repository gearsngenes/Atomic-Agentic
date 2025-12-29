from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Mapping, Optional, Union, Tuple, Sequence
from typing import Mapping, Optional
from collections.abc import Sequence as Sequence
from collections import OrderedDict
import threading
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4
from ..core.Exceptions import *
from ..core.Invokable import ArgumentMap, AtomicInvokable

logger = logging.getLogger(__name__)

__all__ = [
    "Workflow",
    "WorkflowCheckpoint",
    "BundlingPolicy",
    "MappingPolicy",
    "AbsentValPolicy",
    "NO_VAL",
    "DEFAULT_WF_KEY"
]

# ───────────────────────────────────────────────────────────────────────────────
# Workflow primitive
# ───────────────────────────────────────────────────────────────────────────────
class _NoValSentinel:
    """Sentinel used to mark required output fields in an output schema template."""
    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover
        return "NO_VAL"

NO_VAL: Any = _NoValSentinel()

class BundlingPolicy(str, Enum):
    """Controls whether raw results are bundled into a single output field."""
    BUNDLE = "BUNDLE"
    UNBUNDLE = "UNBUNDLE"

class MappingPolicy(str, Enum):
    """Controls how mapping-shaped raw outputs are interpreted."""
    STRICT = "STRICT"
    IGNORE_EXTRA = "IGNORE_EXTRA"
    MATCH_FIRST_STRICT = "MATCH_FIRST_STRICT"
    MATCH_FIRST_LENIENT = "MATCH_FIRST_LENIENT"

class AbsentValPolicy(str, Enum):
    """Controls how missing output fields are handled."""
    RAISE = "RAISE"
    DROP = "DROP"
    FILL = "FILL"

@dataclass(frozen=True, slots=True)
class WorkflowCheckpoint:
    """A single workflow invocation record."""
    run_id: str
    started_at: datetime
    ended_at: datetime
    elapsed_s: float
    inputs: Mapping[str, Any]
    raw_output: Any
    packaged_output: OrderedDict[str, Any]
    metadata: Dict[str, Any]

DEFAULT_WF_KEY = "result"

class Workflow(AtomicInvokable, ABC):
    """
    Base template-method primitive for Workflow orchestrators.

    Workflows are a deterministic *packaging boundary*:

    - Inputs: always a Mapping[str, Any]
    - Execution: subclasses implement `_invoke(inputs)` and return
      (metadata: Mapping[str, Any], raw: Any)
    - Outputs: normalized and packaged against an ordered `output_schema`
      template using policy-driven packaging rules.

    IO schemas
    ----------
    - `arguments_map` is REQUIRED at construction time and is the authoritative
      source for `input_schema`.
    - `input_schema` mirrors `output_schema` format: OrderedDict[str, Any] where
      each value is either a default or NO_VAL.
    - Public mutation is disallowed: `arguments_map`, `input_schema`, and
      `output_schema` are read-only properties.
    - Subclasses may refresh schemas via `_set_io_schemas(arguments_map=..., output_schema=...)`
      when internal components change.

    Packaging policies
    ------------------
    - BundlingPolicy.BUNDLE: bundle raw output into the single schema key.
      This policy is ONLY applied when output_schema length == 1. Otherwise,
      it is ignored and packaging proceeds in UNBUNDLE mode.
    - BundlingPolicy.UNBUNDLE: attempt to coerce raw output into a Mapping or
      non-text Sequence; then apply MappingPolicy/sequence/scalar rules.

    Validation
    ----------
    Final validation that *no* keys remain set to NO_VAL is performed in `invoke()`,
    not inside the packaging helpers.

    Memory
    ------
    `clear_memory()` clears workflow checkpoints. Subclasses may extend via
    polymorphism.
    """

    def __init__(
        self,
        name: str,
        description: str,
        *,
        output_schema: Optional[Union[List[str], Mapping[str, Any]]] = None,
        bundling_policy: BundlingPolicy = BundlingPolicy.BUNDLE,
        mapping_policy: MappingPolicy = MappingPolicy.STRICT,
        absent_val_policy: AbsentValPolicy = AbsentValPolicy.RAISE,
        default_absent_val: Any = None,
    ) -> None:

        # initialize name, description, arguments-map, return-type
        super().__init__(name = name, description = description)

        # packaging policies
        self._bundling_policy = BundlingPolicy(bundling_policy)
        self._mapping_policy = MappingPolicy(mapping_policy)
        self._absent_val_policy = AbsentValPolicy(absent_val_policy)
        self._default_absent_val = default_absent_val

        # invoke thread lock
        self._invoke_lock = threading.RLock()
        
        # checkpoints
        self._checkpoints: List[WorkflowCheckpoint] = []

        # initialize input/output schemas
        if output_schema is None: output_schema = [DEFAULT_WF_KEY]
        self._normalize_schemas(self.arguments_map, output_schema)

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #
    @property
    def input_schema(self) -> OrderedDict[str, Any]:
        return OrderedDict(self._input_schema)

    @property
    def output_schema(self) -> OrderedDict[str, Any]:
        return OrderedDict(self._output_schema)

    @output_schema.setter
    def output_schema(self, value: Optional[Union[List[str], Mapping[str, Any]]]) -> None:
        if value is None: value = [DEFAULT_WF_KEY]
        self._normalize_schemas(self.arguments_map, value)

    @property
    def bundling_policy(self) -> BundlingPolicy:
        return self._bundling_policy

    @bundling_policy.setter
    def bundling_policy(self, value: BundlingPolicy) -> None:
        self._bundling_policy = BundlingPolicy(value)

    @property
    def mapping_policy(self) -> MappingPolicy:
        return self._mapping_policy

    @mapping_policy.setter
    def mapping_policy(self, value: MappingPolicy) -> None:
        self._mapping_policy = MappingPolicy(value)

    @property
    def absent_val_policy(self) -> AbsentValPolicy:
        return self._absent_val_policy

    @absent_val_policy.setter
    def absent_val_policy(self, value: AbsentValPolicy) -> None:
        self._absent_val_policy = AbsentValPolicy(value)
    
    @property
    def default_absent_val(self) -> Any:
        return self._default_absent_val

    @default_absent_val.setter
    def default_absent_val(self, value: Any) -> None:
        self._default_absent_val = value

    @property
    def checkpoints(self) -> List[WorkflowCheckpoint]:
        return list(self._checkpoints)

    @property
    def latest_checkpoint(self) -> Optional[WorkflowCheckpoint]:
        return self._checkpoints[-1] if self._checkpoints else None

    # ------------------------------------------------------------------ #
    # Public API (Template Method)
    # ------------------------------------------------------------------ #
    def invoke(self, inputs: Mapping[str, Any]) -> OrderedDict[str, Any]:
        """
        Run the invoke method
        """
        # 1) validate is mapping
        if not isinstance(inputs, Mapping):
            raise ValidationError("Workflow.invoke: inputs must be a mapping")

        with self._invoke_lock:
            started = datetime.now(timezone.utc)
            run_id = uuid4().hex

            # 2) run _invoke()
            try:
                metadata, raw = self._invoke(inputs)
            except Exception as exc:
                raise ExecutionError(f"{type(self).__name__}._invoke failed") from exc

            if not isinstance(metadata, Mapping):
                raise ValidationError("Workflow._invoke: returned metadata must be a mapping")

            metadata = dict(metadata) # snapshot metadata to avoid external mutation
            
            # 3) package raw result
            packaged = self.package(raw)

            # 4) validate that no NO_VAL's are present (drop, fill, or raise)
            packaged, missing_key_info = self._validate_packaged(packaged)

            # 5) update checkpoints
            ended = datetime.now(timezone.utc)
            elapsed = (ended - started).total_seconds()
            metadata.update(missing_key_info)
            checkpoint = WorkflowCheckpoint(
                run_id=run_id,
                started_at=started,
                ended_at=ended,
                elapsed_s=elapsed,
                inputs=dict(inputs),
                raw_output=raw,
                packaged_output=OrderedDict(packaged),
                metadata=metadata,
            )
            self._checkpoints.append(checkpoint)

            # 6) return
            return packaged

    @abstractmethod
    def _invoke(self, inputs: Mapping[str, Any]) -> Tuple[Mapping[str, Any], Any]:
        """Subclass-defined execution step; returns (metadata, raw_result)."""
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Memory
    # ------------------------------------------------------------------ #
    def clear_memory(self) -> None:
        """Clear workflow-owned memory (checkpoints). Subclasses may extend."""
        self._checkpoints.clear()

    # ------------------------------------------------------------------ #
    # Final validation (kept separate from packaging)
    # ------------------------------------------------------------------ #
    def _validate_packaged(self, packaged: Mapping[str, Any]) -> Tuple[Mapping[str, Any], Dict[str, Any]]:
        """
        Raise if any output fields remain NO_VAL after packaging.

        This method is intentionally called by `invoke()` and not by `package()`
        to keep packaging responsibilities separate from final contract validation.
        """
        # Find any missing keys
        missing = [k for k, v in packaged.items() if v is NO_VAL]
        # Return as-is if no keys are missing
        if not missing:
            return packaged, {}
        # Raise an error if policy is RAISE
        if self._absent_val_policy == AbsentValPolicy.RAISE:
            raise PackagingError(f"Workflow packaging: missing required output fields: {missing}")

        new_packaged = OrderedDict()
        # Fill in missing values if policy is FILL
        if missing and self._absent_val_policy == AbsentValPolicy.FILL:
            for k in self.output_schema.keys():
                new_packaged[k] = packaged.get(k) if packaged.get(k) is not NO_VAL else self._default_absent_val
        # Drop missing values if policy is DROP
        elif missing and self._absent_val_policy == AbsentValPolicy.DROP:
            for k,v in packaged.items():
                if v is not NO_VAL:
                    new_packaged[k] = v
        meta_data = {"keys_filled": missing} if missing and self._absent_val_policy == AbsentValPolicy.FILL else {"keys_dropped": missing}
        # Return finalized dictionary
        return new_packaged, meta_data

    # ------------------------------------------------------------------ #
    # Packaging
    # ------------------------------------------------------------------ #
    def package(self, raw: Any) -> OrderedDict[str, Any]:
        """
        Package a raw result into the workflow's output schema.

        IMPORTANT: This method does not perform final NO_VAL validation.
        That validation is performed by `invoke()` via `_validate_packaged()`.
        """
        keys = list(self._output_schema.keys())
        if not keys:
            return OrderedDict() # If no keys, return an empty OrderedDict

        # Bundling is ONLY considered when schema length == 1.
        if self._bundling_policy == BundlingPolicy.BUNDLE and len(keys) == 1:
            return OrderedDict([(keys[0], raw)])

        # UNBUNDLE flow (also used when BUNDLE is set but schema length != 1)
        normalized = self._normalize_raw(raw)
        template = OrderedDict(self._output_schema)

        if isinstance(normalized, Mapping):
            return self._package_from_mapping(template, normalized)
        if self._is_non_text_sequence(normalized):
            return self._package_from_sequence(template, normalized)  # type: ignore[arg-type]
        return self._package_from_scalar(template, normalized)

    def _normalize_raw(self, raw: Any) -> Any:
        if isinstance(raw, Mapping) or self._is_non_text_sequence(raw):
            return raw

        # Pydantic v2 models: model_dump()
        md = getattr(raw, "model_dump", None)
        if callable(md):
            try:
                dumped = md()
                if isinstance(dumped, Mapping):
                    return dumped
            except Exception:
                pass

        # NamedTuple: _asdict()
        ad = getattr(raw, "_asdict", None)
        if callable(ad):
            try:
                dumped = ad()
                if isinstance(dumped, Mapping):
                    return dumped
            except Exception:
                pass

        # Dataclass
        if is_dataclass(raw):
            try:
                dumped = asdict(raw)
                if isinstance(dumped, Mapping):
                    return dumped
            except Exception:
                pass

        # Plain object with __dict__
        try:
            dumped = vars(raw)
            if isinstance(dumped, Mapping):
                return dumped
        except Exception:
            pass

        return raw

    def _is_non_text_sequence(self, x: Any) -> bool:
        return isinstance(x, Sequence) and not isinstance(x, (str, bytes, bytearray))

    def _package_from_mapping(
        self,
        template: OrderedDict[str, Any],
        mapping: Mapping[str, Any],
    ) -> OrderedDict[str, Any]:
        schema_keys = list(template.keys())

        if self._mapping_policy == MappingPolicy.STRICT:
            for k in mapping.keys():
                if k not in template:
                    raise PackagingError(f"Workflow packaging (STRICT): unexpected key {k!r}")
            for k in schema_keys:
                if k in mapping:
                    template[k] = mapping[k]
            return template

        if self._mapping_policy == MappingPolicy.IGNORE_EXTRA:
            for k, v in mapping.items():
                if k in template:
                    template[k] = v
            return template

        if self._mapping_policy in (MappingPolicy.MATCH_FIRST_STRICT, MappingPolicy.MATCH_FIRST_LENIENT):
            if self._mapping_policy == MappingPolicy.MATCH_FIRST_STRICT:
                if len(mapping) > len(schema_keys):
                    raise PackagingError(
                        "Workflow packaging (MATCH_FIRST_STRICT): mapping has more keys "
                        f"({len(mapping)}) than output_schema ({len(schema_keys)})"
                    )

            extras_values: List[Any] = []
            for k, v in mapping.items():
                if k in template:
                    template[k] = v
                else:
                    extras_values.append(v)

            missing_keys = [k for k in schema_keys if template[k] is NO_VAL]

            # If already filled, strict rejects any extra values; lenient ignores them.
            if not missing_keys:
                if extras_values and self._mapping_policy == MappingPolicy.MATCH_FIRST_STRICT:
                    raise PackagingError(
                        "Workflow packaging (MATCH_FIRST_STRICT): output_schema already filled "
                        f"but received {len(extras_values)} extra values"
                    )
                return template

            # Too many extras: strict errors; lenient truncates.
            if len(extras_values) > len(missing_keys):
                if self._mapping_policy == MappingPolicy.MATCH_FIRST_LENIENT:
                    extras_values = extras_values[: len(missing_keys)]
                else:
                    raise PackagingError(
                        "Workflow packaging (MATCH_FIRST_*): too many extra values "
                        f"({len(extras_values)}) for remaining schema slots ({len(missing_keys)})"
                    )

            for idx, v in enumerate(extras_values):
                template[missing_keys[idx]] = v

            return template

        raise PackagingError(f"Unknown mapping policy: {self._mapping_policy!r}")

    def _package_from_sequence(
        self,
        template: OrderedDict[str, Any],
        seq: Sequence[Any],
    ) -> OrderedDict[str, Any]:
        keys = list(template.keys())
        values = list(seq)

        # If the sequence fits, positional-fill and return.
        if len(values) <= len(keys):
            for i, v in enumerate(values):
                template[keys[i]] = v
            return template

        # Overflow: policy decides whether we truncate or raise.
        if self._mapping_policy in (MappingPolicy.IGNORE_EXTRA, MappingPolicy.MATCH_FIRST_LENIENT):
            # Ignore extras by truncating to schema length.
            for i, k in enumerate(keys):
                template[k] = values[i]
            return template

        # STRICT and MATCH_FIRST_STRICT both reject overflow for sequences.
        raise PackagingError(
            "Workflow packaging (SEQUENCE): too many values "
            f"({len(values)}) for output_schema ({len(keys)}) under policy {self._mapping_policy.value}"
        )

    def _package_from_scalar(
        self,
        template: OrderedDict[str, Any],
        value: Any,
    ) -> OrderedDict[str, Any]:
        keys = list(template.keys())
        template[keys[0]] = value
        return template

    # ------------------------------------------------------------------ #
    # Schema helpers
    # ------------------------------------------------------------------ #
    def build_args_returns(self) -> Tuple[ArgumentMap, str]:
        return self._get_arguments(), OrderedDict.__name__

    @abstractmethod
    def _get_arguments(self)-> ArgumentMap:
        raise NotImplementedError

    def _normalize_schemas(self,
                           arguments_map: ArgumentMap,
                           output_schema: Union[List[str], Mapping[str, Any]]) -> None:
        """
        Normalize and set:
          - _arguments_map (ordered, meta dicts copied)
          - _input_schema  (defaults from arguments_map else NO_VAL)
          - _output_schema (normalized output template)

        Intentionally protected: callers should be Workflow subclasses.
        """
        # normalize input-schema from an arguments map
        normalized_input_schema = OrderedDict()
        for key, meta in arguments_map.items():
            normalized_input_schema.update({key : meta.get("default", NO_VAL)})

        # normalize output-schema from a mapping or list
        if isinstance(output_schema, Mapping):
            normalized_output_schema = OrderedDict(output_schema)
        else:
            normalized_output_schema = OrderedDict((key, NO_VAL) for key in output_schema)

        # copy argument map (shallow copy of meta dicts) so schemas stay consistent
        self._input_schema = normalized_input_schema
        self._output_schema = normalized_output_schema

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d.update(OrderedDict(
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            bundling_policy=self._bundling_policy.value,
            mapping_policy=self._mapping_policy.value,
            absent_val_policy=self._absent_val_policy.value,
            default_absent_val=self._default_absent_val,
        ))
        return d