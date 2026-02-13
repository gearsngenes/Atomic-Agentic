from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Mapping, Optional, Union, Tuple, Sequence, get_type_hints
from collections.abc import Sequence as Sequence
import threading
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4
from ..core.Exceptions import *
from ..core.Invokable import AtomicInvokable
from ..core.Parameters import ParamSpec
from ..core.sentinels import NO_VAL

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
# Output Schema Normalization
# ───────────────────────────────────────────────────────────────────────────────
def _is_typed_dict_class(obj: Any) -> bool:
    """
    Runtime check for TypedDict classes.
    TypedDict classes have .__annotations__ and .__total__ attributes.
    """
    return isinstance(obj, type) and issubclass(obj, dict) and hasattr(obj, "__annotations__") and hasattr(obj, "__total__")


def _normalize_output_schema(
    schema: Union[type, List[Union[str, ParamSpec]], Mapping[str, Any]],
) -> list[ParamSpec]:
    """
    Normalize output schema to list[ParamSpec].
    
    Accepts:
    - TypedDict class: extract field names from __annotations__, create POSITIONAL_OR_KEYWORD ParamSpec
    - List[str]: convert each string to ParamSpec with NO_VAL default
    - List[ParamSpec]: validate and return as-is
    - List[Union[str, ParamSpec]]: mix of above, normalize all
    - Mapping[str, Any]: old dict format, convert keys to ParamSpec list
    
    Does NOT validate ordering (delegated to AtomicInvokable).
    
    Raises
    ------
    SchemaError
        If schema format is unsupported.
    """
    
    # TypedDict class: extract __annotations__
    if _is_typed_dict_class(schema):
        field_names = list(get_type_hints(schema).keys())
        result = []
        for index, name in enumerate(field_names):
            result.append(ParamSpec(
                name=name,
                index=index,
                kind="POSITIONAL_OR_KEYWORD",
                type="Any",
                default=NO_VAL
            ))
        return result
    
    # List variant: str, ParamSpec, or mixed
    if isinstance(schema, list):
        result = []
        for index, item in enumerate(schema):
            if isinstance(item, str):
                # Convert string to ParamSpec
                result.append(ParamSpec(
                    name=item,
                    index=index,
                    kind="POSITIONAL_OR_KEYWORD",
                    type="Any",
                    default=NO_VAL
                ))
            elif isinstance(item, ParamSpec):
                result.append(item)
            else:
                raise SchemaError(
                    f"Schema list items must be str or ParamSpec, got {type(item).__name__} at index {index}"
                )
        return result
    
    # Dict variant: convert keys to list
    if isinstance(schema, Mapping):
        result = []
        for index, (name, default) in enumerate(schema.items()):
            if not isinstance(name, str):
                raise SchemaError(f"Schema dict keys must be strings, got {type(name).__name__}")
            result.append(ParamSpec(
                name=name,
                index=index,
                kind="POSITIONAL_OR_KEYWORD",
                type="Any",
                default=default
            ))
        return result
    
    raise SchemaError(
        f"output_schema must be TypedDict class, list[str|ParamSpec], or Mapping; got {type(schema).__name__}"
    )


# ───────────────────────────────────────────────────────────────────────────────
# Workflow primitive
# ───────────────────────────────────────────────────────────────────────────────

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
    packaged_output: Dict[str, Any]
    metadata: Dict[str, Any]

DEFAULT_WF_KEY = "<<result>>"

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
    - `parameters` is REQUIRED at construction time and is the authoritative
      source for `input_schema`.
    - `input_schema` mirrors `output_schema` format: Dict[str, Any] where
      each value is either a default or NO_VAL.
    - Public mutation is disallowed: `parameters` and `return_type` are read-only properties.

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

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        name: str,
        description: str,
        parameters: list[ParamSpec],
        *,
        output_schema: Optional[Union[type, List[Union[str, ParamSpec]], Mapping[str, Any]]] = None,
        bundling_policy: BundlingPolicy = BundlingPolicy.BUNDLE,
        mapping_policy: MappingPolicy = MappingPolicy.STRICT,
        absent_val_policy: AbsentValPolicy = AbsentValPolicy.RAISE,
        default_absent_val: Any = None,
    ) -> None:
        # Pass parameters and return_type to parent (AtomicInvokable)
        # Parent will validate parameter ordering and structure
        super().__init__(
            name=name,
            description=description,
            parameters=parameters,
            return_type="Dict[str, Any]",  # Workflows always return mappings
        )
        
        # Normalize output schema to list[ParamSpec] (primary storage)
        if output_schema is None:
            output_schema = [DEFAULT_WF_KEY]
        self._output_schema: list[ParamSpec] = _normalize_output_schema(output_schema)

        # Packaging policies
        self._bundling_policy = BundlingPolicy(bundling_policy)
        self._mapping_policy = MappingPolicy(mapping_policy)
        self._absent_val_policy = AbsentValPolicy(absent_val_policy)
        self._default_absent_val = default_absent_val

        # Invoke thread lock
        self._invoke_lock = threading.RLock()
        
        # Checkpoints
        self._checkpoints: List[WorkflowCheckpoint] = []

    # ------------------------------------------------------------------ #
    # Workflow Properties
    # ------------------------------------------------------------------ #
    @property
    def output_schema(self) -> list[ParamSpec]:
        """Output schema as list of ParamSpec (primary API)."""
        return list(self._output_schema)

    @output_schema.setter
    def output_schema(self, value: Optional[Union[type, List[Union[str, ParamSpec]], Mapping[str, Any]]]) -> None:
        if value is None:
            value = [DEFAULT_WF_KEY]
        self._output_schema = _normalize_output_schema(value)

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
    # Workflow Helpers
    # ------------------------------------------------------------------ #

    #       Runtime invoke helper
    # ------------------------------------------------------------------ #
    @abstractmethod
    def _invoke(self, inputs: Mapping[str, Any]) -> Tuple[Mapping[str, Any], Any]:
        """Subclass-defined execution step; returns (metadata, raw_result)."""
        raise NotImplementedError

    #       validating output before returning
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

        new_packaged = {}
        # Fill in missing values if policy is FILL
        if missing and self._absent_val_policy == AbsentValPolicy.FILL:
            for k in self.output_schema:
                new_packaged[k.name] = packaged.get(k.name) if packaged.get(k.name) is not NO_VAL else self._default_absent_val
        # Drop missing values if policy is DROP
        elif missing and self._absent_val_policy == AbsentValPolicy.DROP:
            for k,v in packaged.items():
                if v is not NO_VAL:
                    new_packaged[k] = v
        meta_data = {"keys_filled": missing} if missing and self._absent_val_policy == AbsentValPolicy.FILL else {"keys_dropped": missing}
        # Return finalized dictionary
        return new_packaged, meta_data


    #       Packaging helpers
    # ------------------------------------------------------------------ #
    def package(self, raw: Any) -> Dict[str, Any]:
        """
        Package a raw result into the workflow's output schema.

        IMPORTANT: This method does not perform final NO_VAL validation.
        That validation is performed by `invoke()` via `_validate_packaged()`.
        """
        # Retrun empty dict if no output schema
        if not len(self._output_schema):
            return {}  # If no fields, return empty dict

        # Unwrap single-key dicts with DEFAULT_WF_KEY
        unwrapped = raw
        while (
            isinstance(unwrapped, Mapping) 
            and len(unwrapped) == 1 
            and list(unwrapped.keys())[0] == [DEFAULT_WF_KEY]
        ):
            unwrapped = unwrapped[DEFAULT_WF_KEY]

        # Bundle if policy set to bundle and schema length == 1
        if self._bundling_policy == BundlingPolicy.BUNDLE and len(self._output_schema) == 1:
            return {self._output_schema[0].name: unwrapped}

        # Normalize raw output to sequence, mapping, or scalar
        normalized = self._normalize_raw(unwrapped)

        # Create template dict from output schema (field_name -> default value)
        template = {spec.name: spec.default for spec in self._output_schema}

        if isinstance(normalized, Mapping):
            return self._package_from_mapping(template, normalized)
        if isinstance(normalized, Sequence) and not isinstance(normalized, (str, bytes, bytearray)):
            return self._package_from_sequence(template, normalized)  # type: ignore[arg-type]
        return self._package_from_scalar(template, normalized)

    def _normalize_raw(self, raw: Any) -> Any:
        if isinstance(raw, Mapping) or isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
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

    def _package_from_mapping(
        self,
        template: Dict[str, Any],
        mapping: Mapping[str, Any],
    ) -> Dict[str, Any]:
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
            # Phase 1: Match by key name - fill template keys that exist in mapping
            extras_values: List[Any] = []
            for k, v in mapping.items():
                if k in template:
                    template[k] = v
                else:
                    extras_values.append(v)

            # Phase 2: Identify remaining missing keys (unfilled schema slots)
            missing_keys = [k for k in schema_keys if template[k] is NO_VAL]

            # Phase 3: If all schema keys are filled, check overflow handling
            if not missing_keys:
                if extras_values and self._mapping_policy == MappingPolicy.MATCH_FIRST_STRICT:
                    raise PackagingError(
                        "Workflow packaging (MATCH_FIRST_STRICT): output_schema already filled "
                        f"but received {len(extras_values)} extra values"
                    )
                return template

            # Phase 4: Fill remaining schema slots positionally from extras
            # If too many extras: STRICT errors, LENIENT truncates
            if len(extras_values) > len(missing_keys):
                if self._mapping_policy == MappingPolicy.MATCH_FIRST_LENIENT:
                    extras_values = extras_values[: len(missing_keys)]
                else:
                    raise PackagingError(
                        "Workflow packaging (MATCH_FIRST_STRICT): too many extra values "
                        f"({len(extras_values)}) for remaining schema slots ({len(missing_keys)})"
                    )

            for idx, v in enumerate(extras_values):
                template[missing_keys[idx]] = v

            return template

        raise PackagingError(f"Unknown mapping policy: {self._mapping_policy!r}")

    def _package_from_sequence(
        self,
        template: Dict[str, Any],
        seq: Sequence[Any],
    ) -> Dict[str, Any]:
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
        template: Dict[str, Any],
        value: Any,
    ) -> Dict[str, Any]:
        keys = list(template.keys())
        template[keys[0]] = value
        return template

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def invoke(self, inputs: Mapping[str, Any]) -> Dict[str, Any]:
        """
        Run the invoke method
        """
        logger.info(f"[{type(self).__name__}.{self.name}.invoke started]")
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
                packaged_output=dict(packaged),
                metadata=metadata,
            )
            self._checkpoints.append(checkpoint)

            logger.info(f"[{type(self).__name__}.{self.name} finished]")

            # 6) return
            return packaged

    def clear_memory(self) -> None:
        """Clear workflow-owned memory (checkpoints). Subclasses may extend."""
        self._checkpoints.clear()

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d.update({
            "input_schema" : self.input_schema,
            "output_schema" : self.output_schema,
            "bundling_policy": self.bundling_policy.value,
            "mapping_policy": self.mapping_policy.value,
            "absent_val_policy": self.absent_val_policy.value,
            "default_absent_val": self.default_absent_val})
        return d