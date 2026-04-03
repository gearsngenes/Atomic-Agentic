from __future__ import annotations

from collections.abc import Mapping
import logging
from typing import Any, Optional, Sequence

from ..core.Exceptions import PackagingError
from ..core.Invokable import AtomicInvokable
from ..core.Parameters import ParamSpec, is_valid_parameter_order, to_paramspec_list
from ..core.sentinels import NO_VAL

logger = logging.getLogger(__name__)

__all__ = ["StructuredInvokable"]


class StructuredInvokable(AtomicInvokable):
    """Wrap an invokable and package its raw output into a mapping."""

    def __init__(
        self,
        component: AtomicInvokable,
        name: Optional[str] = None,
        description: Optional[str] = None,
        *,
        output_schema: Optional[
            type | list[str] | tuple[str, ...] | set[str] | list[ParamSpec]
        ] = None,
        map_single_fields: bool = True,
        map_extras: bool = True,
        ignore_unhandled: bool = False,
        absent_value_mode: str = "RAISE",
        default_absent_value: Any = None,
        none_is_absent: bool = False,
        coerce_to_collection: bool = False,
        filter_extraneous_inputs: Optional[bool] = None,
    ) -> None:
        """Initialize a structured-output wrapper around an invokable."""
        if not isinstance(component, AtomicInvokable):
            raise TypeError(
                f"component must be an AtomicInvokable, got {type(component)!r}"
            )

        resolved_filter = (
            filter_extraneous_inputs
            if filter_extraneous_inputs is not None
            else component.filter_extraneous_inputs
        )

        # Store the wrapped component before any downstream property usage.
        self._component = component

        # Delegate the core invokable contract to AtomicInvokable.
        # - inputs mirror the wrapped component's parameters
        # - return type is always dictionary-shaped for StructuredInvokable
        super().__init__(
            name=name or component.name,
            description=description or component.description,
            parameters=component.parameters,
            return_type="Dict[str, Any]",
            filter_extraneous_inputs=resolved_filter,
        )

        # Packaging contract and policy knobs.
        # These setters are expected to validate and normalize their inputs.
        self.output_schema = output_schema
        self.map_single_fields = map_single_fields
        self.map_extras = map_extras
        self.ignore_unhandled = ignore_unhandled
        self.absent_value_mode = absent_value_mode
        self.default_absent_value = default_absent_value
        self.none_is_absent = none_is_absent
        self.coerce_to_collection = coerce_to_collection

    @property
    def component(self) -> AtomicInvokable:
        """The wrapped component."""
        return self._component

    @property
    def description(self) -> str:
        """The seed description plus a schema summary."""
        parts: list[str] = []

        for spec in self._output_schema:
            if spec.kind == "VAR_POSITIONAL":
                parts.append(f"*{spec.name}")
            elif spec.kind == "VAR_KEYWORD":
                parts.append(f"**{spec.name}")
            else:
                part = spec.name
                if spec.default is not NO_VAL:
                    part += f"={spec.default!r}"
                parts.append(part)

        schema_summary = ", ".join(parts) if parts else "<empty>"
        return f"{self._description}\nOutput schema: [{schema_summary}]"

    @description.setter
    def description(self, value: str) -> None:
        """Set the seed description."""
        if not isinstance(value, str):
            raise TypeError(
                f"description must be a string, got {type(value).__name__}"
            )
        if not value.strip():
            raise ValueError("description cannot be empty")
        self._description = value.strip()

    @property
    def output_schema(self) -> list[ParamSpec]:
        """The normalized output schema."""
        return list(self._output_schema)

    @output_schema.setter
    def output_schema(
        self,
        value: Optional[type | list[str] | tuple[str, ...] | set[str] | list[ParamSpec]],
    ) -> None:
        """Normalize, validate, and set the output schema."""
        normalized = to_paramspec_list(value)
        is_valid_parameter_order(normalized)
        self._output_schema = normalized

    @property
    def named_output_fields(self) -> list[ParamSpec]:
        """The non-variadic output fields."""
        return [
            spec
            for spec in self._output_schema
            if spec.kind not in {"VAR_POSITIONAL", "VAR_KEYWORD"}
        ]

    @property
    def output_vararg(self) -> Optional[str]:
        """The output vararg field name."""
        spec = next(
            (item for item in self._output_schema if item.kind == "VAR_POSITIONAL"),
            None,
        )
        return spec.name if spec is not None else None

    @property
    def output_varkwarg(self) -> Optional[str]:
        """The output varkwarg field name."""
        spec = next(
            (item for item in self._output_schema if item.kind == "VAR_KEYWORD"),
            None,
        )
        return spec.name if spec is not None else None

    @property
    def output_has_varargs(self) -> bool:
        """Whether the output schema has a vararg sink."""
        return self.output_vararg is not None

    @property
    def output_has_varkwargs(self) -> bool:
        """Whether the output schema has a varkwarg sink."""
        return self.output_varkwarg is not None

    @property
    def map_single_fields(self) -> bool:
        """Whether single fields may map collection-shaped raw outputs."""
        return self._map_single_fields

    @map_single_fields.setter
    def map_single_fields(self, value: bool) -> None:
        """Set the single-field mapping mode."""
        if not isinstance(value, bool):
            raise TypeError(
                f"map_single_fields must be a bool, got {type(value).__name__}"
            )
        self._map_single_fields = value

    @property
    def map_extras(self) -> bool:
        """Whether extras may backfill missing named fields first."""
        return self._map_extras

    @map_extras.setter
    def map_extras(self, value: bool) -> None:
        """Set the extras-mapping mode."""
        if not isinstance(value, bool):
            raise TypeError(
                f"map_extras must be a bool, got {type(value).__name__}"
            )
        self._map_extras = value

    @property
    def absent_value_mode(self) -> str:
        """The missing-value policy."""
        return self._absent_value_mode

    @absent_value_mode.setter
    def absent_value_mode(self, value: str) -> None:
        """Validate and set the missing-value policy."""
        if not isinstance(value, str):
            raise TypeError(
                f"absent_value_mode must be a string, got {type(value).__name__}"
            )

        normalized = value.strip().upper()
        if normalized not in {"RAISE", "DROP", "FILL"}:
            raise ValueError(
                "absent_value_mode must be one of: 'RAISE', 'DROP', 'FILL'"
            )

        self._absent_value_mode = normalized

    @property
    def default_absent_value(self) -> Any:
        """The fill value for missing fields."""
        return self._default_absent_value

    @default_absent_value.setter
    def default_absent_value(self, value: Any) -> None:
        """Set the default absent value."""
        self._default_absent_value = value

    @property
    def none_is_absent(self) -> bool:
        """Whether ``None`` is treated as absent."""
        return self._none_is_absent

    @none_is_absent.setter
    def none_is_absent(self, value: bool) -> None:
        """Set whether ``None`` is treated as absent."""
        if not isinstance(value, bool):
            raise TypeError(
                f"none_is_absent must be a bool, got {type(value).__name__}"
            )
        self._none_is_absent = value

    @property
    def coerce_to_collection(self) -> bool:
        """Whether object-like outputs may be coerced to collections."""
        return self._coerce_to_collection

    @coerce_to_collection.setter
    def coerce_to_collection(self, value: bool) -> None:
        """Set the collection-coercion mode."""
        if not isinstance(value, bool):
            raise TypeError(
                f"coerce_to_collection must be a bool, got {type(value).__name__}"
            )
        self._coerce_to_collection = value

    @property
    def ignore_unhandled(self) -> bool:
        """Whether unsinkable extras may be silently dropped."""
        return self._ignore_unhandled

    @ignore_unhandled.setter
    def ignore_unhandled(self, value: bool) -> None:
        """Set whether unsinkable extras may be silently dropped."""
        if not isinstance(value, bool):
            raise TypeError(
                f"ignore_unhandled must be a bool, got {type(value).__name__}"
            )
        self._ignore_unhandled = value

    def invoke(self, inputs: Mapping[str, Any]) -> dict[str, Any]:
        """Synchronously invoke the wrapped component and return a packaged mapping.

        This method is the sync execution boundary for ``StructuredInvokable``.
        It preserves the standard dict-first invocation contract inherited from
        :class:`AtomicInvokable` by first normalizing/filtering the provided
        ``inputs`` through :meth:`filter_inputs`. The filtered mapping is then
        passed unchanged to the wrapped component's synchronous
        :meth:`component.invoke` method.

        After the wrapped component returns a raw result, this method delegates
        all output-shaping concerns to the structured packaging pipeline:
        :meth:`package` is responsible for turning the raw value into a
        dictionary-shaped handoff aligned with the current normalized
        ``output_schema``, and :meth:`handle_missing_values` applies the final
        missing-value remediation policy (for example raise, drop, or fill).

        The public sync path is intentionally thin. It does not implement any
        packaging rules itself, and it does not special-case particular raw
        result shapes. Those responsibilities belong exclusively to the
        downstream packaging helpers so that sync and async invocation paths
        remain behaviorally identical apart from how the wrapped component is
        executed.

        This method acquires ``self._invoke_lock`` for the duration of the sync
        call, matching the library's existing sync invocation pattern for other
        invokable primitives.

        Parameters
        ----------
        inputs : Mapping[str, Any]
            Dict-first input payload for the wrapped component.

        Returns
        -------
        dict[str, Any]
            Final packaged output after schema-driven packaging and missing-value
            handling have both been applied.
        """
        with self._invoke_lock:
            logger.info(f"[{self.full_name} started]")

            filtered_inputs = self.filter_inputs(inputs)
            raw_result = self.component.invoke(filtered_inputs)
            packaged = self.package(raw_result)
            final_output = self.handle_missing_values(packaged)

            logger.info(f"[{self.full_name} finished]")
            return final_output

    async def async_invoke(self, inputs: Mapping[str, Any]) -> dict[str, Any]:
        """Asynchronously invoke the wrapped component and return a packaged mapping.

        This method is the async analog of :meth:`invoke`. It preserves the same
        dict-first contract and the same packaging semantics while changing only
        the execution path used to obtain the wrapped component's raw result.
        Inputs are first normalized and filtered through :meth:`filter_inputs`,
        then passed to :meth:`component.async_invoke`.

        Once the awaited raw result is available, packaging remains fully
        synchronous: :meth:`package` converts the raw value into the normalized
        mapping handoff defined by the current ``output_schema``, and
        :meth:`handle_missing_values` applies the configured final missing-value
        policy. This keeps the async path behaviorally aligned with the sync path
        and ensures that output-structuring logic lives in one place only.

        The method intentionally delegates all result-shaping behavior to the
        packaging helpers rather than duplicating policy logic inline. The only
        semantic difference from the sync path is that wrapped component
        execution occurs through ``await self.component.async_invoke(...)``.

        Parameters
        ----------
        inputs : Mapping[str, Any]
            Dict-first input payload for the wrapped component.

        Returns
        -------
        dict[str, Any]
            Final packaged output after schema-driven packaging and missing-value
            handling have both been applied.
        """
        logger.info(f"[Async {self.full_name} started]")

        filtered_inputs = self.filter_inputs(inputs)
        raw_result = await self.component.async_invoke(filtered_inputs)
        packaged = self.package(raw_result)
        final_output = self.handle_missing_values(packaged)

        logger.info(f"[Async {self.full_name} finished]")
        return final_output

    def package(self, raw: Any) -> dict[str, Any]:
        """Package a raw result into the normalized mapping output.

        This method converts the wrapped component's raw output into a dictionary
        shaped by the current normalized ``output_schema`` and the active
        packaging-policy knobs.

        Packaging responsibilities
        --------------------------
        - Resolve named output fields plus any declared variadic sinks.
        - Optionally coerce object-like raw values into mapping/sequence form.
        - Package the raw value in one of four modes:
        whole-value single-field, mapping, sequence, or scalar.
        - Optionally backfill missing named fields from ordinary mapping extras.
        - Apply late ``none_is_absent`` normalization to named fields.
        - Route any remaining extras into declared variadic sinks, or drop/raise
        according to ``ignore_unhandled``.

        Notes
        -----
        ``package()`` may return named fields whose values are still ``NO_VAL``.
        Final raise/drop/fill remediation is handled later by
        :meth:`handle_missing_values`.
        """
        # ------------------------------------------------------------------
        # Step 1: Snapshot the normalized output contract.
        #
        # - named_fields: ordinary non-variadic output fields
        # - output_vararg: declared VAR_POSITIONAL sink name, if any
        # - output_varkwarg: declared VAR_KEYWORD sink name, if any
        #
        # Variadic sinks are only emitted if they actually receive values.
        # ------------------------------------------------------------------
        schema = self.output_schema
        if not schema:
            return {}

        named_fields = [
            spec for spec in schema
            if spec.kind not in {"VAR_POSITIONAL", "VAR_KEYWORD"}
        ]
        output_vararg = next(
            (spec.name for spec in schema if spec.kind == "VAR_POSITIONAL"),
            None,
        )
        output_varkwarg = next(
            (spec.name for spec in schema if spec.kind == "VAR_KEYWORD"),
            None,
        )

        # ------------------------------------------------------------------
        # Step 2: Initialize named fields from schema defaults.
        #
        # Named fields start with their ParamSpec default if present; otherwise
        # they start unresolved as NO_VAL.
        # ------------------------------------------------------------------
        packaged: dict[str, Any] = {
            spec.name: (spec.default if spec.default is not NO_VAL else NO_VAL)
            for spec in named_fields
        }

        # Leftover extras accumulated during packaging.
        positional_extras: list[Any] = []
        mapping_extras: dict[str, Any] = {}

        # Explicit variadic payloads extracted directly from a mapping-shaped raw
        # source under the declared sink names.
        explicit_vararg_items: list[Any] = []
        explicit_varkwarg_items: dict[str, Any] = {}

        # ------------------------------------------------------------------
        # Step 3: Optionally coerce object-like raw values into collection form.
        #
        # This is a best-effort preprocessing step. It only runs when the raw
        # value is not already a Mapping and not already a non-string Sequence.
        # ------------------------------------------------------------------
        source = raw
        is_non_string_sequence = isinstance(source, Sequence) and not isinstance(
            source, (str, bytes, bytearray)
        )

        if (
            self.coerce_to_collection
            and not isinstance(source, Mapping)
            and not is_non_string_sequence
        ):
            model_dump = getattr(source, "model_dump", None)
            if callable(model_dump):
                try:
                    candidate = model_dump(mode="python")
                except TypeError:
                    candidate = model_dump()
                if isinstance(candidate, Mapping) or (
                    isinstance(candidate, Sequence)
                    and not isinstance(candidate, (str, bytes, bytearray))
                ):
                    source = candidate
            elif callable(getattr(source, "_asdict", None)):
                candidate = source._asdict()
                if isinstance(candidate, Mapping):
                    source = candidate
            elif hasattr(source, "__dict__"):
                try:
                    candidate = {
                        str(key): value
                        for key, value in vars(source).items()
                        if not str(key).startswith("_")
                    }
                except TypeError:
                    candidate = None
                if isinstance(candidate, Mapping):
                    source = candidate

        is_mapping_source = isinstance(source, Mapping)
        is_sequence_source = isinstance(source, Sequence) and not isinstance(
            source, (str, bytes, bytearray)
        )

        # ------------------------------------------------------------------
        # Step 4: Choose the packaging mode.
        #
        # If there is exactly one named field and map_single_fields is False,
        # treat the whole raw object as the value for that field, even if it is
        # mapping- or sequence-shaped.
        # ------------------------------------------------------------------
        if len(named_fields) == 1 and not self.map_single_fields:
            packaged[named_fields[0].name] = source

        # ------------------------------------------------------------------
        # Step 5A: Mapping mode.
        #
        # Exact named-field matching happens first. Ordinary unmatched entries
        # become mapping extras. If the raw mapping already contains keys that
        # match declared variadic sink names, those values are treated as explicit
        # sink payloads rather than as ordinary extras.
        # ------------------------------------------------------------------
        elif is_mapping_source:
            source_mapping = dict(source)

            # Peel off explicit raw payload for the declared output vararg sink.
            if output_vararg is not None and output_vararg in source_mapping:
                raw_explicit_vararg = source_mapping.pop(output_vararg)
                if not isinstance(raw_explicit_vararg, (list, tuple)):
                    raise ValueError(
                        f"{self.full_name}: raw mapping value for output vararg "
                        f"'{output_vararg}' must be a list or tuple, got "
                        f"{type(raw_explicit_vararg).__name__}."
                    )
                explicit_vararg_items = list(raw_explicit_vararg)

            # Peel off explicit raw payload for the declared output varkwarg sink.
            if output_varkwarg is not None and output_varkwarg in source_mapping:
                raw_explicit_varkwarg = source_mapping.pop(output_varkwarg)
                if not isinstance(raw_explicit_varkwarg, Mapping):
                    raise ValueError(
                        f"{self.full_name}: raw mapping value for output varkwarg "
                        f"'{output_varkwarg}' must be a mapping, got "
                        f"{type(raw_explicit_varkwarg).__name__}."
                    )
                explicit_varkwarg_items = {
                    str(key): value for key, value in raw_explicit_varkwarg.items()
                }

            # Exact named-field extraction.
            named_field_names = {spec.name for spec in named_fields}
            for spec in named_fields:
                if spec.name in source_mapping:
                    packaged[spec.name] = source_mapping[spec.name]

            # Everything left over becomes ordinary mapping extras.
            mapping_extras = {
                str(key): value
                for key, value in source_mapping.items()
                if key not in named_field_names
            }

            # Optional backfill from ordinary mapping extras only.
            if self.map_extras and mapping_extras:
                missing_named_fields = [
                    spec.name
                    for spec in named_fields
                    if packaged[spec.name] is NO_VAL
                ]

                if missing_named_fields:
                    remaining_extras: dict[str, Any] = {}
                    missing_index = 0

                    for extra_key, extra_value in mapping_extras.items():
                        if missing_index < len(missing_named_fields):
                            target_name = missing_named_fields[missing_index]
                            packaged[target_name] = extra_value
                            missing_index += 1
                        else:
                            remaining_extras[extra_key] = extra_value

                    mapping_extras = remaining_extras

        # ------------------------------------------------------------------
        # Step 5B: Sequence mode.
        #
        # Non-string sequences fill named fields positionally in schema order.
        # Any remaining items become positional extras.
        # ------------------------------------------------------------------
        elif is_sequence_source:
            source_items = list(source)

            for index, spec in enumerate(named_fields):
                if index < len(source_items):
                    packaged[spec.name] = source_items[index]

            if len(source_items) > len(named_fields):
                positional_extras = source_items[len(named_fields):]

        # ------------------------------------------------------------------
        # Step 5C: Scalar mode.
        #
        # Scalar values can only be placed when the target is unambiguous:
        # - exactly one named field exists, or
        # - exactly one named field is still missing.
        # Otherwise the scalar cannot be reliably mapped.
        # ------------------------------------------------------------------
        else:
            missing_named_fields = [
                spec.name
                for spec in named_fields
                if packaged[spec.name] is NO_VAL
            ]

            if len(missing_named_fields) == 1:
                packaged[missing_named_fields[0]] = source
            elif len(named_fields) == 1:
                packaged[named_fields[0].name] = source
            elif len(missing_named_fields) > 1:
                raise PackagingError(
                    f"{self.full_name}: too much ambiguity to package scalar output "
                    f"without explicit mapping: {len(missing_named_fields)} named "
                    f"fields are missing."
                )
            elif not self.ignore_unhandled:
                raise PackagingError(
                    f"{self.full_name}: too much ambiguity to package scalar output "
                    f"without explicit mapping: all {len(named_fields)} named fields "
                    f"are already present."
                )

        # ------------------------------------------------------------------
        # Step 6: Late none_is_absent normalization for named fields only.
        #
        # This runs after exact assignment and backfilling so that None values can
        # still participate in normal packaging before being reinterpreted as
        # absent. Variadic sink payloads are filtered later, right before emission.
        # ------------------------------------------------------------------
        if self.none_is_absent:
            for field_name, value in list(packaged.items()):
                if value is None:
                    packaged[field_name] = NO_VAL

        # ------------------------------------------------------------------
        # Step 7: Build and emit final variadic sink payloads.
        #
        # Each sink may receive:
        # - explicit raw payload already present under the sink name
        # - leftovers produced by packaging
        #
        # None values are omitted from final variadic payloads when
        # none_is_absent=True. Variadic sinks are only emitted if non-empty.
        # ------------------------------------------------------------------

        # Final positional sink payload.
        final_vararg_items = list(explicit_vararg_items)
        if positional_extras:
            final_vararg_items.extend(positional_extras)

        if self.none_is_absent and final_vararg_items:
            final_vararg_items = [
                value for value in final_vararg_items
                if value is not None
            ]

        if final_vararg_items:
            if output_vararg is not None:
                packaged[output_vararg] = tuple(final_vararg_items)
            elif not self.ignore_unhandled:
                raise PackagingError(
                    f"{self.full_name}: unhandled positional extras remain "
                    f"but no output vararg sink is declared: {final_vararg_items!r}"
                )

        # Final keyword sink payload.
        if explicit_varkwarg_items and mapping_extras:
            overlapping_keys = set(explicit_varkwarg_items).intersection(mapping_extras)
            if overlapping_keys:
                raise PackagingError(
                    f"{self.full_name}: explicit output varkwarg payload and leftover "
                    f"mapping extras contain overlapping keys: "
                    f"{sorted(overlapping_keys)!r}"
                )

        final_varkwarg_items = dict(explicit_varkwarg_items)
        if mapping_extras:
            final_varkwarg_items.update(mapping_extras)

        if self.none_is_absent and final_varkwarg_items:
            final_varkwarg_items = {
                key: value
                for key, value in final_varkwarg_items.items()
                if value is not None
            }

        if final_varkwarg_items:
            if output_varkwarg is not None:
                packaged[output_varkwarg] = final_varkwarg_items
            elif not self.ignore_unhandled:
                raise PackagingError(
                    f"{self.full_name}: unhandled mapping extras remain "
                    f"but no output varkwarg sink is declared: {final_varkwarg_items!r}"
                )

        # ------------------------------------------------------------------
        # Step 8: Return the packaged result as-is.
        #
        # Any remaining NO_VALs in named fields are intentional and will be
        # resolved later by handle_missing_values().
        # ------------------------------------------------------------------
        return packaged

    def handle_missing_values(self, packaged: Mapping[str, Any]) -> dict[str, Any]:
        """Apply the configured missing-value policy to packaged output."""
        resolved = dict(packaged)
        missing_keys = [key for key, value in resolved.items() if value is NO_VAL]

        if not missing_keys:
            return resolved

        mode = self.absent_value_mode

        if mode == "RAISE":
            raise ValueError(
                f"{self.full_name}: packaged output is missing required field(s): {missing_keys}"
            )

        if mode == "DROP":
            for key in missing_keys:
                resolved.pop(key, None)
            return resolved

        if mode == "FILL":
            for key in missing_keys:
                resolved[key] = self.default_absent_value
            return resolved

        raise ValueError(
            f"{self.full_name}: invalid absent_value_mode {mode!r}"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize this wrapper, its component, and its packaging policy."""
        data = super().to_dict()
        data.update({
            "component": self.component.to_dict(),
            "output_schema": [spec.to_dict() for spec in self._output_schema],
            "map_single_fields": self.map_single_fields,
            "map_extras": self.map_extras,
            "ignore_unhandled": self.ignore_unhandled,
            "absent_value_mode": self.absent_value_mode,
            "default_absent_value": self.default_absent_value,
            "none_is_absent": self.none_is_absent,
            "coerce_to_collection": self.coerce_to_collection,
        })
        return data
