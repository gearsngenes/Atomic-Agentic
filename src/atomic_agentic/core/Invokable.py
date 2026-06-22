from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Mapping, Dict, Sequence, Optional, TypeVar
import threading
import asyncio
from uuid import uuid4
import logging

from ..constants.core import IDENTIFIER_PATTERN, NO_VAL
from ..models.parameters import ParamSpec
from ..utils.parameters import _validate_parameter_order, to_paramspec_list
from ..exceptions import PackagingError
from ..models.results import AtomicResult, CommandResult, StructuredResult


logger = logging.getLogger(__name__)
_R = TypeVar("_R", bound=AtomicResult)

# Deprecated aliases removed in v2: use `ParamSpec` and the `parameters` property.


class AtomicInvokable(ABC):
    """
    Basal invokable contract for Tools, Agents, and Workflows.

    Overview
    --------
    `AtomicInvokable` defines the minimal, language-level contract that all
    executable primitives in this codebase must satisfy. It standardises:

    - **identity**: validated `name` and `description` (human-friendly strings).
    - **interface**: a single execution entrypoint
      ``invoke(inputs: Mapping[str, Any])`` returning an ``AtomicResult``-family
      successful-invocation envelope.
    - **parameters & return type**: declared at construction time as concrete
      `parameters: list[ParamSpec]` and `return_type: str`.
    - **metadata serialization**: default ``to_dict()`` implementation for
      persisting metadata.

    Parameters and Schema
    ---------------------
    Parameters are declared as a list of ``ParamSpec`` objects at construction time.
    Each ``ParamSpec`` is self-sufficient and contains:

      - ``name`` (str): parameter name; must be a valid Python identifier
      - ``index`` (int): position in the parameter sequence (0-based)
      - ``kind`` (str): parameter classification—one of:

        - ``POSITIONAL_ONLY``: cannot be passed by name (``/``-style)
        - ``POSITIONAL_OR_KEYWORD``: may be passed by name or position
        - ``KEYWORD_ONLY``: must be passed by name (``*``-style)
        - ``VAR_POSITIONAL``: accepts ``*args`` (unnamed)
        - ``VAR_KEYWORD``: accepts ``**kwargs`` (named)

      - ``type`` (str): human-readable type annotation, e.g. ``"int"`` or ``"List[str]"``
      - ``default`` (Any): default value if parameter is optional; ``NO_VAL`` if required

    The list order is canonical and defines parameter sequence. ``index`` is stored for
    redundancy (enabling self-sufficiency) and should match list position.

    Dict-First Invocation Contract
    -------------------------------
    ``invoke(inputs)`` accepts a ``Mapping[str, Any]`` where keys correspond to
    parameter names. The contract is "dict-first": callers provide a mapping, not
    ``(*args, **kwargs)``. Implementations (subclasses) are responsible for:

      - Validating required parameters are present
      - Handling default values
      - Converting the dict to appropriate execution arguments or provider payloads
      - Raising clear, typed exceptions on invalid inputs (use
        ``ToolInvocationError``, ``AgentInvocationError``, etc.)
      - Returning an ``AtomicResult``-family object whose ``.result`` field contains
        the caller-facing payload

    Return Type Contract
    --------------------
    ``return_type`` describes the caller-facing payload stored in
    ``AtomicResult.result``. It does not describe the result envelope class itself.

    Architecture Notes
    -------------------
    - The class intentionally does not expose a human-readable ``signature`` string
      inside ``to_dict()`` to minimize churn when persisting metadata; use the
      ``signature`` property for logging and UIs.
    - **Backward Compatibility**: Legacy aliases (previously ``ParameterMap``,
      ``ArgumentMap``, and ``ArgSpec``) have been removed in v2; prefer
      ``ParamSpec`` and the `parameters` property on invokable instances.
    """

    def __init__(
        self,
        *,
        name: str,
        description: str,
        namespace: str = "default",
        parameters: list[ParamSpec],
        return_type: str,
        filter_extraneous_inputs: bool = True,
    ) -> None:
        # setters include validation
        self.name = name
        self.description = description

        # Validate and store namespace — same identifier rules as name.
        if not isinstance(namespace, str) or not namespace.strip():
            raise ValueError("namespace must be a non-empty string")
        if not IDENTIFIER_PATTERN.fullmatch(namespace):
            raise ValueError(
                f"namespace must be alphanumeric/underscore and not start with a digit; "
                f"got {namespace!r}"
            )
        self._namespace: str = namespace

        # Validate parameters
        if not isinstance(parameters, list):
            raise TypeError(
                f"{type(self).__name__}: parameters must be a list[ParamSpec], got {type(parameters)!r}"
            )
        if not all(isinstance(p, ParamSpec) for p in parameters):
            raise TypeError(
                f"{type(self).__name__}: all parameters must be ParamSpec instances"
            )

        # Validate parameter names are unique and valid
        param_names = [p.name for p in parameters]
        if len(param_names) != len(set(param_names)):
            raise TypeError(
                f"{type(self).__name__}: duplicate parameter names detected: {param_names}"
            )
        for p in parameters:
            if not isinstance(p.name, str) or not p.name:
                raise TypeError(
                    f"{type(self).__name__}: parameter names must be non-empty strings"
                )
            if not IDENTIFIER_PATTERN.fullmatch(p.name):
                raise ValueError(
                    f"{type(self).__name__}: parameter name {p.name!r} is not a valid identifier"
                )

        # Validate indices are consistent with list position
        for i, p in enumerate(parameters):
            if p.index != i:
                raise TypeError(
                    f"{type(self).__name__}: parameter at position {i} has mismatched index {p.index}"
                )

        # Validate parameter ordering (will raise SchemaError if invalid)
        _validate_parameter_order(parameters)

        # Validate return type
        if not isinstance(return_type, str):
            raise TypeError(
                f"{type(self).__name__}: return_type must be a str, got {type(return_type)!r}"
            )

        self._parameters: list[ParamSpec] = parameters
        self._return_type: str = return_type
        self.filter_extraneous_inputs = filter_extraneous_inputs
        # invoke lock
        self._invoke_lock = threading.RLock()
        # unique identifier for this invokable instance
        self._instance_id = str(uuid4())

    # ---------------------------------------------------------------- #
    # Name + description with validation
    # ---------------------------------------------------------------- #
    @property
    def name(self) -> str:
        """The canonical name of this invokable."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("name must be a non-empty string")
        if not IDENTIFIER_PATTERN.fullmatch(value):
            raise ValueError(
                f"name must be alphanumeric/underscore and not start with a digit; got {value!r}"
            )
        self._name = value

    @property
    def description(self) -> str:
        """Human-friendly description."""
        return self._description

    @description.setter
    def description(self, value: str) -> None:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("description must be a non-empty string")
        self._description = value.strip()

    @property
    def namespace(self) -> str:
        """Grouping label for this invokable, used as the middle segment of ``full_name``."""
        return self._namespace

    @property
    def full_name(self) -> str:
        """Fully-qualified name of the form ``Type.namespace.name``."""
        return f"{type(self).__name__}.{self._namespace}.{self._name}"
    
    @property
    def instance_id(self) -> str:
        """Unique identifier for this invokable instance."""
        return self._instance_id
    
    # ---------------------------------------------------------------- #
    # Parameters and return type (primary API)
    # ---------------------------------------------------------------- #
    @property
    def parameters(self) -> list[ParamSpec]:
        """Primary parameter specification as an ordered list of ParamSpec objects.

        Returns a list of ``ParamSpec`` in signature order. The list is copied
        to prevent external mutation.
        """
        return list(self._parameters)

    @property
    def has_varargs(self) -> bool:
        """Whether this invokable accepts variable positional arguments (*args)."""
        return any(p.kind == ParamSpec.VAR_POSITIONAL for p in self._parameters)

    @property
    def has_varkwargs(self) -> bool:
        """Whether this invokable accepts variable keyword arguments (**kwargs)."""
        return any(p.kind == ParamSpec.VAR_KEYWORD for p in self._parameters)

    @property
    def return_type(self) -> str:
        """Payload return type stored inside ``AtomicResult.result``."""
        return self._return_type

    @property
    def filter_extraneous_inputs(self) -> bool:
        """Whether to filter extraneous inputs not used by the component's parameters."""
        return self._filter_extraneous_inputs
    
    @filter_extraneous_inputs.setter
    def filter_extraneous_inputs(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError(
                f"{type(self).__name__}.filter_extraneous_inputs must be a bool, "
                f"got {type(value)!r}"
            )
        self._filter_extraneous_inputs = value

    # ---------------------------------------------------------------- #
    # Secondary Access compatibility properties
    # ---------------------------------------------------------------- #
    @property
    def parameters_map(self) -> dict[str, ParamSpec]:
        """Secondary access: parameters as a dict mapping name -> ParamSpec.

        Prefer ``parameters`` for ordered list access.
        """
        return {spec.name: spec for spec in self._parameters}

    # ---------------------------------------------------------------- #
    # Signature helper
    # ---------------------------------------------------------------- #
    @property
    def signature(self) -> str:
        """
        Returns a signature like:

            ClassName.name(param1: Type, param2: Type = default) -> return_type
        """
        params = []
        for spec in self._parameters:
            ptype = spec.type or "Any"
            default_marker = ""
            if spec.default is not NO_VAL:
                default_marker = f" = {spec.default!r}"
            
            if spec.kind == ParamSpec.VAR_POSITIONAL:
                params.append(f"*{spec.name}: {ptype}{default_marker}")
            elif spec.kind == ParamSpec.VAR_KEYWORD:
                params.append(f"**{spec.name}: {ptype}{default_marker}")
            else:
                params.append(f"{spec.name}: {ptype}{default_marker}")
        
        params_str = ", ".join(params)
        return f"{self.full_name}({params_str}) -> {self.return_type}"

    # ---------------------------------------------------------------- #
    # Abstract contract
    # ---------------------------------------------------------------- #
    @abstractmethod
    def invoke(self, inputs: Mapping[str, Any]) -> AtomicResult:
        """Perform work and return an AtomicResult-family envelope."""
        raise NotImplementedError

    async def async_invoke(self, inputs: Mapping[str, Any]) -> AtomicResult:
        """
        Default async compatibility wrapper.

        Returns an AtomicResult-family envelope, mirroring `invoke(...)`.
        This preserves the current sync-first implementation by running
        `invoke(inputs)` in a worker thread.
        """
        return await asyncio.to_thread(self.invoke, inputs)

    def make_result(
        self,
        result: Any,
        started_at: datetime,
        ended_at: datetime,
        **result_kwargs: Any,
    ) -> AtomicResult:
        """
        Construct this invokable's default AtomicResult-family envelope.

        Subclasses may override this hook to choose a more specific result class
        or add subclass-specific result fields while preserving the public
        invocation template.
        """
        return self._make_result(
            result=result,
            started_at=started_at,
            ended_at=ended_at,
            **result_kwargs,
        )

    def _make_result(
        self,
        *,
        result: Any,
        started_at: datetime,
        ended_at: datetime,
        result_cls: type[_R] = AtomicResult,
        **result_kwargs: Any,
    ) -> _R:
        """
        Construct an AtomicResult-family object for this invokable.

        This helper only centralizes the common envelope construction details:
        injecting this invokable's ``instance_id`` as ``invoker_id`` and passing
        through caller-provided timing values and explicit subclass fields.

        It intentionally does not capture time, wrap exceptions, record traces,
        manage lifecycle state, or choose result classes from instance-level
        registry/configuration.
        """
        if not isinstance(result_cls, type) or not issubclass(result_cls, AtomicResult):
            raise TypeError(
                f"result_cls must be a subclass of AtomicResult, got {result_cls!r}"
            )
        return result_cls(
            result=result,
            invoker_id=self.instance_id,
            started_at=started_at,
            ended_at=ended_at,
            **result_kwargs,
        )

    # ---------------------------------------------------------------- #
    # Input Filtering and Validation
    # ---------------------------------------------------------------- #
    def filter_inputs(self, inputs: Mapping[str, Any]) -> Dict[str, Any]:
        """
        Filter and normalize dict-first inputs according to this invokable's
        declared parameter contract.

        Behavior
        --------
        - Inputs must be a Mapping.
        - Known parameter keys are retained.
        - Explicit *args payloads must be list or tuple.
        - Explicit **kwargs payloads must be a Mapping.
        - Unknown keys are merged into the **kwargs payload when VAR_KEYWORD exists.
        - Unknown keys are dropped when no VAR_KEYWORD exists and filtering is enabled.
        - Unknown keys raise when no VAR_KEYWORD exists and filtering is disabled.
        - Explicit **kwargs payload keys may not overlap with loose unknown input keys.
        """
        if not isinstance(inputs, Mapping):
            raise TypeError(
                f"{type(self).__name__}.invoke: inputs must be a mapping, "
                f"got {type(inputs)!r}"
            )

        parameters = self.parameters
        param_specs = {param.name: param for param in parameters}
        param_names = set(param_specs)

        vararg_spec = next(
            (param for param in parameters if param.kind == ParamSpec.VAR_POSITIONAL),
            None,
        )
        varkwarg_spec = next(
            (param for param in parameters if param.kind == ParamSpec.VAR_KEYWORD),
            None,
        )

        vararg_name = vararg_spec.name if vararg_spec is not None else None
        varkwarg_name = varkwarg_spec.name if varkwarg_spec is not None else None

        filtered: Dict[str, Any] = {}

        for param_name in param_names:
            if param_name in inputs:
                filtered[param_name] = inputs[param_name]

        if vararg_name is not None and vararg_name in filtered:
            value = filtered[vararg_name]
            if not isinstance(value, (list, tuple)):
                raise TypeError(
                    f"{self.full_name}: explicit VAR_POSITIONAL input "
                    f"{vararg_name!r} must be a list or tuple, got {type(value)!r}"
                )

        if varkwarg_name is not None and varkwarg_name in filtered:
            value = filtered[varkwarg_name]
            if not isinstance(value, Mapping):
                raise TypeError(
                    f"{self.full_name}: explicit VAR_KEYWORD input "
                    f"{varkwarg_name!r} must be a mapping, got {type(value)!r}"
                )

        extra_keys = [key for key in inputs if key not in param_names]
        extras = {key: inputs[key] for key in extra_keys}

        if varkwarg_name is not None:
            explicit = filtered.get(varkwarg_name, {})
            overlapping_keys = set(explicit).intersection(extras)
            if overlapping_keys:
                raise TypeError(
                    f"{self.full_name}: explicit VAR_KEYWORD input "
                    f"{varkwarg_name!r} and extra input keys overlap: "
                    f"{sorted(overlapping_keys)!r}"
                )

            merged = dict(explicit)
            merged.update(extras)
            filtered[varkwarg_name] = merged
        elif extras and not self._filter_extraneous_inputs:
            raise TypeError(
                f"{self.full_name}: unexpected input key(s): {sorted(extras)!r}"
            )

        return filtered

    # ---------------------------------------------------------------- #
    # Default metadata serialization
    # ---------------------------------------------------------------- #
    def to_dict(self) -> Dict[str, Any]:
        """
        Minimal metadata serialization.

        Does *not* include `signature` by default to reduce churn in persisted metadata.
        """
        return {
            "type": type(self).__name__,
            "instance_id": self.instance_id,
            "name": self.name,
            "namespace": self.namespace,
            "description": self.description,
            "parameters": [spec.to_dict() for spec in self._parameters],
            "return_type": self.return_type,
            "filter_extraneous_inputs": self._filter_extraneous_inputs,
        }

    # ---------------------------------------------------------------- #
    # Unified repr/str
    # ---------------------------------------------------------------- #
    def __str__(self) -> str:
        return f"<{self.signature} - {self.description}>"

    def __repr__(self) -> str:
        return f"{self.signature}: {self.description}"

    # ---------------------------------------------------------------- #
    # callable contract
    # ---------------------------------------------------------------- #
    def __call__(self, *args: Any, **kwargs: Any) -> AtomicResult:
        """
        Allows the invokable to be called like a regular function.

        Check for varargs/kwargs parameters and construct the inputs dict
        accordingly before invoking. The return value mirrors ``invoke(...)``:
        an AtomicResult-family envelope whose ``.result`` field contains the
        caller-facing payload.
        """
        inputs = self._args_kwargs_to_dict(*args, **kwargs)
        return self.invoke(inputs)

    async def async_call(self, *args: Any, **kwargs: Any) -> AtomicResult:
        """
        Async analog of __call__.

        Bind normal call-style args/kwargs into the dict-first inputs shape,
        then delegate to async_invoke(). The return value mirrors
        ``async_invoke(...)``.
        """
        inputs = self._args_kwargs_to_dict(*args, **kwargs)

        return await self.async_invoke(inputs)

    @staticmethod
    def _unwrap_result_payload(value: Any) -> Any:
        """
        Return the caller-facing payload from an AtomicResult-family value.

        During staged result integration, some child invokables may already
        return AtomicResult-family objects while others still return raw payloads.
        This helper gives migrated callers one narrow compatibility rule.
        """
        if isinstance(value, AtomicResult):
            return value.result
        return value

    def _args_kwargs_to_dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """
        Convert Python call-style (*args, **kwargs) into the dict-first input shape.

        Behavior
        --------
        - Positional arguments bind to POSITIONAL_ONLY and POSITIONAL_OR_KEYWORD
        parameters in declaration order.
        - Extra positional arguments populate VAR_POSITIONAL when available.
        - Too many positional arguments raise when no VAR_POSITIONAL exists.
        - Keyword arguments matching POSITIONAL_OR_KEYWORD or KEYWORD_ONLY parameters
        bind directly.
        - Keyword arguments matching POSITIONAL_ONLY parameters raise.
        - Keyword arguments not matching non-variadic keyword-bindable parameters
        populate VAR_KEYWORD when available.
        - Unhandled keyword arguments raise when no VAR_KEYWORD exists.
        - VAR_POSITIONAL and VAR_KEYWORD parameter names receive no special keyword
        treatment; as keywords, they are ordinary extra keywords.
        """
        positional_values = list(args)
        keyword_values = dict(kwargs)

        parameters = self.parameters

        positional_capable = [
            param
            for param in parameters
            if param.kind in {
                ParamSpec.POSITIONAL_ONLY,
                ParamSpec.POSITIONAL_OR_KEYWORD,
            }
        ]
        positional_only_names = {
            param.name
            for param in parameters
            if param.kind == ParamSpec.POSITIONAL_ONLY
        }
        keyword_bindable_names = {
            param.name
            for param in parameters
            if param.kind in {
                ParamSpec.POSITIONAL_OR_KEYWORD,
                ParamSpec.KEYWORD_ONLY,
            }
        }

        vararg_spec = next(
            (param for param in parameters if param.kind == ParamSpec.VAR_POSITIONAL),
            None,
        )
        varkwarg_spec = next(
            (param for param in parameters if param.kind == ParamSpec.VAR_KEYWORD),
            None,
        )

        inputs: dict[str, Any] = {}

        positional_count = len(positional_capable)
        cutoff = min(len(positional_values), positional_count)

        if len(positional_values) > positional_count and vararg_spec is None:
            raise TypeError(
                f"{self.full_name} takes at most {positional_count} positional "
                f"arguments but {len(positional_values)} were given"
            )

        for index in range(cutoff):
            param = positional_capable[index]
            inputs[param.name] = positional_values[index]

        if vararg_spec is not None and positional_values[cutoff:]:
            inputs[vararg_spec.name] = tuple(positional_values[cutoff:])

        extra_keywords: dict[str, Any] = {}

        for key, value in keyword_values.items():
            if key in positional_only_names:
                raise TypeError(
                    f"{self.full_name} got positional-only argument {key!r} "
                    "passed as keyword"
                )

            if key in keyword_bindable_names:
                if key in inputs:
                    raise TypeError(
                        f"{self.full_name} got multiple values for argument {key!r}"
                    )
                inputs[key] = value
                continue

            extra_keywords[key] = value

        if extra_keywords:
            if varkwarg_spec is None:
                raise TypeError(
                    f"{self.full_name} got unexpected keyword arguments: "
                    f"{', '.join(extra_keywords.keys())}"
                )
            inputs[varkwarg_spec.name] = extra_keywords

        return inputs

    def _dict_to_args_kwargs(
        self,
        inputs: Mapping[str, Any],
    ) -> tuple[tuple[Any, ...], Dict[str, Any]]:
        """
        Convert dict-first inputs into Python call-style (*args, **kwargs).

        Behavior
        --------
        - Inputs are normalized through filter_inputs().
        - POSITIONAL_ONLY parameters are appended to args.
        - POSITIONAL_OR_KEYWORD parameters are appended to args when an explicit
        VAR_POSITIONAL payload is present; otherwise they are placed in kwargs.
        - VAR_POSITIONAL payloads extend args.
        - KEYWORD_ONLY parameters are placed in kwargs.
        - VAR_KEYWORD payloads update kwargs.
        - Missing required non-variadic parameters raise TypeError.
        - Missing optional non-variadic parameters use their declared defaults.
        - VAR_POSITIONAL and VAR_KEYWORD parameters do not receive defaults; absence
        means no additional positional or keyword arguments.
        """
        if not isinstance(inputs, Mapping):
            raise TypeError(
                f"{type(self).__name__}._dict_to_args_kwargs: inputs must be a mapping, "
                f"got {type(inputs)!r}"
            )

        data = self.filter_inputs(inputs)
        parameters = self.parameters

        vararg_spec = next(
            (param for param in parameters if param.kind == ParamSpec.VAR_POSITIONAL),
            None,
        )
        has_explicit_varargs = (
            vararg_spec is not None and vararg_spec.name in data
        )

        args: list[Any] = []
        kwargs: Dict[str, Any] = {}
        missing: list[str] = []

        for param in parameters:
            if param.kind == ParamSpec.POSITIONAL_ONLY:
                if param.name in data:
                    args.append(data[param.name])
                elif param.default is not NO_VAL:
                    args.append(param.default)
                else:
                    missing.append(param.name)
                continue

            if param.kind == ParamSpec.POSITIONAL_OR_KEYWORD:
                if param.name in data:
                    value = data[param.name]
                elif param.default is not NO_VAL:
                    value = param.default
                else:
                    missing.append(param.name)
                    continue

                if has_explicit_varargs:
                    args.append(value)
                else:
                    kwargs[param.name] = value
                continue

            if param.kind == ParamSpec.VAR_POSITIONAL:
                if param.name not in data:
                    continue

                value = data[param.name]
                if not isinstance(value, (list, tuple)):
                    raise TypeError(
                        f"{self.full_name}: VAR_POSITIONAL input {param.name!r} "
                        f"must be a list or tuple, got {type(value)!r}"
                    )

                args.extend(value)
                continue

            if param.kind == ParamSpec.KEYWORD_ONLY:
                if param.name in data:
                    kwargs[param.name] = data[param.name]
                elif param.default is not NO_VAL:
                    kwargs[param.name] = param.default
                else:
                    missing.append(param.name)
                continue

            if param.kind == ParamSpec.VAR_KEYWORD:
                if param.name not in data:
                    continue

                value = data[param.name]
                if not isinstance(value, Mapping):
                    raise TypeError(
                        f"{self.full_name}: VAR_KEYWORD input {param.name!r} "
                        f"must be a mapping, got {type(value)!r}"
                    )

                overlapping_keys = set(kwargs).intersection(value)
                if overlapping_keys:
                    raise TypeError(
                        f"{self.full_name}: VAR_KEYWORD input {param.name!r} "
                        f"would overwrite bound keyword argument(s): "
                        f"{sorted(overlapping_keys)!r}"
                    )

                kwargs.update(value)
                continue

            raise TypeError(
                f"{self.full_name}: unsupported parameter kind {param.kind!r} "
                f"for parameter {param.name!r}"
            )

        if missing:
            raise TypeError(
                f"{self.full_name}: missing required argument(s): {missing!r}"
            )

        return tuple(args), kwargs


class Command(AtomicInvokable):
    """
    No-input command wrapper around one AtomicInvokable and one fixed input mapping.

    `Command` implements the command pattern for AtomicInvokable instances:

    - construction receives a wrapped executor and a fixed input mapping;
    - the fixed input mapping is validated through the executor's own
      `filter_inputs(...)` path and then shallow-copied;
    - the Command itself exposes no parameters;
    - caller-provided runtime inputs are never accepted;
    - invocation delegates to the wrapped executor with the fixed input mapping.

    This is useful when an invokable should be registered, stored, composed, or
    passed around as a zero-argument executable action.
    """

    def __init__(
        self,
        executor: AtomicInvokable,
        fixed_inputs: Mapping[str, Any],
        name: Optional[str] = None,
        description: Optional[str] = None,
        namespace: Optional[str] = None,
    ) -> None:
        if not isinstance(executor, AtomicInvokable):
            raise TypeError(
                f"{type(self).__name__}: executor must be an AtomicInvokable, "
                f"got {type(executor)!r}"
            )

        if not isinstance(fixed_inputs, Mapping):
            raise TypeError(
                f"{type(self).__name__}: fixed_inputs must be a mapping, "
                f"got {type(fixed_inputs)!r}"
            )

        # Validate and normalize the fixed inputs through the executor's own
        # input-filtering contract. This rejects bad shapes and invalid extras
        # according to the executor's declared parameter policy.
        filtered_fixed_inputs = executor.filter_inputs(fixed_inputs)

        # `filter_inputs(...)` normalizes shape/extras, but missing required
        # parameters are normally rejected later by dict-to-call binding.
        # Validate that here so invalid commands fail at construction time.
        self._validate_fixed_inputs_bindable(
            executor=executor,
            inputs=filtered_fixed_inputs,
        )

        self._executor: AtomicInvokable = executor
        self._fixed_inputs: Dict[str, Any] = dict(filtered_fixed_inputs)

        resolved_name = name or f"{executor.name}_command"
        resolved_description = (
            description
            or f"Command wrapper for {executor.full_name} with fixed inputs."
        )

        super().__init__(
            name=resolved_name,
            description=resolved_description,
            namespace=namespace or executor.namespace,  # inherit when not supplied
            parameters=[],
            return_type=executor.return_type,
            filter_extraneous_inputs=False,
        )

    # ---------------------------------------------------------------- #
    # Command properties
    # ---------------------------------------------------------------- #
    @property
    def executor(self) -> AtomicInvokable:
        """The wrapped invokable executed by this command."""
        return self._executor

    @property
    def fixed_inputs(self) -> Dict[str, Any]:
        """A shallow copy of the fixed executor input mapping."""
        return dict(self._fixed_inputs)

    @property
    def filter_extraneous_inputs(self) -> bool:
        """
        Commands always reject runtime caller inputs.

        This property is intentionally fixed to False so that the empty
        parameter list means "no inputs accepted", not "drop all inputs".
        """
        return self._filter_extraneous_inputs

    @filter_extraneous_inputs.setter
    def filter_extraneous_inputs(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError(
                f"{type(self).__name__}.filter_extraneous_inputs must be a bool, "
                f"got {type(value)!r}"
            )
        if value is not False:
            raise ValueError(
                f"{type(self).__name__}.filter_extraneous_inputs is fixed to False."
            )
        self._filter_extraneous_inputs = False

    # ---------------------------------------------------------------- #
    # Construction validation
    # ---------------------------------------------------------------- #
    @staticmethod
    def _validate_fixed_inputs_bindable(
        *,
        executor: AtomicInvokable,
        inputs: Mapping[str, Any],
    ) -> None:
        """
        Validate that the fixed inputs are complete enough to invoke the executor.

        `AtomicInvokable.filter_inputs(...)` owns mapping-shape validation,
        extraneous-key policy, and variadic payload normalization. Required
        non-variadic parameters are checked here so an invalid Command fails
        during construction rather than first invocation.
        """
        missing: list[str] = []

        for spec in executor.parameters:
            if spec.kind in {ParamSpec.VAR_POSITIONAL, ParamSpec.VAR_KEYWORD}:
                continue
            if spec.default is not NO_VAL:
                continue
            if spec.name not in inputs:
                missing.append(spec.name)

        if missing:
            raise TypeError(
                f"{type(executor).__name__}.{executor.name}: command fixed_inputs "
                f"missing required input key(s): {missing!r}"
            )

    # ---------------------------------------------------------------- #
    # Result construction
    # ---------------------------------------------------------------- #
    def make_result(
        self,
        result: Any,
        started_at: datetime,
        ended_at: datetime,
        **result_kwargs: Any,
    ) -> CommandResult:
        """Construct a CommandResult envelope for this command's invocation."""
        return self._make_result(
            result=result,
            started_at=started_at,
            ended_at=ended_at,
            result_cls=CommandResult,
            **result_kwargs,
        )

    # ---------------------------------------------------------------- #
    # Invocation
    # ---------------------------------------------------------------- #
    def invoke(self, inputs: Mapping[str, Any]) -> CommandResult:
        """
        Invoke the wrapped executor with this command's fixed input mapping.

        Runtime inputs must be an empty mapping. The validation is routed through
        `self.filter_inputs(inputs)` so errors remain consistent with the base
        AtomicInvokable contract. The executor's returned AtomicResult is
        unwrapped — `executor_result.result` becomes this command's payload,
        and `executor_result.run_id` plus the executor's `instance_id` are
        carried forward for cross-envelope tracing.
        """
        with self._invoke_lock:
            logger.info("[%s started]", self.full_name)
            started_at = datetime.now(timezone.utc)

            self.filter_inputs(inputs)
            executor_result = self.executor.invoke(dict(self._fixed_inputs))

            ended_at = datetime.now(timezone.utc)
            logger.info("[%s finished]", self.full_name)
            return self.make_result(
                result=executor_result.result,
                started_at=started_at,
                ended_at=ended_at,
                executor_run_id=executor_result.run_id,
                executor_id=self.executor.instance_id,
            )

    async def async_invoke(self, inputs: Mapping[str, Any]) -> CommandResult:
        """
        Async command invocation — see `invoke` for the unwrap/lifecycle contract.

        This delegates to the wrapped executor's native async path instead of
        relying on AtomicInvokable's default sync-to-thread wrapper.
        """
        logger.info("[%s started]", self.full_name)
        started_at = datetime.now(timezone.utc)

        self.filter_inputs(inputs)
        fixed_inputs = dict(self._fixed_inputs)

        executor_result = await self.executor.async_invoke(fixed_inputs)

        ended_at = datetime.now(timezone.utc)
        logger.info("[%s finished]", self.full_name)
        return self.make_result(
            result=executor_result.result,
            started_at=started_at,
            ended_at=ended_at,
            executor_run_id=executor_result.run_id,
            executor_id=self.executor.instance_id,
        )

    # ---------------------------------------------------------------- #
    # Serialization
    # ---------------------------------------------------------------- #
    def to_dict(self) -> Dict[str, Any]:
        """
        Return a metadata/debug snapshot for this command.

        `fixed_inputs` are included directly as a shallow dictionary. This is
        useful for introspection, but callers should avoid storing secrets in
        fixed inputs if serialized metadata may be logged or persisted.
        """
        data = super().to_dict()
        data.update(
            {
                "executor": self.executor.to_dict(),
                "fixed_inputs": dict(self._fixed_inputs),
            }
        )
        return data


class StructuredInvokable(AtomicInvokable):
    """Wrap an invokable and package its raw output into a mapping."""

    RAISE = "RAISE"
    DROP = "DROP"
    FILL = "FILL"

    _ABSENT_VALUE_MODES: frozenset[str] = frozenset({RAISE, DROP, FILL})

    PASSTHROUGH = [ParamSpec(
        name="__passthrough_mapping__",
        index=0,
        kind=ParamSpec.VAR_KEYWORD,
        type="Mapping[str, Any]",)]

    def __init__(
        self,
        component: AtomicInvokable,
        name: Optional[str] = None,
        description: Optional[str] = None,
        *,
        namespace: Optional[str] = None,
        output_schema: Optional[
            type | list[str] | tuple[str, ...] | set[str] | list[ParamSpec]
        ] = None,
        map_single_fields: bool = True,
        map_extras: bool = True,
        ignore_unhandled: bool = False,
        absent_value_mode: str = RAISE,
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
            namespace=namespace or component.namespace,  # inherit when not supplied
            parameters=component.parameters,
            return_type="dict[str, Any]",
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
        _validate_parameter_order(normalized)
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
        """The canonical uppercase missing-value policy."""
        return self._absent_value_mode

    @absent_value_mode.setter
    def absent_value_mode(self, value: str) -> None:
        """Validate, normalize, and set the missing-value policy."""
        if not isinstance(value, str):
            raise TypeError(
                f"absent_value_mode must be a string, got {type(value).__name__}"
            )

        normalized = value.strip().upper()
        if normalized not in self._ABSENT_VALUE_MODES:
            raise ValueError(
                "absent_value_mode must be one of: 'RAISE', 'DROP', 'FILL' "
                "(case-insensitive)"
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

    def make_result(
        self,
        result: dict[str, Any],
        started_at: datetime,
        ended_at: datetime,
        **result_kwargs: Any,
    ) -> StructuredResult:
        """
        Construct this wrapper's ``StructuredResult`` envelope.

        ``result`` is the packaged, missing-value-resolved output mapping stored
        in ``StructuredResult.result``. Packaging diagnostics gathered during the
        invocation — the pre-packaging payload, unresolved named fields, and the
        wrapped component's run identity — are stored as explicit
        StructuredInvokable-specific result fields. Mirrors
        ``LLMEngine.make_result``'s validate-then-delegate shape: this hook
        validates the kwargs ``invoke``/``async_invoke`` assemble, then fixes
        ``result_cls=StructuredResult`` and delegates to ``_make_result``.
        """
        unexpected = set(result_kwargs) - {"unpackaged_result", "missing_keys", "component_run_id"}
        if unexpected:
            raise PackagingError(
                f"make_result: unexpected result kwarg(s): {sorted(unexpected)!r}."
            )

        if "unpackaged_result" not in result_kwargs:
            raise PackagingError("StructuredInvokable.make_result: unpackaged_result is required.")
        unpackaged_result = result_kwargs["unpackaged_result"]

        missing_keys = result_kwargs.get("missing_keys", ())
        if not isinstance(missing_keys, tuple) or not all(isinstance(k, str) for k in missing_keys):
            raise PackagingError(
                "StructuredInvokable.make_result: missing_keys must be a tuple of str."
            )

        component_run_id = result_kwargs.get("component_run_id")
        if component_run_id is not None and not isinstance(component_run_id, str):
            raise PackagingError(
                "StructuredInvokable.make_result: component_run_id must be a str or None."
            )

        return self._make_result(
            result=result,
            started_at=started_at,
            ended_at=ended_at,
            result_cls=StructuredResult,
            unpackaged_result=unpackaged_result,
            missing_keys=missing_keys,
            component_run_id=component_run_id,
        )

    def invoke(self, inputs: Mapping[str, Any]) -> StructuredResult:
        """Synchronously invoke the wrapped component and return a StructuredResult.

        1) Filter caller inputs through this wrapper's input contract.
        2) Invoke the wrapped component — its return is an AtomicResult-family
           object; this method extracts ``.result``/``.run_id`` from it before
           packaging (every AtomicInvokable returns an AtomicResult).
        3) Package the unwrapped payload according to the output schema and
           packaging policies, then resolve unresolved named fields per
           ``absent_value_mode``.
        4) Assemble and return this invocation's StructuredResult via
           ``make_result(...)``, carrying the packaged output plus packaging
           diagnostics (unpackaged_result, missing_keys, component_run_id).
        """
        with self._invoke_lock:
            logger.info(f"[{self.full_name} started]")
            started_at = datetime.now(timezone.utc)

            filtered_inputs = self.filter_inputs(inputs)
            component_result = self.component.invoke(filtered_inputs)
            unpackaged_result = component_result.result
            component_run_id = component_result.run_id

            packaged = self.package(unpackaged_result)
            missing_keys = tuple(key for key, value in packaged.items() if value is NO_VAL)
            final_output = self.handle_missing_values(packaged)

            ended_at = datetime.now(timezone.utc)
            logger.info(f"[{self.full_name} finished]")

            return self.make_result(
                result=dict(final_output),
                started_at=started_at,
                ended_at=ended_at,
                unpackaged_result=unpackaged_result,
                missing_keys=missing_keys,
                component_run_id=component_run_id,
            )

    async def async_invoke(self, inputs: Mapping[str, Any]) -> StructuredResult:
        """Asynchronous analog of :meth:`invoke` — see its docstring for the lifecycle."""
        logger.info(f"[Async {self.full_name} started]")
        started_at = datetime.now(timezone.utc)

        filtered_inputs = self.filter_inputs(inputs)
        component_result = await self.component.async_invoke(filtered_inputs)
        unpackaged_result = component_result.result
        component_run_id = component_result.run_id

        packaged = self.package(unpackaged_result)
        missing_keys = tuple(key for key, value in packaged.items() if value is NO_VAL)
        final_output = self.handle_missing_values(packaged)

        ended_at = datetime.now(timezone.utc)
        logger.info(f"[Async {self.full_name} finished]")

        return self.make_result(
            result=dict(final_output),
            started_at=started_at,
            ended_at=ended_at,
            unpackaged_result=unpackaged_result,
            missing_keys=missing_keys,
            component_run_id=component_run_id,
        )

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

        # -------------------------------------------------------------------
        # Step 3.5: Special passthrough mode for mapping sources with a passthrough schema.
        #
        # Special passthrough mode for mapping sources with a passthrough schema.
        # In this mode, the raw mapping is returned as-is with no packaging or
        # missing-value handling applied. This is an escape hatch for maximum flexibility
        # when the source is already a mapping and the schema is just a generic passthrough.
        # -------------------------------------------------------------------
        if is_mapping_source and self.output_schema == self.PASSTHROUGH:
            packaged = {}
            for key, value in source.items():
                if not isinstance(key, str):
                    raise PackagingError(
                        f"{self.full_name}: passthrough schema requires string keys in raw mapping source, "
                        f"got key {key!r} of type {type(key).__name__}"
                    )
                packaged[str(key)] = NO_VAL if value is None and self.none_is_absent else value
            return packaged

        if self.output_schema == self.PASSTHROUGH:
            raise PackagingError(
                f"{self.full_name}: passthrough schema can only be used with mapping-shaped raw outputs"
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
        """Apply the configured missing-value policy to unresolved ``NO_VAL`` fields."""
        resolved = dict(packaged)
        missing_keys = [key for key, value in resolved.items() if value is NO_VAL]

        if not missing_keys:
            return resolved

        mode = self.absent_value_mode

        if mode == self.RAISE:
            raise ValueError(
                f"{self.full_name}: packaged output is missing required field(s): {missing_keys}"
            )

        if mode == self.DROP:
            for key in missing_keys:
                resolved.pop(key, None)
            return resolved

        if mode == self.FILL:
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
