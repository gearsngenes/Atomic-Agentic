from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, Dict
import re
import threading
import asyncio
from uuid import uuid4

from .sentinels import NO_VAL
from .Parameters import ParamSpec, is_valid_parameter_order

# Canonical mapping of parameter name -> ParamSpec (legacy, for backward compat)
ParameterMap = dict[str, ParamSpec]

# Legacy aliases for backward compatibility
ArgumentMap = ParameterMap  # Deprecated: use ParameterMap instead
ArgSpec = ParamSpec  # Deprecated: use ParamSpec instead

# Valid name: like a Python identifier (letters, digits, underscores), must not start with a digit
_VALID_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class AtomicInvokable(ABC):
    """
    Basal invokable contract for Tools, Agents, and Workflows.

    Overview
    --------
    `AtomicInvokable` defines the minimal, language-level contract that all
    executable primitives in this codebase must satisfy. It standardises:

    - **identity**: validated `name` and `description` (human-friendly strings).
    - **interface**: a single execution entrypoint ``invoke(inputs: Mapping[str, Any])``.
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
      - Converting the dict to appropriate ``(*args, **kwargs)`` for execution
      - Raising clear, typed exceptions on invalid inputs (use
        ``ToolInvocationError``, ``AgentInvocationError``, etc.)

    Architecture Notes
    -------------------
    - The class intentionally does not expose a human-readable ``signature`` string
      inside ``to_dict()`` to minimize churn when persisting metadata; use the
      ``signature`` property for logging and UIs.
    - **Backward Compatibility**: Legacy aliases ``ArgumentMap`` and ``ArgSpec`` are
      defined module-level but **DEPRECATED**. Use ``ParameterMap`` and ``ParamSpec``
      in new code.
    """

    def __init__(
        self,
        *,
        name: str,
        description: str,
        parameters: list[ParamSpec],
        return_type: str,
        filter_extraneous_inputs: bool = True,
    ) -> None:
        # setters include validation
        self.name = name
        self.description = description

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
            if not _VALID_NAME.match(p.name):
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
        is_valid_parameter_order(parameters)

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
        if not _VALID_NAME.match(value):
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
    def full_name(self) -> str:
        """Fully-qualified name (for logging)."""
        return f"{type(self).__name__}.{self.name}"
    
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
        return any(p.kind == "VAR_POSITIONAL" for p in self._parameters)

    @property
    def has_varkwargs(self) -> bool:
        """Whether this invokable accepts variable keyword arguments (**kwargs)."""
        return any(p.kind == "VAR_KEYWORD" for p in self._parameters)

    @property
    def return_type(self) -> str:
        """Return type (string) of this invokable."""
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
    # Legacy backward compatibility properties
    # ---------------------------------------------------------------- #
    @property
    def parameters_map(self) -> ParameterMap:
        """Legacy alternative viewing mechanism: parameters as a dict mapping name -> ParamSpec.

        This property is provided for backward compatibility. New code should prefer
        the ``parameters`` property which provides the ordered list directly.
        """
        return {spec.name: spec for spec in self._parameters}

    @property
    def arguments_map(self) -> ArgumentMap:
        """Deprecated legacy property. Use ``parameters_map`` or ``parameters`` instead."""
        return self.parameters_map

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
            
            if spec.kind == "VAR_POSITIONAL":
                params.append(f"*{spec.name}: {ptype}{default_marker}")
            elif spec.kind == "VAR_KEYWORD":
                params.append(f"**{spec.name}: {ptype}{default_marker}")
            else:
                params.append(f"{spec.name}: {ptype}{default_marker}")
        
        params_str = ", ".join(params)
        return f"{self.full_name}({params_str}) -> {self.return_type}"

    # ---------------------------------------------------------------- #
    # Abstract contract
    # ---------------------------------------------------------------- #
    @abstractmethod
    def invoke(self, inputs: Mapping[str, Any]) -> Any:
        """Perform work."""
        raise NotImplementedError

    async def async_invoke(self, inputs: Mapping[str, Any]) -> Any:
        """
        Default async compatibility wrapper.

        This preserves the current sync-first implementation by running
        `invoke(inputs)` in a worker thread.
        """
        return await asyncio.to_thread(self.invoke, inputs)

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
    def __call__(self, *args, **kwargs)-> Any:
        """
        Allows the invokable to be called like a regular function
        Check for varargs/kwargs parameters and construct the inputs dict accordingly before invoking.
        """
        inputs = self._to_dict_input_conversion(*args, **kwargs)
        return self.invoke(inputs)
    
    async def async_call(self, *args: Any, **kwargs: Any) -> Any:
        """
        Async analog of __call__:
        bind normal call-style args/kwargs into the dict-first inputs shape,
        then delegate to async_invoke().
        """
        inputs = self._to_dict_input_conversion(*args, **kwargs)

        return await self.async_invoke(inputs)
    
    def _to_dict_input_conversion(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """
        Convert Python call-style (*args, **kwargs) into the dict-first input shape.

        This method is intentionally Python-call-like:
        - extra positional arguments populate VAR_POSITIONAL
        - unknown keyword arguments populate VAR_KEYWORD
        - explicit VAR_POSITIONAL / VAR_KEYWORD field names are not accepted as
        keyword arguments in call-style mode; use invoke({...}) for that.
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

        vararg_name = vararg_spec.name if vararg_spec is not None else None
        varkwarg_name = varkwarg_spec.name if varkwarg_spec is not None else None
        variadic_field_names = {
            name for name in (vararg_name, varkwarg_name) if name is not None
        }

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

            if key in variadic_field_names:
                raise TypeError(
                    f"{self.full_name} got variadic parameter field {key!r} as a keyword; "
                    "pass variadic values positionally or use invoke({...}) for explicit "
                    "dict-first payloads"
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
