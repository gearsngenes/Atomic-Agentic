from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, Dict
import re
import threading

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
        filter_extraneous_inputs: bool = False,
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
        self._filter_extraneous_inputs = filter_extraneous_inputs
        # invoke lock
        self._invoke_lock = threading.RLock()

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
    def return_type(self) -> str:
        """Return type (string) of this invokable."""
        return self._return_type

    @property
    def filter_extraneous_inputs(self) -> bool:
        """Whether to filter extraneous inputs not used by the component's parameters."""
        return self._filter_extraneous_inputs
    
    @filter_extraneous_inputs.setter
    def filter_extraneous_inputs(self, value: bool) -> None:
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

    # ---------------------------------------------------------------- #
    # Input Filtering and Validation
    # ---------------------------------------------------------------- #
    def filter_inputs(self, inputs: Mapping[str, Any]) -> Dict[str, Any]:
        """
        The standardized superclass self.invoke that enforces the dict-first contract and handles extraneous input filtering.
        """
        if not isinstance(inputs, Mapping):
            raise TypeError(f"{type(self).__name__}.invoke: inputs must be a mapping, got {type(inputs)!r}")
        # Check if the component has varargs or kwargs parameters
        has_varparams = any(p.kind in ("VAR_POSITIONAL", "VAR_KEYWORD") for p in self.parameters)
        params = [p.name for p in self.parameters]

        # If filter_extraneous_inputs is True and the component does not have varargs/kwargs, filter out extraneous inputs
        if self._filter_extraneous_inputs:
            if not has_varparams:
                # Filter out extraneous inputs not in the component's parameters
                inputs = {k: v for k, v in inputs.items() if k in params}
        # otherwise, pass all inputs through (including extraneous ones)
        # and let the component handle them (e.g. by raising an error if unexpected)
        return inputs

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
