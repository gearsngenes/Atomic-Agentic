from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from dataclasses import replace
from typing import Any, ClassVar, Iterable, Mapping, Dict
import logging

from ..constants.core import DEFAULT_LOG_LEVEL, IDENTIFIER_PATTERN, NO_VAL
from ..models.parameters import ParamSpec
from ..utils.parameters import _validate_parameter_order
from ..models.results import AtomicResult


logger = logging.getLogger(__name__)

# Deprecated aliases removed in v2: use `ParamSpec` and the `parameters` property.


class AtomicInvokable(ABC):
    """
    Basal invokable contract for Tools, Agents, and Workflows.

    Overview
    --------
    `AtomicInvokable` defines the minimal, language-level contract that all
    executable primitives in this codebase must satisfy. It standardises:

    - **identity**: validated `name` and `description` (human-friendly strings).
    - **interface**: ``invoke(inputs: Mapping[str, Any])`` and its async twin,
      returning an ``AtomicResult``-family successful-invocation envelope.
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

    Mutable defaults
    ----------------
    Names, parameter names, kinds, types and the return type are fixed at
    construction. Defaults are the exception: ``set_defaults`` and
    ``remove_defaults`` change which parameters are required, which is not part of
    the I/O shape. The schema default is authoritative at call time
    (``filter_inputs`` fills it); a wrapped callable's own defaults are untouched.
    A change is local to this invokable and never reaches components it wraps.
    Limitation: a default changed after the invokable is hosted over A2A or after a
    client connects is not propagated to the remote side's advertised schema.

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

    Invocation template
    -------------------
    ``invoke`` / ``async_invoke`` are concrete templates that own start/end
    logging (at ``log_level``), input filtering, timing and result construction.
    Subclasses implement ``_invoke_impl`` and ``_async_invoke_impl``, which
    receive the already-filtered inputs and return a payload dict containing
    ``result`` (plus any extra fields of the subclass's ``result_cls``). The
    template builds ``result_cls(invoker_id=full_name, started_at, ended_at,
    **payload)``. Exceptions from filtering or the hook propagate unchanged.

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

    result_cls: ClassVar[type[AtomicResult]] = AtomicResult

    def __init__(
        self,
        *,
        name: str,
        namespace: str,
        description: str,
        parameters: list[ParamSpec],
        return_type: str,
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be a non-empty string")
        if not IDENTIFIER_PATTERN.fullmatch(name):
            raise ValueError(
                f"name must be alphanumeric/underscore and not start with a digit; got {name!r}"
            )
        self._name = name
        self.__name__ = self._name
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
            if p.name == "return_atomic_result_object":
                raise TypeError(
                    f"{type(self).__name__}: parameter name 'return_atomic_result_object' is "
                    "reserved for the __call__/async_call convenience API and cannot be used "
                    "as a declared parameter name."
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

        self._parameters: list[ParamSpec] = list(parameters)
        self._return_type: str = return_type
        self._log_level: int | None = DEFAULT_LOG_LEVEL

    # ---------------------------------------------------------------- #
    # Name + description with validation
    # ---------------------------------------------------------------- #
    @property
    def name(self) -> str:
        """The canonical name of this invokable."""
        return self._name

    @property
    def description(self) -> str:
        """
        Human-friendly description, augmented with per-parameter bullets and
        subclass/wrapper extras when present.

        1. Start from `self._description` as the base text.
        2. Collect parameters with a non-None `.description`, in declaration
           order.
        3. If any exist, append a blank-line-separated `- name: description`
           bullet block; otherwise this step is a no-op.
        4. Call `self._extra_description()`; if non-empty, append it with a
           single newline separator; otherwise this step is a no-op.
        """
        text = self._description

        # Step 2-3: conditional per-parameter bullet block
        described = [p for p in self._parameters if p.description is not None]
        if described:
            bullets = "\n".join(f"- {p.name}: {p.description}" for p in described)
            text = f"{text}\n\n{bullets}"

        # Step 4: conditional subclass/wrapper extra content
        extra = self._extra_description()
        if extra:
            text = f"{text}\n{extra}"

        return text

    @description.setter
    def description(self, value: str) -> None:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("description must be a non-empty string")
        self._description = value.strip()
        self.__doc__ = self._description

    def _extra_description(self) -> str:
        """
        Override point for wrapper/subclass-specific description additions.

        Default is `""` (no extra content). Subclasses that wrap another
        `AtomicInvokable` instance by composition (e.g. `self._function`) must chain into *that wrapped instance's*
        `_extra_description()` — never `super()` — since wrapping is
        composition, not inheritance, and `super()` cannot reach an unrelated
        class's override. Ordinary (non-wrapper) subclasses may use
        `super()._extra_description()` normally to preserve an ancestor
        class's own extra content.
        """
        return ""

    @property
    def namespace(self) -> str:
        """Grouping label for this invokable, used as the middle segment of ``full_name``."""
        return self._namespace

    @property
    def full_name(self) -> str:
        """Fully-qualified name of the form ``Type.namespace.name``."""
        return f"{type(self).__name__}.{self._namespace}.{self._name}"

    @property
    def log_level(self) -> int | None:
        """Level for start/end invocation logs; None disables them."""
        return self._log_level

    @log_level.setter
    def log_level(self, value: int | None) -> None:
        if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
            raise TypeError(
                f"log_level must be an int logging level or None, got {value!r}"
            )
        self._log_level = value

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
    # Default mutators
    # ---------------------------------------------------------------- #
    def set_defaults(self, defaults: Mapping[str, Any]) -> None:
        """Give each named parameter the mapped default (replacing any existing one).

        All-or-nothing: the resulting schema is validated as a whole before
        anything changes, so a batch may set a positional parameter and the ones
        after it together. Values are shared by reference and are not type-checked;
        ``NO_VAL`` as a value leaves that parameter with no default.

        Raises ``TypeError`` for a non-mapping, ``ValueError`` for unknown or
        variadic names, and ``SchemaError`` if the result breaks parameter ordering.
        """
        if not isinstance(defaults, Mapping):
            raise TypeError(
                f"{type(self).__name__}.set_defaults: defaults must be a mapping, "
                f"got {type(defaults)!r}"
            )
        if not defaults:
            return None

        specs = self.parameters_map
        unknown = [name for name in defaults if name not in specs]
        if unknown:
            raise ValueError(
                f"{type(self).__name__}.set_defaults: unknown parameter(s) {unknown!r}"
            )
        variadic = [
            name for name in defaults
            if specs[name].kind in (ParamSpec.VAR_POSITIONAL, ParamSpec.VAR_KEYWORD)
        ]
        if variadic:
            raise ValueError(
                f"{type(self).__name__}.set_defaults: variadic parameter(s) "
                f"{variadic!r} cannot take a default"
            )

        updated = [
            replace(spec, default=defaults[spec.name]) if spec.name in defaults else spec
            for spec in self._parameters
        ]
        _validate_parameter_order(updated)  # SchemaError propagates; self unchanged
        self._parameters = updated  # the only mutation, one rebind

    def remove_defaults(self, names: Iterable[str]) -> None:
        """Make each named parameter required again.

        Same all-or-nothing validation as ``set_defaults``. A name with no
        default raises, so a typo cannot silently do nothing. Removing a default
        also overrides a default the wrapped callable has on its own: the schema
        decides what is required.

        Raises ``TypeError`` for a bare string or non-iterable, and ``ValueError``
        for unknown, variadic or default-less names.
        """
        if isinstance(names, (str, bytes)):
            raise TypeError(
                f"{type(self).__name__}.remove_defaults: names must be an iterable of "
                "parameter names, not a bare string"
            )
        names = list(names)
        if not names:
            return None

        specs = self.parameters_map
        no_default = [
            name for name in names
            if name in specs
            and specs[name].kind not in (ParamSpec.VAR_POSITIONAL, ParamSpec.VAR_KEYWORD)
            and specs[name].default is NO_VAL
        ]
        if no_default:
            raise ValueError(
                f"{type(self).__name__}.remove_defaults: parameter(s) {no_default!r} "
                "have no default"
            )

        # Unknown and variadic names raise in set_defaults, which also validates ordering.
        self.set_defaults({name: NO_VAL for name in names})

    # ---------------------------------------------------------------- #
    # Signature helper
    # ---------------------------------------------------------------- #
    @property
    def signature(self) -> str:
        """
        Returns a signature like:
            name(a: T, b: T, /, c: T, d: T = 1, *args: T, e: T, f: T = 2, **kwargs: T) -> return_type

        Mirrors ``inspect.Signature.__str__``'s own convention: a bare
        ``/`` marks the end of a run of POSITIONAL_ONLY parameters (only
        when at least one preceded it), and a bare ``*`` marks the start
        of KEYWORD_ONLY parameters -- but only when no VAR_POSITIONAL
        (``*name``) already opened keyword-only territory, since a second
        bare ``*`` right after it would be redundant/invalid.
        """
        params = []
        emitted_positional_only = False
        seen_var_positional = False
        emitted_keyword_only_marker = False

        for spec in self._parameters:
            ptype = " | ".join(spec.type)
            default_marker = ""
            if spec.default is not NO_VAL:
                default_marker = f" = {spec.default!r}"

            if spec.kind == ParamSpec.POSITIONAL_ONLY:
                emitted_positional_only = True
            elif emitted_positional_only:
                params.append("/")
                emitted_positional_only = False

            if (
                spec.kind == ParamSpec.KEYWORD_ONLY
                and not seen_var_positional
                and not emitted_keyword_only_marker
            ):
                params.append("*")
                emitted_keyword_only_marker = True

            if spec.kind == ParamSpec.VAR_POSITIONAL:
                seen_var_positional = True
                params.append(f"*{spec.name}: {ptype}{default_marker}")
            elif spec.kind == ParamSpec.VAR_KEYWORD:
                params.append(f"**{spec.name}: {ptype}{default_marker}")
            else:
                params.append(f"{spec.name}: {ptype}{default_marker}")

        if emitted_positional_only:
            # Every parameter was POSITIONAL_ONLY -- the closing "/" never
            # got emitted inside the loop above (it only fires on the
            # *next* differently-kinded parameter), so emit it here.
            params.append("/")

        params_str = ", ".join(params)
        return f"{self.name}({params_str}) -> {self.return_type}"

    @property
    def fullname_signature(self) -> str:
        """
        Returns a signature like:

            ClassName.name(a: T, b: T, /, c: T, d: T = 1, *args: T, e: T, f: T = 2, **kwargs: T) -> return_type

        Mirrors ``inspect.Signature.__str__``'s own convention: a bare
        ``/`` marks the end of a run of POSITIONAL_ONLY parameters (only
        when at least one preceded it), and a bare ``*`` marks the start
        of KEYWORD_ONLY parameters -- but only when no VAR_POSITIONAL
        (``*name``) already opened keyword-only territory, since a second
        bare ``*`` right after it would be redundant/invalid.
        """
        params = []
        emitted_positional_only = False
        seen_var_positional = False
        emitted_keyword_only_marker = False

        for spec in self._parameters:
            ptype = " | ".join(spec.type)
            default_marker = ""
            if spec.default is not NO_VAL:
                default_marker = f" = {spec.default!r}"

            if spec.kind == ParamSpec.POSITIONAL_ONLY:
                emitted_positional_only = True
            elif emitted_positional_only:
                params.append("/")
                emitted_positional_only = False

            if (
                spec.kind == ParamSpec.KEYWORD_ONLY
                and not seen_var_positional
                and not emitted_keyword_only_marker
            ):
                params.append("*")
                emitted_keyword_only_marker = True

            if spec.kind == ParamSpec.VAR_POSITIONAL:
                seen_var_positional = True
                params.append(f"*{spec.name}: {ptype}{default_marker}")
            elif spec.kind == ParamSpec.VAR_KEYWORD:
                params.append(f"**{spec.name}: {ptype}{default_marker}")
            else:
                params.append(f"{spec.name}: {ptype}{default_marker}")

        if emitted_positional_only:
            # Every parameter was POSITIONAL_ONLY -- the closing "/" never
            # got emitted inside the loop above (it only fires on the
            # *next* differently-kinded parameter), so emit it here.
            params.append("/")

        params_str = ", ".join(params)
        return f"{self.full_name}({params_str}) -> {self.return_type}"

    # ---------------------------------------------------------------- #
    # Abstract contract
    # ---------------------------------------------------------------- #
    def invoke(self, inputs: Mapping[str, Any]) -> AtomicResult:
        """
        Run one invocation through the template: stamp start, log, filter inputs,
        run ``_invoke_impl``, stamp end, log, build ``result_cls``. Exceptions
        propagate unchanged and suppress the end log.
        """
        started_at = datetime.now(timezone.utc)
        if self._log_level is not None:
            logger.log(self._log_level, "[%s started]", self.full_name)
        filtered = self.filter_inputs(inputs)
        payload = self._invoke_impl(filtered)
        ended_at = datetime.now(timezone.utc)
        if self._log_level is not None:
            logger.log(self._log_level, "[%s finished]", self.full_name)
        return self.result_cls(
            invoker_id=self.full_name,
            started_at=started_at,
            ended_at=ended_at,
            **payload,
        )

    async def async_invoke(self, inputs: Mapping[str, Any]) -> AtomicResult:
        """Async twin of ``invoke``; awaits ``_async_invoke_impl``."""
        started_at = datetime.now(timezone.utc)
        if self._log_level is not None:
            logger.log(self._log_level, "[Async %s started]", self.full_name)
        filtered = self.filter_inputs(inputs)
        payload = await self._async_invoke_impl(filtered)
        ended_at = datetime.now(timezone.utc)
        if self._log_level is not None:
            logger.log(self._log_level, "[Async %s finished]", self.full_name)
        return self.result_cls(
            invoker_id=self.full_name,
            started_at=started_at,
            ended_at=ended_at,
            **payload,
        )

    @abstractmethod
    def _invoke_impl(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Do the work on filtered inputs; return a payload dict (``result`` required)."""
        raise NotImplementedError

    @abstractmethod
    async def _async_invoke_impl(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Async twin of ``_invoke_impl``."""
        raise NotImplementedError

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
        - Unknown keys are dropped when no VAR_KEYWORD exists.
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

        # inject defaults for absent non-variadic params
        for param in parameters:
            if param.kind in {ParamSpec.VAR_POSITIONAL, ParamSpec.VAR_KEYWORD}:
                continue
            if param.name not in filtered and param.default is not NO_VAL:
                filtered[param.name] = param.default

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
            "name": self.name,
            "namespace": self.namespace,
            "description": self._description,
            "parameters": [spec.to_dict() for spec in self._parameters],
            "return_type": self.return_type,
        }

    # ---------------------------------------------------------------- #
    # Unified repr/str
    # ---------------------------------------------------------------- #
    def __str__(self) -> str:
        """Constructor-style dump using the augmented `description` getter
        (per-parameter bullets and any `_extra_description()` included) for
        the most comprehensive human-readable detail. Unlike `to_dict()`,
        which uses raw `_description` to avoid duplicating bullets in
        persisted metadata, `__str__` favors completeness over compactness."""
        return (
            f"{type(self).__name__}("
            f"name={self.name!r}, "
            f"namespace={self.namespace!r}, "
            f"parameters={self.parameters!r}, "
            f"return_type={self.return_type!r}, "
            f"description={self.description!r})"
        )

    def __repr__(self) -> str:
        """Short, single-line, unambiguous form. `full_name` already leads
        with the class name (`Type.namespace.name`), so no separate class
        name prefix is added."""
        return f"<{self.full_name}>"

    # ---------------------------------------------------------------- #
    # callable contract
    # ---------------------------------------------------------------- #
    def __call__(
        self,
        *args: Any,
        return_atomic_result_object: bool = False,
        **kwargs: Any,
    ) -> Any:
        """
        Allows the invokable to be called like a regular function.

        Check for varargs/kwargs parameters and construct the inputs dict
        accordingly before invoking. By default returns the unwrapped
        caller-facing payload (``.result``), not the ``AtomicResult``
        envelope — pass ``return_atomic_result_object=True`` to get the full
        envelope back, matching ``invoke(...)``'s return shape.
        ``return_atomic_result_object`` is consumed here directly and never
        reaches ``invoke(...)``.
        """
        inputs = self._args_kwargs_to_dict(*args, **kwargs)
        result = self.invoke(inputs)
        if return_atomic_result_object:
            return result
        return self._unwrap_result_payload(result)

    async def async_call(
        self,
        *args: Any,
        return_atomic_result_object: bool = False,
        **kwargs: Any,
    ) -> Any:
        """
        Async analog of __call__.

        Bind normal call-style args/kwargs into the dict-first inputs shape,
        then delegate to async_invoke(). Same unwrap-by-default behavior as
        __call__.
        """
        inputs = self._args_kwargs_to_dict(*args, **kwargs)
        result = await self.async_invoke(inputs)
        if return_atomic_result_object:
            return result
        return self._unwrap_result_payload(result)

    @staticmethod
    def _unwrap_result_payload(value: Any) -> Any:
        """
        Return the caller-facing payload from an ``AtomicResult``-family value.

        If ``value`` is an ``AtomicResult`` instance, returns ``value.result``.
        Non-result values are returned unchanged.
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
