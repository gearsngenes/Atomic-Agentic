from __future__ import annotations
import logging

from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
)

from ..core.Exceptions import *
from ..core.Invokable import AtomicInvokable, ParameterMap
from ..core.Parameters import ParamSpec, extract_io
from ..core.sentinels import NO_VAL


logger = logging.getLogger(__name__)


# ───────────────────────────────────────────────────────────────────────────────
# Tool Invokable
# ───────────────────────────────────────────────────────────────────────────────
class Tool(AtomicInvokable):
    """Concrete base Tool primitive.

    This class provides a *dict-first* invocation interface around an underlying
    callable. It implements the template method::

        invoke(inputs) -> to_arg_kwarg(inputs) -> execute(args, kwargs)

    Subclasses may override the helper hook ``_build_tool_signature()`` to customize
    how parameter and return type schemas are built (e.g., from MCP metadata or remote
    agent specifications). They may also override ``to_arg_kwarg()`` and ``execute()``
    to customize invocation semantics. The public ``invoke()`` method must not be overridden.

    Schema
    ------
    The parameter schema is exposed via :attr:`parameters` as an ordered list of
    :class:`ParamSpec` objects (see :mod:`atomic_agentic.core.Parameters`).

    Each :class:`ParamSpec` is self-sufficient, containing:
      - ``name``    : ``str`` – the parameter name
      - ``index``   : ``int`` – the parameter position.
      - ``kind``    : ``str`` – one of ``POSITIONAL_ONLY``, ``POSITIONAL_OR_KEYWORD``,
        ``KEYWORD_ONLY``, ``VAR_POSITIONAL``, ``VAR_KEYWORD``.
      - ``type``    : ``str`` – a human-readable type name, e.g. ``"int"``.
      - ``default`` : optional – the default value, or the shared sentinel ``NO_VAL``
        when no default is present.

    ``return_type`` is always stored as a string.
    """

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        function: Callable[..., Any],
        name: Optional[str] = None,
        namespace: Optional[str] = None,
        description: Optional[str] = None,
    ) -> None:
        if not callable(function):
            raise ToolDefinitionError(f"Tool function must be callable, got {type(function)!r}")

        # Underlying callable and identity
        self._function: Callable[..., Any] = function
        self._namespace: str = namespace or "default"
        self._module, self._qualname = self._get_mod_qual(function)

        # Prepare name and description (AtomicInvokable requires non-empty description)
        inferred_name = name or getattr(function, "__name__", "unnamed_callable") or "unnamed_callable"
        inferred_description = (description or getattr(function, "__doc__", "") or "undescribed").strip() or "undescribed"

        # Build tool signature (template method)
        parameters, return_type = self._build_tool_signature()

        # Delegate name/description validation and schema setup to parent
        super().__init__(
            name=inferred_name,
            description=inferred_description,
            parameters=parameters,
            return_type=return_type,
        )

    # ------------------------------------------------------------------ #
    # Tool Properties
    # ------------------------------------------------------------------ #
    @property
    def namespace(self) -> str:
        return self._namespace

    @namespace.setter
    def namespace(self, value: str) -> None:
        self._namespace = value

    @property
    def function(self) -> Callable[..., Any]:
        return self._function

    @function.setter
    def function(self, func: Callable[..., Any]) -> None:
        """Update the underlying callable and refresh schema & identity."""
        if not callable(func):
            raise ToolDefinitionError(f"Tool function must be callable, got {type(func)!r}")
        self._function = func
        self._module, self._qualname = self._get_mod_qual(func)

        # Rebuild and validate the parameter schema
        parameters, return_type = self._build_tool_signature()
        
        if not isinstance(return_type, str):
            raise TypeError(
                f"{type(self).__name__}.return_type must be str, got {type(return_type)!r}"
            )

        # Update internal state
        self._parameters = parameters
        self._return_type = return_type

    @property
    def module(self) -> Optional[str]:
        return self._module

    @property
    def qualname(self) -> Optional[str]:
        return self._qualname

    @property
    def full_name(self) -> str:
        """Fully-qualified tool name of the form ``Type.namespace.name``."""
        return f"{type(self).__name__}.{self._namespace}.{self._name}"

    # ------------------------------------------------------------------ #
    # Signature Building (Template Method)
    # ------------------------------------------------------------------ #
    def _build_tool_signature(self) -> tuple[list[ParamSpec], str]:
        """Build tool signature from underlying callable.

        This is the template method that subclasses can override to build
        signatures from alternative sources (e.g., MCP schemas, remote agents).

        Base implementation extracts schema from self._function using extract_io().

        Returns
        -------
        tuple[list[ParamSpec], str]
            (parameters list, return_type string)
        """
        return extract_io(self._function)

    # ------------------------------------------------------------------ #
    # Tool Helpers
    # ------------------------------------------------------------------ #
    def _get_mod_qual(self, function: Callable[..., Any]) -> tuple[Optional[str], Optional[str]]:
        """Determine ``(module, qualname)`` for callable-based tools.

        Subclasses that do not use Python import identity (e.g. MCPProxyTool)
        should override this to return ``(None, None)``.
        """
        module = getattr(function, "__module__", None)
        qualname = getattr(function, "__qualname__", None)
        return module, qualname

    def to_arg_kwarg(self, inputs: Mapping[str, Any]) -> tuple[tuple[Any, ...], Dict[str, Any]]:
        """
        Default implementation for mapping input dicts to ``(*args, **kwargs)``.

        Summary of policy:

        - Required parameters (no ``default`` and not VAR_*) must be present in
            the input mapping or :class:`ToolInvocationError` is raised.
        - Unknown keys are accepted only if the wrapped callable declares a
            ``**kwargs`` (VAR_KEYWORD) parameter; otherwise an error is raised.
        - ``POSITIONAL_ONLY`` parameters (``/``-style) are always passed
            positionally and will be pulled out of the mapping into ``*args``.
        - ``POSITIONAL_OR_KEYWORD`` and ``KEYWORD_ONLY`` parameters are passed
            as keyword arguments (i.e., in ``**kwargs``) when present in the mapping.
        - ``VAR_POSITIONAL`` (``*args``) is handled by expecting the mapping to
            contain a sequence value under the var-positional parameter name;
            that sequence is appended to the positional ``args`` tuple.
        - ``VAR_KEYWORD`` (``**kwargs``) collects all remaining unknown keys and
            (optionally) merges with an explicit mapping provided under its own name
            in the inputs.

        Examples:

        - Given ``def f(a, b, /, c=3)`` and inputs ``{"a":1, "b":2}``, the
            function will be called as ``f(1, 2)`` (returned ``args=(1,2), kwargs={}``).
        - Given ``def g(x, *rest, **kw)`` and inputs ``{"x": 1, "rest": [2,3],
            "foo": "bar"}``, the call will be ``g(1, 2, 3, foo="bar")`` (``args=(1,2,3)``,
            ``kwargs={"foo":"bar"}``).
        - Missing required parameters (not present and without a default) raise
            ``ToolInvocationError``.

        Notes / edge-cases:

        - The method expects the input keys named exactly as the function's
            parameter names; it does not perform type coercion beyond Python's
            normal call-time behavior.
        - For ``VAR_POSITIONAL`` the input must be a sequence (list/tuple); the
            sequence's elements are appended to positional args.
        - For ``VAR_KEYWORD``, if the caller provides a mapping under the
            parameter's own name, it is merged with unknown keys (unknown keys
            take precedence if duplicate names appear).

        This method iterates parameters in list order with no sorting required.
        """
        data: Dict[str, Any] = dict(inputs)

        # Build param names set
        param_names = {spec.name for spec in self._parameters}

        # Find VAR_* specs
        varpos_spec = next((s for s in self._parameters if s.kind == "VAR_POSITIONAL"), None)
        varkw_spec = next((s for s in self._parameters if s.kind == "VAR_KEYWORD"), None)
        varpos_name = varpos_spec.name if varpos_spec else None
        varkw_name = varkw_spec.name if varkw_spec else None

        # Unknown key handling
        unknown_keys = set(data.keys()) - param_names
        if unknown_keys and varkw_name is None:
            raise ToolInvocationError(f"{self._name}: unknown parameters: {sorted(unknown_keys)}")

        # Required parameter check (exclude VAR_*)
        required_names = [
            spec.name
            for spec in self._parameters
            if spec.kind not in {"VAR_POSITIONAL", "VAR_KEYWORD"}
            and spec.default is NO_VAL
        ]
        missing = [name for name in required_names if name not in data]
        if missing:
            raise ToolInvocationError(f"{self._name}: missing required parameters: {missing}")

        args: List[Any] = []
        kwargs: Dict[str, Any] = {}

        # Process parameters IN LIST ORDER (no sorting!)
        for spec in self._parameters:
            name = spec.name
            kind = spec.kind
            has_default = spec.default is not NO_VAL
            value_provided = name in data

            if kind == "VAR_POSITIONAL":
                # Expect a sequence under this key, if provided
                if value_provided:
                    val = data[name]
                    if not isinstance(val, (list, tuple)):
                        raise ToolInvocationError(
                            f"{self._name}: var-positional parameter '{name}' must be a list or tuple"
                        )
                    args.extend(list(val))
                continue

            if kind == "VAR_KEYWORD":
                # Handled after all named parameters
                continue

            if not value_provided and has_default:
                val = spec.default
            elif value_provided:
                val = data[name]
            else:
                # Required parameters should already have been validated above
                continue

            if kind == "POSITIONAL_ONLY":
                args.append(val)
            else:
                # POS_OR_KEYWORD or KEYWORD_ONLY – pass as keyword argument
                kwargs[name] = val

        # Handle VAR_KEYWORD by collecting unknown keys + any explicit mapping under its own name
        if varkw_name is not None:
            extra_kwargs: Dict[str, Any] = {}
            # If caller explicitly supplied a mapping for the VAR_KEYWORD parameter name, merge it.
            if varkw_name in data:
                explicit = data[varkw_name]
                if not isinstance(explicit, Mapping):
                    raise ToolInvocationError(
                        f"{self._name}: var-keyword parameter '{varkw_name}' must be a mapping if provided"
                    )
                extra_kwargs.update(dict(explicit))
            # Include unknown keys
            for key in unknown_keys:
                if key in extra_kwargs or key in kwargs:
                    raise ToolInvocationError(
                        f"{self._name}: duplicate key '{key}' in **kwargs aggregation"
                    )
                extra_kwargs[key] = data[key]
            kwargs.update(extra_kwargs)

        return tuple(args), kwargs

    def execute(self, args: tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        """Execute the underlying callable.

        Subclasses may override this to change *how* a tool is executed (for
        example, by making a remote MCP call or invoking an Agent), but should
        not change the high-level semantics.
        """
        try:
            result = self._function(*args, **kwargs)
        except Exception as e:  # pragma: no cover - thin wrapper
            raise ToolInvocationError(f"{self.full_name}: invocation failed: {e}") from e
        return result

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def invoke(self, inputs: Mapping[str, Any]) -> Any:
        """Invoke the tool using a dict-like mapping of inputs.

        This is the *only* public entrypoint for execution. Subclasses must not
        override this method; instead they can customise :meth:`to_arg_kwarg`
        and :meth:`execute`.
        """
        logger.info(f"[{self.full_name} started]")
        if not isinstance(inputs, Mapping):
            raise ToolInvocationError(f"{self._name}: inputs must be a mapping")
        args, kwargs = self.to_arg_kwarg(inputs)
        result = self.execute(args, kwargs)
        logger.info(f"[{self.full_name} finished]")
        return result

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        """Serialize this tool's header and argument schema.

        Note that deserialisation will handled by a future :mod:`Factory`; this method does
        *not* perform any persistibility checks and will not raise solely
        because :attr:`is_persistible` is ``False``.
        """
        d = super().to_dict()
        d.update({
            "namespace": self.namespace,
            "module": self.module,
            "qualname": self.qualname,
        })
        return d
