from __future__ import annotations
import logging

import inspect
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
        filter_extraneous_inputs: bool = True,
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
            filter_extraneous_inputs=filter_extraneous_inputs,
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
        Map dict-first inputs to (*args, **kwargs) for the wrapped callable.

        Binding policy
        --------------
        Normal mode (no explicit varargs payload present):
        - POSITIONAL_ONLY parameters -> args
        - POSITIONAL_OR_KEYWORD and KEYWORD_ONLY parameters -> kwargs

        Varargs mode (explicit VAR_POSITIONAL payload present):
        - POSITIONAL_ONLY and POSITIONAL_OR_KEYWORD parameters -> args
        - then append the explicit varargs payload
        - KEYWORD_ONLY parameters -> kwargs

        Common rules
        ------------
        - Unknown input keys raise.
        - Explicit VAR_POSITIONAL payload must be a non-string sequence.
        - Explicit VAR_KEYWORD payload must be a mapping.
        - Duplicate keys introduced by explicit **kwargs raise.
        """

        if not isinstance(inputs, Mapping):
            raise ToolInvocationError(f"{self.full_name}: inputs must be a mapping")

        data: Dict[str, Any] = dict(inputs)
        param_specs = self._parameters
        param_names = {spec.name for spec in param_specs}

        vararg_spec = next((p for p in param_specs if p.kind == "VAR_POSITIONAL"), None)
        varkw_spec = next((p for p in param_specs if p.kind == "VAR_KEYWORD"), None)

        positional_only = [p for p in param_specs if p.kind == "POSITIONAL_ONLY"]
        pos_or_kw = [p for p in param_specs if p.kind == "POSITIONAL_OR_KEYWORD"]
        kw_only = [p for p in param_specs if p.kind == "KEYWORD_ONLY"]

        has_explicit_varargs = vararg_spec is not None and vararg_spec.name in data

        # ------------------------------------------------------------------
        # Reject unknown keys
        # ------------------------------------------------------------------
        unknown_keys = sorted(set(data.keys()) - param_names)
        if unknown_keys:
            raise ToolInvocationError(
                f"{self.full_name}: unknown parameters: {unknown_keys}"
            )

        args: List[Any] = []
        kwargs: Dict[str, Any] = {}
        missing: List[str] = []

        # ------------------------------------------------------------------
        # Phase 1: POSITIONAL_ONLY -> args
        # ------------------------------------------------------------------
        for spec in positional_only:
            if spec.name in data:
                args.append(data[spec.name])
            elif spec.default is not NO_VAL:
                args.append(spec.default)
            else:
                missing.append(spec.name)

        # ------------------------------------------------------------------
        # Phase 2: POSITIONAL_OR_KEYWORD binding
        # ------------------------------------------------------------------
        if has_explicit_varargs:
            for spec in pos_or_kw:
                if spec.name in data:
                    args.append(data[spec.name])
                elif spec.default is not NO_VAL:
                    args.append(spec.default)
                else:
                    missing.append(spec.name)
        else:
            for spec in pos_or_kw:
                if spec.name in data:
                    kwargs[spec.name] = data[spec.name]
                elif spec.default is not NO_VAL:
                    kwargs[spec.name] = spec.default
                else:
                    missing.append(spec.name)

        # ------------------------------------------------------------------
        # Phase 3: KEYWORD_ONLY -> kwargs
        # ------------------------------------------------------------------
        for spec in kw_only:
            if spec.name in data:
                kwargs[spec.name] = data[spec.name]
            elif spec.default is not NO_VAL:
                kwargs[spec.name] = spec.default
            else:
                missing.append(spec.name)

        if missing:
            raise ToolInvocationError(
                f"{self.full_name}: missing required parameters: {missing}"
            )

        # ------------------------------------------------------------------
        # Phase 4: explicit *args payload
        # ------------------------------------------------------------------
        if has_explicit_varargs:
            raw_varargs = data[vararg_spec.name]

            if isinstance(raw_varargs, (str, bytes, bytearray)):
                raise ToolInvocationError(
                    f"{self.full_name}: var-positional parameter '{vararg_spec.name}' "
                    "must be a non-string sequence"
                )

            if not isinstance(raw_varargs, (list, tuple)):
                raise ToolInvocationError(
                    f"{self.full_name}: var-positional parameter '{vararg_spec.name}' "
                    "must be a list or tuple"
                )

            args.extend(raw_varargs)

        # ------------------------------------------------------------------
        # Phase 5: explicit **kwargs payload
        # ------------------------------------------------------------------
        if varkw_spec is not None and varkw_spec.name in data:
            raw_varkw = data[varkw_spec.name]

            if not isinstance(raw_varkw, Mapping):
                raise ToolInvocationError(
                    f"{self.full_name}: var-keyword parameter '{varkw_spec.name}' "
                    "must be a mapping if provided"
                )

            for key, value in raw_varkw.items():
                if key in kwargs:
                    raise ToolInvocationError(
                        f"{self.full_name}: duplicate key '{key}' in explicit **kwargs mapping"
                    )
                kwargs[key] = value

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

    async def async_execute(
        self,
        args: tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Any:
        """
        Async execution hook for the underlying callable.

        - If the callable returns an awaitable, await it.
        - Otherwise, return the plain result directly.
        """
        try:
            result = self._function(*args, **kwargs)
            if inspect.isawaitable(result):
                result = await result
        except Exception as e:  # pragma: no cover - thin wrapper
            raise ToolInvocationError(f"{self.full_name}: invocation failed: {e}") from e
        return result

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    async def async_invoke(self, inputs: Mapping[str, Any]) -> Any:
        """
        Async analog of invoke() for tools.

        Mirrors the current sync Tool.invoke flow:
        - filter inputs
        - bind to args/kwargs
        - execute
        """
        logger.info(f"[Async {self.full_name} started]")
        inputs = self.filter_inputs(inputs)
        args, kwargs = self.to_arg_kwarg(inputs)
        result = await self.async_execute(args, kwargs)
        logger.info(f"[Async {self.full_name} finished]")
        return result

    def invoke(self, inputs: Mapping[str, Any]) -> Any:
        """Invoke the tool using a dict-like mapping of inputs.

        This is the *only* public entrypoint for execution. Subclasses must not
        override this method; instead they can customise :meth:`to_arg_kwarg`
        and :meth:`execute`.
        """
        with self._invoke_lock:
            logger.info(f"[{self.full_name} started]")
            # Filter inputs
            inputs = self.filter_inputs(inputs)
            # Separate positional and keyword arguments according to the tool's parameter schema
            args, kwargs = self.to_arg_kwarg(inputs)
            # Execute and return result
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
