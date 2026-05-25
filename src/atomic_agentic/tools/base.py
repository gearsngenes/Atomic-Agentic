from __future__ import annotations

import asyncio
import inspect
import logging
from typing import (
    Any,
    Callable,
    Dict,
    Mapping,
    Optional,
)

from ..core.Exceptions import ToolDefinitionError, ToolInvocationError
from ..core.Invokable import AtomicInvokable
from ..core.Parameters import ParamSpec, extract_io
from ..core.utils import run_coro_sync


logger = logging.getLogger(__name__)


# ───────────────────────────────────────────────────────────────────────────────
# Tool Invokable
# ───────────────────────────────────────────────────────────────────────────────
class Tool(AtomicInvokable):
    """Concrete base Tool primitive.

    ``Tool`` provides a dict-first invocation interface around either a plain
    Python callable or another ``AtomicInvokable``. It implements the template
    method::

        invoke(inputs) -> to_arg_kwarg(inputs) -> execute(args, kwargs)

    Plain callable-backed tools derive their schema from Python signature
    introspection through ``extract_io(...)``. Invokable-backed tools instead
    reuse the wrapped invokable's declared ``parameters`` and ``return_type`` so
    the exposed Tool schema matches the wrapped component's public contract.

    Subclasses may override ``_build_tool_signature()`` to customize how
    parameter and return type schemas are built, such as from MCP metadata or
    remote agent metadata. They may also override ``to_arg_kwarg()``,
    ``execute()``, or ``async_execute()`` to customize transport or invocation
    semantics. The public ``invoke()`` method should not be overridden.

    Schema
    ------
    The parameter schema is exposed via :attr:`parameters` as an ordered list of
    :class:`ParamSpec` objects.

    Each :class:`ParamSpec` is self-sufficient, containing:

    - ``name``: parameter name
    - ``index``: parameter position
    - ``kind``: one of ``POSITIONAL_ONLY``, ``POSITIONAL_OR_KEYWORD``,
      ``KEYWORD_ONLY``, ``VAR_POSITIONAL``, or ``VAR_KEYWORD``
    - ``type``: human-readable type name, such as ``"int"``
    - ``default``: default value, or the shared ``NO_VAL`` sentinel when no
      default is present

    Execution
    ---------
    Sync execution calls the stored function target. For invokable-backed tools,
    this uses the wrapped invokable's ``__call__`` path. Async execution calls
    ``async_call(...)`` for invokable-backed tools, awaits native async callables
    directly, and offloads sync callables to a worker thread.

    Serialization
    -------------
    ``to_dict()`` includes ``wraps_invokable``. When true, it also includes the
    wrapped invokable's own ``to_dict()`` output under ``"invokable_function"``
    for transparency.
    """

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        function: AtomicInvokable | Callable[..., Any],
        name: Optional[str] = None,
        namespace: Optional[str] = None,
        description: Optional[str] = None,
        filter_extraneous_inputs: bool = True,
    ) -> None:
        """Initialize a Tool from a plain callable or AtomicInvokable.

        Parameters
        ----------
        function:
            Plain callable or ``AtomicInvokable`` to expose as a Tool.

            - Plain callables are introspected with ``extract_io(...)``.
            - Invokable-backed tools reuse the invokable's declared schema and
              call through the invokable's normal call-style API.

        name:
            Optional Tool name override. If omitted, plain callables use
            ``function.__name__`` and invokables use ``function.name``.

        namespace:
            Optional Tool namespace. Defaults to ``"default"``.

        description:
            Optional Tool description override. If omitted, plain callables use
            their docstring or a fallback description, and invokables use
            ``function.description``.

        filter_extraneous_inputs:
            Whether unknown inputs are filtered before invocation when no
            ``VAR_KEYWORD`` parameter is present.
        """
        if not callable(function):
            raise ToolDefinitionError(f"Tool function must be callable, got {type(function)!r}")

        # Underlying callable or invokable-backed execution target and identity.
        self._function: AtomicInvokable | Callable[..., Any] = function
        self._namespace: str = namespace or "default"
        self._module, self._qualname = self._get_mod_qual(function)

        inferred_name = (
            name.strip()
            if isinstance(name, str) and name.strip()
            else None
        )
        inferred_description = (
            description.strip()
            if isinstance(description, str) and description.strip()
            else None
        )

        # Prepare name and description. AtomicInvokable parent validation
        # requires both to resolve to non-empty strings.
        if isinstance(function, AtomicInvokable):
            inferred_name = inferred_name or function.name
            inferred_description = inferred_description or function.description
        else:
            inferred_name = (
                inferred_name
                or getattr(function, "__name__", None)
                or "unnamed_callable"
            )

            doc = getattr(function, "__doc__", None)
            inferred_description = inferred_description or (
                doc.strip()
                if isinstance(doc, str) and doc.strip()
                else "No description available."
            )

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
    def wraps_invokable(self) -> bool:
        """Return whether this Tool wraps an ``AtomicInvokable`` target."""
        return isinstance(self._function, AtomicInvokable)

    @property
    def function(self) -> AtomicInvokable | Callable[..., Any]:
        """Underlying plain callable or ``AtomicInvokable`` execution target."""
        return self._function

    @function.setter
    def function(self, func: AtomicInvokable | Callable[..., Any]) -> None:
        """Update the execution target and refresh schema/import metadata."""
        if not callable(func):
            raise ToolDefinitionError(f"Tool function must be callable, got {type(func)!r}")

        module, qualname = self._get_mod_qual(func)

        if isinstance(func, AtomicInvokable):
            parameters = list(func.parameters)
            return_type = func.return_type
        else:
            parameters, return_type = extract_io(func)

        if not isinstance(return_type, str):
            raise TypeError(
                f"{type(self).__name__}.return_type must be str, got {type(return_type)!r}"
            )

        self._function = func
        self._module = module
        self._qualname = qualname
        self._parameters = parameters
        self._return_type = return_type

    @property
    def module(self) -> Optional[str]:
        """Best-effort module identity for the wrapped callable or invokable target."""
        return self._module

    @property
    def qualname(self) -> Optional[str]:
        """Best-effort qualified-name identity for the wrapped callable or invokable target."""
        return self._qualname

    @property
    def full_name(self) -> str:
        """Fully-qualified tool name of the form ``Type.namespace.name``."""
        return f"{type(self).__name__}.{self._namespace}.{self._name}"

    # ------------------------------------------------------------------ #
    # Signature Building (Template Method)
    # ------------------------------------------------------------------ #
    def _build_tool_signature(self) -> tuple[list[ParamSpec], str]:
        """Build this Tool's parameter and return schema.

        Plain callable-backed tools derive schema through ``extract_io(...)``.
        Invokable-backed tools reuse the wrapped invokable's declared
        ``parameters`` and ``return_type`` instead of introspecting ``__call__``,
        ``invoke``, or ``async_call``.

        Subclasses can override this template hook to build signatures from
        alternative metadata sources, such as MCP schemas or remote agent
        metadata.

        Returns
        -------
        tuple[list[ParamSpec], str]
            Ordered parameter specs and return type string.
        """
        # If the tool wraps an AtomicInvokable, use the invokable-declared
        # schema rather than introspecting callables. Do not call
        # `extract_io()` on invokable instances.
        if self.wraps_invokable:
            parameters = list(self._function.parameters)
            return_type = self._function.return_type
        else:
            parameters, return_type = extract_io(self._function)

        return parameters, return_type

    # ------------------------------------------------------------------ #
    # Tool Helpers
    # ------------------------------------------------------------------ #
    def _get_mod_qual(
        self,
        function: AtomicInvokable | Callable[..., Any],
    ) -> tuple[Optional[str], Optional[str]]:
        """Determine ``(module, qualname)`` for callable- or invokable-backed tools.

        For invokable-backed tools, identity is derived from the bound
        ``invoke`` method because the invokable object itself may not expose a
        useful import path. Subclasses that do not use Python import identity,
        such as MCP-backed tools, should override this method.
        """
        # For invokable objects, derive import identity from the bound
        # `invoke` method for more useful module/qualname metadata. Use the
        # passed `function` argument (not `self._function`) to avoid relying
        # on instance state during construction.
        if isinstance(function, AtomicInvokable):
            module = getattr(function.invoke, "__module__", None)
            qualname = getattr(function.invoke, "__qualname__", None)
        else:
            module = getattr(function, "__module__", None)
            qualname = getattr(function, "__qualname__", None)

        return module, qualname

    def to_arg_kwarg(self, inputs: Mapping[str, Any]) -> tuple[tuple[Any, ...], Dict[str, Any]]:
        """Map dict-first inputs to normal call-style ``(*args, **kwargs)``.

        The base implementation delegates binding to
        ``AtomicInvokable._dict_to_args_kwargs()``. For plain callables, the
        resulting args/kwargs are passed directly to the callable. For
        invokable-backed tools, they are passed to the wrapped invokable's
        ``__call__`` path.

        Subclasses may override this method when their execution transport is
        intentionally dict-first rather than Python-call-style, such as
        ``AdapterTool``, ``MCPProxyTool``, or ``PyA2AtomicTool``.

        Raises
        ------
        TypeError
            If the input mapping cannot be bound to this tool's declared
            parameter contract.
        """
        args, kwargs = self._dict_to_args_kwargs(inputs)
        return args, kwargs

    def execute(self, args: tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        """Synchronously execute the underlying target.

        Plain callable-backed tools call the stored callable directly.
        Invokable-backed tools call the wrapped invokable's ``__call__`` path,
        which converts call-style args/kwargs back into the invokable's
        dict-first ``invoke(...)`` contract.

        Subclasses may override this to change *how* a tool is executed, such
        as by making a remote MCP call or invoking a transport client.

        If the target returns an awaitable, the sync path runs it to completion
        using a private event-loop bridge.
        """
        try:
            result = self._function(*args, **kwargs)

            if inspect.isawaitable(result):
                result = run_coro_sync(result)

        except ToolInvocationError:
            raise
        except Exception as e:  # pragma: no cover - thin wrapper
            raise ToolInvocationError(f"{self.full_name}: invocation failed: {e}") from e

        return result

    async def async_execute(
        self,
        args: tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Any:
        """Asynchronously execute the underlying target.

        Execution dispatch:

        - Invokable-backed tools call the wrapped invokable's
          ``async_call(*args, **kwargs)`` path.
        - Native async callables are awaited directly.
        - Sync callables are offloaded to a worker thread.
        - Awaitable results are awaited before returning.
        """
        try:
            if self.wraps_invokable:
                result = await self._function.async_call(*args, **kwargs)
            elif inspect.iscoroutinefunction(self._function):
                result = await self._function(*args, **kwargs)
            else:
                result = await asyncio.to_thread(self._function, *args, **kwargs)

            if inspect.isawaitable(result):
                result = await result

        except ToolInvocationError:
            raise
        except Exception as e:  # pragma: no cover - thin wrapper
            raise ToolInvocationError(f"{self.full_name}: invocation failed: {e}") from e

        return result

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    async def async_invoke(self, inputs: Mapping[str, Any]) -> Any:
        """Asynchronously invoke the tool using dict-first inputs.

        Mirrors the sync ``invoke(...)`` flow:

        1. Filter inputs.
        2. Bind dict-first inputs to call-style args/kwargs.
        3. Dispatch through ``async_execute(...)``.

        ``async_execute(...)`` owns the distinction between invokable-backed
        tools, native async callables, and sync callables.
        """
        logger.info(f"[Async {self.full_name} started]")
        inputs = self.filter_inputs(inputs)
        args, kwargs = self.to_arg_kwarg(inputs)
        result = await self.async_execute(args, kwargs)
        logger.info(f"[Async {self.full_name} finished]")
        return result

    def invoke(self, inputs: Mapping[str, Any]) -> Any:
        """Synchronously invoke the tool using dict-first inputs.

        This is the main public execution entrypoint. Subclasses should not
        override this method; instead, they should customize ``to_arg_kwarg()``
        and ``execute()`` when they need different binding or transport
        semantics.
        """
        with self._invoke_lock:
            logger.info(f"[{self.full_name} started]")
            inputs = self.filter_inputs(inputs)
            args, kwargs = self.to_arg_kwarg(inputs)
            result = self.execute(args, kwargs)
            logger.info(f"[{self.full_name} finished]")
            return result

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        """Serialize this tool's metadata and argument schema.

        The base metadata includes identity, schema, namespace, import-path
        hints, and whether this Tool wraps an ``AtomicInvokable``.

        If ``wraps_invokable`` is true, the wrapped invokable's own
        ``to_dict()`` output is included under ``"invokable_function"`` for
        transparency. This method does not guarantee that the Tool is
        reconstructable; reconstruction is left to future factory logic.
        """
        d = super().to_dict()
        d.update({
            "wraps_invokable": self.wraps_invokable,
            "namespace": self.namespace,
            "module": self.module,
            "qualname": self.qualname,
        })

        # If wrapping an invokable, include the invokable's own serialization
        # under the agreed key name.
        if self.wraps_invokable:
            d["invokable_function"] = self._function.to_dict()

        return d
