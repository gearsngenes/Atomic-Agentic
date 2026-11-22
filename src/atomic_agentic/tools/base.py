from __future__ import annotations

import asyncio
import inspect
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Mapping,
    Optional,
)

from ..exceptions import ToolDefinitionError, ToolInvocationError
from ..invokables.atomic_invokable import AtomicInvokable
from ..models.parameters import ParamSpec
from ..utils.core import run_coro_sync
from ..utils.parameters import extract_io
from ..models.results.tools import ToolResult


# ───────────────────────────────────────────────────────────────────────────────
# Tool Invokable
# ───────────────────────────────────────────────────────────────────────────────
class Tool(AtomicInvokable):
    """Concrete base Tool primitive.

    ``Tool`` provides a dict-first invocation interface around a plain
    Python callable *or* an ``AtomicInvokable``. It implements the template
    method::

        ``invoke(inputs)`` (inherited template) -> ``_invoke_impl`` -> ``to_arg_kwarg`` -> ``execute`` -> ``ToolResult``

    Schema comes from the wrapped target: an ``AtomicInvokable`` contributes
    its declared ``parameters``/``return_type``, and any other callable is
    introspected through ``extract_io(...)``. Name/description inference reads ``__name__``/``__doc__`` off
    the wrapped target — for an ``AtomicInvokable`` these are kept in sync
    with ``.name``/``.description`` by the base class itself, so no special
    handling is needed here. ``Tool`` stays otherwise blind to what kind of
    callable it wraps, with one narrow, deliberate exception: `async_execute`
    dispatches to `async_call` for an ``AtomicInvokable`` target, since a
    plain `iscoroutinefunction` check would miss it and silently thread-offload
    the *sync* ``__call__`` instead of awaiting the real async path.

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
    Filtered dict-first inputs are bound into ordinary Python call-style
    ``(*args, **kwargs)`` before calling the stored callable.

    ``execute(...)`` and ``async_execute(...)`` return the result payload dict
    (``{"result": <value>}``; proxies add their fixed transport fields).
    Public ``invoke(...)`` and ``async_invoke(...)`` wrap that payload in a
    ``ToolResult`` whose ``.result`` field contains the caller-facing value.

    Subclasses select their result class with ``result_cls``; proxy subclasses
    also put their fixed transport fields in the dict their ``execute`` returns.
    """

    result_cls: ClassVar[type[ToolResult]] = ToolResult

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        function: AtomicInvokable | Callable[..., Any],
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> None:
        """Initialize a Tool from a plain callable or an AtomicInvokable.

        Parameters
        ----------
        function:
            Plain callable or ``AtomicInvokable`` to expose as a Tool.
            A plain callable is introspected with ``extract_io(...)``.

        name:
            Optional Tool name override. If omitted, defaults to
            ``function.__name__``.

        description:
            Optional Tool description override. If omitted, defaults to the
            callable's docstring or a fallback description.
        """
        if not callable(function):
            raise ToolDefinitionError(f"Tool function must be callable, got {type(function)!r}")

        # Underlying callable or AtomicInvokable execution target and identity.
        self._function: AtomicInvokable | Callable[..., Any] = function
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
        )

    @classmethod
    def of(
        cls,
        fn: Callable[..., Any] | None = None,
        /,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        """Decorator form of the constructor.

        ``@Tool.of`` and ``@Tool.of()`` / ``@Tool.of(name=..., description=...)``
        bind the decorated name to a ``Tool`` built exactly as
        ``Tool(fn, name, description)`` (same name and description inference).
        ``name`` and ``description`` are keyword-only. A method loses ``self``
        binding (``Tool`` is not a descriptor). Intended for ``Tool`` itself;
        subclasses with a different constructor (the proxy tools) should not
        be used with it.

        1. If ``fn`` is None, return a decorator that builds the Tool from the
           function it receives.
        2. If ``fn`` is not callable (e.g. ``@Tool.of("name")``), raise
           ``TypeError``: only keywords configure the decorator.
        3. Otherwise return ``cls(fn, name, description)``.
        """
        # 1) Keyword form: configure now, build when the function arrives.
        if fn is None:
            def decorator(f: Callable[..., Any]):
                return cls(f, name, description)

            return decorator

        # 2) A positional non-callable is almost certainly @Tool.of("name").
        if not callable(fn):
            raise TypeError(
                "Tool.of: keyword arguments only; use @Tool.of(name=...), not a "
                f"positional value (got {type(fn).__name__})."
            )

        # 3) Bare form.
        return cls(fn, name, description)

    # ------------------------------------------------------------------ #
    # Tool Properties
    # ------------------------------------------------------------------ #
    @property
    def function(self) -> AtomicInvokable | Callable[..., Any]:
        """Underlying plain callable or AtomicInvokable execution target."""
        return self._function

    @property
    def module(self) -> Optional[str]:
        """Best-effort module identity for the wrapped callable."""
        return self._module

    @property
    def qualname(self) -> Optional[str]:
        """Best-effort qualified-name identity for the wrapped callable."""
        return self._qualname

    def _extra_description(self) -> str:
        """Chain into the wrapped invokable's own extra description.

        AtomicInvokable-backed tools surface `self._function`'s own
        `_extra_description()` verbatim. Plain callable-backed tools have no
        wrapped instance to chain to, so this returns `""`.
        """
        if isinstance(self._function, AtomicInvokable):
            return self._function._extra_description()
        return ""

    # ------------------------------------------------------------------ #
    # Signature Building (Template Method)
    # ------------------------------------------------------------------ #
    def _build_tool_signature(self) -> tuple[list[ParamSpec], tuple[str, ...]]:
        """Build this Tool's parameter and return schema.

        An ``AtomicInvokable`` contributes its own declared ``parameters``
        and ``return_type``; any other callable is introspected through
        ``extract_io(...)``.

        Subclasses can override this template hook to build signatures from
        alternative metadata sources, such as MCP schemas or remote agent
        metadata.

        Returns
        -------
        tuple[list[ParamSpec], tuple[str, ...]]
            Ordered parameter specs and the return type tuple.
        """
        if isinstance(self._function, AtomicInvokable):
            return self._function.parameters, self._function.return_type
        return extract_io(self._function)

    # ------------------------------------------------------------------ #
    # Tool Helpers
    # ------------------------------------------------------------------ #
    def _get_mod_qual(
        self,
        function: Callable[..., Any],
    ) -> tuple[Optional[str], Optional[str]]:
        """Determine ``(module, qualname)`` for a callable-backed tool.

        Subclasses that do not use Python import identity, such as
        MCP-backed tools, should override this method.
        """
        module = getattr(function, "__module__", None)
        qualname = getattr(function, "__qualname__", None)
        return module, qualname

    def to_arg_kwarg(self, inputs: Mapping[str, Any]) -> tuple[tuple[Any, ...], Dict[str, Any]]:
        """Map filtered dict-first inputs into execution arguments.

        Binds inputs into normal Python call-style ``(*args, **kwargs)``
        using ``AtomicInvokable._dict_to_args_kwargs()``.

        Subclasses may override this method when their execution transport has
        a different binding shape, such as MCP-backed or A2A-backed proxy tools.

        Raises
        ------
        TypeError
            If the input mapping cannot be bound to this tool's declared
            parameter contract.
        """
        args, kwargs = self._dict_to_args_kwargs(inputs)
        return args, kwargs

    def execute(self, args: tuple[Any, ...], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronously execute the underlying target; returns ``{"result": value}``.

        Calls the stored callable directly with ``(*args, **kwargs)``.

        Subclasses may override this to change *how* a tool is executed, such
        as by making a remote MCP call or invoking a transport client.

        If the target returns an awaitable, the sync path runs it to completion
        using the shared sync-over-async bridge.
        """
        try:
            result = self._function(*args, **kwargs)
            if inspect.isawaitable(result):
                result = run_coro_sync(result)

        except ToolInvocationError:
            raise
        except Exception as e:  # pragma: no cover - thin wrapper
            raise ToolInvocationError(f"{self.full_name}: invocation failed: {e}") from e

        return {"result": result}

    async def async_execute(
        self,
        args: tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Asynchronously execute the underlying target; returns ``{"result": value}``.

        Execution dispatch:

        - AtomicInvokable targets use their own native async_call.
        - Native async callables are awaited directly.
        - Sync callables are offloaded to a worker thread.
        - Awaitable results are awaited before returning.
        """
        try:
            if isinstance(self._function, AtomicInvokable):
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

        return {"result": result}

    # ------------------------------------------------------------------ #
    # Invoke hooks
    # ------------------------------------------------------------------ #
    def _invoke_impl(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """to_arg_kwarg -> execute; ``execute`` returns the result payload."""
        args, kwargs = self.to_arg_kwarg(inputs)
        return self.execute(args, kwargs)

    async def _async_invoke_impl(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Async twin: awaits ``async_execute``."""
        args, kwargs = self.to_arg_kwarg(inputs)
        return await self.async_execute(args, kwargs)

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        """Serialize this tool's metadata and argument schema.

        The base metadata includes identity, schema, and import-path hints.
        This method does not guarantee that the Tool is reconstructable; reconstruction is left to future factory logic.
        """
        d = super().to_dict()
        d.update({
            "module": self.module,
            "qualname": self.qualname,
        })
        return d
