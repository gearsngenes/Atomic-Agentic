from __future__ import annotations

import asyncio
import inspect
import logging
from datetime import datetime, timezone
from typing import (
    Any,
    Callable,
    Dict,
    Mapping,
    Optional,
)

from ..exceptions import ToolDefinitionError, ToolInvocationError
from ..core.Invokable import AtomicInvokable
from ..core.core_api import extract_io
from ..models.parameters import ParamSpec
from ..utils.core import run_coro_sync
from ..models.results.tools import ToolResult


logger = logging.getLogger(__name__)


# ───────────────────────────────────────────────────────────────────────────────
# Tool Invokable
# ───────────────────────────────────────────────────────────────────────────────
class Tool(AtomicInvokable):
    """Concrete base Tool primitive.

    ``Tool`` provides a dict-first invocation interface around a plain
    Python callable *or* an ``AtomicInvokable``. It implements the template
    method::

        invoke(inputs) -> to_arg_kwarg(inputs) -> execute(args, kwargs) -> ToolResult

    Schema is derived through ``extract_io(...)``, which recognizes
    ``AtomicInvokable`` inputs directly (reusing their declared
    ``parameters``/``return_type``) and falls back to signature introspection
    otherwise. Name/description inference reads ``__name__``/``__doc__`` off
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

    ``execute(...)`` and ``async_execute(...)`` return the raw execution payload.
    Public ``invoke(...)`` and ``async_invoke(...)`` wrap that payload in a
    ``ToolResult`` whose ``.result`` field contains the caller-facing value.

    Subclasses customize result-envelope selection by overriding ``make_result(...)``.
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
    ) -> None:
        """Initialize a Tool from a plain callable or an AtomicInvokable.

        Parameters
        ----------
        function:
            Plain callable or ``AtomicInvokable`` to expose as a Tool,
            introspected with ``extract_io(...)``.

        name:
            Optional Tool name override. If omitted, defaults to
            ``function.__name__``.

        namespace:
            Optional Tool namespace. Defaults to ``"default"``.

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

        # namespace is stored by AtomicInvokable; no local storage needed
        # Delegate name/description validation and schema setup to parent
        super().__init__(
            name=inferred_name,
            description=inferred_description,
            namespace=namespace or "default",
            parameters=parameters,
            return_type=return_type,
        )

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
    def _build_tool_signature(self) -> tuple[list[ParamSpec], str]:
        """Build this Tool's parameter and return schema.

        Schema is derived through ``extract_io(...)`` on the underlying
        callable.

        Subclasses can override this template hook to build signatures from
        alternative metadata sources, such as MCP schemas or remote agent
        metadata.

        Returns
        -------
        tuple[list[ParamSpec], str]
            Ordered parameter specs and return type string.
        """
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

    def execute(self, args: tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        """Synchronously execute the underlying target.

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

        return result

    async def async_execute(
        self,
        args: tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Any:
        """Asynchronously execute the underlying target.

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

        return result

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    async def async_invoke(self, inputs: Mapping[str, Any]) -> ToolResult:
        """Asynchronously invoke the tool using dict-first inputs.

        Mirrors the sync ``invoke(...)`` flow:

        1. Filter inputs.
        2. Convert filtered inputs into the execution shape through
           ``to_arg_kwarg(...)``.
        3. Dispatch through ``async_execute(...)``.
        4. Wrap the raw execution payload in ``ToolResult``.

        ``async_execute(...)`` owns the distinction between AtomicInvokable
        targets, native async callables, and sync callables.
        """
        started_at = datetime.now(timezone.utc)

        logger.info(f"[Async {self.full_name} started]")
        inputs = self.filter_inputs(inputs)
        args, kwargs = self.to_arg_kwarg(inputs)
        result = await self.async_execute(args, kwargs)
        ended_at = datetime.now(timezone.utc)

        logger.info(f"[Async {self.full_name} finished]")
        return self.make_result(
            result=result,
            started_at=started_at,
            ended_at=ended_at,
        )

    def invoke(self, inputs: Mapping[str, Any]) -> ToolResult:
        """Synchronously invoke the tool using dict-first inputs.

        This is the main public execution entrypoint. Subclasses should not
        override this method; instead, they should customize ``to_arg_kwarg()``
        and ``execute()`` when they need different binding or transport
        semantics.

        The returned ``ToolResult.result`` contains the raw execution payload
        that this method previously returned directly.
        """
        with self._invoke_lock:
            started_at = datetime.now(timezone.utc)

            logger.info(f"[{self.full_name} started]")
            inputs = self.filter_inputs(inputs)
            args, kwargs = self.to_arg_kwarg(inputs)
            result = self.execute(args, kwargs)
            ended_at = datetime.now(timezone.utc)

            logger.info(f"[{self.full_name} finished]")
            return self.make_result(
                result=result,
                started_at=started_at,
                ended_at=ended_at,
            )

    def make_result(
        self,
        result: Any,
        started_at: datetime,
        ended_at: datetime,
        **result_kwargs: Any,
    ) -> ToolResult:
        """Construct a ToolResult envelope for this tool's invocation."""
        return self._make_result(
            result=result,
            started_at=started_at,
            ended_at=ended_at,
            result_cls=ToolResult,
            **result_kwargs,
        )

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        """Serialize this tool's metadata and argument schema.

        The base metadata includes identity, schema, namespace, and
        import-path hints. This method does not guarantee that the Tool is
        reconstructable; reconstruction is left to future factory logic.
        """
        d = super().to_dict()
        # "namespace" is emitted by super().to_dict(); not repeated here
        d.update({
            "module": self.module,
            "qualname": self.qualname,
        })
        return d
