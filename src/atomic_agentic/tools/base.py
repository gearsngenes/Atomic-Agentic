from __future__ import annotations
import inspect
from collections import OrderedDict
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    get_args,
    get_origin,
)

from ..core.Exceptions import *
from ..core.Invokable import AtomicInvokable, ArgumentMap


# ───────────────────────────────────────────────────────────────────────────────
# Tool primitive
# ───────────────────────────────────────────────────────────────────────────────
class Tool(AtomicInvokable):
    """Concrete base Tool primitive.

    This class provides a *dict-first* invocation interface around an underlying
    callable. It implements the template method::

        invoke(inputs) -> to_arg_kwarg(inputs) -> execute(args, kwargs)

    Subclasses such as MCPProxyTool and AgentTool are expected to override only
    the helper hooks used at construction time (``_get_mod_qual``,
    ``_build_args_returns`` (preferred) or ``_build_io_schemas`` for compatibility,
    ``_compute_is_persistible``) and, where necessary,
    the :meth:`to_arg_kwarg` / :meth:`execute` methods. The public
    :meth:`invoke` method must not be overridden.

    The argument schema is exposed via :attr:`arguments_map` as an
    OrderedDict whose values are JSON-serialisable metadata dictionaries with
    the following keys:

    - ``index``  : ``int`` – the parameter position.
    - ``kind``   : ``str`` – one of
      ``{"POSITIONAL_ONLY", "POSITIONAL_OR_KEYWORD", "KEYWORD_ONLY", "VAR_POSITIONAL", "VAR_KEYWORD"}``.
    - ``type``   : ``str`` – a human-readable type name, e.g. ``"int"``.
    - ``default``: optional – present only if the parameter has a default.

    ``return_type`` is always stored as a string as well.
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

        # Delegate name/description validation and arguments/return type setup
        super().__init__(name=inferred_name, description=inferred_description)

        # Compute persistibility flag now that module/qualname are set
        self._is_persistible_internal: bool = self._compute_is_persistible()


    # ------------------------------------------------------------------ #
    # Properties
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
        args, ret = self.build_args_returns()
        self._arguments_map, self._return_type = args, ret
        self._is_persistible_internal = self._compute_is_persistible()

    @property
    def module(self) -> Optional[str]:
        return self._module

    @property
    def qualname(self) -> Optional[str]:
        return self._qualname

    @property
    def is_persistible(self) -> bool:
        """Whether this Tool can be reconstructed from its serialized form."""
        return self._is_persistible_internal

    @property
    def full_name(self) -> str:
        """Fully-qualified tool name of the form ``Type.namespace.name``."""
        return f"{type(self).__name__}.{self._namespace}.{self._name}"

    @property
    def signature(self) -> str:
        """Human-readable signature derived from ``arguments_map`` and
        ``return_type``."""
        params: List[str] = []
        for name, meta in self._arguments_map.items():
            kind = meta.get("kind", "")
            type_name = meta.get("type", "Any")
            default_marker = ""
            if "default" in meta:
                default = str(meta['default'])
                default_marker = f" = {default}"
            if kind == "VAR_POSITIONAL":
                param_str = f"*{name}: {type_name}{default_marker}"
            elif kind == "VAR_KEYWORD":
                param_str = f"**{name}: {type_name}{default_marker}"
            else:
                param_str = f"{name}: {type_name}{default_marker}"
            params.append(param_str)
        params_str = ", ".join(params)
        return f"{self.full_name}({params_str}) -> {self._return_type}"

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def invoke(self, inputs: Mapping[str, Any]) -> Any:
        """Invoke the tool using a dict-like mapping of inputs.

        This is the *only* public entrypoint for execution. Subclasses must not
        override this method; instead they can customise :meth:`to_arg_kwarg`
        and :meth:`execute`.
        """
        if not isinstance(inputs, Mapping):
            raise ToolInvocationError(f"{self._name}: inputs must be a mapping")
        args, kwargs = self.to_arg_kwarg(inputs)
        result = self.execute(args, kwargs)
        return result

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _get_mod_qual(self, function: Callable[..., Any]) -> tuple[Optional[str], Optional[str]]:
        """Determine ``(module, qualname)`` for callable-based tools.

        Subclasses that do not use Python import identity (e.g. MCPProxyTool)
        should override this to return ``(None, None)``.
        """
        module = getattr(function, "__module__", None)
        qualname = getattr(function, "__qualname__", None)
        return module, qualname

    def _format_annotation(self, ann: Any) -> str:
        """Convert a type annotation into a readable string.

        Behaviour:
        - If missing/empty → 'Any'.
        - If already a string → returned as-is.
        - If a parameterized / generic type (e.g. List[Dict[str, int]] or dict[str, int]):
          builds the full nested structure string.
        - If a plain class → its name (e.g. 'int', 'MyModel').
        - Otherwise → best-effort str(ann).
        """

        # Missing / unknown annotation
        if ann is inspect._empty or ann is None:
            return "Any"

        # Forward reference or explicit string annotation
        if isinstance(ann, str):
            return ann

        # typing / generic / PEP 585 parameterized types
        origin = get_origin(ann)
        if origin is not None:
            # Recursively format origin and args
            origin_str = self._format_annotation(origin)
            args = get_args(ann)
            if not args:
                return origin_str
            args_str = ", ".join(self._format_annotation(a) for a in args)
            return f"{origin_str}[{args_str}]"

        # Plain classes / types
        module = getattr(ann, "__module__", None)
        name = getattr(ann, "__name__", None)
        if module == "builtins" and name:
            # int, str, dict, list, etc.
            return name
        if name:
            # Custom or library class
            return name

        # Fallback: best-effort string representation
        return str(ann)

    def build_args_returns(self) -> tuple[ArgumentMap, str]:
        """Construct ``arguments_map`` and ``return_type`` from the wrapped
        callable's signature.

        This is the new canonical hook (replaces ``_build_io_schemas``).
        """
        sig = inspect.signature(self._function)
        arg_map: ArgumentMap = OrderedDict()

        for index, (name, param) in enumerate(sig.parameters.items()):
            kind_name = param.kind.name  # e.g. "POSITIONAL_ONLY"
            ann = param.annotation
            default = param.default

            # Decide the source of the type information:
            # 1) annotation if present,
            # 2) otherwise the default's type if present,
            # 3) otherwise "Any".
            if ann is not inspect._empty:
                raw_type = ann
            elif default is not inspect._empty:
                raw_type = type(default)
            else:
                raw_type = inspect._empty

            type_str = self._format_annotation(raw_type)

            meta: Dict[str, Any] = {
                "index": index,
                "kind": kind_name,
                "type": type_str,
            }
            if default is not inspect._empty:
                meta["default"] = default

            arg_map[name] = meta

        # Return type: annotation if present, else 'Any'
        ret_ann = sig.return_annotation
        return_type = self._format_annotation(ret_ann)

        return arg_map, return_type

    def _compute_is_persistible(self) -> bool:
        """Default persistibility check for callable-based tools.

        A Tool is considered persistible if its function has both ``__module__``
        and ``__qualname__`` and does not appear to be a local/helper function.
        Subclasses can override this with their own criteria.
        """
        if not self._module or not self._qualname:
            return False
        # Heuristic: local/helper functions usually contain '<locals>' in qualname.
        if "<locals>" in self._qualname:
            return False
        return True

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

        This docstring documents behaviour only; the implementation is intentionally
        left as-is to preserve existing semantics and tests.
        """
        data: Dict[str, Any] = dict(inputs)

        # Compute parameter ordering and kinds
        param_items = sorted(self._arguments_map.items(), key=lambda kv: kv[1].get("index", 0))
        param_names = {name for name, _ in param_items}

        varpos_name: Optional[str] = None
        varkw_name: Optional[str] = None
        for name, meta in param_items:
            kind = meta.get("kind")
            if kind == "VAR_POSITIONAL" and varpos_name is None:
                varpos_name = name
            elif kind == "VAR_KEYWORD" and varkw_name is None:
                varkw_name = name

        # Unknown key handling
        unknown_keys = set(data.keys()) - param_names
        if unknown_keys and varkw_name is None:
            raise ToolInvocationError(f"{self._name}: unknown parameters: {sorted(unknown_keys)}")

        # Required parameter check (exclude VAR_*)
        required_names = [
            name
            for name, meta in param_items
            if meta.get("kind") not in {"VAR_POSITIONAL", "VAR_KEYWORD"}
            and "default" not in meta
        ]
        missing = [name for name in required_names if name not in data]
        if missing:
            raise ToolInvocationError(f"{self._name}: missing required parameters: {missing}")

        args: List[Any] = []
        kwargs: Dict[str, Any] = {}

        # First, handle named parameters in order
        for name, meta in param_items:
            kind = meta.get("kind")
            has_default = "default" in meta
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
                val = meta["default"]
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
            "is_persistible": self.is_persistible
        })
        return d
