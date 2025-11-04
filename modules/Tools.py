# Tools.py
from __future__ import annotations

import asyncio
import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, get_type_hints, get_origin, get_args, TypedDict
from collections import OrderedDict

# External integrations (MCP) + local modules
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

# from modules.Agents import Agent
from modules.Plugins import *  # Provides Plugin-shaped dicts (see TypedDict below)


# ───────────────────────────────────────────────────────────────────────────────
# Public API
# ───────────────────────────────────────────────────────────────────────────────
__all__ = ["Tool", "ToolFactory"]

# ───────────────────────────────────────────────────────────────────────────────
# Tool
# ───────────────────────────────────────────────────────────────────────────────
from typing import Any, Mapping, Callable, OrderedDict as OrderedDictType, Dict, List, Optional, Tuple
from collections import OrderedDict
import inspect


class ToolError(Exception):
    """Base exception for Tool-related errors."""


class ToolDefinitionError(ToolError):
    """Raised when a callable is incompatible at Tool construction time."""


class ToolInvocationError(ToolError):
    """Raised when inputs are invalid for invocation or binding fails."""


# Sentinel to mark "no default" for parameters
NO_DEFAULT = object()


class Tool:
    """
    Tool
    ----
    Stateless wrapper around a Python callable with a dict-first invocation API.
    At construction, Tool builds an **arguments map** describing each named
    parameter (annotation, kind, default, and declaration position). During
    `invoke()`, Tool splits the provided mapping into a positional-only list
    (`args`) and a keyword dict (`kwargs`) according to that map, then calls
    `func(*args, **kwargs)`.

    Primary API
    -----------
    __init__(func: Callable, name: str, description: str = "",
             tool_type: str = "python", source: str = "local")

    invoke(inputs: Mapping[str, Any]) -> Any
        Deterministic binding by parameter **names**:
          • Positional-only params → collected into *args by **declaration order**,
            enforcing a contiguous prefix (no gaps).
          • Positional-or-keyword & keyword-only params → placed in **kwargs** by name.
          • If the function declares *args, extra positionals must be provided via
            '_args': list|tuple.
          • If the function declares **kwargs, extra keyword pairs must be provided
            via '_kwargs': Mapping.

    Metadata (tags)
    ---------------
    • type: str      # classification tag (e.g., "python", "agent", "workflow")
    • source: str    # provenance tag (e.g., "local", "remote:mcp", "plugin:my_pkg")

    Arguments Map (constructor-built, cached)
    -----------------------------------------
    arguments_map: OrderedDict[str, ArgSpec]  (read-only property)
      Each entry captures:
        - index: int                         # declaration index (0-based)
        - kind: inspect._ParameterKind       # EXACT enum (not a string)
        - ann:  type | Any                   # from annotations; Any if absent
        - has_default: bool
        - default: value | NO_DEFAULT

    Call Plan (cached)
    ------------------
    • posonly_order: List[str]               # positional-only names in declaration order
    • p_or_kw_names: List[str]               # positional-or-keyword names
    • kw_only_names: List[str]               # keyword-only names
    • required_names: set[str]               # required among p_or_kw + kw_only (no defaults)
    • has_varargs: bool                      # whether *args exists
    • has_varkw: bool                        # whether **kwargs exists
    • varargs_name: Optional[str]
    • varkw_name: Optional[str]

    Strictness
    ----------
    - Unknown top-level keys are **errors** (even if the function has **kwargs);
      callers must place extras explicitly under '_kwargs'.
    - Duplicate provision (top-level and '_kwargs' for the same name) is an error.
    - Reserved keys must match required container types:
        '_args' -> list|tuple, '_kwargs' -> Mapping.

    Notes
    -----
    - Tool is **stateless**; it exposes no memory APIs.
    - The `description` getter intentionally returns the user-provided text only.
      Prompt-time schema/context composition is handled by Tool-Agents.
    """

    # -------------------------
    # Construction & metadata
    # -------------------------
    def __init__(
        self,
        func: Callable,
        name: str,
        description: str = "",
        type: str = "function",
        source: str = "local",
    ) -> None:
        self._func: Callable = func
        self._name: str = name
        self._description: str = description
        self._type: str = type
        self._source: str = source

        # Build signature and call plan once (deterministic & fast at runtime)
        try:
            self._sig: inspect.Signature = inspect.signature(inspect.unwrap(func))
        except Exception as e:
            raise ToolDefinitionError(f"{name}: cannot introspect callable signature: {e}") from e

        (
            self._arguments_map,
            self.posonly_order,
            self.p_or_kw_names,
            self.kw_only_names,
            self.required_names,
            self.has_varargs,
            self.varargs_name,
            self.has_varkw,
            self.varkw_name,
        ) = self._build_arguments_map_and_plan(self._sig)

    # Read-only tags and doc
    @property
    def name(self) -> str:
        """Tool name (read-only)."""
        return self._name

    @property
    def description(self) -> str:
        """Natural-language description (read-only)."""
        return self._description

    @property
    def type(self) -> str:
        """Classification tag (read-only). E.g., 'python', 'agent', 'workflow'."""
        return self._type

    @property
    def source(self) -> str:
        """Provenance tag (read-only). E.g., 'local', 'remote:mcp', 'plugin:my_pkg'."""
        return self._source

    @property
    def full_name(self) -> str:
        """Fully-qualified tool key used by planners/orchestrators."""
        return f"{self._type}.{self._source}.{self._name}"

    @property
    def func(self) -> Callable:
        """Underlying callable (read-only). Prefer `invoke()` for validation."""
        return self._func

    @property
    def arguments_map(self):
        """Read-only accessor for the constructor-built arguments map."""
        return self._arguments_map

    # -------------------------
    # Core invocation
    # -------------------------
    def invoke(self, inputs: Mapping[str, Any]) -> Any:
        """
        Invoke the wrapped callable by splitting `inputs` into:
          - args:  positional-only values (contiguous prefix by declaration order),
                   plus optional extras from '_args' if *args exists.
          - kwargs: all positional-or-keyword and keyword-only parameters by name,
                    plus optional extras from '_kwargs' if **kwargs exists.

        Strict validation:
          - Unknown top-level keys are errors (use '_kwargs' explicitly if **kwargs exists).
          - Reserved keys '_args' and '_kwargs' require correct container types.
          - Duplicate provision across top-level and '_kwargs' is an error.

        Returns:
            Any: result of `self._func(*args, **kwargs)`

        Raises:
            ToolInvocationError: on invalid inputs or binding mistakes.
        """
        if not isinstance(inputs, Mapping):
            raise ToolInvocationError(f"{self._name}: inputs must be a mapping")

        ARGS_KEY = "_args"
        KWARGS_KEY = "_kwargs"

        # Validate reserved keys early
        if ARGS_KEY in inputs:
            if not self.has_varargs:
                raise ToolInvocationError(f"{self._name}: '{ARGS_KEY}' provided but function declares no *args")
            extra_args = inputs[ARGS_KEY]
            if not isinstance(extra_args, (list, tuple)):
                raise ToolInvocationError(f"{self._name}: '{ARGS_KEY}' must be list or tuple")
        else:
            extra_args = ()

        if KWARGS_KEY in inputs:
            if not self.has_varkw:
                raise ToolInvocationError(f"{self._name}: '{KWARGS_KEY}' provided but function declares no **kwargs")
            extra_kwargs = inputs[KWARGS_KEY]
            if not isinstance(extra_kwargs, Mapping):
                raise ToolInvocationError(f"{self._name}: '{KWARGS_KEY}' must be a mapping")
        else:
            extra_kwargs = {}

        provided_names = set(inputs.keys()) - {ARGS_KEY, KWARGS_KEY}
        known_names = set(self._arguments_map.keys())
        unknown = sorted(provided_names - known_names)

        # Strict: unknown top-level keys are not allowed even if **kwargs exists
        if unknown:
            if not self.has_varkw:
                raise ToolInvocationError(f"{self._name}: unexpected keys: {unknown}")
            raise ToolInvocationError(
                f"{self._name}: unexpected keys {unknown}; place extras under '{KWARGS_KEY}' because function accepts **kwargs"
            )

        # Required named parameters must be present (pos-only handled separately)
        missing = sorted(self.required_names - provided_names)
        if missing:
            raise ToolInvocationError(f"{self._name}: missing required keys: {missing}")

        # -------------------------
        # Build args (positional-only), gap-safe
        # -------------------------
        args: List[Any] = []
        seen_gap = False
        for pname in self.posonly_order:
            present = pname in inputs
            if not present:
                seen_gap = True
                continue
            if seen_gap:
                # e.g., 'a' missing but 'b' present (both positional-only) — illegal
                raise ToolInvocationError(
                    f"{self._name}: positional-only gap: '{pname}' supplied after an earlier positional-only was missing"
                )
            args.append(inputs[pname])

        # Prevent conflicts when *also* supplying extra positional args via '_args'
        if extra_args:
            consume = min(len(extra_args), len(self.p_or_kw_names))
            if consume:
                conflicting = [self.p_or_kw_names[i] for i in range(consume) if self.p_or_kw_names[i] in provided_names]
                if conflicting:
                    raise ToolInvocationError(
                        f"{self._name}: '_args' will bind positionally to {conflicting} "
                        "but those are also provided by name; use either '_args' or named keys, not both."
                    )

        # -------------------------
        # Build kwargs (pos_or_kw + kw_only)
        # -------------------------
        kwargs: Dict[str, Any] = {}
        for pname in self.p_or_kw_names + self.kw_only_names:
            if pname in inputs:
                kwargs[pname] = inputs[pname]
            # else: omit; Python applies default if available

        # Merge explicit var-keyword extras if any
        if extra_kwargs:
            dupes = sorted(set(kwargs.keys()) & set(extra_kwargs.keys()))
            if dupes:
                raise ToolInvocationError(
                    f"{self._name}: duplicate keys supplied both as named inputs and in '{KWARGS_KEY}': {dupes}"
                )
            kwargs.update(extra_kwargs)  # type: ignore[arg-type]

        # Finally call the function
        try:
            return self._func(*args, **kwargs)
        except TypeError as e:
            # Surface actionable context for signature mismatches
            raise ToolInvocationError(f"{self._name}: invocation failed: {e}") from e

    # -------------------------
    # Introspection & serialization
    # -------------------------
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize Tool metadata and signature plan for registries and UIs.
        Notes:
          - 'default' uses sentinel string '<NO_DEFAULT>' when absent.
          - 'ann' stores the annotation's repr for readability.
          - 'kind' is emitted as the enum name (e.g., 'POSITIONAL_ONLY').
        """
        def ann_repr(t: Any) -> str:
            if t is inspect._empty:
                return "Any"
            try:
                return getattr(t, "__name__", repr(t))
            except Exception:
                return repr(t)

        argmap_serialized = OrderedDict()
        for name, spec in self._arguments_map.items():
            kind_enum: inspect._ParameterKind = spec["kind"]
            argmap_serialized[name] = {
                "index": spec["index"],
                "kind": kind_enum.name,
                "ann": ann_repr(spec["ann"]),
                "has_default": spec["has_default"],
                "default": "<NO_DEFAULT>" if spec["default"] is NO_DEFAULT else repr(spec["default"]),
            }

        return {
            "name": self._name,
            "description": self._description,
            "type": self._type,
            "source": self._source,
            "signature": str(self._sig),
            "arguments_map": argmap_serialized,
            "posonly_order": list(self.posonly_order),
            "p_or_kw_names": list(self.p_or_kw_names),
            "kw_only_names": list(self.kw_only_names),
            "required_names": sorted(self.required_names),
            "has_varargs": self.has_varargs,
            "varargs_name": self.varargs_name,
            "has_varkw": self.has_varkw,
            "varkw_name": self.varkw_name,
        }

    # -------------------------
    # Internal: build arg map + call plan (ENUM kinds)
    # -------------------------
    def _build_arguments_map_and_plan(
        self, sig: inspect.Signature
    ) -> Tuple[
        OrderedDict[str, Dict[str, Any]],      # arguments_map
        List[str],                             # posonly_order
        List[str],                             # p_or_kw_names
        List[str],                             # kw_only_names
        set,                                   # required_names
        bool, Optional[str],                   # has_varargs, varargs_name
        bool, Optional[str],                   # has_varkw,   varkw_name
    ]:
        """
        Partition parameters by kind (ENUM), capture annotations/defaults,
        and precompute call-time ordering and requirements.
        """
        arguments_map: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        posonly_order: List[str] = []
        p_or_kw_names: List[str] = []
        kw_only_names: List[str] = []
        required_names: set = set()

        has_varargs = False
        varargs_name: Optional[str] = None
        has_varkw = False
        varkw_name: Optional[str] = None

        index = 0
        for pname, p in sig.parameters.items():
            if p.kind is p.VAR_POSITIONAL:
                has_varargs = True
                varargs_name = pname
                continue
            if p.kind is p.VAR_KEYWORD:
                has_varkw = True
                varkw_name = pname
                continue

            ann = p.annotation if p.annotation is not inspect._empty else Any
            default = p.default if p.default is not inspect._empty else NO_DEFAULT
            kind_enum = p.kind  # store the EXACT enum

            if kind_enum is p.POSITIONAL_ONLY:
                posonly_order.append(pname)
            elif kind_enum is p.POSITIONAL_OR_KEYWORD:
                p_or_kw_names.append(pname)
            elif kind_enum is p.KEYWORD_ONLY:
                kw_only_names.append(pname)
            else:
                # Should be unreachable (varargs/varkw handled above)
                raise ToolDefinitionError(f"Unexpected parameter kind for {pname}: {kind_enum!r}")

            arguments_map[pname] = {
                "index": index,
                "kind": kind_enum,   # enum stored here
                "ann": ann,
                "has_default": default is not NO_DEFAULT,
                "default": default,
            }
            index += 1

        # Required named = those without defaults among pos_or_kw + kw_only
        for pname in p_or_kw_names + kw_only_names:
            if arguments_map[pname]["default"] is NO_DEFAULT:
                required_names.add(pname)

        return (
            arguments_map,
            posonly_order,
            p_or_kw_names,
            kw_only_names,
            required_names,
            has_varargs,
            varargs_name,
            has_varkw,
            varkw_name,
        )
