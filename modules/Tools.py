# Tools.py
from __future__ import annotations

import inspect
from typing import Any, Mapping, Callable, OrderedDict as Dict, List, Optional, Tuple
from collections import OrderedDict


# ───────────────────────────────────────────────────────────────────────────────
# Public API
# ───────────────────────────────────────────────────────────────────────────────
__all__ = ["Tool"]

# ───────────────────────────────────────────────────────────────────────────────
# Tool
# ───────────────────────────────────────────────────────────────────────────────
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

    At construction, Tool builds an **arguments map** (names → {index, kind, ann,
    default}) and a call-plan (posonly / p_or_kw / kw_only / required / varargs /
    varkw). During `invoke()`, inputs are split deterministically into `*args`
    (contiguous positional-only prefix) and `**kwargs` (named), honoring strict
    unknown-key rules.

    New (schema-derived) signature string
    -------------------------------------
    • `signature` is now a canonical, **schema-derived** string composed from
      `arguments_map` + `return_type`, not from `inspect.Signature`.
    • Grammar:  `name(p1:Type, p2?:Type, *varargs, **varkw) -> ReturnType`
        - Required params: `name:Type`
        - Optional params (has default): `name?:Type`
        - Varargs/varkw included only if declared.
      This is for **display/telemetry**; runtime validation relies on the call-plan.

    Return type
    -----------
    • `return_type` reflects the callable's return annotation if present, else `Any`.
      For adapters that override the plan (e.g., AgentTool / MCPProxyTool), they
      should set an appropriate `return_type` and call `_rebuild_signature_str()`.

    Primary API
    -----------
    __init__(func: Callable, name: str, description: str = "",
             type: str = "function", source: str = "default")

    invoke(inputs: Mapping[str, Any]) -> Any

    Introspection
    -------------
    • name, description, type, source (read-only tags)
    • arguments_map (read-only OrderedDict view)
    • signature (schema-derived string)
    • return_type (read-only, display only)

    Strictness (unchanged)
    ----------------------
    - Unknown top-level keys are errors (even if **kwargs exists); extras must go
      under `_kwargs`.
    - Duplicate supply across top-level and `_kwargs` is an error.
    - `_args` must be list/tuple; `_kwargs` must be Mapping.
    """

    # -------------------------
    # Construction
    # -------------------------
    def __init__(
        self,
        func: Callable,
        name: str,
        description: str = "",
        type: str = "function",
        source: str = "default",
    ) -> None:
        self._func: Callable = func
        self._name: str = name
        self._description: str = description
        self._type: str = type
        self._source: str = source

        # Build call plan once (still uses inspect internally for parameters).
        try:
            sig = inspect.signature(inspect.unwrap(func))
        except Exception as e:
            raise ToolDefinitionError(f"{name}: could not inspect callable: {e}") from e

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
        ) = self._build_arguments_map_and_plan(sig)

        # New: derive return type (display-only) from annotation; default Any
        self._return_type:type = self._func.__annotations__.get("return", Any)

        # New: schema-derived signature string (from arguments_map & return_type)
        self._sig_str: str = ""
        self._rebuild_signature_str()

    # -------------------------
    # Read-only tags and doc
    # -------------------------
    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def type(self) -> str:
        return self._type

    @property
    def source(self) -> str:
        return self._source

    @property
    def full_name(self) -> str:
        return f"{self._type}.{self._source}.{self._name}"

    @property
    def func(self) -> Callable:
        return self._func

    @property
    def arguments_map(self):
        return self._arguments_map

    # New: return type (display only)
    @property
    def return_type(self) -> Any:
        return self._return_type

    # New: canonical signature string
    @property
    def signature(self) -> str:
        return self._sig_str

    # -------------------------
    # Core invocation (unchanged)
    # -------------------------
    def invoke(self, inputs: Mapping[str, Any]) -> Any:
        if not isinstance(inputs, Mapping):
            raise ToolInvocationError(f"{self._name}: inputs must be a mapping")

        ARGS_KEY = "_args"
        KWARGS_KEY = "_kwargs"

        # Validate reserved keys early
        if ARGS_KEY in inputs and not isinstance(inputs[ARGS_KEY], (list, tuple)):
            raise ToolInvocationError(f"{self._name}: '{ARGS_KEY}' must be a list or tuple")
        if KWARGS_KEY in inputs and not isinstance(inputs[KWARGS_KEY], Mapping):
            raise ToolInvocationError(f"{self._name}: '{KWARGS_KEY}' must be a Mapping")

        # Unknown top-level keys are not allowed (extras belong under _kwargs if any)
        provided_names = set(inputs.keys()) - {ARGS_KEY, KWARGS_KEY}
        known_names = set(self._arguments_map.keys())
        unknown = sorted(provided_names - known_names)
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

        # Build args (positional-only), gap-safe
        args: List[Any] = []
        seen_gap = False
        for pname in self.posonly_order:
            present = pname in inputs
            if not present:
                seen_gap = True
                continue
            if seen_gap:
                raise ToolInvocationError(
                    f"{self._name}: positional-only parameters must be a contiguous prefix; missing a value before '{pname}'"
                )
            args.append(inputs[pname])

        # Extra *args if declared
        if self.has_varargs and ARGS_KEY in inputs:
            extra_args = inputs[ARGS_KEY]
            if not isinstance(extra_args, (list, tuple)):
                raise ToolInvocationError(f"{self._name}: '{ARGS_KEY}' must be list or tuple")
            args.extend(list(extra_args))

        # Build kwargs
        kwargs: Dict[str, Any] = {}
        for pname in (self.p_or_kw_names + self.kw_only_names):
            if pname in inputs:
                kwargs[pname] = inputs[pname]

        # Extra **kwargs if declared
        if self.has_varkw and KWARGS_KEY in inputs:
            extra_kwargs = inputs[KWARGS_KEY]
            if not isinstance(extra_kwargs, Mapping):
                raise ToolInvocationError(f"{self._name}: '{KWARGS_KEY}' must be a Mapping")
            dupes = set(kwargs.keys()) & set(extra_kwargs.keys())
            if dupes:
                raise ToolInvocationError(
                    f"{self._name}: duplicate keys supplied both as named inputs and in '{KWARGS_KEY}': {dupes}"
                )
            kwargs.update(extra_kwargs)  # type: ignore[arg-type]

        # Final call
        try:
            return self._func(*args, **kwargs)
        except ToolInvocationError:
            raise
        except Exception as e:
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
          - 'signature' is the canonical schema-derived string (display only).
          - 'return_type' is display-only.
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
            "signature": self.signature,
            "return_type": self._type_to_str(self._return_type),
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
        bool, Optional[str],                   # has_varkw, varkw_name
    ]:
        arguments_map: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        posonly_order: List[str] = []
        p_or_kw_names: List[str] = []
        kw_only_names: List[str] = []
        required_names: set = set()
        has_varargs: bool = False
        varargs_name: Optional[str] = None
        has_varkw: bool = False
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
                raise ToolDefinitionError(f"Unexpected parameter kind for {pname}: {kind_enum!r}")

            arguments_map[pname] = {
                "index": index,
                "kind": kind_enum,
                "ann": ann,
                "has_default": default is not NO_DEFAULT,
                "default": default,
            }
            index += 1

        # Required named parameters (no default) among p_or_kw + kw_only
        for pname in (p_or_kw_names + kw_only_names):
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

    # -------------------------
    # Internal: signature-string builder (schema-derived)
    # -------------------------
    def _rebuild_signature_str(self) -> None:
        """Refresh the canonical signature string from the current plan + return_type.

        Rules:
        - Required param:  name:Type
        - Optional param:  name?:Type  (and if an explicit default exists: ' = <repr(default)>')
        - Varargs/varkw appended as *name / **name
        """
        ordered = sorted(self._arguments_map.items(), key=lambda kv: kv[1]["index"])
        parts: List[str] = []

        for pname, spec in ordered:
            ann_str = self._type_to_str(spec.get("ann", inspect._empty))
            has_default = bool(spec.get("has_default", False))
            default_val = spec.get("default", NO_DEFAULT)

            # Base token: required by default
            if not has_default:
                token = f"{pname}: {ann_str}"
            else:
                token = f"{pname}?: {ann_str}"
                # Only show '= ...' when an explicit default is present (not NO_DEFAULT)
                if default_val is not NO_DEFAULT:
                    try:
                        token += f" = {repr(default_val)}"
                    except Exception:
                        token += " = <default>"

            parts.append(token)

        # Varargs / varkw tokens (use declared names if present)
        if self.has_varargs:
            parts.append(f"*{self.varargs_name or 'args'}")
        if self.has_varkw:
            parts.append(f"**{self.varkw_name or 'kwargs'}")

        rtype_str = self._type_to_str(self._return_type)
        self._sig_str = f"{self.full_name}({', '.join(parts)}) -> {rtype_str}"

    @staticmethod
    def _type_to_str(t: Any) -> str:
        """Best-effort readable type name for display-only contexts."""
        if t is inspect._empty:
            return "Any"
        try:
            # Prefer simple names for builtins / classes
            n = getattr(t, "__name__", None)
            if isinstance(n, str):
                return n
            # Fallback to repr for typing constructs / generics
            return repr(t)
        except Exception:
            return "Any"
