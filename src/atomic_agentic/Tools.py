# Tools.py
from __future__ import annotations

from .__utils__ import _canonize_annotation, _jsonify_default, KIND_TO_MODE
from ._exceptions import ToolDefinitionError, ToolInvocationError
import inspect
from collections import OrderedDict
from typing import (
    Any,
    Mapping,
    Callable,
    List,
    Optional,
    Tuple
)

# ───────────────────────────────────────────────────────────────────────────────
# Public API
# ───────────────────────────────────────────────────────────────────────────────
__all__ = ["Tool"]

# ───────────────────────────────────────────────────────────────────────────────
# Tool
# ───────────────────────────────────────────────────────────────────────────────
class Tool:
    """
    Stateless wrapper around a Python callable with a dict-first invocation API.

    Construction builds:
      • arguments_map (name → {index, mode/kind, ann, ann_meta?, has_default, default? [JSON-safe]})
      • call-plan (posonly / p_or_kw / kw_only / required / varargs / varkw)
      • return_type (canonical string)
      • signature (canonical, display-only)

    Parameter kinds follow Python's calling convention (pos-only, pos-or-kw, kw-only, *args, **kwargs).  :contentReference[oaicite:8]{index=8}
    """

    # -------------------------
    # Construction
    # -------------------------
    def __init__(
        self,
        func: Callable,
        name: str,
        description: str = "",
        source: str = "default",
    ) -> None:
        self._func: Callable = func
        self._name: str = (name or func.__name__) or "unnamed_callable"
        self._description: str = (description or func.__doc__) or ""
        self._source: str = source
        
        self.module = getattr(self.func, "__module__", None)
        self.qualname = getattr(self.func, "__qualname__", None)

        if self.module is None or self.qualname is None:
            raise TypeError(
                f"Callable {self.func!r} must be an accessible and retrievable function. "
                "Provided callable has no `__module__` or `__qualname__`; "
                f"cannot safely dehydrate."
            )

        # Build call plan once (unwrap to reach original if decorated)
        try:
            sig = inspect.signature(inspect.unwrap(func))  # :contentReference[oaicite:9]{index=9}
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

        # Derive canonical return type string
        raw_ret = getattr(self._func, "__annotations__", {}).get("return", inspect._empty)
        self._return_type: str = _canonize_annotation(raw_ret)[0]

        # Build signature string
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
        return f"{type(self).__name__}.{self._source}.{self._name}"

    @property
    def func(self) -> Callable:
        return self._func

    @property
    def arguments_map(self) -> OrderedDict[str, Any]:
        return self._arguments_map

    @property
    def return_type(self) -> str:
        return self._return_type

    @property
    def signature(self) -> str:
        return self._sig_str

    # -------------------------
    # Core invocation
    # -------------------------
    def invoke(self, inputs: Mapping[str, Any]) -> Any:
        """
        Invoke the underlying callable using a dict of inputs.
        Rules:
        - Top-level keys must match known parameter names; extras go under '_kwargs' if **varkw exists.
        - Positional-only params must form a contiguous prefix (we read them by name, in order).
        - Optional reserved keys:
            _args: list/tuple for extra *args (if function declares VAR_POSITIONAL)
            _kwargs: mapping for extra **kwargs (if function declares VAR_KEYWORD)
        """
        if not isinstance(inputs, Mapping):
            raise ToolInvocationError(f"{self._name}: inputs must be a mapping")

        ARGS_KEY = "_args"
        KWARGS_KEY = "_kwargs"

        # Validate reserved keys early
        if ARGS_KEY in inputs and not isinstance(inputs[ARGS_KEY], (list, tuple)):
            raise ToolInvocationError(f"{self._name}: '{ARGS_KEY}' must be a list or tuple")
        if KWARGS_KEY in inputs and not isinstance(inputs[KWARGS_KEY], Mapping):
            raise ToolInvocationError(f"{self._name}: '{KWARGS_KEY}' must be a Mapping")

        # Unknown top-level keys are not allowed; extras must go in _kwargs if **varkw exists
        provided_names = set(inputs.keys()) - {ARGS_KEY, KWARGS_KEY}
        known_names = set(self._arguments_map.keys())
        unknown = sorted(provided_names - known_names)
        if unknown:
            if not self.has_varkw:
                raise ToolInvocationError(f"{self._name}: unexpected keys: {unknown}")
            raise ToolInvocationError(
                f"{self._name}: unexpected keys {unknown}; place extras under '{KWARGS_KEY}' because function accepts **kwargs"
            )

        # Required named parameters (no default) among p_or_kw + kw_only
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
            args.extend(list(extra_args))

        # Build kwargs
        kwargs: OrderedDict[str, Any] = OrderedDict()
        for pname in (self.p_or_kw_names + self.kw_only_names):
            if pname in inputs:
                kwargs[pname] = inputs[pname]

        # Extra **kwargs if declared
        if self.has_varkw and KWARGS_KEY in inputs:
            extra_kwargs = inputs[KWARGS_KEY]
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
    def to_dict(self) -> OrderedDict[str, Any]:
        """
        Serialize Tool metadata and signature plan for registries and UIs.

        Notes
        -----
        - 'ann' is a canonical Python-flavored string (not repr of type objects).
        - 'ann_meta' is included for complex heads ('callable', 'annotated', 'literal').
        - 'default' is included only when a parameter has a default; it is JSON-safe already.
        - 'mode' is a JSON-stable parameter kind token; 'kind_name' is for human display.
        """

        return OrderedDict(
            # Tool Type
            tool_type = type(self).__name__,
            # Top-level metadata
            name=self._name,
            description=self._description,
            source=self._source,
            signature=self.signature,
            return_type=self._return_type,
            # Reconstructive metadata
            module=self.module,
            qualname=self.qualname,
            # Arguments metadata
            arguments_map=self.arguments_map,
            posonly_order=list(self.posonly_order),
            p_or_kw_names=list(self.p_or_kw_names),
            kw_only_names=list(self.kw_only_names),
            required_names=sorted(self.required_names),
            has_varargs=self.has_varargs,
            varargs_name=self.varargs_name,
            has_varkw=self.has_varkw,
            varkw_name=self.varkw_name,
        )

    # -------------------------
    # Internal: build arg map + call plan (JSON-safe at creation)
    # -------------------------
    def _build_arguments_map_and_plan(
        self, sig: inspect.Signature
    ) -> Tuple[
        OrderedDict[str, OrderedDict[str, Any]],      # arguments_map
        List[str],                                    # posonly_order
        List[str],                                    # p_or_kw_names
        List[str],                                    # kw_only_names
        set,                                          # required_names
        bool, Optional[str],                          # has_varargs, varargs_name
        bool, Optional[str],                          # has_varkw, varkw_name
    ]:
        arguments_map: OrderedDict[str, OrderedDict[str, Any]] = OrderedDict()
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

            raw_ann = p.annotation if p.annotation is not inspect._empty else inspect._empty
            ann_str, ann_meta = _canonize_annotation(raw_ann)
            kind_enum = p.kind  # exact enum
            mode = KIND_TO_MODE.get(kind_enum, "pos_or_kw")

            has_def = p.default is not inspect._empty
            default_json = _jsonify_default(p.default) if has_def else None

            if kind_enum is p.POSITIONAL_ONLY:
                posonly_order.append(pname)
            elif kind_enum is p.POSITIONAL_OR_KEYWORD:
                p_or_kw_names.append(pname)
            elif kind_enum is p.KEYWORD_ONLY:
                kw_only_names.append(pname)
            else:
                raise ToolDefinitionError(f"Unexpected parameter kind for {pname}: {kind_enum!r}")

            entry: OrderedDict[str, Any] = OrderedDict(
                index=index,
                kind=kind_enum,      # internal convenience
                mode=mode,           # JSON-stable token
                ann=ann_str,
                has_default=has_def,
            )
            if ann_meta is not None:
                entry["ann_meta"] = ann_meta
            if has_def:
                entry["default"] = default_json  # JSON-safe now

            arguments_map[pname] = entry
            index += 1

        # Required named parameters (no default) among p_or_kw + kw_only
        for pname in (p_or_kw_names + kw_only_names):
            if not arguments_map[pname]["has_default"]:
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
        """Refresh the canonical signature string from the current plan + return_type."""
        ordered = sorted(self._arguments_map.items(), key=lambda kv: kv[1]["index"])
        parts: List[str] = []

        for pname, spec in ordered:
            ann_str = spec.get("ann", "any")
            has_default = bool(spec.get("has_default", False))
            token: str
            if not has_default:
                token = f"{pname}: {ann_str}"
            else:
                token = f"{pname}?: {ann_str}"
                if "default" in spec:
                    try:
                        token += f" = {repr(spec['default'])}"
                    except Exception:
                        token += " = <default>"
            parts.append(token)

        if self.has_varargs:
            parts.append(f"*{self.varargs_name or 'args'}")
        if self.has_varkw:
            parts.append(f"**{self.varkw_name or 'kwargs'}")

        self._sig_str = f"{self.full_name}({', '.join(parts)}) -> {self._return_type}"
