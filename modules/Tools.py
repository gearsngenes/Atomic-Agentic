# Tools.py
from __future__ import annotations

import inspect
import re
import types
import json
import os
import pathlib
import datetime
import decimal
import uuid
import enum
import dataclasses
from collections import OrderedDict
from typing import (
    Any,
    Mapping,
    Callable,
    List,
    Optional,
    Tuple,
    get_origin,
    get_args,
    Annotated,
    Literal,
    ClassVar,
    Type as TypingType,
    Union,
)

# ───────────────────────────────────────────────────────────────────────────────
# Public API
# ───────────────────────────────────────────────────────────────────────────────
__all__ = ["Tool", "ToolError", "ToolDefinitionError", "ToolInvocationError"]

# Optional attrs support
try:
    import attrs as _attrs  # type: ignore
except Exception:  # pragma: no cover - optional
    _attrs = None  # type: ignore

# ───────────────────────────────────────────────────────────────────────────────
# Exceptions & sentinels
# ───────────────────────────────────────────────────────────────────────────────
class ToolError(Exception):
    """Base exception for Tool-related errors."""


class ToolDefinitionError(ToolError):
    """Raised when a callable is incompatible at Tool construction time."""


class ToolInvocationError(ToolError):
    """Raised when inputs are invalid for invocation or binding fails."""


# Sentinel to mark "no default" for parameters
NO_DEFAULT = object()

# ───────────────────────────────────────────────────────────────────────────────
# Parameter kind → JSON-stable "mode" token (for REST/MCP/A2A)
# ───────────────────────────────────────────────────────────────────────────────
KIND_TO_MODE = {
    inspect.Parameter.POSITIONAL_ONLY: "posonly",
    inspect.Parameter.POSITIONAL_OR_KEYWORD: "pos_or_kw",
    inspect.Parameter.KEYWORD_ONLY: "kwonly",
}
MODE_TO_KIND = {v: k for k, v in KIND_TO_MODE.items()}

# ───────────────────────────────────────────────────────────────────────────────
# Canonical type normalization (annotation → string + meta)
# ───────────────────────────────────────────────────────────────────────────────
# Canonical heads we emit for `ann`
_CANON_SCALARS = {
    "any", "object", "none", "bool", "int", "float", "complex", "str", "bytes", "bytearray", "memoryview",
}
_CANON_CONTAINERS = {"list", "tuple", "dict", "set", "frozenset"}
_CANON_ABSTRACTS = {"iterable", "iterator", "generator"}
_CANON_MISC = {"callable", "pathlike", "date", "datetime", "time", "timedelta", "decimal", "uuid", "enum"}

# Synonyms we fold to canonical heads (objects and strings)
_CANON_SYNONYMS: dict[str, set[Any]] = {
    # scalars
    "any": {Any, "any", "Any"},
    "object": {object, "object", "builtins.object"},
    "none": {type(None), getattr(types, "NoneType", type(None)), None, "none", "None", "NoneType", "type(None)", "null"},
    "bool": {bool, "bool"},
    "int": {int, "int"},
    "float": {float, "float"},
    "complex": {complex, "complex"},
    "str": {str, "str", "string"},
    "bytes": {bytes, "bytes"},
    "bytearray": {bytearray, "bytearray"},
    "memoryview": {memoryview, "memoryview"},
    # concrete containers (fold ABCs to these)
    "list": {"list", list, "List"},
    "tuple": {"tuple", tuple, "Tuple"},
    "dict": {"dict", dict, "Dict", "Mapping", "collections.abc.Mapping", "typing.Mapping"},
    "set": {"set", set, "Set"},
    "frozenset": {"frozenset", frozenset, "FrozenSet", "typing.FrozenSet"},
    # abstracts
    "iterable": {"iterable", "Iterable", "collections.abc.Iterable"},
    "iterator": {"iterator", "Iterator", "collections.abc.Iterator"},
    "generator": {"generator", "Generator", "collections.abc.Generator", types.GeneratorType},
    # callable & paths
    "callable": {"callable", "Callable", "typing.Callable", "collections.abc.Callable", types.FunctionType},
    "pathlike": {"pathlike", os.PathLike, pathlib.Path, "Path", "PathLike"},
    # dates & misc
    "date": {"date", datetime.date},
    "datetime": {"datetime", datetime.datetime},
    "time": {"time", datetime.time},
    "timedelta": {"timedelta", datetime.timedelta},
    "decimal": {"decimal", decimal.Decimal, "Decimal"},
    "uuid": {"uuid", uuid.UUID, "UUID"},
    "enum": {"enum", enum.Enum, "Enum"},
}

# For typing origins
_ORIGIN_TO_HEAD = {
    list: "list",
    tuple: "tuple",
    dict: "dict",
    set: "set",
    frozenset: "frozenset",
}

# Strip these prefixes from textual annotations
_STRIP_PREFIXES = (
    "typing.",
    "builtins.",
    "collections.",
    "collections.abc.",
)

# Simple parsers for stringized annotations
_GENERIC_RE = re.compile(r"^\s*([a-zA-Z_][\w\.]*)\s*\[(.*)\]\s*$", re.S)
_CALLABLE_RE = re.compile(r"^\s*callable\s*\[(.*)\]\s*$", re.S)  # callable[[args], ret] or callable[..., ret]


def _strip_known_prefixes(s: str) -> str:
    t = s.strip()
    for pref in _STRIP_PREFIXES:
        if t.startswith(pref):
            t = t[len(pref):]
    return t


def _split_top_level(s: str, sep: str = ",") -> List[str]:
    """Split by sep at top level, respecting brackets and simple quotes."""
    out: List[str] = []
    buf: List[str] = []
    depth = 0
    i, n = 0, len(s)
    in_squote = False
    in_dquote = False
    while i < n:
        ch = s[i]
        if ch == "'" and not in_dquote:
            in_squote = not in_squote
            buf.append(ch)
        elif ch == '"' and not in_squote:
            in_dquote = not in_dquote
            buf.append(ch)
        elif not in_squote and not in_dquote:
            if ch in "[(":
                depth += 1
                buf.append(ch)
            elif ch in "])":
                depth -= 1
                buf.append(ch)
            elif ch == sep and depth == 0:
                out.append("".join(buf).strip())
                buf = []
            else:
                buf.append(ch)
        else:
            buf.append(ch)
        i += 1
    if buf:
        out.append("".join(buf).strip())
    return out


def _canon_head_for_object(obj: Any) -> Optional[str]:
    """Map a runtime class/ABC/etc. to a canonical head if possible."""
    for head, syns in _CANON_SYNONYMS.items():
        if obj in syns:
            return head
        try:
            if isinstance(obj, type) and any(obj is s for s in syns if isinstance(s, type)):
                return head
        except Exception:
            pass
    return None


def _canonize_str_annotation(text: str) -> Tuple[str, Optional[dict]]:
    """Parse and normalize a textual annotation into `ann` string + optional meta."""
    t = _strip_known_prefixes(text).strip()
    t_low = t.lower()

    # simple literals
    if t_low in {"none", "nonetype", "type(none)", "null"}:
        return "none", None
    if t_low in {"any"}:
        return "any", None
    if t_low in {"object"}:
        return "object", None

    # callable textual forms
    m = _CALLABLE_RE.match(t_low)
    if m:
        inner = m.group(1).strip()
        meta: dict = {"args": "...", "return": "any"}
        parts = _split_top_level(inner)
        if len(parts) == 2:
            a, r = parts
            r = _strip_known_prefixes(r).strip()
            if a.strip().startswith("[") and a.strip().endswith("]"):
                a_inner = a.strip()[1:-1]
                arg_tokens = [tok for tok in _split_top_level(a_inner) if tok]
                meta["args"] = [_canonize_str_annotation(tok)[0] for tok in arg_tokens]
            elif a.strip() == "...":
                meta["args"] = "..."
            else:
                meta["args"] = "..."
            meta["return"] = _canonize_str_annotation(r)[0]
        return "callable", meta

    # generics: head[...]
    gm = _GENERIC_RE.match(t_low)
    if gm:
        head_raw, inner = gm.group(1), gm.group(2)
        head = _strip_known_prefixes(head_raw).lower()
        items = [tok for tok in _split_top_level(inner) if tok]

        if head in {"list", "set", "frozenset"}:
            child = _canonize_str_annotation(items[0])[0] if items else "any"
            return f"{head}[{child}]", None
        if head == "tuple":
            if not items:
                return "tuple", None
            children = ", ".join(_canonize_str_annotation(tok)[0] for tok in items)
            return f"tuple[{children}]", None
        if head == "dict":
            key = _canonize_str_annotation(items[0])[0] if items else "any"
            val = _canonize_str_annotation(items[1])[0] if len(items) > 1 else "any"
            return f"dict[{key}, {val}]", None
        if head in {"union", "optional"}:
            parts = [_canonize_str_annotation(tok)[0] for tok in items]
            parts = sorted(parts)
            if head == "optional":
                base = parts[0] if parts else "any"
                return f"optional[{base}]", None
            non_none = [p for p in parts if p != "none"]
            if len(parts) == 2 and len(non_none) == 1 and "none" in parts:
                return f"optional[{non_none[0]}]", None
            return f"union[{', '.join(parts)}]", None
        if head == "annotated":
            base = _canonize_str_annotation(items[0])[0] if items else "any"
            extras = items[1:] if len(items) > 1 else []
            return f"annotated[{base}]", {"extras": extras} if extras else None
        if head == "literal":
            return f"literal[{', '.join(items)}]", {"args": items} if items else None
        if head in {"type", "classvar"} and items:
            base = _canonize_str_annotation(items[0])[0]
            return f"{head}[{base}]", None

        ch = _canon_head_for_object(head)
        if ch:
            children = ", ".join(_canonize_str_annotation(tok)[0] for tok in items)
            return f"{ch}[{children}]", None
        return "object", None

    ch = _canon_head_for_object(t_low) or _canon_head_for_object(t)
    if ch:
        return ch, None
    return "object", None


def _canonize_annotation(ann: Any) -> Tuple[str, Optional[dict]]:
    """
    Convert any annotation (class/typing/ABC/string) to a canonical string `ann`
    and optional `ann_meta`.

    Rules:
    - Missing → "any"; None/NoneType → "none"; unknown → "object".
    - Builtins-as-generics (PEP 585) are the canonical heads for containers.
    - Union/Optional normalized; Callable parsed into args/return.
    """
    if ann is inspect._empty:
        return "any", None

    if isinstance(ann, str):
        return _canonize_str_annotation(ann)

    if ann is None or ann is type(None) or getattr(ann, "__name__", None) == "NoneType":
        return "none", None

    origin = get_origin(ann)

    # Union / X|Y (PEP 604)
    if origin in (Union, types.UnionType):
        parts = [_canonize_annotation(a)[0] for a in get_args(ann)]
        parts = sorted(parts)
        non_none = [p for p in parts if p != "none"]
        if len(parts) == 2 and len(non_none) == 1 and "none" in parts:
            return f"optional[{non_none[0]}]", None
        return f"union[{', '.join(parts)}]", None

    # Callable
    from collections import abc as c_abc
    if origin in (Callable, c_abc.Callable):
        args = get_args(ann)
        meta: dict = {"args": "...", "return": "any"}
        if args:
            params, ret = args
            if params is Ellipsis:
                meta["args"] = "..."
            else:
                meta["args"] = [_canonize_annotation(p)[0] for p in params]
            meta["return"] = _canonize_annotation(ret)[0]
        return "callable", meta

    # ClassVar, Annotated, Literal, Type[T]
    if origin is ClassVar:
        base = _canonize_annotation(get_args(ann)[0])[0] if get_args(ann) else "any"
        return f"classvar[{base}]", None

    if origin is Annotated:
        args = get_args(ann)
        base = _canonize_annotation(args[0])[0] if args else "any"
        extras = [repr(x) for x in args[1:]] if len(args) > 1 else []
        return f"annotated[{base}]", {"extras": extras} if extras else None

    if origin is Literal:
        vals = [repr(x) for x in get_args(ann)]
        return f"literal[{', '.join(vals)}]", {"args": vals} if vals else None

    if origin in (TypingType, type):
        base = _canonize_annotation(get_args(ann)[0])[0] if get_args(ann) else "any"
        return f"type[{base}]", None

    # PEP 585 generics
    if origin in _ORIGIN_TO_HEAD:
        head = _ORIGIN_TO_HEAD[origin]
        args = [_canonize_annotation(a)[0] for a in get_args(ann)]
        if head in {"list", "set", "frozenset"}:
            inner = args[0] if args else "any"
            return f"{head}[{inner}]", None
        if head == "tuple":
            return (f"tuple[{', '.join(args)}]" if args else "tuple"), None
        if head == "dict":
            k = args[0] if args else "any"
            v = args[1] if len(args) > 1 else "any"
            return f"dict[{k}, {v}]", None

    # ABCs and concrete classes (fold)
    ch = _canon_head_for_object(ann)
    if ch:
        if ch == "callable":
            return "callable", {"args": "...", "return": "any"}
        return ch, None

    return "object", None

# ───────────────────────────────────────────────────────────────────────────────
# JSON-safe default conversion at build time
# ───────────────────────────────────────────────────────────────────────────────
def _iso8601_duration_from_timedelta(td: datetime.timedelta) -> str:
    """
    Serialize timedelta to ISO-8601 duration (e.g., 'PT3H5M7.25S').
    Chosen for broad interop with JSON Schema's "duration" format.  :contentReference[oaicite:3]{index=3}
    """
    total = td.total_seconds()
    sign = "-" if total < 0 else ""
    total = abs(total)
    days, rem = divmod(total, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)
    s = "P"
    if days:
        s += f"{int(days)}D"
    # include time component whenever sub-day component exists (or to avoid bare 'P')
    s += "T"
    if hours:
        s += f"{int(hours)}H"
    if minutes:
        s += f"{int(minutes)}M"
    if float(seconds).is_integer():
        s += f"{int(seconds)}S"
    else:
        s_fmt = f"{seconds:.9f}".rstrip("0").rstrip(".")
        s += f"{s_fmt}S"
    return sign + s


def _is_json_roundtrippable(x: object) -> bool:
    try:
        json.dumps(x)
        return True
    except Exception:
        return False


def _jsonify_default(obj: object, *, _seen: set[int] | None = None, _depth: int = 0, _max_depth: int = 8):
    """
    Return a JSON-encodable value for `obj`. Best-effort, safe, and deterministic.
    - Uses model_dump()/dict() for Pydantic models, dataclasses.asdict, attrs.asdict,
      to_dict()/__dict__ for common containers, and special handling for datetime/UUID/Decimal/Enum/timedelta.
    - Coerces mapping keys to strings. Falls back to repr(obj) when necessary.  :contentReference[oaicite:4]{index=4}
    """
    if _seen is None:
        _seen = set()
    oid = id(obj)
    if oid in _seen:
        return "<recursion>"
    _seen.add(oid)

    # Primitives
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    # Well-known specials
    if isinstance(obj, enum.Enum):
        return obj.name  # per spec: enum serialized by name
    if isinstance(obj, (datetime.datetime, datetime.date, datetime.time)):
        return obj.isoformat()
    if isinstance(obj, datetime.timedelta):
        return _iso8601_duration_from_timedelta(obj)
    if isinstance(obj, uuid.UUID):
        return str(obj)
    if isinstance(obj, decimal.Decimal):
        return str(obj)
    if isinstance(obj, (os.PathLike, pathlib.Path)):
        return os.fspath(obj)

    # Containers
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [_jsonify_default(v, _seen=_seen, _depth=_depth + 1, _max_depth=_max_depth) for v in obj]
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            out[str(k)] = _jsonify_default(v, _seen=_seen, _depth=_depth + 1, _max_depth=_max_depth)
        return out

    # Depth cap
    if _depth >= _max_depth:
        return repr(obj)

    # Pydantic v2 → v1
    try:
        if hasattr(obj, "model_dump") and callable(getattr(obj, "model_dump")):
            return _jsonify_default(obj.model_dump(), _seen=_seen, _depth=_depth + 1, _max_depth=_max_depth)  # :contentReference[oaicite:5]{index=5}
        if hasattr(obj, "dict") and callable(getattr(obj, "dict")):
            return _jsonify_default(obj.dict(), _seen=_seen, _depth=_depth + 1, _max_depth=_max_depth)
    except Exception:
        pass

    # dataclasses
    try:
        if dataclasses.is_dataclass(obj):
            return _jsonify_default(dataclasses.asdict(obj), _seen=_seen, _depth=_depth + 1, _max_depth=_max_depth)  # :contentReference[oaicite:6]{index=6}
    except Exception:
        pass

    # attrs
    try:
        if _attrs is not None and _attrs.has(type(obj)):  # type: ignore[attr-defined]
            return _jsonify_default(_attrs.asdict(obj), _seen=_seen, _depth=_depth + 1, _max_depth=_max_depth)  # :contentReference[oaicite:7]{index=7}
    except Exception:
        pass

    # Conventional hooks
    try:
        if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
            return _jsonify_default(obj.to_dict(), _seen=_seen, _depth=_depth + 1, _max_depth=_max_depth)
    except Exception:
        pass

    # __dict__
    try:
        if hasattr(obj, "__dict__"):
            return _jsonify_default(vars(obj), _seen=_seen, _depth=_depth + 1, _max_depth=_max_depth)
    except Exception:
        pass

    # Final fallback: keep if JSON-able else repr
    return obj if _is_json_roundtrippable(obj) else repr(obj)

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
        type: str = "function",
        source: str = "default",
    ) -> None:
        self._func: Callable = func
        self._name: str = name
        self._description: str = description
        self._type: str = type
        self._source: str = source

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
        return f"{self._type}.{self._source}.{self._name}"

    @property
    def func(self) -> Callable:
        return self._func

    @property
    def arguments_map(self):
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
        # argmap_serialized: OrderedDict[str, Any] = OrderedDict()
        # for name, spec in sorted(self._arguments_map.items(), key=lambda kv: kv[1]["index"]):
        #     kind_enum: inspect._ParameterKind = spec["kind"]
        #     entry = {
        #         "index": spec["index"],
        #         "mode": spec.get("mode", KIND_TO_MODE.get(kind_enum, "pos_or_kw")),
        #         "kind_name": kind_enum.name,  # optional, human friendly
        #         "ann": spec.get("ann", "any"),
        #         "has_default": spec.get("has_default", False),
        #     }
        #     if "ann_meta" in spec and spec["ann_meta"] is not None:
        #         entry["ann_meta"] = spec["ann_meta"]
        #     if spec.get("has_default", False) and "default" in spec:
        #         entry["default"] = spec["default"]  # already JSON-safe
        #     argmap_serialized[name] = entry

        return OrderedDict(
            name=self._name,
            description=self._description,
            type=self._type,
            source=self._source,
            signature=self.signature,
            return_type=self._return_type,
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
        arguments_map: "OrderedDict[str, OrderedDict[str, Any]]" = OrderedDict()
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
