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
import asyncio
from queue import Queue
import threading
from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession
from typing import (
    Any,
    Callable,
    List,
    Optional,
    Tuple,
    Union,
    get_origin,
    get_args,
    Annotated,
    Literal,
    ClassVar,
    Type as TypingType,
)
# Optional attrs support
try:
    import attrs as _attrs  # type: ignore
except Exception:  # pragma: no cover - optional
    _attrs = None  # type: ignore

from .Exceptions import *

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

#------------------------MCPProxyTool Helpers------------------------#
# ───────────────────────────────────────────────────────────────────────────────
# Helpers (self-contained; ensure there is ONLY ONE definition of each)
# ───────────────────────────────────────────────────────────────────────────────
def _run_sync(coro):
    """
    Run a coroutine synchronously with robust loop handling:
      • If no loop or the current loop is CLOSED → create a fresh loop, run, shutdown, close.
      • If a loop is RUNNING → execute in a worker thread via asyncio.run().
      • Else → run_until_complete on the existing idle loop.

    This avoids 'Event loop is closed' after earlier asyncio.run(...) usage.
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = None

    # No loop or closed loop → make a temporary one
    if loop is None or loop.is_closed():
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(coro)
        finally:
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            finally:
                asyncio.set_event_loop(None)
                loop.close()

    # Loop is running in this thread → run in a worker thread
    if loop.is_running():
        q: "Queue[Tuple[str, Any]]" = Queue()

        def _worker():
            try:
                q.put(("ok", asyncio.run(coro)))
            except BaseException as exc:  # surface original exception to caller
                q.put(("err", exc))

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        status, payload = q.get()
        t.join()
        if status == "err":
            raise payload
        return payload

    # Loop exists and is idle → run directly
    return loop.run_until_complete(coro)


def _normalize_url(u: str) -> str:
    """
    Ensure a valid MCP streamable-HTTP URL. If no path is provided, default to '/mcp'.
    """
    from urllib.parse import urlparse, urlunparse
    parts = urlparse(u.strip())
    if not parts.scheme or not parts.netloc:
        raise ToolDefinitionError(f"Invalid MCP URL: {u!r}")
    if not parts.path or parts.path == "/":
        parts = parts._replace(path="/mcp")
    return urlunparse(parts)



def _extract_structured_or_text(result: Any) -> Optional[Any]:
    """
    OLD behavior, restored and streamlined:

    1) If `structuredContent` exists, return it.
       • If it's exactly {'result': X}, unwrap to X.
    2) Else if `content` exists, concatenate its 'text' blocks.
    3) Else return None (caller can fall back to model_dump()/raw).
    """
    # Prefer structuredContent (attribute or mapping)
    sc = getattr(result, "structuredContent", None)
    if sc is None and isinstance(result, dict):
        sc = result.get("structuredContent")
    if sc is not None:
        if isinstance(sc, dict) and set(sc.keys()) == {"result"}:
            return sc["result"]
        return sc

    # Fallback to unstructured content blocks (join text)
    content = getattr(result, "content", None)
    if content is None and isinstance(result, dict):
        content = result.get("content")
    if isinstance(content, list):
        texts = [p.get("text") for p in content if isinstance(p, dict) and "text" in p]
        if texts:
            return "".join(texts)

    return None


def _to_plain(result: Any) -> Any:
    """Last resort: pydantic v2 models → dict; otherwise pass result through."""
    try:
        if hasattr(result, "model_dump") and callable(result.model_dump):
            return result.model_dump()
    except Exception:
        pass
    return result

# ---------- MCP discovery (list tool names) ----------
async def _async_discover_mcp_tool_names(url: str, headers: Optional[dict]) -> List[str]:
    async with streamablehttp_client(url=url, headers=headers or None) as transport:
        read, write = _extract_rw(transport)
        async with ClientSession(read, write) as session:
            # REQUIRED handshake
            await session.initialize()
            tools_resp = await session.list_tools()
            tool_objs = getattr(tools_resp, "tools", tools_resp)
            return [t.name for t in tool_objs]

def _discover_mcp_tool_names(url: str, headers: Optional[dict]) -> List[str]:
    return _run_sync(_async_discover_mcp_tool_names(url, headers))

_JSON_TO_PY = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "object": dict,
    "array": list,
    "null": type(None),
}

# ---------- URL helpers ----------
from urllib.parse import urlparse, urlunparse
def _is_http_url(s: str) -> bool:
    try:
        p = urlparse(s)
        return p.scheme in ("http", "https") and bool(p.netloc)
    except Exception:
        return False