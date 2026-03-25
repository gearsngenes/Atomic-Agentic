from __future__ import annotations

import asyncio
import threading
from typing import (
    Any,
    Awaitable,
    Dict,
    Mapping,
    Literal,
    TypeVar,
)
from mcp import types as mcp_types

from ..core.Parameters import ParamSpec
from ..core.sentinels import NO_VAL

MCPExtractionMode = Literal["extract_result", "structured_content", "content_blocks"]

T = TypeVar("T")

def _run_coro_sync(coro: Awaitable[T]) -> T:
    """
    Run an async coroutine from sync code, even if a loop is already running
    in the current thread.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result_box: list[T] = []
    error_box: list[BaseException] = []

    def runner() -> None:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            result_box.append(loop.run_until_complete(coro))
        except BaseException as exc:  # noqa: BLE001
            error_box.append(exc)
        finally:
            loop.close()

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()

    if error_box:
        raise error_box[0]
    if not result_box:
        raise RuntimeError("Coroutine completed without producing a result.")

    return result_box[0]


def _json_schema_type_to_str(schema: Mapping[str, Any]) -> str:
    """
    Best-effort conversion from a JSON Schema fragment to a Python-ish type string.
    """
    if not isinstance(schema, Mapping):
        return "Any"

    primitive_map: Dict[str, str] = {
        "string": "str",
        "number": "float",
        "integer": "int",
        "boolean": "bool",
        "object": "Dict[str, Any]",
        "array": "List[Any]",
        "null": "None",
    }

    raw_type = schema.get("type")

    if isinstance(raw_type, str):
        if raw_type == "array":
            items = schema.get("items")
            inner = _json_schema_type_to_str(items) if isinstance(items, Mapping) else "Any"
            return f"List[{inner}]"
        return primitive_map.get(raw_type, "Any")

    if isinstance(raw_type, (list, tuple)):
        parts: list[str] = []
        for item in raw_type:
            if not isinstance(item, str):
                continue
            if item == "array":
                items = schema.get("items")
                inner = _json_schema_type_to_str(items) if isinstance(items, Mapping) else "Any"
                parts.append(f"List[{inner}]")
            else:
                parts.append(primitive_map.get(item, "Any"))
        return " | ".join(dict.fromkeys(parts)) if parts else "Any"

    return "Any"


def _plain_mcp_value(value: Any) -> Any:
    """
    Convert MCP SDK objects into plain Python values for serialization-friendly
    metadata snapshots.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, Mapping):
        return {str(key): _plain_mcp_value(item) for key, item in value.items()}

    if isinstance(value, list):
        return [_plain_mcp_value(item) for item in value]

    if isinstance(value, tuple):
        return tuple(_plain_mcp_value(item) for item in value)

    if isinstance(value, set):
        return {_plain_mcp_value(item) for item in value}

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump(mode="python")
        except TypeError:
            dumped = model_dump()
        return _plain_mcp_value(dumped)

    if hasattr(value, "__dict__"):
        return {
            str(key): _plain_mcp_value(item)
            for key, item in vars(value).items()
            if not str(key).startswith("_")
        }

    return value


def _infer_mcp_extraction_mode(output_schema: Any) -> MCPExtractionMode:
    """
    Decide how the proxy should extract the final AA-facing value.

    Rules
    -----
    - Missing / non-mapping output schema -> raw content blocks
    - Singleton object-like schema with one "result" property -> unwrap "result"
    - Any other mapping schema -> return structuredContent as-is
    """
    if not isinstance(output_schema, Mapping):
        return "content_blocks"

    properties = output_schema.get("properties")
    if isinstance(properties, Mapping) and len(properties) == 1 and "result" in properties:
        return "extract_result"

    return "structured_content"


def _infer_mcp_return_type(output_schema: Any) -> str:
    """
    Infer the AA-facing return type string from an MCP output schema.

    The return type matches the final proxy behavior:
    - content_blocks     -> List[ContentBlock]
    - extract_result     -> inferred type of the nested "result" schema
    - structured_content -> concrete inferred non-object type when possible,
                            otherwise Dict[str, Any]
    """
    extraction_mode = _infer_mcp_extraction_mode(output_schema)

    if extraction_mode == "content_blocks":
        return "List[ContentBlock]"

    if not isinstance(output_schema, Mapping):
        return "List[ContentBlock]"

    if extraction_mode == "extract_result":
        properties = output_schema.get("properties")
        if isinstance(properties, Mapping):
            result_schema = properties.get("result")
            if isinstance(result_schema, Mapping):
                return _json_schema_type_to_str(result_schema)
        return "Any"

    output_type = output_schema.get("type")

    if isinstance(output_type, str):
        if output_type == "object":
            return "Dict[str, Any]"
        return _json_schema_type_to_str(output_schema)

    if isinstance(output_type, (list, tuple)):
        if "object" in output_type:
            return "Dict[str, Any]"
        return _json_schema_type_to_str(output_schema)

    properties = output_schema.get("properties")
    if isinstance(properties, Mapping):
        return "Dict[str, Any]"

    return _json_schema_type_to_str(output_schema)


def _build_mcp_tool_metadata(raw_tool: mcp_types.Tool) -> Dict[str, Any]:
    """
    Convert one raw MCP Tool object into the client hub's processed metadata shape.
    """
    input_schema = raw_tool.inputSchema if isinstance(raw_tool.inputSchema, Mapping) else {}
    output_schema = getattr(raw_tool, "outputSchema", None)

    properties = input_schema.get("properties")
    required_raw = input_schema.get("required")
    required = (
        {str(item) for item in required_raw}
        if isinstance(required_raw, (list, tuple, set))
        else set()
    )

    parameters: list[ParamSpec] = []
    if isinstance(properties, Mapping):
        for index, (raw_name, raw_meta) in enumerate(properties.items()):
            name = str(raw_name)
            meta_schema = raw_meta if isinstance(raw_meta, Mapping) else {}

            # Required params default to NO_VAL. Optional params use the schema
            # default if provided, otherwise None.
            if "default" in meta_schema:
                default = meta_schema.get("default")
            elif name in required:
                default = NO_VAL
            else:
                default = None

            parameters.append(
                ParamSpec(
                    name=name,
                    index=index,
                    kind="KEYWORD_ONLY",
                    type=_json_schema_type_to_str(meta_schema),
                    default=default,
                )
            )

    extraction_mode = _infer_mcp_extraction_mode(output_schema)

    raw_metadata: Dict[str, Any] = {
        "name": raw_tool.name,
        "description": raw_tool.description,
        "inputSchema": _plain_mcp_value(raw_tool.inputSchema),
        "outputSchema": _plain_mcp_value(output_schema),
    }

    annotations = getattr(raw_tool, "annotations", None)
    if annotations is not None:
        raw_metadata["annotations"] = _plain_mcp_value(annotations)

    meta = getattr(raw_tool, "_meta", None)
    if meta is not None:
        raw_metadata["_meta"] = _plain_mcp_value(meta)

    title = getattr(raw_tool, "title", None)
    if title is not None:
        raw_metadata["title"] = _plain_mcp_value(title)

    return {
        "description": str(raw_tool.description or ""),
        "parameters": parameters,
        "return_type": _infer_mcp_return_type(output_schema),
        "extraction_mode": extraction_mode,
        "raw_metadata": raw_metadata,
    }


def _normalize_mcp_call_result(raw: mcp_types.CallToolResult) -> Dict[str, Any]:
    """
    Normalize a raw MCP CallToolResult into the stable hub envelope.

    Notes
    -----
    - `content` is preserved as raw MCP content blocks.
    - `structuredContent` is preserved exactly as returned by the SDK.
    - `isError` is preserved exactly as returned by the SDK.
    """
    return {
        "content": list(raw.content),
        "structuredContent": raw.structuredContent,
        "isError": raw.isError,
    }
