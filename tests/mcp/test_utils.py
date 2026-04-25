from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

from atomic_agentic.core.Parameters import ParamSpec
from atomic_agentic.core.sentinels import NO_VAL
from atomic_agentic.mcp.utils import (
    _build_mcp_tool_metadata,
    _infer_mcp_extraction_mode,
    _infer_mcp_return_type,
    _json_schema_type_to_str,
    _normalize_mcp_call_result,
    _plain_mcp_value,
    _run_coro_sync,
)


class DumpableObject:
    def model_dump(self, mode: str = "python") -> dict[str, Any]:
        return {"mode": mode, "value": 123}


class TestRunCoroSync:
    def test_run_coro_sync_returns_result_without_running_loop(self) -> None:
        async def sample() -> int:
            return 42

        assert _run_coro_sync(sample()) == 42

    def test_run_coro_sync_works_when_event_loop_is_already_running(self) -> None:
        async def outer() -> str:
            async def inner() -> str:
                return "ok"

            return _run_coro_sync(inner())

        assert asyncio.run(outer()) == "ok"


class TestJsonSchemaTypeToStr:
    def test_non_mapping_schema_returns_any(self) -> None:
        assert _json_schema_type_to_str("not-a-schema") == "Any"  # type: ignore[arg-type]

    def test_primitive_schema_types(self) -> None:
        assert _json_schema_type_to_str({"type": "string"}) == "str"
        assert _json_schema_type_to_str({"type": "integer"}) == "int"
        assert _json_schema_type_to_str({"type": "number"}) == "float"
        assert _json_schema_type_to_str({"type": "boolean"}) == "bool"
        assert _json_schema_type_to_str({"type": "object"}) == "Dict[str, Any]"
        assert _json_schema_type_to_str({"type": "null"}) == "None"

    def test_array_schema_type_infers_inner_type(self) -> None:
        assert _json_schema_type_to_str(
            {"type": "array", "items": {"type": "string"}}
        ) == "List[str]"

    def test_union_schema_type_deduplicates_parts(self) -> None:
        assert _json_schema_type_to_str(
            {"type": ["string", "null", "string"]}
        ) == "str | None"

    def test_unknown_schema_type_returns_any(self) -> None:
        assert _json_schema_type_to_str({"type": "unknown"}) == "Any"


class TestPlainMcpValue:
    def test_plain_scalars_return_unchanged(self) -> None:
        assert _plain_mcp_value(None) is None
        assert _plain_mcp_value("x") == "x"
        assert _plain_mcp_value(1) == 1
        assert _plain_mcp_value(1.5) == 1.5
        assert _plain_mcp_value(True) is True

    def test_plain_nested_mapping_and_sequence(self) -> None:
        value = {"a": [1, {"b": 2}], "c": ("x", "y")}

        assert _plain_mcp_value(value) == {
            "a": [1, {"b": 2}],
            "c": ("x", "y"),
        }

    def test_model_dump_object_is_converted(self) -> None:
        assert _plain_mcp_value(DumpableObject()) == {
            "mode": "python",
            "value": 123,
        }

    def test_dunder_dict_object_is_converted_without_private_attrs(self) -> None:
        obj = SimpleNamespace(public=1, _private=2)

        assert _plain_mcp_value(obj) == {"public": 1}


class TestMcpExtractionAndReturnType:
    def test_non_mapping_output_schema_uses_content_blocks(self) -> None:
        assert _infer_mcp_extraction_mode(None) == "content_blocks"
        assert _infer_mcp_return_type(None) == "List[ContentBlock]"

    def test_single_result_property_extracts_result(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "result": {"type": "integer"},
            },
        }

        assert _infer_mcp_extraction_mode(schema) == "extract_result"
        assert _infer_mcp_return_type(schema) == "int"

    def test_object_schema_uses_structured_content_dict(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "score": {"type": "number"},
            },
        }

        assert _infer_mcp_extraction_mode(schema) == "structured_content"
        assert _infer_mcp_return_type(schema) == "Dict[str, Any]"

    def test_non_object_schema_returns_inferred_type(self) -> None:
        schema = {"type": "array", "items": {"type": "boolean"}}

        assert _infer_mcp_extraction_mode(schema) == "structured_content"
        assert _infer_mcp_return_type(schema) == "List[bool]"


class TestBuildMcpToolMetadata:
    def test_builds_metadata_from_required_and_optional_schema_fields(self) -> None:
        raw_tool = SimpleNamespace(
            name="search",
            description="Search documents.",
            inputSchema={
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 5},
                    "debug": {"type": "boolean"},
                },
            },
            outputSchema={
                "type": "object",
                "properties": {
                    "result": {"type": "string"},
                },
            },
            annotations={"safe": True},
            _meta={"source": "unit-test"},
            title="Search",
        )

        metadata = _build_mcp_tool_metadata(raw_tool)

        assert metadata["description"] == "Search documents."
        assert metadata["return_type"] == "str"
        assert metadata["extraction_mode"] == "extract_result"
        assert metadata["raw_metadata"]["name"] == "search"
        assert metadata["raw_metadata"]["annotations"] == {"safe": True}
        assert metadata["raw_metadata"]["_meta"] == {"source": "unit-test"}
        assert metadata["raw_metadata"]["title"] == "Search"

        params = metadata["parameters"]
        assert [(p.name, p.kind, p.type, p.default) for p in params] == [
            ("query", ParamSpec.KEYWORD_ONLY, "str", NO_VAL),
            ("top_k", ParamSpec.KEYWORD_ONLY, "int", 5),
            ("debug", ParamSpec.KEYWORD_ONLY, "bool", None),
        ]


class TestNormalizeMcpCallResult:
    def test_normalizes_call_tool_result_envelope(self) -> None:
        raw = SimpleNamespace(
            content=["block-1", "block-2"],
            structuredContent={"answer": 42},
            isError=False,
        )

        assert _normalize_mcp_call_result(raw) == {
            "content": ["block-1", "block-2"],
            "structuredContent": {"answer": 42},
            "isError": False,
        }
