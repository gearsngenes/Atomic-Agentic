from __future__ import annotations

from typing import Any, Callable, Mapping

from atomic_agentic.llm import LLMEngine
from atomic_agentic.models.results import LLMModelData, TokenUsage

__all__ = ["FakeLLMEngine", "echo_latest_user"]


class FakeLLMEngine(LLMEngine):
    """Canonical scripted/dynamic fake LLMEngine for tests.

    Two response modes, `response_fn` taking silent priority when both are
    supplied at construction:

    - Dynamic (`response_fn`): called fresh on every LLM call with that
      call's messages; whatever it returns (or raises) becomes the response.
    - Static/scripted (`responses`): popped one per call; raises
      `RuntimeError("No scripted LLM responses remain.")` on exhaustion.

    Exception injection needs no dedicated mechanism in either mode: a
    `response_fn` that raises propagates naturally; a popped `responses`
    item that is a `BaseException` instance is raised instead of returned.
    """

    def __init__(
        self,
        responses: list[Any] | None = None,
        *,
        response_fn: Callable[[list[dict[str, str]]], Any] | None = None,
        prepare_result: Mapping[str, Any] | Any | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.responses = list(responses) if responses is not None else [" hello "]
        self.response_fn = response_fn
        self.prepare_result = prepare_result
        self.payloads: list[Any] = []
        self.prepare_calls: list[str] = []
        self.detach_calls: list[Mapping[str, Any]] = []
        self.calls: list[list[dict[str, str]]] = []
        self.call_count = 0

    def _build_provider_payload(
        self,
        messages: list[dict[str, str]],
        attachments: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Any]:
        copied = [dict(m) for m in messages]
        self.calls.append(copied)
        payload = {"messages": copied, "attachments": dict(attachments)}
        self.payloads.append(payload)
        return payload

    def _resolve_response(self, messages: list[dict[str, str]]) -> Any:
        """Shared by `_call_provider`/`_call_provider_async` — no real I/O
        happens in either mode, so both call this directly."""
        if self.response_fn is not None:
            result = self.response_fn(messages)
        else:
            if not self.responses:
                raise RuntimeError("No scripted LLM responses remain.")
            result = self.responses.pop(0)

        if isinstance(result, BaseException):
            raise result
        return result

    def _call_provider(self, payload: Any) -> Any:
        self.call_count += 1
        return self._resolve_response(payload["messages"])

    async def _call_provider_async(self, payload: Any) -> Any:
        self.call_count += 1
        return self._resolve_response(payload["messages"])

    def _extract_text(self, response: Any) -> Any:
        # Deliberately Any, not str: some tests script a non-string response
        # to exercise the base class's own "_extract_text must return str"
        # validation.
        return response

    def _extract_token_usage(self, response: Any) -> TokenUsage:
        return TokenUsage(
            input_tokens=1, generated_tokens=1, total_tokens=2, response_tokens=1
        )

    def _get_model_data(self) -> LLMModelData:
        return LLMModelData(provider="fake")

    def _prepare_attachment(self, path: str) -> Mapping[str, Any]:
        self.prepare_calls.append(path)
        if self.prepare_result is not None:
            return self.prepare_result
        return {"path": path, "prepared": True}

    def _on_detach(self, meta: Mapping[str, Any]) -> None:
        self.detach_calls.append(meta)

    def _should_retry(self, exc: Exception, attempt: int) -> bool:
        if attempt > self._max_retries:
            return False
        return isinstance(exc, (TimeoutError, ConnectionError))


def echo_latest_user(prefix: str = "ECHO") -> Callable[[list[dict[str, str]]], str]:
    """Factory producing a `response_fn`-shaped closure that echoes the
    latest user-role message content with the given prefix."""

    def _fn(messages: list[dict[str, str]]) -> str:
        latest_user = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
        )
        return f"{prefix}: {latest_user}"

    return _fn
