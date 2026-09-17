"""
Example: response_schema for BasicAgent

Demonstrates BasicAgent's construction-time response_schema knob, which
requests provider-native, schema-constrained structured output for every
reply this agent produces. response_schema is frozen at construction, like
every other BasicAgent config knob -- there is no supported way to change
it after the fact.

Three cases:
  1) An object schema (the common case) -- a structured verdict consumed
     by post_invoke.
  2) A bare-primitive root schema (no wrapping object) against Anthropic --
     not every provider supports this. Anthropic, Gemini, and Mistral do;
     OpenAI does not.
  3) The same bare-primitive schema against OpenAI, deliberately left
     unhandled beyond a try/except -- AA performs no schema validation of
     its own (see LLMEngine's own "Structured-output request contract"),
     so an unsupported schema fails as that provider's own native error,
     surfaced through AA unmodified.
"""
from __future__ import annotations

from dotenv import load_dotenv

from atomic_agentic.agents import BasicAgent
from atomic_agentic.llm import AnthropicEngine, OpenAIEngine
from atomic_agentic.exceptions import LLMEngineError

load_dotenv()


def record_verdict(*, result: dict) -> dict:
    """post_invoke hook -- result here is already the structured dict
    response_schema requested, not raw text."""
    return {"received": result}


def main() -> None:
    print("=== 1) Object schema (the common case) ===")
    agent = BasicAgent(
        name="summarizer",
        namespace="examples",
        description="Summarizes text into a structured verdict.",
        llm_engine=AnthropicEngine("claude-sonnet-4-5-20250929"),
        role_prompt="You are a concise technical reviewer.",
        response_schema={
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "confidence": {"type": "number"},
            },
            "required": ["summary", "confidence"],
            "additionalProperties": False,
        },
        post_invoke=record_verdict,
    )
    result = agent.invoke(
        {"prompt": "Summarize in one sentence: Python is a dynamically typed language."}
    )
    print("result.result:", result.result)
    print("type:", type(result.result).__name__)

    print("\n=== 2) Bare primitive schema, Anthropic (supported) ===")
    bare_agent = BasicAgent(
        name="bare_int_agent",
        namespace="examples",
        description="Replies with a single integer, nothing else.",
        llm_engine=AnthropicEngine("claude-sonnet-4-5-20250929"),
        role_prompt="You are terse. Answer only with the requested number.",
        response_schema={"type": "integer"},
    )
    bare_result = bare_agent.invoke({"prompt": "How many continents are there?"})
    print("result.result:", bare_result.result)
    print("type:", type(bare_result.result).__name__)

    print("\n=== 3) Same bare primitive schema, OpenAI (unsupported) ===")
    print("AA performs no schema validation of its own -- an unsupported")
    print("schema fails as the provider's own native error, not an AA one.")
    openai_agent = BasicAgent(
        name="bare_int_agent_openai",
        namespace="examples",
        description="Replies with a single integer, nothing else.",
        llm_engine=OpenAIEngine("gpt-4o-mini"),
        role_prompt="You are terse. Answer only with the requested number.",
        response_schema={"type": "integer"},
    )
    try:
        openai_agent.invoke({"prompt": "How many continents are there?"})
    except LLMEngineError as e:
        print("FAILED as expected:", type(e).__name__, "-", str(e)[:200])


if __name__ == "__main__":
    main()
