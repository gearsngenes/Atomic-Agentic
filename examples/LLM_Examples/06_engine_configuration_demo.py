import os
import tempfile

from dotenv import load_dotenv

from atomic_agentic.exceptions import LLMEngineError
from openai import OpenAI

from atomic_agentic.adapters import to_llm
from atomic_agentic.llm import OpenAIEngine

load_dotenv()

# Requires OPENAI_API_KEY in your environment (or .env). This demo uses
# OpenAIEngine; runtime-parameter names are each provider's own, so another
# engine takes its own names. `strict` exists on OpenAIEngine and MistralEngine
# only; attach/detach exist on the OpenAI, Anthropic, Gemini and Mistral
# engines only. `update_runtime_parameters` and `require_usage` are on every
# engine (Anthropic always reports usage).
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise SystemExit("Set OPENAI_API_KEY to run this demo.")

engine = OpenAIEngine(model="gpt-4.1-mini", api_key=api_key)

# --- to_llm: build an engine from a pre-built SDK client (no network call) ---
# The engine class follows the client's SDK type; `model` is required except
# for a llama_cpp.Llama client, which already carries its model path.
built = to_llm(OpenAI(api_key="dummy-key"), "gpt-4.1-mini")
print("to_llm built:", type(built).__name__, built.name, built.model_data)

# --- Runtime parameters: the provider's own call parameters ---
# None removes a key; `stream` is reserved because streaming is unsupported.
engine.update_runtime_parameters(temperature=0.2, max_output_tokens=200)
print("runtime parameters:", engine.runtime_parameters)
engine.update_runtime_parameters(temperature=None)
print("after removing temperature:", engine.runtime_parameters)
try:
    engine.update_runtime_parameters(stream=True)
except LLMEngineError as exc:
    print("rejected:", exc)

# --- Messages contract ---
# A non-empty list; roles system/user/assistant; at least one user message;
# non-blank content, except a blank user message while attachments exist.
try:
    engine.invoke({"messages": [{"role": "system", "content": "No user turn."}]})
except LLMEngineError as exc:
    print("rejected:", exc)
try:
    engine.invoke({"messages": [{"role": "user", "content": "  "}]})
except LLMEngineError as exc:
    print("rejected:", exc)

# --- strict: structured-output conformance (OpenAI and Mistral) ---
schema = {
    "type": "object",
    "properties": {"capital": {"type": "string"}},
    "required": ["capital"],
    "additionalProperties": False,
}
engine.strict = True
result = engine.invoke({
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "output_structure": schema,
})
print("structured result:", result.result)

# --- require_usage: tolerate providers that omit usage ---
# With False, a response that has no usage object reports all-zero token usage
# instead of raising. This response does report usage, so the counts are real.
engine.require_usage = False
result = engine.invoke({"messages": [{"role": "user", "content": "Say hi."}]})
print("token usage:", result.token_usage)

# --- attach: a file rides along on every later call ---
with tempfile.TemporaryDirectory() as folder:
    notes = os.path.join(folder, "notes.txt")
    with open(notes, "w", encoding="utf-8") as fh:
        fh.write("The secret word is 'pelican'.")

    engine.attach(notes)
    print("attachments:", list(engine.attachments))
    result = engine.invoke({
        "messages": [{"role": "user", "content": "What is the secret word in the attached file?"}]
    })
    print("answer:", result.result)
    engine.detach(notes)
