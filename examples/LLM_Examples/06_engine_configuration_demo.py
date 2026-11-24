import os
import tempfile

from dotenv import load_dotenv

from atomic_agentic.exceptions import LLMEngineError
from atomic_agentic.llm import OpenAIEngine

load_dotenv()

# Requires OPENAI_API_KEY in your environment (or .env). The same calls work on
# every engine; attach/detach exist only on the OpenAI, Anthropic, Gemini and
# Mistral engines.
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise SystemExit("Set OPENAI_API_KEY to run this demo.")

engine = OpenAIEngine(model="gpt-4.1-mini", api_key=api_key)

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

# --- Messages contract: at least one user message is required ---
try:
    engine.invoke({"messages": [{"role": "system", "content": "No user turn."}]})
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
