# Agent Examples — Engines + Agent (Persona + Optional Memory)

> Scope: This folder demonstrates the **lowest layer (LLM Engines)** and the stateful **Agent** wrapper.

## Scope

* How engines act as **stateless, provider-agnostic adapters** with a single `invoke(messages, file_paths)` contract. 
* How the `Agent` wraps an engine to add a **role/system prompt**, **optional chat memory**, and **file path tracking** before delegating to the engine. 

---

## Quick start

### 1) Chat agent (role prompt + memory)

```bash
python ./chatbot_agent_test.py
```

This script defines an engine (e.g., OpenAI/Gemini/Mistral/llama.cpp), creates an `Agent` with a role prompt and `context_enabled=True`, then runs a chat loop in the terminal.

### 2) File summarization

```bash
python ./file_summarization.py
```

This script demonstrates `attach(path) → invoke(prompt) → detach(path)` for a PDF summary workflow.

> Environment variables: set `OPENAI_API_KEY`, `GOOGLE_API_KEY` (Gemini), `MISTRAL_API_KEY`. See the root README’s setup.

---
## LLM's and Agents

### LLM Engines (Stateless LLM)

* One shape: `invoke(messages, file_paths=None) -> str`. Engines **do not** keep history or file state; the Agent owns those. 
* Per call, engines validate and ingest `file_paths` per provider, upload when necessary, inline text/code when allowed, and **best-effort delete** transient uploads.

### Agent (Stateful, LLM-Driven Objects)

* Adds **persona** (system/role prompt), **optional memory** of prior turns, and a **list of local file paths** to forward to the engine. 
* `invoke(prompt)` builds messages: `[system?] + [history?] + [user]`, calls `engine.invoke(messages, file_paths=self.file_paths)`, then **persists** the turn if memory is enabled. 

---

## Public API

### Agent

```python
Agent(
    name: str,
    description: str,
    llm_engine,                   # any LLMEngine subclass
    role_prompt: str = "",
    context_enabled: bool = False
)
```

**Properties:** `name`, `description`, `role_prompt`, `context_enabled`, `llm_engine`, `history (copy)`, `file_paths (copy)` 

**Methods:**

* `attach(path) -> bool` / `detach(path) -> bool` — mutate the agent’s tracked file path list only. 
* `invoke(prompt) -> str` — build messages, forward `file_paths`, persist turns if `context_enabled=True`. 
* `clear_memory()` — wipe conversation history; attachments unchanged.

### LLM Engine (abstract contract)

```python
class LLMEngine:
    def invoke(self, messages: list[dict], file_paths: list[str] | None = None) -> str:
        ...
```

**Responsibilities:** validate files, upload/inline per provider, stay stateless (delete temporary uploads), return assistant text.

---

## Provider notes (high level)

* **OpenAIEngine** — Use **Responses API**; PDFs → `input_file`, images → `input_image`, text/code → `input_text` (with cutoff). Use a vision-capable model when sending PDFs/images. All uploads created by the call are cleaned up. 
* **GeminiEngine** — Upload everything supported (PDF, image, text/code, Office, CSV/XLSX) via `client.files.upload(...)`; pass files + text in `contents`; best-effort delete after call. 
* **MistralEngine** — Upload → sign → attach as `document_url`/`image_url`; inline text/code with cutoff; cleanup uploads. 
* **LlamaCppEngine** — Local model; ignores `file_paths` (no remote upload store).

**Illegal types (guideline):** treat archives/executables/db/model weights and `audio/*`, `video/*` as unsupported; engines should raise `TypeError`. 

---

## Behavioral guarantees & gotchas

* **Provider-agnostic I/O:** Agents don’t change when you swap engines—same `invoke` surface.
* **State lives in the Agent:** history + file path list; engines remain stateless (per-call cleanup). 
* **`attach()` doesn’t validate:** real checks happen at engine time; expect `FileNotFoundError` or `TypeError` there. 
* **Inline cutoffs:** engines that inline text/code apply a global character limit; large files may be truncated.

---

## Recipes

**Swap providers (one line):**

### Swap providers — complete recipes

Below are **drop-in** engine initializations. Each snippet also shows a minimal `Agent` setup and a single call. Pick one, copy-paste, and it will work with the same `Agent` API.

> Assumes your project structure exposes these:
>
> ```py
> from modules.LLMEngines import OpenAIEngine, GeminiEngine, MistralEngine, LlamaCppEngine
> from modules.Agents import Agent
> ```

---

#### OpenAI (Responses API)

```python
from modules.LLMEngines import OpenAIEngine
from modules.Agents import Agent

# Requires: OPENAI_API_KEY in environment (dotenv supported by the repo)
llm = OpenAIEngine(
    model="gpt-4o-mini",          # any chat/vision-capable OpenAI model
    # api_key="...",             # optional: override env
    temperature=0.1,
    inline_cutoff_chars=200_000,  # max chars to inline from local text/code files
)

agent = Agent(
    name="openai-agent",
    description="Chat agent using OpenAI Responses API",
    llm_engine=llm,
    role_prompt="You are concise and technical.",
    context_enabled=True,         # remember prior turns (agent-side state)
)

print(agent.invoke("Give me a 2-line explanation of attention in transformers."))
```

---

#### Google Gemini

```python
from modules.LLMEngines import GeminiEngine
from modules.Agents import Agent

# Requires: GOOGLE_API_KEY in environment
llm = GeminiEngine(
    model="gemini-1.5-flash",     # any text/vision-capable Gemini model
    # api_key="...",             # optional: override env
    temperature=0.1,
)

agent = Agent(
    name="gemini-agent",
    description="Chat agent using Gemini (Files API + flat contents)",
    llm_engine=llm,
    role_prompt="You are concise and technical.",
    context_enabled=True,
)

print(agent.invoke("List 3 practical tips for prompt engineering."))
```

---

#### Mistral AI

```python
from modules.LLMEngines import MistralEngine
from modules.Agents import Agent

# Requires: MISTRAL_API_KEY in environment
llm = MistralEngine(
    model="mistral-medium-latest",    # or mistral-large-latest, etc.
    # api_key="...",                 # optional: override env
    temperature=0.1,
    inline_cutoff_chars=200_000,      # for local text/code inlining
    retry_sign_attempts=5,            # signed-URL retries for fresh uploads
    retry_base_delay=0.3,
)

agent = Agent(
    name="mistral-agent",
    description="Chat agent using Mistral (upload → sign → parts)",
    llm_engine=llm,
    role_prompt="Be precise and provide runnable snippets when possible.",
    context_enabled=True,
)

print(agent.invoke("What are two pros and two cons of RAG vs fine-tuning?"))
```

---

#### Llama.cpp (local, no remote uploads)

**Option A — local .gguf path**

```python
from modules.LLMEngines import LlamaCppEngine
from modules.Agents import Agent

llm = LlamaCppEngine(
    model_path="/path/to/model.gguf",  # local file path
    n_ctx=4096,
    verbose=False,
)

agent = Agent(
    name="llamacpp-agent",
    description="Local llama.cpp model",
    llm_engine=llm,
    role_prompt="Answer tersely; cite assumptions.",
    context_enabled=True,
)

print(agent.invoke("Summarize the benefits of quantization for edge inference."))
```

**Option B — pull from Hugging Face repo**

```python
from modules.LLMEngines import LlamaCppEngine
from modules.Agents import Agent

llm = LlamaCppEngine(
    repo_id="TheBloke/phi-3-mini-4k-instruct-GGUF",  # example
    filename="phi-3-mini-4k-instruct.Q4_K_M.gguf",   # pick a file that exists in the repo
    n_ctx=4096,
    verbose=False,
)

agent = Agent(
    name="llamacpp-agent",
    description="Local llama.cpp model from HF repo",
    llm_engine=llm,
    role_prompt="Keep answers under 100 words.",
    context_enabled=False,  # single-turn mode
)

print(agent.invoke("Name three safe prompt-guard patterns for LLM apps."))
```


**Toggle memory:**

```python
agent.context_enabled = True  # or False
```

**File flow:**

```python
agent.attach("path/to.pdf")
out = agent.invoke("Summarize the document.")
agent.detach("path/to.pdf")
```

(From the summarization example.) 

---

## Error semantics

Typical exceptions from engines:

* `FileNotFoundError` — missing local path.
* `TypeError` — illegal/unsupported file type.
* Provider SDK exceptions during upload/sign/delete.

See each concrete engine for exact behavior.

---

## Extending: add a new engine

Implement a subclass of `LLMEngine` that honors the contract above: accept `messages` + `file_paths`, validate and ingest files for your provider, **delete temporary uploads**, and return text. Keep your public call surface consistent for that provider. 

---

## See also

Top-level README for folder layout and environment setup.
