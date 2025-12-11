"""
Agents
======

An **Agent** is a small, stateful unit of software that talks to a Large Language
Model (LLM) using a chosen persona (a system role-prompt). It accepts a single
**input mapping** (a dict-like object), converts that mapping into a plain
**prompt string** using a **pre-invoke Tool**, and then asks the LLM for a response.

Because LLMs are probabilistic, Agent outputs are **non-deterministic**.

Why schema-first (dict inputs)?
-------------------------------
- A single input shape (`Mapping[str, Any]`) is predictable to call and compose.
- Pre-invoke Tools can adapt *richer schemas and types* (e.g., `{topic, style, audience}`) into
  the final prompt **without** changing the Agent’s method signature.
- Workflows and ToolAdapters can treat Agents as boxes that accept `Mapping[str, Any]`
  and return arbitrary Python objects, while only the pre-invoke Tool and LLMEngine
  know about the concrete prompt structure or model protocols.

Design overview
---------------
- Agents are thin coordination layers around:
  - an `LLMEngine` instance (how to call the model),
  - a `pre_invoke` Tool (how to build the prompt),
  - optional role prompt / persona,
  - stored conversation history (list of messages),
  - external attachments (delegated to the engine).

- Agents do **not** own their own concurrency or dedicated threads; they are
  synchronous call units.

- History is stored as a flat list of `{"role": str, "content": str}` dicts.

How `invoke` works (high-level)
-------------------------------
Given `agent.invoke(inputs)`:
1) **Input validation**: `inputs` must be a `Mapping`; otherwise we raise.
2) **Pre-invoke Tool** turns the mapping into a **prompt string**.
   - The default Tool is *strict* and only accepts `{"prompt": str}`.
   - Provide your own Tool to accept richer keys.
3) **Message assembly**:
   - Optional system persona (`role_prompt`) as the first message.
   - If `context_enabled`, we include the **last N turns** from history, where each
     *turn* is a user->assistant pair. Stored history itself is **never trimmed**.
   - Finally, we append the **user prompt** for the current call.
4) **Engine call**: we invoke the configured `LLMEngine` with the messages and, if any,
   local **attachments** (file paths). A small compatibility shim allows engines that
   expect `(messages, file_paths=...)`, `(messages, files=...)`, or just `(messages)`.
5) **Result**: we expect the engine to return a **str**. If `context_enabled` is on,
   we append this turn (user prompt, assistant reply) to the stored history and return
   the text (non-deterministic by nature).

History policy
--------------
- **Stored history is never trimmed.**
- `history_window` is a **send window measured in turns** (pairs of user/assistant
  messages) that limits what is **sent** to the engine, not what is stored.
- Set `history_window=0` to send no previous turns.
- Set `context_enabled=False` to both **avoid sending** previous turns and **avoid
  recording** new turns.

Attachments
-----------
- Use `attach(path)` and `detach(path)` to manage a simple list of file paths.
- The Agent does **not** check for file existence by default.
- Engines decide how (or whether) to use attachments.

Common mistakes
---------------
- Passing a raw string to `invoke` → **error**. Always pass a mapping, e.g.
  `{"prompt": "Summarize this."}`.
- Passing extra keys with the default pre-invoke Tool → **error**. Install your own
  Tool that accepts your keys and returns a prompt string.
- Expecting deterministic outputs from the Agent/LLM → results will vary.

Quickstart
----------
>>> from LLMEngines import OpenAIEngine
>>> from Tools import Tool
>>> from Agents import Agent
...
>>> def to_prompt(topic: str, style: str) -> str:
...     return f"Write about '{topic}' in a {style} style."
...
>>> prompt_tool = Tool(func=to_prompt, name="to_prompt", description="topic/style -> prompt")
>>> agent.pre_invoke = prompt_tool
>>> agent.invoke({"topic": "unit testing", "style": "pragmatic"})
'...'

Thread-safety
-------------
This class is **not thread-safe**. It mutates internal history and attachments.
Create one Agent per concurrent lane, or protect it with external synchronization.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Callable, Union
import logging
from collections import OrderedDict

# Local imports (adjust the module paths if your project structure differs)
from .LLMEngines import *
from .Tools import *
from .Exceptions import *
from .Primitives import Agent

__all__ = ["Agent"]


logger = logging.getLogger(__name__)