"""Core-only behavior: every optional dependency is blocked in a child process.

Each test runs a fresh interpreter so import state is clean. The prelude hides
the provider SDKs and python-a2a from both the import system and
``importlib.metadata``, mimicking an install with no extras.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"

PRELUDE = '''
import importlib.abc
import importlib.metadata as md
import sys

_BLOCKED = ("python_a2a", "openai", "anthropic", "google.genai", "mistralai",
            "litellm", "llama_cpp", "huggingface_hub")
_DISTS = {"python-a2a", "openai", "anthropic", "google-genai", "mistralai",
          "litellm", "llama-cpp-python", "huggingface-hub"}

class _Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if any(name == b or name.startswith(b + ".") for b in _BLOCKED):
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)

sys.meta_path.insert(0, _Block())
_real_version = md.version

def _version(package):
    if package in _DISTS:
        raise md.PackageNotFoundError(package)
    return _real_version(package)

md.version = _version
'''


def run_blocked(snippet: str, *extra_args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *extra_args, "-c", PRELUDE + snippet],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(SRC)},
        timeout=120,
    )


def test_all_core_packages_import_without_extras() -> None:
    result = run_blocked(
        "import atomic_agentic, atomic_agentic.llm, atomic_agentic.tools, "
        "atomic_agentic.agents, atomic_agentic.composites, atomic_agentic.mcp, "
        "atomic_agentic.a2a\nprint('ok')"
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_python_a2a_classes_exist_but_raise_install_hint() -> None:
    snippet = '''
import json
from atomic_agentic.a2a import PyA2AtomicClient, PyA2AtomicHost
out = {}
for label, call in (
    ("client", lambda: PyA2AtomicClient("http://x")),
    ("host", lambda: PyA2AtomicHost([], name="n", description="d")),
):
    try:
        call()
        out[label] = "no error"
    except ImportError as exc:
        out[label] = str(exc)
print(json.dumps(out))
'''
    result = run_blocked(snippet)
    assert result.returncode == 0, result.stderr
    messages = json.loads(result.stdout.strip().splitlines()[-1])
    for label in ("client", "host"):
        assert "pip install atomic-agentic[python-a2a]" in messages[label]
        assert "`python-a2a`" in messages[label]


def test_toolify_and_isinstance_are_unaffected_without_python_a2a() -> None:
    snippet = '''
from atomic_agentic.a2a import PyA2AtomicClient
from atomic_agentic.tools import toolify, batch_toolify

def add(a: int, b: int) -> int:
    return a + b

tool = toolify(add)
tools = batch_toolify([add])
print(tool.name, len(tools), isinstance(object(), PyA2AtomicClient))
'''
    result = run_blocked(snippet)
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["add", "1", "False"]


def test_each_engine_reports_its_sdk_and_extra() -> None:
    snippet = '''
import json
from atomic_agentic.exceptions import LLMEngineError
from atomic_agentic.llm import (AnthropicEngine, GeminiEngine, LiteLLMEngine,
                                LlamaCppEngine, MistralEngine, OpenAIEngine)
cases = {
    "openai": lambda: OpenAIEngine(model="m"),
    "anthropic": lambda: AnthropicEngine(model="m"),
    "gemini": lambda: GeminiEngine(model="m"),
    "mistral": lambda: MistralEngine(model="m"),
    "litellm": lambda: LiteLLMEngine(model="m"),
    "llama": lambda: LlamaCppEngine(model_path="x.gguf"),
}
out = {}
for extra, build in cases.items():
    try:
        build()
        out[extra] = "no error"
    except LLMEngineError as exc:
        out[extra] = str(exc)
print(json.dumps(out))
'''
    result = run_blocked(snippet)
    assert result.returncode == 0, result.stderr
    messages = json.loads(result.stdout.strip().splitlines()[-1])
    dists = {
        "openai": "openai",
        "anthropic": "anthropic",
        "gemini": "google-genai",
        "mistral": "mistralai",
        "litellm": "litellm",
        "llama": "llama-cpp-python",
    }
    for extra, dist in dists.items():
        assert dist in messages[extra]
        assert f"pip install atomic-agentic[{extra}]" in messages[extra]


def test_importing_does_not_emit_the_python_a2a_deprecation() -> None:
    snippet = '''
import warnings
warnings.filterwarnings("error", message=".*python-a2a.*", category=DeprecationWarning)
import atomic_agentic.a2a, atomic_agentic.tools, atomic_agentic.agents
print("ok")
'''
    result = run_blocked(snippet)
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
