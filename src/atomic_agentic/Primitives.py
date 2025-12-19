from __future__ import annotations

import logging
import os
import random
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Mapping, Optional, Union, get_origin, get_args
from .Exceptions import *
from typing import Mapping, Callable, Optional
from collections import OrderedDict
import inspect
import threading
from collections.abc import Mapping as ABCMapping, Sequence as ABCSequence
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4


logger = logging.getLogger(__name__)

__all__ = [
    "LLMEngine",
    "Tool",
    "Agent",
    "Workflow",
    "WorkflowCheckpoint",
    "BundlingPolicy",
    "MappingPolicy",
    "NO_VAL",
    "DEF_RES_KEY"
]


# ───────────────────────────────────────────────────────────────────────────────
# LLMEngine primitive
# ───────────────────────────────────────────────────────────────────────────────
class LLMEngine(ABC):
    """
    Base template-method primitive for LLM provider adapters.

    Engines are stateless with respect to conversation history: the Agent owns
    the message history. An engine instance represents a particular provider +
    model configuration plus a persistent set of attachments.

    Public contract
    ---------------
    - `invoke(messages: list[{"role": str, "content": str}]) -> str`
      is the *only* required public entrypoint for making a call.
    - Attachments are managed via `attach` / `detach` / `clear_attachments`.
    """

    # Attachment policy defaults
    # --------------------------
    # Subclasses are expected to override `allowed_attachment_exts` with the set
    # of extensions their provider can meaningfully consume (e.g. {".pdf", ".png"}).
    # `illegal_attachment_exts` is a coarse security/robustness guard applied
    # before provider-specific checks.
    illegal_attachment_exts: set[str] = {
        ".zip", ".tar", ".gz", ".tgz", ".rar", ".7z",
        ".exe", ".dll", ".so", ".bin", ".o",
        ".db", ".sqlite",
        ".h5", ".pt", ".pth", ".onnx",
    }
    allowed_attachment_exts: Optional[set[str]] = None

    def __init__(
        self,
        *,
        name: Optional[str] = None,
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
        retry_backoff_base: float = 0.5,
        retry_backoff_max: float = 8.0,
    ) -> None:
        """
        Parameters
        ----------
        name:
            Optional human-friendly identifier for logging/introspection.
        timeout_seconds:
            Suggested per-call timeout; subclasses should honor this where
            their provider SDKs allow it.
        max_retries:
            Maximum number of *retries* after the initial call (so total
            attempts is `max_retries + 1`).
        retry_backoff_base:
            Base seconds for exponential backoff (approx base * 2^(attempt-1)).
        retry_backoff_max:
            Upper bound in seconds for backoff delay.
        """
        self._name = name or type(self).__name__
        self._timeout_seconds = float(timeout_seconds)
        self._max_retries = int(max_retries)
        self._retry_backoff_base = float(retry_backoff_base)
        self._retry_backoff_max = float(retry_backoff_max)
        # Persistent mapping: local path -> provider-specific metadata
        self._attachments: Dict[str, Mapping[str, Any]] = {}

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #
    @property
    def name(self) -> str:
        """Human-friendly identifier for this engine instance."""
        return self._name

    @property
    def attachments(self) -> Mapping[str, Mapping[str, Any]]:
        """
        Read-only view of currently attached paths.

        Keys are local paths; values are provider-specific metadata dicts as
        returned by `_prepare_attachment`.
        """
        # Return a shallow copy to discourage mutation of internal state.
        return dict(self._attachments)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def attach(self, path: str) -> Mapping[str, Any]:
        """
        Attach a local file path to this engine.

        The base implementation:
        - validates the path and extension using `_validate_attachment_path`;
        - delegates to `_prepare_attachment` to build provider-specific metadata;
        - stores and returns that metadata.

        The *shape* of the metadata mapping is entirely determined by the subclass.
        """
        if not isinstance(path, str) or not path:
            raise LLMEngineError("LLMEngine.attach: path must be a non-empty string")
        if path in self._attachments:
            return self._attachments[path]

        self._validate_attachment_path(path)
        meta = self._prepare_attachment(path)
        if not isinstance(meta, Mapping):
            raise LLMEngineError(
                f"{type(self).__name__}._prepare_attachment must return a mapping; "
                f"got {type(meta)!r}"
            )

        self._attachments[path] = meta
        logger.debug("LLMEngine %s attached %s", self._name, path)
        return meta

    def detach(self, path: str) -> bool:
        """
        Detach a previously attached path.

        Calls `_on_detach` with the stored metadata for provider-specific cleanup.
        Returns True if an attachment was removed, False if the path was not
        attached.
        """
        meta = self._attachments.pop(path, None)
        if meta is None:
            return False
        try:
            self._on_detach(meta)
        except Exception as exc:  # pragma: no cover - best-effort cleanup
            logger.debug(
                "LLMEngine %s._on_detach raised %r for %s; ignoring",
                self._name,
                exc,
                path,
            )
        logger.debug("LLMEngine %s detached %s", self._name, path)
        return True

    def clear_attachments(self) -> None:
        """Detach all currently attached paths."""
        for path in list(self._attachments.keys()):
            self.detach(path)

    def invoke(self, messages: List[Dict[str, str]]) -> str:
        """
        Template method that defines the engine invocation lifecycle.

        Steps:
        1. Normalize and validate the input `messages`.
        2. Snapshot current attachments.
        3. Ask the subclass to build a provider-specific payload.
        4. Call the provider with retries/timeouts.
        5. Extract and normalize the assistant text.

        Subclasses **must not** override this method; they customize behavior
        via the protected hooks documented below.
        """
        try:
            normalized = self._normalize_messages(messages)
            attachments = dict(self._attachments)
            payload = self._build_provider_payload(normalized, attachments)
            response = self._call_with_retries(payload)
            text = self._extract_text(response)

            if not isinstance(text, str):
                raise LLMEngineError(
                    f"{type(self).__name__}._extract_text must return str; "
                    f"got {type(text)!r}"
                )
            return text.strip()
        except LLMEngineError:
            # Already normalized; bubble up unchanged.
            raise
        except Exception as exc:
            raise LLMEngineError(f"{self._name}.invoke failed") from exc


    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _normalize_messages(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """
        Validate and normalize a sequence of chat messages.

        - Ensures `messages` is a list of mappings.
        - Ensures each entry has string `role` and `content` keys.
        - Normalizes `role` to lowercase.
        """
        if not isinstance(messages, list):
            raise LLMEngineError("LLMEngine.invoke: messages must be a list")
        if not messages:
            raise LLMEngineError("LLMEngine.invoke: messages must not be empty")

        normalized: List[Dict[str, str]] = []
        for idx, msg in enumerate(messages):
            if not isinstance(msg, Mapping):
                raise LLMEngineError(
                    f"LLMEngine.invoke: message {idx} is not a mapping (got {type(msg)!r})"
                )
            role = msg.get("role")
            content = msg.get("content")
            if not isinstance(role, str) or not isinstance(content, str):
                raise LLMEngineError(
                    "LLMEngine.invoke: each message must have 'role' and 'content' as strings"
                )
            normalized.append({"role": role.lower(), "content": content})
        return normalized

    def _validate_attachment_path(self, path: str) -> None:
        """
        Generic path/extension validation used by `attach`.

        Subclasses are expected to further restrict supported types by setting
        `allowed_attachment_exts` and/or overriding this method.
        """
        if not os.path.isfile(path):
            raise LLMEngineError(
                f"LLMEngine.attach: path does not exist or is not a file: {path!r}"
            )

        _, ext = os.path.splitext(path)
        ext = ext.lower()

        if ext and ext in self.illegal_attachment_exts:
            raise LLMEngineError(
                f"LLMEngine.attach: extension {ext!r} is not allowed for safety/robustness"
            )

        allowed_exts = self.allowed_attachment_exts
        if allowed_exts is not None and ext not in allowed_exts:
            raise LLMEngineError(
                f"LLMEngine.attach: extension {ext!r} is not supported by {self._name}"
            )

    def _call_with_retries(self, payload: Any) -> Any:
        """
        Call `_call_provider` with a basic retry/backoff loop.

        Subclasses can customize retry behavior by overriding `_should_retry`
        or by setting `max_retries`/backoff parameters in the constructor.
        """
        attempt = 0
        while True:
            attempt += 1
            try:
                return self._call_provider(payload)
            except LLMEngineError:
                # Already normalized; do not re-wrap or retry.
                raise
            except Exception as exc:
                if not self._should_retry(exc, attempt):
                    raise
                sleep = min(
                    self._retry_backoff_base * (2 ** (attempt - 1)),
                    self._retry_backoff_max,
                )
                # Add a little jitter to avoid thundering herds.
                sleep *= random.uniform(0.8, 1.2)
                logger.debug(
                    "LLMEngine %s attempt %d failed with %r; retrying in %.2fs",
                    self._name,
                    attempt,
                    exc,
                    sleep,
                )
                time.sleep(sleep)

    def _should_retry(self, exc: Exception, attempt: int) -> bool:
        """
        Decide whether a failed `_call_provider` should be retried.

        Default policy:
        - Do not exceed `self._max_retries`.
        - Retry on basic timeout/connection-style errors.
        Subclasses may override this to recognize provider-specific error types.
        """
        if attempt > self._max_retries:
            return False

        # Simple baseline: retry on common transient conditions. We avoid importing
        # provider SDK exceptions here; subclasses can override for finer control.
        return isinstance(exc, (TimeoutError, ConnectionError))

    # --------------------------------------------------------------------- #
    # Abstract Helpers
    # --------------------------------------------------------------------- #
    @abstractmethod
    def _build_provider_payload(self, messages: List[Dict[str, str]], attachments: Mapping[str, Mapping[str, Any]]) -> Any:
        """
        Convert normalized messages and attachments into the provider-specific
        request payload.
        """
        raise NotImplementedError

    @abstractmethod
    def _call_provider(self, payload: Any) -> Any:
        """
        Perform a single call to the underlying provider using the given payload.

        This method should honor `self._timeout_seconds` where possible.
        """
        raise NotImplementedError

    @abstractmethod
    def _extract_text(self, response: Any) -> str:
        """
        Extract the assistant's textual reply from a provider response object.
        """
        raise NotImplementedError

    @abstractmethod
    def _prepare_attachment(self, path: str) -> Mapping[str, Any]:
        """
        Prepare a local path for reuse with this engine.

        Implementations typically:
        - validate the path and extension vs provider capabilities,
        - perform any remote upload or inlining,
        - return an opaque metadata mapping used later by `_build_provider_payload`.
        """
        raise NotImplementedError
    
    @abstractmethod
    def _on_detach(self, meta: Mapping[str, Any]) -> None:
        """
        Optional hook called when an attachment is detached.

        Subclasses may implement provider-specific cleanup (e.g. remote delete).
        Default implementation is a no-op.
        """
        # Intentionally a no-op by default.
        raise NotImplementedError


    # --------------------------------------------------------------------- #
    # Serialization
    # --------------------------------------------------------------------- #
    def to_dict(self) -> OrderedDict[str, Any]:
        """
        Shallow, non-secret configuration snapshot for debugging / logging.
        """
        return OrderedDict({
            "name": self._name,
            "timeout_seconds": self._timeout_seconds,
            "max_retries": self._max_retries,
            "attachments": self._attachments,
            "provider": type(self).__name__,
        })


# ───────────────────────────────────────────────────────────────────────────────
# Tool primitive
# ───────────────────────────────────────────────────────────────────────────────
ArgumentMap = OrderedDict[str, Dict[str, Any]]

class Tool:
    """Concrete base Tool primitive.

    This class provides a *dict-first* invocation interface around an underlying
    callable. It implements the template method::

        invoke(inputs) -> to_arg_kwarg(inputs) -> execute(args, kwargs)

    Subclasses such as MCPProxyTool and AgentTool are expected to override only
    the helper hooks used at construction time (``_get_mod_qual``,
    ``_build_io_schemas``, ``_compute_is_persistible``) and, where necessary,
    the :meth:`to_arg_kwarg` / :meth:`execute` methods. The public
    :meth:`invoke` method must not be overridden.

    The argument schema is exposed via :attr:`arguments_map` as an
    OrderedDict whose values are JSON-serialisable metadata dictionaries with
    the following keys:

    - ``index``  : ``int`` – the parameter position.
    - ``kind``   : ``str`` – one of
      ``{"POSITIONAL_ONLY", "POSITIONAL_OR_KEYWORD", "KEYWORD_ONLY", "VAR_POSITIONAL", "VAR_KEYWORD"}``.
    - ``type``   : ``str`` – a human-readable type name, e.g. ``"int"``.
    - ``default``: optional – present only if the parameter has a default.

    ``return_type`` is always stored as a string as well.
    """

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        function: Callable[..., Any],
        name: Optional[str] = None,
        namespace: Optional[str] = None,
        description: Optional[str] = None,
    ) -> None:
        if not callable(function):
            raise ToolDefinitionError(f"Tool function must be callable, got {type(function)!r}")
        self._function: Callable[..., Any] = function
        self._name: str = name or getattr(function, "__name__", "unnamed_callable") or "unnamed_callable"
        self._namespace: str = namespace or "default"
        self._description: str = description or (getattr(function, "__doc__", None) or "")

        # Identity in import space (may be overridden by subclasses).
        self._module, self._qualname = self._get_mod_qual(function)

        # Build argument schema and return type from the current function.
        self._arguments_map, self._return_type = self._build_io_schemas()

        # Persistibility flag exposed as a public property.
        self._is_persistible_internal: bool = self._compute_is_persistible()


    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #
    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    @property
    def namespace(self) -> str:
        return self._namespace

    @namespace.setter
    def namespace(self, value: str) -> None:
        self._namespace = value

    @property
    def description(self) -> str:
        return self._description

    @description.setter
    def description(self, value: str) -> None:
        self._description = value

    @property
    def function(self) -> Callable[..., Any]:
        return self._function

    @function.setter
    def function(self, func: Callable[..., Any]) -> None:
        """Update the underlying callable and refresh schema & identity."""
        if not callable(func):
            raise ToolDefinitionError(f"Tool function must be callable, got {type(func)!r}")
        self._function = func
        self._module, self._qualname = self._get_mod_qual(func)
        self._arguments_map, self._return_type = self._build_io_schemas()
        self._is_persistible_internal = self._compute_is_persistible()

    @property
    def arguments_map(self) -> ArgumentMap:
        """Ordered mapping of parameter name → {index, kind, type, default?}."""
        return self._arguments_map

    @property
    def return_type(self) -> str:
        """Return type (always as a string)."""
        return self._return_type

    @property
    def module(self) -> Optional[str]:
        return self._module

    @property
    def qualname(self) -> Optional[str]:
        return self._qualname

    @property
    def is_persistible(self) -> bool:
        """Whether this Tool can be reconstructed from its serialized form."""
        return self._is_persistible_internal

    @property
    def full_name(self) -> str:
        """Fully-qualified tool name of the form ``Type.namespace.name``."""
        return f"{type(self).__name__}.{self._namespace}.{self._name}"

    @property
    def signature(self) -> str:
        """Human-readable signature derived from ``arguments_map`` and
        ``return_type``."""
        params: List[str] = []
        for name, meta in self._arguments_map.items():
            kind = meta.get("kind", "")
            type_name = meta.get("type", "Any")
            default_marker = ""
            if "default" in meta:
                default = str(meta['default'])
                default_marker = f" = {default}"
            if kind == "VAR_POSITIONAL":
                param_str = f"*{name}: {type_name}{default_marker}"
            elif kind == "VAR_KEYWORD":
                param_str = f"**{name}: {type_name}{default_marker}"
            else:
                param_str = f"{name}: {type_name}{default_marker}"
            params.append(param_str)
        params_str = ", ".join(params)
        return f"{self.full_name}({params_str}) -> {self._return_type}"


    # ------------------------------------------------------------------ #
    # Stringification
    # ------------------------------------------------------------------ #
    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"{self.signature}: {self._description}"

    __str__ = __repr__

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def invoke(self, inputs: Mapping[str, Any]) -> Any:
        """Invoke the tool using a dict-like mapping of inputs.

        This is the *only* public entrypoint for execution. Subclasses must not
        override this method; instead they can customise :meth:`to_arg_kwarg`
        and :meth:`execute`.
        """
        if not isinstance(inputs, Mapping):
            raise ToolInvocationError(f"{self._name}: inputs must be a mapping")
        args, kwargs = self.to_arg_kwarg(inputs)
        result = self.execute(args, kwargs)
        return result


    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _get_mod_qual(self, function: Callable[..., Any]) -> tuple[Optional[str], Optional[str]]:
        """Determine ``(module, qualname)`` for callable-based tools.

        Subclasses that do not use Python import identity (e.g. MCPProxyTool)
        should override this to return ``(None, None)``.
        """
        module = getattr(function, "__module__", None)
        qualname = getattr(function, "__qualname__", None)
        return module, qualname

    def _format_annotation(self, ann: Any) -> str:
        """Convert a type annotation into a readable string.

        Behaviour:
        - If missing/empty → 'Any'.
        - If already a string → returned as-is.
        - If a parameterized / generic type (e.g. List[Dict[str, int]] or dict[str, int]):
          builds the full nested structure string.
        - If a plain class → its name (e.g. 'int', 'MyModel').
        - Otherwise → best-effort str(ann).
        """

        # Missing / unknown annotation
        if ann is inspect._empty or ann is None:
            return "Any"

        # Forward reference or explicit string annotation
        if isinstance(ann, str):
            return ann

        # typing / generic / PEP 585 parameterized types
        origin = get_origin(ann)
        if origin is not None:
            # Recursively format origin and args
            origin_str = self._format_annotation(origin)
            args = get_args(ann)
            if not args:
                return origin_str
            args_str = ", ".join(self._format_annotation(a) for a in args)
            return f"{origin_str}[{args_str}]"

        # Plain classes / types
        module = getattr(ann, "__module__", None)
        name = getattr(ann, "__name__", None)
        if module == "builtins" and name:
            # int, str, dict, list, etc.
            return name
        if name:
            # Custom or library class
            return name

        # Fallback: best-effort string representation
        return str(ann)

    def _build_io_schemas(self) -> tuple[ArgumentMap, str]:
        """Construct ``arguments_map`` and ``return_type`` from the wrapped
        callable's signature.

        Rules:
        - If an annotation is present, it *always* defines the type string.
        - If no annotation but a default value exists, the type string is
          derived from ``type(default)``.
        - If neither is present, the type string is 'Any'.
        """

        sig = inspect.signature(self._function)
        arg_map: ArgumentMap = OrderedDict()

        for index, (name, param) in enumerate(sig.parameters.items()):
            kind_name = param.kind.name  # e.g. "POSITIONAL_ONLY"
            ann = param.annotation
            default = param.default

            # Decide the source of the type information:
            # 1) annotation if present,
            # 2) otherwise the default's type if present,
            # 3) otherwise "Any".
            if ann is not inspect._empty:
                raw_type = ann
            elif default is not inspect._empty:
                raw_type = type(default)
            else:
                raw_type = inspect._empty

            type_str = self._format_annotation(raw_type)

            meta: Dict[str, Any] = {
                "index": index,
                "kind": kind_name,
                "type": type_str,
            }
            if default is not inspect._empty:
                meta["default"] = default

            arg_map[name] = meta

        # Return type: annotation if present, else 'Any'
        ret_ann = sig.return_annotation
        return_type = self._format_annotation(ret_ann)

        return arg_map, return_type

    def _compute_is_persistible(self) -> bool:
        """Default persistibility check for callable-based tools.

        A Tool is considered persistible if its function has both ``__module__``
        and ``__qualname__`` and does not appear to be a local/helper function.
        Subclasses can override this with their own criteria.
        """
        if not self._module or not self._qualname:
            return False
        # Heuristic: local/helper functions usually contain '<locals>' in qualname.
        if "<locals>" in self._qualname:
            return False
        return True

    def to_arg_kwarg(self, inputs: Mapping[str, Any]) -> tuple[tuple[Any, ...], Dict[str, Any]]:
        """Default implementation for mapping input dicts to ``(*args, **kwargs)``.

        The base policy is:

        - Required parameters (those without ``default`` and not VAR_*) must be present.
        - Unknown keys raise if there is no VAR_KEYWORD parameter; otherwise they
          are accepted and passed through in ``**kwargs``.
        - POSITIONAL_ONLY parameters are always passed positionally.
        - POSITIONAL_OR_KEYWORD and KEYWORD_ONLY parameters are passed as
          keywords (Python accepts this for both kinds).
        - VAR_POSITIONAL expects the mapping to contain the parameter name with
          a sequence value; these are appended to ``*args``.
        - VAR_KEYWORD collects all remaining unknown keys into ``**kwargs``.
        """
        data: Dict[str, Any] = dict(inputs)

        # Compute parameter ordering and kinds
        param_items = sorted(self._arguments_map.items(), key=lambda kv: kv[1].get("index", 0))
        param_names = {name for name, _ in param_items}

        varpos_name: Optional[str] = None
        varkw_name: Optional[str] = None
        for name, meta in param_items:
            kind = meta.get("kind")
            if kind == "VAR_POSITIONAL" and varpos_name is None:
                varpos_name = name
            elif kind == "VAR_KEYWORD" and varkw_name is None:
                varkw_name = name

        # Unknown key handling
        unknown_keys = set(data.keys()) - param_names
        if unknown_keys and varkw_name is None:
            raise ToolInvocationError(f"{self._name}: unknown parameters: {sorted(unknown_keys)}")

        # Required parameter check (exclude VAR_*)
        required_names = [
            name
            for name, meta in param_items
            if meta.get("kind") not in {"VAR_POSITIONAL", "VAR_KEYWORD"}
            and "default" not in meta
        ]
        missing = [name for name in required_names if name not in data]
        if missing:
            raise ToolInvocationError(f"{self._name}: missing required parameters: {missing}")

        args: List[Any] = []
        kwargs: Dict[str, Any] = {}

        # First, handle named parameters in order
        for name, meta in param_items:
            kind = meta.get("kind")
            has_default = "default" in meta
            value_provided = name in data

            if kind == "VAR_POSITIONAL":
                # Expect a sequence under this key, if provided
                if value_provided:
                    val = data[name]
                    if not isinstance(val, (list, tuple)):
                        raise ToolInvocationError(
                            f"{self._name}: var-positional parameter '{name}' must be a list or tuple"
                        )
                    args.extend(list(val))
                continue

            if kind == "VAR_KEYWORD":
                # Handled after all named parameters
                continue

            if not value_provided and has_default:
                val = meta["default"]
            elif value_provided:
                val = data[name]
            else:
                # Required parameters should already have been validated above
                continue

            if kind == "POSITIONAL_ONLY":
                args.append(val)
            else:
                # POS_OR_KEYWORD or KEYWORD_ONLY – pass as keyword argument
                kwargs[name] = val

        # Handle VAR_KEYWORD by collecting unknown keys + any explicit mapping under its own name
        if varkw_name is not None:
            extra_kwargs: Dict[str, Any] = {}
            # If caller explicitly supplied a mapping for the VAR_KEYWORD parameter name, merge it.
            if varkw_name in data:
                explicit = data[varkw_name]
                if not isinstance(explicit, Mapping):
                    raise ToolInvocationError(
                        f"{self._name}: var-keyword parameter '{varkw_name}' must be a mapping if provided"
                    )
                extra_kwargs.update(dict(explicit))
            # Include unknown keys
            for key in unknown_keys:
                if key in extra_kwargs or key in kwargs:
                    raise ToolInvocationError(
                        f"{self._name}: duplicate key '{key}' in **kwargs aggregation"
                    )
                extra_kwargs[key] = data[key]
            kwargs.update(extra_kwargs)

        return tuple(args), kwargs

    def execute(self, args: tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        """Execute the underlying callable.

        Subclasses may override this to change *how* a tool is executed (for
        example, by making a remote MCP call or invoking an Agent), but should
        not change the high-level semantics.
        """
        try:
            result = self._function(*args, **kwargs)
        except Exception as e:  # pragma: no cover - thin wrapper
            raise ToolInvocationError(f"{self.full_name}: invocation failed: {e}") from e
        return result


    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        """Serialize this tool's header and argument schema.

        Note that deserialisation will handled by a future :mod:`Factory`; this method does
        *not* perform any persistibility checks and will not raise solely
        because :attr:`is_persistible` is ``False``.
        """
        return {
            "tool_type": type(self).__name__,
            "name": self.name,
            "namespace": self.namespace,
            "description": self.description,
            "signature": self.signature,
            "module": self._module,
            "qualname": self._qualname,
            "is_persistible": self.is_persistible,
        }


# ───────────────────────────────────────────────────────────────────────────────
# Agent Primitive
# ───────────────────────────────────────────────────────────────────────────────
def identity_pre(*, prompt: str) -> str:
    if not isinstance(prompt, str):
        raise ValueError("prompt must be a string")
    return prompt

def identity_post(*, result: Any) -> Any:
    """
    Default post-invoke identity function.

    This function accepts a single argument named ``result`` and returns it
    unchanged. It is wrapped as a Tool and used when no explicit ``post_invoke``
    Tool is provided.
    """
    return result


class Agent:
    """
    Schema-driven LLM Agent.

    An Agent is a stateful software unit that points to an LLM and carries a persona
    (system role-prompt). It accepts a single **input mapping** and uses a **pre-invoke
    Tool** to convert that mapping into a **prompt string** before invoking the engine.

    Core behavior:
    - `invoke(inputs: Mapping[str, Any]) -> Any`
      1) Validate inputs (must be a Mapping).
      2) `pre_invoke.invoke(inputs) -> str` (default strict: requires `{"prompt": str}`).
      3) Build messages: [system?] + [last N turns] + [user(prompt)].
      4) Call `llm_engine.invoke(messages)` to obtain a raw result (base Agent expects `str`).
      5) Pass the raw result through `post_invoke` (a single-parameter Tool) to obtain the final output.
      6) Record the turn if `context_enabled`; return the final result.

    Inputs and schema
    -----------------
    - Inputs are always a mapping (`Mapping[str, Any]`).
    - The schema is defined entirely by the `pre_invoke` Tool, which:
      - validates and normalizes the incoming mapping,
      - converts it into a prompt string.

    History and context
    -------------------
    - The Agent keeps an in-memory history of messages as a flat list of dicts:
      `{"role": "user"|"assistant"|"system", "content": str}`.
    - `history_window` controls how many *turns* (user+assistant pairs) from the tail
      of the history are included when building messages.
    - History is append-only; no trimming or summarization is performed by default.

    Parameters
    ----------
    name : str
        Logical name for this agent (read-only).
    description : str
        Short, human-readable description (read-only).
    llm_engine : LLMEngine
        Engine used to perform the model call. Must be an instance of `LLMEngine`.
    role_prompt : Optional[str], default None
        Optional system persona. If None or empty, no system message is sent.
    context_enabled : bool, default True
        If True, the agent includes and logs conversation context (turns).
        If False, the agent sends no prior turns and does not log new ones.
    history_window : int, default 50
        Send-window measured in **turns** (user+assistant pairs). `0` sends no turns.
        Stored history is never trimmed.
    pre_invoke : Optional[Tool or Callable], default None
        Tool that converts the input mapping to a `str` prompt. If None, a strict
        identity Tool is created that accepts exactly `{"prompt": str}`.
    post_invoke : Optional[Tool or Callable], default None
        Tool that converts the raw result from :meth:`_invoke` into the final
        return value. It must accept exactly one parameter. If None, a default
        identity Tool is used that returns its single argument unchanged.

    Properties (selected)
    ---------------------
    name : str (read-only)
    description : str (read-only)
    role_prompt : Optional[str] (read-write)
    llm_engine : LLMEngine (read-write, type-enforced)
    context_enabled : bool (read-write)
    history_window : int (read-write; see semantics above)
    history : List[Dict[str, str]] (read-only view)
    attachments : Dict[str, Dict[str, Any]] (read-only view)
    pre_invoke : Tool (read-write)
    post_invoke : Tool (read-write)
    """

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        name: str,
        description: str,
        llm_engine: LLMEngine,
        role_prompt: Optional[str] = None,
        context_enabled: bool = True,
        *,
        pre_invoke: Optional[Tool | Callable] = None,
        post_invoke: Optional[Tool | Callable] = None,
        history_window: Optional[int] = None,
    ) -> None:
        
        # set the core attributes
        self._name = name
        self._description = description
        self._llm_engine: LLMEngine = llm_engine
        self._role_prompt: Optional[str] = role_prompt or "You are a helpful AI assistant"
        self._context_enabled: bool = context_enabled

        # history_window: strict int semantics (>= 0). 'None' means (send-all) mode.
        if history_window is not None and (not isinstance(history_window, int) or history_window < 0):
            raise AgentError("history_window must be an int >= 0 or be 'None'.")
        self._history_window: Optional[int] = history_window

        # Stored message history (flat list of role/content dicts).
        # We never trim storage; we only limit what we *send* to the engine.
        self._history: List[Dict[str, str]] = []

        # Per-invoke buffer for the newest run. This is populated inside `_invoke`
        # and committed to `_history` by `_update_history` if `context_enabled` is True.
        self._newest_history: List[Dict[str, str]] = []
        
        #invoke lock
        self._invoke_lock = threading.RLock()

        # ~~~ Pre-Invoke Tool preparation ~~~
        # turn callable to pre-invoke tool
        if pre_invoke is not None and isinstance(pre_invoke, Callable):
            pre_invoke = Tool(
                function=pre_invoke,
                name="pre_invoke",
                namespace=self._name,
                description="Pre-invoke callable adapted to Tool",
            )
        # identity pre-invoke by default -> requires {"prompt": str} 
        if pre_invoke is None:
            pre_invoke: Tool = Tool(identity_pre,
                                    "pre_invoke",
                                    self.name,
                                    identity_pre.__doc__)
        # keep pre_invoke as-is if its a Tool
        elif isinstance(pre_invoke, Tool):
            pre_invoke = pre_invoke
        # raise error otherwise
        else:
            raise ValueError("pre_invoke must be a Tools.Tool instance or a callable object.")
        # if the return type isn't 'any' or 'string' or 'str', then raise an error.
        if pre_invoke.return_type.lower() not in ["str", "any", "string"]:
            raise AgentError(
                f"pre_invoke Tool for agent {self._name} must return a type str (or type 'Any' as superset containing 'str'); "
                f"got {self.pre_invoke.return_type}."
            )
        # set pre-invoke
        self._pre_invoke: Tool = pre_invoke
        
        # ~~~ Post-Invoke Tool Preparation ~~~
        # turn callable to post-invoke tool
        if post_invoke is not None and isinstance(post_invoke, Callable):
            post_invoke = Tool(post_invoke,
                               "post_invoke",
                               self.name,
                               identity_post.__doc__ or f"Post-invoke callable adapted to Tool")
        # identity post-invoke by default -> requires single {"result": Any}
        if post_invoke is None:
            post_tool: Tool = Tool(identity_post,
                                   "post_invoke",
                                   self.name,
                                   identity_post.__doc__)
        # keep post_invoke as-is if its a Tool
        elif isinstance(post_invoke, Tool):
            post_tool = post_invoke
        # raise error otherwise
        else:
            raise ValueError("post_invoke must be a Tools.Tool instance or a callable object.")
        # if post_invoke's arguments map is not one parameter long, then raise an error
        post_arg_map = post_tool.arguments_map
        if len(post_arg_map) != 1:
            raise AgentError(
                f"post_invoke Tool for agent {self._name} must accept exactly one parameter; "
                f"got {len(post_arg_map)}."
            )
        # set post-invoke
        self._post_invoke: Tool = post_tool
        # cache the single parameter name for efficient calls.
        self._post_param_name: str = next(iter(post_arg_map.keys()))

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #
    @property
    def name(self) -> str:
        """Read-only agent name."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    @property
    def description(self) -> str:
        """Return description agent description."""
        return self._description

    @description.setter
    def description(self, val: str) -> None:
        """Set description"""
        self._description = val.strip() or f"A helpful AI assistant named {self._name}"

    @property
    def role_prompt(self) -> Optional[str]:
        """Optional system persona (first message if set)."""
        return self._role_prompt

    @role_prompt.setter
    def role_prompt(self, value: str) -> None:
        if not isinstance(value, str):
            raise TypeError(f"role_prompt must be a 'str', but got '{type(value).__name__}'")
        self._role_prompt = value.strip() or "You are a helpful AI assistant"
    
    @property
    def llm_engine(self) -> LLMEngine:
        """LLMEngine used for this agent."""
        return self._llm_engine

    @llm_engine.setter
    def llm_engine(self, engine: LLMEngine) -> None:
        if not isinstance(engine, LLMEngine):
            raise TypeError("llm_engine must be an instance of LLMEngine.")
        self._llm_engine = engine

    @property
    def context_enabled(self) -> bool:
        """
        Whether the agent uses conversation context
        (sends prior turns and logs new ones).
        """
        return self._context_enabled

    @context_enabled.setter
    def context_enabled(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise ValueError("context_enabled must be a bool.")
        self._context_enabled = value

    @property
    def history_window(self) -> int:
        """
        Number of *turns* (user+assistant pairs) to include from the tail of the
        conversation history when building messages for the engine.
        """
        return self._history_window

    @history_window.setter
    def history_window(self, value: int) -> None:
        if not isinstance(value, int) or value < 0:
            raise ValueError("history_window must be an int >= 0.")
        self._history_window = value

    @property
    def history(self) -> List[Dict[str, str]]:
        """Return a shallow copy of the stored message history (never trimmed)."""
        return list(self._history)

    @property
    def attachments(self) -> Dict[str, Dict[str, Any]]:
        """A shallow copy of the current attachment paths."""
        return self.llm_engine.attachments

    @property
    def pre_invoke(self) -> Tool:
        """
        Tool that converts the input mapping into a **prompt string**.

        By default this is a strict identity tool that requires *exactly*:
            {"prompt": <str>}
        and returns that string.

        You can set this to any Tool instance that:
        - accepts a Mapping[str, Any]-like object, and
        - returns a `str` when invoked.

        For convenience, setting it to a plain callable will wrap it in a Tool.
        """
        return self._pre_invoke

    @pre_invoke.setter
    def pre_invoke(self, tool: Union[Tool, Callable[..., str]]) -> None:
        """
        Set the pre-invoke Tool.

        Parameters
        ----------
        tool : Tool or callable
            - If a Tool, it is used as-is.
            - If a callable, it is wrapped as a Tool with a simple signature.
        """
        if isinstance(tool, Callable):
            tool = Tool(
                function=tool,
                name="pre_invoke",
                namespace=self.name,
                description=tool.__doc__ or f"Pre-invoke callable adapted to Tool for agent {self.name}",
            )
        elif not isinstance(tool, Tool):
            raise ValueError("pre_invoke must be a Tools.Tool instance or a callable object.")
        if tool.return_type.lower() not in ['any', 'str', 'string']:
            raise AgentError(f"pre_invoke must return a type 'str' after preprocessing the inputs. Got '{tool.return_type}'")
        self._pre_invoke = tool

    @property
    def post_invoke(self) -> Tool:
        """
        Tool that converts the raw result from :meth:`_invoke` into the final
        return value for :meth:`invoke`.

        This Tool must accept exactly one parameter. The base Agent enforces
        this constraint when the Tool is set.
        """
        return self._post_invoke

    @post_invoke.setter
    def post_invoke(self, tool: Union[Tool, Callable[..., Any]]) -> None:
        """
        Set the post-invoke Tool.

        Parameters
        ----------
        tool : Tool or callable
            - If a Tool, it is used as-is.
            - If a callable, it is wrapped as a Tool with a simple signature.
        """
        if isinstance(tool, Callable):
            tool = Tool(
                function=tool,
                name="post_invoke",
                namespace=self.name,
                description=tool.__doc__ or f"Post-invoke callable adapted to Tool for agent {self.name}",
            )
        elif not isinstance(tool, Tool):
            raise ValueError("post_invoke must be a Tools.Tool instance or a callable object.")

        post_arg_map = tool.arguments_map
        if len(post_arg_map) != 1:
            raise AgentError(
                f"post_invoke Tool for agent {self.name} must accept exactly one parameter; "
                f"got {len(post_arg_map)}."
            )

        self._post_invoke = tool
        self._post_param_name = next(iter(post_arg_map.keys()))
    
    @property
    def arguments_map(self) -> OrderedDict[str, Any]:
        """
        Mirror of the pre_invoke Tool's arguments map.

        Exposed primarily so that planners and higher-level workflows can inspect
        the agent's input schema (required names, ordering, etc.).
        """
        return OrderedDict(self.pre_invoke.arguments_map)


    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def attach(self, path: str) -> Mapping[str, Any]:
        """
        Add a file path to attachments via the underlying engine.

        Returns
        -------
        bool
            True if added, False if it was already present.
        """
        return self._llm_engine.attach(path)

    def detach(self, path: str) -> bool:
        """
        Remove a file path from attachments via the underlying engine.

        Returns
        -------
        bool
            True if removed, False if it was not present.
        """
        return self._llm_engine.detach(path)

    def clear_attachments(self) -> None:
        return self.llm_engine.clear_attachments()

    def clear_memory(self) -> None:
        """Clear the stored message history."""
        with self._invoke_lock:
            self._history.clear()
            self._newest_history.clear()
    
    def invoke(self, inputs: Mapping[str, Any]) -> Any:
        """Invoke the Agent with a single **input mapping**.

        Steps
        -----
        1) Validate ``inputs`` is a :class:`~collections.abc.Mapping`.
        2) ``prompt = pre_invoke.invoke(inputs)`` → must be a ``str``.
           - If the Tool raises :class:`ToolInvocationError`, it propagates unchanged.
           - Other exceptions are wrapped as :class:`AgentInvocationError`.
        3) Build the messages list from the optional role prompt, the windowed
           history, and the current user ``prompt``.
        4) Delegate to :meth:`_invoke`, which performs the actual engine call and
           populates ``_newest_history`` for this run.
        5) If ``context_enabled`` is True, call update history with the _newest_history to commit
           the newest turn(s) into persistent history.
        6) Run ``post_invoke`` on the raw result to obtain the final output and
           return it.

        Parameters
        ----------
        inputs : Mapping[str, Any]
            Input mapping to be adapted to a prompt string via ``pre_invoke``.

        Returns
        -------
        Any
            The Agent's response. For the base Agent, this is the result of the
            ``post_invoke`` Tool applied to the raw LLM output.

        Raises
        ------
        TypeError
            If ``inputs`` is not a Mapping.
        ToolInvocationError
            If the pre- or post-invoke Tool rejects the inputs.
        AgentInvocationError
            For unexpected runtime errors in Tools or the engine.
        """
        
        # main invoke lock        
        with self._invoke_lock:
            logger.info(f"[{type(self).__name__}.{self.name}.invoke started]")
            if not isinstance(inputs, Mapping):
                raise TypeError("Agent.invoke expects a Mapping[str, Any].")
            # Preprocess inputs to prompt string
            try:
                logger.debug(f"Agent.{self.name}.pre_invoke preprocessing inputs")
                prompt = self._pre_invoke.invoke(inputs)
            except ToolInvocationError:
                raise
            except Exception as e:  # pragma: no cover
                raise AgentInvocationError(f"pre_invoke Tool failed: {e}") from e

            if not isinstance(prompt, str):
                raise AgentInvocationError(
                    f"pre_invoke returned non-string (type={type(prompt)!r}); a prompt string is required"
                )
            # Empty any prior history buffer
            self._newest_history.clear()

            # Build messages
            logger.debug(f"Agent.{self.name} building messages for class '{type(self).__name__}'")
            messages: List[Dict[str, str]] = [{"role": "system", "content": self.role_prompt}]

            if self._context_enabled and self._history:
                if self._history_window is None:
                    prior = self._history
                elif self._history_window > 0:
                    prior = self._history[-(self._history_window * 2):]
                else:
                    prior = self._history
                messages.extend(prior)

            user_msg = {"role": "user", "content": prompt}
            messages.append(user_msg)

            # Invoke core logic
            logger.debug(f"Agent.{self.name} performing logic for class '{type(self).__name__}'")
            raw_result = self._invoke(messages=messages)

            # Update history if enabled
            if self._context_enabled:
                logger.debug(f"Agent.{self.name} updating history")
                self._history.extend(self._newest_history)
            self._newest_history.clear()

            # Postprocess raw result
            try:
                logger.debug(f"Agent.{self.name}.post_invoke postprocessing result")
                result = self._post_invoke.invoke({self._post_param_name: raw_result})
            except ToolInvocationError:
                raise
            except Exception as e:  # pragma: no cover
                raise AgentInvocationError(f"post_invoke Tool failed: {e}") from e

            # Final logging and return
            logger.info(f"[{type(self).__name__}.{self.name}.invoke finished]")
            return result

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _invoke(self, messages: List[Dict[str, str]]) -> Any:
        """Internal call path used by :meth:`invoke`.

        This base implementation:
        - Invokes the configured LLM engine with the provided ``messages``.
        - Populates ``_newest_history`` with the *current turn only*
          (user + assistant messages).

        Subclasses may override this method to implement more complex behavior,
        but **must not** mutate ``self._history`` directly. Instead, they should
        either:
        - append messages to ``self._newest_history``, or
        - store richer run data on ``self`` for :meth:`_update_history` to
          summarize and commit.
        """
        # 1) Call engine (attachments are managed by the engine itself)
        try:
            logger.debug(f"[Agent - {self.name}]._invoke: Invoking LLM")
            text = self._llm_engine.invoke(messages)
        except Exception as e:  # pragma: no cover - engine-specific failures
            raise AgentInvocationError(f"engine invocation failed: {e}") from e

        # 2) Engine contract: base Agent expects a string
        if not isinstance(text, str):
            raise AgentInvocationError(
                f"engine returned non-string (type={type(text)!r}); a string is required"
            )

        # 3) Record the current turn into the per-invoke buffer.
        # The base Agent simply stores the user prompt and raw assistant text.
        user_msg = messages[-1]
        
        self._newest_history.append(user_msg)
        self._newest_history.append({"role": "assistant", "content": text})

        return text

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #
    def to_dict(self) -> OrderedDict[str, Any]:
        """A minimal diagnostic snapshot of this agent (safe to log/serialize)."""
        return OrderedDict(
            agent_type=type(self).__name__,
            name=self._name,
            description=self._description,
            role_prompt=self._role_prompt,
            pre_invoke=self._pre_invoke.to_dict(),
            post_invoke=self._post_invoke.to_dict(),
            llm=self._llm_engine.to_dict() if self._llm_engine else None,
            context_enabled=self._context_enabled,
            history_window=self._history_window,
            history=self._history,
        )


# ───────────────────────────────────────────────────────────────────────────────
# Workflow primitive
# ───────────────────────────────────────────────────────────────────────────────
class _NoValSentinel:
    """Sentinel used to mark required output fields in an output schema template."""
    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover
        return "NO_VAL"

NO_VAL: Any = _NoValSentinel()

class BundlingPolicy(str, Enum):
    """Controls whether raw results are bundled into a single output field."""
    BUNDLE = "BUNDLE"
    UNBUNDLE = "UNBUNDLE"

class MappingPolicy(str, Enum):
    """Controls how mapping-shaped raw outputs are interpreted."""
    STRICT = "STRICT"
    IGNORE_EXTRA = "IGNORE_EXTRA"
    MATCH_FIRST_STRICT = "MATCH_FIRST_STRICT"
    MATCH_FIRST_LENIENT = "MATCH_FIRST_LENIENT"

@dataclass(frozen=True, slots=True)
class WorkflowCheckpoint:
    """A single workflow invocation record."""
    run_id: str
    started_at: datetime
    ended_at: datetime
    elapsed_s: float
    inputs: Mapping[str, Any]
    raw_output: Any
    packaged_output: OrderedDict[str, Any]
    metadata: Dict[str, Any]

DEF_RES_KEY = "__RESULT__"

class Workflow(ABC):
    """
    Base template-method primitive for Workflow orchestrators.

    Workflows are a deterministic *packaging boundary*:

    - Inputs: always a Mapping[str, Any]
    - Execution: subclasses implement `_invoke(inputs)` and return
      (metadata: Mapping[str, Any], raw: Any)
    - Outputs: normalized and packaged against an ordered `output_schema`
      template using policy-driven packaging rules.

    IO schemas
    ----------
    - `arguments_map` is REQUIRED at construction time and is the authoritative
      source for `input_schema`.
    - `input_schema` mirrors `output_schema` format: OrderedDict[str, Any] where
      each value is either a default or NO_VAL.
    - Public mutation is disallowed: `arguments_map`, `input_schema`, and
      `output_schema` are read-only properties.
    - Subclasses may refresh schemas via `_set_io_schemas(arguments_map=..., output_schema=...)`
      when internal components change.

    Packaging policies
    ------------------
    - BundlingPolicy.BUNDLE: bundle raw output into the single schema key
      (requires output_schema length == 1).
    - BundlingPolicy.UNBUNDLE: attempt to coerce raw output into a Mapping or
      non-text Sequence; then apply MappingPolicy/sequence/scalar rules.

    Validation
    ----------
    Final validation that *no* keys remain set to NO_VAL is performed in `invoke()`,
    not inside the packaging helpers.

    Memory
    ------
    `clear_memory()` clears workflow checkpoints. Subclasses may extend via
    polymorphism.
    """

    def __init__(
        self,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        arguments_map: Mapping[str, Mapping[str, Any]],
        output_schema: Optional[Union[List[str], Mapping[str, Any]]] = None,
        bundling_policy: BundlingPolicy = BundlingPolicy.UNBUNDLE,
        mapping_policy: MappingPolicy = MappingPolicy.STRICT,
    ) -> None:
        self._name = name or type(self).__name__
        self._description = description

        if output_schema is None:
            output_schema = [DEF_RES_KEY]

        self._bundling_policy = BundlingPolicy(bundling_policy)
        self._mapping_policy = MappingPolicy(mapping_policy)

        self._invoke_lock = threading.RLock()
        self._checkpoints: List[WorkflowCheckpoint] = []

        # Normalized IO state (set by _set_io_schemas)
        self._arguments_map: OrderedDict[str, Dict[str, Any]]
        self._input_schema: OrderedDict[str, Any]
        self._output_schema: OrderedDict[str, Any]

        self._set_io_schemas(arguments_map=arguments_map, output_schema=output_schema)

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #
    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> Optional[str]:
        return self._description

    @property
    def arguments_map(self) -> Mapping[str, Mapping[str, Any]]:
        # Shallow copy to discourage external mutation.
        return OrderedDict(self._arguments_map)

    @property
    def input_schema(self) -> OrderedDict[str, Any]:
        return OrderedDict(self._input_schema)

    @property
    def output_schema(self) -> OrderedDict[str, Any]:
        return OrderedDict(self._output_schema)

    @property
    def output_keys(self) -> tuple[str, ...]:
        return tuple(self._output_schema.keys())

    @property
    def bundling_policy(self) -> BundlingPolicy:
        return self._bundling_policy

    @property
    def mapping_policy(self) -> MappingPolicy:
        return self._mapping_policy

    @property
    def checkpoints(self) -> tuple[WorkflowCheckpoint, ...]:
        return tuple(self._checkpoints)

    @property
    def latest_checkpoint(self) -> Optional[WorkflowCheckpoint]:
        return self._checkpoints[-1] if self._checkpoints else None

    # ------------------------------------------------------------------ #
    # Public API (Template Method)
    # ------------------------------------------------------------------ #
    def invoke(self, inputs: Mapping[str, Any]) -> OrderedDict[str, Any]:
        """
        Template method:
        1) Validate inputs is mapping
        2) Run _invoke() -> (metadata, raw)
        3) Package raw -> packaged (may still contain NO_VAL)
        4) Validate packaged contains no NO_VAL
        5) Save checkpoint
        6) Return packaged
        """
        if not isinstance(inputs, ABCMapping):
            raise ValidationError("Workflow.invoke: inputs must be a mapping")

        with self._invoke_lock:
            started = datetime.now(timezone.utc)
            run_id = uuid4().hex

            try:
                metadata, raw = self._invoke(inputs)
            except Exception as exc:
                raise ExecutionError(f"{type(self).__name__}._invoke failed") from exc

            if not isinstance(metadata, ABCMapping):
                raise ValidationError("Workflow._invoke: returned metadata must be a mapping")

            packaged = self.package(raw)

            # Final completeness validation happens HERE, not in packaging helpers.
            self._validate_packaged(packaged)

            ended = datetime.now(timezone.utc)
            elapsed = (ended - started).total_seconds()

            checkpoint = WorkflowCheckpoint(
                run_id=run_id,
                started_at=started,
                ended_at=ended,
                elapsed_s=elapsed,
                inputs=dict(inputs),
                raw_output=raw,
                packaged_output=OrderedDict(packaged),
                metadata=dict(metadata),  # snapshot
            )
            self._checkpoints.append(checkpoint)
            return packaged

    @abstractmethod
    def _invoke(self, inputs: Mapping[str, Any]) -> tuple[Mapping[str, Any], Any]:
        """Subclass-defined execution step; returns (metadata, raw_result)."""
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Memory
    # ------------------------------------------------------------------ #
    def clear_memory(self) -> None:
        """Clear workflow-owned memory (checkpoints). Subclasses may extend."""
        with self._invoke_lock:
            self._checkpoints.clear()

    # ------------------------------------------------------------------ #
    # Final validation (kept separate from packaging)
    # ------------------------------------------------------------------ #
    def _validate_packaged(self, packaged: Mapping[str, Any]) -> None:
        """
        Raise if any output fields remain NO_VAL after packaging.

        This method is intentionally called by `invoke()` and not by `package()`
        to keep packaging responsibilities separate from final contract validation.
        """
        missing = [k for k, v in packaged.items() if v is NO_VAL]
        if missing:
            raise PackagingError(f"Workflow packaging: missing required output fields: {missing}")

    # ------------------------------------------------------------------ #
    # Packaging
    # ------------------------------------------------------------------ #
    def package(self, raw: Any) -> OrderedDict[str, Any]:
        """
        Package a raw result into the workflow's output schema.

        IMPORTANT: This method does not perform final NO_VAL validation.
        That validation is performed by `invoke()` via `_validate_packaged()`.
        """
        keys = list(self._output_schema.keys())
        if not keys:
            raise SchemaError("Workflow has empty output_schema; cannot package results")

        # Bundling short-circuit
        if self._bundling_policy == BundlingPolicy.BUNDLE:
            if len(keys) != 1:
                raise SchemaError(
                    "BundlingPolicy.BUNDLE requires output_schema of length 1 "
                    f"(got {len(keys)})"
                )
            return OrderedDict([(keys[0], raw)])

        # UNBUNDLE flow
        normalized = self._normalize_raw(raw)
        template = OrderedDict(self._output_schema)

        if isinstance(normalized, ABCMapping):
            return self._package_from_mapping(template, normalized)
        if self._is_non_text_sequence(normalized):
            return self._package_from_sequence(template, normalized)  # type: ignore[arg-type]
        return self._package_from_scalar(template, normalized)

    def _normalize_raw(self, raw: Any) -> Any:
        if isinstance(raw, ABCMapping) or self._is_non_text_sequence(raw):
            return raw

        # Pydantic v2 models: model_dump()
        md = getattr(raw, "model_dump", None)
        if callable(md):
            try:
                dumped = md()
                if isinstance(dumped, ABCMapping):
                    return dumped
            except Exception:
                pass

        # NamedTuple: _asdict()
        ad = getattr(raw, "_asdict", None)
        if callable(ad):
            try:
                dumped = ad()
                if isinstance(dumped, ABCMapping):
                    return dumped
            except Exception:
                pass

        # Dataclass
        if is_dataclass(raw):
            try:
                dumped = asdict(raw)
                if isinstance(dumped, ABCMapping):
                    return dumped
            except Exception:
                pass

        # Plain object with __dict__
        try:
            dumped = vars(raw)
            if isinstance(dumped, ABCMapping):
                return dumped
        except Exception:
            pass

        return raw

    def _is_non_text_sequence(self, x: Any) -> bool:
        return isinstance(x, ABCSequence) and not isinstance(x, (str, bytes, bytearray))

    def _package_from_mapping(
        self,
        template: OrderedDict[str, Any],
        mapping: Mapping[str, Any],
    ) -> OrderedDict[str, Any]:
        schema_keys = list(template.keys())

        if self._mapping_policy == MappingPolicy.STRICT:
            for k in mapping.keys():
                if k not in template:
                    raise PackagingError(f"Workflow packaging (STRICT): unexpected key {k!r}")
            for k in schema_keys:
                if k in mapping:
                    template[k] = mapping[k]
            return template

        if self._mapping_policy == MappingPolicy.IGNORE_EXTRA:
            for k, v in mapping.items():
                if k in template:
                    template[k] = v
            return template

        if self._mapping_policy in (MappingPolicy.MATCH_FIRST_STRICT, MappingPolicy.MATCH_FIRST_LENIENT):
            if self._mapping_policy == MappingPolicy.MATCH_FIRST_STRICT:
                if len(mapping) > len(schema_keys):
                    raise PackagingError(
                        "Workflow packaging (MATCH_FIRST_STRICT): mapping has more keys "
                        f"({len(mapping)}) than output_schema ({len(schema_keys)})"
                    )

            extras_values: List[Any] = []
            for k, v in mapping.items():
                if k in template:
                    template[k] = v
                else:
                    extras_values.append(v)

            missing_keys = [k for k in schema_keys if template[k] is NO_VAL]

            # If already filled, strict rejects any extra values; lenient ignores them.
            if not missing_keys:
                if extras_values and self._mapping_policy == MappingPolicy.MATCH_FIRST_STRICT:
                    raise PackagingError(
                        "Workflow packaging (MATCH_FIRST_STRICT): output_schema already filled "
                        f"but received {len(extras_values)} extra values"
                    )
                return template

            # Too many extras: strict errors; lenient truncates.
            if len(extras_values) > len(missing_keys):
                if self._mapping_policy == MappingPolicy.MATCH_FIRST_LENIENT:
                    extras_values = extras_values[: len(missing_keys)]
                else:
                    raise PackagingError(
                        "Workflow packaging (MATCH_FIRST_*): too many extra values "
                        f"({len(extras_values)}) for remaining schema slots ({len(missing_keys)})"
                    )

            for idx, v in enumerate(extras_values):
                template[missing_keys[idx]] = v

            return template

        raise PackagingError(f"Unknown mapping policy: {self._mapping_policy!r}")

    def _package_from_sequence(
        self,
        template: OrderedDict[str, Any],
        seq: ABCSequence[Any],
    ) -> OrderedDict[str, Any]:
        keys = list(template.keys())
        values = list(seq)

        # If the sequence fits, positional-fill and return.
        if len(values) <= len(keys):
            for i, v in enumerate(values):
                template[keys[i]] = v
            return template

        # Overflow: policy decides whether we truncate or raise.
        if self._mapping_policy in (MappingPolicy.IGNORE_EXTRA, MappingPolicy.MATCH_FIRST_LENIENT):
            # Ignore extras by truncating to schema length.
            for i, k in enumerate(keys):
                template[k] = values[i]
            return template

        # STRICT and MATCH_FIRST_STRICT both reject overflow for sequences.
        raise PackagingError(
            "Workflow packaging (SEQUENCE): too many values "
            f"({len(values)}) for output_schema ({len(keys)}) under policy {self._mapping_policy.value}"
        )

    def _package_from_scalar(
        self,
        template: OrderedDict[str, Any],
        value: Any,
    ) -> OrderedDict[str, Any]:
        keys = list(template.keys())
        template[keys[0]] = value
        return template

    # ------------------------------------------------------------------ #
    # Schema helpers
    # ------------------------------------------------------------------ #
    def _set_io_schemas(
        self,
        *,
        arguments_map: ArgumentMap,
        output_schema: Union[List[str], Mapping[str, Any]],
    ) -> None:
        """
        Normalize and set:
          - _arguments_map (ordered, meta dicts copied)
          - _input_schema  (defaults from arguments_map else NO_VAL)
          - _output_schema (normalized output template)

        Intentionally protected: callers should be Workflow subclasses.
        """
        input_schema: OrderedDict[str, Any] = OrderedDict()

        for name, meta in arguments_map.items():
            input_schema[name] = meta.get("default", default = NO_VAL)

        normalized_out = self._normalize_output_schema(output_schema)
        if len(normalized_out) == 0:
            raise SchemaError("Workflow.output_schema must contain at least one key")

        self._arguments_map = arguments_map
        self._input_schema = input_schema
        self._output_schema = normalized_out

    def _normalize_output_schema(
        self, schema: Union[List[str], Mapping[str, Any]]
    ) -> OrderedDict[str, Any]:
        if isinstance(schema, list):
            if not schema:
                raise SchemaError("output_schema list must be non-empty")
            out = OrderedDict()
            for k in schema:
                if not isinstance(k, str) or not k:
                    raise SchemaError("output_schema list keys must be non-empty strings")
                if k in out:
                    raise SchemaError(f"duplicate output_schema key: {k!r}")
                out[k] = NO_VAL
            return out

        if isinstance(schema, ABCMapping):
            if not schema:
                raise SchemaError("output_schema mapping must be non-empty")
            out = OrderedDict()
            for k, default in schema.items():
                if not isinstance(k, str) or not k:
                    raise SchemaError("output_schema mapping keys must be non-empty strings")
                if k in out:
                    raise SchemaError(f"duplicate output_schema key: {k!r}")
                out[k] = default
            return out

        raise SchemaError(
            "output_schema must be a list[str] or mapping[str, Any]; "
            f"got {type(schema)!r}"
        )

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #
    def to_dict(self) -> OrderedDict[str, Any]:
        return OrderedDict(
            workflow_type=type(self).__name__,
            name=self._name,
            description=self._description,
            arguments_map=self.arguments_map,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            bundling_policy=self._bundling_policy.value,
            mapping_policy=self._mapping_policy.value,
        )
