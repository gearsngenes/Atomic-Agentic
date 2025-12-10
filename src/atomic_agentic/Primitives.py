from __future__ import annotations

import logging
import os
import random
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Mapping, Optional, get_origin, get_args
from .Exceptions import LLMEngineError, ToolDefinitionError, ToolInvocationError
from typing import Mapping, Callable, Optional
from collections import OrderedDict
import inspect

logger = logging.getLogger(__name__)

__all__ = [
    "LLMEngine",
]


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

    # --------------------------------------------------------------------- #
    # Public surface
    # --------------------------------------------------------------------- #

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

    # Attach / detach ----------------------------------------------------- #

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

    # Template `invoke` --------------------------------------------------- #

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
        start = time.time()
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
        finally:
            duration = time.time() - start
            logger.debug(
                "LLMEngine %s.invoke completed in %.3fs", self._name, duration
            )

    # --------------------------------------------------------------------- #
    # Shared helpers used by the template
    # --------------------------------------------------------------------- #

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
    # Abstract hooks for subclasses
    # --------------------------------------------------------------------- #

    @abstractmethod
    def _build_provider_payload(
        self,
        messages: List[Dict[str, str]],
        attachments: Mapping[str, Mapping[str, Any]],
    ) -> Any:
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

    def _on_detach(self, meta: Mapping[str, Any]) -> None:
        """
        Optional hook called when an attachment is detached.

        Subclasses may implement provider-specific cleanup (e.g. remote delete).
        Default implementation is a no-op.
        """
        # Intentionally a no-op by default.
        return None

    # --------------------------------------------------------------------- #
    # Introspection
    # --------------------------------------------------------------------- #

    def to_dict(self) -> Dict[str, Any]:
        """
        Shallow, non-secret configuration snapshot for debugging / logging.
        """
        return {
            "name": self._name,
            "timeout_seconds": self._timeout_seconds,
            "max_retries": self._max_retries,
            "attachments": list(self._attachments.keys()),
            "provider": type(self).__name__,
        }


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

    @property
    def namespace(self) -> str:
        return self._namespace

    @property
    def description(self) -> str:
        return self._description

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
                default_marker = f" = {str(meta["default"])}"
            if kind == "VAR_POSITIONAL":
                param_str = f"*{name}: {type_name}{default_marker}"
            elif kind == "VAR_KEYWORD":
                param_str = f"**{name}: {type_name}{default_marker}"
            else:
                param_str = f"{name}: {type_name}{default_marker}"
            params.append(param_str)
        params_str = ", ".join(params)
        return f"{self.full_name}({params_str}) -> {self._return_type}"

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"{self.signature}: {self._description}"

    __str__ = __repr__

    # ------------------------------------------------------------------ #
    # Template method
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
        return self.execute(args, kwargs)

    # ------------------------------------------------------------------ #
    # Overridable helpers
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

    # ------------------------------------------------------------------ #
    # Dict → (*args, **kwargs) mapping
    # ------------------------------------------------------------------ #

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

    # ------------------------------------------------------------------ #
    # Execution
    # ------------------------------------------------------------------ #

    def execute(self, args: tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        """Execute the underlying callable.

        Subclasses may override this to change *how* a tool is executed (for
        example, by making a remote MCP call or invoking an Agent), but should
        not change the high-level semantics.
        """
        try:
            return self._function(*args, **kwargs)
        except ToolInvocationError:
            # Allow explicit ToolInvocationError to propagate unchanged.
            raise
        except Exception as e:  # pragma: no cover - thin wrapper
            raise ToolInvocationError(f"{self._name}: invocation failed: {e}") from e

    # ------------------------------------------------------------------ #
    # Introspection / serialization
    # ------------------------------------------------------------------ #

    def to_dict(self) -> Dict[str, Any]:
        """Serialize this tool's header and argument schema.

        Note that deserialisation is handled by :mod:`Factory`; this method does
        *not* perform any persistibility checks and will not raise solely
        because :attr:`is_persistible` is ``False``.
        """
        return {
            "tool_type": type(self).__name__,
            "name": self._name,
            "namespace": self._namespace,
            "description": self._description,
            "signature": self.signature,
            "module": self._module,
            "qualname": self._qualname,
            "is_persistible": self.is_persistible,
        }
