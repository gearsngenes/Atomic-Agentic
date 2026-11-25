from __future__ import annotations

from typing import Any

from .. import llm
from ..constants.llm import LLM_CLIENT_ROUTES
from ..llm import LLMEngine, LlamaCppEngine
from ..utils.llm import import_client_types

__all__ = ["to_llm"]


def _resolve_engine_class(client: Any) -> type[LLMEngine]:
    """
    Return the engine class for a pre-built SDK client.

    1. For each route in ``LLM_CLIENT_ROUTES``, import its SDK; an SDK that
       cannot be imported is skipped (and remembered for the error).
    2. ``isinstance(client, <the SDK's client classes>)``; the first match
       returns its engine class, so heavy SDKs listed later are never imported
       for a client that matches an earlier one.
    3. No match: ``TypeError`` naming the client type, the supported client
       classes, and the SDKs that were not importable with their install hint.
    """
    unavailable: list[str] = []
    for module_path, class_names, engine_name, package, extra in LLM_CLIENT_ROUTES:
        client_types, error = import_client_types(module_path, class_names)
        if error is not None:
            unavailable.append(f"{package} (pip install atomic-agentic[{extra}])")
            continue
        if isinstance(client, client_types):
            return getattr(llm, engine_name)

    supported = ", ".join(
        f"{module_path}.{class_name}"
        for module_path, class_names, *_ in LLM_CLIENT_ROUTES
        for class_name in class_names
    )
    message = (
        f"to_llm: unsupported client type {type(client).__name__!r}. "
        f"Supported clients: {supported}."
    )
    if unavailable:
        message += f" Not importable here, so not checked: {'; '.join(unavailable)}."
    raise TypeError(message)


def to_llm(
    client: Any,
    model: str | None = None,
    name: str | None = None,
    description: str | None = None,
) -> LLMEngine:
    """
    Build the matching engine from a pre-built SDK client.

    The engine class is chosen by an ``isinstance`` check against each
    importable provider SDK's client classes (``TypeError`` for an unsupported
    client, including ``None``, a ``str`` or an engine; the message lists any
    SDK that could not be imported). ``model`` is required for every client
    except a ``Llama`` client, which already carries its model path and takes
    none. ``name`` and ``description`` fall through (``None`` selects the
    engine default). Settings beyond these four are set on the returned engine
    (``update_runtime_parameters``, attributes).
    """
    engine_cls = _resolve_engine_class(client)

    # A Llama client carries its model path.
    if engine_cls is LlamaCppEngine:
        if model is not None:
            raise TypeError(
                "to_llm: a Llama client already carries its model path; do not pass `model`."
            )
        return engine_cls(name=name, description=description, llm=client)

    if not isinstance(model, str):
        raise TypeError(
            f"to_llm: `model` is required for a {type(client).__name__} client."
        )
    return engine_cls(model, name, description, client)
