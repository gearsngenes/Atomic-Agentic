from __future__ import annotations

import builtins
from typing import Any

from ..tools import Tool
from ..constants.agents import (
    ALLOWED_ATTR_METHODS,
    ALLOWED_PY_BUILTINS,
    ATTR_CALL_ALIAS,
    PY_BUILTIN_ALIAS,
)

__all__ = [
    "builtin_call_tool",
    "attr_call_tool",
    "call_python_builtin",
]


def _reject_callable_arguments(args: tuple, kwargs: dict) -> None:
    """
    Reject a callable argument at a dispatcher. A bound method read off a
    value (``s.format``) is a plain attribute read, so it can reach a
    callable-taking builtin or method (``map``, ``sorted(key=...)``) and be
    invoked there, bypassing the per-class method table. Nothing legitimate
    passes a callable: the argument eval has no builtins, so ``str``/``int``
    can't even be named.
    """
    for value in (*args, *kwargs.values()):
        if callable(value):
            raise ValueError("callable arguments are not available here")


def call_python_builtin(name: str, args: tuple, kwargs: dict) -> Any:
    """
    Dispatch body for PlanActAgent's approved-Python-builtin calls.

    ``args``/``kwargs`` are the real target call's own positional/keyword
    arguments, passed as opaque packed values (a tuple and a dict) rather
    than splatted into this function's own parameter list -- a real call's
    keyword argument named ``name`` would otherwise collide with this
    dispatcher's own ``name`` parameter during ``Tool``-level binding
    (the class this collision could produce was a spurious ``TypeError``
    on an otherwise legitimate call). Unpacked only internally, right at
    the real invocation.

    Raises ``ValueError`` for a name outside ``ALLOWED_PY_BUILTINS`` --
    the authoritative runtime gate; ``utils/script.py``'s
    ``_build_call_step`` already guarantees this can't happen for a
    step it built, but this check doesn't trust that upstream guarantee.
    """
    if name not in ALLOWED_PY_BUILTINS:
        raise ValueError(f"python builtin {name!r} is not available here")
    if name == "type" and (len(args) != 1 or kwargs):
        raise ValueError("type() takes exactly one positional argument here")
    _reject_callable_arguments(args, kwargs)
    fn = getattr(builtins, name)
    return fn(*args, **kwargs)


# Never registered into any agent's toolbox -- resolved directly by
# PlanActAgent.prepare()/_gather_batch_results() via the PY_BUILTIN_ALIAS
# sentinel, never through get_tool(). Nothing references this Tool by a
# full_name string, only by direct object reference from agents/planact.py.
builtin_call_tool = Tool(
    function=call_python_builtin,
    name=PY_BUILTIN_ALIAS,
    description=(
        "Internal PlanActAgent dispatcher for approved Python builtin calls. "
        "Never registered into any agent's toolbox -- resolved directly by "
        "PlanActAgent.prepare()/_gather_batch_results() via the "
        "PY_BUILTIN_ALIAS sentinel, never through get_tool()."
    ),
)


def _call_attr_method(obj: Any, method_name: str, args: tuple, kwargs: dict) -> Any:
    """
    Dispatch body for PlanActAgent's attribute/method-call steps
    (``obj.method(...)``).

    ``args``/``kwargs`` are the real target method's own positional/keyword
    arguments, passed as opaque packed values (a tuple and a dict) rather
    than splatted into this function's own parameter list -- a real call's
    keyword argument named ``obj``/``method_name`` would otherwise collide
    with this dispatcher's own same-named parameters during ``Tool``-level
    binding (e.g. ``node.attach(obj=child)`` would raise a spurious
    "multiple values for argument 'obj'" ``TypeError`` on an otherwise
    legitimate call). Unpacked only internally, right at the real
    invocation.

    Raises ``ValueError`` unless the receiver's most specific class (walking
    its MRO, so a subclass resolves to its nearest table class) is in
    ``ALLOWED_ATTR_METHODS`` and lists ``method_name`` -- the authoritative
    runtime gate; it doesn't trust that parse time (``utils/script.py``)
    already rejected unknown or dunder names. The method is called off that
    table class (``str.strip(obj, ...)``), never off the instance, so a
    subclass override or instance attribute can't change what runs.
    """
    for cls in type(obj).__mro__:
        allowed = ALLOWED_ATTR_METHODS.get(cls)
        if allowed is not None:
            break
    else:
        raise ValueError(f"method calls on {type(obj).__name__!r} values are not available here")
    if method_name not in allowed:
        raise ValueError(f"method {cls.__name__}.{method_name} is not available here")
    _reject_callable_arguments(args, kwargs)
    return getattr(cls, method_name)(obj, *args, **kwargs)


# Never registered into any agent's toolbox -- resolved directly by
# PlanActAgent.prepare()/_gather_batch_results() via the ATTR_CALL_ALIAS
# sentinel, never through get_tool(). Same treatment as builtin_call_tool.
attr_call_tool = Tool(
    function=_call_attr_method,
    name=ATTR_CALL_ALIAS,
    description=(
        "Internal PlanActAgent dispatcher for attribute/method calls on a "
        "value the plan already holds. Never registered into any agent's "
        "toolbox -- resolved directly by PlanActAgent.prepare()/"
        "_gather_batch_results() via the ATTR_CALL_ALIAS sentinel, never "
        "through get_tool()."
    ),
)
