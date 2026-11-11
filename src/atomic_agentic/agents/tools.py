from __future__ import annotations

import builtins
from typing import Any, Literal

from ..tools import Tool
from ..constants.agents import (
    ALLOWED_ATTR_METHODS,
    ALLOWED_PY_BUILTINS,
    ATTR_CALL_ALIAS,
    PY_BUILTIN_ALIAS,
    RETURN_TOOL_DESCRIPTION,
    RETURN_TOOL_FULL_NAME,
    RETURN_TOOL_NAME,
    RETURN_TOOL_NAMESPACE,
    RETURN_VALUE_FIELD,
)

__all__ = [
    "return_tool",
    "builtin_call_tool",
    "attr_call_tool",
    "call_python_builtin",
    "make_sequence",
    "make_dict",
]


def _return(val: Any) -> Any:
    return val


# The executable Tool instance lives in this module; RETURN_TOOL_FULL_NAME is
# used for canonical runtime identity checks.
return_tool = Tool(
    function=_return,
    name=RETURN_TOOL_NAME,
    namespace=RETURN_TOOL_NAMESPACE,
    description=RETURN_TOOL_DESCRIPTION,
)

if return_tool.full_name != RETURN_TOOL_FULL_NAME:
    raise RuntimeError(
        f"return_tool.full_name mismatch: expected {RETURN_TOOL_FULL_NAME!r}, "
        f"got {return_tool.full_name!r}."
    )

_return_param_names = [spec.name for spec in return_tool.parameters]
if _return_param_names != [RETURN_VALUE_FIELD]:
    raise RuntimeError(
        f"return_tool parameter mismatch: expected {[RETURN_VALUE_FIELD]!r}, "
        f"got {_return_param_names!r}."
    )


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
    Dispatch body for ScriptActAgent's approved-Python-builtin calls.

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
    ``rewrite_builtin_calls`` already guarantees this can't happen for a
    slot it rewrote, but this check doesn't trust that upstream guarantee.
    """
    if name not in ALLOWED_PY_BUILTINS:
        raise ValueError(f"python builtin {name!r} is not available here")
    if name == "type" and (len(args) != 1 or kwargs):
        raise ValueError("type() takes exactly one positional argument here")
    _reject_callable_arguments(args, kwargs)
    fn = getattr(builtins, name)
    return fn(*args, **kwargs)


# Never registered into any agent's toolbox -- resolved directly by
# ScriptActAgent.prepare()/_gather_batch_results() via the PY_BUILTIN_ALIAS
# sentinel, never through get_tool(). No return_tool-style identity assert
# needed: nothing references this Tool by a full_name string, only by
# direct object reference from agents/scriptact.py.
builtin_call_tool = Tool(
    function=call_python_builtin,
    name=PY_BUILTIN_ALIAS,
    namespace="scriptact_agent",
    description=(
        "Internal ScriptActAgent dispatcher for approved Python builtin calls. "
        "Never registered into any agent's toolbox -- resolved directly by "
        "ScriptActAgent.prepare()/_gather_batch_results() via the "
        "PY_BUILTIN_ALIAS sentinel, never through get_tool()."
    ),
)


def _call_attr_method(obj: Any, method_name: str, args: tuple, kwargs: dict) -> Any:
    """
    Dispatch body for ScriptActAgent's attribute/method-call slots
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
# ScriptActAgent.prepare()/_gather_batch_results() via the ATTR_CALL_ALIAS
# sentinel, never through get_tool(). Same treatment as builtin_call_tool.
attr_call_tool = Tool(
    function=_call_attr_method,
    name=ATTR_CALL_ALIAS,
    namespace="scriptact_agent",
    description=(
        "Internal ScriptActAgent dispatcher for attribute/method calls on a "
        "value the plan already holds. Never registered into any agent's "
        "toolbox -- resolved directly by ScriptActAgent.prepare()/"
        "_gather_batch_results() via the ATTR_CALL_ALIAS sentinel, never "
        "through get_tool()."
    ),
)


def make_sequence(*items: Any, sequence_type: Literal["list", "tuple", "set"] = "list") -> list | tuple | set:
    """
    Builds a list, tuple, or set from the given items. ALWAYS call this tool if a future
    tool requires one of these container types as input that was not already built by a previous tool call.
    Each item from items becomes one element of the container, in the order given.

    "sequence_type" selects the container type -- exactly "list", "tuple", or "set"
    (case-sensitive) -- and must always be given as its own keyword
    argument. Each remaining value becomes one element of the container,
    in the order given, and must always be given positionally, never as a
    keyword argument.

    Raises ValueError if "sequence_type" is not one of the three allowed values.
    """
    # sequence_type is deliberately keyword-only (*items precedes it): a
    # (sequence_type, *items) ordering is a real footgun -- naming
    # sequence_type by its own parameter name while leaving items
    # positional collides under Python's own calling convention (the first
    # positional value binds to sequence_type, by left-to-right declared
    # position, before the explicit keyword is ever applied) -- TypeError:
    # got multiple values for argument 'sequence_type'. Making sequence_type
    # keyword-only removes the ambiguity structurally: items can only ever
    # be positional, sequence_type can only ever be a keyword.
    if sequence_type == "list":
        return list(items)
    elif sequence_type == "tuple":
        return tuple(items)
    elif sequence_type == "set":
        return set(items)
    else:
        raise ValueError(
            f"make_sequence: sequence_type must be 'list', 'tuple', or 'set'; got {sequence_type!r}."
        )


def make_dict(**pairs: Any) -> dict:
    """
    Builds a dict from the given keyword arguments. ALWAYS call this tool if a future
    tool or return value requires a dict as input that was not already built by a 
    previous tool call. Each pair in pairs contains the key and value for one entry
    in the dict. The keyword is the key, and its value is that key's value. The order
    of the entries in the dict is the order the keyword arguments were given in the call.
    """
    # Split into its own tool along calling-convention lines (**kwargs, not
    # *args) rather than folding into one make_collection(kind, *items,
    # **pairs) tool, since a single tool whose correct calling convention
    # depends on a runtime kind value would reintroduce the same
    # implicit-contract ambiguity make_sequence's own sequence_type/items
    # ordering avoids structurally (see its own comment).
    return dict(**pairs)
