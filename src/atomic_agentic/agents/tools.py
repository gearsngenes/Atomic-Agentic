from __future__ import annotations

import builtins
from typing import Any

from ..tools import Tool
from ..constants.agents import (
    ATTR_CALL_ALIAS,
    DUNDER_ATTRIBUTE_PATTERN,
    EXCLUDED_PY_BUILTINS,
    PY_BUILTIN_ALIAS,
    RETURN_TOOL_DESCRIPTION,
    RETURN_TOOL_FULL_NAME,
    RETURN_TOOL_NAME,
    RETURN_TOOL_NAMESPACE,
    RETURN_VALUE_FIELD,
)

__all__ = [
    "identity_pre_tool",
    "identity_post_tool",
    "return_tool",
    "builtin_call_tool",
    "attr_call_tool",
    "call_python_builtin",
]


def identity_pre(*, prompt: str) -> str:
    """
    Default pre-invoke identity function.

    Requires exactly ``{"prompt": str}`` and returns the prompt string
    unchanged. Wrapped as a Tool and used when no explicit ``pre_invoke``
    Tool is provided to ``Agent``.
    """
    if not isinstance(prompt, str):
        raise ValueError("prompt must be a string")
    return prompt

identity_pre_tool = Tool(
    function=identity_pre,
    name="identity_pre",
    namespace="base_agent",
    description="Default pre-invoke identity function that requires {'prompt': str} and returns the prompt string.",
)


def identity_post(*, result: Any) -> Any:
    """
    Default post-invoke identity function.

    This function accepts a single argument named ``result`` and returns it
    unchanged. It is wrapped as a Tool and used when no explicit ``post_invoke``
    Tool is provided.
    """
    return result

identity_post_tool = Tool(
    function=identity_post,
    name="identity_post",
    namespace="base_agent",
    description="Default post-invoke identity function that accepts a single argument 'result' and returns it unchanged.",
)


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


def _call_py_builtin(name: str, args: tuple, kwargs: dict) -> Any:
    """
    Dispatch body for ScriptAgent's approved-Python-builtin calls.

    ``args``/``kwargs`` are the real target call's own positional/keyword
    arguments, passed as opaque packed values (a tuple and a dict) rather
    than splatted into this function's own parameter list -- a real call's
    keyword argument named ``name`` would otherwise collide with this
    dispatcher's own ``name`` parameter during ``Tool``-level binding
    (the class this collision could produce was a spurious ``TypeError``
    on an otherwise legitimate call). Unpacked only internally, right at
    the real invocation.

    Raises ``ValueError`` for an excluded or nonexistent builtin name --
    the authoritative runtime gate; ``utils/script.py``'s
    ``rewrite_builtin_calls`` already guarantees this can't happen for a
    slot it rewrote, but this check doesn't trust that upstream guarantee.
    """
    if name in EXCLUDED_PY_BUILTINS or not hasattr(builtins, name):
        raise ValueError(f"python builtin {name!r} is not available here")
    fn = getattr(builtins, name)
    return fn(*args, **kwargs)


# Never registered into any agent's toolbox -- resolved directly by
# ScriptAgent.prepare()/_gather_batch_results() via the PY_BUILTIN_ALIAS
# sentinel, never through get_tool(). No return_tool-style identity assert
# needed: nothing references this Tool by a full_name string, only by
# direct object reference from agents/script.py.
builtin_call_tool = Tool(
    function=_call_py_builtin,
    name=PY_BUILTIN_ALIAS,
    namespace="script_agent",
    description=(
        "Internal ScriptAgent dispatcher for approved Python builtin calls. "
        "Never registered into any agent's toolbox -- resolved directly by "
        "ScriptAgent.prepare()/_gather_batch_results() via the "
        "PY_BUILTIN_ALIAS sentinel, never through get_tool()."
    ),
)


def _call_attr_method(obj: Any, method_name: str, args: tuple, kwargs: dict) -> Any:
    """
    Dispatch body for ScriptAgent's attribute/method-call slots
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

    Raises ``ValueError`` for a dunder method name -- the authoritative
    runtime gate, matching ``_call_py_builtin``'s own posture: dunder names
    are already rejected at parse time (``utils/script.py``'s
    ``_hoist_calls``/``_build_call_slot`` checks), but this is the one
    remaining sandbox-escape surface in the whole grammar, so this check
    doesn't trust that upstream guarantee either.
    """
    if DUNDER_ATTRIBUTE_PATTERN.fullmatch(method_name):
        raise ValueError(f"attribute/method name {method_name!r} is not available here")
    return getattr(obj, method_name)(*args, **kwargs)


# Never registered into any agent's toolbox -- resolved directly by
# ScriptAgent.prepare()/_gather_batch_results() via the ATTR_CALL_ALIAS
# sentinel, never through get_tool(). Same treatment as builtin_call_tool.
attr_call_tool = Tool(
    function=_call_attr_method,
    name=ATTR_CALL_ALIAS,
    namespace="script_agent",
    description=(
        "Internal ScriptAgent dispatcher for attribute/method calls on a "
        "value the plan already holds. Never registered into any agent's "
        "toolbox -- resolved directly by ScriptAgent.prepare()/"
        "_gather_batch_results() via the ATTR_CALL_ALIAS sentinel, never "
        "through get_tool()."
    ),
)


def call_python_builtin(name: str, *args: Any, **kwargs: Any) -> Any:
    """
    DagAgent's sanctioned computation escape hatch -- its value-expression
    grammar permits no function/method calls at all (see
    utils.agents.reject_unsupported_forms(forbid_calls=True)), so real
    computation happens here instead: a normal, budgeted, plan-visible
    dispatched call, never smuggled into an expression value.

    Unlike ScriptAgent's builtin_call_tool/attr_call_tool, this is a plain,
    ordinary Python function -- not a manually constructed Tool, and not
    resolved through a bypass sentinel. DagAgent.__init__ auto-registers it
    on every instance via the normal register_tool(...) path (toolify()'d,
    dispatched through the ordinary get_tool()/tool.invoke() path like any
    other registered tool).

    A plain *args/**kwargs splat is safe here -- deliberately not the
    packed-tuple/dict signature builtin_call_tool/attr_call_tool's
    dispatch bodies use. Those needed packing because their own fixed
    identifying parameters (name+args+kwargs as one unit; obj/method_name)
    sit in the same positional/keyword namespace as the real wrapped
    call's own arguments, and a wrapped call's own keyword genuinely could
    collide with "obj"/"method_name" in ordinary usage. This function has
    only one fixed leading parameter (name, always the wire schema's first
    positional argument), and no real Python builtin's own call needs a
    keyword literally named "name" passed through it, so the same
    collision risk doesn't apply.
    """
    if name in EXCLUDED_PY_BUILTINS or not hasattr(builtins, name):
        raise ValueError(f"python builtin {name!r} is not available here")
    fn = getattr(builtins, name)
    return fn(*args, **kwargs)
