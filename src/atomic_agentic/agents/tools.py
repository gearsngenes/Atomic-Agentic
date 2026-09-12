from __future__ import annotations

import builtins
from typing import Any

from ..tools import Tool
from ..constants.agents import (
    ATTR_CALL_ALIAS,
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


def _call_py_builtin(name: str, *args: Any, **kwargs: Any) -> Any:
    """
    Dispatch body for ScriptAgent's approved-Python-builtin calls.

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


def _call_attr_method(obj: Any, method_name: str, *args: Any, **kwargs: Any) -> Any:
    """
    Dispatch body for ScriptAgent's attribute/method-call slots
    (``obj.method(...)``). Dunder method names are already rejected at
    parse time (``utils/script.py``'s ``_hoist_calls``/``_build_call_slot``
    checks) -- this does not re-check, mirroring ``_call_py_builtin``'s own
    split between the authoritative runtime gate and an upstream guarantee.
    """
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
