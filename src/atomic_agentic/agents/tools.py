from __future__ import annotations

import builtins
from typing import Any, Literal

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
    "make_sequence",
    "make_dict",
    "get_item",
]


def identity_pre(prompt: str) -> str:
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


def identity_post(result: Any) -> Any:
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
    Call a Python builtin function by name and return its result.

    "name" must be the exact name of an available Python builtin (e.g.
    "len", "round", "sorted") -- never free-form text, and never another
    expression to evaluate. Any remaining positional and keyword arguments
    are passed through to that builtin exactly as given.

    Raises ValueError if "name" is not the name of an available builtin.
    """
    # A plain *args/**kwargs splat is safe here -- deliberately not the
    # packed-tuple/dict signature ScriptAgent's own builtin_call_tool/
    # attr_call_tool dispatch bodies use. Those needed packing because their
    # own fixed identifying parameters (name+args+kwargs as one unit;
    # obj/method_name) sit in the same positional/keyword namespace as the
    # real wrapped call's own arguments, and a wrapped call's own keyword
    # genuinely could collide with "obj"/"method_name" in ordinary usage.
    # This function has only one fixed leading parameter (name), and no
    # real Python builtin's own call needs a keyword literally named "name"
    # passed through it, so the same collision risk doesn't apply.
    if name in EXCLUDED_PY_BUILTINS or not hasattr(builtins, name):
        raise ValueError(f"python builtin {name!r} is not available here")
    fn = getattr(builtins, name)
    return fn(*args, **kwargs)


def make_sequence(*items: Any, kind: Literal["list", "tuple", "set"]) -> list | tuple | set:
    """
    Build a list, tuple, or set from the given items.

    "kind" selects the container type -- exactly "list", "tuple", or "set"
    (case-sensitive) -- and must always be given as its own keyword
    argument. Each remaining value becomes one element of the container,
    in the order given, and must always be given positionally, never as a
    keyword argument.

    Raises ValueError if "kind" is not one of the three allowed values.
    """
    # kind is deliberately keyword-only (*items precedes it) -- live
    # cross-provider smoke testing (OpenAI and Anthropic, independently, in
    # different concrete ways) confirmed a (kind, *items) ordering is a real
    # footgun, not just a theoretical one: naming kind by its own parameter
    # name while leaving items positional collides under Python's own
    # calling convention (the first positional value binds to kind, by
    # left-to-right declared position, before the explicit keyword is ever
    # applied) -- TypeError: got multiple values for argument 'kind'. Making
    # kind keyword-only removes the ambiguity structurally: items can only
    # ever be positional, kind can only ever be a keyword.
    if kind == "list":
        return list(items)
    elif kind == "tuple":
        return tuple(items)
    elif kind == "set":
        return set(items)
    else:
        raise ValueError(
            f"make_sequence: kind must be 'list', 'tuple', or 'set'; got {kind!r}."
        )


def make_dict(**pairs: Any) -> dict:
    """
    Build a dict from the given keyword arguments.

    Each keyword argument becomes one dict entry -- the keyword is the
    key, its value is that key's value -- in the order given.
    """
    # Split into its own tool along calling-convention lines (**kwargs, not
    # *args) rather than folding into one make_collection(kind, *items,
    # **pairs) tool, since a single tool whose correct calling convention
    # depends on a runtime kind value would reintroduce exactly the kind of
    # implicit-contract ambiguity make_sequence's own kind/items ordering
    # bug demonstrated live.
    return dict(**pairs)


def get_item(container: Any, key: Any) -> Any:
    """
    Return one element from a container -- container[key].

    For a list or tuple, "key" is an integer index. For a dict, "key" is
    one of its keys. (A set has no positional or key-based access at all,
    so it cannot be used here.)

    Raises whatever error the container itself raises for an invalid key
    or index (e.g. IndexError, KeyError, TypeError).
    """
    # No defensive wrapping -- a KeyError/IndexError/TypeError this raises
    # naturally surfaces as-is, per this codebase's "let natural exceptions
    # surface" discipline. No get_attr sibling tool exists (kept the tool
    # surface smaller; revisit only if live testing shows a real need) --
    # get_item alone covers both list/tuple-by-index and dict-by-key access,
    # the two shapes make_sequence/make_dict actually produce.
    return container[key]
