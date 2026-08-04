from __future__ import annotations

from typing import Any

from ..tools import Tool
from ..constants.agents import (
    RETURN_TOOL_DESCRIPTION,
    RETURN_TOOL_FULL_NAME,
    RETURN_TOOL_NAME,
    RETURN_TOOL_NAMESPACE,
    RETURN_VALUE_FIELD,
)

__all__ = ["identity_pre_tool", "identity_post_tool", "return_tool"]


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
