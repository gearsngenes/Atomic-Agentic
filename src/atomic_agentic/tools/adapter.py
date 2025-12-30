from collections import OrderedDict
from typing import (
    Any,
    Callable,
    Dict,
    Mapping)
from .base import Tool, ArgumentMap
from ..core.Invokable import AtomicInvokable
from ..core.Exceptions import ToolInvocationError


# ───────────────────────────────────────────────────────────────────────────────
# Adapter Tool
# ───────────────────────────────────────────────────────────────────────────────
class AdapterTool(Tool):
    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def __init__(self, component: AtomicInvokable, namespace: str|None = None):
        # set private variable
        self._component = component
        # set core attributes
        super().__init__(component.invoke, component.name, namespace, component.description)

    # ------------------------------------------------------------------ #
    # Tool Properties
    # ------------------------------------------------------------------ #
    @property
    def function(self) -> Callable:
        return self._function

    # ------------------------------------------------------------------ #
    # Adapter-Tool Properties
    # ------------------------------------------------------------------ #
    @property
    def component(self) -> AtomicInvokable:
        return self._component
    
    @component.setter
    def component(self, value: AtomicInvokable)-> None:
        self._component = value
        self._function = self._component.invoke
        self._name = self._component.name
        self._description = value.description
        # Identity in import space (may be overridden by subclasses)
        self._module, self._qualname = self._get_mod_qual(self.function)
        # Build argument schema and return type from the current function.
        self._arguments_map, self._return_type = self.build_args_returns()
        # Persistibility flag exposed as a public property.
        self._is_persistible = self._compute_is_persistible()

    # ------------------------------------------------------------------ #
    # Atomic-Invokable Helpers
    # ------------------------------------------------------------------ #
    def build_args_returns(self) -> tuple[ArgumentMap, str]:
        """Construct ``arguments_map`` and ``return_type`` from the wrapped
        callable's signature.

        Rules:
        - If an annotation is present, it *always* defines the type string.
        - If no annotation but a default value exists, the type string is
          derived from ``type(default)``.
        - If neither is present, the type string is 'Any'.
        """
        return self.component.arguments_map, self.component.return_type

    def _compute_is_persistible(self) -> bool:
        """Default persistibility check for callable-based tools.
        Check the component's persistibility
        """
        return self.component.is_persistible

    # ------------------------------------------------------------------ #
    # Tool Helpers
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
        return tuple([]), dict(inputs)

    def execute(self, args: tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        """Execute the underlying callable.

        Subclasses may override this to change *how* a tool is executed (for
        example, by making a remote MCP call or invoking an Agent), but should
        not change the high-level semantics.
        """
        result = self._function(inputs = kwargs)
        return result

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #
    def to_dict(self)-> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "component": self.component.to_dict()
        })
        return base
