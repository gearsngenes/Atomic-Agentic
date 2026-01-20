from __future__ import annotations
from typing import (
    Any,
    Callable,
    Dict,
    Mapping)
from .base import Tool
from ..core.Invokable import AtomicInvokable
from ..core.Parameters import ParamSpec, extract_io
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
        super().__init__(component.invoke,
                         component.name,
                         namespace,
                         component.description)

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
        # Rebuild signature from the component
        parameters, return_type = self._build_tool_signature()
        self._parameters = parameters
        self._return_type = return_type

    # ------------------------------------------------------------------ #
    # Signature Building (Template Method)
    # ------------------------------------------------------------------ #
    def _build_tool_signature(self) -> tuple[list[ParamSpec], str]:
        """Extract signature from the wrapped component.

        For AdapterTool, the signature comes from the component's parameters and return type.
        This allows AdapterTool to expose the wrapped component's interface.
        """
        return self.component.parameters, self.component.return_type

    # ------------------------------------------------------------------ #
    # Tool Helpers
    # ------------------------------------------------------------------ #
    def to_arg_kwarg(self, inputs: Mapping[str, Any]) -> tuple[tuple[Any, ...], Dict[str, Any]]:
        """Map input dict to args/kwargs for the wrapped component.

        For AdapterTool, we only ever pass dictionary inputs to the `.invoke`
        method so we return an empty tuple & the original inputs.
        """
        return tuple([]), dict(inputs)

    def execute(self, args: tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        """Execute the wrapped component.

        The component is invoked via its invoke() method with the kwargs dict.
        The args tuple is ignored since components expect dict-based inputs.
        """
        return self._function(inputs=kwargs)

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #
    def to_dict(self)-> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "component": self.component.to_dict()
        })
        return base
