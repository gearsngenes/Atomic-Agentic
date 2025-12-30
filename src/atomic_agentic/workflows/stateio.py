from __future__ import annotations
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Type,
    Tuple,
    Union,
    get_type_hints,
)

from ..core.Exceptions import SchemaError
from ..core.Invokable import AtomicInvokable
from .base import (
    BundlingPolicy,
    MappingPolicy,
    AbsentValPolicy,
)
from .basic import BasicFlow

#------------------------------------------------------------------------------#
# Stateful Graph Node Wrapper
#------------------------------------------------------------------------------#
def _is_typed_dict_class(obj: Any) -> bool:
    """
    Runtime check for TypedDict classes.
    TypedDict classes in runtime have .__annotations__ and .__total__ attributes.
    """
    return isinstance(obj, type) and issubclass(obj, dict) and hasattr(obj, "__annotations__") and hasattr(obj, "__total__")

def _extract_state_keys(state_schema: Union[Type[Any], List[str],]) -> List[str]:
    """
    Normalize a state_schema into a list of state keys.
    Accepts:
      - TypedDict subclass => extract keys from __annotations__
      - list[str] => use list directly

    Raises SchemaError for unsupported types.
    """
    if isinstance(state_schema, list):
        # list of keys
        return list(state_schema)

    if _is_typed_dict_class(state_schema):
        # TypedDict: use get_type_hints to resolve forward refs
        return list(get_type_hints(state_schema).keys())

    raise SchemaError(
        f"state_schema must be TypedDict subclass, list[str], or mapping; got {state_schema!r}"
    )

class StateIOFlow(BasicFlow):
    """
    A specialized workflow intended for stateful graph nodes (e.g., LangGraph).
    It treats the component as a node that takes a state dict and emits a state update dict.
    Missing output values are always dropped (AbsentValPolicy.DROP).

    The provided `state_schema` defines the universe of keys the node can handle
    for both input and output.

    The component's *declared* input keys (excluding VAR_POSITIONAL/VAR_KEYWORD)
    must be a subset of these state keys.
    """
    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        component: AtomicInvokable,
        *,
        state_schema: Union[Type[Any], List[str]],
        mapping_policy: MappingPolicy = MappingPolicy.STRICT,
        bundling_policy: BundlingPolicy = BundlingPolicy.UNBUNDLE,
    ):
        # Standardize the expected in/out state schema
        self._state_schema:List[str] = _extract_state_keys(state_schema)
        
        # Check component
        # Compute "real" input keys from arguments_map (exclude VAR_* parameters)
        self.component = component

        # fill in a basic state schema
        super().__init__(
            component=component,
            output_schema=self._state_schema,
            mapping_policy=mapping_policy,
            bundling_policy=bundling_policy,
            absent_val_policy=AbsentValPolicy.DROP,
        )

    # ------------------------------------------------------------------ #
    # Workflow Properties
    # ------------------------------------------------------------------ #
    @property
    def absent_val_policy(self) -> AbsentValPolicy:
        return AbsentValPolicy.DROP

    @property
    def output_schema(self) -> Mapping[str, Any]:
        return dict(self._output_schema)

    # ------------------------------------------------------------------ #
    # BasicFlow Properties
    # ------------------------------------------------------------------ #
    @property
    def component(self) -> AtomicInvokable:
        return self._component
    
    @component.setter
    def component(self, candidate: AtomicInvokable) -> None:
        unknown_inputs = set(candidate.arguments_map.keys()).difference(self.state_schema)
        for unknown in unknown_inputs:
            if candidate.arguments_map[unknown].get("kind") not in {"VAR_POSITIONAL", "VAR_KEYWORD"}:
                raise SchemaError(
                    f"A non-arg/kwarg variable was detected in {self.name}'s component that "
                    f"doesn't exist in the expected schema {self._state_schema}: {unknown}")
        self._component = candidate
        self._arguments_map, self._return_type = self.build_args_returns()
        self._is_persistible = self._compute_is_persistible()

    # ------------------------------------------------------------------ #
    # StateIOFlow Properties
    # ------------------------------------------------------------------ #
    @property
    def state_schema(self) -> Tuple[str, ...]:
        return self._state_schema

    # ------------------------------------------------------------------ #
    # StateIOFlow helpers
    # ------------------------------------------------------------------ #
    def _invoke(self, inputs: Mapping[str, Any]) -> Tuple[Mapping[str, Any], Any]:
        """
        Filter the incoming state down to the component's declared non-var inputs,
        then invoke the wrapped component.
        """
        # if has *args, **kwargs, then accept whole state
        if any(key not in self.state_schema for key in inputs):
            raise ValueError(f"Expected input keys to be a subset of {self.state_schema}, but got "
                             f"the following keys, instead: {inputs.keys()}")
        if any(meta["kind"] in {"VAR_POSITIONAL", "VAR_KEYWORD"} for meta in self.arguments_map.values()):
            raw = self.component.invoke(inputs)
            unused_inputs = {}
        #otherwise filtered
        else:
            filtered_inputs = {k: inputs[k] for k in self.arguments_map.keys() if k in inputs}
            unused_keys = set(filtered_inputs.keys()).difference(self.state_schema)
            unused_inputs = {k:inputs[k] for k in unused_keys}
            raw = self.component.invoke(filtered_inputs)
        return {"unused_inputs": unused_inputs}, raw
    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #
    def to_dict(self)-> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            "state_schema": self._state_schema
        })
        return d