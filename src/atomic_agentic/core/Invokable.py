from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any, Mapping, Dict
import re

ArgumentMap = OrderedDict[str, dict[str, Any]]

# Valid name: like a Python identifier (letters, digits, underscores), must not start with a digit
_VALID_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class AtomicInvokable(ABC):
    """
    Basal invokable contract for Tools / Agents / Workflows.

    Requires:
        - name (valid, non-empty)
        - description (non-empty)
        - build_args_returns() -> (ArgumentMap, return_type)
        - invoke(inputs)

    Provides:
        - properties with safe setters
        - unified __repr__ / __str__
        - default metadata .to_dict()
    """

    def __init__(
        self,
        *,
        name: str,
        description: str,
    ) -> None:
        # setters include validation
        self.name = name
        self.description = description

        args, ret = self.build_args_returns()

        if not isinstance(args, OrderedDict):
            raise TypeError(
                f"{type(self).__name__}.build_args_returns must return OrderedDict"
            )
        if not isinstance(ret, str):
            raise TypeError(
                f"{type(self).__name__}.build_args_returns must return str for return_type"
            )

        self._arguments_map: ArgumentMap = args
        self._return_type: str = ret
        self._is_persistible: bool = self._compute_is_persistible()

    # ---------------------------------------------------------------- #
    # Name + description with validation
    # ---------------------------------------------------------------- #
    @property
    def name(self) -> str:
        """The canonical name of this invokable."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("name must be a non-empty string")
        if not _VALID_NAME.match(value):
            raise ValueError(
                f"name must be alphanumeric/underscore and not start with a digit; got {value!r}"
            )
        self._name = value

    @property
    def description(self) -> str:
        """Human-friendly description."""
        return self._description

    @description.setter
    def description(self, value: str) -> None:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("description must be a non-empty string")
        self._description = value.strip()

    @property
    def is_persistible(self) -> bool:
        self._is_persistible
    
    # ---------------------------------------------------------------- #
    # Exposed read-only derived properties
    # ---------------------------------------------------------------- #
    @property
    def arguments_map(self) -> ArgumentMap:
        """Ordered arg metadata produced by build_args_returns()."""
        return self._arguments_map

    @property
    def return_type(self) -> str:
        """Return type (string) produced by build_args_returns()."""
        return self._return_type

    # ---------------------------------------------------------------- #
    # Signature helper
    # ---------------------------------------------------------------- #
    @property
    def signature(self) -> str:
        """
        Returns a signature like:

            name(arg1: Type, arg2: Type = default) -> return_type
        """
        params = []
        for param, meta in self.arguments_map.items():
            ptype = meta.get("type", "Any")
            default = meta.get("default")
            if default is not None:
                params.append(f"{param}: {ptype} = {default!r}")
            else:
                params.append(f"{param}: {ptype}")
        params_str = ", ".join(params)
        return f"{type(self).__name__}.{self.name}({params_str}) -> {self.return_type}"

    # ---------------------------------------------------------------- #
    # Abstract contract
    # ---------------------------------------------------------------- #
    @abstractmethod
    def _compute_is_persistible(self) -> bool:
        raise NotImplementedError
    
    @abstractmethod
    def build_args_returns(self) -> tuple[ArgumentMap, str]:
        """Return (arguments_map, return_type)."""
        raise NotImplementedError

    @abstractmethod
    def invoke(self, inputs: Mapping[str, Any]) -> Any:
        """Perform work."""
        raise NotImplementedError

    # ---------------------------------------------------------------- #
    # Default metadata serialization
    # ---------------------------------------------------------------- #
    def to_dict(self) -> Dict[str, Any]:
        """
        Minimal metadata serialization.

        Does *not* include `signature` by default, per request.
        """
        return {
            "type": type(self).__name__,
            "name": self.name,
            "description": self.description,
            "arguments_map": self.arguments_map,
            "return_type": self.return_type,
        }

    # ---------------------------------------------------------------- #
    # Unified repr/str
    # ---------------------------------------------------------------- #
    def __str__(self) -> str:
        return f"<{self.signature} - {self.description}>"

    def __repr__(self) -> str:
        return f"{self.signature}: {self.description}"
