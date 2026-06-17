from importlib.metadata import PackageNotFoundError, version

try:  # populated when installed or when a wheel is built
    __version__ = version("atomic-agentic")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

from .constants.core import NO_VAL
from .models.parameters import ParamSpec
from .utils.parameters import to_paramspec_list, extract_io, is_valid_parameter_order
from .core.Invokable import (
    AtomicInvokable,
    Command,
    StructuredInvokable,
)

__all__ = [
    # Sentinels
    "NO_VAL",
    # Parameters and parameter utilities
    "ParamSpec",
    "to_paramspec_list",
    "extract_io",
    "is_valid_parameter_order",
    # Invokable core types
    "AtomicInvokable",
    # Command invokable type
    "Command",
    # Structured invokable types
    "StructuredInvokable",
    ]