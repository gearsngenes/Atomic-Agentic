from importlib.metadata import PackageNotFoundError, version

try:  # populated when installed or when a wheel is built
    __version__ = version("atomic-agentic")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

from .core.sentinels import NO_VAL
from .core.Parameters import ParamSpec, to_paramspec_list, extract_io, is_valid_parameter_order
from .core.Invokable import AtomicInvokable, ParameterMap, ArgumentMap, ArgSpec
from .workflows.StructuredInvokable import StructuredInvokable, StructuredResultDict

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
    # Structured invokable types
    "StructuredInvokable",
    "StructuredResultDict",
    # Legacy aliases for backward compatibility
    "ArgSpec",  # Deprecated: use ParamSpec instead
    "ArgumentMap",  # Deprecated: use ParameterMap instead
    "ParameterMap", # Deprecated: no longer used 
    ]