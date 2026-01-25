from importlib.metadata import PackageNotFoundError, version

try:  # populated when installed or when a wheel is built
    __version__ = version("atomic-agentic")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

from .core.Invokable import AtomicInvokable, NO_VAL, ParamSpec, ParameterMap, ArgumentMap, ArgSpec

__all__ = [
    "NO_VAL",
    "AtomicInvokable",
    "ParamSpec",
    "ParameterMap",
    # Legacy aliases for backward compatibility
    "ArgSpec",  # Deprecated: use ParamSpec instead
    "ArgumentMap",  # Deprecated: use ParameterMap instead
    ]