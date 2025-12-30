from importlib.metadata import PackageNotFoundError, version

try:  # populated when installed or when a wheel is built
    __version__ = version("atomic-agentic")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

from .core.Invokable import AtomicInvokable, NO_VAL, ArgSpec, ArgumentMap

__all__ = [
    "NO_VAL",
    "AtomicInvokable",
    "ArgSpec",
    ]