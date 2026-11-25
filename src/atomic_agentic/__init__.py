from importlib.metadata import PackageNotFoundError, version

try:  # populated when installed or when a wheel is built
    __version__ = version("atomic-agentic")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

from .constants.core import NO_VAL
from .models.parameters import ParamSpec
from .utils.parameters import (
    extract_io,
    is_valid_parameter_order,
    parameter_collisions,
    parameter_overlap,
    semantically_compatible,
    semantically_identical,
    to_parameter_list,
)
from .invokables.atomic_invokable import AtomicInvokable

__all__ = [
    # Sentinels
    "NO_VAL",
    # Parameters and parameter utilities
    "ParamSpec",
    "to_parameter_list",
    "extract_io",
    "is_valid_parameter_order",
    "semantically_compatible",
    "semantically_identical",
    "parameter_overlap",
    "parameter_collisions",
    # Invokable core types
    "AtomicInvokable",
    ]