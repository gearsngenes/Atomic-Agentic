"""Attachment validation and structured-output-schema helpers for LLM engine
adapters."""

import copy
import mimetypes
import os
from typing import Any, Mapping, Optional

from ..constants.llm import (
    LIST_OF_SCHEMAS_KEYS,
    NAME_KEYED_CONTAINER_KEYS,
    NAME_KEYED_LAYER,
    OPAQUE_VALUE,
    SCHEMA_LIST_LAYER,
    SINGLE_SCHEMA_KEYS,
    SINGLE_SCHEMA_LAYER,
)


def validate_attachment_path(
    path: str,
    illegal_exts: set[str],
    allowed_exts: set[str] | None,
    illegal_mime_prefixes: tuple[str, ...],
) -> None:
    """
    Validate a local file path for engine attachment.

    Raises ``ValueError`` on any violation. Engine subclasses call this from
    their ``_validate_attachment_path`` overrides and wrap ``ValueError`` into
    ``LLMEngineError`` at the engine boundary.

    Validation order:
    1. File must exist and be a regular file.
    2. Extension must not be in ``illegal_exts``.
    3. If ``allowed_exts`` is given, extension must be in it.
    4. If ``illegal_mime_prefixes`` is non-empty, MIME type must not match.
    """
    if not os.path.isfile(path):
        raise ValueError(f"path does not exist or is not a file: {path!r}")

    _, ext = os.path.splitext(path)
    ext = ext.lower()

    if ext and ext in illegal_exts:
        raise ValueError(f"extension {ext!r} is not permitted")

    if allowed_exts is not None and ext not in allowed_exts:
        raise ValueError(f"extension {ext!r} is not supported")

    if illegal_mime_prefixes:
        mime, _ = mimetypes.guess_type(path)
        mime = mime or ""
        if any(mime.startswith(prefix) for prefix in illegal_mime_prefixes):
            raise ValueError(f"MIME type {mime!r} is not permitted")


def clean_structure_template(
    template: Mapping[str, Any],
    permitted_keys: Optional[frozenset[str]],
    omitted_keys: frozenset[str],
) -> dict[str, Any]:
    """
    Strip JSON-Schema keywords a provider doesn't support from a structured-
    output request template, without ever touching a user-defined field name.

    ``permitted_keys`` (``None`` = unrestricted) and ``omitted_keys`` (empty =
    no-op) together decide which *keyword* keys survive at every schema-object
    level. Neither ``template`` nor anything nested inside it is mutated --
    the return value is an entirely independent structure.

    Raises ``ValueError`` if ``template`` is not a ``Mapping``.
    """
    if not isinstance(template, Mapping):
        raise ValueError(
            f"clean_structure_template: template must be a Mapping, got {type(template)!r}"
        )
    return _prune_schema_object(dict(template), permitted_keys, omitted_keys)


def _schema_value_kind(key: str, value: Any) -> str:
    """
    Classify which recursion layer ``key``'s ``value`` routes into, so the
    caller never has to re-derive "am I entering a JSON-Schema *keyword*
    layer (safe to prune) or a user-configured *name* layer (never prune,
    e.g. ``properties``' own keys)" inline.

    Returns one of ``NAME_KEYED_LAYER`` / ``SCHEMA_LIST_LAYER`` /
    ``SINGLE_SCHEMA_LAYER`` / ``OPAQUE_VALUE`` (``constants/llm.py``). The
    three non-opaque kinds always route into further JSON-Schema keyword
    layers (directly, or via a list/name-keyed layer of them) -- only
    ``OPAQUE_VALUE`` is a leaf, copied as-is without further inspection.
    """
    if key in NAME_KEYED_CONTAINER_KEYS and isinstance(value, Mapping):
        return NAME_KEYED_LAYER
    if key in LIST_OF_SCHEMAS_KEYS and isinstance(value, (list, tuple)):
        return SCHEMA_LIST_LAYER
    if key == "items" and isinstance(value, list):
        # Tuple-form `items` (pre-2020-12 draft) -- list of schemas.
        return SCHEMA_LIST_LAYER
    if key in SINGLE_SCHEMA_KEYS and isinstance(value, Mapping):
        return SINGLE_SCHEMA_LAYER
    return OPAQUE_VALUE


def _prune_schema_object(
    schema: Mapping[str, Any],
    permitted_keys: Optional[frozenset[str]],
    omitted_keys: frozenset[str],
) -> dict[str, Any]:
    """
    Recursive worker for ``clean_structure_template``. Every key of ``schema``
    is treated as a JSON-Schema keyword layer -- only ever called directly on
    a dict already known to be a schema object, never on a name-keyed
    container's own dict (see the ``NAME_KEYED_LAYER`` branch below, which
    recurses into each *value*, not the container itself -- the container's
    own keys are a user-configured layer and are never pruned).
    """
    result: dict[str, Any] = {}
    for key, value in schema.items():
        # Keyword-survival check -- drop the key (and its value) entirely.
        if key in omitted_keys or (permitted_keys is not None and key not in permitted_keys):
            continue

        kind = _schema_value_kind(key, value)
        if kind == NAME_KEYED_LAYER:
            # `value`'s keys are user-defined names (a user-configured
            # layer) -- never checked against permitted/omitted. Only each
            # named entry's own sub-schema (a keyword layer again) is.
            new_value = {
                name: _prune_schema_object(subschema, permitted_keys, omitted_keys)
                if isinstance(subschema, Mapping)
                else copy.deepcopy(subschema)
                for name, subschema in value.items()
            }
        elif kind == SCHEMA_LIST_LAYER:
            new_value = [
                _prune_schema_object(item, permitted_keys, omitted_keys)
                if isinstance(item, Mapping)
                else copy.deepcopy(item)
                for item in value
            ]
        elif kind == SINGLE_SCHEMA_LAYER:
            new_value = _prune_schema_object(value, permitted_keys, omitted_keys)
        else:
            # Opaque value (scalars, `required` lists, boolean-form
            # `additionalProperties`, or a shape none of the layer kinds
            # matched) -- copied independently, not walked further.
            new_value = copy.deepcopy(value)

        result[key] = new_value

    return result
