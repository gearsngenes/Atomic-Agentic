"""
StructuredInvokable Example: Schema Transformation and Packaging Controls

This script demonstrates how to wrap a basic tool with StructuredInvokable,
showing how to transform raw outputs (scalars, tuples, lists) into schema-enforced dicts.
It also highlights how to configure packaging controls for different use-cases.
"""
from atomic_agentic.workflows import StructuredInvokable

# --- Example 1: Scalar output to single-field dict ---
def add_one(x: int) -> int:
    """Return the input integer plus one."""
    return x + 1

structured_scalar = StructuredInvokable(
    component=add_one,
    output_schema=["result"],
)

print("\n--- Scalar to dict ---")
print("Structured output:", dict(structured_scalar.invoke({"x": 5})))

# --- Example 2: Tuple output to named fields ---
def min_max_sum(x: int, y: int) -> tuple:
    """Return the minimum, maximum, and sum of two integers."""
    return (min(x, y), max(x, y), x + y)

structured_tuple = StructuredInvokable(
    component=min_max_sum,
    output_schema=["min", "max", "sum"],
)

print("\n--- Tuple to dict ---")
print("Structured output:", dict(structured_tuple.invoke({"x": 3, "y": 7})))

# --- Example 3: List output with missing fields (absent_value_mode) ---
def first_two(items: list) -> list:
    """Return the first two items of a list."""
    return items[:2]

schema = ["a", "b", "c"]
structured_list_fill = StructuredInvokable(
    component=first_two,
    output_schema=schema,
    absent_value_mode=StructuredInvokable.FILL,
    default_absent_value="<missing>",
)

structured_list_drop = StructuredInvokable(
    component=first_two,
    output_schema=schema,
    absent_value_mode=StructuredInvokable.DROP,
)

print("\n--- List to dict (fill missing) ---")
print("Structured output (fill):", dict(structured_list_fill.invoke({"items": [1, 2]})))

print("\n--- List to dict (drop missing) ---")
print("Structured output (drop):", dict(structured_list_drop.invoke({"items": [1, 2]})))


# --- Example 4: Mapping with extra fields (ignore_unhandled) ---
def stats(x: int) -> dict:
    """Return a mapping with an extra field."""
    return {"a": x, "b": x * 2, "extra": 42}

# Any remaining extras will cause an error unless ignore_unhandled=True.
structured_stats_strict = StructuredInvokable(
    component=stats,
    output_schema=["a", "b"],
    ignore_unhandled=False,  # This will raise PackagingError if extras remain
)

structured_stats_ignore = StructuredInvokable(
    component=stats,
    output_schema=["a", "b"],
    map_extras=True,
    ignore_unhandled=True,  # This will silently drop any extras
)

print("\n--- Mapping with extras (ignore_unhandled=False, expect error) ---")
try:
    result = structured_stats_strict.invoke({"x": 5})
    print("Structured output:", dict(result))
except Exception as e:
    print("Error:", e)

print("\n--- Mapping with extras (ignore_unhandled=True, extras dropped) ---")
result = structured_stats_ignore.invoke({"x": 5})
print("Structured output:", dict(result))
# There is no .extras field; extras are simply dropped if not handled.

# --- Example 5: None as absent value ---
def maybe_value(flag: bool) -> int | None:
    """Return 42 if flag is True, otherwise return None."""
    return 42 if flag else None

structured_none = StructuredInvokable(
    component=maybe_value,
    output_schema=["value"],
    none_is_absent=True,
    absent_value_mode=StructuredInvokable.FILL,
    default_absent_value="ABSENT",
)

print("\n--- None as absent value ---")
print("Structured output (flag=True):", dict(structured_none.invoke({"flag": True})))
print("Structured output (flag=False):", dict(structured_none.invoke({"flag": False})))

# --- Example 6: Mismatched keys with map_extras=True ---

def mismatched_keys() -> dict:
    """This tool returns keys 'a' and 'b'"""
    return {"a": 10, "b": 20}

# Schema expects 'c' and 'd', but tool returns 'a' and 'b'.
schema_cd = ["c", "d"]

structured_mismatch_extras = StructuredInvokable(
    component=mismatched_keys,
    output_schema=schema_cd,
    map_extras=True,
)

structured_mismatch_noextras = StructuredInvokable(
    component=mismatched_keys,
    output_schema=schema_cd,
    map_extras=False,
    ignore_unhandled=True,
)

print("\n--- Mismatched keys (map_extras=True, extras fill schema) ---")
result = structured_mismatch_extras.invoke({})
print("Structured output:", dict(result))  # Should fill c=10, d=20

print("\n--- Mismatched keys (map_extras=False, expect error) ---")
try:
    result = structured_mismatch_noextras.invoke({})
    print("Structured output:", dict(result))
except Exception as e:
    print("Error:", e)