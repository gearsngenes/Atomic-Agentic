from atomic_agentic.tools import Tool

def sample_func(id: int, text: str, *args, **kwargs) -> str:
    """Convert input text to uppercase and display all inputs."""
    result = f"[id={id}] text={text.upper()}"

    if args:
        for i, arg in enumerate(args):
            result += f" | arg-{i}={arg}"

    if kwargs:
        for key, value in kwargs.items():
            result += f" | {key}={value}"

    return result

sample_tool = Tool(sample_func)

print("--- TOOL OBJECT ---")
print(sample_tool)
print()

print("--- SIGNATURE ---")
print(sample_tool.signature)
print()


def run_case(label: str, fn) -> None:
    """Prints whatever fn() returns as-is.

    Used for call-style (`sample_tool(...)`) and `return_atomic_result_object=True`
    cases below, where the return shape differs deliberately from `.invoke(...)`.
    """
    print(f"--- {label} ---")
    try:
        print(fn())
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}")
    print()


def run_invoke_case(label: str, fn) -> None:
    """Prints fn().result — for `.invoke(...)`/`.async_invoke(...)` cases,
    which always return the full AtomicResult envelope."""
    print(f"--- {label} ---")
    try:
        result = fn()
        print(result.result)
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}")
    print()


# ------------------------------------------------------------------ #
# __call__ vs invoke(...): the real distinction
# ------------------------------------------------------------------ #
# `.invoke(...)`/`.async_invoke(...)` always return the full AtomicResult
# envelope (.result, .run_id, .started_at/.ended_at, ...). `sample_tool(...)`/
# `await sample_tool.async_call(...)` are the "interpreted convenience path" —
# by default they unwrap straight to the raw .result payload, since calling
# something like a plain function is normally not asking for provenance
# metadata back. Pass `return_atomic_result_object=True` to get the full
# envelope from the call-style path too.

run_case(
    "CALL: returns the unwrapped value directly (no .result needed)",
    lambda: sample_tool(1, "hello"),
)

run_invoke_case(
    "INVOKE: returns the full envelope (.result needed)",
    lambda: sample_tool.invoke({"id": 1, "text": "hello"}),
)

run_case(
    "CALL + return_atomic_result_object=True: envelope, same as invoke(...)",
    lambda: sample_tool(1, "hello", return_atomic_result_object=True),
)

# ------------------------------------------------------------------ #
# __call__ samples — argument-binding styles, all unwrapped by default
# ------------------------------------------------------------------ #

run_case(
    "CALL: positional only",
    lambda: sample_tool(1, "hello"),
)

run_case(
    "CALL: positional + extra varargs",
    lambda: sample_tool(2, "world", "extra-a", 99, True),
)

run_case(
    "CALL: keyword for named params",
    lambda: sample_tool(id=3, text="keyword call"),
)

run_case(
    "CALL: mixed positional + keyword extras",
    lambda: sample_tool(4, "mixed", flag=True, source="demo"),
)

run_case(
    "CALL: mixed positional + varargs + kwargs",
    lambda: sample_tool(5, "combo", "x", "y", mode="test", retries=2),
)

run_case(
    "CALL: positional id + keyword text",
    lambda: sample_tool(6, text="semi-mixed"),
)

# ------------------------------------------------------------------ #
# invoke(...) samples — always the full envelope
# ------------------------------------------------------------------ #

run_invoke_case(
    "INVOKE: minimal named inputs",
    lambda: sample_tool.invoke({"id": 7, "text": "invoke basic"}),
)

run_invoke_case(
    "INVOKE: explicit *args payload under declared vararg name",
    lambda: sample_tool.invoke({
        "id": 8,
        "text": "invoke args",
        "args": ["alpha", 123, False],
    }),
)

run_invoke_case(
    "INVOKE: explicit **kwargs payload under declared varkwarg name",
    lambda: sample_tool.invoke({
        "id": 9,
        "text": "invoke kwargs",
        "kwargs": {"lang": "en", "priority": "high"},
    }),
)

run_invoke_case(
    "INVOKE: explicit *args and **kwargs payloads together",
    lambda: sample_tool.invoke({
        "id": 10,
        "text": "invoke full",
        "args": ("tail-1", "tail-2"),
        "kwargs": {"debug": True, "count": 3},
    }),
)

run_invoke_case(
    "INVOKE: named params + empty vararg/varkwarg payloads",
    lambda: sample_tool.invoke({
        "id": 11,
        "text": "empty payloads",
        "args": [],
        "kwargs": {},
    }),
)

# ------------------------------------------------------------------ #
# negative / validation samples
# ------------------------------------------------------------------ #

run_invoke_case(
    "ERROR: missing required parameter",
    lambda: sample_tool.invoke({"id": 12}),
)

run_invoke_case(
    "ERROR: unknown parameter in invoke",
    lambda: sample_tool.invoke({"id": 13, "text": "bad", "extra": "nope"}),
)

run_invoke_case(
    "ERROR: invalid explicit varargs payload",
    lambda: sample_tool.invoke({
        "id": 14,
        "text": "bad args",
        "args": "not-a-sequence",
    }),
)

run_invoke_case(
    "ERROR: invalid explicit varkwargs payload",
    lambda: sample_tool.invoke({
        "id": 15,
        "text": "bad kwargs",
        "kwargs": ["not", "a", "mapping"],
    }),
)
