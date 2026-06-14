"""
Sample script to exercise Tool invocation logic.

This example demonstrates:

1. Dict-first invocation for positional-only, positional-or-keyword,
   keyword-only, *args, and **kwargs parameters.
2. The current AtomicInvokable/Tool binding contract:
   - *args payloads use the callable's actual VAR_POSITIONAL parameter name.
   - **kwargs payloads use the callable's actual VAR_KEYWORD parameter name.
   - There is no reserved "_args" / "_kwargs" convention by default.
3. The effect of filter_extraneous_inputs:
   - Default True: unknown keys are dropped when no **kwargs exists.
   - Default True + **kwargs exists: unknown top-level keys are collected into **kwargs.
   - False: unknown keys are retained and binding errors if no **kwargs can consume them.
"""

from __future__ import annotations

from typing import Any

from atomic_agentic.core.Exceptions import ToolDefinitionError, ToolInvocationError
from atomic_agentic.core.constants import NO_VAL
from atomic_agentic.tools import Tool


# ---------- Target callables with diverse signatures ----------

def mix_sig(a: Any, b: Any, /, c: Any, d: int = 10, *, e: Any, fkw: int = 0) -> tuple:
    """Pos-only a,b; pos-or-kw c,d; kw-only e,fkw."""
    return ("mix_sig", a, b, c, d, e, fkw)


def kwargs_only(*, x: Any, y: int = 1) -> int:
    """Keyword-only x required; y defaulted."""
    return x + y


def with_varargs(u: Any, /, v: Any, w: int = 2, *args: Any, x: Any, y: int = 3, **z: Any) -> dict:
    """
    u: positional-only required
    v,w: positional-or-keyword, w defaulted
    *args: extra positionals
    x,y: keyword-only, x required and y defaulted
    **z: extra keywords
    """
    return {
        "fn": "with_varargs",
        "u": u,
        "v": v,
        "w": w,
        "args": list(args),
        "x": x,
        "y": y,
        "z": dict(z),
    }


def defaults_only(a: int = 1, b: int = 2) -> int:
    """All defaults; returns product."""
    return a * b


class Accumulator:
    def add(self, a: Any, /, b: Any, *, scale: int = 1) -> int:
        """Bound method with pos-only + kw-only."""
        return int((a + b) * scale)


# ---------- Wrap them as Tools ----------

# Default filtering behavior: filter_extraneous_inputs=True.
t_mix = Tool(mix_sig, name="mix_sig", description="Mixed kinds", namespace="local")
t_kwargs = Tool(kwargs_only, name="kwargs_only", description="KW-only", namespace="local")
t_var = Tool(with_varargs, name="with_varargs", description="Varargs + Varkw", namespace="local")
t_defs = Tool(defaults_only, name="defaults_only", description="Defaults only", namespace="local")

# Explicitly unfiltered variants, used to demonstrate stricter error behavior
# when unknown keys are retained and the callable has no **kwargs sink.
t_mix_unfiltered = Tool(
    mix_sig,
    name="mix_sig_unfiltered",
    description="Mixed kinds, filtering disabled",
    namespace="local",
    filter_extraneous_inputs=False,
)

t_kwargs_unfiltered = Tool(
    kwargs_only,
    name="kwargs_only_unfiltered",
    description="KW-only, filtering disabled",
    namespace="local",
    filter_extraneous_inputs=False,
)

t_defs_unfiltered = Tool(
    defaults_only,
    name="defaults_only_unfiltered",
    description="Defaults only, filtering disabled",
    namespace="local",
    filter_extraneous_inputs=False,
)

acc = Accumulator()
t_add = Tool(acc.add, name="acc_add", description="Bound method add", namespace="local")


# ---------- Helpers ----------

def run_case(label: str, tool: Tool, inputs: dict[str, Any]) -> None:
    print(f"\n=== {label} ===")
    try:
        out = tool.invoke(inputs)
        print("OK:", out.result)
    except (ToolInvocationError, ToolDefinitionError, TypeError, ValueError) as exc:
        print("ERR:", exc)


def show_signature(tool: Tool) -> None:
    print(f"\n-- {tool.name} call plan --")
    print("signature:", tool.signature)
    print("parameters:")
    for param in tool.parameters:
        default_str = "default=NO_VAL" if param.default is NO_VAL else f"default={param.default!r}"
        print(f"  {param.name}: {param.kind}, type={param.type}, {default_str}")


# ---------- Run examples ----------

if __name__ == "__main__":
    # Inspect signatures/call-plans.
    show_signature(t_mix)
    show_signature(t_kwargs)
    show_signature(t_var)
    show_signature(t_defs)
    show_signature(t_add)

    # ------------------------------------------------------------------
    # Happy-path cases
    # ------------------------------------------------------------------

    run_case(
        "mix_sig: happy path",
        t_mix,
        {"a": 1, "b": 2, "c": 3, "e": "hello"},
    )
    # Expect: ("mix_sig", 1, 2, 3, 10, "hello", 0)

    run_case(
        "kwargs_only: happy path",
        t_kwargs,
        {"x": 5},
    )
    # Expect: 6

    run_case(
        "with_varargs: use *args via actual parameter name 'args' and **kwargs via actual parameter name 'z'",
        t_var,
        {
            "u": 9,
            "v": 8,
            # w omitted -> default 2
            "args": [10, 20],
            "x": "X",
            # y omitted -> default 3
            "z": {"alpha": 1, "beta": 2},
        },
    )
    # Expect: args=[10, 20], z={"alpha": 1, "beta": 2}

    run_case(
        "defaults_only: empty inputs",
        t_defs,
        {},
    )
    # Expect: 2

    run_case(
        "acc.add bound method",
        t_add,
        {"a": 3, "b": 7, "scale": 2},
    )
    # Expect: (3 + 7) * 2 = 20

    # ------------------------------------------------------------------
    # Intentional required-argument errors
    # ------------------------------------------------------------------

    run_case(
        "mix_sig: missing required kw-only 'e'",
        t_mix,
        {"a": 1, "b": 2, "c": 3},
    )

    run_case(
        "mix_sig: positional-only gap (b present, a missing)",
        t_mix,
        {"b": 2, "c": 3, "e": "hello"},
    )

    # ------------------------------------------------------------------
    # Demonstrate current **kwargs behavior
    # ------------------------------------------------------------------

    run_case(
        "with_varargs: unknown top-level key is collected into **z",
        t_var,
        {"u": 1, "v": 2, "x": "X", "oops": 123},
    )
    # Expect: z={"oops": 123}

    run_case(
        "with_varargs: explicit z payload is expanded into **z",
        t_var,
        {"u": 1, "v": 2, "x": "X", "z": {"alpha": 1, "beta": 2}},
    )
    # Expect: z={"alpha": 1, "beta": 2}

    run_case(
        "with_varargs: duplicate between named x and explicit z payload",
        t_var,
        {"u": 1, "v": 2, "x": "X", "z": {"x": "duplicate"}},
    )
    # Depending on binding validation, this should error before execution or
    # during Python call binding because x is supplied twice.

    run_case(
        "with_varargs: _args and _kwargs are ordinary unknown keys, not reserved channels",
        t_var,
        {
            "u": 9,
            "v": 8,
            "_args": [10, 20],
            "x": "X",
            "_kwargs": {"alpha": 1, "beta": 2},
        },
    )
    # Expect under current contract:
    # args=[] and z={"_args": [...], "_kwargs": {...}}

    # ------------------------------------------------------------------
    # Demonstrate default filter_extraneous_inputs=True
    # ------------------------------------------------------------------

    run_case(
        "defaults_only: unknown key is dropped when filtering is enabled",
        t_defs,
        {"unused": 999},
    )
    # Expect: 2

    run_case(
        "mix_sig: _args is dropped when filtering is enabled and no *args exists",
        t_mix,
        {"a": 1, "b": 2, "c": 3, "e": "hello", "_args": [99]},
    )
    # Expect: OK, because _args is extraneous and filtering is enabled.

    run_case(
        "kwargs_only: _kwargs is dropped when filtering is enabled and no **kwargs exists",
        t_kwargs,
        {"x": 1, "_kwargs": {"z": 10}},
    )
    # Expect: 2, because _kwargs is extraneous and filtering is enabled.

    # ------------------------------------------------------------------
    # Demonstrate filter_extraneous_inputs=False
    # ------------------------------------------------------------------

    run_case(
        "defaults_only_unfiltered: unknown key errors when filtering is disabled",
        t_defs_unfiltered,
        {"unused": 999},
    )

    run_case(
        "mix_sig_unfiltered: _args errors when filtering is disabled and no *args exists",
        t_mix_unfiltered,
        {"a": 1, "b": 2, "c": 3, "e": "hello", "_args": [99]},
    )

    run_case(
        "kwargs_only_unfiltered: _kwargs errors when filtering is disabled and no **kwargs exists",
        t_kwargs_unfiltered,
        {"x": 1, "_kwargs": {"z": 10}},
    )
