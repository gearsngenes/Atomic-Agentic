"""
Sample script to exercise the new Tool class invocation logic.

Adjust the import below to match your project layout:
from modules.Tools import Tool, ToolInvocationError, ToolDefinitionError
"""
import sys, os
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from atomic_agentic.Tools import Tool, ToolInvocationError, ToolDefinitionError


# ---------- Target callables with diverse signatures ----------

def mix_sig(a, b, /, c, d=10, *, e, fkw=0) -> tuple:
    """Pos-only a,b; pos-or-kw c,d; kw-only e,fkw"""
    return ("mix_sig", a, b, c, d, e, fkw)

def kwargs_only(*, x, y=1) -> int:
    """Keyword-only x (required), y (default)"""
    return x + y

def with_varargs(u, /, v, w=2, *args, x, y=3, **z) -> dict:
    """
    u: pos-only (required)
    v,w: pos-or-kw (w default)
    *args: extra positionals
    x,y: kw-only (y default)
    **z: extra keywords
    """
    return {
        "fn": "with_varargs",
        "u": u, "v": v, "w": w,
        "args": list(args),
        "x": x, "y": y,
        "z": dict(z),
    }

def defaults_only(a=1, b=2) -> int:
    """All defaults; returns product."""
    return a * b

class Accumulator:
    def add(self, a, /, b, *, scale=1) -> int:
        """Bound method with pos-only + kw-only."""
        return int((a + b) * scale)


# ---------- Wrap them as Tools ----------

t_mix = Tool(mix_sig, name="mix_sig", description="Mixed kinds", source="local")
t_kwargs = Tool(kwargs_only, name="kwargs_only", description="KW-only", source="local")
t_var = Tool(with_varargs, name="with_varargs", description="Varargs + Varkw", source="local")
t_defs = Tool(defaults_only, name="defaults_only", description="Defaults only", source="local")

acc = Accumulator()
t_add = Tool(acc.add, name="acc.add", description="Bound method add", source="local")


# ---------- Helpers ----------

def run_case(label, tool: Tool, inputs:dict):
    print(f"\n=== {label} ===")
    try:
        out = tool.invoke(inputs)
        print("OK:", out)
    except (ToolInvocationError, ToolDefinitionError) as e:
        print("ERR:", e)


def show_signature(tool):
    meta = tool.to_dict()
    print(f"\n-- {tool.name} call plan --")
    print("signature:", meta["signature"])
    print("posonly_order:", meta["posonly_order"])
    print("p_or_kw_names:", meta["p_or_kw_names"])
    print("kw_only_names:", meta["kw_only_names"])
    print("required_names:", meta["required_names"])
    print("arguments_map (name -> {index, kind, has_default, default}):")
    for name, spec in meta["arguments_map"].items():
        print(f"  {name}: idx={spec['index']}, kind={spec['kind']}, "
              f"has_default={spec['has_default']}, default={spec.get('default', None)}")


# ---------- Happy-path cases ----------

if __name__ == "__main__":
    # Inspect signatures/call-plans
    show_signature(t_mix)
    show_signature(t_kwargs)
    show_signature(t_var)
    show_signature(t_defs)
    show_signature(t_add)

    # 1) Mixed kinds: provide requireds; use default for d, fkw
    run_case(
        "mix_sig: happy path",
        t_mix,
        {"a": 1, "b": 2, "c": 3, "e": "hello"}
    )
    # Expect: ("mix_sig", 1, 2, 3, 10, "hello", 0)

    # 2) KW-only: provide just required x; y uses default
    run_case(
        "kwargs_only: happy path",
        t_kwargs,
        {"x": 5}
    )
    # Expect: 6

    # 3) Varargs & Varkw: use all features
    run_case(
        "with_varargs: use *args via _args and **kwargs via _kwargs",
        t_var,
        {
            "u": 9, "v": 8,            # u pos-only; v pos-or-kw
            # w omitted -> default 2
            "_args": [10, 20],         # goes into *args
            "x": "X",                  # kw-only (required)
            # y omitted -> default 3
            "_kwargs": {"alpha": 1, "beta": 2}  # extra keys into **z
        }
    )
    # Expect: dict with args=[10,20], x="X", y=3, z={"alpha":1,"beta":2}

    # 4) Defaults only: call with empty mapping
    run_case(
        "defaults_only: empty inputs",
        t_defs,
        {}
    )
    # Expect: 2

    # 5) Bound method (pos-only + kw-only)
    run_case(
        "acc.add bound method",
        t_add,
        {"a": 3, "b": 7, "scale": 2}
    )
    # Expect: (3+7)*2 = 20

    # ---------- Intentional error cases ----------

    # E1) Missing required kw-only 'e'
    run_case(
        "mix_sig: missing required kw-only 'e'",
        t_mix,
        {"a": 1, "b": 2, "c": 3}
    )

    # E2) Positional-only gap: provide b but omit a
    run_case(
        "mix_sig: positional-only gap (b present, a missing)",
        t_mix,
        {"b": 2, "c": 3, "e": "hello"}
    )

    # E3) Unknown top-level key when **kwargs exists (should ask for _kwargs)
    run_case(
        "with_varargs: unknown top-level key (must use _kwargs)",
        t_var,
        {"u": 1, "v": 2, "x": "X", "oops": 123}
    )

    # E4) Duplicate between named and _kwargs
    run_case(
        "with_varargs: duplicate key across named and _kwargs",
        t_var,
        {"u": 1, "v": 2, "x": "X", "_kwargs": {"x": "duplicate"}}
    )

    # E5) _args provided but function has no *args
    run_case(
        "mix_sig: _args provided but no *args in signature",
        t_mix,
        {"a": 1, "b": 2, "c": 3, "e": "hello", "_args": [99]}
    )

    # E6) _kwargs provided but function has no **kwargs
    run_case(
        "kwargs_only: _kwargs provided but no **kwargs in signature",
        t_kwargs,
        {"x": 1, "_kwargs": {"z": 10}}
    )
