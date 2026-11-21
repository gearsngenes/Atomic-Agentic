"""
04_Router.py

Router: a selector plus a dict of routes, with plain functions (no LLM, no keys).

Demonstrates:
- a selector that returns a string key and routes that are plain Tools;
  RouterResult.result / .route / .branch and the union return_type
- any hashable key works (tuples, None)
- parameters= so a branch can receive an input the selector does not declare
- a Pipeline as a route (nesting) and the ValueError for an unknown route

Run from the repo root:
    python examples/Composite_Examples/04_Router.py
"""

from __future__ import annotations

from atomic_agentic.composites import Pipeline, Router
from atomic_agentic.models.parameters import ParamSpec
from atomic_agentic.tools import Tool


# ---------------------------------------------------------------------
# Selectors and branches
# ---------------------------------------------------------------------
def classify(request: str) -> str:
    text = request.lower()
    if "refund" in text or "invoice" in text:
        return "billing"
    if "outage" in text or "lawyer" in text:
        return "escalation"
    return "general"


def handle_billing(request: str) -> str:
    return f"Billing team will review: {request}"


def handle_general(request: str) -> str:
    return f"Support answered: {request}"


def handle_escalation(request: str, notes: str = "(no notes)") -> dict[str, str]:
    return {"queue": "on-call", "request": request, "notes": notes}


def string_key_demo() -> Router:
    print("\n=== 1. String keys ===")
    router = Router(
        "support",
        "Route a support request to a team.",
        selector=classify,
        routes={
            "billing": Tool(function=handle_billing, name="billing", description="Billing."),
            "general": Tool(function=handle_general, name="general", description="General."),
            "escalation": Tool(function=handle_escalation, name="escalation", description="Escalate."),
        },
    )
    print("return_type:", router.return_type)
    for request in ("Please refund my invoice", "How do I reset my password?"):
        result = router.invoke({"request": request})
        print(f"result={result.result!r}")
        print(f"  route={result.route!r} branch={result.branch!r} ({type(result).__name__})")
    return router


# ---------------------------------------------------------------------
# 2. Any hashable key
# ---------------------------------------------------------------------
def bucket(score: int) -> tuple[str, int] | None:
    if score < 0:
        return None
    return ("tier", 1 if score < 50 else 2)


def non_string_key_demo() -> None:
    print("\n=== 2. Tuple and None keys ===")
    router = Router(
        "scoring",
        "Route by a tuple key; None means no score.",
        selector=bucket,
        routes={
            None: lambda score: "invalid score",
            ("tier", 1): lambda score: "basic plan",
            ("tier", 2): lambda score: "premium plan",
        },
    )
    for score in (-5, 10, 90):
        result = router.invoke({"score": score})
        print(f"score={score:3} -> route={result.route!r} result={result.result!r}")


# ---------------------------------------------------------------------
# 3. parameters= for an input only a branch needs
# ---------------------------------------------------------------------
def parameters_demo() -> None:
    print("\n=== 3. parameters= ===")
    routes = {
        "billing": handle_billing,
        "general": handle_general,
        "escalation": handle_escalation,
    }
    plain = Router("plain_support", "Selector-only inputs.", classify, routes)
    inputs = {"request": "Total outage in region 2", "notes": "customer is a VIP"}
    print("without parameters=:", plain.invoke(inputs).result)

    declared = [
        ParamSpec(name="request", index=0, kind=ParamSpec.KEYWORD_ONLY, type=("str",)),
        ParamSpec(name="notes", index=1, kind=ParamSpec.KEYWORD_ONLY, type=("str",)),
    ]
    with_notes = Router("noted_support", "Accepts notes too.", classify, routes, parameters=declared)
    print("with parameters=   :", with_notes.invoke(inputs).result)


# ---------------------------------------------------------------------
# 4. A Pipeline as a route, and an unknown route
# ---------------------------------------------------------------------
def normalize(request: str) -> dict[str, str]:
    return {"request": request.strip().upper()}


def nesting_demo() -> None:
    print("\n=== 4. Pipeline branch and unknown route ===")
    shout_then_answer = Pipeline("shout_then_answer", "Normalize then answer.", [normalize, handle_general])
    router = Router(
        "nested_support",
        "One branch is a Pipeline.",
        selector=classify,
        routes={"billing": handle_billing, "general": shout_then_answer},
    )
    result = router.invoke({"request": "  where is my parcel  "})
    print(f"result={result.result!r} branch={result.branch!r}")

    # "escalation" is a selector value with no route here
    try:
        router.invoke({"request": "I will call my lawyer"})
    except ValueError as exc:
        print("ValueError:", exc)


def main() -> None:
    string_key_demo()
    non_string_key_demo()
    parameters_demo()
    nesting_demo()


if __name__ == "__main__":
    main()
