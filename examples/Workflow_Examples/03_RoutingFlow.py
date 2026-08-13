"""
03_RoutingFlow.py

Beginner-friendly RoutingFlow example.
Demonstrates routing support requests to different branches based on topic
and urgency: int-indexed and dict-keyed branch topologies, trace-based run
inspection, and how RoutingFlow reconciles branch parameters the router
doesn't declare. A name only one branch needs is never required on the
outer schema -- it becomes optional with default=None (and "None" is added
to its declared type, so the signature is honest about it), *unless* every
branch that needs it happens to agree on the same real default, in which
case that shared value is used instead.
"""

from __future__ import annotations
from pprint import pprint
from atomic_agentic.tools import Tool
from atomic_agentic.workflows import RoutingFlow

# ──────────────────────────────────────────────────────────────
# Router functions: return the selector used to pick a branch
# ──────────────────────────────────────────────────────────────
def choose_branch(topic: str, urgency: int = 0) -> int:
    """Return the branch index to use for the given request."""
    normalized = topic.strip().lower()
    if urgency >= 8:
        return 2  # urgent/escalation path
    if any(word in normalized for word in ("bill", "refund", "payment", "invoice")):
        return 0  # billing path
    return 1  # general support path

def choose_branch_key(topic: str, urgency: int = 0) -> str:
    """Return the branch key to use for the given request."""
    normalized = topic.strip().lower()
    if urgency >= 8:
        return "escalation"
    if any(word in normalized for word in ("bill", "refund", "payment", "invoice")):
        return "billing"
    return "general"

# ──────────────────────────────────────────────────────────────
# Branch functions: each declares the SAME (topic, urgency) shape as
# the router.
# ──────────────────────────────────────────────────────────────
def handle_billing(topic: str, urgency: int = 0) -> str:
    return (
        f"Billing team selected.\n"
        f"Issue: {topic}\n"
        f"Urgency: {urgency}\n"
        f"Action: review payment history, invoice status, and refund eligibility."
    )

def handle_general_support(topic: str, urgency: int = 0) -> str:
    return (
        f"General support selected.\n"
        f"Issue: {topic}\n"
        f"Urgency: {urgency}\n"
        f"Action: collect reproduction details and suggest next troubleshooting steps."
    )

def handle_escalation(topic: str, urgency: int = 0) -> str:
    return (
        f"Escalation team selected.\n"
        f"Issue: {topic}\n"
        f"Urgency: {urgency}\n"
        f"Action: prioritize immediate human follow-up and incident review."
    )

# A branch that needs MORE than the router does. `notes` isn't part of the
# router's own schema and no other branch declares it either, so it
# becomes optional on the OUTER schema: default=None, and "None" is added
# to its declared type (unlike the type-signature lie a plain
# default=None-with-untouched-type would be). filter_inputs
# (core/Invokable.py) injects declared defaults for absent params at EVERY
# layer, so an omitted `notes` arrives at this branch as a real notes=None
# value, not a missing argument. Branches meant to receive a
# router-absent parameter should handle None defensively for anything the
# router doesn't guarantee.
def handle_escalation_with_notes(topic: str, notes: str, urgency: int = 0) -> str:
    return (
        f"Escalation team selected.\n"
        f"Issue: {topic}\n"
        f"Urgency: {urgency}\n"
        f"Handoff notes: {notes}\n"
        f"Action: prioritize immediate human follow-up and incident review."
    )

# ──────────────────────────────────────────────────────────────
# Build router tools and branch tools
# ──────────────────────────────────────────────────────────────
router_tool = Tool(
    function=choose_branch,
    name="choose_branch",
    namespace="support",
    description="Return the branch index for the incoming support request.",
)

router_tool_by_key = Tool(
    function=choose_branch_key,
    name="choose_branch_key",
    namespace="support",
    description="Return the branch key for the incoming support request.",
)

billing_tool = Tool(
    function=handle_billing,
    name="handle_billing",
    namespace="support",
    description="Handle billing-related issues.",
)

general_tool = Tool(
    function=handle_general_support,
    name="handle_general_support",
    namespace="support",
    description="Handle general support issues.",
)

escalation_tool = Tool(
    function=handle_escalation,
    name="handle_escalation",
    namespace="support",
    description="Handle urgent escalations.",
)

escalation_notes_tool = Tool(
    function=handle_escalation_with_notes,
    name="handle_escalation_with_notes",
    namespace="support",
    description="Handle urgent escalations with handoff notes.",
)

# ──────────────────────────────────────────────────────────────
# Build the RoutingFlows.
# ──────────────────────────────────────────────────────────────
# List-configured branches: router returns an int index into branches.
flow = RoutingFlow(
    name="support_router",
    namespace="examples",
    description="Route support requests to one fixed branch based on router output.",
    router=router_tool,
    branches=[billing_tool, general_tool, escalation_tool],
)

# Dict-configured branches: router returns a key into branches.
flow_by_key = RoutingFlow(
    name="support_router_by_key",
    namespace="examples",
    description="Route support requests to one fixed branch keyed by router output.",
    router=router_tool_by_key,
    branches={
        "billing": billing_tool,
        "general": general_tool,
        "escalation": escalation_tool,
    },
)

# ──────────────────────────────────────────────────────────────
# Example invocations
# ──────────────────────────────────────────────────────────────
examples = [
    {"topic": "Need a refund for duplicate invoice", "urgency": 2},
    {"topic": "The dashboard keeps logging me out", "urgency": 3},
    {"topic": "Production outage affecting all customers", "urgency": 10},
]

print("\n########## List-configured branches (int selector) ##########")
for i, payload in enumerate(examples, start=1):
    result = flow.invoke(payload)
    print(f"\n=== Run {i} ===")
    print("Inputs:", payload)
    print("Run ID:", result.run_id)
    print("Result:", result.result)
    print("Selected branch:", result.selected_branch)
    # trace is exactly 2 entries when include_trace is on (the default):
    # trace[0] is the router's own result, trace[1] is the selected branch's.
    for j, node_result in enumerate(result.trace):
        print(f"  trace[{j}]: run_id={node_result.run_id}, result={node_result.result!r}")

print("\n########## Dict-configured branches (key selector) ##########")
for i, payload in enumerate(examples, start=1):
    result = flow_by_key.invoke(payload)
    print(f"\n=== Run {i} ===")
    print("Inputs:", payload)
    print("Run ID:", result.run_id)
    print("Result:", result.result)
    print("Selected branch:", result.selected_branch)
    for j, node_result in enumerate(result.trace):
        print(f"  trace[{j}]: run_id={node_result.run_id}, result={node_result.result!r}")

# ──────────────────────────────────────────────────────────────
# escalation_notes_tool needs `notes`, which the router never looks at and
# no other branch declares either. It becomes optional on the outer
# schema -- default=None, "None" added to its declared type.
# ──────────────────────────────────────────────────────────────
print("\n########## A branch needing more than the router ##########")
open_flow = RoutingFlow(
    name="support_router_open",
    namespace="examples",
    description="Escalation branch needs handoff notes the router doesn't use.",
    router=router_tool,
    branches=[billing_tool, general_tool, escalation_notes_tool],
)
print("Reconciled outer parameters:", [p.name for p in open_flow.parameters])
notes_param = next(p for p in open_flow.parameters if p.name == "notes")
print(f"'notes' param: type={notes_param.type!r}, default={notes_param.default!r} (optional -- synthesized, not carried through as required)")

urgent_payload = {"topic": "Production outage affecting all customers", "urgency": 10, "notes": "Customer X escalated via phone"}
result = open_flow.invoke(urgent_payload)
print("\nWith notes supplied:", result.result)

result_no_notes = open_flow.invoke({"topic": "Production outage affecting all customers", "urgency": 10})
print("\nWithout notes supplied:", result_no_notes.result)
print("(no error -- the synthesized default=None was injected and passed straight through)")

print("\n=== Flow snapshot (list-configured) ===")
pprint(flow.to_dict())
