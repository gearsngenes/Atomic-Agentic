"""
04_RoutingFlow.py

Beginner-friendly RoutingFlow example.
Demonstrates routing support requests to different branches based on topic and urgency, with clear output and result inspection.
"""

from __future__ import annotations
from pprint import pprint
from dotenv import load_dotenv
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
# Branch functions: each returns a string response
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

# ──────────────────────────────────────────────────────────────
# Build the RoutingFlows
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
    print("Selected index:", result.selected_key)
    print("Chosen branch run:", result.chosen_branch_run)
    print("Router run:", result.router_run_id)
    print("Router decision:", flow.get_router_decision(result.run_id))

print("\n########## Dict-configured branches (key selector) ##########")
for i, payload in enumerate(examples, start=1):
    result = flow_by_key.invoke(payload)
    print(f"\n=== Run {i} ===")
    print("Inputs:", payload)
    print("Run ID:", result.run_id)
    print("Result:", result.result)
    print("Selected key:", result.selected_key)
    print("Chosen branch run:", result.chosen_branch_run)
    print("Router run:", result.router_run_id)
    print("Router decision:", flow_by_key.get_router_decision(result.run_id))

print("\n=== Flow snapshot (list-configured) ===")
pprint(flow.to_dict())
