"""
05_RoutingFlow.py

Beginner-friendly RoutingFlow example.
Demonstrates routing support requests to different branches based on topic and urgency, with clear output and metadata inspection.
"""

from __future__ import annotations
from pprint import pprint
from dotenv import load_dotenv
from atomic_agentic.tools import Tool
from atomic_agentic.workflows.StructuredInvokable import StructuredInvokable
from atomic_agentic.workflows.basic import BasicFlow
from atomic_agentic.workflows.routing import RoutingFlow

# ──────────────────────────────────────────────────────────────
# Router function: returns the branch index to use
# ──────────────────────────────────────────────────────────────
def choose_branch(topic: str, urgency: int = 0) -> int:
    """Return the branch index to use for the given request."""
    normalized = topic.strip().lower()
    if urgency >= 8:
        return 2  # urgent/escalation path
    if any(word in normalized for word in ("bill", "refund", "payment", "invoice")):
        return 0  # billing path
    return 1  # general support path

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
# Build router tool and branch flows
# ──────────────────────────────────────────────────────────────
router_tool = Tool(
    function=choose_branch,
    name="choose_branch",
    namespace="support",
    description="Return the branch index for the incoming support request.",
)

billing_branch = BasicFlow(
    component=StructuredInvokable(
        component=Tool(
            function=handle_billing,
            name="handle_billing",
            namespace="support",
            description="Handle billing-related issues.",
        ),
        name="billing_branch_component",
        description="Structured billing branch output.",
        output_schema=["response"],
    )
)

general_branch = BasicFlow(
    component=StructuredInvokable(
        component=Tool(
            function=handle_general_support,
            name="handle_general_support",
            namespace="support",
            description="Handle general support issues.",
        ),
        name="general_branch_component",
        description="Structured general support branch output.",
        output_schema=["response"],
    )
)

escalation_branch = BasicFlow(
    component=StructuredInvokable(
        component=Tool(
            function=handle_escalation,
            name="handle_escalation",
            namespace="support",
            description="Handle urgent escalations.",
        ),
        name="escalation_branch_component",
        description="Structured escalation branch output.",
        output_schema=["response"],
    )
)

# ──────────────────────────────────────────────────────────────
# Build the RoutingFlow
# ──────────────────────────────────────────────────────────────
flow = RoutingFlow(
    name="support_router",
    description="Route support requests to one fixed branch based on router output.",
    router=router_tool,
    branches=[billing_branch, general_branch, escalation_branch],
)

# ──────────────────────────────────────────────────────────────
# Example invocations
# ──────────────────────────────────────────────────────────────
examples = [
    {"topic": "Need a refund for duplicate invoice", "urgency": 2},
    {"topic": "The dashboard keeps logging me out", "urgency": 3},
    {"topic": "Production outage affecting all customers", "urgency": 10},
]

for i, payload in enumerate(examples, start=1):
    result = flow.invoke(payload)
    print(f"\n=== Run {i} ===")
    print("Inputs:", payload)
    print("Run ID:", result.run_id)
    print("Selected index:", flow.get_router_decision(result.run_id))
    print("Result:", dict(result))
    checkpoint = flow.get_checkpoint(result.run_id)
    if checkpoint is not None:
        print("Checkpoint metadata:")
        pprint(checkpoint.metadata)

print("\n=== Flow snapshot ===")
pprint(flow.to_dict())
