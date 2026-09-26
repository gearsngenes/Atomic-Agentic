"""04_incident_responder.py

ReActAgent as an on-call incident-response assistant: investigates a
reported service outage against a small, deterministic mock monitoring
backend, discovers the real root cause may be an upstream dependency (not
the service that was actually reported), and reacts to a real, legitimate
tool failure -- a restart refused because the target is still in its
cooldown window -- by escalating instead of retrying, rather than treating
the failure as a mistake to recover from blindly.

Unlike 01 (deterministic math, no failures), 02 (delegating to two LLM
sub-agents), and 03 (cross-turn memory), this example is the first to
exercise `fail_fast=False`'s actual reason for existing: a tool call that
fails for a real, in-context business reason, which the model must read
(`YOUR LAST CALL FAILED`) and route around -- never a model mistake, and
never something a fixed upfront plan could have anticipated, since which
service is actually broken and whether a fix is even allowed right now
are both only knowable by looking.
"""
import logging
from pprint import pprint

from atomic_agentic.agents import ReActAgent

from shared_engine import llm_engine

logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------- #
# A small, deterministic mock monitoring backend -- no real API needed,
# same self-contained spirit as this folder's other examples' tools.
# ---------------------------------------------------------------------- #
_SERVICES: dict[str, dict] = {
    "checkout-api": {
        "status": "down", "latency_ms": None, "error_rate": 1.0,
        "depends_on": ["payments-gateway"],
    },
    "payments-gateway": {
        "status": "degraded", "latency_ms": 2400, "error_rate": 0.35,
        "depends_on": [],
    },
    "auth-service": {
        "status": "healthy", "latency_ms": 80, "error_rate": 0.0,
        "depends_on": [],
    },
}
# payments-gateway was restarted 4 minutes ago and is still cooling down --
# restarting it again right now will be refused.
_RESTART_COOLDOWNS: set[str] = {"payments-gateway"}


def check_service_health(service: str) -> dict:
    """Return {"status": "healthy"|"degraded"|"down", "latency_ms": int|None,
    "error_rate": float} for a monitored service. Raises KeyError if the
    given name isn't a monitored service."""
    if service not in _SERVICES:
        raise KeyError(f"{service!r} is not a monitored service.")
    info = _SERVICES[service]
    return {
        "status": info["status"],
        "latency_ms": info["latency_ms"],
        "error_rate": info["error_rate"],
    }


def check_dependencies(service: str) -> list:
    """Return the list of upstream services this service depends on --
    empty if it has none. The real root cause of an outage is often an
    upstream dependency, not the service that was actually reported."""
    if service not in _SERVICES:
        raise KeyError(f"{service!r} is not a monitored service.")
    return list(_SERVICES[service]["depends_on"])


def restart_service(service: str) -> str:
    """Restart a service. Raises RuntimeError if it was restarted too
    recently and is still in its cooldown window -- restart refused, not
    retryable right now."""
    if service not in _SERVICES:
        raise KeyError(f"{service!r} is not a monitored service.")
    if service in _RESTART_COOLDOWNS:
        raise RuntimeError(
            f"{service!r} was restarted within the last 10 minutes and is "
            "still in its cooldown window -- restart refused."
        )
    _SERVICES[service].update(status="healthy", latency_ms=90, error_rate=0.0)
    return f"{service} restarted successfully and is now healthy."


def page_oncall(message: str) -> str:
    """Page the on-call engineer with a message when a service can't be
    remediated directly. Returns a confirmation string."""
    return f"On-call paged: {message}"


responder = ReActAgent(
    name="IncidentResponder",
    namespace="examples",
    description="Investigates and remediates service incidents, escalating when it can't fix something directly.",
    llm_engine=llm_engine,
    # A failed restart here is real, recoverable information -- not a
    # mistake -- so a tolerated failure must redirect the next call
    # (escalate) rather than abort the whole run. This is this family's
    # own default; set explicitly since it's the entire point of this
    # example.
    fail_fast=False,
    tool_calls_limit=8,
    context_enabled=True,
)
responder.register_tools([check_service_health, check_dependencies, restart_service, page_oncall])

task = (
    "The service 'checkout-api' is reporting errors. Investigate and resolve the issue if you "
    "can, or escalate to the on-call engineer if you can't.\n\n"
    "Check the reported service's health first. If it's not healthy, check what it depends on -- "
    "the real problem may be an upstream service, not the one reported. Only attempt to restart "
    "a service once you've identified which one is actually unhealthy.\n\n"
    "If a restart fails, do not retry it -- page the on-call engineer instead, explaining what "
    "you found and why you couldn't fix it directly.\n\n"
    "When you're done, return a short incident summary: what was wrong, what you tried, and how "
    "it was resolved (fixed directly, or escalated)."
)

final_result = responder.invoke({"prompt": task})

print(f"\nIncident summary: {final_result.result}")

record = responder.get_conversation()[-1]
print("\nCalls made:")
pprint(record.statements)

if record.failed_statements:
    print("\nCalls that failed (and were reacted to, not retried):")
    pprint(record.failed_statements)
