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

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
# Services whose health has actually been checked this run -- enforced by
# restart_service below, not just advised in prose. A written-down
# investigation order is not reliable enough on its own across every model;
# making the requirement a real, catchable failure lets this family's own
# reactive-correction path (see the module docstring) do the enforcing.
_HEALTH_CHECKED: set[str] = set()


def check_service_health(service: str) -> dict:
    """Return {"status": "healthy"|"degraded"|"down", "latency_ms": int|None,
    "error_rate": float} for a monitored service. Raises KeyError if the
    given name isn't a monitored service."""
    if service not in _SERVICES:
        raise KeyError(f"{service!r} is not a monitored service.")
    _HEALTH_CHECKED.add(service)
    info = _SERVICES[service]
    return {
        "status": info["status"],
        "latency_ms": info["latency_ms"],
        "error_rate": info["error_rate"],
    }


def check_dependencies(service: str) -> list:
    """Return the list of upstream services this service depends on --
    empty if it has none. After calling this, check the health of EACH
    returned service too, with check_service_health -- one of them, not
    the service you started with, may be the real root cause. Getting
    this list is not enough on its own to decide what's actually broken."""
    if service not in _SERVICES:
        raise KeyError(f"{service!r} is not a monitored service.")
    return list(_SERVICES[service]["depends_on"])


def restart_service(service: str) -> str:
    """Restart a service. Raises RuntimeError in three cases, checked in
    this order: (1) this service's own health hasn't been checked yet
    this run -- call check_service_health on it first; (2) it has a
    dependency whose health hasn't been checked yet -- the real problem
    is often there, not in the service you're about to restart; (3) it
    was restarted too recently and is still in its cooldown window --
    restart refused, not retryable right now."""
    if service not in _SERVICES:
        raise KeyError(f"{service!r} is not a monitored service.")
    if service not in _HEALTH_CHECKED:
        raise RuntimeError(
            f"You haven't checked {service!r}'s own health yet -- call "
            "check_service_health on it before restarting it."
        )
    unchecked_deps = [d for d in _SERVICES[service]["depends_on"] if d not in _HEALTH_CHECKED]
    if unchecked_deps:
        raise RuntimeError(
            f"{service!r} depends on {unchecked_deps!r}, whose health you "
            "haven't checked yet -- check each dependency's health before "
            "restarting anything; the real problem may be there instead."
        )
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
    "Check the reported service's health first. If it's not healthy, then check EACH DEPENDENCY -- "
    "the real problem may be an upstream service, not the one reported. Only attempt to restart "
    "a service once you've identified which one is actually unhealthy.\n\n"
    "If a restart fails, do not retry it -- page the on-call engineer instead, explaining what "
    "you found and why you couldn't fix it directly.\n\n"
    "When you're done, call return with one plain sentence you write yourself, summarizing what "
    "was wrong and how it was resolved -- do not try to combine several bound values into one "
    "value with '+'; just describe the outcome in your own words."
)

final_result = responder.invoke({"prompt": task})

print(f"\nIncident summary: {final_result.result}")

record = responder.get_conversation()[-1]
print("\nCalls made:")
print(record.render_as_code())

if record.failed_statements:
    print("\nCalls that failed (and were reacted to, not retried):")
    pprint(record.failed_statements)
