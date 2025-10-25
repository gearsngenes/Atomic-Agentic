# examples/Workflow_Examples/08_LangGraphFlow_latest_nonnull.py
# Run: python 08_LangGraphFlow_latest_nonnull.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from typing_extensions import Annotated
from typing import Optional
from datetime import datetime, timezone, timedelta

from langgraph.graph import StateGraph
from modules.Workflows import LangGraphFlow
from modules.Tools import Tool

# ───────────────────────── Root reducer: latest non-null wins ─────────────────────────
def latest_non_null(old: dict, new: dict) -> dict:
    """Merge two dict states; overwrite only when new value is not None."""
    merged = dict(old)
    for k, v in new.items():
        if v is not None:
            merged[k] = v
    return merged

SCHEMA = ["name", "email", "phone", "address", "last_seen", "score"]

# ───────────────────────── Tools (all single-param; always return full schema) ─────────
def seed_local_profile(
    name: str = "Alex Doe",
    email: Optional[str] = None,
    phone: Optional[str] = None,
    address: Optional[str] = None,
    last_seen: Optional[str] = None,
    score: Optional[int] = None,
) -> dict:
    """Seed a sparse local record; fill missing schema keys safely."""
    last_seen = last_seen or datetime.now(timezone.utc).isoformat()
    return {
        "name": name,
        "email": email,
        "phone": phone,
        "address": address,
        "last_seen": last_seen,
        "score": score,
    }

def crm_enrich(
    name: str,
    email: Optional[str] = None,
    phone: Optional[str] = None,
    address: Optional[str] = None,
    last_seen: Optional[str] = None,
    score: Optional[int] = None,
) -> dict:
    """Simulate CRM update: maybe a fresher phone, email unknown (None)."""
    newer_phone = "+1-555-0107"     # use None to simulate 'no update'
    newer_email = None
    return {
        "name": name,
        "email": newer_email if newer_email is not None else email,
        "phone": newer_phone if newer_phone is not None else phone,
        "address": address,
        "last_seen": last_seen,
        "score": score,
    }

def geocode_enrich(
    name: str,
    email: Optional[str] = None,
    phone: Optional[str] = None,
    address: Optional[str] = None,
    last_seen: Optional[str] = None,
    score: Optional[int] = None,
) -> dict:
    """Simulate geocoder update: maybe normalized address; preserve others."""
    normalized_address = "123 Maple St, Springfield, IL 62704"  # or None for 'no update'
    return {
        "name": name,
        "email": email,
        "phone": phone,
        "address": normalized_address if normalized_address is not None else address,
        "last_seen": last_seen,
        "score": score,
    }

def finalize_profile(
    name: str,
    email: Optional[str] = None,
    phone: Optional[str] = None,
    address: Optional[str] = None,
    last_seen: Optional[str] = None,
    score: Optional[int] = None,
) -> dict:
    """Finalize: refresh last_seen, compute naive completeness score."""
    now = datetime.now(timezone.utc).isoformat()
    completeness = int(bool(email)) + int(bool(phone)) + int(bool(address))
    new_score = completeness * 10
    return {
        "name": name,
        "email": email,
        "phone": phone,
        "address": address,
        "last_seen": now,
        "score": new_score,
    }

# Wrap as Tools (names become node ids via LangGraphFlow.add_node)
seed_tool    = Tool(name="seed_local_profile", func=seed_local_profile, type="function", source="demo")
crm_tool     = Tool(name="crm_enrich",         func=crm_enrich,         type="function", source="demo")
geo_tool     = Tool(name="geocode_enrich",     func=geocode_enrich,     type="function", source="demo")
final_tool   = Tool(name="finalize_profile",   func=finalize_profile,   type="function", source="demo")

def main():
    # Builder with root-level reducer so branches can write concurrently
    RootState = Annotated[dict, latest_non_null]
    builder = StateGraph(RootState)

    # LangGraphFlow: explicit, uniform schema (required by your class)
    flow = LangGraphFlow(
        name="profile_enrichment",
        description="seed → (crm_enrich | geocode_enrich) → finalize_profile",
        schema=SCHEMA,
        graph=builder,
    )

    # Register nodes (add_node wraps Tools → ToolFlow, enforces single-param + schema equality)
    flow.add_node(seed_tool)
    flow.add_node(crm_tool)
    flow.add_node(geo_tool)
    flow.add_node(final_tool)

    # Diamond
    flow.set_entry_point(seed_tool.name)
    flow.add_edge(seed_tool.name, crm_tool.name)
    flow.add_edge(seed_tool.name, geo_tool.name)
    flow.add_edge(crm_tool.name, final_tool.name)
    flow.add_edge(geo_tool.name, final_tool.name)
    flow.set_finish_point(final_tool.name)

    # Initial record — LangGraphFlow requires all schema keys present, so
    # set optional fields to None when absent.
    initial = {
        "name": "Alex Doe",
        "email": None,
        "phone": None,
        "address": None,
        "last_seen": (datetime.now(timezone.utc) - timedelta(days=7)).isoformat(),
        "score": None,
    }

    result = flow.invoke(initial)

    print("Final packaged result:")
    for k in SCHEMA:
        print(f"  {k}: {result.get(k)}")

if __name__ == "__main__":
    main()