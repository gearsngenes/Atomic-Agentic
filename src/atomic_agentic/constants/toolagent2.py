from __future__ import annotations

# =============================================================================
# ToolAgent2 blackboard-slot-v2 reserved literals
# =============================================================================
# Used by:
# - models/agents/toolagent2_models.py: BlackboardSlotV2.tool default alias
# - utils/toolagent2.py: parse_statement_to_slots hoisting/rhs_assign logic
#
# Reserved namespaces: RHS_ASSIGN_ALIAS can never be a real registered tool
# alias; HOISTED_NAME_PREFIX can never be a model-chosen identifier. Both
# enforcement points live outside this file's scope.

RHS_ASSIGN_ALIAS = "rhs_assign"
HOISTED_NAME_PREFIX = "_HOIST_"

__all__ = [
    "RHS_ASSIGN_ALIAS",
    "HOISTED_NAME_PREFIX",
]
