from __future__ import annotations

# =============================================================================
# ToolAgent2 blackboard-slot-v2 reserved literals
# =============================================================================
# Used by:
# - models/agents/toolagent2_models.py: BlackboardSlotV2.tool default alias
# - utils/toolagent2.py: parse_statement_to_slots hoisting/rhs_assign/return/
#   task-result-reference logic
# - agents/toolagent2.py: render_turn/_initialize_task cross-invocation
#   result addressing (TASK_RESULT_PREFIX)
#
# Reserved namespaces: RHS_ASSIGN_ALIAS/RETURN_ALIAS can never be real
# registered tool aliases; HOISTED_NAME_PREFIX/TASK_RESULT_PREFIX can never
# be a model-chosen identifier. Enforcement points live outside this file's
# scope.

RHS_ASSIGN_ALIAS = "rhs_assign"
HOISTED_NAME_PREFIX = "_HOIST_"
RETURN_ALIAS = "return"
TASK_RESULT_PREFIX = "task_result_"

# Fallback continuation-note text (agents/toolagent2.py's checkpoint-triggered
# reactive continuation): used only when an explicit `# CHECKPOINT` marker is
# not followed by a triple-quoted explanation -- never used for a
# resolution/execution failure, which always surfaces its own real, dynamic
# reason instead of this generic text.
DEFAULT_CONTINUATION_NOTE = (
    "It was deemed necessary to pause code writing and execution to "
    "accurately determine the next steps for completing the task. Review "
    "the work completed so far and continue writing code based on what "
    "you can now reason."
)

__all__ = [
    "RHS_ASSIGN_ALIAS",
    "HOISTED_NAME_PREFIX",
    "RETURN_ALIAS",
    "TASK_RESULT_PREFIX",
    "DEFAULT_CONTINUATION_NOTE",
]
