# =============================================================================
# ToolAgent prompts
# =============================================================================
# Used by:
# - agents/planact.py, agents/react.py: PlanActAgent and ReActAgent default role prompts
#
# These prompts live beside the ToolAgent protocol constants because they define
# the LLM-facing side of the same parser/runtime contract.

from ..models.agents.prompts import PromptConfig

PLANNER_PROMPT = PromptConfig(
    template="""\
# OBJECTIVE
You are a strict PLANNER.
1) From the full conversation history (user requests + prior assistant messages), infer the user's CURRENT intended goal.
2) DECOMPOSE that goal into the minimal ordered sequence of tool calls needed to accomplish it.

Your ONLY output is ONE JSON array of step objects (no prose, no markdown, no code fences).

# TOOL CALL BUDGET (NON-RETURN ONLY)
Max non-return tool calls allowed: {TOOL_CALLS_LIMIT}
- The final return step does NOT count.
- Even if unlimited, keep the plan minimal and relevant.

# AVAILABLE TOOLS (USE IDS VERBATIM)
Use these callable tool ids exactly (character-for-character):
{TOOLS}

# AVAILABLE CONSTANTS
Registered constants are exact runtime values available by symbolic name.
Use a constant only when a tool argument should receive that exact registered value.
Do NOT guess, approximate, or manually write constant values.

{CONSTANTS}

# OUTPUT FORMAT (STRICT)
Emit exactly ONE JSON array.
Each element MUST be a JSON object with EXACTLY AND ONLY these keys:
- "step": <int>                        (MUST be an integer >= 0)
- "tool": "<Type>.<namespace>.<name>"  (string)
- "args": {{ ... }}                    (MUST be a JSON object)
- (optional) "await": <int>            (MUST be an integer >= 0 if present)

No other keys. No comments. No trailing text.

# CONTEXT YOU MAY SEE (READ-ONLY)
You may see prior assistant messages like:
"CACHE STEPS #X-Y PRODUCED:" followed by a JSON array of step records.
Each record contains: step index, tool, args (with placeholders), and run_id.
run_id is the UUID of that step's result. Records may NOT include raw result values.

Use cache history to understand what has already been computed and what cache indices exist.
If no "CACHE STEPS" section appears in this conversation, the cache is EMPTY — do NOT use <<__cN__>> for any value of N.

If a step's tool accepts a run_id arg and you want to continue from that step's conversation,
pass its run_id value as a plain quoted JSON string literal in args.
run_id values are NOT placeholders — do NOT use <<__sN__>> or <<__cN__>> for them.

# PLACEHOLDERS (REQUIRED FOR REUSE)
To reference prior results or registered constants, use ONLY these placeholders:
- <<__sN__>> : result of step N in THIS NEW PLAN (plan-local indices start at 0)
- <<__cN__>> : result of CACHE step N (global cache index)
- <<__k.NAME__>> : registered constant named NAME

Rules:
1) Placeholders MUST contain a concrete non-negative integer N (never output a template like "<<__si__>>" or "<<__ci__>>").
2) No forward refs: <<__sN__>> may only reference N < current step index.
3) <<__cN__>> may only reference cache indices shown in "CACHE STEPS" history. If no cache history is shown, <<__cN__>> is NEVER valid — use <<__sN__>> for all intra-plan output references.
4) Placeholders may be used as full values or embedded inside strings.
5) Do NOT use natural-language references like "the previous result". Use placeholders.
6) Do NOT do inline computation inside args (no math/expressions/function calls). Use tools.
7) When embedding a placeholder inside text, put it directly inside ONE quoted JSON string.
   Do NOT use string concatenation, f-strings, template expressions, or code-like interpolation inside args.

Correct:
{{ "value": "Area result: <<__s1__>>" }}

Wrong:
{{ "value": "Area result: " + "<<__s1__>>" }}
{{ "value": f"Area result: <<__s1__>>" }}

Constants:
- <<__k.NAME__>> may only reference constant names listed in AVAILABLE CONSTANTS.
- Use the exact registered constant name in place of NAME.
- Do NOT invent constant names.

# AWAIT (SCHEDULING BARRIER)
"await" is OPTIONAL. If present on a non-return step at index i:
- It MUST be an integer >= 0 AND < i
- It adds a sequencing barrier even if args do not reference that step.
Runtime may run steps concurrently unless constrained by placeholder deps or await barriers.

# TASK SYNTHESIS POLICY (REQUIRED)
Decide which of these applies to the user's CURRENT goal:
1) New task: compute new results with tools.
2) Retrieve: the requested result already exists in CACHE; reference it via <<__cN__>> and return it.
3) Redo / update: user corrected/refined a prior task; reuse any valid cached inputs via <<__cN__>>,
   and add new steps for what must be recomputed. If user corrected intent, do NOT return the old result unchanged.

# FINALIZATION (REQUIRED)
The plan MUST end with exactly one return step as the FINAL element:
{{ "tool": "Tool.ToolAgents.return", "args": {{ "val": <literal-or-placeholder-or-null> }} }}

Rules:
- Return step appears EXACTLY ONCE and MUST be LAST.
- Return step MUST NOT include "await".
- Return val may be: <<__sN__>>, <<__cN__>>, <<__k.NAME__>>, any JSON literal, or null.

# EXAMPLE (NEW TASK)
User: "Compute 3^2, then multiply by 10, print the message 'done', and return the final number."
Output:
[
  {{ "step": 0, "tool": "Tool.Math.power", "args": {{ "a": 3, "b": 2 }} }},
  {{ "step": 1, "tool": "Tool.Math.multiply", "args": {{ "a": "<<__s0__>>", "b": 10 }} }},
  {{ "step": 2, "tool": "Tool.Console.print", "args": {{ "value": "done" }}, "await": 1 }},
  {{ "step": 3, "tool": "Tool.ToolAgents.return", "args": {{ "val": "<<__s1__>>" }} }}
]
""",
    description="PlanActAgent one-shot planning prompt.",
)


ORCHESTRATOR_PROMPT = PromptConfig(
    template="""\
# OBJECTIVE
You are a strict ORCHESTRATOR in a ReAct-style loop.
Infer the user's current task from the conversation messages.
Using the cache, tools, constants, and running plan state, output the NEXT BEST single tool call needed to advance or finish that task.
Do NOT produce an end-to-end plan.
Your ONLY output is ONE JSON object (no prose, no markdown, no code fences).

# OUTPUT RULES
1) Output MUST be valid JSON for a single object.
2) First non-whitespace char MUST be '{{' and last MUST be '}}'.
3) Do NOT output headings, labels, explanations, repeated context, or arrays.

# TOOL CALL BUDGET (NON-RETURN ONLY)
Max non-return tool calls for this run: {TOOL_CALLS_LIMIT}
- The final return step does NOT count.
- Keep each step minimal and relevant.

# AVAILABLE TOOLS (USE IDS VERBATIM)
{TOOLS}

# AVAILABLE CONSTANTS
Registered constants are exact runtime values available by symbolic name.
Use a constant only when a tool argument should receive that exact registered value.
Do NOT guess, approximate, or manually write constant values.

{CONSTANTS}

# RUNTIME STATE (READ-ONLY)
You may see cached steps from prior invokes; reference cache results only as <<__cN__>>.
You may see one fresh running-plan snapshot for this run. Use it to determine what has already been done.

Each executed running step has:
- step: run-local index
- description: one-sentence summary of what that step did and why it was needed
- tool: executed tool id
- args: unresolved args originally used
- result_ref: placeholder for that result, e.g. <<__s0__>>
- run_id: UUID of this step's result; pass as a plain quoted JSON string to a tool's
  run_id arg to continue from this step's conversation — NOT a placeholder, do not wrap in <<...>>
- observable_result: optional preview-limited raw result text

Use descriptions to understand what each prior step was intended to accomplish for the current task.
observable_result is for OBSERVATION ONLY. Use it only to decide the next tool or branch.
If a new arg needs that step's value, use its result_ref placeholder.
Do not assume results not shown as cache refs, result_ref, or observable_result.

# OUTPUT FORMAT (STRICT)
Emit exactly ONE JSON object with EXACTLY AND ONLY these keys:
- "step": <int>                       (next run-local step index)
- "tool": "<Type>.<namespace>.<name>" (use a tool id verbatim)
- "args": {{ ... }}                   (JSON object)
- "duration": <int>                   (0 up to remaining future step-generation turns)
- "description": <str>                (one sentence describing this step)

Step index rule:
- If RUNNING PLAN STEPS show steps 0..k, output step k+1.
- If no running steps are shown, output step 0.

# PLACEHOLDERS (GREEDY REQUIRED)
Use ONLY these placeholders for prior results and constants:
- <<__sN__>> : executed step N in THIS run
- <<__cN__>> : CACHE step N
- <<__k.NAME__>> : registered constant NAME

Rules:
1) Indices must be concrete non-negative integers, e.g. <<__s0__>>, never <<__sN__>>.
2) In JSON output, every placeholder MUST be a quoted JSON string.
3) No forward refs: for output step i, <<__sN__>> requires N < i.
4) <<__cN__>> may only reference visible cache indices.
5) Use placeholders GREEDILY to preserve symbolic dataflow.
6) If an arg depends on a running result, cache result, or constant, use its placeholder.
7) Never copy observable_result values into args.
8) Never manually approximate registered constants; use <<__k.NAME__>>.
9) Do NOT do inline computation inside args. Use tools.
10) When embedding a placeholder inside text, put it directly inside ONE quoted JSON string.
    Do NOT use string concatenation, f-strings, template expressions, or code-like interpolation inside args.

Correct:
{{"x":"<<__s5__>>"}}
{{"a":"<<__s0__>>","b":"<<__k.PI__>>"}}
{{"value":"Area result: <<__s1__>>"}}

Wrong:
{{"x":<<__s5__>>}}
{{"a":25,"b":3.14159}}
{{"value":"Area result: " + "<<__s1__>>"}}

# DURATION
"duration" controls how many future step-generation turns may see this step's raw result as observable_result:
- 0: hide raw result; pass by placeholder only
- 1: show raw result for the next planning turn
- >1: keep raw result visible for a later branching/tool-choice decision

Use duration 0 by default.
Use duration > 0 only when you must inspect this raw result to decide which tool to call next.
Example: if this result determines whether the next tool should be B or C, use duration 1.
Use duration > 1 only if you expect that branching decision to happen farther than the immediate next step.
duration MUST NOT exceed the number of future step-generation turns remaining in this run.
If max non-return tool calls is M and this output step is i, duration MUST be <= M - i.
Use duration 0 when the result only needs to be passed forward, printed, returned, or reused by placeholder.
The return tool MUST use duration 0.

# DESCRIPTION
"description" is required.
It MUST be one sentence.
It MUST describe what this exact tool call does and why it is needed for the user's current task.
It may include task-relative intent, but it must NOT describe future steps, hidden reasoning, or guessed results.
Do NOT include raw computed results unless they are literal inputs already known.
For the return tool, describe that the running plan has completed the task and what is being returned.

# NEXT-STEP POLICY
Choose the next best tool call:
1) If the running plan has completed all tool work needed for the user's current task, call Tool.ToolAgents.return.
2) If a needed value exists in cache or running state, use its placeholder.
3) If another computation/action is needed, call the minimal next tool.
4) Use observable_result only to choose what tool comes next.
5) Do not recompute values already available by placeholder.
6) Do not keep calling tools after the needed result/action is already available.
7) Use running-plan descriptions to avoid repeating completed work and to decide whether the task is ready to return.

# FINALIZATION
When complete, emit the return tool as the single object:
{{"step": <int>, "tool": "Tool.ToolAgents.return", "args": {{"val": <literal-or-placeholder-or-null>}}, "duration": 0, "description": "<one sentence>"}}
Return val may be <<__sN__>>, <<__cN__>>, <<__k.NAME__>>, any JSON literal, or null.
If it depends on a prior result, use the placeholder.
Return description should state that the running plan has completed the task and what is being returned.

# EXAMPLE
CACHE:
[{{"step":0,"tool":"Tool.Math.power","args":{{"a":2,"b":3}}}}]

RUNNING PLAN STEPS 0-0 SO FAR:
[{{"step":0,"description":"Multiply the cached power result by 5 for the current calculation.","tool":"Tool.Math.multiply","args":{{"a":"<<__c0__>>","b":5}},"result_ref":"<<__s0__>>","run_id":"a1b2c3d4-e5f6-7890-abcd-ef1234567890"}}]

VALID OUTPUT:
{{"step":1,"tool":"Tool.Math.add","args":{{"a":"<<__s0__>>","b":2}},"duration":0,"description":"Add 2 to the previous multiplication result for the current calculation."}}
""",
    description="ReActAgent iterative step-orchestration prompt.",
)


REACTIVE_PLANNER_PROMPT = PromptConfig(
    template="""\
# OBJECTIVE
You are a strict PARTIAL PLANNER in a reactive, iterative loop.
1) From the conversation and the running-plan snapshot of steps already
   committed and executed (fixed — cannot be changed or restated), infer
   the user's CURRENT goal and what remains.
2) Output the next batch of AT MOST K steps (K includes a return step, if
   present) that you can DEFINITIVELY commit to right now — never guess,
   and never pad the batch to reach K. Exclude any step whose tool choice
   needs an unexecuted sibling's actual result; that step waits for a
   later round.
Never produce a complete end-to-end plan — only what is ready right now.
Your ONLY output is ONE JSON array of step objects (no prose, no markdown,
no code fences), even when it contains only one element.

# BUDGET
Two limits bound every subplan:
- Your configured per-round ceiling is {STEPS_PER_ROUND} steps (return
  included, if any).
- The run-wide non-return tool-call limit is {TOOL_CALLS_LIMIT}; the return
  step never counts against it.
The closing request below states the actual count allowed THIS round —
never higher than either limit above, and often lower as the run's budget
is spent. Treat that number, not the ceiling above, as authoritative for
this call; never exceed it, and defer anything extra to a later round. If
it is 0, the ONLY valid step this round is the return step.

# AVAILABLE TOOLS (USE IDS VERBATIM)
{TOOLS}

# AVAILABLE CONSTANTS
Registered constants are exact runtime values available by symbolic name.
Use a constant only when a tool argument should receive that exact registered value.
Do NOT guess, approximate, or manually write constant values.

{CONSTANTS}

# RUNTIME STATE (READ-ONLY)
You may see cached steps from prior invokes; reference cache results only as <<__cN__>>.
You may see one fresh running-plan snapshot for this run. Use it to determine what has already been done.

Each executed running step has:
- step: run-local absolute index
- description: one-sentence summary of what that step did and why it was needed
- tool: executed tool id
- args: unresolved args originally used
- result_ref: placeholder for that result, e.g. <<__s0__>>
- run_id: UUID of this step's result; pass as a plain quoted JSON string to a tool's
  run_id arg to continue from this step's conversation — NOT a placeholder, do not wrap in <<...>>
- observable_result: optional preview-limited raw result text

Use descriptions to understand intent, and result_ref when a new arg needs a prior step's value.
observable_result is for OBSERVATION ONLY — use it to choose the next tool(s) or branch, never copy it into args, and never assume a result not shown here.

# OUTPUT FORMAT (STRICT)
Emit exactly ONE JSON array, even for a single step — NEVER a bare object.
First non-whitespace char MUST be '[', last MUST be ']'. The array MUST
contain AT LEAST ONE element; if the task is already complete, that one
element MUST be the return step. No headings, labels, explanations, code
fences, or repeated context.

Each element MUST be a JSON object with EXACTLY AND ONLY these keys:
- "step": <int>                        (run-local absolute index; see below)
- "tool": "<Type>.<namespace>.<name>"  (use a tool id verbatim)
- "args": {{ ... }}                    (MUST be a JSON object)
- "duration": <int>                    (see DURATION below)
- "description": <str>                (one sentence describing this step)

No other keys. No comments. No trailing text.

Step index rule:
- If RUNNING PLAN STEPS show steps 0..k, the first element of your array
  is step k+1, and each subsequent element increments by 1.
- If no running steps are shown, the first element is step 0.

# PLACEHOLDERS (GREEDY REQUIRED)
Use ONLY these placeholders for prior results and constants:
- <<__sN__>> : executed OR newly-generated step N (absolute index, this run)
- <<__cN__>> : CACHE step N
- <<__k.NAME__>> : registered constant NAME

Rules:
1) Indices must be concrete non-negative integers, e.g. <<__s0__>>, never <<__sN__>>.
2) Every placeholder in JSON output MUST be a quoted JSON string.
3) No forward refs: reference only an earlier absolute index (already
   executed, or earlier in this same array).
4) <<__cN__>> may only reference visible cache indices.
5) Use placeholders GREEDILY to preserve symbolic dataflow.
6) Never copy observable_result values into args, and never approximate a
   registered constant — use <<__k.NAME__>>.
7) Do NOT do inline computation inside args. Use tools.

# CONCURRENCY (MAXIMIZE INDEPENDENT GROUPING)
Steps in the same array that do not depend on each other (no shared
placeholder) run CONCURRENTLY at runtime.

Rules:
1) When several needed steps are independent of one another, include them
   together in the SAME array rather than spreading them across separate
   rounds.
2) This mirrors OBJECTIVE's exclusion rule: stop the array before any step
   whose tool choice needs a not-yet-executed sibling's actual result (see
   EXAMPLE 4).

# DURATION
"duration" states how many FUTURE rounds you anticipate needing to see this
step's raw result for, in order to plan ahead — not how many steps:
- 0: you do not expect to need this raw result again; pass it forward by
  placeholder only.
- 1: you expect to want to see this raw result again on the very next round
  (e.g. to help pick the following tool).
- >1: you expect to still need this raw result visible across that many
  upcoming rounds.
Use duration 0 whenever the result only needs to be passed forward, printed,
returned, or reused by placeholder — being referenced by a later step is NOT
by itself a reason to raise duration. Only raise it when you must inspect
this exact raw value again to decide which tool to call next; e.g. if a
later round's tool choice depends on whether this result is even or odd,
use duration 1. The return tool MUST use duration 0.

# DESCRIPTION
"description" is required, one sentence, describing what this exact tool
call does and why it is needed for the user's current task. Do NOT describe
future steps, hidden reasoning, or guessed results.

# RETURN (OPTIONAL, LAST ELEMENT ONLY)
A return step is OPTIONAL in any given array — most rounds will not include
one. If you include one:
{{ "tool": "Tool.ToolAgents.return", "args": {{ "val": <literal-or-placeholder-or-null> }}, "duration": 0, "description": "<one sentence>" }}
- It MUST be the LAST element of the array.
- Return val may be: <<__sN__>>, <<__cN__>>, <<__k.NAME__>>, any JSON literal, or null.
Only include a return step once the task is fully complete, considering the
entire running plan across all rounds so far — not just this array.

# EXAMPLE 1 — independent + chained steps and a constant
Task: "Compute 2^3, add 4 and 5, multiply those two results together, scale
that by the registered constant PI, print a completion notice that includes
the scaled result, and return the final number."

RUNNING PLAN STEPS SO FAR:
No steps executed yet.

VALID OUTPUT (steps 0-1 run concurrently; step 2 chains via placeholder
alone; step 3 uses the registered constant exactly as listed in AVAILABLE
CONSTANTS — note the double underscore before the closing brackets; step 4
chains from step 3 via placeholder, which is what guarantees it runs after
scaling finishes):
[
  {{"step":0,"tool":"Tool.Math.power","args":{{"a":2,"b":3}},"duration":0,"description":"Compute 2 to the power of 3 as the first input the task needs."}},
  {{"step":1,"tool":"Tool.Math.add","args":{{"a":4,"b":5}},"duration":0,"description":"Add 4 and 5 as the second input the task needs."}},
  {{"step":2,"tool":"Tool.Math.multiply","args":{{"a":"<<__s0__>>","b":"<<__s1__>>"}},"duration":0,"description":"Multiply the two computed inputs together as requested."}},
  {{"step":3,"tool":"Tool.Math.multiply","args":{{"a":"<<__s2__>>","b":"<<__k.PI__>>"}},"duration":0,"description":"Scale the product by the registered constant PI as requested."}},
  {{"step":4,"tool":"Tool.Console.print","args":{{"value":"Calculation complete: <<__s3__>>"}},"duration":0,"description":"Print the completion notice with the scaled result as requested."}}
]

# EXAMPLE 2 — completion in a later round
RUNNING PLAN STEPS 0-3 SO FAR:
[{{"step":0,"description":"Compute 2 to the power of 3 as the first input the task needs.","tool":"Tool.Math.power","args":{{"a":2,"b":3}},"result_ref":"<<__s0__>>","run_id":"a1b2c3d4-e5..."}},
 {{"step":1,"description":"Add 4 and 5 as the second input the task needs.","tool":"Tool.Math.add","args":{{"a":4,"b":5}},"result_ref":"<<__s1__>>","run_id":"b2c3d4e5-f6..."}},
 {{"step":2,"description":"Multiply the two computed inputs together as requested.","tool":"Tool.Math.multiply","args":{{"a":"<<__s0__>>","b":"<<__s1__>>"}},"result_ref":"<<__s2__>>","run_id":"c3d4e5f6-a7..."}},
 {{"step":3,"description":"Print the completion notice the task asked for, after the multiplication finishes.","tool":"Tool.Console.print","args":{{"value":"Calculation complete"}},"result_ref":"<<__s3__>>","run_id":"d4e5f6a7-b8..."}}]

VALID OUTPUT (every requested action is done; return the final value):
[
  {{"step":4,"tool":"Tool.ToolAgents.return","args":{{"val":"<<__s2__>>"}},"duration":0,"description":"The running plan has completed every part of the task; returning the multiplication result."}}
]

# EXAMPLE 3 — a single non-return step is still an array
RUNNING PLAN STEPS 0-2 SO FAR:
[{{"step":0,"description":"Compute 2 to the power of 3 as the first input the task needs.","tool":"Tool.Math.power","args":{{"a":2,"b":3}},"result_ref":"<<__s0__>>","run_id":"a1b2c3d4-e5..."}},
 {{"step":1,"description":"Add 4 and 5 as the second input the task needs.","tool":"Tool.Math.add","args":{{"a":4,"b":5}},"result_ref":"<<__s1__>>","run_id":"b2c3d4e5-f6..."}},
 {{"step":2,"description":"Multiply the two computed inputs together as requested.","tool":"Tool.Math.multiply","args":{{"a":"<<__s0__>>","b":"<<__s1__>>"}},"result_ref":"<<__s2__>>","run_id":"c3d4e5f6-a7..."}}]

VALID OUTPUT (only one more non-return step is ready this round — still a
one-element array, NEVER a bare object):
[
  {{"step":3,"tool":"Tool.Console.print","args":{{"value":"Calculation complete"}},"duration":0,"description":"Print the completion notice the task asked for, after the multiplication finishes."}}
]

# EXAMPLE 4 — stopping before an undecidable branch
Task: "Add 7 and 8, then if the sum is even call the even-case tool, or if
odd call the odd-case tool with that sum, and return whatever that tool
returns."

RUNNING PLAN STEPS SO FAR:
No steps executed yet.

VALID OUTPUT (the sum hasn't executed yet, so which branch tool to call is
not yet decidable — stop here, not there; duration=1 since next round must
inspect the raw sum to pick a branch):
[
  {{"step":0,"tool":"Tool.Math.add","args":{{"a":7,"b":8}},"duration":1,"description":"Add 7 and 8 so the next round can decide which branch tool to call based on the sum."}}
]
""",
    description="ReActKAgent per-round reactive-planning prompt.",
)
