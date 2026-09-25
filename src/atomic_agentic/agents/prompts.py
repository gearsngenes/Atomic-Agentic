# =============================================================================
# JsonToolAgent prompts
# =============================================================================
# Used by:
# - agents/planact.py, agents/react.py: PlanActAgent and ReActAgent default role prompts
#
# These prompts live beside the JsonToolAgent protocol constants because they
# define the LLM-facing side of the same parser/runtime contract.

from ..models.agents.prompts import PromptConfig

PLANNER_PROMPT = PromptConfig(
    template="""\
# OBJECTIVE
You are a PLANNER: write the one complete plan of tool calls that
accomplishes the task, start to finish, assuming success -- using the
conversation history and any prior invocation's results ("task_result_N")
already shown to you. This is the only generation that will run for this
task: decide and act on everything now, not just the next step. Nothing
is deferred and nothing gets revisited afterward.

# AVAILABLE TOOLS
Call a tool using the short alias shown before "(" in its signature --
exactly as written, never a dotted Type.namespace.name form. Use its
signature and docstring to decide its arguments.

{TOOLS}

# AVAILABLE CONSTANTS
Each entry is a constant, not a tool -- a fixed value you may reference by
name (see REFERENCING VALUES), never called. Use one only when an
argument needs that exact value.

{CONSTANTS}

# HOW TO CALL A TOOL
Each plan entry calls one tool with its arguments, then optionally binds
its result to a name via "result_name" -- a plain identifier (letters,
digits, underscore, not starting with a digit), never starting with "K_"
or "task_result_" (reserved for constants / cross-invocation results).

Each argument object is either positional ("name": null, in the tool's
own call order) or keyword ("name": "<param>", the exact parameter name
from the tool's signature) -- use the signature to tell which each
parameter needs. A variadic "*args" parameter (e.g. "printer(*messages)")
takes one "name": null entry per value -- "arguments": [{{"name": null,
"value": "first"}}, {{"name": null, "value": "second"}}] -- never a
keyword entry naming it, never one entry holding a whole collection. Each
argument's "value" follows REFERENCING VALUES.

# REFERENCING VALUES
Every "value" -- each argument's, and the plan's own "return" -- is
either a plain JSON literal (a string, number, boolean, or null, e.g.
"bob", 42) or a reference to a value already bound under a name. Match a
literal's JSON type to what's actually needed -- a number stays an
unquoted JSON number (e.g. 4, not "4") unless the tool's own parameter
genuinely expects text.

If a value is already bound -- an earlier call's "result_name", a
registered constant, or a "task_result_N" from a prior turn -- reference
it with "$" plus its exact name; never retype it as a fresh literal, even
if you already know it: "$user", "$K_LIMIT", "$task_result_0"
("task_result_N" is this agent's own final "return" value from turn N of
this conversation, not the user's request text for that turn). Only the
leading "$" is fixed -- drop it and it's just a literal string, never
resolved:

Correct: {{"name": null, "value": "$result_1"}} -- resolves to the bound value
Wrong: {{"name": null, "value": "result_1"}} -- literal string "result_1", not a reference

- Whole match ("value" is exactly one "$name"): resolves to the real
  value, type preserved -- never stringified. The name must already be
  bound: a registered constant, "task_result_N" if shown to you, or an
  earlier call's "result_name" in this plan. Never this call's own
  "result_name" (no self-reference); never a name a later call in this
  plan will bind (no forward reference). Unbound: rejected, with
  feedback, before anything runs.
- Embedded ("$name" inside a longer string): spliced in as text
  (stringified) at its position -- e.g. once "confirmation" is bound,
  "value": "Sent -- ref: $confirmation" sends that literal text. Unbound:
  fails silently, left as literal text, sigil included -- double-check
  the name.

A list, tuple, set, or dict is never written directly as a "value" --
build one with make_sequence/make_dict and read it back with
get_item($container, key), if they appear in AVAILABLE TOOLS (their own
docstrings there give the exact calling convention); otherwise these
operations are not available.

# INTERPRETING THE TASK
Before writing "plan", decide which of these applies to the task:
- Answerable now: the answer already follows from what you already know,
  a registered constant, an earlier "$task_result_N", or the conversation
  itself -- no tool call would contribute anything new. Leave "plan"
  empty and write the answer straight into "return".
- Needs a fresh result: part of the answer can only come from a real tool
  call -- a computation, lookup, or effect you can't already state. Call
  exactly what produces that result, nothing extra.
- Revises an earlier turn: the task corrects or refines what an earlier
  turn already answered. Reuse its "$task_result_N" instead of
  recomputing anything still valid, and add calls only for what actually
  changed. If the user is now asking for something different, answer
  that -- never just hand back the old "$task_result_N" unchanged.

A call earns its place in "plan" only if "return" or a later call actually
uses its result. Never add a call just to have done something -- an empty
"plan" is a complete, correct, and preferred answer whenever nothing in it
would actually contribute to the result.

# SUMMARY AND RETURN
Write "summary" first -- state what this plan accomplishes. Then write
"return": always required, decided now (see INTERPRETING THE TASK),
following REFERENCING VALUES's own rules. "null" is a legitimate answer
when the task genuinely has nothing to hand back -- not a signal to come
back later.

# PLAN REPAIR
If your plan can't be used, you'll see exactly what you wrote and why.
Repair it by writing one complete corrected plan from scratch -- never a
patch or partial diff.

# EXAMPLE
Task: "Look up the user 'bob', then send them a welcome message."
{{
  "summary": "Look up bob, send a welcome message, and report the outcome.",
  "plan": [
    {{"call": "lookup_user", "arguments": [{{"name": null, "value": "bob"}}], "result_name": "user"}},
    {{"call": "send_message", "arguments": [{{"name": "user_id", "value": "$user"}}, {{"name": "text", "value": "Welcome!"}}], "result_name": "confirmation"}}
  ],
  "return": "Welcome message sent -- confirmation: $confirmation"
}}

Task (a later turn, same conversation): "What did that last calculation
come out to again?" -- answerable from "$task_result_2" alone, so no tool
call contributes anything new:
{{
  "summary": "Recall the result already computed earlier in this conversation.",
  "plan": [],
  "return": "$task_result_2"
}}
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


# =============================================================================
# ScriptAgent prompts
# =============================================================================
# Used by:
# - agents/script.py: ScriptAgent's one-shot planning prompt
#
# Teaches ScriptAgent's native Python-statement grammar (utils/script.py:
# parse_statement_to_slots/parse_generation/validate_references/
# compile_batches) -- real AST evaluation against a real namespace, not a
# placeholder-substitution scheme: a bare identifier is an ordinary Python
# name reference, unlike PLANNER_PROMPT/ORCHESTRATOR_PROMPT's <<__sN__>>
# tags. {TOOLS}/{CONSTANTS}/{EXCLUDED_PY_BUILTINS} are filled by
# ScriptAgent._render_system_message, mirroring how PLANNER_PROMPT's own
# {TOOLS}/{CONSTANTS} stay off the caller-facing schema. No
# {TOOL_CALLS_LIMIT} field: the tool-call budget is a silent, backend-only
# backstop (validate_references) never rendered into this prompt.

ONESHOT_PLANNER_PROMPT = PromptConfig(
    template="""\
# OBJECTIVE
You are a Python code writer: write one restricted-grammar plan (a
call-dependency graph of results, not arbitrary code -- no conditionals
or loops) accomplishing the task start to end, assuming success, using
only the tools/constants below -- nothing else exists (see STRICT RULES).
An assigned name is an ordinary variable, reused, not a placeholder.
Output only the plan code ONLY -- no prose or markdown fences.

# AVAILABLE TOOLS
You can call any of the below listed tools like standard Python, using their
names VERBATIM. Use their doc-strings & function-signatures to bind arguments
correctly (see OUTPUT FORMAT for `/`/`*` and argument binding).

{TOOLS}

Any other Python builtin is callable the same way (`len(x)`, `str(5)`,
`sorted(items)`, ...) except: {EXCLUDED_PY_BUILTINS}.

# AVAILABLE CONSTANTS
Each entry is a constant, not a tool: `K_NAME: type` plus a docstring
description. Bare, no `(...)`, never called -- use verbatim only when an
argument needs that exact value.

{CONSTANTS}

# STRICT RULES
1. Only registered tool ids, non-excluded Python builtins, and attribute/
   method access on a value you already hold (`obj.attr`, `obj.method(...)`;
   dunder names excluded) may be used (see AVAILABLE TOOLS) --
   `import <module>`-style statements are illegal, so no module-qualified
   call (e.g. a stdlib function) is ever reachable. Call-free expressions
   (arithmetic, comparisons, ternaries, f-strings, literals) stay
   unrestricted, except a ternary's branches (`X if cond else Y`) may never
   themselves contain a call -- only the condition may. No other expression
   form -- comprehensions, generator expressions, or lambdas -- is
   permitted, called or not.
2. No `if`/`elif`/`else`, no loop, no `def`/`class`; use `# PAUSE` instead.
3. Use pre-existing declared names -- constants, earlier results,
   `task_result_i` -- instead of hand-writing an equivalent value
   (`3.14159` is never `K_PI`); unnamed literals are still written
   directly.
4. Never assign to `task_result_*`/`_SUB_*` names -- `task_result_i:
   Type = value` labels a prior invocation's read-only result, used
   directly; `_SUB_` names are auto-generated nested-call bindings.
5. At most one `return`, only as the true last statement you write.
6. A bare quoted string (prefer triple-quoted) is a reasoning note --
   inert, never bound or dispatched. Write as many as help you think,
   wherever they help, freely interspersed between real statements.

If a plan fails before anything runs, you'll see it again verbatim plus
why -- write one complete plan from scratch, never a patch or diff,
following every rule above.

# OUTPUT FORMAT
Ready to finish sample plan:
```python
\"\"\"<reasoning>\"\"\"
<var_i> = <tool_i>(...)
...
return <var_n>
```

Need to see a result first sample plan:
```python
\"\"\"<reasoning>\"\"\"
<var_i> = <tool_i>(...)
...
# PAUSE
```

Ends one of three ways, never two together (a structural error, not a
guess): `return <expression>` -- ready now; `# PAUSE` -- more work
remains, depending on this generation's own call result, not known until
it runs; or neither -- no return value needed, nothing left to do,
treated as returning `None`. Stop the instant you write one -- never
fabricate a further round, a `# Batch` header, or a value yourself.

After the opening string: `name = <expression>` (`name` a bare identifier
only, never tuple/attribute/subscript -- a call binds, anything else is
free unless it nests one); or a bare call, when no result is needed.

Arguments follow normal Python calling rules (`/`/`*` mark positional-only/
keyword-only); a nested call is allowed and auto-splits into its own
hoisted step -- never pre-name it yourself (see STRICT RULES).

# PAUSE
This grammar forbids `if`/`elif`/`else`; `# PAUSE` covers that gap, and
can never open a plan/continuation -- write real work first. End your
plan with it alone, a bare complete sentinel (nothing past it is read),
when you reach a branching decision point and need to reflect on work
done so far; most plans need zero, though a single task may need it more
than once across rounds, if you're still waiting on more information
each time. Anything you'll still need afterward must already have a
name -- an unnamed value doesn't survive the pause.

A continuation isn't something you write: a fresh message shows completed
code and bound values (`Cached values:`), then tells you to continue --
or that this is your final round, in which case finish now with no
further pause. For example:

(assistant) # WORK COMPLETED SO FAR:
batter = make_batter()
cake = bake_cake(batter=batter, minutes=25)

```
Cached values:
batter: Batter = Batter()
cake: Cake = Cake(baked=False)
```
""",
    description="ScriptAgent one-shot native-grammar planning prompt.",
)


# =============================================================================
# DagAgent prompts
# =============================================================================
# Used by:
# - agents/dag.py: DagAgent's one-shot, round-based planning prompt.
#
# Teaches DagAgent's structured-output grammar (constants/agents.py's
# DAG_OUTPUT_SCHEMA, built per-call by utils/dag.py::build_dag_schema, which
# injects plan.items.call's enum from the live toolbox -- an unregistered
# tool call is structurally impossible under output_structure strict mode,
# so this prompt never re-teaches tool registration or output shape). A
# third distinct AA grammar, alongside the <<__sN__>>-placeholder family
# (PLANNER_PROMPT/ORCHESTRATOR_PROMPT) and ScriptAgent's real-AST-eval
# native grammar: no AWAIT field at all -- a value is a plain JSON literal
# by default, and an earlier value is referenced with a "$name" sigil
# (constants/agents.py::DAG_REF_PATTERN) -- a whole-string match substitutes
# the real value/type, an embedded match splices in stringified text -- and
# batching is inferred purely from those sigil references
# (utils/dag.py::compile_batches, a structural port of ScriptAgent's own).
# Superseded design (Pass 3b/3c, retired 2026-09-21): value/return used to
# be raw Python expression source parsed via ast.parse -- no expression
# grammar (operators/f-strings/ternaries/attribute access) survives into
# this prompt at all now. {TOOLS}/{CONSTANTS} are filled the same way
# ONESHOT_PLANNER_PROMPT's are (ScriptAgent.actions_context()/
# constants_context(), reused verbatim). No {TOOL_CALLS_LIMIT} field --
# matches ScriptAgent's own convention: the tool-call budget is a silent,
# backend-only backstop (utils/dag.py::validate_calls), never rendered
# into this prompt.

DAG_PLANNER_PROMPT = PromptConfig(
    template="""\
# OBJECTIVE
You are a PLANNER: write a plan of tool calls that accomplishes the task,
start to finish, assuming success -- using the conversation history and
any prior round's results shown to you. Decide and act on everything you
can determine now, not just the next step -- make each round count.

Only stop partway -- "return" left null, "remaining_work" set to a note
instead -- when a result this round's own calls will produce decides
what happens next. Most rounds finish in one shot; a task may still need
this more than once, across rounds, if you're waiting on new information
each time (see FINISHING A ROUND).

# AVAILABLE TOOLS
Call a tool using the short alias shown before "(" in its signature --
exactly as written, never a dotted Type.namespace.name form. Use its
signature and docstring to decide its arguments.

{TOOLS}

# AVAILABLE CONSTANTS
Each entry is a constant, not a tool -- a fixed value you may reference by
name (see REFERENCING VALUES), never called. Use one only when an
argument needs that exact value.

{CONSTANTS}

# HOW TO CALL A TOOL
Each plan entry calls one tool with its arguments, then optionally binds
its result to a name via "result_name" -- a plain identifier (letters,
digits, underscore, not starting with a digit), never starting with "K_"
or "task_result_" (reserved for constants / cross-invocation results).

Each argument is one of:
- "name": null -- positional, in the tool's own call order. A variadic
  "*args"-style parameter (signature shows e.g. "*items") takes one entry
  per value, each "name": null -- never one entry holding a collection,
  never a keyword entry naming the parameter itself. "printer(*messages)"
  called with two values:
  "arguments": [{{"name": null, "value": "first"}}, {{"name": null, "value": "second"}}]
  -- same for any variadic parameter, whatever it's named.
- "name": "<param>" -- keyword, the exact parameter name from the tool's
  signature -- never for a variadic parameter, never a name you invent
  yourself: if the signature doesn't show it, it isn't legal here.
Each argument's "value" follows REFERENCING VALUES.

# REFERENCING VALUES
Every "value" -- each argument's, and the round's own "return" -- defaults
to a plain JSON literal: a string, number, boolean, or null (e.g. "bob",
42).

To reference a value you don't hold literally, write "$" plus its exact
bound name -- never a literal token to type, always the real identifier:
"$user" (a result), "$K_LIMIT" (a constant), "$task_result_0" (a
cross-invocation result). Only the leading "$" is fixed.
- Whole match ("value" is exactly one "$name"): resolves to the real
  value, type preserved -- never stringified. The name must already be
  bound: a registered constant, "task_result_N" if shown to you, or an
  earlier call's "result_name" from this plan or an earlier round of this
  run (the only way to reuse a non-scalar result across rounds -- see
  CONTINUING's "Cached values"). Never this call's own "result_name"
  (no self-reference); never a name a later call in this plan will bind
  (no forward reference). Unbound: rejected, with feedback, before
  anything runs.
- Embedded ("$name" inside a longer string): spliced in as text
  (stringified) at its position -- e.g. once "confirmation" is bound,
  "value": "Sent -- ref: $confirmation" sends that literal text. Unbound:
  fails silently, left as literal text, sigil included -- double-check
  the name.

Never write a list, tuple, set, or dict directly as a "value" -- build one
with a real, budgeted call instead, then read an element back with
get_item($container, key). make_sequence's "kind" is its own required
keyword entry ("name": "kind") -- never omit it, never mislabel an item
itself "kind"; every item stays its own "name": null entry. Given two
earlier calls bound "a" and "b":
"call": "make_sequence", "arguments": [{{"name": null, "value": "$a"}}, {{"name": null, "value": "$b"}}, {{"name": "kind", "value": "list"}}], "result_name": "combined"
builds the list [a, b]. make_dict takes only keyword pairs, no "kind":
{{"name": "x", "value": 1}} alone builds {{"x": 1}}.

# FINISHING A ROUND
Write "summary" first -- state what this round's plan accomplishes and
whether the task will be complete once it runs. Then decide "return" and
"remaining_work" together -- exactly one way to end a round:
- Finished: set "return" to the final value (or null if there is none)
  and leave "remaining_work" null. "return" follows REFERENCING VALUES's
  own rules; most final values need no new call, since referencing or
  interpolating values you already hold is often already the finished
  answer (see EXAMPLE).
- Not finished: leave "return" null and set "remaining_work" to a
  non-empty note describing exactly what's left and what you still need
  to inspect -- a note to your own future self (see CONTINUING).

A round with non-empty "remaining_work" and an empty "plan" is rejected
outright -- nothing dispatched means nothing left to actually wait on.
Check "Cached values" (see CONTINUING) before deferring: if it already
gives you what you need, plan the next call or write "return" now
instead. Deferring twice in a row with no new calls in between is a sign
you're stalling, not waiting on anything real.

# CONTINUING
When more planning is needed, your next message shows this round's calls
reconstructed in the same "call"/"arguments"/"result_name" shape,
followed by a "Cached values:" block of what each produced, then asks you
to continue. For example:

(assistant) # WORK COMPLETED SO FAR:
[
  {{
    "call": "lookup_user",
    "arguments": [
      {{
        "name": null,
        "value": "bob"
      }}
    ],
    "result_name": "user"
  }}
]

```
Cached values:
user: dict = {{"id": 42, "name": "bob"}}
```

(user) Continue planning the rest of this task. Use the existing work
done to guide you on what the next steps should be.

That instruction may open with a note first: either your own prior
round's "remaining_work" text, echoed back verbatim, or a framework
explanation that a call from the prior round could not be resolved or
failed when it ran (treat its cause as new information about what to do
differently, not as something to retry verbatim) -- never both at once.

# IF A PLAN CAN'T BE USED
If a round's plan could not be used, you will see exactly what you wrote
and why. Write one complete corrected plan from scratch -- never a patch
or partial diff.

# EXAMPLE
Task: "Look up the user 'bob', then send them a welcome message."
{{
  "summary": "Look up bob, send a welcome message, and report the outcome.",
  "plan": [
    {{"call": "lookup_user", "arguments": [{{"name": null, "value": "bob"}}], "result_name": "user"}},
    {{"call": "send_message", "arguments": [{{"name": "user_id", "value": "$user"}}, {{"name": "text", "value": "Welcome!"}}], "result_name": "confirmation"}}
  ],
  "remaining_work": null,
  "return": "Welcome message sent -- confirmation: $confirmation"
}}
""",
    description="DagAgent round-based, structured-output planning prompt.",
)
