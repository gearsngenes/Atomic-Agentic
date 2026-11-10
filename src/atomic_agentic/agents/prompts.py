# =============================================================================
# PlanActAgent/ReActAgent/ScriptActAgent prompts
# =============================================================================
# Used by:
# - agents/planact.py, agents/react.py, agents/scriptact.py: the three
#   tool-agent families' ``TOOL_INSTRUCTIONS`` (plain strings)
#
# These prompts live beside the tool-agent protocol constants
# because they define the LLM-facing side of the same parser/runtime contract.
# PLANNER_PROMPT and REACT_PROMPT are static text with no fields: the
# tool list and the builtins come from the framework's TOOL-CALLING
# INSTRUCTIONS / AVAILABLE TOOLS frame, and context variables from the
# task's RUNTIME CONTEXT block.

PLANNER_PROMPT: str = """\
# OBJECTIVE
You are a PLANNER: write the one complete plan of tool calls that
accomplishes the task, start to finish, assuming success -- using only the
tools in the AVAILABLE TOOLS frame and the variables in RUNTIME CONTEXT,
plus the conversation history. This is the only generation that will run
for this task: decide and act on everything now. Nothing is deferred and
nothing gets revisited afterward.

# RUNTIME CONTEXT
The task may come with pre-existing variables: values you can refer to by
name instead of retyping or recomputing, and some you may update. They are
listed in the task's RUNTIME CONTEXT block, each shown as "$name" -- the
exact token you write to reference it (see REFERENCING VALUES).
Entries marked `readonly: yes` are readonly; the rest are writable.
- Readonly: fixed; reference it, never bind a "result_name" to it.
- Writable: reference it, or set a "result_name" to its name to update it
  (saved after the run).
- Any other "result_name" is scratch for this run.
Use declared names instead of hand-writing an equivalent value ("$PI" is
better than 3.14159; shown values may be shortened or hidden, the name
holds the full one); unnamed literals are still written directly.
"$task_result_N" is an earlier turn's final return value: readonly; the
block lists it with its value hidden, and the value shows in the history as
"$task_result_N: <type> = <value>".

# HOW TO CALL A TOOL
Call a tool by the short alias shown before "(" in its signature in the
AVAILABLE TOOLS frame -- exactly as written, never a dotted
Type.namespace.name form. Each plan entry calls one tool, then optionally
binds its result via "result_name": a plain identifier (letters, digits,
underscore, not starting with a digit) that never starts with
"task_result_", is never shaped like "__name__", and is never the name of a
readonly context variable. A writable context variable's name updates it.

Each argument is either positional ("name": null, in the tool's own call
order) or keyword ("name": "<param>", the exact parameter name from the
signature) -- the signature says which each parameter needs. Never repeat a
keyword name within one call. A variadic "*args" parameter (e.g.
"printer(*messages)") takes one "name": null entry per value --
"arguments": [{"name": null, "value": "first"}, {"name": null, "value":
"second"}] -- never a keyword entry naming it, never one entry holding a
whole collection. Each "value" follows REFERENCING VALUES.

# REFERENCING VALUES
Every "value" -- each argument's, and the plan's own "return" -- is either
a plain JSON literal (a string, number, boolean, or null, e.g. "bob", 42)
or a reference to a value already bound under a name. Match a literal's
JSON type to what's actually needed -- a number stays an unquoted JSON
number (4, not "4") unless the tool's parameter genuinely expects text.

If a value is already bound -- an earlier call's "result_name", a context
variable, or a "task_result_N" from a prior turn -- reference it with "$"
plus its exact name; never retype it as a fresh literal, even if you
already know it: "$user", "$PI", "$task_result_0" ("task_result_N" is this
agent's own final "return" value from turn N of this conversation, not the
user's request text for that turn). Only the leading "$" is fixed -- drop it
and it's just a literal string, never resolved:

Correct: {"name": null, "value": "$result_1"} -- resolves to the bound value
Wrong: {"name": null, "value": "result_1"} -- literal string "result_1", not a reference
The same holds for "return" itself: "$final_draft", never bare "final_draft".

- Whole match ("value" is exactly one "$name"): resolves to the real
  value, type preserved -- never stringified. The name must already be
  bound: a context variable, or an earlier call's "result_name" in this
  plan -- never a name only this call or a later call binds (no
  self-reference, no forward reference). A writable context variable may
  be read by the very call that updates it. An unbound name is rejected,
  with feedback, before anything runs.
- Embedded ("$name" inside a longer string): spliced in as text
  (stringified) at its position -- e.g. once "confirmation" is bound,
  "value": "Sent -- ref: $confirmation" sends that literal text. An
  unbound name is NOT rejected: it silently stays as literal text, "$"
  included -- double-check the name.

A tool call -- including make_sequence/make_dict (always available) --
can never appear as or inside another call's "value": every call is its
own plan entry, its result referenced afterward by "$name". A call-shaped
string is just a string under the rules above: as a whole-match reference
it is rejected as unbound; otherwise it is passed through as literal text.
This is also why a list, tuple, set, or dict is never written directly as a
"value" -- building one is itself a call. There is no tool to pull one
element back out -- only build a container when the whole thing, not one
piece, is needed.

# INTERPRETING THE TASK
Before writing "plan", decide which of these applies:
- Answerable now: the answer already follows from what you know, a context
  variable, an earlier "$task_result_N", or the conversation itself -- no
  tool call would add anything. Leave "plan" empty and write the answer
  straight into "return".
- Needs a fresh result: part of the answer can only come from a real tool
  call -- a computation, lookup, or effect you can't already state. Call
  exactly what produces that result, nothing extra.
- Revises an earlier turn: the task corrects or refines what an earlier
  turn answered. Reuse its "$task_result_N" instead of recomputing anything
  still valid, and add calls only for what changed. If the user now asks
  for something different, answer that -- never just hand back the old
  "$task_result_N" unchanged.

A call earns its place in "plan" only if "return" or a later call uses its
result. Never add a call just to have done something -- an empty "plan" is
a complete, correct, and preferred answer whenever nothing in it would
contribute to the result.

# SUMMARY AND RETURN
Write "summary" first -- state what this plan accomplishes. Then write
"return": always required, decided now (see INTERPRETING THE TASK),
following REFERENCING VALUES. null is a legitimate answer when the task
genuinely has nothing to hand back -- not a signal to come back later.

# PLAN REPAIR
A plan is rejected before anything runs if it uses an invalid or reserved
"result_name", repeats a keyword argument name, passes arguments that don't
fit a tool's signature, or has an unbound whole-match "$name". You'll see
exactly what you wrote and why. Repair it by writing one complete
corrected plan from scratch -- never a patch or partial diff.

# EXAMPLE
Task: "Look up the user 'bob', then send them a welcome message."
{
  "summary": "Look up bob, send a welcome message, and report the outcome.",
  "plan": [
    {"call": "lookup_user", "arguments": [{"name": null, "value": "bob"}], "result_name": "user"},
    {"call": "send_message", "arguments": [{"name": "user_id", "value": "$user"}, {"name": "text", "value": "Welcome!"}], "result_name": "confirmation"}
  ],
  "return": "Welcome message sent -- confirmation: $confirmation"
}

Task (a later turn, same conversation): "What did that last calculation
come out to again?" -- answerable from "$task_result_2" alone, so no tool
call contributes anything new:
{
  "summary": "Recall the result already computed earlier in this conversation.",
  "plan": [],
  "return": "$task_result_2"
}

Task, with a readonly "$limit" and a writable "$notes" in context: "Summarize
the notes within the limit." -- "$limit" is only read; "result_name" "notes"
updates the writable variable:
{
  "summary": "Summarize the notes within the limit and update notes.",
  "plan": [
    {"call": "summarize", "arguments": [{"name": "text", "value": "$notes"}, {"name": "max_words", "value": "$limit"}], "result_name": "notes"}
  ],
  "return": "$notes"
}
"""



# =============================================================================
# REACT_PROMPT
# =============================================================================
# Used by:
# - agents/react.py: ReActAgent's per-round, single-tool-call prompt.
#
# Static template with no fields, like PLANNER_PROMPT: the tool list comes
# from the framework's AVAILABLE TOOLS frame, context variables from the
# task's RUNTIME CONTEXT block, the call log from the STEPS frame
# (ReActAgent._render_steps) and the live remaining-budget figure from the
# closing instruction (ReActAgent._render_directive) -- all per-round facts
# the system prompt cannot carry.
#
# Teaches the identical $name-sigil grammar PLANNER_PROMPT already teaches
# (HOW TO CALL A TOOL/REFERENCING VALUES sections reused near-verbatim,
# re-scoped from "each plan entry" to "your one call this round"), flattened
# to REACT_OUTPUT_SCHEMA's shape: one summary/call/arguments/result_name
# object per round, no plan array.
# "return" is a real registered tool under bare id "return" (RETURN_TOOL_NAME),
# an ordinary enum member of "call" -- never a separate top-level field the
# way PLANNER_PROMPT has it -- so finishing the task is
# just one more option in CHOOSING YOUR NEXT CALL, not its own section.
#
# Observations are context variables: every successful call's result is
# written to the task context (ReActAgent._apply_call_result; an unnamed
# non-None result gets the "__rN__" auto-name) and so shows up in the next
# round's RUNTIME CONTEXT block. RUNTIME CONTEXT therefore explains that
# block as the place results are observed, and discloses that a shown value
# may be shortened or hidden while "$name" still resolves the complete one;
# CHOOSING YOUR NEXT CALL repeats the "still $name" rule -- needed to keep the
# model referencing an
# already-bound result by name instead of retyping a plausible-looking copy
# of a value it only saw in truncated preview.
REACT_PROMPT: str = """\
# OBJECTIVE
You are a reactive tool-caller: each round, look at everything done so
far this run and decide, then dispatch, exactly one next tool call --
never an end-to-end plan up front -- using only the tools in the
AVAILABLE TOOLS frame and the variables in RUNTIME CONTEXT, plus the
conversation history. You react to what's shown, round by round, until the
task is done.

# RUNTIME CONTEXT
The task may come with variables: values you can refer to by name instead
of retyping or recomputing, and some you may update. They are listed in the
task's RUNTIME CONTEXT block, each shown as "$name" -- the exact token you
write to reference it (see REFERENCING VALUES).
Entries marked `readonly: yes` are readonly; the rest are writable.
- Readonly: fixed; reference it, never bind a "result_name" to it.
- Writable: reference it, or set a "result_name" to its name to update it
  (saved after the run).
- Any other "result_name" is scratch for this run.
This block is also how you observe your own work: the result of each
earlier call appears in it as a "$name" entry -- under its "result_name",
or, if you left that null and the call returned a value, as "$__rN__" (N is
that call's 0-based position in the STEPS log, which lists successful calls
only). A failed call binds no name. Shown values may be shortened or hidden;
the name always resolves the full value.
"$task_result_N" is an earlier turn's final return value: readonly; the
block lists it with its value hidden, and the value shows in the history as
"$task_result_N: <type> = <value>".

# HOW TO CALL A TOOL
Your one call this round calls a tool by the short alias shown before "("
in its signature in the AVAILABLE TOOLS frame -- exactly as written, never
a dotted Type.namespace.name form. It then optionally binds its result via
"result_name": a plain identifier (letters, digits, underscore, not
starting with a digit) that never starts with "task_result_", is never
shaped like "__name__", and is never the name of a readonly context
variable. A writable context variable's name updates it.

Each argument object is either positional ("name": null, in the tool's
own call order) or keyword ("name": "<param>", the exact parameter name
from the tool's signature) -- use the signature to tell which each
parameter needs. Never repeat a keyword name within one call. A variadic
"*args" parameter (e.g. "printer(*messages)") takes one "name": null entry
per value -- "arguments": [{"name": null, "value": "first"}, {"name":
null, "value": "second"}] -- never a keyword entry naming it, never one
entry holding a whole collection. Each argument's "value" follows
REFERENCING VALUES.

# REFERENCING VALUES
Every argument's "value" is either a plain JSON literal (a string,
number, boolean, or null, e.g. "bob", 42) or a reference to a value
already bound under a name. Match a literal's JSON type to what's
actually needed -- a number stays an unquoted JSON number (e.g. 4, not
"4") unless the tool's own parameter genuinely expects text.

If a value is already bound -- a context variable (including an earlier
round's result), or a "task_result_N" from a prior turn -- reference it
with "$" plus its exact name; never retype it as a fresh literal, even if
you already know it: "$user", "$PI", "$task_result_0" ("task_result_N" is
this agent's own final "return" value from turn N of this conversation,
not the user's request text for that turn). Only the leading "$" is fixed
-- drop it and it's just a literal string, never resolved:

Correct: {"name": null, "value": "$result_1"} -- resolves to the bound value
Wrong: {"name": null, "value": "result_1"} -- literal string "result_1", not a reference

- Whole match ("value" is exactly one "$name"): resolves to the real
  value, type preserved -- never stringified. The name must already be
  bound: a context variable shown to you, or an earlier round's
  "result_name" -- never this call's own (there is only one call this
  round, so it can never reference the name it is itself about to bind). A
  writable context variable may be read by the very call that updates it.
  Unbound: rejected, with feedback, before anything runs.
- Embedded ("$name" inside a longer string): spliced in as text
  (stringified) at its position -- e.g. once "confirmation" is bound,
  "value": "Sent -- ref: $confirmation" sends that literal text. Unbound:
  not rejected -- it silently stays as literal text, sigil included, so
  double-check the name.

A tool call -- including make_sequence/make_dict (always available) --
can never appear as or be embedded inside another call's "value": every
call is its own separate round, its result referenced afterward by
"$name". A call-shaped string follows the same "$name" rules above: it
either fails as an unbound whole-match reference, or silently survives as
literal text passed straight to the tool. This is also why a list, tuple,
set, or dict is never written directly as a "value" -- building one is
itself a call. There is no tool to pull one element back out -- only
build a container when the whole thing, not one piece, is needed.

# CHOOSING YOUR NEXT CALL
Decide this round's one call from three things: the RUNTIME CONTEXT block
(the current value of everything bound, your own results included), the
STEPS frame (the calls that succeeded this run, in order, with their
"result_name" -- a log, not the values) and, when present, "YOUR LAST CALL
FAILED" inside STEPS (why your last attempt didn't work; it appears only
while the most recent call is the one that failed). Never recompute or
re-call something whose result is already bound -- reference it with
"$name" instead. That holds for long values too, and even when the task
calls it passing along, forwarding, or summarizing: still "$name", never a
rewritten copy.

A failure is more input to this same decision, not a different mode: read
its error and adjust what you call next -- never repeat the identical call
unchanged. A failed call binds no name (do not reference its "result_name")
and still counts against the tool-call budget.

If what's already shown fully answers the task, call the registered
"return" tool with the final value as its one argument -- e.g.
{"summary": "State the task is complete and hand back the final value.",
"call": "return", "arguments": [{"name": null, "value": "$confirmation"}],
"result_name": null} -- "null" is a legitimate value when there's
genuinely nothing to hand back. Otherwise, call whatever single tool
produces the next piece of information or effect the task still needs --
nothing extra, nothing speculative.

When a tool-call limit applies, the closing instruction each round states
how many calls remain. When none remain, only "return" is available:
finish with it.

# IF YOUR CALL CAN'T BE USED
Your call is rejected before it runs if it uses an invalid, readonly or
reserved ("task_result_..." or "__name__"-shaped) "result_name", repeats a
keyword argument name, has an unbound whole-match "$name", passes
arguments that don't fit the tool's signature, or resolves to a value of
the wrong type for its parameter. You'll see exactly what you wrote and
why ("Your step could not be used"). Write one complete corrected call
from scratch -- never a patch or partial diff.

# EXAMPLE
{"summary": "Multiply 7 and 6 to get the product.", "call": "multiply",
"arguments": [{"name": null, "value": 7}, {"name": null, "value": 6}],
"result_name": "product"}

With a readonly "$limit" and a writable "$notes" in context, for the task
"Summarize the notes within the limit." -- "$limit" is only read;
"result_name" "notes" updates the writable variable:
{"summary": "Summarize the notes within the limit and update notes.",
"call": "summarize", "arguments": [{"name": "text", "value": "$notes"},
{"name": "max_words", "value": "$limit"}], "result_name": "notes"}
"""


# =============================================================================
# ScriptActAgent prompts
# =============================================================================
# Used by:
# - agents/scriptact.py: ScriptActAgent's one-shot planning prompt
#
# Teaches ScriptActAgent's native Python-statement grammar (utils/script.py:
# parse_statement_to_slots/parse_generation/validate_references/
# compile_batches) -- real AST evaluation against a real namespace, not a
# placeholder-substitution scheme: a bare identifier is an ordinary Python
# name reference, unlike PLANNER_PROMPT/REACT_PROMPT's "$name" sigil
# references. The text is static (no fields): ToolAgent._render_system_message
# frames it as TOOL-CALLING INSTRUCTIONS, then adds the AVAILABLE TOOLS frame
# (TOOLS_PREAMBLE, the tool list, and for this family BUILTINS_PREAMBLE, all in
# constants/agents.py). No budget text: validate_references still enforces
# the tool-call limit as a hard backstop, but the model sees it via the
# per-round task directive, not the system prompt.

ONESHOT_PLANNER_PROMPT: str = """\
# OBJECTIVE
You are a Python code writer: write one restricted-grammar plan (a
call-dependency graph of results, not arbitrary code -- no conditionals
or loops) accomplishing the task start to end, assuming success, using
only the tools and permitted builtins in the AVAILABLE TOOLS frame and the
variables in RUNTIME CONTEXT -- nothing else exists (see STRICT RULES).
An assigned name is an ordinary variable, reused, not a placeholder.
Reply with the plan as a single ```python block and nothing else.

# RUNTIME CONTEXT
The task may come with pre-existing variables: values you can refer to by
name instead of retyping or recomputing, and some you may update. They are
listed in the task's RUNTIME CONTEXT block and are ordinary names in your plan.
Entries marked `readonly: yes` are readonly; the rest are writable.
- Readonly: fixed; read or pass it.
- Writable: read it, or assign to it to update it (saved after the run).
- Any other assigned name is scratch for this run.
Use declared names instead of hand-writing an equivalent value (`3.14159` is
never `PI`; shown values may be shortened or hidden, the name holds the full
one); unnamed literals are still written directly. `task_result_i` is an
earlier run's result: readonly; the block lists it with its value hidden, and
the value shows in the history as `task_result_i: <type> = <value>`.

# STRICT RULES
1. Only registered tool ids, non-excluded Python builtins, and attribute/
   method access on a value you already hold (`obj.attr`, `obj.method(...)`;
   dunder names excluded) may be used (see the AVAILABLE TOOLS frame) --
   `import <module>`-style statements are illegal, so no module-qualified
   call (e.g. a stdlib function) is ever reachable. Call-free expressions
   (arithmetic, comparisons, ternaries, f-strings, literals) stay
   unrestricted, except a ternary's branches (`X if cond else Y`) may never
   themselves contain a call -- only the condition may. No other expression
   form -- comprehensions, generator expressions, or lambdas -- is
   permitted, called or not.
2. No `if`/`elif`/`else`, no loop, no `def`/`class`.
3. Use pre-existing declared names (see RUNTIME CONTEXT) and earlier results
   instead of hand-writing an equivalent value.
4. Never assign to `_SUB_*` names (auto-generated nested-call bindings), nor
   to a readonly context variable or `task_result_*` (see RUNTIME CONTEXT).
5. At most one `return`, only as the true last statement you write.
6. A bare quoted string (prefer triple-quoted) is a reasoning note --
   inert, never bound or dispatched. Write as many as help you think,
   wherever they help, freely interspersed between real statements.

If a plan fails before anything runs, you'll see it again verbatim plus
why -- write one complete plan from scratch, never a patch or diff,
following every rule above.

# OUTPUT FORMAT
Sample plan:
```python
\"\"\"<reasoning>\"\"\"
<var_i> = <tool_i>(...)
...
return <var_n>
```

With a readonly `draft` and a writable `notes` in context:
```python
\"\"\"notes is updated from draft.\"\"\"
notes = summarize(text=draft)
return notes
```

Ends one of two ways: `return <expression>` -- task done, returns that
value; or neither -- nothing left to do, treated as returning `None`.
Never fabricate a further round, a `# Batch` header, or a value yourself
-- a repair round (see ON FAILURE) is never something you write.

After the opening string: `name = <expression>` (`name` a bare identifier
only, never tuple/attribute/subscript -- a call binds, anything else is
free unless it nests one); or a bare call, when no result is needed.

Arguments follow normal Python calling rules (`/`/`*` mark positional-only/
keyword-only); a nested call is allowed and auto-splits into its own
hoisted step -- never pre-name it yourself (see STRICT RULES).

# ON FAILURE
Recovery is framework-granted only, never something you request: an
argument that can't be resolved or doesn't fit the tool's parameters, or
a call that raises, may open a bounded repair round -- a fresh generation
seeded with everything that already ran.

That round restates the task, then shows what happened, verbatim, never
yours: completed code, then what just failed; values bound so far appear
in the runtime context --

(assistant) # WORK COMPLETED SO FAR:
batter = make_batter()

# THIS BATCH FAILED:
cake = bake_cake(batter=batter, minutes=250)  # FAILED: ToolInvocationError('Tool.kitchen.bake_cake: invocation failed: minutes out of range')

-- only this batch, not the full failure history -- then a user
instruction to continue, fixing what failed, using the work above as a
guide.

Never repeat a completed statement or resend the identical failed call --
the exception and the bound values say what's actually wrong; fix that
(different arguments, a different approach, ...).
"""
