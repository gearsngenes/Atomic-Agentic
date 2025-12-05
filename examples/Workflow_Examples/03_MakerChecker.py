import logging
from atomic_agentic.Agents import Agent
from atomic_agentic.LLMEngines import OpenAIEngine
from atomic_agentic.Workflows import MakerChecker, AgentFlow, ToolFlow
from atomic_agentic.Tools import Tool
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level = logging.INFO)#logging.DEBUG)

# ---- LLM ----
LLM = OpenAIEngine(model="gpt-4o", temperature=0.7)

# ---- Prompts ----
WRITER = """
You are a creative story writer. You take story ideas & outlines and create full-blown story drafts.
Alternatively, if you receive revision notes on the latest draft you've written, then use them as
guidance to refine your previous draft and incorporate them into the next one.

Output draft requirements
- The draft should be a minimum of 1500 words and a maximum of 2500.
- Return ONLY the draft itself, no prose, or commentary
""".strip()

CRITIC = """
You are a meticulous and unforgiving story editor. You review story-drafts by the author and provide
constructive feedback on how to improve the story. Evaluate any stories you are given based on 1) plot
cohesiveness, 2) character well-roundedness, 3) creativity, and most importantly, 4) narrative satisfaction.
Does the plot make sense? Are the characters relatable or at least easy to engage with, and how well
does the story handle any loose ends or narrative arcs? You are a ruthless reviewer/critic in order to weed
out any issues. After your review, ONLY IF IT MEETS YOUR CRITERIA, do you end your notes with
"<<APPROVED>>" to indicate the draft is ready.
""".strip()

# ---- Agents ----
writer_agent = Agent(
    name="Writer",
    description="Writes stories based on user input & revision notes",
    llm_engine=LLM,
    role_prompt=WRITER,
    context_enabled=True,
)

editor_agent = Agent(
    name="Editor",
    description="Provides feedback on story drafts",
    llm_engine=LLM,
    role_prompt=CRITIC,
    context_enabled=True,
)

# ---- Judge (Tool) ----
def is_approved(prompt: str) -> bool:
    return "<<APPROVED>>" in prompt

approver_tool = Tool(func = is_approved, name = "approver")

# ---- MakerChecker pipeline ----
workflow = MakerChecker(
    name="Story-Generator",
    description="Creates and refines a short story based on user input",
    maker=writer_agent,
    checker=editor_agent,
    max_revisions=3,
    judge=approver_tool,
    output_schema=["final_draft"],
    bundle_all=False
)

# ---- Run ----
user_prompt = input("Enter a prompt for the story: ")
result = workflow.invoke({"prompt": user_prompt})  # dict-only input

# final result conforms to workflow.output_schema
final_draft = result["final_draft"]

print("\n---FINAL DRAFT---\n")
print(final_draft)
print("\n---END OF DRAFT---\n")