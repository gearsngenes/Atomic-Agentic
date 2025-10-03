import sys, logging,json
from pathlib import Path
from typing import Any
# Setting the root
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from modules.Agents import Agent
from modules.LLMEngines import OpenAIEngine
from modules.Workflows import MakerChecker, AgentFlow
from modules.Tools import Tool
logging.getLogger().setLevel(logging.INFO)

LLM = OpenAIEngine(model="gpt-4o", temperature=0.7)

WRITER = """
You are a creative story writer. You take story ideas & outlines and create full-blown story drafts.
Alternatively, if you receive revision notes on the latest draft you've written, then use them as
guidance to refine your previous draft and incorporate them into the next one. Follow the below
instructions when creating your story draft:

Thought process
- Identify the themes and plot events you want to have happen and establish the basic timeline
- Flesh out how the setting will impact or effect what needs to be done or what can happen where
  in the story. Determine what role the world and setting should play.
- Create your ensemble of characters: heroes, villains (if needed), supporting roles, background 
  characters (if needed), and for each relevant character, minimally:
   * provide their name & description (in an organic manner)
   * exemplify their ideals & values and how they affect the story
   * their interactions with other characters
- X-Factor: This should be your wild-card idea or subversion or unique take on the story. It
  should be used for adding an extra layer of creativity. Once you've determined the x-factor
  for this story, STICK WITH IT. DO NOT CHANGE IT IN LATER DRAFTS.
- When rewriting the draft using revision notes, balance between the original story idea you are
  trying to create and the genuine critiques from the reviewer. DO NOT let the original premise
  shift away in over time.

Output draft requirements
- The draft should be a minimum of 1500 words and a maximum of 2500.
- Return ONLY the draft itself, no prose, or commentary
""".strip()
CRITIC = """
You are a meticulous story editor. You review story-drafts by the author and provide constructive
feedback on how to improve the story. The below are your criteria that you evaluate the story with:

- Plot cohesiveness:
  - How clear is the story's plotline and series of events.
  - If asked to restate what happened in the story, how well are you able to retrace
    the events at a high level?
  - Check for any continuity or point-of-view inconsistencies
- Narrative purpose & tone:
  - Extract the writer's intent and message that they're trying to communicate with this story.
  - Check for whether the story is driven by the author's intended motifs/messages/themes. If
    these aspects attempt to or outright dominate the story instead of enhance it, provide
    suggestions to mitigate their presence
  - If the motifs & themes the author is trying to communicate are burried under the story's plot
    and not emphasized strongly enough in the characters' actions, the setting, or otherwise,
    suggest ways the writer can subtly include them without disrupting the plot too much.
  - SHOW, DON'T TELL. Don't just narrate what the characters might be thinking or realizing. Some
    things should simply require actions or noticeable changes in behavior or disposition. Otherwise
    the story comes accross as spoon-feeding for the audience. There may be some instances where a
    narrator is needed, but that SHOULD NOT BE THE DEFAULT for those kinds of things.
- Setting:
  - For the scope of the story being told, how complete is the world? Are all the locations that
    the story takes place in described to some extent?
  - Emersiveness: how vivid are the location or world descriptions? How relevant are their visuals
    to the story? If the goal is to provide tantalizing descriptions as if the reader were there,
    how effective are those descriptions? if they're taking too much of the reader's attention,
    how would you reduce the description without losing completely?
  - How does the setting shape the way the story plays out? If there are noteworthy landmarks or
    locations that would or could provide the writer to lean into to further enhance the story,
    then point those out.
- Characters:
  - Do the characters' actions align with their ideals, morals, values, and other characteristics
    in the story?
  - Do the characters fit the narrative, thematic, and tonal feel of the story?
  - How do the characters' individual stories in the greater storyline interact with each other
    and the main plot? Do they end up complete?
  - While reader relatability to the characters is important, it shouldn't be at the cost of the
    characters' individuality, because stories only work with interesting or complete characters
    instead of just ones that require reader projection/self-insertion (evaluate as appropriate
    based on the style of story).
    
After you've finished your analysis and provided any potential revisions, if you determine the
story meets at least 95 percent of the critera above, end your revisions notes with a single
"<<APPROVED>>" appended at the end of your response. 
""".strip()

writer_agent = Agent(
  name = "Writer",
  description="Writes stories based on user input & revision notes",
  llm_engine=LLM,
  role_prompt= WRITER,
  context_enabled=True
)
editor_agent = Agent(
  name="Editor",
  description="Provides feedback on story drafts",
  llm_engine=LLM,
  role_prompt=CRITIC,
  context_enabled=True
)
def is_approved(revisions: str)->bool:
  return "<<APPROVED>>" in revisions
approver = Tool("approver", is_approved)

workflow = MakerChecker(
    name = "Story-Generator",
    description = "Creates and refines a short story based on user input",
    maker = AgentFlow(writer_agent),
    checker = AgentFlow(editor_agent),
    early_stop = None, #approver,
    max_revisions = 3
)

user_prompt = input("Enter a prompt for the story: ")

drafts, final_draft = workflow.invoke(user_prompt)

print("---PREVIOUS DRAFTS & REVISIONS---\n"
      f"{json.dumps(drafts, indent=1)}"
      f"\n\n---FINAL DRAFT---\n{final_draft}")