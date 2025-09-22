import sys, logging
from pathlib import Path
from typing import Any
# Setting the root
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from modules.Agents import Agent
from modules.LLMEngines import OpenAIEngine
from modules.Workflows import MakerChecker

logging.getLogger().setLevel(logging.INFO)

LLM = OpenAIEngine(model="gpt-4o")

maker = Agent(
    name = "StoryMaker",
    description = "Creates a short story based on user input",
    role_prompt = "You are a creative writer. Based on the user's prompt, craft an engaging and imaginative short story. Ensure the story has a clear beginning, middle, and end, and incorporates interesting characters and settings.",
    llm_engine = LLM,
    context_enabled=True
)
checker = Agent(
    name = "StoryChecker",
    description = "Reviews and suggests improvements for the short story created by StoryMaker",
    role_prompt = "You are a meticulous editor. Review the story provided by the StoryMaker for coherence, creativity, and engagement. Provide constructive feedback and suggest improvements. If the story is good, simply say 'The story is well-written and engaging.'",
    llm_engine = LLM,
    context_enabled=True
)
workflow = MakerChecker(
    name = "StoryMakerChecker",
    description = "Creates and refines a short story based on user input",
    maker=maker,
    checker=checker,
    max_revisions = 2
)

user_prompt = input("Enter a prompt for the story: ")
final_draft = workflow.invoke(workflow.invoke(user_prompt))

print(final_draft)