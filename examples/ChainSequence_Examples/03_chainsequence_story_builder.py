import sys
from pathlib import Path
import os
# Set root to repo root
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from modules.Agents import Agent, ChainSequenceAgent
from modules.LLMEngines import *

# define a global llm engine to give to each of our agents
llm_engine = OpenAIEngine(model = "gpt-4o-mini")

OUTPUT_DIR = "examples/output_markdowns"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Role prompts for each agent type
OUTLINER_PROMPT = """
You are a story outliner. Given a story prompt, respond ONLY with a bullet-point outline of the story's major beats, no extra text.
"""
WRITER_PROMPT = """
You are a story writer. Given a bullet-point outline, write a full draft of the story. Respond ONLY with the story text, no extra commentary. If instead of an outline you get revision notes from a third party reviewer agent, then rewrite your last draft with those critiques addressed. 
"""
REVIEWER_PROMPT = """
You are a story reviewer. Given a story draft, respond ONLY with a list of constructive feedback points for improvement. No extra text.
"""
REWRITER_PROMPT = """
You are a story rewriter. Given a story draft and a list of feedback points, revise the story to address the feedback. Respond ONLY with the revised story text.
"""
# Define our agents
outliner = Agent(name="Outliner", llm_engine=llm_engine, role_prompt=OUTLINER_PROMPT)
writer = Agent(name="Writer", llm_engine=llm_engine, role_prompt=WRITER_PROMPT, context_enabled=True)
reviewer = Agent(name="Reviewer", llm_engine=llm_engine, role_prompt=REVIEWER_PROMPT, context_enabled=True)

# helper save method
def save_markdown(text, filename):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)
    return f"Saved to {filename}"

chain_story_writer = ChainSequenceAgent("Sequential Story Writer")
# Build our sequential chain: outline, write, then review and rewrite twice
chain_story_writer.add(outliner)
chain_story_writer.add(writer)
chain_story_writer.add(reviewer)
chain_story_writer.add(writer)
chain_story_writer.add(reviewer)
chain_story_writer.add(writer)

# Build chain sequence
if __name__ == "__main__":
    # 1. get story idea
    story_prompt = input("Story Idea (brief description): ")
    # 2. create story
    final_draft = chain_story_writer.invoke(story_prompt)
    
    output_filename = f"{OUTPUT_DIR}/chainsequence_story.md"
    save_markdown(final_draft, output_filename)
    print("Final draft saved at:", output_filename)
