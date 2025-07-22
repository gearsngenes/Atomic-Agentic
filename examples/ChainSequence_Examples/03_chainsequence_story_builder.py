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
You are a story writer. Given a bullet-point outline, write a full draft of the story. Respond ONLY with the story text, no extra commentary.
"""
REVIEWER_PROMPT = """
You are a story reviewer. Given a story draft, respond ONLY with a list of constructive feedback points for improvement. No extra text.
"""
REWRITER_PROMPT = """
You are a story rewriter. Given a story draft and a list of feedback points, revise the story to address the feedback. Respond ONLY with the revised story text.
"""

# Preprocessors for each agent

def save_markdown(text, filename):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)
    return f"Saved to {filename}"

# Build chain sequence
if __name__ == "__main__":
    story_prompt = "A young fox must outwit a pack of wolves to save its family."
    output_filename = f"{OUTPUT_DIR}/chainsequence_story.md"

    # Outliner agent
    outliner = ChainSequenceAgent(seed=Agent(name="Outliner", llm_engine=llm_engine, role_prompt=OUTLINER_PROMPT))

    # Writer agent
    writer = ChainSequenceAgent(seed=Agent(name="Writer", llm_engine=llm_engine, role_prompt=WRITER_PROMPT))
    outliner.talks_to(writer)

    # Single review-rewrite pair
    reviewer = ChainSequenceAgent(
        seed=Agent(name="Reviewer-1", llm_engine=llm_engine, role_prompt=REVIEWER_PROMPT))
    rewriter = ChainSequenceAgent(
        seed=Agent(name="Rewriter-1", llm_engine=llm_engine, role_prompt=REWRITER_PROMPT))
    
    writer.talks_to(reviewer)
    reviewer.talks_to(rewriter)

    # Final output agent: saves markdown
    final_draft = outliner.invoke(story_prompt)
    save_markdown(final_draft, output_filename)
    print("Final draft saved at:", output_filename)
