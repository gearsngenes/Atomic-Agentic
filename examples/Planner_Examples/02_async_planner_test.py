import sys, os, logging, time, json
from pathlib import Path
# Setting the root
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# ----------------- Atomic Agents ----------------
from modules.Agents import *
from modules.PlannerAgents import AgenticPlannerAgent

# ----------------- Setup Logging ----------------
logging.basicConfig(level=logging.INFO)

# define a global llm engine to give to each of our agents
llm_engine = OpenAIEngine(model = "gpt-4o-mini")

# Define a haiku-writing agent
haiku_writer = PrePostAgent(
    name        = "HaikuWriter",
    llm_engine     = llm_engine,
    role_prompt = (
        "You are a master of writing haiku. Given a topic, write a "
        "3-line haiku about it, following a 5-syllable, 7-syllable, "
        "5-syllable rhythmic patter. Do not simply re-state the topic "
        "in the haiku, be creative! Example, given the topic "
        "'The morning sun', a valid haiku could be:\n\n"
        "A gol-den orb shines\n"
        "With red streaks a-cross the sky,\n"
        "Crisp, bright, mor-ning light\n\n"
        "Be sure to follow the 5-7-5 syllable structure, and "
        "hyphenate every multi-syllable word used.\n"
        "Once you have finished writing the haiku, place the haiku "
        "and the topic into a valid JSON object in the format below, with "
        "NO additional tags, quotation marks, or comments:\n"
        """{"title":"<haiku topic here>", "haiku":"<haiku poem here>"}"""
    )
)

# Define and register a formatted print function for haikus + the json.loads method
def print_haiku(haiku_json):
    print(f"---\n**{haiku_json['title']}**\n{haiku_json['haiku']}\n---")
haiku_writer.add_poststep(json.loads)
haiku_writer.add_poststep(print_haiku)

# Create a PlannerAgent that uses the haiku writer and print tool
async_batch_writer = AgenticPlannerAgent(
    name        = "AsyncBatchHaikuPlanner",
    llm_engine  = llm_engine,
    granular    = False,     # Toggle to enable adding individual tools & plugins
    is_async    = True,   # Toggle between async and sync planning
)

# Register the haiku writer agent

async_batch_writer.register(tool=haiku_writer)

if __name__ == "__main__":
    # List of haiku topics to write about
    topics = [
        "The morning frost",
        "A gentle breeze",
        "Falling leaves",
        "Winter snowfall",
        "A quiet pond",
        "A blooming flower",
        "The starry night",
        "A roaring fire",
        "An ancient mountain",
        "A running river",
    ]
    
    # Compose the prompt for the planner agent: batch-writing haikus
    prompt = (
        "Given the following list of topics, separately send each topic to the HaikuWriter. "
        "The HaikuWriter should be given only ONE HAIKU TOPIC at a time. "
        "Topics:\n"
        + "\n- Topic ".join(f'{i}: {t}' for i, t in enumerate(topics)) + "."
    )

    # Invoke the planner agent and time it
    start = time.time()
    result = async_batch_writer.invoke(prompt)
    end = time.time()
    print("Time taken:", end - start, "seconds")