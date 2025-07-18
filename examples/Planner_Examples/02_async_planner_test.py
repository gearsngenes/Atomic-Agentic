import sys, os, logging, time
from pathlib import Path
# Setting the root
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# ----------------- Atomic Agents ----------------
from modules.Agents import Agent, PlannerAgent
from modules.LLMNuclei import *

# ----------------- Setup Logging ----------------
logging.basicConfig(level=logging.INFO)

# define a global nucleus to give to each of our agents
nucleus = OpenAINucleus(model = "gpt-4o-mini")

# Define a haiku-writing agent
haiku_writer = Agent(
    name        = "HaikuWriter",
    nucleus     = nucleus,
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
        "hyphenate every multi-syllable word used."
    )
)

# Create a PlannerAgent that uses the haiku writer and print tool
async_batch_writer = PlannerAgent(
    name =      "AsyncBatchHaikuPlanner",
    nucleus =   nucleus,
    is_async =  True,   # Toggle between async and sync planning
)

# Register the haiku writer agent
async_batch_writer.register_agent(
    agent =         haiku_writer,
    description =   "Writes a haiku about a given topic."
)

# Define and register a simple formatted print function for haikus
def print_haiku(haiku_topic, haiku):
    print(f"---\n**{haiku_topic}**\n{haiku}\n---")

async_batch_writer.register_tool(
    name =          "print_haiku",
    func =          print_haiku,
    description =   "Prints a haiku formatted with topic as the title."
)

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
        "Use the haiku print method to print the formatted haikus after they are written."
        "Topics:\n"
        + "\n- Topic ".join(f'{i}: {t}' for i, t in enumerate(topics)) + "."
    )

    # Invoke the planner agent and time it
    start = time.time()
    result = async_batch_writer.invoke(prompt)
    end = time.time()
    print("Time taken:", end - start, "seconds")