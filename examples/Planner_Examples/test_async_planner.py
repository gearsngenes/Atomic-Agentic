import sys
from pathlib import Path
# Setting the root
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import logging
import time
# ----------------- Atomic Agents ----------------
from modules.Agents import Agent, PlannerAgent

# ----------------- Setup Logging ----------------
logging.basicConfig(level=logging.INFO)

# Define a haiku-writing agent
haiku_writer = Agent(
    name="HaikuWriter",
    role_prompt=(
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

# Define a simple formatted print function for haikus
def print_haiku(haiku_topic, haiku):
    print(f"---\n**{haiku_topic}**\n{haiku}\n---")

# Create a PlannerAgent that uses the haiku writer and print tool
agent = PlannerAgent(
    "AsyncHaikuPlanner",
    is_async=True,  # Toggle to False to use synchronous planning. Test
                    # how long it takes to complete with async vs sync
)

# Register the haiku writer agent
agent.register_agent(
    haiku_writer,
    description = "Writes a haiku about a given topic."
)

# Register the print_haiku tool
agent.register_tool(
    "print_haiku",
    print_haiku,
    "Prints a haiku formatted with topic as the title."
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
    result = agent.invoke(prompt)
    end = time.time()
    print("Time taken:", end - start, "seconds")