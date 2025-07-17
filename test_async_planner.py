from atomic_agents.Agents import Agent, PlannerAgent
from atomic_agents.Plugins import ConsolePlugin, MathPlugin
import logging

logging.basicConfig(level=logging.INFO)
# Haiku writer agent
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

poem_writer = Agent(
    name="PoemWriter",
    role_prompt=(
        "You are a master of writing poems. Given a topic, write a "
        "short poem about it, following a rhythmic pattern. Do not simply re-state the topic "
        "in the poem, be creative!"
    )
)

if __name__ == "__main__":
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
    # Toggle 'is_async' to test both sync and async planners
    agent = PlannerAgent("AsyncHaikuPlanner", is_async=True, debug=False)
    agent.register_plugin(ConsolePlugin())
    agent.register_agent(haiku_writer, description="Writes a haiku about a given topic.")

    # Compose the prompt for the planner agent
    prompt = (
        "Given the following list of topics, separately send each topic to the HaikuWriter. "
        "The HaikuWriter should be given only ONE HAIKU TOPIC at a time. "
        "Print the resulting haikus after they are written, formatted exactly as such:\n"
        "---\n**<Topic here>**\n"
        "<haiku here>\n---"
        "Topics:\n"
        + "\n- Topic ".join(f'{i}: {t}' for i, t in enumerate(topics)) + "."
    )

    import time
    start = time.time()
    result = agent.invoke(prompt)
    end = time.time()
    print("Time taken:", end - start, "seconds")