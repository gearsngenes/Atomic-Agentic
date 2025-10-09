import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from modules.A2Agents import A2AProxyAgent

def main():
    # Address and ports of agents to test
    agents = {
        "trivia": {"port": 6000, "task": "Hello, World!"},
        "shakespeare": {"port": 5000, "task": "Who are you, and what are capable of?"},
        "planner": {"port": 7000, "task": """
                    Perform the following tasks:
                    1)  Give me a fun fact about sea cucumbers".
                    2)  Write a sonnet about bluebirds
                    3)  Return both results in the format:
                        Trivia Result: 
                        <trivia result here>

                        Sonnet Result: 
                        <sonnet result here>"""}
    }
    
    choice = "planner"  # change to "shakespeare" to test the shakespeare agent
    
    # Get the port and task for the chosen agent
    port, task = agents[choice]["port"], agents[choice]["task"]
    # Create the A2A client
    a2a_client = A2AProxyAgent(f"http://localhost:{port}")
    
    # Give the task to the agent and print the response
    response = a2a_client.invoke(task)
    print(f"{choice}'s response:\n", response)
if __name__ == "__main__":
    main()
