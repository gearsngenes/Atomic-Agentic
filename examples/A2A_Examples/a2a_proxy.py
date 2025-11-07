import sys
from pathlib import Path
import json

# Ensure local modules can be imported when running from examples/
sys.path.append(str(Path(__file__).resolve().parents[2]))

from modules.A2Agents import A2AProxyAgent

def main():
    """
    Exercise: call a remote A2A server that wraps a schema-driven seed Agent.

    New contract:
      - Proxy .invoke(...) expects a MAPPING, not a string.
      - The payload is sent as a single function_call parameter named "payload".
      - .invoke(...) returns the RAW A2A Message; use helpers to parse output.
    """
    agents = {
        "shakespeare": {
            "port": 5000,
            # Base Agent's pre-invoke Tool expects {"prompt": <str>}
            "inputs": {"prompt": "Who art thou, and what canst thou accomplish?"}
        },
        # Add others as you bring them up (must expose same 'invoke/payload' surface)
        "trivia": {"port": 6000, "inputs": {"prompt": "Hello, World!"}},
        "planner": {"port": 7000, "inputs": {
            "prompt": ("Give me a fun-fact and write a sonnet, and return the two strings as a dictionary,"
            " with the keys being the names of the tools called to create each of them") }},
    }

    choice = "shakespeare"
    host = "localhost"
    port = agents[choice]["port"]
    inputs = agents[choice]["inputs"]

    # IMPORTANT: pass full scheme in URL (http://)
    proxy = A2AProxyAgent(url=f"http://{host}:{port}", name="A2AProxy")

    # A2A response
    print("Arg-map:", proxy.arguments_map)
    result = proxy.invoke(inputs)  # returns python_a2a Message
    print(f"\n[{choice}] RAW A2A response:\n{result}\n{type(result).__name__}")

if __name__ == "__main__":
    main()
