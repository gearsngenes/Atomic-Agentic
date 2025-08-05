# atomic_agentic_client.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from modules.Agents import A2AProxyAgent

def main():
    # Point at the root (no '/a2a' suffix—the client adds it)
    a2a_client = A2AProxyAgent("http://localhost:5000")
    # Ask the orchestrator to do something
    request = input("Give a request to your Atomic-Agentic Shakespearean agent. Type 'q' to quit.\nRequest: ")
    while request.lower().strip() != 'q':
        reply = a2a_client.invoke(request)
        print("Shakespearean Agent:", reply)
        request = input("Request: ")

if __name__ == "__main__":
    main()
