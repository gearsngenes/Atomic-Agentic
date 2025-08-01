# atomic_agentic_client.py

from python_a2a import A2AClient

def main():
    # Point at the root (no '/a2a' suffix—the client adds it)
    client = A2AClient("http://localhost:5000")
    # Ask the orchestrator to do something
    request = input("Give a request to your Atomic-Agentic Shakespearean agent. Type 'q' to quit.\nRequest: ")
    while request.lower().strip() != 'q':
        reply = client.ask(request)
        print("Shakespearean Agent:", reply)
        request = input("Request: ")

if __name__ == "__main__":
    main()
