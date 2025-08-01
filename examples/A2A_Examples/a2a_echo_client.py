# atomic_agentic_client.py

from python_a2a import A2AClient

def main():
    # Point at the root (no '/a2a' suffix—the client adds it)
    client = A2AClient("http://localhost:5000")
    # Ask the echo agent
    response = client.ask("Hello, World!")
    print("Echo agent response: ", response)
if __name__ == "__main__":
    main()
