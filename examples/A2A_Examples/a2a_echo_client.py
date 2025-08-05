import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from modules.Agents import A2AProxyAgent

def main():
    # Point at the root (no '/a2a' suffix—the client adds it)
    a2a_client = A2AProxyAgent("http://localhost:6000")
    # Ask the echo agent
    response = a2a_client.invoke("Hello, World!")
    print("Echo agent response: ", response)
if __name__ == "__main__":
    main()
