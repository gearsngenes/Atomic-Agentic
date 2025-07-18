import sys
from pathlib import Path
# Set root to repo root
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from modules.Agents import Agent

assistant = Agent("Agent Atom", "You are a helpful and enthusiastic assistant named Agent Atom.", context_enabled=True)
print(f"Chat with {assistant.name}! To exit the conversation, type 'q' or 'exit'!")
query = input("You: ")
while query.strip().lower() not in ['q', 'exit']:
    print(f"{assistant.name}: {assistant.invoke(query)}")
    query = input("You: ")