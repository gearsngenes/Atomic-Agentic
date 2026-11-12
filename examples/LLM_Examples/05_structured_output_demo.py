import os
from dotenv import load_dotenv
from atomic_agentic.llm import (
    OpenAIEngine,
    GeminiEngine,
    MistralEngine,
    AnthropicEngine,
    LlamaCppEngine,
)
import logging
from pprint import pprint

load_dotenv()
logging.basicConfig(level=logging.INFO)

# --- Shared JSON-Schema: a lightweight tabletop RPG character sheet ---
# Deliberately uses only keywords enforced identically across every
# provider (type/properties/required/additionalProperties/items/description)
# -- no minItems/maxItems/not/properties-free nested objects -- so the same
# schema works unmodified across all five routes below. Every nested object
# (abilities, each spell) carries its own "required" + "additionalProperties":
# False, since OpenAI's strict mode enforces that recursively.
CHARACTER_SCHEMA = {
    "type": "object",
    "description": "A lightweight tabletop RPG character sheet, built from a short premise.",
    "properties": {
        "name": {"type": "string", "description": "The character's full name."},
        "role": {
            "type": "string",
            "description": "Their class or role (e.g. 'wandering healer', 'rogue archivist').",
        },
        "physical_description": {
            "type": "string",
            "description": "A short physical description.",
        },
        "motivation": {
            "type": "string",
            "description": "What drives the character; their core motivation.",
        },
        "abilities": {
            "type": "object",
            "description": "Core ability scores, 1-20.",
            "properties": {
                "strength": {"type": "integer", "description": "Physical power, 1-20."},
                "dexterity": {"type": "integer", "description": "Agility and reflexes, 1-20."},
                "intelligence": {"type": "integer", "description": "Reasoning and knowledge, 1-20."},
                "charisma": {"type": "integer", "description": "Force of personality, 1-20."},
            },
            "required": ["strength", "dexterity", "intelligence", "charisma"],
            "additionalProperties": False,
        },
        "spells": {
            "type": "array",
            "description": "A few signature spells or special abilities the character knows.",
            "items": {
                "type": "object",
                "description": "One spell or special ability.",
                "properties": {
                    "name": {"type": "string", "description": "The spell's name."},
                    "description": {
                        "type": "string",
                        "description": "A short description of its effect.",
                    },
                },
                "required": ["name", "description"],
                "additionalProperties": False,
            },
        },
    },
    "required": [
        "name",
        "role",
        "physical_description",
        "motivation",
        "abilities",
        "spells",
    ],
    "additionalProperties": False,
}

# --- The task: build a character from a short premise ---
CHARACTER_PREMISE = (
    "A former royal guard who abandoned her post after a magical plague "
    "swept the capital, and now wanders the wilds seeking a cure."
)

messages = [
    {
        "role": "system",
        "content": (
            "You are a tabletop RPG character designer. Given a short "
            "premise, invent a complete, internally consistent character "
            "that fits it."
        ),
    },
    {
        "role": "user",
        "content": f"Build a character from this premise: {CHARACTER_PREMISE}",
    },
]

# --- Initialize one LLM engine (uncomment the one you want to use) ---
llm = OpenAIEngine(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4.1")
# llm = GeminiEngine(api_key=os.getenv("GOOGLE_API_KEY"), model="gemini-2.5-flash")
# llm = MistralEngine(api_key=os.getenv("MISTRAL_API_KEY"), model="mistral-small-latest")
# llm = AnthropicEngine(api_key=os.getenv("ANTHROPIC_API_KEY"), model="claude-sonnet-4-6")
# llm = LlamaCppEngine(model_path=os.getenv("LLAMA_MODEL_PATH"), repo_id="unsloth/phi-4-GGUF", filename="phi-4-Q4_K_M.gguf", n_ctx=512, verbose=False, n_threads=4)

# --- Invoke the engine with output_structure for schema-constrained output ---
result = llm.invoke({"messages": messages, "output_structure": CHARACTER_SCHEMA})
print("STRUCTURED CHARACTER SHEET:")
pprint(result.result)
print("\nLLM RESULT OBJECT:")
pprint(result)
