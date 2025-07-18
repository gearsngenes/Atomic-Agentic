import os
from dotenv import load_dotenv
from abc import ABC, abstractmethod
from openai import OpenAI

# CONSTANTS
load_dotenv()
DEFAULT_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class LLMNucleus(ABC):
    @abstractmethod
    def invoke(self, messages: list[dict]) -> str:
        pass

class OpenAINucleus(LLMNucleus):
    def __init__(self, model: str, api_key: str|None = None, temperature: float = 0.1):
        self.llm = OpenAI(api_key=api_key if api_key else DEFAULT_OPENAI_API_KEY)
        self.model = model
        self.temperature = temperature

    def invoke(self, messages: list[dict[str,str]]) -> str:
        response = self.llm.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
        )
        return response.choices[0].message.content.strip()

class AzureOpenAINucleus(LLMNucleus):
    def __init__(self, api_key: str, endpoint: str, api_version: str, model: str):
        # Placeholder for Azure OpenAI client setup
        self.api_key = api_key
        self.endpoint = endpoint
        self.api_version = api_version
        self.model = model
        # self.llm = AzureOpenAI(...)

    def invoke(self, messages: list[dict]) -> str:
        # Implement Azure OpenAI invocation logic here
        raise NotImplementedError("AzureOpenAINucleus.invoke not implemented")

class BedrockNucleus(LLMNucleus):
    def __init__(self, access_key: str, secret_key: str, region: str, model: str):
        # Placeholder for Bedrock client setup
        self.access_key = access_key
        self.secret_key = secret_key
        self.region = region
        self.model = model
        # self.llm = Bedrock(...)

    def invoke(self, messages: list[dict]) -> str:
        # Implement Bedrock invocation logic here
        raise NotImplementedError("BedrockNucleus.invoke not implemented")
