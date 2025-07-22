from llama_cpp import Llama
import os
from dotenv import load_dotenv
from abc import ABC, abstractmethod
from openai import OpenAI

# CONSTANTS
load_dotenv()
DEFAULT_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class LLMEngine(ABC):
    @abstractmethod
    def invoke(self, messages: list[dict]) -> str:
        pass

class OpenAIEngine(LLMEngine):
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

class LlamaCppEngine(LLMEngine):
    def __init__(self,
                    model_path: str = None,
                    repo_id: str = None, 
                    filename: str = None, 
                    n_ctx: int = 2048, 
                    verbose = False):
        self.llm = None
        # Prefer file path if available
        if model_path:
            self.llm = Llama(model_path=model_path, n_ctx=n_ctx, verbose=verbose)
        elif repo_id and filename:
            self.llm = Llama.from_pretrained(repo_id=repo_id, filename=filename, n_ctx=n_ctx, verbose=verbose)
        else:
            raise ValueError("Must provide either model_path or both repo_id and filename.")

    def invoke(self, messages: list[dict]) -> str:
        if not self.llm:
            raise RuntimeError("Llama model not loaded.")
        # llama-cpp-python expects messages in OpenAI format
        response = self.llm.create_chat_completion(messages=messages)
        return response["choices"][0]["message"]["content"].strip()

class AzureOpenAIEngine(LLMEngine):
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

class BedrockEngine(LLMEngine):
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
