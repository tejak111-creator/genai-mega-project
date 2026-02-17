#LLM/Application Service Layer(MODEL ABSTRACTION LAYER)
"""
LLM adapter.
Later: OpenAI/local model implementations
"""
#this layer turns prompt to generated text
from typing import Protocol
from app.core.config import settings 

class LLMProvider(Protocol):
    def generate(self,prompt:str) -> str:
        ...
class StubProvider:
    def generate(self, prompt: str) -> str:
        return f"[stub:{settings.llm_model}] {prompt}"
def get_provider() -> LLMProvider:
    if settings.llm_provider == "stub":
        return StubProvider()
    raise ValueError("Unsupported provider")

"""
def generate_text(prompt: str) -> str:
    #Stub
    return f"[stub:{settings.llm_provider}:{settings.llm_model}] You said: {prompt}"
"""