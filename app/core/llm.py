from __future__ import annotations

"""
LLM adapter layer (Model Abstraction Layer)

- Today: Stub provider for development
- Later: OpenAI / Bedrock / local model providers
"""

from typing import Protocol
from app.core.config import settings
from app.core.cache import SimpleCache

_llm_cache = SimpleCache()


class LLMProvider(Protocol):
    def generate(self, prompt: str) -> str:
        ...


class StubProvider:
    """
    Development provider: returns deterministic output.
    Now includes LLM response caching.
    """

    def generate(self, prompt: str) -> str:
        # 1) Check cache
        cached = _llm_cache.get("llm", settings.llm_provider, settings.llm_model, prompt)
        if cached is not None:
            return cached

        # 2) Generate (stub)
        result = f"[stub:{settings.llm_model}] {prompt}"

        # 3) Save cache
        _llm_cache.set(result, "llm", settings.llm_provider, settings.llm_model, prompt)

        return result


def get_provider() -> LLMProvider:
    if settings.llm_provider == "stub":
        return StubProvider()
    raise ValueError("Unsupported provider")