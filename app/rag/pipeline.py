from __future__ import annotations

from dataclasses import dataclass
from typing import List

from app.rag.retriever import Retriever
from app.rag.prompt_builder import build_rag_prompt
from app.rag.vector_store import SearchResult

@dataclass(frozen=True)
class RagResult:
    answer: str
    results: List[SearchResult]

#PIPELINE = ORCHESTRATION LAYER. THIS IS BACKEND SYSTEM DESIGN>

class RagPipeline:
    """
    Orchestrates Retrieval + Prompt + LLM.
    """
    def __init__(self, retriever: Retriever, llm_provider) -> None:
        self.retriever = retriever
        self.llm_provider=llm_provider

    def run(self, question: str, top_k: int = 3) -> RagResult:
        results = self.retriever.retrieve(question, top_k=top_k)
        prompt = build_rag_prompt(question, results)
        answer = self.llm_provider.generate(prompt)

        return RagResult(answer=answer, results=results)