from __future__ import annotations

from dataclasses import dataclass
from typing import List

from app.rag.loader import load_text_files
from app.rag.chunker import chunk_document
from app.rag.models import DocumentChunk
from app.rag.embeddings import get_embedding_provider
from app.rag.vector_store import FaissVectorStore, SearchResult
from app.rag.retrieval_cache import get_cached_results, set_cached_results
from app.core.llm import get_provider


@dataclass
class RagConfig:
    index_dir: str = "data/index"
    chunk_size: int = 400
    chunk_overlap: int = 80


@dataclass
class RagRunResult:
    answer: str
    results: List[SearchResult]


class RagPipeline:
    def __init__(self, config: RagConfig):
        self.config = config
        self.embedder = get_embedding_provider()

        # Try load index; if not present, FaissVectorStore.load should return None or empty store
        self.store = FaissVectorStore.load(self.config.index_dir)

    def build_from_files(self, paths: List[str]) -> None:
        """
        Build (or rebuild) FAISS index from scratch.
        Then save to disk.
        """
        all_chunks: List[DocumentChunk] = []

        for path in paths:
            text = load_text_files(path)
            doc_id = path.split("/")[-1].split("\\")[-1]

            chunks = chunk_document(
                doc_id=doc_id,
                text=text,
                chunk_size=self.config.chunk_size,
                overlap=self.config.chunk_overlap,
            )
            all_chunks.extend(chunks)

        vectors = self.embedder.embed([c.text for c in all_chunks])
        dim = int(vectors[0].shape[0]) if vectors else 0

        store = FaissVectorStore(embedding_dim=dim)
        store.add(vectors, all_chunks)

        # Save for next startup
        store.save(self.config.index_dir)
        self.store = store

    def retrieve(self, query: str, top_k: int = 3) -> List[SearchResult]:
        cached = get_cached_results(query, top_k)
        if cached is not None:
            return cached

        if self.store is None:
            raise RuntimeError("Vector store not built. Call build_from_files() first.")

        results = self.store.search(query, self.embedder, top_k=top_k)
        set_cached_results(query, top_k, results)
        return results

    def run(self, question: str, top_k: int = 3) -> RagRunResult:
        results = self.retrieve(question, top_k=top_k)

        context = "\n\n".join(
            f"[{i+1}] doc={r.chunk.doc_id} chunk={r.chunk.chunk_id}\n{r.chunk.text}"
            for i, r in enumerate(results)
        )

        prompt = (
            "You are a helpful assistant. Use ONLY the context.\n"
            "If the answer is not in the context, say you don't know.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )

        llm = get_provider()
        answer = llm.generate(prompt).strip()

        return RagRunResult(answer=answer, results=results)