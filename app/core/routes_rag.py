from __future__ import annotations
from fastapi import APIRouter, Header, HTTPException

from app.core.guardrails import basic_input_guard
from app.core.llm import get_provider
from app.rag.api_models import RagChatRequest, RagChatResponse, RagSource
from app.rag.embeddings import SentenceTransformerProvider
from app.rag.vector_store import FaissVectorStore
from app.rag.retriever import Retriever
from app.rag.pipeline import RagPipeline
from app.rag.loader import load_text_files
from app.rag.chunker import chunk_document
from app.rag.embeddings import get_embedding_provider

router = APIRouter(tags=["rags"])

def _build_store_from_sample() -> tuple[FaissVectorStore, SentenceTransformerProvider]:
    text = load_text_files("data/sample.txt")
    chunks = chunk_document(text, doc_id="sample.txt")
    
    embedder = get_embedding_provider()
    vectors = embedder.embed([c.text for c in chunks])

    dim = vectors[0].shape[0]
    store=FaissVectorStore(embedding_dim=dim)
    store.add(vectors=vectors, chunks=chunks)
    return store, embedder

_STORE, _EMBEDDER = _build_store_from_sample()

@router.post("/rag/chat", response_model=RagChatResponse)
def rag_chat(req: RagChatRequest, x_request_id: str | None = Header(default=None)) -> RagChatRequest:
    request_id = x_request_id or "missing-x-request-id"

    guard = basic_input_guard(req.question)
    if not guard.allowed:
        raise HTTPException(status_code=400, detail=f"blocked: {guard.reason}")
    
    retriever = Retriever(store=_STORE, embedder= _EMBEDDER)
    llm= get_provider()
    pipeline = RagPipeline(retriever=retriever, llm_provider=llm)

    result = pipeline.run(req.question, top_k=req.top_k)

    sources = []
    for r in result.results:
        preview = r.chunk.text[:160].replace("\n", " ")
        sources.append(
            RagSource(
                doc_id=r.chunk.doc_id,
                chunk_id=r.chunk.chunk_id,
                score=r.score,
                preview=preview,
            )
        )
    return RagChatResponse(answer=result.answer, sources=sources, request_id=request_id)