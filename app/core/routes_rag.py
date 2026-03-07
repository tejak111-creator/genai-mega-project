from __future__ import annotations

from fastapi import APIRouter, Header, HTTPException
from pathlib import Path

from app.core.guardrails import basic_input_guard
from app.rag.api_models import RagChatRequest, RagChatResponse, RagSource
from app.rag.pipeline import RagPipeline, RagConfig

router = APIRouter(tags=["rags"])


# -------------------------
# Pipeline Initialization
# -------------------------

pipeline = RagPipeline(
    RagConfig(index_dir="data/index")
)


# -------------------------
# Ensure index exists lazily
# -------------------------

BASE_DIR = Path(__file__).resolve().parents[2]   # /app inside container
INDEX_DIR = BASE_DIR / "data" / "index"
SAMPLE_FILE = BASE_DIR / "data" / "sample.txt"


def ensure_index():
    if not (INDEX_DIR / "vectors.faiss").exists():
        if not SAMPLE_FILE.exists():
            raise FileNotFoundError(f"{SAMPLE_FILE} not found")
        pipeline.build_from_files([str(SAMPLE_FILE)])


# -------------------------
# RAG Chat Endpoint
# -------------------------

@router.post("/rag/chat", response_model=RagChatResponse)
def rag_chat(
    req: RagChatRequest,
    x_request_id: str | None = Header(default=None),
) -> RagChatResponse:

    request_id = x_request_id or "missing-x-request-id"

    # Guardrails
    guard = basic_input_guard(req.question)
    if not guard.allowed:
        raise HTTPException(status_code=400, detail=f"blocked: {guard.reason}")

    # Ensure index only when needed
    try:
        ensure_index()
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Run pipeline
    result = pipeline.run(req.question, top_k=req.top_k)

    # Convert sources
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

    return RagChatResponse(
        answer=result.answer,
        sources=sources,
        request_id=request_id,
    )