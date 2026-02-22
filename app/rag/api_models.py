from __future__ import annotations

from pydantic import BaseModel, Field
from typing import List, Optional

class RagChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User Question")
    top_k: int = Field(3, ge=1, le=10, description="How many chunks to retrieve")

class RagSource(BaseModel):
    doc_id: str
    chunk_id: int
    score: float
    preview: str

class RagChatResponse(BaseModel):
    answer: str
    sources: List[RagSource]
    request_id: Optional[str] = None