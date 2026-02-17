#API Layer
from fastapi import APIRouter, Header, HTTPException
from app.core.models import ChatRequest, ChatResponse
from app.core.guardrails import basic_input_guard
from app.core.llm import get_provider

router = APIRouter(tags=["chat"])
@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, x_request_id: str | None = Header(default=None)) -> ChatResponse:
    request_id = x_request_id or "missing-x-request-id"

    guard = basic_input_guard(req.prompt)
    if not guard.allowed:
        raise HTTPException(status_code=400, detail=f"blocked: {guard.reason}")
    provider=get_provider()
    text = provider.generate(req.prompt)
    return ChatResponse(response=text, request_id=request_id)

"""
Client
   ↓
FastAPI route (/chat)
   ↓
Parse JSON → ChatRequest
   ↓
Read header → x-request-id
   ↓
Guardrail layer
   ↓
If blocked → HTTP 400
   ↓
If allowed → LLM layer
   ↓
Wrap result → ChatResponse
   ↓
Return JSON
"""