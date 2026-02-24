from __future__ import annotations

from fastapi import APIRouter
from app.agents.tools import Tool, ToolRegistry
from app.agents.sample_tools import calculator, text_length
from app.agents.agent_core import SimpleAgent

router = APIRouter(tags=["agent"])

registry = ToolRegistry()
registry.register(Tool("calculator", "Evaluate math expression", calculator))
registry.register(Tool("text_length", "Get length of text", text_length))

agent = SimpleAgent(registry)

@router.post("/agent/run")
def run_agent(question: str):
    result = agent.run(question)
    return {"answer": result}