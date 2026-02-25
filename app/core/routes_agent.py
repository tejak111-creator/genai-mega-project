from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app.agents.tools import Tool, ToolRegistry
from app.agents.sample_tools import calculator, text_length
from app.agents.agent_core import MultiStepAgent


logger = logging.getLogger("app")

router = APIRouter(tags=["agent"])


# -------------------------
# Tool registry setup
# -------------------------

registry = ToolRegistry()

registry.register(
    Tool("calculator", "Evaluate math expression", calculator)
)

registry.register(
    Tool("text_length", "Get length of text", text_length)
)


# -------------------------
# Agent
# -------------------------

agent = MultiStepAgent(registry)


# -------------------------
# Route
# -------------------------

@router.post("/agent/run")
def run_agent(question: str):
    try:
        result = agent.run(question)
        return {"answer": result}

    except Exception as e:
        logger.exception("Agent run failed")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )