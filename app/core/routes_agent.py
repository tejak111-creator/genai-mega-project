from __future__ import annotations

import logging
from pydantic import BaseModel

from fastapi import APIRouter, HTTPException
from app.agents.tool_schema import ToolSpec, Tool
from app.agents.tools import ToolRegistry
from app.agents.sample_tools import calculator, text_length
from app.agents.agent_core import FunctionCallingAgent


logger = logging.getLogger("app")

router = APIRouter(tags=["agent"])


# -------------------------
# Tool registry setup
# -------------------------

registry = ToolRegistry()

registry.register(
    Tool(
        ToolSpec(
            name="calculator",
            description="Evaluate a math expression like '2+2'",
            parameters={
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        ),
        calculator,
    )
)

registry.register(
    Tool(
        ToolSpec(
            name="text_length",
            description="Return length of a text string",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        ),
        text_length,
    )
)



# -------------------------
# Agent
# -------------------------

agent = FunctionCallingAgent(registry)


class AgentRequest(BaseModel):
    question: str 
# -------------------------
# Route
# -------------------------

@router.post("/agent/run")
def run_agent(req: AgentRequest):
    try:
        result = agent.run(req.question)
        return {"answer": result}

    except Exception as e:
        logger.exception("Agent run failed")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )