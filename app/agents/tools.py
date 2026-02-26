from __future__ import annotations

from typing import Dict
from app.agents.tool_schema import Tool


class ToolRegistry:
    def __init__(self):
        self.tools: Dict[str, Tool] = {}

    def register(self, tool: Tool):
        self.tools[tool.spec.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self.tools:
            raise ValueError(f"Tool {name} not found")
        return self.tools[name]

    def list_specs(self) -> str:
        lines = []
        for t in self.tools.values():
            lines.append(f"{t.spec.name}: {t.spec.description} params={list(t.spec.parameters.keys())}")
        return "\n".join(lines)

    def as_openai_like(self) -> list[dict]:
        """
        Returns OpenAI-like tool schema objects (for teaching & logging).
        """
        out = []
        for t in self.tools.values():
            out.append(
                {
                    "name": t.spec.name,
                    "description": t.spec.description,
                    "parameters": t.spec.parameters,
                }
            )
        return out