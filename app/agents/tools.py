from __future__ import annotations

from typing import Any, Dict, Callable

class Tool:
    """
    Represents a callable tool the agent can use.
    """
    def __init__(self, name: str, description: str, func: Callable[..., Any]):
        self.name = name
        self.description = description
        self.func = func
    def run(self, **kwargs) -> Any:
        return self.func(**kwargs)
class ToolRegistry:
    """
    Stores all available tools.
    """
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
    def register(self, tool: Tool):
        self.tools[tool.name] = tool
    def get(self, name: str) -> Tool:
        if name not in self.tools:
            raise ValueError(f"Tool {name} not found")
        return self.tools[name]
    def list_descriptions(self) -> str:
        lines = []
        for t in self.tools.values():
            lines.append(f"{t.name}: {t.description}")
        return "\n".join(lines)