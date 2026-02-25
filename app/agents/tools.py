from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict


@dataclass
class Tool:
    name: str
    description: str
    fn: Callable[..., Any]

    def run(self, *args, **kwargs) -> Any:
        """
        Execute the underlying tool function.

        We accept *args/**kwargs so different tools can have different signatures.
        MultiStepAgent will try kwargs first; if needed it can fall back to positional.
        """
        return self.fn(*args, **kwargs)


class ToolRegistry:
    def __init__(self):
        self.tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        return self.tools[name]

    def list_descriptions(self) -> str:
        return "\n".join(
            [f"- {t.name}: {t.description}" for t in self.tools.values()]
        )