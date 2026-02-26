from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional
import inspect

@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: Dict[str, Any] #JSON Schema like Dict

"""
ToolSpec = A schema that tells the model what a tool does, what inputs it needs, and how to call it.

It’s like an API contract for AI agents.
"""
class Tool:
    """
    Function-calling style tool:
    -spec: name/description/parameters
    -func: callable
    Validates kwargs against function signature.
    """
    def __init__(self, spec: ToolSpec, func: Callable[..., Any]):
        self.spec = spec
        self.func = func
        self._param_names = set(inspect.signature(func).parameters.keys())

    def run(self, **kwargs) -> Any:
        filtered = { k: v for k, v in kwargs.items() if k in self._param_names}
        return self.func(**filtered)
    