from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class ToolCall:
    tool_name: str
    arguments: Dict[str, Any]


def parse_tool_call(text: str) -> ToolCall:
    """
    Expect STRICT JSON like:
    {"tool_name": "calculator", "arguments": {"expression": "2+2"}}
    """
    data = json.loads(text)
    if "tool_name" not in data or "arguments" not in data:
        raise ValueError("Invalid tool call JSON: must include tool_name and arguments")
    if not isinstance(data["arguments"], dict):
        raise ValueError("arguments must be an object")
    return ToolCall(tool_name=str(data["tool_name"]), arguments=data["arguments"])