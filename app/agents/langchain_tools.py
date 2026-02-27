from __future__ import annotations

from typing import List

from langchain_core.tools import tool


@tool
def calculator(expression: str) -> str:
    """Evaluate a simple math expression like '2+2'."""
    try:
        # VERY basic eval for demo purposes; in prod use a safe parser
        return str(eval(expression, {"__builtins__": {}}))
    except Exception as e:
        return f"error: {e}"


@tool
def text_length(text: str) -> int:
    """Return length of a text string."""
    return len(text)


def get_langchain_tools() -> List:
    return [calculator, text_length]