from __future__ import annotations


def calculator(expression: str = "", **kwargs) -> str:
    expr = expression.strip()

    allowed = set("0123456789+-*/(). %")
    if any(ch not in allowed for ch in expr):
        return "Invalid expression"

    try:
        return str(eval(expr, {"__builtins__": {}}))
    except Exception as e:
        return f"Error: {e}"


def text_length(text: str = "", **kwargs) -> str:
    return str(len(text))